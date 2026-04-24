// C++ verze sinusoid_response testu. Měří amplitude ratio + phase lag
// mezi target a reálnou pozicí kloubu pro ověření, zda ~200 ms delay
// naměřený z Pythonu není artefakt Python/cyclonedds bindingů.
//
// Používá stejné DDS topic a IDL struct jako deploy binary (unitree_hg).
// Cmd loop běží v dedicated real-time thread s precise chrono-based
// pacing. CRC se počítá inline. Subscriber callback přímo z Unitree SDK.
//
// Build: ./scripts/cpp/recompile.sh
// Spuštění (real robot):
//   ./scripts/cpp/build/sinusoid_response --iface eth0 --joint 0 \
//       --freqs 0.3,0.5,1,1.5,2 --amplitude 0.15
// Spuštění (MuJoCo sim):
//   ./scripts/cpp/build/sinusoid_response --iface lo --joint 0 \
//       --freqs 0.3,0.5,1,1.5,2 --amplitude 0.15

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <unitree/dds_wrapper/common/crc.h>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

using LowCmdMsg = unitree_hg::msg::dds_::LowCmd_;
using LowStateMsg = unitree_hg::msg::dds_::LowState_;

constexpr int NUM_MOTOR = 29;
constexpr int CONTROL_HZ = 500;
constexpr double CONTROL_DT = 1.0 / CONTROL_HZ;
constexpr float KD_DAMPING = 1.0f;

// Sim-matched PD gains (z deploy.yaml balance_height_v6). Musí odpovídat,
// jinak měříme jinou dynamiku než policy v provozu.
static const std::array<float, NUM_MOTOR> KP_DEFAULT = {{
    40.2f, 99.1f, 40.2f, 99.1f, 28.5f, 28.5f,       // 0-5   L leg
    40.2f, 99.1f, 40.2f, 99.1f, 28.5f, 28.5f,       // 6-11  R leg
    40.2f, 28.5f, 28.5f,                             // 12-14 waist
    40.0f, 40.0f, 40.0f, 40.0f, 40.0f, 40.0f, 40.0f, // 15-21 L arm
    40.0f, 40.0f, 40.0f, 40.0f, 40.0f, 40.0f, 40.0f, // 22-28 R arm
}};

static const std::array<float, NUM_MOTOR> KD_DEFAULT = {{
    2.6f, 6.3f, 2.6f, 6.3f, 1.8f, 1.8f,
    2.6f, 6.3f, 2.6f, 6.3f, 1.8f, 1.8f,
    2.6f, 1.8f, 1.8f,
    5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
    5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
}};

static const char* MOTOR_NAMES[NUM_MOTOR] = {
    "left_hip_pitch_joint",      "left_hip_roll_joint",     "left_hip_yaw_joint",
    "left_knee_joint",           "left_ankle_pitch_joint",  "left_ankle_roll_joint",
    "right_hip_pitch_joint",     "right_hip_roll_joint",    "right_hip_yaw_joint",
    "right_knee_joint",          "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",           "waist_roll_joint",        "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint","left_shoulder_yaw_joint",
    "left_elbow_joint",          "left_wrist_roll_joint",   "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",      "right_shoulder_pitch_joint","right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",  "right_elbow_joint",       "right_wrist_roll_joint",
    "right_wrist_pitch_joint",   "right_wrist_yaw_joint",
};

struct Sample {
    double t;
    double q;
    double dq;
    double tau_est;
    double q_target;
};

class SineTester {
public:
    SineTester() : cmd_pub_("rt/lowcmd"), state_sub_("rt/lowstate") {
        for (int i = 0; i < NUM_MOTOR; ++i) {
            kp_scale_[i].store(0.0f);
            q_target_[i].store(0.0f);
            q_real_[i].store(0.0f);
            dq_real_[i].store(0.0f);
            tau_est_[i].store(0.0f);
        }
        std::memset(&cmd_, 0, sizeof(cmd_));
    }

    ~SineTester() { stop_cmd_thread(); }

    void init(const std::string& iface) {
        unitree::robot::ChannelFactory::Instance()->Init(0, iface);

        cmd_pub_.InitChannel();

        state_sub_.InitChannel(
            [this](const void* msg) { on_state(static_cast<const LowStateMsg*>(msg)); }, 10);

        // Wait for first state.
        auto t0 = std::chrono::steady_clock::now();
        while (!mode_machine_set_) {
            if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(5)) {
                throw std::runtime_error(
                    "Žádný LowState během 5 s — zkontroluj iface / propojení.");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        // Targets = current q (aby ramp-up nezpůsobil skok).
        for (int i = 0; i < NUM_MOTOR; ++i) q_target_[i].store(q_real_[i].load());
        std::cout << "[DDS] inicializováno, mode_machine=" << static_cast<int>(mode_machine_)
                  << "\n";
    }

    void start_cmd_thread() {
        running_.store(true);
        cmd_thread_ = std::thread(&SineTester::cmd_loop, this);
    }

    void stop_cmd_thread() {
        running_.store(false);
        if (cmd_thread_.joinable()) cmd_thread_.join();
    }

    void emergency_damping() {
        for (int i = 0; i < NUM_MOTOR; ++i) kp_scale_[i].store(0.0f);
    }

    // Lineární rampa kp_scale všech kloubů.
    void ramp_all_pd(float from_s, float to_s, double duration) {
        if (duration <= 0) {
            for (int i = 0; i < NUM_MOTOR; ++i) kp_scale_[i].store(to_s);
            return;
        }
        int n = std::max(1, static_cast<int>(duration * CONTROL_HZ));
        auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k <= n; ++k) {
            double frac = static_cast<double>(k) / n;
            float s = from_s + (to_s - from_s) * static_cast<float>(frac);
            for (int i = 0; i < NUM_MOTOR; ++i) kp_scale_[i].store(s);
            auto t_target = t0 + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                      std::chrono::duration<double>(k * duration / n));
            std::this_thread::sleep_until(t_target);
        }
    }

    // Ulož aktuální pozice všech kloubů jako target (drží robot v místě).
    void hold_current_positions() {
        for (int i = 0; i < NUM_MOTOR; ++i) q_target_[i].store(q_real_[i].load());
    }

    // Sinusoid test na jeden kloub. Během testu main thread zapisuje
    // q_target[joint_idx] s frekvencí CONTROL_HZ; cmd thread publikuje
    // na DDS; state callback zaznamenává vzorky.
    std::vector<Sample> sine_test(int joint_idx, double amplitude,
                                   double frequency, double duration,
                                   double ramp_cycles = 2.0) {
        double q0 = q_real_[joint_idx].load();
        for (int i = 0; i < NUM_MOTOR; ++i) q_target_[i].store(q_real_[i].load());

        {
            std::lock_guard<std::mutex> lk(record_mutex_);
            record_buffer_.clear();
            record_buffer_.reserve(static_cast<size_t>(duration * 600));
            record_joint_ = joint_idx;
        }
        record_start_ = std::chrono::steady_clock::now();
        recording_.store(true);

        double omega = 2.0 * M_PI * frequency;
        double ramp_dur = (frequency > 0) ? ramp_cycles / frequency : 0.0;

        auto t_start = std::chrono::steady_clock::now();
        auto next_tick = t_start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                        std::chrono::duration<double>(CONTROL_DT));
        while (true) {
            double t = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - t_start)
                           .count();
            if (t >= duration) break;
            double amp_now = (t >= ramp_dur) ? amplitude : amplitude * (t / ramp_dur);
            q_target_[joint_idx].store(
                static_cast<float>(q0 + amp_now * std::sin(omega * t)));
            std::this_thread::sleep_until(next_tick);
            next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                             std::chrono::duration<double>(CONTROL_DT));
        }

        recording_.store(false);
        q_target_[joint_idx].store(static_cast<float>(q0));

        std::vector<Sample> out;
        {
            std::lock_guard<std::mutex> lk(record_mutex_);
            out = record_buffer_;
        }
        return out;
    }

    // Pomalý návrat joint target z aktuální pozice na target.
    void slow_return(int joint_idx, double to_q, double duration) {
        double from_q = q_target_[joint_idx].load();
        if (duration <= 0) {
            q_target_[joint_idx].store(static_cast<float>(to_q));
            return;
        }
        int n = std::max(1, static_cast<int>(duration * CONTROL_HZ));
        auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k <= n; ++k) {
            double frac = static_cast<double>(k) / n;
            q_target_[joint_idx].store(
                static_cast<float>(from_q + (to_q - from_q) * frac));
            auto t_target = t0 + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                      std::chrono::duration<double>(k * duration / n));
            std::this_thread::sleep_until(t_target);
        }
    }

    double cmd_rate_hz() const { return cmd_rate_hz_.load(); }

private:
    void on_state(const LowStateMsg* msg) {
        if (!mode_machine_set_) {
            mode_machine_ = msg->mode_machine();
            mode_machine_set_ = true;
        }
        for (int i = 0; i < NUM_MOTOR; ++i) {
            q_real_[i].store(msg->motor_state()[i].q());
            dq_real_[i].store(msg->motor_state()[i].dq());
            tau_est_[i].store(msg->motor_state()[i].tau_est());
        }

        if (recording_.load()) {
            int j = record_joint_;
            double t = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - record_start_)
                           .count();
            Sample s{t,
                     static_cast<double>(msg->motor_state()[j].q()),
                     static_cast<double>(msg->motor_state()[j].dq()),
                     static_cast<double>(msg->motor_state()[j].tau_est()),
                     static_cast<double>(q_target_[j].load())};
            std::lock_guard<std::mutex> lk(record_mutex_);
            record_buffer_.push_back(s);
        }
    }

    void cmd_loop() {
        // Static fields once.
        cmd_.mode_pr() = 0;
        cmd_.mode_machine() = mode_machine_;
        for (int i = 0; i < NUM_MOTOR; ++i) {
            cmd_.motor_cmd()[i].mode() = 1;
            cmd_.motor_cmd()[i].dq() = 0.0f;
            cmd_.motor_cmd()[i].tau() = 0.0f;
        }

        auto next_tick = std::chrono::steady_clock::now() +
                         std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                             std::chrono::duration<double>(CONTROL_DT));

        int iter_count = 0;
        auto last_rate_report = std::chrono::steady_clock::now();

        while (running_.load()) {
            for (int i = 0; i < NUM_MOTOR; ++i) {
                float s = kp_scale_[i].load();
                cmd_.motor_cmd()[i].kp() = s * KP_DEFAULT[i];
                cmd_.motor_cmd()[i].kd() = s * KD_DEFAULT[i] + (1.0f - s) * KD_DAMPING;
                cmd_.motor_cmd()[i].q() = q_target_[i].load();
            }
            cmd_.crc() = crc32_core(reinterpret_cast<uint32_t*>(&cmd_),
                                    (sizeof(LowCmdMsg) >> 2) - 1);
            cmd_pub_.Write(cmd_);

            iter_count++;
            auto now = std::chrono::steady_clock::now();
            double since_report =
                std::chrono::duration<double>(now - last_rate_report).count();
            if (since_report >= 1.0) {
                cmd_rate_hz_.store(iter_count / since_report);
                iter_count = 0;
                last_rate_report = now;
            }

            std::this_thread::sleep_until(next_tick);
            next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                             std::chrono::duration<double>(CONTROL_DT));
        }
    }

    unitree::robot::ChannelPublisher<LowCmdMsg> cmd_pub_;
    unitree::robot::ChannelSubscriber<LowStateMsg> state_sub_;
    LowCmdMsg cmd_;

    std::array<std::atomic<float>, NUM_MOTOR> kp_scale_;
    std::array<std::atomic<float>, NUM_MOTOR> q_target_;
    std::array<std::atomic<float>, NUM_MOTOR> q_real_;
    std::array<std::atomic<float>, NUM_MOTOR> dq_real_;
    std::array<std::atomic<float>, NUM_MOTOR> tau_est_;

    std::atomic<uint8_t> mode_machine_{0};
    std::atomic<bool> mode_machine_set_{false};

    std::atomic<bool> running_{false};
    std::thread cmd_thread_;
    std::atomic<double> cmd_rate_hz_{0.0};

    std::atomic<bool> recording_{false};
    std::mutex record_mutex_;
    std::vector<Sample> record_buffer_;
    int record_joint_{0};
    std::chrono::steady_clock::time_point record_start_;
};

// ---------- Analýza: least-squares sinus fit ----------

struct AnalysisResult {
    size_t n_samples;
    size_t n_fit;
    double A_q;
    double A_ratio;
    double phase_lag_rad;
    double delay_ms;
    double bias;
    double r_squared;
};

AnalysisResult analyze_sine(const std::vector<Sample>& samples,
                            double frequency, double amplitude_cmd,
                            double skip_ratio = 0.35) {
    AnalysisResult r{};
    r.n_samples = samples.size();
    if (samples.size() < 20 || frequency <= 0) return r;

    size_t skip = std::max<size_t>(1, static_cast<size_t>(skip_ratio * samples.size()));
    size_t n = samples.size() - skip;
    r.n_fit = n;

    double omega = 2.0 * M_PI * frequency;
    // Normal equations coefficients.
    double S0 = 0, Ssin = 0, Scos = 0;
    double Sss = 0, Scc = 0, Ssc = 0;
    double Sq = 0, Sq_s = 0, Sq_c = 0;
    for (size_t i = skip; i < samples.size(); ++i) {
        double t = samples[i].t;
        double q = samples[i].q;
        double s = std::sin(omega * t);
        double c = std::cos(omega * t);
        S0 += 1.0;
        Ssin += s; Scos += c;
        Sss += s * s; Scc += c * c; Ssc += s * c;
        Sq += q; Sq_s += q * s; Sq_c += q * c;
    }
    // Solve 3x3: [[S0, Ssin, Scos], [Ssin, Sss, Ssc], [Scos, Ssc, Scc]] @ [a,b,c] = [Sq, Sq_s, Sq_c]
    double M[3][3] = {{S0, Ssin, Scos}, {Ssin, Sss, Ssc}, {Scos, Ssc, Scc}};
    double B[3] = {Sq, Sq_s, Sq_c};
    // Gauss elimination (in-place).
    for (int i = 0; i < 3; ++i) {
        int pivot = i;
        for (int k = i + 1; k < 3; ++k)
            if (std::abs(M[k][i]) > std::abs(M[pivot][i])) pivot = k;
        if (pivot != i) {
            std::swap(M[i], M[pivot]);
            std::swap(B[i], B[pivot]);
        }
        if (std::abs(M[i][i]) < 1e-12) return r; // singular
        for (int k = i + 1; k < 3; ++k) {
            double f = M[k][i] / M[i][i];
            for (int j = i; j < 3; ++j) M[k][j] -= f * M[i][j];
            B[k] -= f * B[i];
        }
    }
    double x[3];
    for (int i = 2; i >= 0; --i) {
        double s = B[i];
        for (int j = i + 1; j < 3; ++j) s -= M[i][j] * x[j];
        x[i] = s / M[i][i];
    }
    double a = x[0], b = x[1], c = x[2];
    double A_q = std::hypot(b, c);
    double phase_lag = std::atan2(-c, b);  // positive = q lags q_target

    // R².
    double ss_res = 0, ss_tot = 0, q_mean = Sq / S0;
    for (size_t i = skip; i < samples.size(); ++i) {
        double t = samples[i].t;
        double q = samples[i].q;
        double pred = a + b * std::sin(omega * t) + c * std::cos(omega * t);
        ss_res += (q - pred) * (q - pred);
        ss_tot += (q - q_mean) * (q - q_mean);
    }
    double r_sq = (ss_tot > 0) ? 1.0 - ss_res / ss_tot : 0.0;

    r.A_q = A_q;
    r.A_ratio = (amplitude_cmd > 0) ? A_q / amplitude_cmd : 0.0;
    r.phase_lag_rad = phase_lag;
    r.delay_ms = phase_lag / omega * 1000.0;
    r.bias = a;
    r.r_squared = r_sq;
    return r;
}

// ---------- CLI parsing ----------

struct Args {
    std::string iface;
    int joint = 0;
    std::vector<double> freqs;
    double amplitude = 0.15;
    double cycles = 8.0;
    double min_duration = 4.0;
    double max_duration = 20.0;
    double ramp_time = 1.0;
    double pre_wait = 0.5;
    bool yes = false;
    std::string outdir = "sine_response_logs_cpp";
};

std::vector<double> parse_csv_doubles(const std::string& s) {
    std::vector<double> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::stod(item));
    }
    return out;
}

Args parse_args(int argc, char** argv) {
    Args a;
    auto eat = [&](int& i, const char* name) {
        if (i + 1 >= argc) {
            std::cerr << "chybí hodnota pro " << name << "\n";
            std::exit(1);
        }
        return std::string(argv[++i]);
    };
    std::string freqs_str = "0.3,0.5,1,1.5,2";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--iface") a.iface = eat(i, "--iface");
        else if (arg == "--joint") a.joint = std::stoi(eat(i, "--joint"));
        else if (arg == "--freqs" || arg == "--frequencies")
            freqs_str = eat(i, "--freqs");
        else if (arg == "--amplitude") a.amplitude = std::stod(eat(i, "--amplitude"));
        else if (arg == "--cycles") a.cycles = std::stod(eat(i, "--cycles"));
        else if (arg == "--min-duration")
            a.min_duration = std::stod(eat(i, "--min-duration"));
        else if (arg == "--max-duration")
            a.max_duration = std::stod(eat(i, "--max-duration"));
        else if (arg == "--ramp-time") a.ramp_time = std::stod(eat(i, "--ramp-time"));
        else if (arg == "--pre-wait") a.pre_wait = std::stod(eat(i, "--pre-wait"));
        else if (arg == "--outdir") a.outdir = eat(i, "--outdir");
        else if (arg == "--yes" || arg == "-y") a.yes = true;
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: sinusoid_response --iface <eth> --joint <0-28> "
                         "[--freqs 0.3,0.5,1,1.5,2] [--amplitude 0.15] "
                         "[--cycles 8] [--ramp-time 1] [--pre-wait 0.5] [--yes]\n";
            std::exit(0);
        } else {
            std::cerr << "Neznámý argument: " << arg << "\n";
            std::exit(1);
        }
    }
    if (a.iface.empty()) {
        std::cerr << "--iface je povinný (např. eth0 nebo lo)\n";
        std::exit(1);
    }
    if (a.joint < 0 || a.joint >= NUM_MOTOR) {
        std::cerr << "--joint mimo 0-" << (NUM_MOTOR - 1) << "\n";
        std::exit(1);
    }
    a.freqs = parse_csv_doubles(freqs_str);
    if (a.freqs.empty()) {
        std::cerr << "--freqs prázdný\n";
        std::exit(1);
    }
    return a;
}

bool confirm(const std::string& msg) {
    std::cout << msg << " [y/N] " << std::flush;
    std::string r;
    if (!std::getline(std::cin, r)) return false;
    return (r == "y" || r == "Y" || r == "yes");
}

void save_csv(const std::string& path, const std::vector<Sample>& samples) {
    std::ofstream f(path);
    f << "t,q,dq,tau_est,q_target\n";
    f << std::fixed << std::setprecision(6);
    for (const auto& s : samples) {
        f << s.t << "," << s.q << "," << s.dq << "," << s.tau_est << "," << s.q_target
          << "\n";
    }
}

// Phase unwrap (monotonic phase lag).
std::vector<double> unwrap_phase(const std::vector<double>& phases) {
    std::vector<double> out = phases;
    if (out.size() <= 1) return out;
    double accum = 0.0;
    double last = out[0];
    for (size_t i = 1; i < out.size(); ++i) {
        while (out[i] + accum < last - M_PI) accum += 2 * M_PI;
        while (out[i] + accum > last + M_PI) accum -= 2 * M_PI;
        out[i] += accum;
        last = out[i];
    }
    return out;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    std::cout << "============================================================\n";
    std::cout << "BEZPEČNOSTNÍ UPOZORNĚNÍ — C++ SINUSOID RESPONSE TEST\n";
    std::cout << "============================================================\n";
    std::cout << "• Kloub: " << MOTOR_NAMES[args.joint] << " (index " << args.joint
              << ")\n";
    std::cout << "• Amplituda: ±" << args.amplitude << " rad ≈ ±"
              << (args.amplitude * 180.0 / M_PI) << "°\n";
    std::cout << "• Frekvence [Hz]:";
    for (double f : args.freqs) std::cout << " " << f;
    std::cout << "\n• Robot MUSÍ být ZAVĚŠENÝ; kolem kloubu volný prostor.\n\n";
    if (!args.yes && !confirm("Pokračovat?")) {
        std::cout << "Zrušeno.\n";
        return 0;
    }

    std::system(("mkdir -p " + args.outdir).c_str());

    try {
        SineTester tester;
        tester.init(args.iface);
        tester.start_cmd_thread();

        tester.hold_current_positions();
        std::cout << "[ramp] ramp up PD...\n";
        tester.ramp_all_pd(0.0f, 1.0f, args.ramp_time);
        if (args.pre_wait > 0) std::this_thread::sleep_for(
            std::chrono::duration<double>(args.pre_wait));

        struct FreqResult {
            double freq;
            AnalysisResult ana;
        };
        std::vector<FreqResult> results;

        for (size_t k = 0; k < args.freqs.size(); ++k) {
            double f = args.freqs[k];
            double dur = std::max(args.min_duration,
                                  std::min(args.max_duration, args.cycles / f));
            std::cout << "\n── [" << (k + 1) << "/" << args.freqs.size() << "] f=" << f
                      << " Hz  cycles≈" << (dur * f) << "  duration=" << dur << "s\n";
            auto samples = tester.sine_test(args.joint, args.amplitude, f, dur);
            auto a = analyze_sine(samples, f, args.amplitude);
            results.push_back({f, a});
            std::cout << std::fixed << std::setprecision(3)
                      << "    A_ratio=" << a.A_ratio << "  phase="
                      << (a.phase_lag_rad * 180.0 / M_PI) << "°  "
                      << "delay=" << std::setprecision(1) << a.delay_ms << " ms  "
                      << "R²=" << std::setprecision(3) << a.r_squared << "\n";

            std::ostringstream fn;
            fn << args.outdir << "/" << std::setw(2) << std::setfill('0') << args.joint
               << "_" << MOTOR_NAMES[args.joint] << "_f" << std::setprecision(2) << f
               << ".csv";
            save_csv(fn.str(), samples);
            // Mezi frekvencemi klid.
            std::this_thread::sleep_for(std::chrono::duration<double>(0.5));
        }

        std::cout << "\n[ramp] ramp down PD...\n";
        tester.ramp_all_pd(1.0f, 0.0f, args.ramp_time);
        tester.stop_cmd_thread();

        // Phase unwrap.
        std::vector<double> phases;
        for (const auto& r : results) phases.push_back(r.ana.phase_lag_rad);
        auto phases_uw = unwrap_phase(phases);

        std::cout << "\n";
        std::cout << std::string(96, '=') << "\n";
        std::cout << " Sinusoid frequency response (C++) — " << MOTOR_NAMES[args.joint]
                  << "\n";
        std::cout << " Kp=" << KP_DEFAULT[args.joint] << ", Kd=" << KD_DEFAULT[args.joint]
                  << ", amplitude=±" << args.amplitude << " rad\n";
        std::cout << " cmd loop ≈ " << static_cast<int>(tester.cmd_rate_hz()) << " Hz\n";
        std::cout << std::string(96, '=') << "\n";
        std::cout << std::left;
        std::cout << std::setw(9) << " f (Hz)" << std::setw(10) << "A_ratio"
                  << std::setw(11) << "|H|[dB]" << std::setw(12) << "phase(°)"
                  << std::setw(12) << "unwrap(°)" << std::setw(12) << "delay(ms)"
                  << std::setw(8) << "R²" << std::setw(7) << "n_fit" << "\n";
        std::cout << std::string(96, '-') << "\n";
        double sum_delay_lowf = 0;
        int count_lowf = 0;
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            double dB = 20.0 * std::log10(std::max(r.ana.A_ratio, 1e-6));
            double omega = 2.0 * M_PI * r.freq;
            double delay_uw = phases_uw[i] / omega * 1000.0;
            std::cout << std::fixed << std::setprecision(2) << std::setw(9) << r.freq
                      << std::setprecision(3) << std::setw(10) << r.ana.A_ratio
                      << std::setprecision(2) << std::setw(11) << dB
                      << std::setprecision(1) << std::setw(12)
                      << (r.ana.phase_lag_rad * 180.0 / M_PI) << std::setw(12)
                      << (phases_uw[i] * 180.0 / M_PI) << std::setw(12) << delay_uw
                      << std::setprecision(3) << std::setw(8) << r.ana.r_squared
                      << std::setw(7) << r.ana.n_fit << "\n";
            if (r.freq <= 2.0) {
                sum_delay_lowf += delay_uw;
                count_lowf++;
            }
        }
        std::cout << std::string(96, '-') << "\n";
        if (count_lowf > 0) {
            std::cout << " Avg delay (≤2 Hz, operační pásmo) ≈ "
                      << std::setprecision(1) << (sum_delay_lowf / count_lowf) << " ms\n";
        }
        std::cout << std::string(96, '=') << "\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "CHYBA: " << e.what() << "\n";
        return 1;
    }
}
