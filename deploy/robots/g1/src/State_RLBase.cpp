#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "arm_collision_checker.h"
#include <unordered_map>

namespace {
static std::unique_ptr<ArmCollisionChecker> g_collision_checker;
static double last_safe_arms[G1_NUM_ARM_MOTORS] = {};
static bool was_colliding = false;
}

#ifdef WITH_LCM_ARMS
#include <lcm/lcm-cpp.hpp>
#include "arm_action_lcmt.hpp"
#include <atomic>
#include <mutex>

namespace {

constexpr int ARM_JOINT_START = 15;
constexpr int NUM_ARM_JOINTS = 14;

// Fallback arm gains used only when the deploy.yaml stiffness/damping arrays
// cover only legs/waist (<29 entries). New configs should include all 29 joints.
constexpr float ARM_KP_FALLBACK = 40.0f;
constexpr float ARM_KD_FALLBACK = 10.0f;

struct ArmReceiver {
    lcm::LCM lcm;
    std::thread thread;
    std::atomic<bool> running{false};
    std::mutex mutex;
    double positions[NUM_ARM_JOINTS] = {};
    bool has_data = false;

    ArmReceiver() {
        if (!lcm.good()) {
            spdlog::error("LCM initialization failed for arm receiver");
            return;
        }
        lcm.subscribe("arm_action", &ArmReceiver::handle, this);
        running = true;
        thread = std::thread([this] {
            while (running) {
                lcm.handleTimeout(100);
            }
        });
        spdlog::info("Arm LCM receiver started (channel: 'arm_action', joints {}-{})",
                      ARM_JOINT_START, ARM_JOINT_START + NUM_ARM_JOINTS - 1);
    }

    ~ArmReceiver() {
        running = false;
        if (thread.joinable())
            thread.join();
    }

    void handle(const lcm::ReceiveBuffer*, const std::string&, const arm_action_lcmt* msg) {
        std::lock_guard<std::mutex> lock(mutex);
        for (int i = 0; i < NUM_ARM_JOINTS; ++i)
            positions[i] = msg->act[i];
        has_data = true;
    }

    bool get_positions(double out[NUM_ARM_JOINTS]) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!has_data) return false;
        for (int i = 0; i < NUM_ARM_JOINTS; ++i)
            out[i] = positions[i];
        return true;
    }
};

static std::unique_ptr<ArmReceiver> g_arm_receiver;

} // anonymous namespace
#endif // WITH_LCM_ARMS


namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {1.0f, 0.0f, 0.0f}},
        {"s", {-1.0f, 0.0f, 0.0f}},
        {"a", {0.0f, 1.0f, 0.0f}},
        {"d", {0.0f, -1.0f, 0.0f}},
        {"q", {0.0f, 0.0f, 1.0f}},
        {"e", {0.0f, 0.0f, -1.0f}}
    };
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if (key_commands.find(key) != key_commands.end())
    {
        cmd = key_commands[key];
    }
    return cmd;
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );

#ifdef WITH_LCM_ARMS
    if (param::receive_arms && !g_arm_receiver) {
        g_arm_receiver = std::make_unique<ArmReceiver>();
    }
#else
    if (param::receive_arms) {
        spdlog::error("--receive-arms requires building with -DWITH_LCM_ARMS=ON (and liblcm-dev installed)");
        std::exit(1);
    }
#endif

    // Init collision checker (needed when arms are received or render requested).
    bool need_collision = (param::receive_arms && !param::disable_collisions)
                          || param::render_collisions;
    if (need_collision && !g_collision_checker) {
        g_collision_checker = std::make_unique<ArmCollisionChecker>();
        auto xml_path = param::config_dir / "g1_collision.xml";
        if (!g_collision_checker->init(xml_path.string())) {
            spdlog::error("Collision checker init failed — disabling");
            g_collision_checker.reset();
        } else if (param::render_collisions) {
            g_collision_checker->start_render();
        }
    }
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    // action.size() may be smaller than joint_ids_map.size() when the policy
    // only controls a subset of joints (e.g. legs/waist but not arms).
    auto num_action_joints = std::min(action.size(), env->robot->data.joint_ids_map.size());

    // Rate-limit: interpolate max delta from tight (startup) to permissive.
    // At 50Hz: 0.01 rad/step = 0.5 rad/s, 0.2 rad/step = 10 rad/s.
    constexpr float MAX_DELTA_START = 0.01f;   // 0.5 rad/s
    constexpr float MAX_DELTA_FULL = 0.2f;     // 10 rad/s (safety cap)
    const float alpha = ramp_factor();
    const float max_delta = MAX_DELTA_START + alpha * (MAX_DELTA_FULL - MAX_DELTA_START);

    // Log joint positions every ~10s.
    static int log_counter_ = 0;
    log_counter_++;
    if (log_counter_ % 5000 == 0) {
        std::string pos_str = "[Joint positions] ";
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        for (int i = 0; i < 29; ++i) {
            pos_str += fmt::format("{:.4f}", lowstate->msg_.motor_state()[i].q());
            if (i < 28) pos_str += ", ";
        }
        spdlog::info(pos_str);
    }

    for(size_t i(0); i < num_action_joints; i++) {
        int motor_id = env->robot->data.joint_ids_map[i];
        float current_cmd = lowcmd->msg_.motor_cmd()[motor_id].q();
        float desired = action[i];
        float delta = desired - current_cmd;
        // Clamp the step size.
        if (delta > max_delta) delta = max_delta;
        if (delta < -max_delta) delta = -max_delta;
        lowcmd->msg_.motor_cmd()[motor_id].q() = current_cmd + delta;
    }

#ifdef WITH_LCM_ARMS
    if (g_arm_receiver) {
        // Prefer gains from deploy.yaml (stiffness/damping arrays covering all 29
        // joints). Fall back to hardcoded values for legacy configs with only
        // 15 entries.
        const auto& kp_arr = env->robot->data.joint_stiffness;
        const auto& kd_arr = env->robot->data.joint_damping;
        const bool use_cfg_gains = (kp_arr.size() >= ARM_JOINT_START + NUM_ARM_JOINTS)
                                && (kd_arr.size() >= ARM_JOINT_START + NUM_ARM_JOINTS);

        // Set arm joint gains (keeps arms active even before first LCM message)
        for (int i = 0; i < NUM_ARM_JOINTS; ++i) {
            const int idx = ARM_JOINT_START + i;
            const float kp = use_cfg_gains ? kp_arr[idx] : ARM_KP_FALLBACK;
            const float kd = use_cfg_gains ? kd_arr[idx] : ARM_KD_FALLBACK;
            lowcmd->msg_.motor_cmd()[idx].kp() = kp;
            lowcmd->msg_.motor_cmd()[idx].kd() = kd;
            lowcmd->msg_.motor_cmd()[idx].dq() = 0;
            lowcmd->msg_.motor_cmd()[idx].tau() = 0;
        }

        // Apply LCM arm positions (arms stay at FixStand position until first message)
        double arm_pos[NUM_ARM_JOINTS];
        if (g_arm_receiver->get_positions(arm_pos)) {
            // Per-joint collision resolution: each joint is tested independently
            // so a blocked shoulder doesn't freeze elbow/wrist, and arms are independent.
            if (g_collision_checker && !param::disable_collisions) {
                float all_motor_pos[G1_NUM_MOTORS];
                float imu_quat[4];
                {
                    std::lock_guard<std::mutex> lock(lowstate->mutex_);
                    for (int i = 0; i < G1_NUM_MOTORS; ++i)
                        all_motor_pos[i] = lowstate->msg_.motor_state()[i].q();
                    for (int i = 0; i < 4; ++i)
                        imu_quat[i] = lowstate->msg_.imu_state().quaternion()[i];
                }

                double resolved[NUM_ARM_JOINTS];
                bool colliding = g_collision_checker->resolve_arms(
                    all_motor_pos, last_safe_arms, arm_pos, resolved, imu_quat);

                if (colliding) {
                    if (!was_colliding)
                        spdlog::warn("Arm collision: {}", g_collision_checker->last_contacts());
                } else {
                    if (was_colliding)
                        spdlog::info("Arm collision cleared");
                }
                was_colliding = colliding;

                for (int i = 0; i < NUM_ARM_JOINTS; ++i) {
                    arm_pos[i] = resolved[i];
                    last_safe_arms[i] = resolved[i];
                }

            }

            for (int i = 0; i < NUM_ARM_JOINTS; ++i) {
                lowcmd->msg_.motor_cmd()[ARM_JOINT_START + i].q() = arm_pos[i];
            }
        }

        // Always update render from lowstate (even before first LCM message).
        if (g_collision_checker) {
            float all_motor_pos[G1_NUM_MOTORS];
            float imu_quat[4];
            {
                std::lock_guard<std::mutex> lock(lowstate->mutex_);
                for (int i = 0; i < G1_NUM_MOTORS; ++i)
                    all_motor_pos[i] = lowstate->msg_.motor_state()[i].q();
                for (int i = 0; i < 4; ++i)
                    imu_quat[i] = lowstate->msg_.imu_state().quaternion()[i];
            }
            g_collision_checker->update_render_state(all_motor_pos, imu_quat, was_colliding);
        }
    }
#endif
}
