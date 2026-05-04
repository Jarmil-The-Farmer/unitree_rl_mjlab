#pragma once

#include <mujoco/mujoco.h>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "inspire_hand_ctrl.hpp"

class InspireHandModbusServer
{
public:
    InspireHandModbusServer(mjModel *model, mjData *data, int port,
                            int left_device_id, int right_device_id)
        : model_(model),
          data_(data),
          port_(port),
          left_device_id_(left_device_id),
          right_device_id_(right_device_id)
    {
        left_.device_id = left_device_id_;
        right_.device_id = right_device_id_;
        init_hand_state(left_);
        init_hand_state(right_);
        initialized_ = find_finger_joints();
    }

    ~InspireHandModbusServer()
    {
        stop();
    }

    bool available() const
    {
        return initialized_;
    }

    void start()
    {
        if (!initialized_ || running_) {
            return;
        }

        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0) {
            std::cerr << "[inspire_modbus] socket() failed: " << std::strerror(errno) << std::endl;
            return;
        }

        int opt = 1;
        setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        addr.sin_port = htons(static_cast<uint16_t>(port_));

        if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            std::cerr << "[inspire_modbus] bind port " << port_
                      << " failed: " << std::strerror(errno) << std::endl;
            ::close(listen_fd_);
            listen_fd_ = -1;
            return;
        }

        if (::listen(listen_fd_, 8) < 0) {
            std::cerr << "[inspire_modbus] listen() failed: " << std::strerror(errno) << std::endl;
            ::close(listen_fd_);
            listen_fd_ = -1;
            return;
        }

        running_ = true;
        accept_thread_ = std::thread([this] { accept_loop(); });
        std::cout << "[inspire_modbus] listening on 0.0.0.0:" << port_
                  << " (left unit " << left_device_id_
                  << ", right unit " << right_device_id_ << ")" << std::endl;

        start_dds_subscribers();
    }

    void stop()
    {
        running_ = false;
        if (listen_fd_ >= 0) {
            ::shutdown(listen_fd_, SHUT_RDWR);
            ::close(listen_fd_);
            listen_fd_ = -1;
        }
        if (accept_thread_.joinable()) {
            accept_thread_.join();
        }
    }

    void apply_to_sim()
    {
        if (!initialized_ || !data_) {
            return;
        }

        std::array<std::array<int16_t, kNumFingers>, kNumHands> targets;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            targets[kLeft] = left_.angle_set;
            targets[kRight] = right_.angle_set;
        }

        const double dt = model_->opt.timestep > 1e-6 ? model_->opt.timestep : kDefaultTimestepSec;
        const double alpha = 1.0 - std::exp(-dt / kMotorTimeConstantSec);
        const double max_step = kMaxMotorVelocityRadPerSec * dt;
        std::array<std::array<int16_t, kNumFingers>, kNumHands> actual_angles{};

        for (int hand = 0; hand < kNumHands; ++hand) {
            for (int finger = 0; finger < kNumFingers; ++finger) {
                const double normalized_close =
                    1.0 - std::clamp(static_cast<double>(targets[hand][finger]) / 1000.0, 0.0, 1.0);

                for (const FingerJoint& joint : finger_joints_[hand][finger]) {
                    const double target =
                        joint.open_qpos + normalized_close * (joint.closed_qpos - joint.open_qpos);
                    const double current = data_->qpos[joint.qpos_adr];
                    const double filtered_target = current + alpha * (target - current);
                    const double delta = std::clamp(filtered_target - current, -max_step, max_step);
                    data_->qpos[joint.qpos_adr] = current + delta;
                    data_->qvel[joint.qvel_adr] = delta / dt;
                }

                actual_angles[hand][finger] = sim_finger_angle_value(hand, finger);
            }
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            left_.angle_act = actual_angles[kLeft];
            left_.pos_act = actual_angles[kLeft];
            right_.angle_act = actual_angles[kRight];
            right_.pos_act = actual_angles[kRight];
        }
    }

private:
    static constexpr int kLeft = 0;
    static constexpr int kRight = 1;
    static constexpr int kNumHands = 2;
    static constexpr int kNumFingers = 6;
    static constexpr int kRegisterCount = 6000;
    static constexpr double kDefaultTimestepSec = 0.002;
    static constexpr double kRegularFingerMaxBendRad = 1.5707963267948966; // 90 deg
    static constexpr double kThumbBendMaxRad = 0.7853981633974483;          // 45 deg
    static constexpr double kMotorTimeConstantSec = 0.12;
    static constexpr double kMaxMotorVelocityRadPerSec = 6.0;

    static constexpr uint16_t kAddrResetError = 1004;
    static constexpr uint16_t kAddrPosSet = 1474;
    static constexpr uint16_t kAddrAngleSet = 1486;
    static constexpr uint16_t kAddrForceSet = 1498;
    static constexpr uint16_t kAddrSpeedSet = 1522;
    static constexpr uint16_t kAddrPosAct = 1534;
    static constexpr uint16_t kAddrAngleAct = 1546;
    static constexpr uint16_t kAddrForceAct = 1582;
    static constexpr uint16_t kAddrCurrent = 1594;
    static constexpr uint16_t kAddrError = 1606;
    static constexpr uint16_t kAddrStatus = 1612;
    static constexpr uint16_t kAddrTemperature = 1618;

    struct FingerJoint {
        int qpos_adr = -1;
        int qvel_adr = -1;
        double open_qpos = 0.0;
        double closed_qpos = 0.0;
    };

    struct HandState {
        int device_id = 1;
        std::array<int16_t, kNumFingers> pos_set{};
        std::array<int16_t, kNumFingers> angle_set{};
        std::array<int16_t, kNumFingers> pos_act{};
        std::array<int16_t, kNumFingers> angle_act{};
        std::array<int16_t, kNumFingers> force_set{};
        std::array<int16_t, kNumFingers> speed_set{};
        std::array<int16_t, kNumFingers> current{};
        std::array<int16_t, kNumFingers> err{};
        std::array<int16_t, kNumFingers> status{};
        std::array<int16_t, kNumFingers> temperature{};
    };

    using InspireHandSubscriber =
        unitree::robot::ChannelSubscriber<inspire::inspire_hand_ctrl>;

    static void init_hand_state(HandState& state)
    {
        state.pos_set.fill(1000);
        state.angle_set.fill(1000);
        state.pos_act.fill(1000);
        state.angle_act.fill(1000);
        state.force_set.fill(500);
        state.speed_set.fill(500);
        state.current.fill(0);
        state.err.fill(0);
        state.status.fill(2);
        state.temperature.fill(25);
    }

    bool find_finger_joints()
    {
        static const std::array<std::array<std::vector<std::string>, kNumFingers>, kNumHands> names = {{
            {{
                {"left_little_1_joint", "left_little_2_joint"},
                {"left_ring_1_joint", "left_ring_2_joint"},
                {"left_middle_1_joint", "left_middle_2_joint"},
                {"left_index_1_joint", "left_index_2_joint"},
                {"left_thumb_2_joint", "left_thumb_3_joint", "left_thumb_4_joint"},
                {"left_thumb_1_joint"},
            }},
            {{
                {"right_little_1_joint", "right_little_2_joint"},
                {"right_ring_1_joint", "right_ring_2_joint"},
                {"right_middle_1_joint", "right_middle_2_joint"},
                {"right_index_1_joint", "right_index_2_joint"},
                {"right_thumb_2_joint", "right_thumb_3_joint", "right_thumb_4_joint"},
                {"right_thumb_1_joint"},
            }},
        }};

        int found = 0;
        for (int hand = 0; hand < kNumHands; ++hand) {
            for (int finger = 0; finger < kNumFingers; ++finger) {
                for (size_t joint_slot = 0; joint_slot < names[hand][finger].size(); ++joint_slot) {
                    const std::string& name = names[hand][finger][joint_slot];
                    const int joint_id = mj_name2id(model_, mjOBJ_JOINT, name.c_str());
                    if (joint_id < 0) {
                        continue;
                    }

                    FingerJoint joint;
                    joint.qpos_adr = model_->jnt_qposadr[joint_id];
                    joint.qvel_adr = model_->jnt_dofadr[joint_id];
                    const double range_min = model_->jnt_range[2 * joint_id];
                    const double range_max = model_->jnt_range[2 * joint_id + 1];
                    joint.open_qpos = range_min;
                    joint.closed_qpos = closed_qpos_for(finger, joint_slot, range_min, range_max);
                    finger_joints_[hand][finger].push_back(joint);
                    found++;
                }
            }
        }

        if (found == 0) {
            std::cout << "[inspire_modbus] no Inspire finger joints found; server disabled" << std::endl;
            return false;
        }

        std::cout << "[inspire_modbus] found " << found << " finger joints" << std::endl;
        return true;
    }

    static double closed_qpos_for(int finger, size_t joint_slot,
                                  double range_min, double range_max)
    {
        const double range_delta = range_max - range_min;
        const double direction = range_delta >= 0.0 ? 1.0 : -1.0;
        const double xml_span = std::abs(range_delta);
        double desired_span = xml_span;

        if (finger >= 0 && finger <= 3) {
            desired_span = kRegularFingerMaxBendRad;
        } else if (finger == 4) {
            desired_span = kThumbBendMaxRad;
        } else if (finger == 5) {
            // Thumb rotation is already reasonable in the MuJoCo model.
            desired_span = xml_span;
        }

        (void)joint_slot;
        return range_min + direction * std::min(xml_span, desired_span);
    }

    int16_t sim_finger_angle_value(int hand, int finger) const
    {
        const auto& joints = finger_joints_[hand][finger];
        if (joints.empty()) {
            return 1000;
        }

        double close_sum = 0.0;
        int count = 0;
        for (const FingerJoint& joint : joints) {
            const double span = joint.closed_qpos - joint.open_qpos;
            if (std::abs(span) < 1e-9) {
                continue;
            }
            const double close =
                std::clamp((data_->qpos[joint.qpos_adr] - joint.open_qpos) / span, 0.0, 1.0);
            close_sum += close;
            count++;
        }

        if (count == 0) {
            return 1000;
        }

        const double average_close = close_sum / static_cast<double>(count);
        return clamp_hand_value(static_cast<int16_t>(std::lround((1.0 - average_close) * 1000.0)));
    }

    HandState& hand_for_unit(uint8_t unit_id)
    {
        if (unit_id == left_device_id_) {
            return left_;
        }
        return right_;
    }

    HandState& hand_for_index(int hand_index)
    {
        return hand_index == kLeft ? left_ : right_;
    }

    int hand_index_for_unit(uint8_t unit_id) const
    {
        return unit_id == left_device_id_ ? kLeft : kRight;
    }

    static uint16_t read_u16(const uint8_t *data)
    {
        return static_cast<uint16_t>((data[0] << 8) | data[1]);
    }

    static void push_u16(std::vector<uint8_t>& out, uint16_t value)
    {
        out.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
        out.push_back(static_cast<uint8_t>(value & 0xff));
    }

    static bool recv_exact(int fd, uint8_t *buf, size_t len)
    {
        size_t got = 0;
        while (got < len) {
            const ssize_t n = ::recv(fd, buf + got, len - got, 0);
            if (n <= 0) {
                return false;
            }
            got += static_cast<size_t>(n);
        }
        return true;
    }

    static bool send_all(int fd, const std::vector<uint8_t>& data)
    {
        size_t sent = 0;
        while (sent < data.size()) {
            const ssize_t n = ::send(fd, data.data() + sent, data.size() - sent, MSG_NOSIGNAL);
            if (n <= 0) {
                return false;
            }
            sent += static_cast<size_t>(n);
        }
        return true;
    }

    uint16_t read_register_locked(const HandState& hand, uint16_t address) const
    {
        if (address == 1000) {
            return static_cast<uint16_t>(hand.device_id);
        }
        if (in_range(address, kAddrPosSet)) return to_u16(hand.pos_set[address - kAddrPosSet]);
        if (in_range(address, kAddrAngleSet)) return to_u16(hand.angle_set[address - kAddrAngleSet]);
        if (in_range(address, kAddrForceSet)) return to_u16(hand.force_set[address - kAddrForceSet]);
        if (in_range(address, kAddrSpeedSet)) return to_u16(hand.speed_set[address - kAddrSpeedSet]);
        if (in_range(address, kAddrPosAct)) return to_u16(hand.pos_act[address - kAddrPosAct]);
        if (in_range(address, kAddrAngleAct)) return to_u16(hand.angle_act[address - kAddrAngleAct]);
        if (in_range(address, kAddrForceAct)) return to_u16(hand.force_set[address - kAddrForceAct]);
        if (in_range(address, kAddrCurrent)) return to_u16(hand.current[address - kAddrCurrent]);
        if (in_range(address, kAddrError)) return to_u16(hand.err[address - kAddrError]);
        if (in_range(address, kAddrStatus)) return to_u16(hand.status[address - kAddrStatus]);
        if (in_range(address, kAddrTemperature)) return to_u16(hand.temperature[address - kAddrTemperature]);
        return 0;
    }

    void write_register_locked(HandState& hand, uint16_t address, uint16_t raw_value)
    {
        const int16_t value = from_u16(raw_value);
        if (address == kAddrResetError) {
            hand.err.fill(0);
            return;
        }
        if (write_array_value(hand.pos_set, address, kAddrPosSet, value)) {
            hand.angle_set[address - kAddrPosSet] = clamp_hand_value(value);
            return;
        }
        if (write_array_value(hand.angle_set, address, kAddrAngleSet, value)) return;
        if (write_array_value(hand.force_set, address, kAddrForceSet, value)) return;
        if (write_array_value(hand.speed_set, address, kAddrSpeedSet, value)) return;
    }

    std::vector<uint16_t> read_holding(uint8_t unit_id, uint16_t address, uint16_t quantity)
    {
        std::vector<uint16_t> values;
        values.reserve(quantity);
        std::lock_guard<std::mutex> lock(mutex_);
        const HandState& hand = hand_for_unit(unit_id);
        for (uint16_t i = 0; i < quantity; ++i) {
            values.push_back(read_register_locked(hand, address + i));
        }
        (void)hand_index_for_unit(unit_id);
        return values;
    }

    void write_holding(uint8_t unit_id, uint16_t address, const std::vector<uint16_t>& values)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        HandState& hand = hand_for_unit(unit_id);
        for (size_t i = 0; i < values.size(); ++i) {
            write_register_locked(hand, address + static_cast<uint16_t>(i), values[i]);
        }

        const uint64_t count = ++modbus_rx_count_;
        if (count <= 5 || count % 100 == 0) {
            std::cout << "[inspire_modbus] modbus write #" << count
                      << " unit=" << static_cast<int>(unit_id)
                      << " addr=" << address
                      << " count=" << values.size()
                      << " first=" << (values.empty() ? 0 : values.front())
                      << std::endl;
        }
    }

    void start_dds_subscribers()
    {
        if (dds_started_) {
            return;
        }

        sub_left_ = std::make_shared<InspireHandSubscriber>("rt/inspire_hand/ctrl/l");
        sub_right_ = std::make_shared<InspireHandSubscriber>("rt/inspire_hand/ctrl/r");
        sub_left_->InitChannel([this](const void *message) {
            const auto *cmd = static_cast<const inspire::inspire_hand_ctrl*>(message);
            write_dds_command(kLeft, *cmd);
        }, 10);
        sub_right_->InitChannel([this](const void *message) {
            const auto *cmd = static_cast<const inspire::inspire_hand_ctrl*>(message);
            write_dds_command(kRight, *cmd);
        }, 10);
        dds_started_ = true;
        std::cout << "[inspire_modbus] DDS subscribers active on rt/inspire_hand/ctrl/l|r" << std::endl;
    }

    void write_dds_command(int hand_index, const inspire::inspire_hand_ctrl& msg)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            HandState& hand = hand_for_index(hand_index);
            const size_t angle_count = std::min<size_t>(msg.angle_set().size(), kNumFingers);
            const size_t pos_count = std::min<size_t>(msg.pos_set().size(), kNumFingers);
            const size_t force_count = std::min<size_t>(msg.force_set().size(), kNumFingers);
            const size_t speed_count = std::min<size_t>(msg.speed_set().size(), kNumFingers);

            if (msg.mode() & 0b0001) {
                for (size_t i = 0; i < angle_count; ++i) {
                    hand.angle_set[i] = clamp_hand_value(msg.angle_set()[i]);
                }
            }
            if (msg.mode() & 0b0010) {
                for (size_t i = 0; i < pos_count; ++i) {
                    hand.pos_set[i] = clamp_hand_value(msg.pos_set()[i]);
                    hand.angle_set[i] = hand.pos_set[i];
                }
            }
            if (msg.mode() & 0b0100) {
                for (size_t i = 0; i < force_count; ++i) {
                    hand.force_set[i] = clamp_hand_value(msg.force_set()[i]);
                }
            }
            if (msg.mode() & 0b1000) {
                for (size_t i = 0; i < speed_count; ++i) {
                    hand.speed_set[i] = clamp_hand_value(msg.speed_set()[i]);
                }
            }
        }

        const uint64_t count = ++dds_rx_count_;
        if (count <= 5 || count % 100 == 0) {
            const auto& angles = msg.angle_set();
            std::cout << "[inspire_modbus] DDS rx #" << count
                      << " hand=" << (hand_index == kLeft ? "left" : "right")
                      << " mode=" << static_cast<int>(msg.mode())
                      << " angle0=" << (angles.empty() ? 0 : angles[0])
                      << " angle3=" << (angles.size() > 3 ? angles[3] : 0)
                      << std::endl;
        }
    }

    std::vector<uint8_t> handle_pdu(uint8_t unit_id, const std::vector<uint8_t>& pdu)
    {
        if (pdu.empty()) {
            return {};
        }

        const uint8_t fn = pdu[0];
        if (fn == 0x03 && pdu.size() >= 5) {
            const uint16_t address = read_u16(&pdu[1]);
            const uint16_t quantity = read_u16(&pdu[3]);
            if (quantity == 0 || quantity > 125 || address + quantity > kRegisterCount) {
                return exception_pdu(fn, 0x03);
            }

            const auto regs = read_holding(unit_id, address, quantity);
            std::vector<uint8_t> response{fn, static_cast<uint8_t>(regs.size() * 2)};
            for (uint16_t reg : regs) {
                push_u16(response, reg);
            }
            return response;
        }

        if (fn == 0x06 && pdu.size() >= 5) {
            const uint16_t address = read_u16(&pdu[1]);
            const uint16_t value = read_u16(&pdu[3]);
            write_holding(unit_id, address, {value});
            return pdu;
        }

        if (fn == 0x10 && pdu.size() >= 6) {
            const uint16_t address = read_u16(&pdu[1]);
            const uint16_t quantity = read_u16(&pdu[3]);
            const uint8_t byte_count = pdu[5];
            if (quantity == 0 || quantity > 123 || byte_count != quantity * 2 ||
                pdu.size() < static_cast<size_t>(6 + byte_count)) {
                return exception_pdu(fn, 0x03);
            }

            std::vector<uint16_t> values;
            values.reserve(quantity);
            for (uint16_t i = 0; i < quantity; ++i) {
                values.push_back(read_u16(&pdu[6 + i * 2]));
            }
            write_holding(unit_id, address, values);

            std::vector<uint8_t> response{fn};
            push_u16(response, address);
            push_u16(response, quantity);
            return response;
        }

        return exception_pdu(fn, 0x01);
    }

    static std::vector<uint8_t> exception_pdu(uint8_t fn, uint8_t code)
    {
        return {static_cast<uint8_t>(fn | 0x80), code};
    }

    void accept_loop()
    {
        while (running_) {
            sockaddr_in client_addr{};
            socklen_t addr_len = sizeof(client_addr);
            int client_fd = ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&client_addr), &addr_len);
            if (client_fd < 0) {
                if (running_) {
                    std::cerr << "[inspire_modbus] accept() failed: "
                              << std::strerror(errno) << std::endl;
                }
                continue;
            }

            std::thread([this, client_fd] { client_loop(client_fd); }).detach();
        }
    }

    void client_loop(int client_fd)
    {
        while (running_) {
            uint8_t header[7];
            if (!recv_exact(client_fd, header, sizeof(header))) {
                break;
            }

            const uint16_t transaction_id = read_u16(&header[0]);
            const uint16_t protocol_id = read_u16(&header[2]);
            const uint16_t length = read_u16(&header[4]);
            const uint8_t unit_id = header[6];
            if (protocol_id != 0 || length < 2 || length > 260) {
                break;
            }

            std::vector<uint8_t> pdu(length - 1);
            if (!recv_exact(client_fd, pdu.data(), pdu.size())) {
                break;
            }

            const auto response_pdu = handle_pdu(unit_id, pdu);
            if (response_pdu.empty()) {
                break;
            }

            std::vector<uint8_t> response;
            push_u16(response, transaction_id);
            push_u16(response, protocol_id);
            push_u16(response, static_cast<uint16_t>(response_pdu.size() + 1));
            response.push_back(unit_id);
            response.insert(response.end(), response_pdu.begin(), response_pdu.end());

            if (!send_all(client_fd, response)) {
                break;
            }
        }

        ::shutdown(client_fd, SHUT_RDWR);
        ::close(client_fd);
    }

    static bool in_range(uint16_t address, uint16_t start)
    {
        return address >= start && address < start + kNumFingers;
    }

    static uint16_t to_u16(int16_t value)
    {
        return static_cast<uint16_t>(value);
    }

    static int16_t from_u16(uint16_t value)
    {
        return static_cast<int16_t>(value);
    }

    static int16_t clamp_hand_value(int16_t value)
    {
        return static_cast<int16_t>(std::clamp<int>(value, 0, 1000));
    }

    static bool write_array_value(std::array<int16_t, kNumFingers>& array,
                                  uint16_t address, uint16_t start, int16_t value)
    {
        if (!in_range(address, start)) {
            return false;
        }
        array[address - start] = clamp_hand_value(value);
        return true;
    }

    mjModel *model_ = nullptr;
    mjData *data_ = nullptr;
    int port_ = 6000;
    int left_device_id_ = 2;
    int right_device_id_ = 1;
    bool initialized_ = false;

    std::array<std::array<std::vector<FingerJoint>, kNumFingers>, kNumHands> finger_joints_{};
    HandState left_;
    HandState right_;
    std::mutex mutex_;
    std::atomic<uint64_t> modbus_rx_count_{0};
    std::atomic<uint64_t> dds_rx_count_{0};

    std::atomic<bool> running_{false};
    int listen_fd_ = -1;
    std::thread accept_thread_;
    bool dds_started_ = false;
    std::shared_ptr<InspireHandSubscriber> sub_left_;
    std::shared_ptr<InspireHandSubscriber> sub_right_;
};
