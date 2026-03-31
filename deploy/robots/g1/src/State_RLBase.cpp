#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>

#ifdef WITH_LCM_ARMS
#include <lcm/lcm-cpp.hpp>
#include "arm_action_lcmt.hpp"
#include <atomic>
#include <mutex>

namespace {

constexpr int ARM_JOINT_START = 15;
constexpr int NUM_ARM_JOINTS = 14;

// Arm gains matching FixStand config defaults
constexpr float ARM_KP = 40.0f;
constexpr float ARM_KD = 10.0f;

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
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }

#ifdef WITH_LCM_ARMS
    if (g_arm_receiver) {
        // Set arm joint gains (keeps arms active even before first LCM message)
        for (int i = 0; i < NUM_ARM_JOINTS; ++i) {
            lowcmd->msg_.motor_cmd()[ARM_JOINT_START + i].kp() = ARM_KP;
            lowcmd->msg_.motor_cmd()[ARM_JOINT_START + i].kd() = ARM_KD;
            lowcmd->msg_.motor_cmd()[ARM_JOINT_START + i].dq() = 0;
            lowcmd->msg_.motor_cmd()[ARM_JOINT_START + i].tau() = 0;
        }

        // Apply LCM arm positions (arms stay at FixStand position until first message)
        double arm_pos[NUM_ARM_JOINTS];
        if (g_arm_receiver->get_positions(arm_pos)) {
            for (int i = 0; i < NUM_ARM_JOINTS; ++i) {
                lowcmd->msg_.motor_cmd()[ARM_JOINT_START + i].q() = arm_pos[i];
            }
        }
    }
#endif
}
