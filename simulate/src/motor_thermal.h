#pragma once

// Simulated motor thermal model for the MuJoCo bridge.
//
// This is a C++ port of the winding-temperature model used during RL training
// (see src/tasks/velocity/mdp/thermal.py). It lets the simulator publish a
// motor temperature in the DDS LowState that rises over time with joint load,
// just like a real robot.
//
// Model (per thermal node i), first-order lumped winding temperature:
//     P_i  = k_i * sum_{a in sources(i)} tau_a^2
//     dT_i = dt * (P_i - (T_i - T_amb)) / tau_th_i        (R_th folded into k)
//
// Coupled actuators: the G1 waist (pitch+roll) and each ankle (pitch+roll) are
// 4-bar linkages driven by TWO shared physical motors, so each node in a
// coupled group is heated by the COMBINED squared torque of the group. This
// mirrors the training model exactly.
//
// In addition to the calibrated 15 leg+waist joints, every remaining actuator
// (arms, wrists, etc.) gets a generic fallback node so its temperature also
// rises with load — uncalibrated, but better than a constant.
//
// A second, slower "casing" temperature is derived from the winding so the two
// temperature channels of the real MotorState (temperature[0]=casing,
// temperature[1]=winding) can both be populated.

#include <mujoco/mujoco.h>

#include <cmath>
#include <string>
#include <vector>

class MotorThermalSim
{
public:
    struct Params {
        double k;        // heating gain G = k*R_th [K/(N*m)^2/s scaled by tau_th]
        double tau_th;   // winding thermal time constant [s]
    };

    // T_amb: ambient temperature [C]. Matches MotorThermalCfg.T_amb (30 C).
    explicit MotorThermalSim(mjModel *model, double T_amb = 30.0)
        : T_amb_(T_amb)
    {
        const int nu = model->nu;
        T_.assign(nu, T_amb_);
        T_case_.assign(nu, T_amb_);
        k_.assign(nu, 0.0);
        tau_th_.assign(nu, 0.0);
        sources_.assign(nu, {});
        _build_nodes(model);
    }

    int num_motors() const { return static_cast<int>(T_.size()); }

    double winding(int i) const { return T_[i]; }
    double casing(int i) const { return T_case_[i]; }

    // Over-temperature status word, mirroring real-robot fault flags.
    // 0 = ok, 1 = warning (>80 C), 2 = critical (>115 C, training termination).
    uint32_t status(int i) const
    {
        if (T_[i] > 115.0) return 2u;
        if (T_[i] > 80.0) return 1u;
        return 0u;
    }

    // Integrate all nodes forward by dt seconds using the current actuator
    // forces. dt is the elapsed simulation time since the previous call.
    void step(mjData *data, double dt)
    {
        if (dt <= 0.0) return;
        if (dt > 0.1) dt = 0.1;  // guard against large jumps (e.g. after pause)

        for (int i = 0; i < num_motors(); ++i) {
            double tau2_sum = 0.0;
            for (int s : sources_[i]) {
                const double tau = data->actuator_force[s];
                tau2_sum += tau * tau;
            }
            const double P = k_[i] * tau2_sum;
            const double dT = dt * (P - (T_[i] - T_amb_)) / tau_th_[i];
            T_[i] += dT;

            // Casing tracks the winding with attenuation and a slower lag.
            const double target = T_amb_ + 0.7 * (T_[i] - T_amb_);
            const double tau_case = 3.0 * tau_th_[i];
            T_case_[i] += dt * (target - T_case_[i]) / tau_case;
        }
    }

private:
    double T_amb_;
    std::vector<double> T_;        // winding temperature per actuator [C]
    std::vector<double> T_case_;   // casing temperature per actuator [C]
    std::vector<double> k_;
    std::vector<double> tau_th_;
    std::vector<std::vector<int>> sources_;  // source actuator ids per node

    // Resolve the actuator id driving a given joint. Tries an actuator named
    // exactly like the joint, then falls back to a substring match on the
    // joint stem (without the trailing "_joint").
    static int _actuator_id(mjModel *model, const std::string &joint)
    {
        int id = mj_name2id(model, mjOBJ_ACTUATOR, joint.c_str());
        if (id >= 0) return id;

        std::string stem = joint;
        const std::string suffix = "_joint";
        if (stem.size() > suffix.size() &&
            stem.compare(stem.size() - suffix.size(), suffix.size(), suffix) == 0) {
            stem = stem.substr(0, stem.size() - suffix.size());
        }
        for (int i = 0; i < model->nu; ++i) {
            const char *name = mj_id2name(model, mjOBJ_ACTUATOR, i);
            if (name && std::string(name).find(stem) != std::string::npos) {
                return i;
            }
        }
        return -1;
    }

    void _set_node(mjModel *model, const std::string &joint, const Params &p,
                   const std::vector<std::string> &source_joints)
    {
        const int act = _actuator_id(model, joint);
        if (act < 0) return;  // joint not present in this scene

        k_[act] = p.k;
        tau_th_[act] = p.tau_th;
        std::vector<int> srcs;
        for (const auto &sj : source_joints) {
            const int sid = _actuator_id(model, sj);
            if (sid >= 0) srcs.push_back(sid);
        }
        if (srcs.empty()) srcs.push_back(act);
        sources_[act] = srcs;
    }

    // Build the 15 calibrated leg+waist nodes (params copied verbatim from the
    // training DEFAULT_PARAMS), then a generic fallback for every other motor.
    void _build_nodes(mjModel *model)
    {
        // Calibrated params, keyed by joint function (left/right share params).
        const Params hip_pitch{0.090, 27.0};
        const Params hip_roll{0.032, 19.0};
        const Params hip_yaw{0.059, 18.0};
        const Params knee{0.043, 37.0};
        const Params ankle_pitch{0.067, 15.0};
        const Params ankle_roll{0.090, 18.0};
        const Params waist_yaw{0.059, 18.0};
        const Params waist_pitch{0.067, 13.0};
        const Params waist_roll{0.107, 20.0};

        // Independent leg joints (each driven by its own torque), left + right.
        for (const std::string side : {"left", "right"}) {
            _set_node(model, side + "_hip_pitch_joint", hip_pitch,
                      {side + "_hip_pitch_joint"});
            _set_node(model, side + "_hip_roll_joint", hip_roll,
                      {side + "_hip_roll_joint"});
            _set_node(model, side + "_hip_yaw_joint", hip_yaw,
                      {side + "_hip_yaw_joint"});
            _set_node(model, side + "_knee_joint", knee, {side + "_knee_joint"});
        }

        // Coupled ankle: pitch+roll on each leg share two physical motors.
        for (const std::string side : {"left", "right"}) {
            const std::vector<std::string> group = {
                side + "_ankle_pitch_joint", side + "_ankle_roll_joint"};
            _set_node(model, side + "_ankle_pitch_joint", ankle_pitch, group);
            _set_node(model, side + "_ankle_roll_joint", ankle_roll, group);
        }

        // Waist: yaw is independent; pitch+roll are a coupled pair.
        _set_node(model, "waist_yaw_joint", waist_yaw, {"waist_yaw_joint"});
        const std::vector<std::string> waist_group = {"waist_pitch_joint",
                                                      "waist_roll_joint"};
        _set_node(model, "waist_pitch_joint", waist_pitch, waist_group);
        _set_node(model, "waist_roll_joint", waist_roll, waist_group);

        // Generic fallback for any remaining actuator (arms, wrists, hands...).
        const Params fallback{0.050, 20.0};
        for (int i = 0; i < num_motors(); ++i) {
            if (tau_th_[i] <= 0.0) {
                k_[i] = fallback.k;
                tau_th_[i] = fallback.tau_th;
                sources_[i] = {i};
            }
        }
    }
};
