// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <atomic>
#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update();
        ramp_step_ = 0;
        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();

            // Inference statistics (logged every ~10s).
            auto stats_start = clock::now();
            auto prev_step_start = clock::now();
            int stats_count = 0;
            double stats_sum_infer_us = 0.0;
            double stats_max_infer_us = 0.0;
            double stats_sum_jitter_us = 0.0;
            double stats_max_jitter_us = 0.0;
            int stats_overrun = 0;
            const double target_period_us = env->step_dt * 1e6;

            while (policy_thread_running)
            {
                auto step_start = clock::now();
                env->step();
                auto step_end = clock::now();
                ramp_step_++;

                double infer_us = std::chrono::duration<double, std::micro>(step_end - step_start).count();
                stats_sum_infer_us += infer_us;
                if (infer_us > stats_max_infer_us) stats_max_infer_us = infer_us;
                if (infer_us > target_period_us) stats_overrun++;

                if (stats_count > 0) {
                    double period_us = std::chrono::duration<double, std::micro>(step_start - prev_step_start).count();
                    double jitter_us = period_us - target_period_us;
                    if (jitter_us < 0) jitter_us = -jitter_us;
                    stats_sum_jitter_us += jitter_us;
                    if (jitter_us > stats_max_jitter_us) stats_max_jitter_us = jitter_us;
                }
                prev_step_start = step_start;
                stats_count++;

                double elapsed_s = std::chrono::duration<double>(step_end - stats_start).count();
                if (elapsed_s >= 10.0) {
                    double fps = stats_count / elapsed_s;
                    double avg_infer_us = stats_sum_infer_us / stats_count;
                    double avg_jitter_us = stats_count > 1 ? stats_sum_jitter_us / (stats_count - 1) : 0.0;
                    spdlog::info("[Inference stats] fps={:.2f}Hz (target={:.1f}Hz), steps={}, infer avg={:.0f}us max={:.0f}us, jitter avg={:.0f}us max={:.0f}us, overruns={}",
                                 fps, 1.0 / env->step_dt, stats_count, avg_infer_us, stats_max_infer_us,
                                 avg_jitter_us, stats_max_jitter_us, stats_overrun);
                    stats_start = step_end;
                    stats_count = 0;
                    stats_sum_infer_us = 0.0;
                    stats_max_infer_us = 0.0;
                    stats_sum_jitter_us = 0.0;
                    stats_max_jitter_us = 0.0;
                    stats_overrun = 0;
                }

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();
    
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

    // Ramp-up duration in steps (2.0s at 50Hz = 100 steps).
    static constexpr int RAMP_STEPS = 100;
    float ramp_factor() const {
        if (ramp_step_ >= RAMP_STEPS) return 1.0f;
        return static_cast<float>(ramp_step_) / static_cast<float>(RAMP_STEPS);
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    bool policy_thread_running = false;
    std::atomic<int> ramp_step_{0};
};

REGISTER_FSM(State_RLBase)
