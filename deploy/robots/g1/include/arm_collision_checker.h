// Arm collision checker using MuJoCo for real-time self-collision prevention.
// Loads a lightweight collision model (g1_collision.xml), sets joint positions
// from robot state, and detects arm-body / arm-arm contacts via mj_forward().
// Optional GLFW rendering shows the collision capsules and contact points.

#pragma once

#include <mujoco/mujoco.h>
#include <spdlog/spdlog.h>

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef WITH_COLLISION_RENDER
#include <GLFW/glfw3.h>
#endif

// G1 29-DOF motor layout.
constexpr int G1_NUM_MOTORS = 29;
constexpr int G1_ARM_START = 15;        // first arm motor index
constexpr int G1_NUM_ARM_MOTORS = 14;   // 7 per arm
constexpr int G1_JOINTS_PER_ARM = 7;

// Joint names in hardware motor index order (0-28).
static const char* const G1_JOINT_NAMES[G1_NUM_MOTORS] = {
    "left_hip_pitch_joint",      // 0
    "left_hip_roll_joint",       // 1
    "left_hip_yaw_joint",        // 2
    "left_knee_joint",           // 3
    "left_ankle_pitch_joint",    // 4
    "left_ankle_roll_joint",     // 5
    "right_hip_pitch_joint",     // 6
    "right_hip_roll_joint",      // 7
    "right_hip_yaw_joint",       // 8
    "right_knee_joint",          // 9
    "right_ankle_pitch_joint",   // 10
    "right_ankle_roll_joint",    // 11
    "waist_yaw_joint",           // 12
    "waist_roll_joint",          // 13
    "waist_pitch_joint",         // 14
    "left_shoulder_pitch_joint", // 15
    "left_shoulder_roll_joint",  // 16
    "left_shoulder_yaw_joint",   // 17
    "left_elbow_joint",          // 18
    "left_wrist_roll_joint",     // 19
    "left_wrist_pitch_joint",    // 20
    "left_wrist_yaw_joint",      // 21
    "right_shoulder_pitch_joint",// 22
    "right_shoulder_roll_joint", // 23
    "right_shoulder_yaw_joint",  // 24
    "right_elbow_joint",         // 25
    "right_wrist_roll_joint",    // 26
    "right_wrist_pitch_joint",   // 27
    "right_wrist_yaw_joint",     // 28
};

// Arm body names — all geoms attached to these bodies are considered arm geoms.
static const char* const ARM_BODY_NAMES[] = {
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
};
constexpr int NUM_ARM_BODIES = 14;


class ArmCollisionChecker {
public:
    ArmCollisionChecker() = default;
    ~ArmCollisionChecker() { shutdown(); }

    // Non-copyable.
    ArmCollisionChecker(const ArmCollisionChecker&) = delete;
    ArmCollisionChecker& operator=(const ArmCollisionChecker&) = delete;

    bool init(const std::string& xml_path) {
        char error[1000] = "";
        model_ = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
        if (!model_) {
            spdlog::error("ArmCollisionChecker: failed to load '{}': {}", xml_path, error);
            return false;
        }
        check_data_ = mj_makeData(model_);

        // Precompute arm geom IDs: collect all geoms belonging to arm bodies.
        for (int i = 0; i < NUM_ARM_BODIES; ++i) {
            int bid = mj_name2id(model_, mjOBJ_BODY, ARM_BODY_NAMES[i]);
            if (bid < 0) {
                spdlog::warn("ArmCollisionChecker: body '{}' not found", ARM_BODY_NAMES[i]);
                continue;
            }
            for (int g = 0; g < model_->body_geomnum[bid]; ++g) {
                int gid = model_->body_geomadr[bid] + g;
                if (model_->geom_contype[gid] > 0 || model_->geom_conaffinity[gid] > 0)
                    arm_geom_ids_.push_back(gid);
            }
        }

        // Build motor-index → qpos-address mapping.
        for (int i = 0; i < G1_NUM_MOTORS; ++i) {
            int jid = mj_name2id(model_, mjOBJ_JOINT, G1_JOINT_NAMES[i]);
            if (jid < 0) {
                spdlog::error("ArmCollisionChecker: joint '{}' not found", G1_JOINT_NAMES[i]);
                shutdown();
                return false;
            }
            joint_qposadr_.push_back(model_->jnt_qposadr[jid]);
        }

        // Freejoint qpos address.
        int fj = mj_name2id(model_, mjOBJ_JOINT, "floating_base_joint");
        fj_qposadr_ = (fj >= 0) ? model_->jnt_qposadr[fj] : -1;

        // Set standing pose.
        reset_freejoint(check_data_);

        spdlog::info("ArmCollisionChecker: loaded '{}' ({} arm geoms, {} joints)",
                     xml_path, arm_geom_ids_.size(), joint_qposadr_.size());
        return true;
    }

    void shutdown() {
        stop_render();
        if (check_data_) { mj_deleteData(check_data_); check_data_ = nullptr; }
        if (model_)      { mj_deleteModel(model_); model_ = nullptr; }
    }

    /// Resolve arm collisions per-joint, proximal to distal.
    /// Each joint is tested independently — a blocked shoulder doesn't freeze
    /// the elbow/wrist, and left arm doesn't affect right arm.
    ///
    /// @param motor_pos   All 29 motor positions from lowstate (float).
    /// @param safe_pos    Last known safe arm positions (14 doubles).
    /// @param desired     Desired arm positions from LCM (14 doubles).
    /// @param out         Result arm positions (14 doubles). Each joint gets
    ///                    desired if safe, or reverts to safe_pos if it causes collision.
    /// @param imu_quat    IMU quaternion [w,x,y,z] (float, may be nullptr).
    /// @return true if any joint was clamped.
    bool resolve_arms(const float* motor_pos, const double* safe_pos,
                      const double* desired, double* out,
                      const float* imu_quat = nullptr) {
        // Start with all arms at safe positions and set legs/waist/freejoint.
        set_positions(motor_pos, safe_pos, imu_quat);

        // Fast path: try all desired at once.
        for (int i = 0; i < G1_NUM_ARM_MOTORS; ++i)
            check_data_->qpos[joint_qposadr_[G1_ARM_START + i]] = desired[i];
        mj_forward(model_, check_data_);

        if (!has_arm_contacts()) {
            for (int i = 0; i < G1_NUM_ARM_MOTORS; ++i)
                out[i] = desired[i];
            last_contact_info_.clear();
            return false;
        }

        // Snapshot contact info before per-joint resolution modifies state.
        last_contact_info_ = get_contact_info();

        // Collision detected — resolve per-joint, each arm independently.
        // Reset to safe positions.
        for (int i = 0; i < G1_NUM_ARM_MOTORS; ++i)
            check_data_->qpos[joint_qposadr_[G1_ARM_START + i]] = safe_pos[i];

        bool any_clamped = false;

        // Process left arm (joints 0-6) then right arm (joints 7-13).
        for (int arm = 0; arm < 2; ++arm) {
            int base = arm * G1_JOINTS_PER_ARM;  // 0 or 7
            for (int j = 0; j < G1_JOINTS_PER_ARM; ++j) {
                int idx = base + j;
                double prev = check_data_->qpos[joint_qposadr_[G1_ARM_START + idx]];
                // Try desired value for this joint.
                check_data_->qpos[joint_qposadr_[G1_ARM_START + idx]] = desired[idx];
                mj_forward(model_, check_data_);

                if (has_arm_contacts()) {
                    // Revert this joint to safe.
                    check_data_->qpos[joint_qposadr_[G1_ARM_START + idx]] = prev;
                    out[idx] = prev;
                    any_clamped = true;
                } else {
                    // Accept desired.
                    out[idx] = desired[idx];
                }
            }
        }
        return any_clamped;
    }

    /// Return a human-readable string of current arm contact pairs (body names).
    std::string get_contact_info() const {
        std::string info;
        // Collect unique body pairs.
        std::vector<std::pair<int,int>> seen;
        for (int i = 0; i < check_data_->ncon; ++i) {
            int g1 = check_data_->contact[i].geom1;
            int g2 = check_data_->contact[i].geom2;
            bool is_arm = false;
            for (int aid : arm_geom_ids_) {
                if (g1 == aid || g2 == aid) { is_arm = true; break; }
            }
            if (!is_arm) continue;
            int b1 = model_->geom_bodyid[g1];
            int b2 = model_->geom_bodyid[g2];
            if (b1 > b2) std::swap(b1, b2);
            bool dup = false;
            for (auto& p : seen) if (p.first == b1 && p.second == b2) { dup = true; break; }
            if (dup) continue;
            seen.push_back({b1, b2});
            const char* n1 = mj_id2name(model_, mjOBJ_BODY, b1);
            const char* n2 = mj_id2name(model_, mjOBJ_BODY, b2);
            if (!info.empty()) info += ", ";
            info += (n1 ? n1 : "?");
            info += " <-> ";
            info += (n2 ? n2 : "?");
        }
        return info;
    }

    // ── Render (optional, requires -DWITH_COLLISION_RENDER) ────────────────

    void start_render() {
#ifdef WITH_COLLISION_RENDER
        if (render_running_) return;
        render_running_ = true;
        render_qpos_.resize(model_->nq, 0.0);
        reset_freejoint_buf();
        render_thread_ = std::thread(&ArmCollisionChecker::render_loop, this);
        spdlog::info("ArmCollisionChecker: render thread started");
#else
        spdlog::error("ArmCollisionChecker: render unavailable (build without -DWITH_COLLISION_RENDER)");
#endif
    }

    void stop_render() {
#ifdef WITH_COLLISION_RENDER
        render_running_ = false;
        if (render_thread_.joinable()) render_thread_.join();
#endif
    }

    /// Push latest state to the render thread (called from control loop).
    /// Uses actual motor positions (all 29 from lowstate) so render matches reality.
    /// collision_active flag controls the overlay text independently.
    void update_render_state(const float* all_motor_pos,
                             const float* imu_quat = nullptr,
                             bool collision_active = false) {
#ifdef WITH_COLLISION_RENDER
        if (!render_running_) return;
        {
            std::lock_guard<std::mutex> lock(render_mutex_);
            if (fj_qposadr_ >= 0) {
                render_qpos_[fj_qposadr_ + 0] = 0;
                render_qpos_[fj_qposadr_ + 1] = 0;
                render_qpos_[fj_qposadr_ + 2] = 0.79;
                if (imu_quat) {
                    for (int i = 0; i < 4; ++i)
                        render_qpos_[fj_qposadr_ + 3 + i] = imu_quat[i];
                } else {
                    render_qpos_[fj_qposadr_ + 3] = 1.0;
                    render_qpos_[fj_qposadr_ + 4] = 0;
                    render_qpos_[fj_qposadr_ + 5] = 0;
                    render_qpos_[fj_qposadr_ + 6] = 0;
                }
            }
            for (int i = 0; i < G1_NUM_MOTORS; ++i)
                render_qpos_[joint_qposadr_[i]] = all_motor_pos[i];
            render_dirty_ = true;
        }
        render_collision_active_ = collision_active;
#else
        (void)all_motor_pos; (void)imu_quat; (void)collision_active;
#endif
    }

private:
    mjModel* model_ = nullptr;
    mjData*  check_data_ = nullptr;

    std::vector<int> arm_geom_ids_;
    std::vector<int> joint_qposadr_;  // [motor_idx] → qpos index
    int fj_qposadr_ = -1;
    std::string last_contact_info_;

public:
    const std::string& last_contacts() const { return last_contact_info_; }
private:

    void reset_freejoint(mjData* d) const {
        if (fj_qposadr_ < 0) return;
        d->qpos[fj_qposadr_ + 0] = 0;
        d->qpos[fj_qposadr_ + 1] = 0;
        d->qpos[fj_qposadr_ + 2] = 0.79;
        d->qpos[fj_qposadr_ + 3] = 1.0;  // w
        d->qpos[fj_qposadr_ + 4] = 0;    // x
        d->qpos[fj_qposadr_ + 5] = 0;    // y
        d->qpos[fj_qposadr_ + 6] = 0;    // z
    }

    void set_positions(const float* motor_pos, const double* arm_pos,
                       const float* imu_quat) {
        if (fj_qposadr_ >= 0) {
            check_data_->qpos[fj_qposadr_ + 0] = 0;
            check_data_->qpos[fj_qposadr_ + 1] = 0;
            check_data_->qpos[fj_qposadr_ + 2] = 0.79;
            if (imu_quat) {
                for (int i = 0; i < 4; ++i)
                    check_data_->qpos[fj_qposadr_ + 3 + i] = imu_quat[i];
            } else {
                check_data_->qpos[fj_qposadr_ + 3] = 1.0;
                check_data_->qpos[fj_qposadr_ + 4] = 0;
                check_data_->qpos[fj_qposadr_ + 5] = 0;
                check_data_->qpos[fj_qposadr_ + 6] = 0;
            }
        }
        // Legs + waist from motor feedback.
        for (int i = 0; i < G1_ARM_START; ++i)
            check_data_->qpos[joint_qposadr_[i]] = motor_pos[i];
        // Arms from desired positions.
        for (int i = 0; i < G1_NUM_ARM_MOTORS; ++i)
            check_data_->qpos[joint_qposadr_[G1_ARM_START + i]] = arm_pos[i];
    }

    bool has_arm_contacts() const {
        for (int i = 0; i < check_data_->ncon; ++i) {
            int g1 = check_data_->contact[i].geom1;
            int g2 = check_data_->contact[i].geom2;
            for (int aid : arm_geom_ids_) {
                if (aid >= 0 && (g1 == aid || g2 == aid))
                    return true;
            }
        }
        return false;
    }

    // ── Render internals ───────────────────────────────────────────────────

#ifdef WITH_COLLISION_RENDER
    std::thread render_thread_;
    std::atomic<bool> render_running_{false};
    std::atomic<bool> render_collision_active_{false};
    std::mutex render_mutex_;
    std::vector<double> render_qpos_;
    bool render_dirty_ = false;

    void reset_freejoint_buf() {
        if (fj_qposadr_ < 0) return;
        render_qpos_[fj_qposadr_ + 2] = 0.79;
        render_qpos_[fj_qposadr_ + 3] = 1.0;
    }

    // Context passed to GLFW callbacks via window user pointer.
    struct RenderCtx {
        mjvCamera cam{};
        bool button_left = false;
        bool button_right = false;
        bool button_middle = false;
        double last_x = 0, last_y = 0;
    };

    static void glfw_mouse_button(GLFWwindow* w, int button, int action, int /*mods*/) {
        auto* ctx = static_cast<RenderCtx*>(glfwGetWindowUserPointer(w));
        bool pressed = (action == GLFW_PRESS);
        if (button == GLFW_MOUSE_BUTTON_LEFT)   ctx->button_left = pressed;
        if (button == GLFW_MOUSE_BUTTON_RIGHT)  ctx->button_right = pressed;
        if (button == GLFW_MOUSE_BUTTON_MIDDLE) ctx->button_middle = pressed;
        glfwGetCursorPos(w, &ctx->last_x, &ctx->last_y);
    }

    static void glfw_cursor_pos(GLFWwindow* w, double xpos, double ypos) {
        auto* ctx = static_cast<RenderCtx*>(glfwGetWindowUserPointer(w));
        double dx = xpos - ctx->last_x;
        double dy = ypos - ctx->last_y;
        ctx->last_x = xpos;
        ctx->last_y = ypos;

        if (ctx->button_left) {
            ctx->cam.azimuth   -= 0.3 * dx;
            ctx->cam.elevation -= 0.3 * dy;
        }
        if (ctx->button_right) {
            ctx->cam.lookat[0] -= 0.003 * dx;
            ctx->cam.lookat[1] += 0.003 * dy;
        }
        if (ctx->button_middle) {
            ctx->cam.distance *= (1.0 + 0.003 * dy);
        }
    }

    static void glfw_scroll(GLFWwindow* w, double /*xoff*/, double yoff) {
        auto* ctx = static_cast<RenderCtx*>(glfwGetWindowUserPointer(w));
        ctx->cam.distance *= (1.0 - 0.05 * yoff);
    }

    void render_loop() {
        if (!glfwInit()) {
            spdlog::error("ArmCollisionChecker: glfwInit failed");
            render_running_ = false;
            return;
        }

        GLFWwindow* window = glfwCreateWindow(1280, 720,
            "Arm Collision Checker", nullptr, nullptr);
        if (!window) {
            spdlog::error("ArmCollisionChecker: glfwCreateWindow failed");
            glfwTerminate();
            render_running_ = false;
            return;
        }
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        // Render data (separate from check_data_ for thread safety).
        mjData* rd = mj_makeData(model_);
        reset_freejoint(rd);

        // Visualization objects.
        mjvCamera cam;
        mjvOption opt;
        mjvScene  scn;
        mjrContext con;
        mjv_defaultCamera(&cam);
        mjv_defaultOption(&opt);
        mjv_defaultScene(&scn);
        mjr_defaultContext(&con);
        mjv_makeScene(model_, &scn, 2000);
        mjr_makeContext(model_, &con, mjFONTSCALE_150);

        // Camera: free orbit, looking at the robot.
        cam.type = mjCAMERA_FREE;
        cam.distance = 3.0;
        cam.elevation = -20.0;
        cam.azimuth = 135.0;
        cam.lookat[2] = 0.5;

        // Show contact markers.
        opt.flags[mjVIS_CONTACTPOINT] = 1;
        opt.flags[mjVIS_CONTACTFORCE] = 1;

        // GLFW callbacks.
        RenderCtx rctx;
        rctx.cam = cam;
        glfwSetWindowUserPointer(window, &rctx);
        glfwSetMouseButtonCallback(window, glfw_mouse_button);
        glfwSetCursorPosCallback(window, glfw_cursor_pos);
        glfwSetScrollCallback(window, glfw_scroll);

        while (render_running_ && !glfwWindowShouldClose(window)) {
            // Copy latest qpos.
            {
                std::lock_guard<std::mutex> lock(render_mutex_);
                if (render_dirty_) {
                    mju_copy(rd->qpos, render_qpos_.data(), model_->nq);
                    render_dirty_ = false;
                }
            }
            mj_forward(model_, rd);

            // Build status text from control-loop collision flag.
            char status[128];
            if (render_collision_active_)
                snprintf(status, sizeof(status), "COLLISION (clamped)");
            else
                snprintf(status, sizeof(status), "SAFE");

            // Render.
            cam = rctx.cam;
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            mjrRect viewport = {0, 0, width, height};

            mjv_updateScene(model_, rd, &opt, nullptr, &cam, mjCAT_ALL, &scn);
            mjr_render(viewport, &scn, &con);

            // Status overlay (top-left).
            mjr_overlay(mjFONT_BIG, mjGRID_TOPLEFT, viewport, status, nullptr, &con);

            glfwSwapBuffers(window);
            glfwPollEvents();
        }

        // Cleanup.
        mjr_freeContext(&con);
        mjv_freeScene(&scn);
        mj_deleteData(rd);
        glfwDestroyWindow(window);
        glfwTerminate();
        render_running_ = false;
    }
#endif  // WITH_COLLISION_RENDER
};
