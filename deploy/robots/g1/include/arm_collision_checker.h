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
constexpr int COLLISION_BISECT_ITERS = 4;

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

// Arm collision geom names in the MuJoCo model.
static const char* const ARM_GEOM_NAMES[] = {
    "left_shoulder_yaw_collision",
    "left_elbow_yaw_collision",
    "left_wrist_collision",
    "left_hand_collision",
    "right_shoulder_yaw_collision",
    "right_elbow_yaw_collision",
    "right_wrist_collision",
    "right_hand_collision",
};
constexpr int NUM_ARM_GEOMS = 8;


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

        // Precompute arm geom IDs.
        for (int i = 0; i < NUM_ARM_GEOMS; ++i) {
            int id = mj_name2id(model_, mjOBJ_GEOM, ARM_GEOM_NAMES[i]);
            if (id < 0)
                spdlog::warn("ArmCollisionChecker: geom '{}' not found", ARM_GEOM_NAMES[i]);
            arm_geom_ids_.push_back(id);
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

    /// Check if desired arm positions collide with the body.
    /// @param motor_pos  All 29 motor positions from lowstate (float).
    /// @param arm_pos    14 desired arm positions from LCM (double).
    /// @param imu_quat   IMU quaternion [w,x,y,z] (float, may be nullptr).
    /// @return true if any arm geom is in contact.
    bool check(const float* motor_pos, const double* arm_pos,
               const float* imu_quat = nullptr) {
        set_positions(motor_pos, arm_pos, imu_quat);
        mj_forward(model_, check_data_);
        return count_arm_contacts() > 0;
    }

    /// Binary-search for the furthest safe arm position along [current_safe → desired].
    /// Writes result into out_safe (14 doubles).
    void find_safe_arms(const double* current_safe, const double* desired,
                        double* out_safe,
                        const float* motor_pos, const float* imu_quat = nullptr) {
        double lo = 0.0, hi = 1.0;
        double test[G1_NUM_ARM_MOTORS];

        for (int iter = 0; iter < COLLISION_BISECT_ITERS; ++iter) {
            double t = (lo + hi) * 0.5;
            for (int i = 0; i < G1_NUM_ARM_MOTORS; ++i)
                test[i] = current_safe[i] + t * (desired[i] - current_safe[i]);
            set_positions(motor_pos, test, imu_quat);
            mj_forward(model_, check_data_);
            if (count_arm_contacts() > 0)
                hi = t;
            else
                lo = t;
        }

        for (int i = 0; i < G1_NUM_ARM_MOTORS; ++i)
            out_safe[i] = current_safe[i] + lo * (desired[i] - current_safe[i]);
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

    int count_arm_contacts() const {
        int count = 0;
        for (int i = 0; i < check_data_->ncon; ++i) {
            int g1 = check_data_->contact[i].geom1;
            int g2 = check_data_->contact[i].geom2;
            for (int aid : arm_geom_ids_) {
                if (aid >= 0 && (g1 == aid || g2 == aid)) {
                    ++count;
                    break;
                }
            }
        }
        return count;
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
