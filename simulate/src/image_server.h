#pragma once

// MuJoCo offscreen renderer + ZMQ PUB image server.
//
// Renders a fixed camera defined in the MJCF (e.g. "head_cam") into an
// off-screen framebuffer, encodes each frame as JPEG with libjpeg-turbo, and
// publishes the JPEG bytes over a ZMQ PUB socket on the configured port.
//
// The wire format matches teleop/image_server/image_server.py running on the
// real robot (Unit_Test == False): each ZMQ message contains exactly the raw
// JPEG bytes -- so the existing ImageClient on the operator side works
// unchanged whether the robot is real or simulated.
//
// Rendering uses an off-screen GLFW window (hidden, single-buffered context).
// The window MUST be created on the main thread (GLFW requirement); see
// main.cc which creates it before sim->RenderLoop() starts. This thread then
// takes ownership of that window's GL context via glfwMakeContextCurrent.
// Using GLFW (instead of EGL) avoids NVIDIA's flaky EGL/X11 path.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>
#include <turbojpeg.h>
#include <zmq.h>

namespace image_server {

struct Config
{
    bool enable = false;
    int port = 5555;
    int fps = 30;
    int width = 640;
    int height = 480;
    int jpeg_quality = 80;
    std::string camera_name = "head_cam";
};

class ImageServer
{
public:
    // `gl_window` must be a hidden GLFW window created on the main thread.
    // Its OpenGL context is taken over by the worker thread via
    // glfwMakeContextCurrent. Pass nullptr to disable rendering.
    ImageServer(mjModel *model,
                mjData *data,
                std::recursive_mutex *sim_mtx,
                GLFWwindow *gl_window,
                Config cfg)
        : m_(model), d_(data), sim_mtx_(sim_mtx), gl_window_(gl_window), cfg_(cfg) {}

    ~ImageServer() { stop(); }

    void start()
    {
        if (running_.exchange(true)) return;
        thread_ = std::thread(&ImageServer::run, this);
    }

    void stop()
    {
        if (!running_.exchange(false)) return;
        if (thread_.joinable()) thread_.join();
    }

private:
    void run()
    {
        if (!gl_window_) {
            std::fprintf(stderr,
                         "[image_server] No GL window provided; image server disabled.\n");
            running_ = false;
            return;
        }

        // Take ownership of the hidden GL context on this thread.
        glfwMakeContextCurrent(gl_window_);

        // Look up camera id.
        cam_id_ = mj_name2id(m_, mjOBJ_CAMERA, cfg_.camera_name.c_str());
        if (cam_id_ < 0) {
            std::fprintf(stderr,
                         "[image_server] Camera '%s' not found in model; image server disabled.\n",
                         cfg_.camera_name.c_str());
            glfwMakeContextCurrent(nullptr);
            running_ = false;
            return;
        }

        // MuJoCo visualization state owned by this thread.
        mjv_defaultCamera(&cam_);
        mjv_defaultOption(&opt_);
        mjv_defaultScene(&scn_);
        mjr_defaultContext(&con_);

        // Bias model offscreen buffer size so mjr_makeContext allocates the size
        // we want (the offscreen framebuffer cannot grow past these values).
        m_->vis.global.offwidth = std::max(m_->vis.global.offwidth, cfg_.width);
        m_->vis.global.offheight = std::max(m_->vis.global.offheight, cfg_.height);

        mjv_makeScene(m_, &scn_, 2000);
        mjr_makeContext(m_, &con_, mjFONTSCALE_100);
        mjr_setBuffer(mjFB_OFFSCREEN, &con_);
        if (con_.currentBuffer != mjFB_OFFSCREEN) {
            std::fprintf(stderr,
                         "[image_server] Offscreen rendering not supported; image server disabled.\n");
            mjr_freeContext(&con_);
            mjv_freeScene(&scn_);
            glfwMakeContextCurrent(nullptr);
            running_ = false;
            return;
        }

        // Use the named MJCF camera.
        cam_.type = mjCAMERA_FIXED;
        cam_.fixedcamid = cam_id_;

        // ZMQ PUB socket.
        zmq_ctx_ = zmq_ctx_new();
        zmq_sock_ = zmq_socket(zmq_ctx_, ZMQ_PUB);
        std::string endpoint = "tcp://*:" + std::to_string(cfg_.port);
        if (zmq_bind(zmq_sock_, endpoint.c_str()) != 0) {
            std::fprintf(stderr, "[image_server] zmq_bind(%s) failed: %s\n",
                         endpoint.c_str(), zmq_strerror(zmq_errno()));
            zmq_close(zmq_sock_);
            zmq_ctx_term(zmq_ctx_);
            mjr_freeContext(&con_);
            mjv_freeScene(&scn_);
            glfwMakeContextCurrent(nullptr);
            running_ = false;
            return;
        }

        // libjpeg-turbo encoder.
        tj_ = tjInitCompress();
        if (!tj_) {
            std::fprintf(stderr, "[image_server] tjInitCompress failed: %s\n", tjGetErrorStr2(nullptr));
            zmq_close(zmq_sock_);
            zmq_ctx_term(zmq_ctx_);
            mjr_freeContext(&con_);
            mjv_freeScene(&scn_);
            glfwMakeContextCurrent(nullptr);
            running_ = false;
            return;
        }

        const int W = cfg_.width;
        const int H = cfg_.height;
        rgb_.assign(static_cast<size_t>(3 * W * H), 0);
        rgb_flipped_.assign(static_cast<size_t>(3 * W * H), 0);

        std::printf("[image_server] PUB tcp://*:%d, camera='%s', %dx%d @ %d fps\n",
                    cfg_.port, cfg_.camera_name.c_str(), W, H, cfg_.fps);

        const auto frame_period = std::chrono::microseconds(1000000 / std::max(1, cfg_.fps));
        auto next_tick = std::chrono::steady_clock::now();
        mjrRect viewport{0, 0, W, H};

        while (running_.load()) {
            next_tick += frame_period;

            // Update abstract scene under sim lock so we read consistent mjData.
            if (sim_mtx_) {
                std::lock_guard<std::recursive_mutex> lock(*sim_mtx_);
                mjv_updateScene(m_, d_, &opt_, nullptr, &cam_, mjCAT_ALL, &scn_);
            } else {
                mjv_updateScene(m_, d_, &opt_, nullptr, &cam_, mjCAT_ALL, &scn_);
            }

            // Render and read pixels (no lock needed -- scn_ is local).
            mjr_render(viewport, &scn_, &con_);
            mjr_readPixels(rgb_.data(), nullptr, viewport, &con_);

            // OpenGL returns rows bottom-to-top; flip to top-down for JPEG.
            const int row_bytes = 3 * W;
            for (int y = 0; y < H; ++y) {
                std::memcpy(rgb_flipped_.data() + y * row_bytes,
                            rgb_.data() + (H - 1 - y) * row_bytes,
                            row_bytes);
            }

            // JPEG-encode and publish.
            unsigned char *jpeg_buf = nullptr;
            unsigned long jpeg_size = 0;
            int rc = tjCompress2(tj_, rgb_flipped_.data(), W, 0, H,
                                 TJPF_RGB, &jpeg_buf, &jpeg_size,
                                 TJSAMP_420, cfg_.jpeg_quality, TJFLAG_FASTDCT);
            if (rc == 0 && jpeg_buf && jpeg_size > 0) {
                zmq_send(zmq_sock_, jpeg_buf, jpeg_size, 0);
            } else {
                std::fprintf(stderr, "[image_server] tjCompress2 failed: %s\n", tjGetErrorStr2(tj_));
            }
            if (jpeg_buf) tjFree(jpeg_buf);

            std::this_thread::sleep_until(next_tick);
        }

        // Cleanup.
        tjDestroy(tj_);
        tj_ = nullptr;
        zmq_close(zmq_sock_);
        zmq_ctx_term(zmq_ctx_);
        zmq_sock_ = nullptr;
        zmq_ctx_ = nullptr;
        mjr_freeContext(&con_);
        mjv_freeScene(&scn_);
        glfwMakeContextCurrent(nullptr);
    }

    // Inputs.
    mjModel *m_ = nullptr;
    mjData *d_ = nullptr;
    std::recursive_mutex *sim_mtx_ = nullptr;
    GLFWwindow *gl_window_ = nullptr;
    Config cfg_;

    // State.
    std::atomic<bool> running_{false};
    std::thread thread_;
    int cam_id_ = -1;

    // MuJoCo render state (owned by this thread).
    mjvScene scn_{};
    mjvCamera cam_{};
    mjvOption opt_{};
    mjrContext con_{};

    // ZMQ + JPEG.
    void *zmq_ctx_ = nullptr;
    void *zmq_sock_ = nullptr;
    tjhandle tj_ = nullptr;

    // Pixel buffers.
    std::vector<unsigned char> rgb_;
    std::vector<unsigned char> rgb_flipped_;
};

}  // namespace image_server
