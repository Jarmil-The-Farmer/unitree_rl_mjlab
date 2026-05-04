#pragma once

#include <iostream>
#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>

namespace param
{

inline struct SimulationConfig
{
    std::string robot;
    std::filesystem::path robot_scene;

    int domain_id;
    std::string interface;

    int use_joystick;
    std::string joystick_type;
    std::string joystick_device;
    int joystick_bits;

    int print_scene_information;

    int enable_elastic_band;
    int band_attached_link = 0;

    // Image server (mirror of teleop image_server.py running on the real robot).
    int enable_image_server = 0;
    int image_server_port = 5555;
    int image_server_fps = 30;
    int image_server_width = 640;
    int image_server_height = 480;
    int image_server_jpeg_quality = 80;
    std::string image_server_camera = "head_cam";

    int enable_inspire_hand_modbus_server = 0;
    int inspire_hand_modbus_port = 6000;
    int inspire_hand_modbus_left_device_id = 2;
    int inspire_hand_modbus_right_device_id = 1;

    void load_from_yaml(const std::string &filename)
    {
        auto cfg = YAML::LoadFile(filename);
        try
        {
            robot = cfg["robot"].as<std::string>();
            robot_scene = cfg["robot_scene"].as<std::string>();
            domain_id = cfg["domain_id"].as<int>();
            interface = cfg["interface"].as<std::string>();
            use_joystick = cfg["use_joystick"].as<int>();
            joystick_type = cfg["joystick_type"].as<std::string>();
            joystick_device = cfg["joystick_device"].as<std::string>();
            joystick_bits = cfg["joystick_bits"].as<int>();
            print_scene_information = cfg["print_scene_information"].as<int>();
            enable_elastic_band = cfg["enable_elastic_band"].as<int>();

            // Optional image server section.
            if (cfg["image_server"]) {
                auto is = cfg["image_server"];
                if (is["enable"]) enable_image_server = is["enable"].as<int>();
                if (is["port"]) image_server_port = is["port"].as<int>();
                if (is["fps"]) image_server_fps = is["fps"].as<int>();
                if (is["width"]) image_server_width = is["width"].as<int>();
                if (is["height"]) image_server_height = is["height"].as<int>();
                if (is["jpeg_quality"]) image_server_jpeg_quality = is["jpeg_quality"].as<int>();
                if (is["camera"]) image_server_camera = is["camera"].as<std::string>();
            }

            // Optional fake Inspire Modbus TCP server. It mirrors the register
            // writes used by inspire_hand_ws and drives MuJoCo finger joints.
            if (cfg["inspire_hand_modbus_server"]) {
                auto ms = cfg["inspire_hand_modbus_server"];
                if (ms["enable"]) enable_inspire_hand_modbus_server = ms["enable"].as<int>();
                if (ms["port"]) inspire_hand_modbus_port = ms["port"].as<int>();
                if (ms["left_device_id"]) inspire_hand_modbus_left_device_id = ms["left_device_id"].as<int>();
                if (ms["right_device_id"]) inspire_hand_modbus_right_device_id = ms["right_device_id"].as<int>();
            }
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }
    }
} config;

/* ---------- Command Line Parameters ---------- */
namespace po = boost::program_options;

//※ This function must be called at the beginning of main() function
inline po::variables_map helper(int argc, char** argv)
{
    po::options_description desc("Unitree Mujoco");
    desc.add_options()
        ("help,h", "Show help message")
        ("domain_id,i", po::value<int>(&config.domain_id), "DDS domain ID; -i 0")
        ("network,n", po::value<std::string>(&config.interface), "DDS network interface; -n eth0")
        ("robot,r", po::value<std::string>(&config.robot), "Robot type; -r go2")
        ("scene,s", po::value<std::filesystem::path>(&config.robot_scene), "Robot scene file; -s scene_terrain.xml")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        exit(0);
    }

    return vm;
}

}
