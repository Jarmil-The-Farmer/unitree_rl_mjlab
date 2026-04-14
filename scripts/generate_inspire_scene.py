#!/usr/bin/env python3
"""Generate scene_g1_inspire.xml for the simulate app.

Uses the same robot spec as training (URDF + programmatic augmentation) and adds
the actuator/sensor/scene setup required by the simulate DDS bridge.

Usage: python scripts/generate_inspire_scene.py
"""

import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco

from src.assets.robots.unitree_g1.g1_inspire_constants import get_inspire_spec

# 29 body joints in DDS bridge order (must match scene_g1.xml exactly).
BODY_JOINTS = [
    ("left_hip_pitch", "left_hip_pitch_joint", -88, 88),
    ("left_hip_roll", "left_hip_roll_joint", -139, 139),
    ("left_hip_yaw", "left_hip_yaw_joint", -88, 88),
    ("left_knee", "left_knee_joint", -139, 139),
    ("left_ankle_pitch", "left_ankle_pitch_joint", -50, 50),
    ("left_ankle_roll", "left_ankle_roll_joint", -50, 50),
    ("right_hip_pitch", "right_hip_pitch_joint", -88, 88),
    ("right_hip_roll", "right_hip_roll_joint", -139, 139),
    ("right_hip_yaw", "right_hip_yaw_joint", -88, 88),
    ("right_knee", "right_knee_joint", -139, 139),
    ("right_ankle_pitch", "right_ankle_pitch_joint", -25, 25),
    ("right_ankle_roll", "right_ankle_roll_joint", -25, 25),
    ("waist_yaw", "waist_yaw_joint", -88, 88),
    ("waist_roll", "waist_roll_joint", -25, 25),
    ("waist_pitch", "waist_pitch_joint", -25, 25),
    ("left_shoulder_pitch", "left_shoulder_pitch_joint", -25, 25),
    ("left_shoulder_roll", "left_shoulder_roll_joint", -25, 25),
    ("left_shoulder_yaw", "left_shoulder_yaw_joint", -25, 25),
    ("left_elbow", "left_elbow_joint", -25, 25),
    ("left_wrist_roll", "left_wrist_roll_joint", -25, 25),
    ("left_wrist_pitch", "left_wrist_pitch_joint", -5, 5),
    ("left_wrist_yaw", "left_wrist_yaw_joint", -5, 5),
    ("right_shoulder_pitch", "right_shoulder_pitch_joint", -25, 25),
    ("right_shoulder_roll", "right_shoulder_roll_joint", -25, 25),
    ("right_shoulder_yaw", "right_shoulder_yaw_joint", -25, 25),
    ("right_elbow", "right_elbow_joint", -25, 25),
    ("right_wrist_roll", "right_wrist_roll_joint", -25, 25),
    ("right_wrist_pitch", "right_wrist_pitch_joint", -5, 5),
    ("right_wrist_yaw", "right_wrist_yaw_joint", -5, 5),
]

OUTPUT = PROJECT_ROOT / "src" / "assets" / "robots" / "unitree_g1" / "xmls" / "scene_g1_inspire.xml"


def main():
    # --- Step 1: Build robot spec (same as training) ---
    print("[1/5] Loading inspire robot spec from URDF...")
    spec = get_inspire_spec()

    # Set pelvis spawn height (matching scene_g1.xml: 0.793m above ground).
    pelvis = spec.body("pelvis")
    pelvis.pos = [0, 0, 0.793]

    # Add "imu" site on pelvis — bridge sensors reference this name.
    # (The spec already has "imu_in_pelvis" but bridge expects "imu".)
    imu_site = pelvis.add_site()
    imu_site.name = "imu"
    imu_site.pos = [0, 0, 0]
    imu_site.size = [0.01, 0, 0]

    # --- Step 2: Export spec to XML ---
    print("[2/5] Exporting spec to XML...")
    xml_string = spec.to_xml()

    # --- Step 3: Parse and modify XML for simulate bridge ---
    print("[3/5] Adding actuators, sensors, and scene elements...")
    root = ET.fromstring(xml_string)
    tree = ET.ElementTree(root)

    # Set model name.
    root.set("model", "scene_g1_inspire")

    # Fix compiler: set meshdir relative to output XML location.
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("angle", "radian")
    compiler.set("meshdir", "../urdf/meshes/")
    # Remove attrs that mj_saveLastXML may add (autolimits, etc.)
    for attr in list(compiler.attrib):
        if attr not in ("angle", "meshdir"):
            del compiler.attrib[attr]

    # Set default joint properties (matching scene_g1.xml).
    for d in root.findall("default"):
        root.remove(d)
    default_elem = ET.SubElement(root, "default")
    joint_def = ET.SubElement(default_elem, "joint")
    joint_def.set("damping", "0.05")
    joint_def.set("armature", "0.01")
    joint_def.set("frictionloss", "0.2")

    # Remove existing actuator/sensor sections (from training spec augmentation).
    for tag in ("actuator", "sensor"):
        for elem in root.findall(tag):
            root.remove(elem)

    # Add motor actuators for 29 body joints.
    act_elem = ET.SubElement(root, "actuator")
    for act_name, jnt_name, lo, hi in BODY_JOINTS:
        m = ET.SubElement(act_elem, "motor")
        m.set("name", act_name)
        m.set("joint", jnt_name)
        m.set("ctrlrange", f"{lo} {hi}")

    # Add sensors in bridge-expected order:
    #   [29 x jointpos] [29 x jointvel] [29 x jointactuatorfrc] [IMU by name]
    sens_elem = ET.SubElement(root, "sensor")

    for act_name, jnt_name, _, _ in BODY_JOINTS:
        s = ET.SubElement(sens_elem, "jointpos")
        s.set("name", f"{act_name}_pos")
        s.set("joint", jnt_name)

    for act_name, jnt_name, _, _ in BODY_JOINTS:
        s = ET.SubElement(sens_elem, "jointvel")
        s.set("name", f"{act_name}_vel")
        s.set("joint", jnt_name)

    for act_name, jnt_name, _, _ in BODY_JOINTS:
        s = ET.SubElement(sens_elem, "jointactuatorfrc")
        s.set("name", f"{act_name}_torque")
        s.set("joint", jnt_name)

    # IMU sensors referencing "imu" site (names must match bridge expectations).
    # gyro/accelerometer use "site" attr; frame* sensors use "objtype"/"objname".
    for tag, name, use_site_attr in [
        ("framequat", "imu_quat", False),
        ("gyro", "imu_gyro", True),
        ("accelerometer", "imu_acc", True),
        ("framepos", "frame_pos", False),
        ("framelinvel", "frame_vel", False),
    ]:
        s = ET.SubElement(sens_elem, tag)
        s.set("name", name)
        if use_site_attr:
            s.set("site", "imu")
        else:
            s.set("objtype", "site")
            s.set("objname", "imu")

    # Scene visual setup (matching scene_g1.xml).
    stat = ET.SubElement(root, "statistic")
    stat.set("center", "1.0 0.7 1.0")
    stat.set("extent", "0.8")

    visual = ET.SubElement(root, "visual")
    hl = ET.SubElement(visual, "headlight")
    hl.set("diffuse", "0.6 0.6 0.6")
    hl.set("ambient", "0.1 0.1 0.1")
    hl.set("specular", "0.9 0.9 0.9")
    rgba_elem = ET.SubElement(visual, "rgba")
    rgba_elem.set("haze", "0.15 0.25 0.35 1")
    glob = ET.SubElement(visual, "global")
    glob.set("azimuth", "-140")
    glob.set("elevation", "-20")

    # Scene assets (skybox + ground material).
    asset2 = ET.SubElement(root, "asset")
    for tag, attrs in [
        ("texture", {"type": "skybox", "builtin": "flat", "rgb1": "0 0 0",
                      "rgb2": "0 0 0", "width": "512", "height": "3072"}),
        ("texture", {"type": "2d", "name": "groundplane", "builtin": "checker",
                      "mark": "edge", "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.2 0.3",
                      "markrgb": "0.8 0.8 0.8", "width": "300", "height": "300"}),
        ("material", {"name": "groundplane", "texture": "groundplane",
                       "texuniform": "true", "texrepeat": "5 5", "reflectance": "0.2"}),
    ]:
        e = ET.SubElement(asset2, tag)
        for k, v in attrs.items():
            e.set(k, v)

    # Scene worldbody (light + floor). MuJoCo merges multiple worldbody sections.
    wb2 = ET.SubElement(root, "worldbody")
    light = ET.SubElement(wb2, "light")
    light.set("pos", "1 0 3.5")
    light.set("dir", "0 0 -1")
    light.set("directional", "true")
    floor = ET.SubElement(wb2, "geom")
    floor.set("name", "floor")
    floor.set("size", "0 0 0.05")
    floor.set("type", "plane")
    floor.set("material", "groundplane")

    # --- Step 4: Write output ---
    print("[4/5] Writing output XML...")
    ET.indent(tree, space="  ")
    xml_str = ET.tostring(root, encoding="unicode")
    OUTPUT.write_text(
        f"<!-- Auto-generated by scripts/generate_inspire_scene.py\n"
        f"     Do not edit manually. Regenerate with:\n"
        f"       python scripts/generate_inspire_scene.py\n"
        f"-->\n{xml_str}\n"
    )

    # --- Step 5: Verify ---
    print("[5/5] Verifying output loads correctly...")
    try:
        verify_model = mujoco.MjModel.from_xml_path(str(OUTPUT))
        nu = verify_model.nu
        ns = verify_model.nsensor
        njnt = verify_model.njnt
        print(f"  joints={njnt}, actuators={nu}, sensors={ns}")
        assert nu == 29, f"Expected 29 actuators, got {nu}"
        assert ns == 29 * 3 + 5, f"Expected {29*3+5} sensors, got {ns}"
        print("  All checks passed!")
    except Exception as e:
        print(f"  VERIFICATION FAILED: {e}")
        return 1

    print(f"\nDone: {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
