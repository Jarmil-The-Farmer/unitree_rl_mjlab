"""Test arm collision detection with interactive MuJoCo viewer.

Spawns G1 Inspire with arm collision capsules ENABLED from start.
Robot is in deep squat with arms at sides — arms should penetrate legs.
Contact points are visualized as red/blue markers.

Press Space to cycle poses, C to toggle arm collisions on/off.

Usage:
  python scripts/test_arm_collisions.py
"""

import time

import mujoco
import mujoco.viewer

from src.assets.robots.unitree_g1.g1_inspire_constants import get_inspire_spec

# ── Build model with arm collision capsules ──

spec = get_inspire_spec()

# Ground plane.
ground = spec.worldbody.add_geom()
ground.type = mujoco.mjtGeom.mjGEOM_PLANE
ground.size = [10, 10, 0.01]
ground.name = "ground"

# Arm collision capsules — same as the ones commented out in g1_inspire_constants.py.
# Added here with collisions ENABLED so we can see contacts.
ARM_CAPSULES = [
  ("left_shoulder_yaw_link",  "left_shoulder_col",  [0.035], [0, 0, -0.08, 0, 0, 0.05]),
  ("right_shoulder_yaw_link", "right_shoulder_col", [0.035], [0, 0, -0.08, 0, 0, 0.05]),
  ("left_elbow_link",         "left_elbow_col",     [0.035], [-0.01, 0, -0.01, 0.08, 0, -0.01]),
  ("right_elbow_link",        "right_elbow_col",    [0.035], [-0.01, 0, -0.01, 0.08, 0, -0.01]),
  ("left_wrist_pitch_link",   "left_wrist_col",     [0.035], [-0.01, 0, 0, 0.06, 0, 0]),
  ("right_wrist_pitch_link",  "right_wrist_col",    [0.035], [-0.01, 0, 0, 0.06, 0, 0]),
]
ARM_CAPSULE_NAMES = []
for body_name, geom_name, size, fromto in ARM_CAPSULES:
  body = spec.body(body_name)
  g = body.add_geom()
  g.name = geom_name
  g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
  g.size = size + [0, 0]
  g.fromto = fromto
  g.group = 3
  g.contype = 1         # ENABLED
  g.conaffinity = 1
  g.condim = 1
  g.rgba = [1, 0, 0, 0.4]
  ARM_CAPSULE_NAMES.append(geom_name)

# Disable gravity.
spec.option.gravity = [0, 0, 0]

model = spec.compile()
data = mujoco.MjData(model)

# Make contact visualization bigger.
model.vis.scale.contactwidth = 0.05
model.vis.scale.contactheight = 0.02

# ── Helpers ──

fj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
fj_adr = model.jnt_qposadr[fj_id]


def set_joint(name, value):
  jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
  if jid >= 0:
    data.qpos[model.jnt_qposadr[jid]] = value


def reset_robot():
  mujoco.mj_resetData(model, data)
  data.qpos[fj_adr + 2] = 0.8
  data.qpos[fj_adr + 3] = 1.0


POSES = [
  ("Arms at sides + deep squat (OVERLAP)", {
    "shoulder_pitch": 0.0, "elbow": 1.57,
    "hip_pitch": -1.5, "knee": 2.8, "ankle_pitch": -0.8,
  }),
  ("Arms at sides (standing)", {
    "shoulder_pitch": 0.0, "elbow": 1.57,
  }),
  ("Arms extended forward", {
    "shoulder_pitch": -1.6, "elbow": 1.57,
  }),
  ("Arms at sides + max squat", {
    "shoulder_pitch": 0.0, "elbow": 1.57,
    "hip_pitch": -1.8, "knee": 2.8, "ankle_pitch": -0.8,
  }),
]


def apply_pose(idx):
  label, joints = POSES[idx]
  reset_robot()
  for side in ("left", "right"):
    for key, val in joints.items():
      set_joint(f"{side}_{key}_joint", val)
  return label


# ── State ──

pose_idx = [0]
collisions_on = [True]


def toggle_collisions():
  collisions_on[0] = not collisions_on[0]
  val = 1 if collisions_on[0] else 0
  for name in ARM_CAPSULE_NAMES:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid >= 0:
      model.geom_contype[gid] = val
      model.geom_conaffinity[gid] = val
  print(f"  [C] Arm collisions: {'ON' if collisions_on[0] else 'OFF'}")


def key_callback(keycode):
  if keycode == ord(" "):
    pose_idx[0] = (pose_idx[0] + 1) % len(POSES)
    label = apply_pose(pose_idx[0])
    print(f"  [Space] -> {label}")
  elif keycode in (ord("c"), ord("C")):
    toggle_collisions()


# ── Initial pose ──

label = apply_pose(0)

# Verify contacts work before opening viewer.
mujoco.mj_step(model, data)
print(f"Initial pose: {label}")
print(f"Contacts detected: {data.ncon}")
for i in range(data.ncon):
  c = data.contact[i]
  g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"unnamed_{c.geom1}"
  g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"unnamed_{c.geom2}"
  print(f"  {g1} <-> {g2}  dist={c.dist:.4f}")

# Re-apply pose (mj_step may have moved things).
apply_pose(0)

print("\nOpening viewer...")
print("Controls: Space = next pose, C = toggle arm collisions")
print("Arm collisions: ON")

# ── Viewer loop ──

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
  viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
  viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

  step_count = 0
  while viewer.is_running():
    # Step physics (needed for contact detection).
    mujoco.mj_step(model, data)

    # Pin the robot in place (gravity is off but contacts push it).
    data.qpos[fj_adr + 2] = 0.8
    data.qpos[fj_adr + 3] = 1.0
    # Re-apply current pose joints each step to keep robot static.
    _, joints = POSES[pose_idx[0]]
    for side in ("left", "right"):
      for key, val in joints.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{key}_joint")
        if jid >= 0:
          data.qpos[model.jnt_qposadr[jid]] = val

    # Print contacts periodically.
    step_count += 1
    if step_count % 100 == 0 and data.ncon > 0:
      arm_contacts = 0
      for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
        if any(n in g1 or n in g2 for n in ARM_CAPSULE_NAMES):
          arm_contacts += 1
          if step_count % 500 == 0:
            print(f"  ARM: {g1} <-> {g2} dist={c.dist:.4f}")
      if arm_contacts > 0 and step_count % 500 == 0:
        print(f"  ({arm_contacts} arm contacts, {data.ncon} total)")

    viewer.sync()
    time.sleep(0.002)
