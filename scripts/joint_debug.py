"""Debug joint positions in MuJoCo viewer.

Spawns a single G1 Inspire robot with configurable initial joint positions.
The MuJoCo viewer opens with joint sliders so you can interactively adjust
each joint and read its position.

Usage:
  python scripts/joint_debug.py
"""

import mujoco
import mujoco.viewer
import numpy as np

from src.assets.robots.unitree_g1.g1_inspire_constants import get_inspire_spec

# ── Joint positions (training defaults from INSPIRE_BALANCE_HOME_KEYFRAME) ──
# Values here match what the RL policy sees during training.
# Edit any value and re-run to see the effect.
JOINT_POS: dict[str, float] = {
  # ── LEFT LEG ──
  "left_hip_pitch_joint": -0.15,       # Kycel, predozadni sklon stehna (- = stehno dozadu)
  "left_hip_roll_joint": 0.0,          # Kycel, unozeneni stehna do stran (+ = noha od tela)
  "left_hip_yaw_joint": 0.0,           # Kycel, rotace stehna kolem svisle osy
  "left_knee_joint": 0.3,              # Koleno, ohyb (+ = ohnuté koleno)
  "left_ankle_pitch_joint": -0.15,     # Kotnik, predozadni naklon chodidla (- = spicka nahoru)
  "left_ankle_roll_joint": 0.0,        # Kotnik, bocni naklon chodidla

  # ── RIGHT LEG ──
  "right_hip_pitch_joint": -0.15,      # Kycel, predozadni sklon stehna (- = stehno dozadu)
  "right_hip_roll_joint": 0.0,         # Kycel, unozeneni stehna do stran (- = noha od tela)
  "right_hip_yaw_joint": 0.0,          # Kycel, rotace stehna kolem svisle osy
  "right_knee_joint": 0.3,             # Koleno, ohyb (+ = ohnuté koleno)
  "right_ankle_pitch_joint": -0.15,    # Kotnik, predozadni naklon chodidla (- = spicka nahoru)
  "right_ankle_roll_joint": 0.0,       # Kotnik, bocni naklon chodidla

  # ── WAIST (PAS/TRUP) ──
  "waist_yaw_joint": 0.0,              # Rotace trupu doleva/doprava
  "waist_roll_joint": 0.0,             # Ukloneni trupu do stran
  "waist_pitch_joint": 0.0,            # Predklon/zaklon trupu

  # ── LEFT ARM ──
  "left_shoulder_pitch_joint": -1.4,   # Rameno, zdvih paze dopredu/dozadu (-1.57 = paze rovne dopredu)
  "left_shoulder_roll_joint": 0.18,    # Rameno, upazeneni (+ = paze od tela)
  "left_shoulder_yaw_joint": 0.0,      # Rameno, rotace paze kolem jeji osy
  "left_elbow_joint": 1.57,            # Loket, ohyb (0 = pravy uhel)
  "left_wrist_roll_joint": 0.0,        # Zapesti, rotace (supinace/pronace)
  "left_wrist_pitch_joint": 0.0,       # Zapesti, ohyb nahoru/dolu
  "left_wrist_yaw_joint": 0.0,         # Zapesti, uhyb do stran

  # ── RIGHT ARM ──
  "right_shoulder_pitch_joint": -1.4,  # Rameno, zdvih paze dopredu/dozadu (-1.57 = paze rovne dopredu)
  "right_shoulder_roll_joint": -0.18,  # Rameno, upazeneni (- = paze od tela)
  "right_shoulder_yaw_joint": 0.0,     # Rameno, rotace paze kolem jeji osy
  "right_elbow_joint": 1.57,           # Loket, ohyb (0 = pravy uhel)
  "right_wrist_roll_joint": 0.0,       # Zapesti, rotace (supinace/pronace)
  "right_wrist_pitch_joint": 0.0,      # Zapesti, ohyb nahoru/dolu
  "right_wrist_yaw_joint": 0.0,        # Zapesti, uhyb do stran

  # ── LEFT HAND (INSPIRE) ──
  "left_thumb_1_joint": 0.0,           # Palec, zakladni kloub - abdukce/addukce
  "left_thumb_2_joint": 0.0,           # Palec, stredni kloub - ohyb
  "left_thumb_3_joint": 0.0,           # Palec, koncovy kloub - ohyb
  "left_thumb_4_joint": 0.0,           # Palec, rotace
  "left_index_1_joint": 0.0,           # Ukazovacek, zakladni kloub (MCP)
  "left_index_2_joint": 0.0,           # Ukazovacek, stredni/koncovy kloub (PIP+DIP)
  "left_middle_1_joint": 0.0,          # Prostrednicek, zakladni kloub (MCP)
  "left_middle_2_joint": 0.0,          # Prostrednicek, stredni/koncovy kloub (PIP+DIP)
  "left_ring_1_joint": 0.0,            # Prstynek, zakladni kloub (MCP)
  "left_ring_2_joint": 0.0,            # Prstynek, stredni/koncovy kloub (PIP+DIP)
  "left_little_1_joint": 0.0,          # Malicek, zakladni kloub (MCP)
  "left_little_2_joint": 0.0,          # Malicek, stredni/koncovy kloub (PIP+DIP)

  # ── RIGHT HAND (INSPIRE) ──
  "right_thumb_1_joint": 0.0,          # Palec, zakladni kloub - abdukce/addukce
  "right_thumb_2_joint": 0.0,          # Palec, stredni kloub - ohyb
  "right_thumb_3_joint": 0.0,          # Palec, koncovy kloub - ohyb
  "right_thumb_4_joint": 0.0,          # Palec, rotace
  "right_index_1_joint": 0.0,          # Ukazovacek, zakladni kloub (MCP)
  "right_index_2_joint": 0.0,          # Ukazovacek, stredni/koncovy kloub (PIP+DIP)
  "right_middle_1_joint": 0.0,         # Prostrednicek, zakladni kloub (MCP)
  "right_middle_2_joint": 0.0,         # Prostrednicek, stredni/koncovy kloub (PIP+DIP)
  "right_ring_1_joint": 0.0,           # Prstynek, zakladni kloub (MCP)
  "right_ring_2_joint": 0.0,           # Prstynek, stredni/koncovy kloub (PIP+DIP)
  "right_little_1_joint": 0.0,         # Malicek, zakladni kloub (MCP)
  "right_little_2_joint": 0.0,         # Malicek, stredni/koncovy kloub (PIP+DIP)
}

# ── Build model ──
spec = get_inspire_spec()

# Add ground plane (visual reference only, robot hovers above it).
ground = spec.worldbody.add_geom()
ground.type = mujoco.mjtGeom.mjGEOM_PLANE
ground.size = [10, 10, 0.01]
ground.rgba = [0.4, 0.4, 0.4, 1.0]
ground.name = "ground"
ground.contype = 0
ground.conaffinity = 0

# Add ambient and directional light.
light = spec.worldbody.add_light()
light.pos = [0, 0, 3]
light.dir = [0, 0, -1]
light.diffuse = [0.8, 0.8, 0.8]
light.ambient = [0.4, 0.4, 0.4]

# Disable gravity so the robot holds its pose without collapsing.
spec.option.gravity = [0, 0, 0]
# Disable contacts so nothing bounces.
spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

model = spec.compile()
data = mujoco.MjData(model)

# Apply joint positions.
for name, value in JOINT_POS.items():
  jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
  if jnt_id == -1:
    print(f"[WARN] Joint '{name}' not found in model, skipping.")
    continue
  qpos_adr = model.jnt_qposadr[jnt_id]
  data.qpos[qpos_adr] = value

# Set freejoint: pelvis at z=0.8 (gravity is off, robot holds pose).
fj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
assert fj_id != -1
fj_adr = model.jnt_qposadr[fj_id]
data.qpos[fj_adr + 2] = 0.8   # z position
data.qpos[fj_adr + 3] = 1.0   # quaternion w
mujoco.mj_forward(model, data)

# Print initial joint positions.
print("\n── Initial joint positions ──")
for i in range(model.njnt):
  name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
  if name == "floating_base_joint":
    continue
  adr = model.jnt_qposadr[i]
  print(f"  {name:40s} = {data.qpos[adr]:+.4f} rad ({np.degrees(data.qpos[adr]):+.1f}deg)")

print("\nOpening MuJoCo viewer... Use the joint sliders in the right panel to adjust.")
print("(Right panel -> expand 'Joint' section)")

mujoco.viewer.launch(model, data)
