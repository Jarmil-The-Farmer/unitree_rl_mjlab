"""Unitree G1 with Inspire Hands constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

from .g1_constants import (
  ACTUATOR_5020,
  ARMATURE_4010,
  DAMPING_4010,
  DAMPING_5020,
  DAMPING_RATIO,
  G1_ACTUATOR_4010,
  G1_ACTUATOR_5020,
  G1_ACTUATOR_7520_14,
  G1_ACTUATOR_7520_22,
  G1_ACTUATOR_ANKLE,
  G1_ACTUATOR_WAIST,
  NATURAL_FREQ,
  STIFFNESS_4010,
  STIFFNESS_5020,
)

##
# MJCF and assets.
##

G1_INSPIRE_URDF: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_g1" / "urdf" / "g1_inspire.urdf"
)
assert G1_INSPIRE_URDF.exists()


def _get_inspire_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, G1_INSPIRE_URDF.parent / "meshes", meshdir)
  return assets


# Collision capsule definitions from g1.xml.
# Each entry: (body_name, geom_name, geom_type, size, pos_or_fromto).
# geom_type: "capsule" uses fromto, "sphere" uses pos.
_COLLISION_GEOMS: list[tuple[str, str, str, list[float], list[float]]] = [
  # Pelvis.
  ("pelvis", "pelvis_collision", "sphere", [0.07], [0, 0, -0.08]),
  # Left leg.
  ("left_hip_roll_link", "left_hip_collision", "capsule",
   [0.06], [0.02, 0, 0, 0.02, 0, -0.08]),
  ("left_hip_yaw_link", "left_thigh_collision", "capsule",
   [0.055], [0, 0, -0.03, -0.06, 0, -0.17]),
  ("left_knee_link", "left_shin_collision", "capsule",
   [0.045], [0.01, 0, 0, 0.01, 0, -0.15]),
  ("left_knee_link", "left_linkage_brace_collision", "capsule",
   [0.03], [0.01, 0, -0.2, 0.01, 0, -0.28]),
  # Right leg.
  ("right_hip_roll_link", "right_hip_collision", "capsule",
   [0.06], [0.02, 0, 0, 0.02, 0, -0.08]),
  ("right_hip_yaw_link", "right_thigh_collision", "capsule",
   [0.055], [0, 0, -0.03, -0.06, 0, -0.17]),
  ("right_knee_link", "right_shin_collision", "capsule",
   [0.045], [0.01, 0, 0, 0.01, 0, -0.15]),
  ("right_knee_link", "right_linkage_brace_collision", "capsule",
   [0.03], [0.01, 0, -0.2, 0.01, 0, -0.28]),
  # Torso.
  ("torso_link", "torso_collision", "capsule",
   [0.09], [0.01, 0, 0.08, 0.01, 0, 0.2]),
  ("torso_link", "head_collision", "sphere", [0.06], [0, 0, 0.43]),
  # Arms.
  ("left_shoulder_yaw_link", "left_shoulder_yaw_collision", "capsule",
   [0.035], [0, 0, -0.08, 0, 0, 0.05]),
  ("right_shoulder_yaw_link", "right_shoulder_yaw_collision", "capsule",
   [0.035], [0, 0, -0.08, 0, 0, 0.05]),
  ("left_elbow_link", "left_elbow_yaw_collision", "capsule",
   [0.035], [-0.01, 0, -0.01, 0.08, 0, -0.01]),
  ("right_elbow_link", "right_elbow_yaw_collision", "capsule",
   [0.035], [-0.01, 0, -0.01, 0.08, 0, -0.01]),
  ("left_wrist_pitch_link", "left_wrist_collision", "capsule",
   [0.035], [-0.01, 0, 0, 0.06, 0, 0]),
  ("right_wrist_pitch_link", "right_wrist_collision", "capsule",
   [0.035], [-0.01, 0, 0, 0.06, 0, 0]),
]

# Foot collision capsules (7 per foot, size=0.01).
_FOOT_GEOMS: dict[str, list[list[float]]] = {
  "left": [
    [0.1, -0.026, -0.025, 0.05, -0.027, -0.025],
    [-0.044, -0.018, -0.025, 0.123, -0.018, -0.025],
    [-0.052, -0.01, -0.025, 0.13, -0.01, -0.025],
    [-0.054, 0, -0.025, 0.132, 0, -0.025],
    [-0.052, 0.01, -0.025, 0.13, 0.01, -0.025],
    [-0.044, 0.018, -0.025, 0.123, 0.018, -0.025],
    [0.1, 0.026, -0.025, 0.05, 0.026, -0.025],
  ],
  "right": [
    [0.1, -0.026, -0.025, 0.05, -0.026, -0.025],
    [-0.044, -0.018, -0.025, 0.123, -0.018, -0.025],
    [-0.052, -0.01, -0.025, 0.13, -0.01, -0.025],
    [-0.054, 0, -0.025, 0.132, 0, -0.025],
    [-0.052, 0.01, -0.025, 0.13, 0.01, -0.025],
    [-0.044, 0.018, -0.025, 0.123, 0.018, -0.025],
    [0.1, 0.026, -0.025, 0.05, 0.026, -0.025],
  ],
}

# Contact exclusions (same as g1.xml).
_CONTACT_EXCLUDES = [
  ("left_elbow_link", "left_wrist_pitch_link"),
  ("right_elbow_link", "right_wrist_pitch_link"),
  ("pelvis", "right_hip_roll_link"),
  ("pelvis", "left_hip_roll_link"),
]


def _augment_inspire_spec(spec: mujoco.MjSpec) -> None:
  """Add freejoint, collision geoms, sites, and sensors to URDF-loaded spec."""
  pelvis = spec.body("pelvis")

  # Freejoint.
  fj = pelvis.add_freejoint()
  fj.name = "floating_base_joint"

  # Disable all existing URDF mesh collision geoms. The URDF generates unnamed
  # mesh geoms for every link which create excessive contacts. We replace them
  # with lightweight named collision capsules below.
  for geom in spec.geoms:
    if not geom.name:
      geom.contype = 0
      geom.conaffinity = 0

  # IMU site on pelvis.
  imu = pelvis.add_site()
  imu.name = "imu_in_pelvis"
  imu.pos = [0.04525, 0, -0.08339]
  imu.size = [0.01, 0, 0]
  imu.rgba = [1, 0, 0, 1]
  imu.group = 5

  # Body collision geoms.
  for body_name, geom_name, geom_type, size, coords in _COLLISION_GEOMS:
    body = spec.body(body_name)
    g = body.add_geom()
    g.name = geom_name
    g.group = 3
    g.priority = 1
    g.condim = 6
    if geom_type == "capsule":
      g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
      g.size = size + [0, 0]
      g.fromto = coords
    else:  # sphere
      g.type = mujoco.mjtGeom.mjGEOM_SPHERE
      g.size = size + [0, 0]
      g.pos = coords

  # Foot collision capsules and sites.
  for side, fromto_list in _FOOT_GEOMS.items():
    ankle = spec.body(f"{side}_ankle_roll_link")
    for i, fromto in enumerate(fromto_list, 1):
      g = ankle.add_geom()
      g.name = f"{side}_foot{i}_collision"
      g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
      g.size = [0.01, 0, 0]
      g.fromto = fromto
      g.group = 3
      g.priority = 1
      g.condim = 6

    foot_site = ankle.add_site()
    foot_site.name = f"{side}_foot"
    foot_site.pos = [0.04, 0, -0.035]
    foot_site.rgba = [1, 0, 0, 1]
    foot_site.group = 5

  # Sensors.
  for name, stype in [
    ("imu_ang_vel", mujoco.mjtSensor.mjSENS_GYRO),
    ("imu_lin_vel", mujoco.mjtSensor.mjSENS_VELOCIMETER),
    ("imu_lin_acc", mujoco.mjtSensor.mjSENS_ACCELEROMETER),
  ]:
    s = spec.add_sensor()
    s.name = name
    s.type = stype
    s.objtype = mujoco.mjtObj.mjOBJ_SITE
    s.objname = "imu_in_pelvis"

  angmom = spec.add_sensor()
  angmom.name = "root_angmom"
  angmom.type = mujoco.mjtSensor.mjSENS_SUBTREEANGMOM
  angmom.objtype = mujoco.mjtObj.mjOBJ_BODY
  angmom.objname = "pelvis"

  # Contact exclusions.
  for body1, body2 in _CONTACT_EXCLUDES:
    excl = spec.add_exclude()
    excl.name = f"exclude_{body1}_{body2}"
    excl.bodyname1 = body1
    excl.bodyname2 = body2


def get_inspire_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(G1_INSPIRE_URDF))
  spec.assets = _get_inspire_assets(spec.meshdir)
  _augment_inspire_spec(spec)
  return spec


##
# Finger actuator config.
##

# Inspire hand finger joints use small motors with effort_limit=10 N*m.
# Use the 4010 actuator parameters scaled for finger joints.
FINGER_ARMATURE = ARMATURE_4010
FINGER_STIFFNESS = STIFFNESS_4010
FINGER_DAMPING = DAMPING_4010

G1_ACTUATOR_FINGER = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_thumb_[1-4]_joint",
    ".*_index_[1-2]_joint",
    ".*_middle_[1-2]_joint",
    ".*_ring_[1-2]_joint",
    ".*_little_[1-2]_joint",
  ),
  stiffness=FINGER_STIFFNESS,
  damping=FINGER_DAMPING,
  effort_limit=10.0,
  armature=FINGER_ARMATURE,
)

##
# Collision config.
##

# Reuse the same collision configs from g1_constants.py since the collision
# geom naming convention is identical.
INSPIRE_FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

INSPIRE_FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Keyframe config.
##

INSPIRE_BALANCE_HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.8),
  joint_pos={
    # Legs: slight backward lean to compensate for forward-extended arms.
    #".*_hip_pitch_joint": -0.15,
    #".*_knee_joint": 0.3,
    #".*_ankle_pitch_joint": -0.15,
    # Arms: straing from body forward
    ".*_shoulder_pitch_joint": -1.4,
    # elbows straight (1.57 = 180 degrees)
    ".*_elbow_joint": 1.57,
    #"left_shoulder_roll_joint": 0.18,
    #"right_shoulder_roll_joint": -0.18,
  },
  joint_vel={".*": 0.0},
)

##
# Final config.
##

G1_INSPIRE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_4010,
    G1_ACTUATOR_WAIST,
    G1_ACTUATOR_ANKLE,
    G1_ACTUATOR_FINGER,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_g1_inspire_balance_robot_cfg() -> EntityCfg:
  """Get G1 + Inspire Hands robot config for balance/teleoperation tasks.

  Arms start in a forward-extended position. The RL velocity policy
  controls only legs and waist; arm and finger joints are passive.
  """
  return EntityCfg(
    init_state=INSPIRE_BALANCE_HOME_KEYFRAME,
    collisions=(INSPIRE_FULL_COLLISION,),
    spec_fn=get_inspire_spec,
    articulation=G1_INSPIRE_ARTICULATION,
  )


G1_INSPIRE_ACTION_SCALE: dict[str, float] = {}
for a in G1_INSPIRE_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    G1_INSPIRE_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_g1_inspire_balance_robot_cfg())

  viewer.launch(robot.spec.compile())
