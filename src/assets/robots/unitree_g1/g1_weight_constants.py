"""Unitree G1 with configurable weight boxes on hands and back.

Replaces the rubber-hand end effectors with two boxes mounted at each wrist
(10x10x10 cm) and adds a backpack-style box on the torso (10x20x20 cm).
Box masses are meant to be randomized per environment via
``dr.body_mass`` so the RL policy learns to balance under variable load.
"""

from __future__ import annotations

import mujoco

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from .g1_constants import (
  FULL_COLLISION,
  G1_ACTION_SCALE,
  G1_ACTUATOR_4010,
  G1_ACTUATOR_5020,
  G1_ACTUATOR_7520_14,
  G1_ACTUATOR_7520_22,
  G1_ACTUATOR_ANKLE,
  G1_ACTUATOR_WAIST,
  get_spec as _get_g1_spec,
)

##
# Weight box specifications.
##

# Body names for per-env mass randomization.
LEFT_HAND_WEIGHT_BODY = "left_hand_weight"
RIGHT_HAND_WEIGHT_BODY = "right_hand_weight"
HAND_WEIGHT_BODIES: tuple[str, str] = (LEFT_HAND_WEIGHT_BODY, RIGHT_HAND_WEIGHT_BODY)
BACK_WEIGHT_BODY = "back_weight"

# Box half-sizes (MuJoCo BOX geom size is half-extent).
# Hand weight: 10 x 10 x 10 cm cube.
_HAND_BOX_HALFSIZE: tuple[float, float, float] = (0.05, 0.05, 0.05)
# Back weight: 10 cm (depth, x) x 20 cm (width, y) x 20 cm (height, z).
_BACK_BOX_HALFSIZE: tuple[float, float, float] = (0.05, 0.10, 0.10)

# Initial (nominal) masses — will be overridden per-env by dr.body_mass.
# Set to the middle of the expected range so the default inertia tensor
# approximates a reasonable value across the randomized range.
_HAND_INIT_MASS = 2.0  # mid of 0-4 kg
_BACK_INIT_MASS = 4.0  # mid of 0-8 kg

# Box mounting positions (relative to parent body frame).
# Hands: placed at the tip of the wrist, roughly where the rubber-hand ends.
_HAND_BOX_POS: tuple[float, float, float] = (0.12, 0.0, 0.0)
# Back: backpack-style, centred behind the torso.
_BACK_BOX_POS: tuple[float, float, float] = (-0.13, 0.0, 0.15)


def _box_inertia(
  mass: float, halfsize: tuple[float, float, float]
) -> tuple[float, float, float]:
  """Principal moments of inertia for a uniform cuboid about its centre."""
  dx, dy, dz = (2 * h for h in halfsize)
  ix = mass * (dy * dy + dz * dz) / 12.0
  iy = mass * (dx * dx + dz * dz) / 12.0
  iz = mass * (dx * dx + dy * dy) / 12.0
  return ix, iy, iz


def _add_weight_box(
  spec: mujoco.MjSpec,
  parent_name: str,
  body_name: str,
  pos: tuple[float, float, float],
  halfsize: tuple[float, float, float],
  mass: float,
  rgba: tuple[float, float, float, float],
) -> None:
  """Add a massive, visual-only box as a fixed child of ``parent_name``."""
  parent = spec.body(parent_name)
  body = parent.add_body()
  body.name = body_name
  body.pos = list(pos)

  # Explicit inertial properties — decouples inertia from geom density so
  # later randomization of body_mass does not desync geom visuals.
  body.mass = mass
  body.ipos = [0.0, 0.0, 0.0]
  body.inertia = list(_box_inertia(mass, halfsize))
  body.explicitinertial = True

  geom = body.add_geom()
  geom.name = f"{body_name}_geom"
  geom.type = mujoco.mjtGeom.mjGEOM_BOX
  geom.size = list(halfsize)
  geom.rgba = list(rgba)
  # Density 0 — mass comes from explicit body inertial above.
  geom.density = 0.0
  # Disable contact — these are payload stand-ins, no collision with the world.
  geom.contype = 0
  geom.conaffinity = 0
  geom.group = 2  # visual group


def _remove_rubber_hands(spec: mujoco.MjSpec) -> None:
  """Delete the visual rubber_hand mesh geoms from the wrist_yaw_links.

  They are mass-less (density=0 via the visual default class) so they do
  not affect dynamics, but they are visually confusing in the balance_weight
  task where the policy carries boxes instead of hands.
  """
  to_delete = [
    g for g in spec.geoms if "rubber_hand" in (g.meshname or "")
  ]
  for g in to_delete:
    spec.delete(g)


def _augment_weight_spec(spec: mujoco.MjSpec) -> None:
  """Attach the three weight boxes to the G1 spec (no rubber hands)."""
  _remove_rubber_hands(spec)
  _add_weight_box(
    spec,
    parent_name="left_wrist_yaw_link",
    body_name=LEFT_HAND_WEIGHT_BODY,
    pos=_HAND_BOX_POS,
    halfsize=_HAND_BOX_HALFSIZE,
    mass=_HAND_INIT_MASS,
    rgba=(0.2, 0.6, 0.9, 1.0),
  )
  _add_weight_box(
    spec,
    parent_name="right_wrist_yaw_link",
    body_name=RIGHT_HAND_WEIGHT_BODY,
    pos=_HAND_BOX_POS,
    halfsize=_HAND_BOX_HALFSIZE,
    mass=_HAND_INIT_MASS,
    rgba=(0.9, 0.3, 0.2, 1.0),
  )
  _add_weight_box(
    spec,
    parent_name="torso_link",
    body_name=BACK_WEIGHT_BODY,
    pos=_BACK_BOX_POS,
    halfsize=_BACK_BOX_HALFSIZE,
    mass=_BACK_INIT_MASS,
    rgba=(0.3, 0.8, 0.3, 1.0),
  )


def get_weight_spec() -> mujoco.MjSpec:
  """Get the G1 spec with weight boxes attached."""
  spec = _get_g1_spec()
  _augment_weight_spec(spec)
  return spec


##
# Keyframe / articulation.
##

# Start with arms hanging down so the hand weights are near the pelvis,
# keeping the initial state conservative for early training.
WEIGHT_BALANCE_HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.8),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    # Arms hanging alongside the body.
    ".*_shoulder_pitch_joint": 0.0,
    ".*_elbow_joint": 0.0,
    "left_shoulder_roll_joint": 0.18,
    "right_shoulder_roll_joint": -0.18,
  },
  joint_vel={".*": 0.0},
)


# Same 29 DoF actuators as the base G1 — the weight boxes are fixed children
# and do not introduce new joints.
G1_WEIGHT_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_4010,
    G1_ACTUATOR_WAIST,
    G1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_g1_weight_robot_cfg() -> EntityCfg:
  """Get G1 robot config with weight boxes on hands and back.

  No hand DoFs — each wrist carries a fixed box whose mass is intended to
  be randomized per-env via ``dr.body_mass``. A backpack-style box is
  mounted on the torso.
  """
  return EntityCfg(
    init_state=WEIGHT_BALANCE_HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_weight_spec,
    articulation=G1_WEIGHT_ARTICULATION,
  )


# Reuse the 29-DoF action scale (no fingers).
G1_WEIGHT_ACTION_SCALE: dict[str, float] = dict(G1_ACTION_SCALE)


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_g1_weight_robot_cfg())
  viewer.launch(robot.spec.compile())
