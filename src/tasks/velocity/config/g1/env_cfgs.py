"""Unitree G1 velocity environment configurations."""

from src.assets.robots import (
  G1_ACTION_SCALE,
  G1_INSPIRE_ACTION_SCALE,
  get_g1_robot_cfg,
  get_g1_inspire_balance_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from src.tasks.velocity.mdp.events import nudge_joints_velocity
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.velocity.mdp.velocity_command import UniformVelocityHeightCommandCfg
from src.tasks.velocity.mdp.rewards import track_base_height, track_linear_velocity_no_z
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def unitree_g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  # Set raycast sensor frame to G1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.15,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.25,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
  }

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg


def unitree_g1_flat_balance_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain balance configuration for teleoperation.

  Based on Unitree-G1-Flat but uses the 29 DoF G1 model with Inspire Hands
  with arms extended forward. The RL velocity policy controls only legs and
  waist; arm joints are excluded from actions and moved with small random
  perturbations to simulate natural teleoperation variation.
  """
  cfg = unitree_g1_flat_env_cfg(play=play)

  # Use G1 + Inspire Hands balance config (arms extended forward, 29 DoF + 24 finger DoF).
  cfg.scene.entities = {"robot": get_g1_inspire_balance_robot_cfg()}

  # Leg and waist joint names (the only joints the RL policy controls).
  # Must match the exact order in the robot model for consistency validation.
  _leg_waist_joint_names = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
  )

  _leg_waist_asset_cfg = SceneEntityCfg("robot", joint_names=_leg_waist_joint_names)

  # Restrict RL policy action space to legs and waist only.
  _balance_action_scale = {
    k: v for k, v in G1_INSPIRE_ACTION_SCALE.items()
    if not any(arm in k for arm in ("shoulder", "elbow", "wrist", "thumb", "index", "middle", "ring", "little"))
  }
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = list(_leg_waist_joint_names)
  joint_pos_action.scale = _balance_action_scale

  # Filter observations to leg/waist joints only. The Inspire model has 53
  # joints but the policy only controls 9; observing all joints adds noise
  # that destabilises training.
  cfg.observations["actor"].terms["joint_pos"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=_leg_waist_joint_names
  )
  cfg.observations["actor"].terms["joint_vel"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=_leg_waist_joint_names
  )
  cfg.observations["critic"].terms["joint_pos"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=_leg_waist_joint_names
  )
  cfg.observations["critic"].terms["joint_vel"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=_leg_waist_joint_names
  )

  # Filter joint-based rewards to leg/waist only.
  cfg.rewards["joint_acc_l2"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=_leg_waist_joint_names
  )
  cfg.rewards["joint_pos_limits"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=_leg_waist_joint_names
  )
  cfg.rewards["stand_still"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=_leg_waist_joint_names
  )

  # Exclude arm joints from pose reward (arms are teleop-controlled, not RL).
  # Also replace std dicts: arm patterns would cause ValueError since no arm
  # joints are present in the filtered joint list.
  cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=_leg_waist_joint_names
  )
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.15,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.25,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
  }

  # At reset, only randomize leg/waist joints; arms stay at forward keyframe.
  cfg.events["reset_robot_joints"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=_leg_waist_joint_names
  )

  # Periodically nudge arm joints with small random velocities. The PD
  # controllers smoothly dampen the perturbation, producing natural gradual
  # motion instead of instantaneous position jumps.
  cfg.events["nudge_arms"] = EventTermCfg(
    func=nudge_joints_velocity,
    mode="interval",
    interval_range_s=(0.2, 1),
    params={
      "velocity_range": (-5, 5),
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(
          ".*_shoulder_pitch_joint",
          ".*_shoulder_roll_joint",
          ".*_shoulder_yaw_joint",
          ".*_elbow_joint",
          ".*_wrist_roll_joint",
          ".*_wrist_pitch_joint",
          ".*_wrist_yaw_joint",
        ),
      ),
    },
  )

  # Stronger balance incentives for the heavy Inspire hands configuration.
  # The forward-extended arms shift COM significantly, requiring more active
  # balance effort than the base G1.
  cfg.rewards["body_orientation_l2"].weight = -3.0
  cfg.rewards["stand_still"].weight = -3.0
  cfg.rewards["body_ang_vel"].weight = -0.15

  # Increase standing training ratio for better balance without velocity input.
  # Default 0.05 (5%) is too low — the robot barely learns to stand still.
  # With heavy Inspire hands, 40% gives enough gradient signal for balance.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.rel_standing_envs = 0.4

  return cfg


def unitree_g1_flat_balance_height_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain balance configuration with height control.

  Based on Unitree-G1-Flat-Balance but adds a target base height command.
  The robot learns to squat/crouch by tracking a randomly sampled target
  height in addition to velocity commands. The command vector is 4D:
  [lin_vel_x, lin_vel_y, ang_vel_z, target_height].
  """
  import math

  cfg = unitree_g1_flat_balance_env_cfg(play=play)

  # Replace the velocity command with the height-aware variant.
  # Preserve the existing velocity ranges and settings.
  old_twist = cfg.commands["twist"]
  assert isinstance(old_twist, UniformVelocityCommandCfg)
  cfg.commands["twist"] = UniformVelocityHeightCommandCfg(
    entity_name=old_twist.entity_name,
    resampling_time_range=old_twist.resampling_time_range,
    rel_standing_envs=old_twist.rel_standing_envs,
    rel_heading_envs=old_twist.rel_heading_envs,
    heading_command=old_twist.heading_command,
    heading_control_stiffness=old_twist.heading_control_stiffness,
    debug_vis=old_twist.debug_vis,
    default_height=0.74,
    ranges=UniformVelocityHeightCommandCfg.Ranges(
      lin_vel_x=old_twist.ranges.lin_vel_x,
      lin_vel_y=old_twist.ranges.lin_vel_y,
      ang_vel_z=old_twist.ranges.ang_vel_z,
      heading=old_twist.ranges.heading,
      base_height=(0.45, 0.78),
    ),
    viz=UniformVelocityCommandCfg.VizCfg(z_offset=1.15),
  )

  # 1) Replace track_linear_velocity with a version that doesn't penalize
  #    z-velocity. The original penalizes vertical motion (2 * z_error²),
  #    which directly conflicts with height changes during squatting.
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=track_linear_velocity_no_z,
    weight=1.0,
    params={"command_name": "twist", "std": math.sqrt(0.25)},
  )

  # 2) Add height tracking reward — main signal for learning to squat.
  cfg.rewards["track_base_height"] = RewardTermCfg(
    func=track_base_height,
    weight=2.0,
    params={"command_name": "twist", "std": math.sqrt(0.05)},
  )

  # 3) Reduce stand_still penalty and make it height-aware. At weight -3.0
  #    it dominates when velocity is zero, punishing the knee/hip bending
  #    needed for squatting. With default_height set, stand_still deactivates
  #    when a non-default height is commanded, allowing squatting in place.
  cfg.rewards["stand_still"].weight = -0.5
  cfg.rewards["stand_still"].params["default_height"] = 0.74

  # 4) Reduce pose reward weight — squatting deviates heavily from default
  #    standing pose; a strong pose reward fights the height command.
  cfg.rewards["pose"].weight = 0.3

  # Relax pose std for standing — squatting requires large knee/hip/ankle
  # deviations from the default standing pose.
  cfg.rewards["pose"].params["std_standing"] = {
    r".*hip_pitch.*": 1.0,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 1.0,
    r".*ankle_pitch.*": 1.0,
    r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.15,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
  }

  return cfg
