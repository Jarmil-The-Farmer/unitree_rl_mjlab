"""Unitree G1 velocity environment configurations."""

from src.assets.robots import (
  G1_ACTION_SCALE,
  G1_INSPIRE_ACTION_SCALE,
  G1_WEIGHT_ACTION_SCALE,
  get_g1_robot_cfg,
  get_g1_inspire_balance_robot_cfg,
  get_g1_weight_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from src.tasks.velocity.mdp.curriculums import arm_pose_randomization_curriculum, event_ranges, reward_weight, standing_balance, waist_yaw_range
from src.tasks.velocity.mdp.events import (
  drive_joints_from_command_channel,
  nudge_joints_position,
  nudge_joints_velocity,
  randomize_arm_pose,
  reset_motor_temperatures,
  step_motor_temperatures,
)
from src.tasks.velocity.mdp.observations import motor_temperatures, payload_masses
from src.tasks.velocity.mdp.terminations import motor_overheat
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.velocity.mdp.velocity_command import UniformVelocityHeightCommandCfg, UniformVelocityHeightWaistCommandCfg
from src.tasks.velocity.mdp.rewards import (
  action_rate_l2_standing,
  base_ang_vel_standing,
  joint_acc_l2_standing,
  joint_deviation_l2,
  motor_overheat_penalty,
  stand_still_lin_vel,
  track_base_height,
  track_linear_velocity_no_z,
)
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

  # Observe leg/waist + arm joints (no fingers — they don't affect balance).
  # The policy controls only legs/waist but needs arm positions to compensate
  # for their weight (8.5 kg, 24% of total mass) during balance.
  _obs_joint_names = _leg_waist_joint_names + (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
  )
  cfg.observations["actor"].terms["joint_pos"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=_obs_joint_names
  )
  cfg.observations["actor"].terms["joint_vel"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=_obs_joint_names
  )
  cfg.observations["critic"].terms["joint_pos"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=_obs_joint_names
  )
  cfg.observations["critic"].terms["joint_vel"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=_obs_joint_names
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
  # Standing pose std: sagittal joints free for balance (matching standing task).
  # Lateral joints tight.
  cfg.rewards["pose"].params["std_standing"] = {
    r".*hip_pitch.*": 10.0,
    r".*hip_roll.*": 0.08,
    r".*hip_yaw.*": 0.08,
    r".*knee.*": 10.0,
    r".*ankle_pitch.*": 10.0,
    r".*ankle_roll.*": 0.05,
    r".*waist_yaw.*": 0.08,
    r".*waist_roll.*": 0.5,
    r".*waist_pitch.*": 10.0,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.05,
    r".*hip_yaw.*": 0.05,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.3,
    r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.1,
    r".*waist_roll.*": 0.15,
    r".*waist_pitch.*": 0.3,  # allow lean during walking
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.1,
    r".*hip_yaw.*": 0.1,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.3,
    r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.15,
    r".*waist_roll.*": 0.2,
    r".*waist_pitch.*": 0.3,
  }

  # At reset, only randomize leg/waist joints; arms are reset separately.
  cfg.events["reset_robot_joints"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=_leg_waist_joint_names
  )

  # Randomize arm pose on every reset. Each env gets a random shoulder_pitch
  # and elbow from the current range (widened by curriculum). PD targets are
  # set to hold the sampled pose. default_joint_pos stays at (0,0) so
  # joint_pos_rel observations reflect absolute arm position.
  _arm_joint_names = (
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_roll_joint",
    ".*_wrist_pitch_joint",
    ".*_wrist_yaw_joint",
  )
  cfg.events["randomize_arm_pose"] = EventTermCfg(
    func=randomize_arm_pose,
    mode="reset",
    params={
      "shoulder_pitch_range": (-1.6, 0.0),
      "elbow_range": (0.0, 1.57),
      "shoulder_roll_range": (0.0, 0.8),
      "asset_cfg": SceneEntityCfg("robot", joint_names=_arm_joint_names),
    },
  )

  # Smoothly move arm joints toward random target positions at constant speed.
  # Called frequently — PD targets move at `speed` rad/s toward a random goal.
  # When goal is reached, a new one is sampled. Simulates teleoperation.
  # Controlled by the same joystick toggle as nudge_arms.
  cfg.events["nudge_arms_position"] = EventTermCfg(
    func=nudge_joints_position,
    mode="interval",
    interval_range_s=(0.05, 0.1),
    params={
      "position_offset_range": (-0.5, 0.5),
      "speed": 0.5,
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

  # Balance incentives matching standing task approach:
  # strong upright signal, disable stand_still (blocks knee bending).
  cfg.rewards["body_orientation_l2"].weight = -5.0
  cfg.rewards["stand_still"].weight = 0.0  # disabled — blocks sagittal joint freedom
  cfg.rewards["body_ang_vel"].weight = -0.15
  cfg.rewards["pose"].weight = 0.3

  # Dedicated penalties for joints that must stay near zero.
  # The averaged pose reward is too weak to prevent hip splay/rotation —
  # these targeted L2 penalties directly punish any deviation.
  cfg.rewards["hip_lateral_deviation"] = RewardTermCfg(
    func=joint_deviation_l2,
    weight=-5.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(
        ".*_hip_roll_joint",
        ".*_hip_yaw_joint",
      )),
    },
  )
  cfg.rewards["waist_lateral_deviation"] = RewardTermCfg(
    func=joint_deviation_l2,
    weight=-4.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(
        "waist_yaw_joint",
      )),
    },
  )

  # Penalize horizontal base velocity when commanded to stand still.
  cfg.rewards["stand_still_lin_vel"] = RewardTermCfg(
    func=stand_still_lin_vel,
    weight=-10.0,
    params={"command_name": "twist", "command_threshold": 0.1},
  )

  # Start with higher standing ratio — standing balance is the priority.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.rel_standing_envs = 0.2

  # Curriculum: gradually shift focus from walking to standing balance.
  # Faster ramp — standing balance with arms is the main goal.
  cfg.curriculum["standing_balance"] = CurriculumTermCfg(
    func=standing_balance,
    params={
      "command_name": "twist",
      "nudge_event_name": "nudge_arms_position",
      "stages": [
        {"step": 0,          "rel_standing_envs": 0.2, "nudge_speed": 0.3},
        {"step": 2000 * 24,  "rel_standing_envs": 0.4, "nudge_speed": 0.5},
        {"step": 4000 * 24,  "rel_standing_envs": 0.6, "nudge_speed": 0.8},
        {"step": 6000 * 24,  "rel_standing_envs": 0.7, "nudge_speed": 1.0},
      ],
    },
  )

  # # Curriculum: arm pose randomization range.
  # # Disabled — nudge_arms_position handles continuous arm motion during training.
  # cfg.curriculum["arm_pose_range"] = CurriculumTermCfg(
  #   func=arm_pose_randomization_curriculum,
  #   params={
  #     "reset_event_name": "randomize_arm_pose",
  #     "stages": [
  #       {"step": 0,          "shoulder_pitch_range": (-0.5, 0.0), "elbow_range": (0.0, 0.5)},
  #       {"step": 1500 * 24,  "shoulder_pitch_range": (-1.0, 0.0), "elbow_range": (0.0, 1.0)},
  #       {"step": 3000 * 24,  "shoulder_pitch_range": (-1.6, 0.0), "elbow_range": (0.0, 1.57)},
  #     ],
  #   },
  # )

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

  cfg.episode_length_s = 30.0

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
      base_height=(0.35, 0.78),
    ),
    viz=UniformVelocityCommandCfg.VizCfg(z_offset=1.15),
  )

  # 0) Keep gait reward and phase observation from base config — helps
  # the robot learn a periodic walking gait.

  # 1) Replace track_linear_velocity with a version that doesn't penalize
  #    z-velocity. The original penalizes vertical motion (2 * z_error²),
  #    which directly conflicts with height changes during squatting.
  #    Boosted weight so walking is more attractive than standing.
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=track_linear_velocity_no_z,
    weight=1.5,
    params={"command_name": "twist", "std": math.sqrt(0.25)},
  )

  # 2) Add height tracking reward — main signal for learning to squat.
  cfg.rewards["track_base_height"] = RewardTermCfg(
    func=track_base_height,
    weight=2.0,
    params={"command_name": "twist", "std": math.sqrt(0.05)},
  )

  # 3) Disable stand_still — it blocks knee/waist bending needed for both
  #    squatting and arm balance compensation.
  cfg.rewards["stand_still"].weight = 0.0

  # 4) Pose reward: sagittal joints completely free, hip_roll slightly
  #    loosened to allow a modest wider stance (not full asymmetry).
  cfg.rewards["pose"].weight = 0.3
  cfg.rewards["pose"].params["std_standing"] = {
    r".*hip_pitch.*": 10.0,
    r".*hip_roll.*": 0.15,  # slightly loosened — modest wider stance
    r".*hip_yaw.*": 0.1,   # keep feet pointing forward
    r".*knee.*": 10.0,
    r".*ankle_pitch.*": 10.0,
    r".*ankle_roll.*": 0.05,
    r".*waist_yaw.*": 0.08,
    r".*waist_roll.*": 0.5,
    r".*waist_pitch.*": 10.0,
  }

  # 5) Strong upright signal — keeps pelvis/hips level (no sideways lean).
  cfg.rewards["body_orientation_l2"].weight = -8.0

  # 6) Penalize horizontal base velocity when commanded to stand still.
  cfg.rewards["stand_still_lin_vel"] = RewardTermCfg(
    func=stand_still_lin_vel,
    weight=-10.0,
    params={"command_name": "twist", "command_threshold": 0.1},
  )

  # 7) Hip lateral deviation — keeps hips symmetric (both legs behave alike).
  #    Covers hip_roll and hip_yaw to prevent asymmetric leg splay.
  cfg.rewards["hip_lateral_deviation"].weight = -3.0

  # 7b) Dedicated penalty for waist_roll — prevents sideways lean at the
  #     waist. Strong weight because torso tilt looks unnatural and is
  #     rarely necessary for balance (pelvis/leg adjustment should compensate).
  cfg.rewards["waist_roll_deviation"] = RewardTermCfg(
    func=joint_deviation_l2,
    weight=-5.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=("waist_roll_joint",)),
    },
  )

  # 7c) Standing-only smoothness penalties — stronger than the global
  # action_rate_l2 / joint_acc_l2, but apply ONLY when commanded velocity
  # is below threshold. Eliminates tremor while standing (especially during
  # arm motion) without inhibiting dynamic walking gait.
  cfg.rewards["action_rate_standing"] = RewardTermCfg(
    func=action_rate_l2_standing,
    weight=-0.05,  # initial; curriculum ramps to -0.15
    params={"command_name": "twist", "command_threshold": 0.1},
  )
  # Anti-oscillation: penalize torso rocking when standing.
  cfg.rewards["base_ang_vel_standing"] = RewardTermCfg(
    func=base_ang_vel_standing,
    weight=-0.5,
    params={"command_name": "twist", "command_threshold": 0.1},
  )
  cfg.rewards["joint_acc_standing"] = RewardTermCfg(
    func=joint_acc_l2_standing,
    weight=-1.0e-7,  # initial; curriculum ramps to -4e-7
    params={
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(
          ".*_hip_.*", ".*_knee_.*", ".*_ankle_.*", "waist_.*",
        ),
      ),
    },
  )

  # 8) Add shoulder_roll randomization to prevent arm crossing.
  cfg.events["randomize_arm_pose"].params["shoulder_roll_range"] = (0.2, 0.8)

  # 9) Observation history — stack proprioceptive snapshots so the policy
  # can infer dynamics (arm inertia, height regime) from recent trajectory.
  _HISTORY_LENGTH = 5
  _HISTORY_TERMS = (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
  )
  _CRITIC_EXTRA_HISTORY_TERMS = ("base_lin_vel",)
  for _group_name, _extras in (
    ("actor", ()),
    ("critic", _CRITIC_EXTRA_HISTORY_TERMS),
  ):
    _group = cfg.observations[_group_name]
    _group.history_length = None  # disable group-level override
    for _term_name in _HISTORY_TERMS + _extras:
      if _term_name in _group.terms:
        _group.terms[_term_name].history_length = _HISTORY_LENGTH

  # 10) Smoothness curriculum: ramp up action_rate and joint_acc penalties
  # in later training to eliminate tremor once basic balance is learned.
  # Global action_rate / joint_acc — kept LOW so walking dynamics aren't
  # suppressed. Standing tremor is handled by the *_standing rewards below.
  cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
    func=reward_weight,
    params={
      "reward_name": "action_rate_l2",
      "weight_stages": [
        {"step": 0,           "weight": -0.05},
        {"step": 6000 * 24,   "weight": -0.08},
        {"step": 12000 * 24,  "weight": -0.10},
      ],
    },
  )
  cfg.curriculum["joint_acc_weight"] = CurriculumTermCfg(
    func=reward_weight,
    params={
      "reward_name": "joint_acc_l2",
      "weight_stages": [
        {"step": 0,           "weight": -2.5e-7},
        {"step": 6000 * 24,   "weight": -3.5e-7},
        {"step": 12000 * 24,  "weight": -5.0e-7},
      ],
    },
  )

  # Standing-only smoothness — gentle anti-tremor signal applied ONLY when
  # commanded velocity is near zero. Kept modest so balance corrections
  # remain affordable; previous strong weights caused the policy to fall
  # rather than make corrections (tremor cost > termination cost).
  cfg.curriculum["action_rate_standing_weight"] = CurriculumTermCfg(
    func=reward_weight,
    params={
      "reward_name": "action_rate_standing",
      "weight_stages": [
        {"step": 0,           "weight": -0.05},
        {"step": 8000 * 24,   "weight": -0.10},
        {"step": 14000 * 24,  "weight": -0.15},
      ],
    },
  )
  cfg.curriculum["joint_acc_standing_weight"] = CurriculumTermCfg(
    func=reward_weight,
    params={
      "reward_name": "joint_acc_standing",
      "weight_stages": [
        {"step": 0,           "weight": -1.0e-7},
        {"step": 8000 * 24,   "weight": -2.5e-7},
        {"step": 14000 * 24,  "weight": -4.0e-7},
      ],
    },
  )

  # 10) Arm nudge curriculum: ramp speed and offset range for more
  # aggressive arm motion in later training.
  cfg.curriculum["standing_balance"].params["stages"] = [
    {"step": 0,           "rel_standing_envs": 0.2, "nudge_speed": 0.3, "nudge_offset_range": (-0.5, 0.5)},
    {"step": 1500 * 24,   "rel_standing_envs": 0.3, "nudge_speed": 0.5, "nudge_offset_range": (-0.5, 0.5)},
    {"step": 3000 * 24,   "rel_standing_envs": 0.4, "nudge_speed": 0.8, "nudge_offset_range": (-0.7, 0.7)},
    {"step": 6000 * 24,   "rel_standing_envs": 0.5, "nudge_speed": 1.2, "nudge_offset_range": (-0.8, 0.8)},
    {"step": 9000 * 24,   "rel_standing_envs": 0.5, "nudge_speed": 1.8, "nudge_offset_range": (-1.0, 1.0)},
    {"step": 14000 * 24,  "rel_standing_envs": 0.5, "nudge_speed": 2.5, "nudge_offset_range": (-1.2, 1.2)},
    # Late-stage "arm hold" — slow arm motion + larger fraction of standing envs
    # to specifically train stable balancing with near-static arm poses.
    {"step": 20000 * 24,  "rel_standing_envs": 0.6, "nudge_speed": 0.5, "nudge_offset_range": (-0.8, 0.8)},
  ]

  # === Motor thermal simulation ===
  # Each step integrates a per-motor first-order thermal model driven by
  # actuator_force. Soft penalty above T_warn, hard termination above T_max.
  # Critic sees temperatures (privileged), actor must infer from proprio.
  cfg.events["reset_motor_temperatures"] = EventTermCfg(
    func=reset_motor_temperatures, mode="reset",
  )
  cfg.events["step_motor_temperatures"] = EventTermCfg(
    func=step_motor_temperatures, mode="step",
  )
  cfg.observations["critic"].terms["motor_temperatures"] = ObservationTermCfg(
    func=motor_temperatures, params={"T_amb": 25.0, "T_scale": 50.0},
  )
  cfg.terminations["motor_overheat"] = TerminationTermCfg(
    func=motor_overheat, params={"T_max": 100.0},
  )
  cfg.rewards["motor_overheat_penalty"] = RewardTermCfg(
    func=motor_overheat_penalty,
    weight=0.0,  # ramped by curriculum below
    params={"T_warn": 70.0, "T_crit": 90.0},
  )
  cfg.curriculum["motor_overheat_weight"] = CurriculumTermCfg(
    func=reward_weight,
    params={
      "reward_name": "motor_overheat_penalty",
      "weight_stages": [
        {"step": 0,           "weight": 0.0},     # let policy learn balance first
        {"step": 3000 * 24,   "weight": -0.001},
        {"step": 6000 * 24,   "weight": -0.005},
        {"step": 10000 * 24,  "weight": -0.02},
      ],
    },
  )

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityHeightCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.3, 0.5)
    twist_cmd.ranges.lin_vel_y = (-0.3, 0.3)
    twist_cmd.ranges.ang_vel_z = (-0.4, 0.4)

  return cfg


def unitree_g1_flat_balance_standing_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 standing-only balance configuration.

  Based on Unitree-G1-Flat-Balance but the robot ONLY learns to stand still
  with arms in any position. No walking, no velocity tracking — pure standing
  balance. Used to isolate and solve the arm-compensation problem.
  """
  cfg = unitree_g1_flat_balance_env_cfg(play=play)

  # 100% standing — no walking at all.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.rel_standing_envs = 1.0

  # Zero velocity ranges — never command walking.
  twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

  # Remove walking/gait rewards — they're meaningless for standing.
  for name in ("track_linear_velocity", "track_angular_velocity",
               "foot_gait", "foot_clearance", "foot_slip", "soft_landing"):
    cfg.rewards.pop(name, None)

  # Disable stand_still — it penalizes ALL joint deviations including
  # knees/hip_pitch which the robot MUST bend to balance with arms.
  cfg.rewards["stand_still"].weight = 0.0

  # Pose reward: only penalize lateral joints (hip_roll, hip_yaw, ankle_roll,
  # waist_yaw). All sagittal joints (hip_pitch, knee, ankle_pitch, waist_pitch)
  # are fully free so the robot can squat and lean as needed.
  cfg.rewards["pose"].weight = 0.3
  cfg.rewards["pose"].params["std_standing"] = {
    r".*hip_pitch.*": 10.0,   # completely free
    r".*hip_roll.*": 0.08,    # slightly relaxed for stability
    r".*hip_yaw.*": 0.08,     # slightly relaxed for stability
    r".*knee.*": 10.0,        # completely free
    r".*ankle_pitch.*": 10.0, # completely free
    r".*ankle_roll.*": 0.05,
    r".*waist_yaw.*": 0.08,
    r".*waist_roll.*": 0.5,
    r".*waist_pitch.*": 10.0, # completely free
  }

  # Body orientation: the main upright signal. Robot must keep torso upright
  # but can achieve this through any combination of knee bend + waist lean.
  cfg.rewards["body_orientation_l2"].weight = -5.0

  # Strong penalty for any horizontal base movement.
  cfg.rewards["stand_still_lin_vel"] = RewardTermCfg(
    func=stand_still_lin_vel,
    weight=-10.0,
    params={"command_name": "twist", "command_threshold": 0.1},
  )

  # Hip lateral deviation — relaxed slightly so robot can fine-tune stance.
  cfg.rewards["hip_lateral_deviation"].weight = -5.0

  # Arm curriculum: start with fully extended arms so robot learns the hardest
  # case first, then widen range to include all positions.
  # Shoulder roll prevents arm crossing when raised.
  cfg.events["randomize_arm_pose"].params["shoulder_pitch_range"] = (-1.6, -1.2)
  cfg.events["randomize_arm_pose"].params["elbow_range"] = (1.2, 1.57)
  cfg.events["randomize_arm_pose"].params["shoulder_roll_range"] = (0.2, 0.8)
  cfg.curriculum["arm_pose_range"] = CurriculumTermCfg(
    func=arm_pose_randomization_curriculum,
    params={
      "reset_event_name": "randomize_arm_pose",
      "stages": [
        {"step": 0,          "shoulder_pitch_range": (-1.6, -1.2), "elbow_range": (1.2, 1.57)},
        {"step": 3000 * 24,  "shoulder_pitch_range": (-1.6, -0.5), "elbow_range": (0.5, 1.57)},
        {"step": 6000 * 24,  "shoulder_pitch_range": (-1.6, 0.0),  "elbow_range": (0.0, 1.57)},
      ],
    },
  )

  # Remove standing_balance curriculum — already 100% standing.
  cfg.curriculum.pop("standing_balance", None)

  # Slow nudge — arms move gently so the robot must hold each position
  # for a long time, not just survive brief transients.
  cfg.events["nudge_arms_position"].params["speed"] = 0.3

  return cfg


def unitree_g1_flat_balance_height_waist_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 balance + height + externally-driven waist_yaw config.

  Based on Unitree-G1-Flat-Balance-Height, with three differences:

  1) ``waist_yaw_joint`` is removed from the RL action set. The joint is
     driven directly from an extra 5th command channel (``waist_yaw_target``),
     written into the PD target every control step via the
     ``drive_waist_yaw`` interval event. This mirrors headset-driven teleop:
     the operator commands the waist angle, the policy only compensates for
     the resulting torso pose.
  2) The command tensor is 5D: ``[lin_vel_x, lin_vel_y, ang_vel_z,
     target_height, waist_yaw_target]``. The 5th channel is sampled per
     resample from ``ranges.waist_yaw`` (curriculum-widened).
  3) Curriculum compressed to ~20k iterations, ending with a "slow arm
     hold" stage so the policy refines balance with near-static extended
     arms (the practical teleop target).
  """
  cfg = unitree_g1_flat_balance_height_env_cfg(play=play)

  cfg.episode_length_s = 30.0

  # === 1) Action set: drop waist_yaw — it's externally driven. ===
  _leg_waist_no_wyaw = (
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
    "waist_roll_joint",
    "waist_pitch_joint",
  )
  _action_scale = {
    k: v for k, v in G1_INSPIRE_ACTION_SCALE.items()
    if not any(arm in k for arm in (
      "shoulder", "elbow", "wrist", "thumb", "index", "middle", "ring", "little"
    )) and "waist_yaw" not in k
  }
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = list(_leg_waist_no_wyaw)
  joint_pos_action.scale = _action_scale

  # Filter joint-based rewards/events to the new (14-joint) action set.
  # waist_yaw stays in the observation joint set (joint_pos/joint_vel terms)
  # — the policy must see where the externally-driven waist actually is.
  # Fresh SceneEntityCfg per consumer: each manager resolves it in-place,
  # so a shared instance would fail the second resolution's consistency check.
  def _filter_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=_leg_waist_no_wyaw)
  cfg.rewards["joint_acc_l2"].params["asset_cfg"] = _filter_cfg()
  cfg.rewards["joint_pos_limits"].params["asset_cfg"] = _filter_cfg()
  cfg.rewards["stand_still"].params["asset_cfg"] = _filter_cfg()
  cfg.rewards["pose"].params["asset_cfg"] = _filter_cfg()

  # Drop the waist_yaw entry from pose std dicts (joint no longer in filter).
  for _std_key in ("std_standing", "std_walking", "std_running"):
    cfg.rewards["pose"].params[_std_key].pop(r".*waist_yaw.*", None)

  # Remove waist_lateral_deviation — it penalized waist_yaw deviation.
  cfg.rewards.pop("waist_lateral_deviation", None)

  # joint_acc_standing currently includes "waist_.*"; restrict to waist_roll/pitch
  # so accelerations of the externally-driven waist_yaw aren't penalized.
  cfg.rewards["joint_acc_standing"].params["asset_cfg"] = SceneEntityCfg(
    "robot",
    joint_names=(
      ".*_hip_.*", ".*_knee_.*", ".*_ankle_.*",
      "waist_roll_joint", "waist_pitch_joint",
    ),
  )

  # reset_robot_joints currently targets 15 joints (including waist_yaw).
  # Restrict to the 14-joint set; waist_yaw qpos is zeroed by sim reset
  # and then chased by PD via the drive_waist_yaw event.
  cfg.events["reset_robot_joints"].params["asset_cfg"] = _filter_cfg()

  # === 2) 5D command (lin_vel xyz, height, waist_yaw_target). ===
  old_twist = cfg.commands["twist"]
  assert isinstance(old_twist, UniformVelocityHeightCommandCfg)
  cfg.commands["twist"] = UniformVelocityHeightWaistCommandCfg(
    entity_name=old_twist.entity_name,
    resampling_time_range=old_twist.resampling_time_range,
    rel_standing_envs=old_twist.rel_standing_envs,
    rel_heading_envs=old_twist.rel_heading_envs,
    heading_command=old_twist.heading_command,
    heading_control_stiffness=old_twist.heading_control_stiffness,
    debug_vis=old_twist.debug_vis,
    default_height=old_twist.default_height,
    ranges=UniformVelocityHeightWaistCommandCfg.Ranges(
      lin_vel_x=old_twist.ranges.lin_vel_x,
      lin_vel_y=old_twist.ranges.lin_vel_y,
      ang_vel_z=old_twist.ranges.ang_vel_z,
      heading=old_twist.ranges.heading,
      base_height=old_twist.ranges.base_height,
      waist_yaw=(0.0, 0.0),  # curriculum gradually widens this from zero
    ),
    viz=UniformVelocityCommandCfg.VizCfg(z_offset=1.15),
  )

  # === 3) Drive waist_yaw PD target from command channel 4 every step. ===
  cfg.events["drive_waist_yaw"] = EventTermCfg(
    func=drive_joints_from_command_channel,
    mode="interval",
    interval_range_s=(0.02, 0.04),
    params={
      "command_name": "twist",
      "command_index": 4,
      "asset_cfg": SceneEntityCfg("robot", joint_names=("waist_yaw_joint",)),
    },
  )

  # === 4) Curriculum compressed to ~20k iterations. ===
  # Standing-heavy throughout: deploy target is teleop where the robot stands
  # most of the time. Adding externally-driven waist_yaw on top of all other
  # perturbations raises the difficulty floor, so rel_standing_envs is
  # bumped (was 0.2-0.6 in the height task) to give the policy more direct
  # standing supervision. Last 5k iter focus on slow arm motion for the
  # extended-arm hold case.
  cfg.commands["twist"].rel_standing_envs = 0.3  # initial; curriculum drives it
  cfg.curriculum["standing_balance"].params["stages"] = [
    {"step": 0,           "rel_standing_envs": 0.3, "nudge_speed": 0.3, "nudge_offset_range": (-0.5, 0.5)},
    {"step": 1000 * 24,   "rel_standing_envs": 0.4, "nudge_speed": 0.5, "nudge_offset_range": (-0.5, 0.5)},
    {"step": 2500 * 24,   "rel_standing_envs": 0.5, "nudge_speed": 0.8, "nudge_offset_range": (-0.7, 0.7)},
    {"step": 5000 * 24,   "rel_standing_envs": 0.6, "nudge_speed": 1.2, "nudge_offset_range": (-0.8, 0.8)},
    {"step": 8000 * 24,   "rel_standing_envs": 0.6, "nudge_speed": 1.8, "nudge_offset_range": (-1.0, 1.0)},
    {"step": 11000 * 24,  "rel_standing_envs": 0.6, "nudge_speed": 2.5, "nudge_offset_range": (-1.2, 1.2)},
    # Late-stage arm hold — slow arm motion, more standing.
    {"step": 15000 * 24,  "rel_standing_envs": 0.7, "nudge_speed": 0.5, "nudge_offset_range": (-0.8, 0.8)},
  ]
  cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
    {"step": 0,           "weight": -0.05},
    {"step": 5000 * 24,   "weight": -0.08},
    {"step": 10000 * 24,  "weight": -0.10},
  ]
  cfg.curriculum["joint_acc_weight"].params["weight_stages"] = [
    {"step": 0,           "weight": -2.5e-7},
    {"step": 5000 * 24,   "weight": -3.5e-7},
    {"step": 10000 * 24,  "weight": -5.0e-7},
  ]
  cfg.curriculum["action_rate_standing_weight"].params["weight_stages"] = [
    {"step": 0,           "weight": -0.05},
    {"step": 6000 * 24,   "weight": -0.10},
    {"step": 11000 * 24,  "weight": -0.15},
  ]
  cfg.curriculum["joint_acc_standing_weight"].params["weight_stages"] = [
    {"step": 0,           "weight": -1.0e-7},
    {"step": 6000 * 24,   "weight": -2.5e-7},
    {"step": 11000 * 24,  "weight": -4.0e-7},
  ]

  # Waist_yaw range: zero for early training (policy first learns standing
  # balance with stationary torso), then linearly widens. Narrowed back to
  # practical teleop range for the late-stage arm-hold phase.
  cfg.curriculum["waist_yaw_range"] = CurriculumTermCfg(
    func=waist_yaw_range,
    params={
      "command_name": "twist",
      "stages": [
        {"step": 0,           "ranges": (0.0, 0.0)},     # no rotation — standing pre-req
        {"step": 1500 * 24,   "ranges": (0.0, 0.0)},     # hold zero through 1500 iter
        {"step": 3500 * 24,   "ranges": (-0.3, 0.3)},    # gentle introduction
        {"step": 7000 * 24,   "ranges": (-0.7, 0.7)},
        {"step": 11000 * 24,  "ranges": (-1.2, 1.2)},
        {"step": 13500 * 24,  "ranges": (-1.4, 1.4)},    # widest
        {"step": 15000 * 24,  "ranges": (-1.0, 1.0)},    # narrow back for hold
      ],
    },
  )

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityHeightWaistCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.3, 0.5)
    twist_cmd.ranges.lin_vel_y = (-0.3, 0.3)
    twist_cmd.ranges.ang_vel_z = (-0.4, 0.4)
    twist_cmd.ranges.waist_yaw = (-1.0, 1.0)

  return cfg


def unitree_g1_flat_balance_weight_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 balance configuration with configurable payload weights.

  Based on Unitree-G1-Flat-Balance-Height but replaces the Inspire Hands
  end effectors with three weight boxes:

  - ``left_hand_weight`` / ``right_hand_weight``: 10x10x10 cm boxes mounted
    at each wrist. Mass is randomized per-episode on reset, in [0, 4] kg
    (independently for left and right — simulates picking up different
    objects in each hand).
  - ``back_weight``: 10x20x20 cm backpack box mounted on the torso.
    Mass is randomized once at startup in [0, 8] kg and stays fixed per env
    (static load).

  The 29-DoF G1 model is used (no fingers). Arms are still randomized and
  nudged exactly like in the balance_height task, so the policy sees a wide
  distribution of arm poses under variable payload.
  """
  cfg = unitree_g1_flat_balance_height_env_cfg(play=play)

  # Swap to the weight-augmented G1 model (no fingers, with payload boxes).
  cfg.scene.entities = {"robot": get_g1_weight_robot_cfg()}

  # Re-derive the leg/waist action scale from the 29-DoF scale dict (the
  # balance parent used G1_INSPIRE_ACTION_SCALE which contains finger
  # patterns that don't exist in this model).
  _balance_action_scale = {
    k: v for k, v in G1_WEIGHT_ACTION_SCALE.items()
    if not any(arm in k for arm in ("shoulder", "elbow", "wrist"))
  }
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = _balance_action_scale

  # --- Payload mass randomization. ---
  # Hand weights: resample on every episode reset to simulate the robot
  # picking up / putting down different objects. Final range: 0-2.5 kg per
  # hand (left and right drawn independently). With kp=40, 2.5 kg causes
  # ~25° sag at the shoulder — the practical limit for stable holding.
  cfg.events["randomize_hand_weights"] = EventTermCfg(
    mode="reset",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", body_names=("left_hand_weight", "right_hand_weight"),
      ),
      "operation": "abs",
      "ranges": (0.0, 0.5),  # curriculum widens this to (0, 2.5)
    },
  )
  # Back weight: sampled once per env at startup, held constant for that
  # env's lifetime (static load, e.g. a fixed backpack). Final range:
  # 0-5 kg (~14% of G1 body mass).
  cfg.events["randomize_back_weight"] = EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("back_weight",)),
      "operation": "abs",
      "ranges": (0.0, 1.0),  # curriculum widens this to (0, 5)
    },
  )

  # --- Payload masses in actor + critic observations. ---
  # Both actor and critic see the current payload masses (3-dim: left
  # hand, right hand, back). For sim-to-real deploy, these values must
  # come from an external source (e.g. a scale sensor, known object
  # weight, or a mass estimator).
  _payload_asset_cfg = SceneEntityCfg(
    "robot",
    body_names=("left_hand_weight", "right_hand_weight", "back_weight"),
  )
  cfg.observations["actor"].terms["payload_masses"] = ObservationTermCfg(
    func=payload_masses,
    params={"asset_cfg": _payload_asset_cfg},
  )
  cfg.observations["critic"].terms["payload_masses"] = ObservationTermCfg(
    func=payload_masses,
    params={"asset_cfg": _payload_asset_cfg},
  )

  # --- Observation history (system identification). ---
  # Stack the last N proprioceptive snapshots so the policy can infer
  # payload mass from the trajectory of joint torques/velocities/gravity,
  # not just a single frame. Noise is applied per-snapshot before being
  # pushed into the history buffer, so each stacked frame carries an
  # independent noise realization (good for robustness). Command and phase
  # are excluded — they're externally supplied and don't carry sys-id
  # signal. payload_masses (privileged) and foot_* sensors are kept as
  # single-frame since the policy doesn't need trajectories of them.
  #
  # NOTE: the base velocity template sets group-level ``history_length=1``,
  # which overrides per-term values. We must clear the group-level override
  # (``None``) so per-term settings take effect.
  _HISTORY_LENGTH = 5
  _HISTORY_TERMS = (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
  )
  _CRITIC_EXTRA_HISTORY_TERMS = ("base_lin_vel",)
  for _group_name, _extras in (
    ("actor", ()),
    ("critic", _CRITIC_EXTRA_HISTORY_TERMS),
  ):
    _group = cfg.observations[_group_name]
    _group.history_length = None  # disable group-level override
    for _term_name in _HISTORY_TERMS + _extras:
      if _term_name in _group.terms:
        _group.terms[_term_name].history_length = _HISTORY_LENGTH

  # --- Curricula that widen the mass ranges over training. ---
  # Hand-weight range is used by a reset-mode event, so resampling happens
  # each episode. Back-weight is startup-only and will keep its initial
  # sampled value per env; the back curriculum only changes what *new*
  # envs (if the sim ever instantiates more) would draw, but we still
  # ramp it for completeness and for evaluation/play use.
  cfg.curriculum["hand_weight_range"] = CurriculumTermCfg(
    func=event_ranges,
    params={
      "event_name": "randomize_hand_weights",
      "stages": [
        {"step": 0,           "ranges": (0.0, 0.5)},
        {"step": 3000 * 24,   "ranges": (0.0, 1.0)},
        {"step": 8000 * 24,   "ranges": (0.0, 1.5)},
        {"step": 14000 * 24,  "ranges": (0.0, 2.0)},
        {"step": 20000 * 24,  "ranges": (0.0, 2.5)},
      ],
    },
  )
  cfg.curriculum["back_weight_range"] = CurriculumTermCfg(
    func=event_ranges,
    params={
      "event_name": "randomize_back_weight",
      "stages": [
        {"step": 0,           "ranges": (0.0, 1.0)},
        {"step": 3000 * 24,   "ranges": (0.0, 2.0)},
        {"step": 8000 * 24,   "ranges": (0.0, 3.0)},
        {"step": 14000 * 24,  "ranges": (0.0, 4.0)},
        {"step": 20000 * 24,  "ranges": (0.0, 5.0)},
      ],
    },
  )

  return cfg
