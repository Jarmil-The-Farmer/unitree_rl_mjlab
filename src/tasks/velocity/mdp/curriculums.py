from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to simpler
  # terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    # "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    # "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    # "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    # "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    # "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    # "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


class StandingBalanceStage(TypedDict):
  step: int
  rel_standing_envs: float | None
  nudge_speed: float | None
  nudge_offset_range: tuple[float, float] | None


def standing_balance(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  nudge_event_name: str,
  stages: list[StandingBalanceStage],
) -> dict[str, torch.Tensor]:
  """Gradually increase standing ratio and arm nudge intensity.

  Early training focuses on walking; later stages shift towards standing
  balance with arm perturbations so the robot learns to stay still while
  arms are being moved.
  """
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  nudge_cfg = env.event_manager.get_term_cfg(nudge_event_name)
  for stage in stages:
    if env.common_step_counter > stage["step"]:
      if stage.get("rel_standing_envs") is not None:
        cfg.rel_standing_envs = stage["rel_standing_envs"]
      if stage.get("nudge_speed") is not None:
        nudge_cfg.params["speed"] = stage["nudge_speed"]
      if stage.get("nudge_offset_range") is not None:
        nudge_cfg.params["position_offset_range"] = stage["nudge_offset_range"]
  return {}


class ArmRangeStage(TypedDict):
  step: int
  shoulder_pitch_range: tuple[float, float]
  elbow_range: tuple[float, float]


def arm_pose_randomization_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reset_event_name: str,
  stages: list[ArmRangeStage],
) -> dict[str, torch.Tensor]:
  """Gradually widen the arm pose randomization range.

  Updates the shoulder_pitch_range and elbow_range parameters of the
  randomize_arm_pose reset event. Early training uses a narrow range
  near (0, 0), later stages cover the full arm workspace.

  Between stages the ranges are linearly interpolated so the transition
  is smooth rather than a sudden jump.
  """
  del env_ids  # Unused.
  step = env.common_step_counter

  # Find the two surrounding stages and interpolate.
  prev = stages[0]
  sp_range = prev["shoulder_pitch_range"]
  el_range = prev["elbow_range"]
  for stage in stages:
    if step >= stage["step"]:
      prev = stage
      sp_range = prev["shoulder_pitch_range"]
      el_range = prev["elbow_range"]
    else:
      t = (step - prev["step"]) / max(stage["step"] - prev["step"], 1)
      sp_range = (
        prev["shoulder_pitch_range"][0] + t * (stage["shoulder_pitch_range"][0] - prev["shoulder_pitch_range"][0]),
        prev["shoulder_pitch_range"][1] + t * (stage["shoulder_pitch_range"][1] - prev["shoulder_pitch_range"][1]),
      )
      el_range = (
        prev["elbow_range"][0] + t * (stage["elbow_range"][0] - prev["elbow_range"][0]),
        prev["elbow_range"][1] + t * (stage["elbow_range"][1] - prev["elbow_range"][1]),
      )
      break

  # Update the reset event's params so next reset samples from the new range.
  event_cfg = env.event_manager.get_term_cfg(reset_event_name)
  event_cfg.params["shoulder_pitch_range"] = sp_range
  event_cfg.params["elbow_range"] = el_range

  return {}


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])
