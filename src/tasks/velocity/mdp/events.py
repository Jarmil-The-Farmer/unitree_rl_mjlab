"""Custom event functions for velocity tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def nudge_joints_velocity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  velocity_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Apply random velocity perturbations to joints without changing positions.

  Unlike ``reset_joints_by_offset``, this does **not** reset joint positions.
  It only writes random velocities, so PD controllers smoothly dampen the
  perturbation and produce natural, gradual motion.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  if len(env_ids) == 0:
    return

  asset: Entity = env.scene[asset_cfg.name]

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)

  joint_vel = torch.empty(
    (len(env_ids), len(asset_cfg.joint_ids)), device=env.device
  ).uniform_(*velocity_range)

  asset.write_joint_velocity_to_sim(
    joint_vel,
    env_ids=env_ids,
    joint_ids=joint_ids,
  )


def randomize_arm_pose(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  shoulder_pitch_range: tuple[float, float],
  elbow_range: tuple[float, float],
  shoulder_roll_range: tuple[float, float] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Randomize arm positions per-env on reset.

  Samples random shoulder_pitch, elbow, and optionally shoulder_roll values.
  For shoulder_roll, left side uses the range as-is (positive = away from body)
  and right side uses the negated range (negative = away from body).

  Does NOT modify default_joint_pos so that joint_pos_rel observations
  reflect the absolute arm position.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  n = len(env_ids)
  if n == 0:
    return

  asset: Entity = env.scene[asset_cfg.name]

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids_t = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
  else:
    joint_ids_t = joint_ids

  # Sample one value per env for each joint type (applied symmetrically).
  sp = torch.empty(n, device=env.device).uniform_(*shoulder_pitch_range)
  el = torch.empty(n, device=env.device).uniform_(*elbow_range)
  sr = None
  if shoulder_roll_range is not None:
    sr = torch.empty(n, device=env.device).uniform_(*shoulder_roll_range)

  # Build position tensor for all arm joints.
  # joint_names may be regex patterns (7) while joint_ids are resolved (14,
  # covering both left and right). Use the resolved entity joint names to
  # match by actual name instead of iterating over the pattern list.
  num_joints = len(asset_cfg.joint_ids)
  joint_pos = torch.zeros(n, num_joints, device=env.device)
  all_joint_names = asset.joint_names
  for i, jid in enumerate(asset_cfg.joint_ids):
    name = all_joint_names[jid]
    if "shoulder_pitch" in name:
      joint_pos[:, i] = sp
    elif "shoulder_roll" in name:
      if sr is not None:
        # Left side: positive roll = away from body.
        # Right side: negative roll = away from body.
        if "right" in name:
          joint_pos[:, i] = -sr
        else:
          joint_pos[:, i] = sr
    elif "elbow" in name:
      joint_pos[:, i] = el
    # Other arm joints (shoulder_yaw, wrist) stay at 0.

  joint_vel = torch.zeros_like(joint_pos)

  asset.write_joint_state_to_sim(
    joint_pos, joint_vel, env_ids=env_ids, joint_ids=joint_ids_t,
  )

  # Set PD targets to hold the sampled pose.
  for i, jid in enumerate(asset_cfg.joint_ids):
    asset.data.joint_pos_target[env_ids, jid] = joint_pos[:, i]


def nudge_joints_position(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  position_offset_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Apply random position offsets to joints and update PD targets.

  Unlike ``nudge_joints_velocity`` which only perturbs velocities and lets PD
  controllers dampen the motion, this function directly moves joints to new
  positions (current + random offset, clamped to limits) and sets PD targets
  to hold them there.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  if len(env_ids) == 0:
    return

  asset: Entity = env.scene[asset_cfg.name]

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids_t = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
  else:
    joint_ids_t = joint_ids

  # Current positions for the selected joints.
  current_pos = asset.data.joint_pos[env_ids][:, asset_cfg.joint_ids]

  # Random offsets.
  offsets = torch.empty_like(current_pos).uniform_(*position_offset_range)
  new_pos = current_pos + offsets

  # Clamp to joint limits.
  limits = asset.data.soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
  new_pos = torch.clamp(new_pos, limits[:, :, 0], limits[:, :, 1])

  # Write new position with zero velocity (no jerking).
  joint_vel = torch.zeros_like(new_pos)
  asset.write_joint_state_to_sim(
    new_pos, joint_vel, env_ids=env_ids, joint_ids=joint_ids_t,
  )

  # Update PD targets to hold the new position.
  for i, jid in enumerate(asset_cfg.joint_ids):
    asset.data.joint_pos_target[env_ids, jid] = new_pos[:, i]


def set_joint_targets_to_default(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Set actuator position targets for specified joints to their default (keyframe) values.

  This prevents PD controllers from driving joints back to 0 after reset,
  which is critical for joints not controlled by the RL policy (e.g. arms).
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  if len(env_ids) == 0:
    return

  asset: Entity = env.scene[asset_cfg.name]
  for jid in asset_cfg.joint_ids:
    asset.data.joint_pos_target[env_ids, jid] = asset.data.default_joint_pos[env_ids, jid]
