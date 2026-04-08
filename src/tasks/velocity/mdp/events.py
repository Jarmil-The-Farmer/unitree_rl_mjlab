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
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset: Entity = env.scene[asset_cfg.name]

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)

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
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Randomize arm shoulder_pitch and elbow positions per-env on reset.

  Samples a random arm pose from the given ranges, writes it into qpos
  and sets PD targets to hold it. Does NOT modify default_joint_pos so
  that joint_pos_rel observations reflect the absolute arm position.
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

  # Sample one shoulder_pitch and one elbow value per env.
  sp = torch.empty(n, device=env.device).uniform_(*shoulder_pitch_range)
  el = torch.empty(n, device=env.device).uniform_(*elbow_range)

  # Build position tensor for all arm joints.
  num_joints = len(asset_cfg.joint_ids)
  joint_pos = torch.zeros(n, num_joints, device=env.device)
  for i, name in enumerate(asset_cfg.joint_names):
    if "shoulder_pitch" in name:
      joint_pos[:, i] = sp
    elif "elbow" in name:
      joint_pos[:, i] = el
    # Other arm joints (shoulder_roll, yaw, wrist) stay at 0.

  joint_vel = torch.zeros_like(joint_pos)

  asset.write_joint_state_to_sim(
    joint_pos, joint_vel, env_ids=env_ids, joint_ids=joint_ids_t,
  )

  # Set PD targets to hold the sampled pose.
  for i, jid in enumerate(asset_cfg.joint_ids):
    asset.data.joint_pos_target[env_ids, jid] = joint_pos[:, i]


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
