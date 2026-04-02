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
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset: Entity = env.scene[asset_cfg.name]
  asset.data.joint_pos_target[env_ids[:, None], asset_cfg.joint_ids] = (
    asset.data.default_joint_pos[env_ids[:, None], asset_cfg.joint_ids]
  )
