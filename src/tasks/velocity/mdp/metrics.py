"""Diagnostic metric functions for velocity tasks.

Each function returns a per-env scalar tensor; MetricsManager logs the
episode-mean under ``Episode_Metrics/<name>``. Use these to track balance
quality (lateral lean, leg spread) without polluting the reward sum.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_spread_lateral(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  foot_site_names: tuple[str, str] = ("left_foot", "right_foot"),
) -> torch.Tensor:
  """Lateral (Y world-frame) distance between the two foot sites."""
  asset: Entity = env.scene[asset_cfg.name]
  site_ids, _ = asset.find_sites(list(foot_site_names))
  foot_pos = asset.data.site_pos_w[:, site_ids, :]  # [B, 2, 3]
  return (foot_pos[:, 0, 1] - foot_pos[:, 1, 1]).abs()


def pelvis_roll_abs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """|Y component of projected gravity in pelvis (root link) frame|.

  Sideways pelvis tilt — non-zero when the pelvis is rolled to one side
  (one hip higher than the other). Range 0..1.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.projected_gravity_b[:, 1].abs()


def pelvis_pitch_abs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """|X component of projected gravity in pelvis (root link) frame|.

  Forward/backward pelvis lean. Range 0..1.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.projected_gravity_b[:, 0].abs()


def waist_roll_abs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Absolute value of the waist_roll joint position [rad].

  Tracks sideways waist (torso) tilt — distinct from pelvis tilt which
  comes from asymmetric hip_roll.
  """
  asset: Entity = env.scene[asset_cfg.name]
  joint_names = list(asset.joint_names)
  if "waist_roll_joint" not in joint_names:
    return torch.zeros(env.num_envs, device=env.device)
  jid = joint_names.index("waist_roll_joint")
  return asset.data.joint_pos[:, jid].abs()


def hip_roll_abs_mean(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Mean absolute hip_roll joint position across both legs.

  High values indicate the legs are spread wide (squat with knees out).
  """
  asset: Entity = env.scene[asset_cfg.name]
  joint_names = list(asset.joint_names)
  ids = [i for i, n in enumerate(joint_names) if "hip_roll" in n]
  return asset.data.joint_pos[:, ids].abs().mean(dim=1)
