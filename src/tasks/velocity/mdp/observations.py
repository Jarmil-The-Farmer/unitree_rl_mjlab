from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def payload_masses(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Per-env masses of selected bodies (privileged critic observation).

  Returns a ``(num_envs, len(body_ids))`` tensor read directly from the
  per-world ``sim.model.body_mass`` buffer. Meant for asymmetric actor/
  critic setups where the critic sees the current payload while the actor
  must infer it from proprioception.
  """
  asset: Entity = env.scene[asset_cfg.name]
  # asset_cfg.body_ids are entity-local; sim.model.body_mass is sim-global.
  # indexing.body_ids maps entity-local -> sim-global.
  global_ids = asset.indexing.body_ids[asset_cfg.body_ids]
  return env.sim.model.body_mass[:, global_ids].to(env.device)


def motor_temperatures(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  T_amb: float = 25.0,
  T_scale: float = 50.0,
) -> torch.Tensor:
  """Simulated motor temperatures as a privileged critic observation.

  Returns ``(num_envs, num_tracked_motors)`` of normalized temperatures
  ``(T - T_amb) / T_scale``. Creates the thermal state with default cfg
  on first call (so the observation manager can determine obs width
  during ``_prepare_terms``).
  """
  from src.tasks.velocity.mdp.thermal import get_or_create as _get_thermal
  state = _get_thermal(env, asset_cfg)
  return (state.T - T_amb) / T_scale


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    # Only check velocity channels (:3) — ignore extra channels like height.
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase

