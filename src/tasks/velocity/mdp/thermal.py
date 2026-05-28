"""Simulated motor thermal model.

Per-joint first-order lumped thermal state driven by ``actuator_force``.
The state lives on the env (``env._motor_thermal``) and is updated every
sim step by a "step" event. Observations, rewards, and terminations read
from this buffer.

Model (per joint i):
    P_i = k_type[i] * coupling[i] * tau_i**2
    dT_i/dt = (P_i * R_th[i] - (T_i - T_amb)) / tau_th[i]

For joints with paired actuators (waist_pitch/roll, ankle_pitch/roll on G1),
the joint torque is shared by two motors. We track one temperature per joint
and use ``coupling=0.5`` so the per-motor squared torque is ``(tau/2)**2``
summed over two motors == ``tau**2 / 2``.

Real-world calibration replaces the (k, R_th, tau_th) tuples per motor type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_STATE_KEY = "_motor_thermal"


@dataclass
class MotorThermalParams:
  """Per-motor-type thermal parameters.

  k: heating coefficient, [W / (N*m)**2]. Roughly ``1 / (Kt**2 * R_phase)``.
  R_th: thermal resistance winding-to-ambient, [K/W].
  tau_th: thermal time constant, [s].
  """

  k: float
  R_th: float
  tau_th: float


# Datasheet-estimate placeholders. Replace per motor type via real-log fit
# (see scripts/fit_motor_thermal.py).
DEFAULT_PARAMS: dict[str, MotorThermalParams] = {
  "5020": MotorThermalParams(k=0.050, R_th=2.0, tau_th=90.0),
  "7520_14": MotorThermalParams(k=0.015, R_th=1.0, tau_th=120.0),
  "7520_22": MotorThermalParams(k=0.008, R_th=0.8, tau_th=150.0),
  "4010": MotorThermalParams(k=0.10, R_th=3.0, tau_th=60.0),
}


# G1 leg + waist joints (15) — joint regex -> (motor_type, coupling).
# coupling=0.5 for joints driven by two parallel 5020 actuators.
G1_LEG_WAIST_THERMAL_MAP: tuple[tuple[str, str, float], ...] = (
  (".*_hip_pitch_joint", "7520_14", 1.0),
  (".*_hip_roll_joint", "7520_22", 1.0),
  (".*_hip_yaw_joint", "7520_14", 1.0),
  (".*_knee_joint", "7520_22", 1.0),
  (".*_ankle_pitch_joint", "5020", 0.5),
  (".*_ankle_roll_joint", "5020", 0.5),
  ("waist_yaw_joint", "7520_14", 1.0),
  ("waist_pitch_joint", "5020", 0.5),
  ("waist_roll_joint", "5020", 0.5),
)


@dataclass
class MotorThermalCfg:
  joint_map: tuple[tuple[str, str, float], ...] = G1_LEG_WAIST_THERMAL_MAP
  params: dict[str, MotorThermalParams] = field(
    default_factory=lambda: dict(DEFAULT_PARAMS)
  )
  T_amb: float = 25.0
  init_T_range: tuple[float, float] = (25.0, 55.0)
  # Per-env multiplicative randomization on (k, R_th, tau_th) for DR.
  dr_range: tuple[float, float] = (0.7, 1.3)


class MotorThermal:
  """Per-env motor temperatures with DR-re-sampleable parameters.

  Lives on the env at ``env._motor_thermal``. Created lazily on first
  reset event call so that the env / asset are fully constructed.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    cfg: MotorThermalCfg,
    asset_cfg: SceneEntityCfg,
  ):
    self.cfg = cfg
    self.asset_name = asset_cfg.name
    asset: Entity = env.scene[asset_cfg.name]

    joint_ids: list[int] = []
    joint_names: list[str] = []
    motor_types: list[str] = []
    couplings: list[float] = []
    for pattern, motor_type, coupling in cfg.joint_map:
      ids, names = asset.find_joints(pattern)
      for jid, name in zip(ids, names):
        if jid in joint_ids:
          continue
        joint_ids.append(jid)
        joint_names.append(name)
        motor_types.append(motor_type)
        couplings.append(coupling)

    self.joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
    self.joint_names = tuple(joint_names)
    self.motor_types = tuple(motor_types)
    self.num_motors = len(joint_ids)

    # The thermal model reads actuator_force[ctrl], not joint torques.
    # Map each tracked joint to its actuator. On G1 the actuator name
    # matches the joint name; fall back to substring match otherwise.
    actuator_names = asset.actuator_names
    actuator_ids: list[int] = []
    for name in joint_names:
      if name in actuator_names:
        actuator_ids.append(actuator_names.index(name))
      else:
        stem = name.removesuffix("_joint")
        matches = [i for i, a in enumerate(actuator_names) if stem in a]
        assert matches, f"No actuator found for joint '{name}'"
        actuator_ids.append(matches[0])
    self.actuator_ids = torch.tensor(
      actuator_ids, device=env.device, dtype=torch.long
    )

    self.k_base = torch.tensor(
      [cfg.params[t].k for t in motor_types], device=env.device
    )
    self.R_base = torch.tensor(
      [cfg.params[t].R_th for t in motor_types], device=env.device
    )
    self.tau_base = torch.tensor(
      [cfg.params[t].tau_th for t in motor_types], device=env.device
    )
    self.coupling = torch.tensor(couplings, device=env.device)

    n = env.num_envs
    m = self.num_motors
    self.k_scale = torch.ones((n, m), device=env.device)
    self.R_scale = torch.ones((n, m), device=env.device)
    self.tau_scale = torch.ones((n, m), device=env.device)
    self.T = torch.full(
      (n, m), cfg.T_amb, device=env.device, dtype=torch.float32
    )

    self.reset(torch.arange(n, device=env.device, dtype=torch.long))

  def reset(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    n = len(env_ids)
    m = self.num_motors
    device = self.T.device
    lo, hi = self.cfg.init_T_range
    self.T[env_ids] = torch.empty((n, m), device=device).uniform_(lo, hi)
    dr_lo, dr_hi = self.cfg.dr_range
    self.k_scale[env_ids] = torch.empty((n, m), device=device).uniform_(dr_lo, dr_hi)
    self.R_scale[env_ids] = torch.empty((n, m), device=device).uniform_(dr_lo, dr_hi)
    self.tau_scale[env_ids] = torch.empty((n, m), device=device).uniform_(dr_lo, dr_hi)

  def step(self, env: ManagerBasedRlEnv, dt: float) -> None:
    asset: Entity = env.scene[self.asset_name]
    tau = asset.data.actuator_force[:, self.actuator_ids]  # [B, M]
    k = self.k_base.unsqueeze(0) * self.k_scale
    R_th = self.R_base.unsqueeze(0) * self.R_scale
    tau_th = self.tau_base.unsqueeze(0) * self.tau_scale

    P = k * self.coupling.unsqueeze(0) * tau.pow(2)
    dT = dt * (P * R_th - (self.T - self.cfg.T_amb)) / tau_th
    self.T = self.T + dT


def get_or_create(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  cfg: MotorThermalCfg | None = None,
) -> MotorThermal:
  state: MotorThermal | None = getattr(env, _STATE_KEY, None)
  if state is None:
    state = MotorThermal(env, cfg or MotorThermalCfg(), asset_cfg)
    setattr(env, _STATE_KEY, state)
  return state
