"""Simulated motor thermal model (winding temperature).

Per-node first-order lumped thermal state driven by ``actuator_force``. The
state lives on the env (``env._motor_thermal``) and is updated every sim step
by a "step" event. Observations, rewards, and terminations read this buffer.

Model (per thermal node i):
    P_i = k_i * sum_{a in sources(i)} tau_a**2
    dT_i/dt = (P_i * R_th_i - (T_i - T_amb)) / tau_th_i

Coupled actuators
-----------------
The G1 waist (pitch+roll) and each ankle (pitch+roll) are 4-bar linkages
driven by TWO shared physical motors. Real-robot logs show that holding a
forward lean (logically attributed to waist_pitch torque, ~30 N*m) heats
*both* waist temperature sensors hard — waist_roll reached 130 C with ~0 of
its own logical torque. So each node in a coupled group is driven by the
COMBINED squared torque of the group, not its own torque alone.

Only the product ``G = k * R_th`` (steady-state gain) and ``tau_th`` are
identifiable from temperature, so calibration fixes ``R_th = 1.0`` and fits
``k = G`` per joint function (see scripts/fit_motor_thermal.py).

Parameter values below are first-pass fits from a single real log (pos1):
waist is well-constrained (large temperature rise); hip/knee are weakly
constrained (small rise) and ankles/waist_yaw are estimates pending more
logs.
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
  """Thermal parameters for one joint function.

  k: heating gain, [K/(N*m)^2/s scaled by tau_th]. Calibrated as G = k*R_th.
  R_th: thermal resistance [K/W]; fixed to 1.0 (folded into k).
  tau_th: thermal time constant of the winding [s].
  """

  k: float
  R_th: float
  tau_th: float


# First-pass params fit from real log pos1 (winding channel, fitted against
# the correct torque source — combined group torque for coupled joints).
# Keyed by joint *function* (left/right share params). REFINE with more logs.
DEFAULT_PARAMS: dict[str, MotorThermalParams] = {
  # Well-constrained (large rise while holding the lean):
  "waist_pitch": MotorThermalParams(k=0.067, R_th=1.0, tau_th=13.0),
  "waist_roll": MotorThermalParams(k=0.107, R_th=1.0, tau_th=20.0),
  # Weakly constrained (only ~10 C rise in 30 s, single short log):
  "hip_pitch": MotorThermalParams(k=0.090, R_th=1.0, tau_th=27.0),
  "hip_roll": MotorThermalParams(k=0.032, R_th=1.0, tau_th=19.0),
  "hip_yaw": MotorThermalParams(k=0.059, R_th=1.0, tau_th=18.0),
  "knee": MotorThermalParams(k=0.043, R_th=1.0, tau_th=37.0),
  # No usable signal in pos1 (near-zero torque) — estimates by analogy:
  # waist_yaw ~ hip_yaw (both 7520_14); ankles ~ waist (both paired 2x5020).
  "waist_yaw": MotorThermalParams(k=0.059, R_th=1.0, tau_th=18.0),
  "ankle_pitch": MotorThermalParams(k=0.067, R_th=1.0, tau_th=15.0),
  "ankle_roll": MotorThermalParams(k=0.090, R_th=1.0, tau_th=18.0),
}


@dataclass
class ThermalNode:
  """One tracked motor temperature.

  joint: the single joint whose temperature sensor this node represents
    (also used as the observation label / ordering).
  param_key: key into ``MotorThermalCfg.params`` (the joint function).
  torque_sources: joint names whose squared torque drive this node's heat.
    For independent joints this is just ``(joint,)``; for coupled linkages
    it is the whole group (e.g. both waist joints).
  """

  joint: str
  param_key: str
  torque_sources: tuple[str, ...]


def _g1_leg_waist_nodes() -> tuple[ThermalNode, ...]:
  """Build the 15 leg+waist thermal nodes for the G1, with coupling."""
  nodes: list[ThermalNode] = []

  # Independent leg joints (each driven by its own torque), left + right.
  for side in ("left", "right"):
    for fn in ("hip_pitch", "hip_roll", "hip_yaw", "knee"):
      jn = f"{side}_{fn}_joint"
      nodes.append(ThermalNode(joint=jn, param_key=fn, torque_sources=(jn,)))

  # Coupled ankle: pitch+roll on each leg share two physical motors.
  for side in ("left", "right"):
    group = (f"{side}_ankle_pitch_joint", f"{side}_ankle_roll_joint")
    nodes.append(ThermalNode(f"{side}_ankle_pitch_joint", "ankle_pitch", group))
    nodes.append(ThermalNode(f"{side}_ankle_roll_joint", "ankle_roll", group))

  # Waist: yaw is independent; pitch+roll are a coupled pair.
  nodes.append(ThermalNode("waist_yaw_joint", "waist_yaw", ("waist_yaw_joint",)))
  waist_group = ("waist_pitch_joint", "waist_roll_joint")
  nodes.append(ThermalNode("waist_pitch_joint", "waist_pitch", waist_group))
  nodes.append(ThermalNode("waist_roll_joint", "waist_roll", waist_group))

  return tuple(nodes)


@dataclass
class MotorThermalCfg:
  nodes: tuple[ThermalNode, ...] = field(default_factory=_g1_leg_waist_nodes)
  params: dict[str, MotorThermalParams] = field(
    default_factory=lambda: dict(DEFAULT_PARAMS)
  )
  T_amb: float = 30.0
  init_T_range: tuple[float, float] = (30.0, 60.0)
  # Per-env multiplicative randomization on (k, R_th, tau_th) for DR.
  dr_range: tuple[float, float] = (0.7, 1.3)


class MotorThermal:
  """Per-env motor winding temperatures with DR-re-sampleable parameters.

  Lives on the env at ``env._motor_thermal``. Created lazily on first access
  so the env / asset are fully constructed.
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
    device = env.device

    self.joint_names = tuple(n.joint for n in cfg.nodes)
    self.num_motors = len(cfg.nodes)

    actuator_names = asset.actuator_names

    def _actuator_id(joint_name: str) -> int:
      if joint_name in actuator_names:
        return actuator_names.index(joint_name)
      stem = joint_name.removesuffix("_joint")
      matches = [i for i, a in enumerate(actuator_names) if stem in a]
      assert matches, f"No actuator found for joint '{joint_name}'"
      return matches[0]

    # Union of all source actuators (column space of the mixing matrix).
    src_actuator_set: list[int] = []
    for node in cfg.nodes:
      for src in node.torque_sources:
        aid = _actuator_id(src)
        if aid not in src_actuator_set:
          src_actuator_set.append(aid)
    self.src_actuator_ids = torch.tensor(
      src_actuator_set, device=device, dtype=torch.long
    )
    col_of = {aid: c for c, aid in enumerate(src_actuator_set)}

    # Mixing matrix: src_mask[node, col] = 1 if that actuator heats the node.
    # P_node = k_node * sum_col src_mask[node, col] * tau[col]**2.
    src_mask = torch.zeros((self.num_motors, len(src_actuator_set)), device=device)
    for i, node in enumerate(cfg.nodes):
      for src in node.torque_sources:
        src_mask[i, col_of[_actuator_id(src)]] = 1.0
    self.src_mask = src_mask

    # Per-node base params.
    self.k_base = torch.tensor(
      [cfg.params[n.param_key].k for n in cfg.nodes], device=device
    )
    self.R_base = torch.tensor(
      [cfg.params[n.param_key].R_th for n in cfg.nodes], device=device
    )
    self.tau_base = torch.tensor(
      [cfg.params[n.param_key].tau_th for n in cfg.nodes], device=device
    )

    n = env.num_envs
    m = self.num_motors
    self.k_scale = torch.ones((n, m), device=device)
    self.R_scale = torch.ones((n, m), device=device)
    self.tau_scale = torch.ones((n, m), device=device)
    self.T = torch.full((n, m), cfg.T_amb, device=device, dtype=torch.float32)

    self.reset(torch.arange(n, device=device, dtype=torch.long))

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
    tau = asset.data.actuator_force[:, self.src_actuator_ids]  # [B, A]
    tau2 = tau.pow(2)
    # Combined squared torque per node via the mixing matrix: [B, M].
    src_power = tau2 @ self.src_mask.t()
    k = self.k_base.unsqueeze(0) * self.k_scale
    R_th = self.R_base.unsqueeze(0) * self.R_scale
    tau_th = self.tau_base.unsqueeze(0) * self.tau_scale
    P = k * src_power
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
