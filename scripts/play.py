"""Script to play RL agent with RSL-RL."""

import math
import os
import select
import sys
import termios
import threading
import tty
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


def _parse_obs_terms_from_yaml(yaml_path: Path, group_name: str) -> list[str] | None:
  """Parse observation term names from a saved env.yaml file."""
  try:
    with open(yaml_path) as f:
      lines = f.readlines()
  except OSError:
    return None
  in_obs = False
  in_group = False
  in_terms = False
  terms: list[str] = []
  for line in lines:
    s = line.rstrip()
    if s == "observations:":
      in_obs = True
      continue
    if in_obs and s == f"  {group_name}:":
      in_group = True
      continue
    if in_group and s == "    terms:":
      in_terms = True
      continue
    if in_terms:
      if s and not s.startswith("      "):
        break
      if s.startswith("      ") and not s.startswith("        "):
        name = s.strip().rstrip(":")
        if name and not name.startswith("#"):
          terms.append(name)
  return terms or None


def _parse_obs_joint_names(yaml_path: Path, group_name: str, term_name: str) -> list[str] | None:
  """Parse joint_names from an observation term in a saved env.yaml file."""
  import re
  try:
    with open(yaml_path) as f:
      content = f.read()
  except OSError:
    return None
  # Find the term's joint_names list.
  pattern = (
    rf'  {group_name}:\s+terms:\s+(?:.*?\n)*?\s+{term_name}:'
    rf'.*?joint_names: !!python/tuple\n((?:\s+- [\w]+\n)+)'
  )
  match = re.search(pattern, content, re.DOTALL)
  if not match:
    return None
  names = [line.strip().lstrip('- ') for line in match.group(1).strip().split('\n')]
  return names if names else None


def _reconcile_obs_with_checkpoint(env_cfg, checkpoint_path: Path):
  """Adjust env_cfg observations to match a checkpoint's saved config.

  Looks for params/env.yaml next to the checkpoint. If found, removes
  observation terms that weren't present during training and adjusts
  joint_names in joint_pos/joint_vel observations to match the checkpoint's
  dimensions.
  """
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  env_yaml = checkpoint_path.parent / "params" / "env.yaml"
  if not env_yaml.exists():
    return
  saved_actor_terms = _parse_obs_terms_from_yaml(env_yaml, "actor")
  if saved_actor_terms is None:
    return
  current_actor_terms = list(env_cfg.observations["actor"].terms.keys())
  if saved_actor_terms == current_actor_terms:
    # Terms match, but joint dimensions may differ. Check joint_names.
    pass
  # Remove terms not in saved config.
  removed = []
  for name in list(env_cfg.observations["actor"].terms.keys()):
    if name not in saved_actor_terms:
      del env_cfg.observations["actor"].terms[name]
      removed.append(name)
  for name in removed:
    if "critic" in env_cfg.observations:
      env_cfg.observations["critic"].terms.pop(name, None)
  if removed:
    print(f"[INFO] Adjusted config to match checkpoint: removed {removed}")
  missing = [n for n in saved_actor_terms if n not in current_actor_terms]
  if missing:
    print(f"[WARN] Checkpoint expects terms not in current config: {missing}")

  # Reconcile joint_names for joint_pos and joint_vel observations.
  for term_name in ("joint_pos", "joint_vel"):
    saved_names = _parse_obs_joint_names(env_yaml, "actor", term_name)
    if saved_names is None:
      continue
    for group_key in ("actor", "critic"):
      group = env_cfg.observations.get(group_key)
      if group is None or term_name not in group.terms:
        continue
      term_cfg = group.terms[term_name]
      current_cfg = term_cfg.params.get("asset_cfg")
      if current_cfg is None:
        continue
      current_names = getattr(current_cfg, "joint_names", None)
      if current_names is not None and list(current_names) != saved_names:
        term_cfg.params["asset_cfg"] = SceneEntityCfg(
          current_cfg.name, joint_names=tuple(saved_names)
        )
        print(f"[INFO] Adjusted {group_key}/{term_name} joint_names: "
              f"{len(list(current_names))} -> {len(saved_names)} joints")


def _log_terminations(env):
  """Print termination reasons to console when any environment resets."""
  unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
  tm = unwrapped.termination_manager
  if not tm.dones.any():
    return
  reasons = []
  for name in tm.active_terms:
    mask = tm.get_term(name)
    if mask.any():
      ids = mask.nonzero(as_tuple=False).squeeze(-1).tolist()
      if isinstance(ids, int):
        ids = [ids]
      reasons.append(f"  {name}: env(s) {ids}")
  if reasons:
    # Also print orientation angle for debugging fell_over.
    robot = unwrapped.scene["robot"]
    pg = robot.data.projected_gravity_b
    tilt_deg = torch.acos(-pg[:, 2]).abs() * (180.0 / math.pi)
    tilt_str = ", ".join(f"{t:.1f}°" for t in tilt_deg.cpu().tolist())
    print(f"[RESET] Termination triggered (tilt: {tilt_str}):")
    for r in reasons:
      print(r)


def _short_joint_label(name: str) -> str:
  """Compact joint label for HUD/console (e.g. left_hip_pitch_joint -> L_hip_pitch)."""
  name = name.removesuffix("_joint")
  name = name.replace("left_", "L_").replace("right_", "R_")
  return name


def _get_motor_thermal(env):
  unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
  return getattr(unwrapped, "_motor_thermal", None)


def _motor_temp_lines(env, env_idx: int, top_k: int = 4):
  """Return list of (label, temp_celsius) for the hottest tracked motors.

  Returns None if the thermal model is not active in the current config.
  """
  state = _get_motor_thermal(env)
  if state is None:
    return None
  T = state.T[env_idx].detach().cpu()
  order = torch.argsort(T, descending=True)
  return [
    (_short_joint_label(state.joint_names[i]), T[i].item())
    for i in order[:top_k].tolist()
  ]


class _WeightController:
  """Console keyboard controller for payload masses in balance_weight task.

  Reads single keys from stdin (raw mode) in a background thread and lets
  the user select which weight box (left hand / right hand / back) to
  edit, then increment or decrement its mass. The main step loop calls
  :meth:`apply` every step, which overwrites the sim ``body_mass`` tensor
  with the user-set values so that randomization events cannot drift the
  masses back on reset.
  """

  _LABELS = {
    "left_hand_weight": "L-hand",
    "right_hand_weight": "R-hand",
    "back_weight": "Back",
  }
  _MAX_MASS = {
    "left_hand_weight": 8.0,
    "right_hand_weight": 8.0,
    "back_weight": 16.0,
  }

  def __init__(self, env, step_size: float = 0.25):
    self._env = env
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    self._sim = unwrapped.sim
    self._sim_model = unwrapped.sim.model
    asset = unwrapped.scene["robot"]

    # Build body-name -> sim-global index map (same for every env).
    self._body_to_idx: dict[str, int] = {}
    for name in self._LABELS:
      if name not in asset.body_names:
        raise RuntimeError(f"Weight body '{name}' not found in robot")
      local = asset.body_names.index(name)
      self._body_to_idx[name] = int(asset.indexing.body_ids[local].item())

    # Current user-set mass for each body (initial: whatever is in sim now).
    self._masses: dict[str, float] = {
      n: float(self._sim_model.body_mass[:, i].mean().item())
      for n, i in self._body_to_idx.items()
    }
    self._selected: str = "left_hand_weight"
    self._step = step_size
    self._dirty = True  # force apply + recompute on first step

    self._running = True
    self._tty_restore = None
    self._thread: threading.Thread | None = None

  # Public API ------------------------------------------------------------

  def start(self) -> None:
    """Start the background stdin reader. Silently no-ops if no TTY."""
    if not sys.stdin.isatty():
      print(
        "[Weights] stdin is not a TTY — keyboard control disabled. "
        "Masses will stay at their randomized values."
      )
      return
    try:
      self._tty_restore = termios.tcgetattr(sys.stdin.fileno())
      tty.setcbreak(sys.stdin.fileno())
    except Exception as e:
      print(f"[Weights] Failed to enter raw tty mode: {e}")
      return
    self._thread = threading.Thread(target=self._stdin_loop, daemon=True)
    self._thread.start()
    self._print_help()
    self._print_status()

  def stop(self) -> None:
    self._running = False
    if self._tty_restore is not None:
      try:
        termios.tcsetattr(
          sys.stdin.fileno(), termios.TCSADRAIN, self._tty_restore
        )
      except Exception:
        pass
      self._tty_restore = None

  def apply(self) -> None:
    """Called every viewer step from the main thread.

    Always writes the user-set masses to ``sim.model.body_mass`` so that
    reset-time randomization events cannot drift values. Recomputes
    constants only when a value actually changed (cheap-ish but nontrivial).
    """
    for name, idx in self._body_to_idx.items():
      self._sim_model.body_mass[:, idx] = self._masses[name]
    if self._dirty:
      self._dirty = False
      try:
        # Import locally — avoids importing at module top if mjlab version
        # doesn't expose the symbol.
        from mjlab.managers.event_manager import RecomputeLevel
        self._sim.recompute_constants(RecomputeLevel.set_const)
      except Exception as e:
        print(f"[Weights] recompute_constants failed: {e}")

  # Private ---------------------------------------------------------------

  def _print_help(self) -> None:
    print(
      "\n[Weights] Controls:\n"
      "  l / r / b       select Left / Right hand / Back weight\n"
      "  + / =           increase selected by step\n"
      "  -               decrease selected by step\n"
      "  0               zero selected\n"
      "  [ / ]           halve / double step size\n"
      "  a               set all to zero\n"
      "  p               print current masses\n"
      "  ?               this help\n"
    )

  def _print_status(self) -> None:
    parts = []
    for name in self._LABELS:
      lbl = self._LABELS[name]
      v = self._masses[name]
      marker = "*" if name == self._selected else " "
      parts.append(f"{marker}{lbl}={v:5.2f}kg")
    print(f"[Weights] step={self._step:.2f}kg  " + "  ".join(parts))

  def _clamp(self, name: str, value: float) -> float:
    return max(0.0, min(self._MAX_MASS[name], value))

  def _adjust(self, delta: float) -> None:
    n = self._selected
    new = self._clamp(n, self._masses[n] + delta)
    if new != self._masses[n]:
      self._masses[n] = new
      self._dirty = True
    self._print_status()

  def _set(self, name: str, value: float) -> None:
    new = self._clamp(name, value)
    if new != self._masses[name]:
      self._masses[name] = new
      self._dirty = True
    self._print_status()

  def _stdin_loop(self) -> None:
    fd = sys.stdin.fileno()
    while self._running:
      try:
        r, _, _ = select.select([fd], [], [], 0.1)
      except Exception:
        break
      if not r:
        continue
      try:
        ch = sys.stdin.read(1)
      except Exception:
        break
      if not ch:
        continue
      if ch in ("l", "L"):
        self._selected = "left_hand_weight"
        self._print_status()
      elif ch in ("r", "R"):
        self._selected = "right_hand_weight"
        self._print_status()
      elif ch in ("b", "B"):
        self._selected = "back_weight"
        self._print_status()
      elif ch in ("+", "="):
        self._adjust(+self._step)
      elif ch == "-":
        self._adjust(-self._step)
      elif ch == "0":
        self._set(self._selected, 0.0)
      elif ch == "a":
        for n in self._LABELS:
          self._set(n, 0.0)
      elif ch == "[":
        self._step = max(0.05, self._step / 2)
        self._print_status()
      elif ch == "]":
        self._step = min(4.0, self._step * 2)
        self._print_status()
      elif ch == "p":
        self._print_status()
      elif ch == "?":
        self._print_help()
      # Ignore all other keys (including Ctrl-C — handled by main thread).


class _TermLoggingViewer(NativeMujocoViewer):
  """NativeMujocoViewer that prints termination reasons to console."""

  def __init__(self, env, policy, *, weight_ctrl: _WeightController | None = None, **kwargs):
    super().__init__(env, policy, **kwargs)
    self._weight_ctrl = weight_ctrl
    self._temp_print_counter = 0

  def _print_motor_temps(self) -> None:
    """Periodically print the hottest motor temperatures to console."""
    self._temp_print_counter += 1
    if self._temp_print_counter % 50 != 0:  # ~1 s at 50 Hz control
      return
    lines = _motor_temp_lines(self.env, self.env_idx, top_k=4)
    if not lines:
      return
    parts = [f"{label}={t:.1f}" for label, t in lines]
    hottest = lines[0][1]
    flag = ""
    if hottest >= 90.0:
      flag = " [CRIT>90]"
    elif hottest >= 70.0:
      flag = " [WARN>70]"
    print(f"[MotorT] " + "  ".join(parts) + flag)

  def _execute_step(self) -> bool:
    if self._weight_ctrl is not None:
      self._weight_ctrl.apply()
    result = super()._execute_step()
    if result:
      _log_terminations(self.env)
    self._print_motor_temps()
    return result


_ARM_MODES = [
  {"name": "Default (0, 0)",       "shoulder_pitch": 0.0,  "elbow": 0.0},
  {"name": "At sides (0, 1.57)",   "shoulder_pitch": 0.0,  "elbow": 1.57},
  {"name": "Extended (-1.6, 1.57)","shoulder_pitch": -1.6, "elbow": 1.57},
]


class _JoystickViewer(_TermLoggingViewer):
  """NativeMujocoViewer with joystick HUD overlay and arm nudge event toggle."""

  def __init__(self, env, policy, *, js_state, cmd_term, nudge_event_indices, has_height=False, has_waist_yaw=False, weight_ctrl: _WeightController | None = None, **kwargs):
    super().__init__(env, policy, weight_ctrl=weight_ctrl, **kwargs)
    self._js = js_state
    self._cmd_term = cmd_term
    self._nudge_indices = nudge_event_indices
    self._nudge_was_on = False
    self._has_height = has_height
    self._has_waist_yaw = has_waist_yaw
    self._arm_mode_idx = 0
    self._prev_arm_mode_idx = 0
    # Resolve arm joint indices once.
    robot = cmd_term.robot
    self._arm_shoulder_ids = [
      i for i, n in enumerate(robot.joint_names) if "shoulder_pitch" in n
    ]
    self._arm_elbow_ids = [
      i for i, n in enumerate(robot.joint_names) if "elbow" in n and "wrist" not in n
    ]
    # Resolve waist_yaw joint index (None if absent from model).
    self._waist_yaw_id = next(
      (i for i, n in enumerate(robot.joint_names) if n == "waist_yaw_joint"),
      None,
    )
    # Disable nudge events initially (set timer to large value).
    if self._nudge_indices:
      em = self.env.unwrapped.event_manager
      for idx in self._nudge_indices:
        em._interval_term_time_left[idx][:] = 1e9

  def _apply_arm_mode(self):
    """Write arm PD targets for the active arm mode.

    Keep ``default_joint_pos`` untouched so ``joint_pos_rel`` observations
    preserve the same meaning as during training: actual arm displacement
    relative to the robot's keyframe default pose.
    """
    mode = _ARM_MODES[self._arm_mode_idx]
    robot = self._cmd_term.robot
    # Use D-pad accumulated shoulder_pitch instead of fixed mode value.
    sp = self._js.get("shoulder_pitch", mode["shoulder_pitch"])
    el = mode["elbow"]
    for idx in self._arm_shoulder_ids:
      robot.data.joint_pos_target[:, idx] = sp
    for idx in self._arm_elbow_ids:
      robot.data.joint_pos_target[:, idx] = el

  def _execute_step(self) -> bool:
    # Toggle nudge_arms events via event manager interval timer.
    if self._nudge_indices:
      nudge_on = self._js["nudge_arms"]
      if nudge_on != self._nudge_was_on:
        self._nudge_was_on = nudge_on
        em = self.env.unwrapped.event_manager
        for idx in self._nudge_indices:
          if nudge_on:
            em._interval_term_time_left[idx][:] = 0.0
          else:
            em._interval_term_time_left[idx][:] = 1e9
    # Cycle arm mode on button press.
    self._arm_mode_idx = self._js.get("arm_mode", 0)
    if self._arm_mode_idx != self._prev_arm_mode_idx:
      self._prev_arm_mode_idx = self._arm_mode_idx
      if not self._nudge_was_on:
        self._apply_arm_mode()
    # Re-apply arm mode every step so it persists after reset
    # (reset_arm_targets would overwrite with keyframe defaults).
    # Skip when nudge is active — nudge_joints_position controls arm targets.
    if not self._nudge_was_on:
      self._apply_arm_mode()
    return super()._execute_step()

  def sync_env_to_viewer(self):
    # Intercept parent's set_texts call to append joystick state, avoiding
    # a double set_texts (which causes flicker).
    v = self.viewer
    if not v or not v.is_running():
      super().sync_env_to_viewer()
      return
    s = self._js
    ct = self._cmd_term
    robot = ct.robot

    # Read velocity vectors for HUD.
    cmd = ct.vel_command_b[self.env_idx].cpu()
    vel = robot.data.root_link_lin_vel_b[self.env_idx].cpu()
    ang = robot.data.root_link_ang_vel_b[self.env_idx, 2].item()

    original_set_texts = v.set_texts

    # Actual base height for HUD.
    if self._has_height:
      actual_height = (
        robot.data.root_link_pos_w[self.env_idx, 2].item()
        - self.env.unwrapped.scene.env_origins[self.env_idx, 2].item()
      )
    # Actual waist_yaw joint position for HUD.
    if self._has_waist_yaw and self._waist_yaw_id is not None:
      actual_waist_yaw = robot.data.joint_pos[self.env_idx, self._waist_yaw_id].item()

    def _patched_set_texts(overlay):
      font, pos, text_1, text_2 = overlay
      arm_mode_name = _ARM_MODES[s.get("arm_mode", 0)]["name"]
      text_1 += (
        "\n \n[O] Velocity\n[X] Heading\n[S] Nudge\n[T] Arms\nShoulder"
        "\n \nCmd Vel\nCur Vel"
      )
      text_2 += (
        f"\n \n"
        f"{'ABSOLUTE' if s['absolute_velocity'] else 'RELATIVE'}\n"
        f"{'ON' if s['heading_align'] else 'OFF'}\n"
        f"{'ON' if s['nudge_arms'] else 'OFF'}\n"
        f"{arm_mode_name}\n"
        f"{s.get('shoulder_pitch', 0.0):.2f} rad"
        f"\n \n"
        f"({cmd[0]:.2f}, {cmd[1]:.2f}, {cmd[2]:.2f})\n"
        f"({vel[0]:.2f}, {vel[1]:.2f}, {ang:.2f})"
      )
      if self._has_height:
        text_1 += "\n \nCmd Height\nCur Height"
        text_2 += f"\n \n{cmd[3]:.2f}m\n{actual_height:.2f}m"
      if self._has_waist_yaw and self._waist_yaw_id is not None and cmd.shape[0] >= 5:
        text_1 += "\n \nCmd Waist\nCur Waist"
        text_2 += f"\n \n{cmd[4]:+.2f} rad\n{actual_waist_yaw:+.2f} rad"
      temp_lines = _motor_temp_lines(self.env, self.env_idx, top_k=4)
      if temp_lines:
        text_1 += "\n \nMotor T (hot)"
        text_2 += "\n \n "
        for label, t in temp_lines:
          mark = "!!" if t >= 90.0 else ("!" if t >= 70.0 else "")
          text_1 += f"\n{label}"
          text_2 += f"\n{t:.1f}C{mark}"
      original_set_texts((font, pos, text_1, text_2))

    v.set_texts = _patched_set_texts
    try:
      super().sync_env_to_viewer()
    finally:
      v.set_texts = original_set_texts


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False
  # If True, read velocity commands from a connected joystick device.
  js: bool = False
  # If True, open a virtual GUI joystick (no physical gamepad needed).
  vjs: bool = False


def _find_latest_checkpoint(log_root_path: Path) -> Path:
  """Return the highest-numbered ``model_*.pt`` in the most recent run dir."""
  if not log_root_path.exists():
    raise FileNotFoundError(
      f"No log directory found at {log_root_path}. Train a policy first or "
      f"pass --checkpoint-file."
    )
  run_dirs = sorted(
    (d for d in log_root_path.iterdir() if d.is_dir()), key=lambda d: d.name
  )
  for run_dir in reversed(run_dirs):
    ckpts = list(run_dir.glob("model_*.pt"))
    if ckpts:
      return max(ckpts, key=lambda p: int(p.stem.split("_")[1]))
  raise FileNotFoundError(
    f"No `model_*.pt` checkpoints found under any run in {log_root_path}."
  )


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")
  else:
    # Always drop the episode time-limit during play so the robot isn't
    # reset every episode_length_s seconds (keeps fell_over etc.).
    env_cfg.terminations.pop("time_out", None)

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      # No checkpoint specified: pick the latest checkpoint from the most
      # recent run directory under the experiment's log root.
      resume_path = _find_latest_checkpoint(log_root_path)
      print(
        f"[INFO]: No checkpoint specified, using latest: {resume_path} "
        f"(run: {resume_path.parent.name})"
      )
    log_dir = resume_path.parent

  # Reconcile observations with checkpoint's saved config.
  if resume_path is not None:
    _reconcile_obs_with_checkpoint(env_cfg, resume_path)

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  # For the balance_weight task, disable randomization events in play mode
  # so the user can set masses manually without reset-time randomization
  # overwriting them (the _WeightController also re-asserts every step,
  # but removing the events avoids an expensive recompute on every reset).
  if task_id == "Unitree-G1-Flat-Balance-Weight":
    for _ev in ("randomize_hand_weights", "randomize_back_weight"):
      env_cfg.events.pop(_ev, None)
    # Also drop the mass curricula — they adjust event params that no
    # longer exist.
    for _cur in ("hand_weight_range", "back_weight_range"):
      env_cfg.curriculum.pop(_cur, None)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  # If joystick control requested, start a background thread that writes
  # joystick values into the velocity command term (overrides random sampling).
  _js_viewer_kwargs = None  # Set below if joystick initializes successfully.
  if cfg.js or cfg.vjs:
    try:
      import math, threading, time

      if cfg.vjs:
        # Virtual GUI joystick (no physical gamepad needed).
        try:
          from virtual_joystick import VirtualJoystickReader
        except Exception:
          from scripts.virtual_joystick import VirtualJoystickReader
        reader = VirtualJoystickReader()
      else:
        # When running `python scripts/play.py`, the `scripts/` dir is
        # usually on `sys.path` as the script directory, so import the
        # local module as `joystick`. Fall back to `scripts.joystick` if
        # that fails (e.g., running from package context).
        try:
          from joystick import JoystickReader
        except Exception:
          from scripts.joystick import JoystickReader
        reader = JoystickReader()
      cmd_term = env.unwrapped.command_manager.get_term("twist")
      robot = cmd_term.robot

      # P-gain for heading alignment (ang_vel_z = gain * heading_error).
      HEADING_ALIGN_GAIN = 2.0

      # PS4 buttons (pygame): 0=Cross, 1=Circle, 2=Triangle, 3=Square
      BTN_HEADING_ALIGN = 0   # Cross (X)
      BTN_ABS_VELOCITY = 1    # Circle
      BTN_ARM_MODE = 2        # Triangle — cycle arm pose mode
      BTN_NUDGE_ARMS = 3      # Square — toggle arm nudge

      # Resolve shoulder_pitch joint limits from the physics model.
      SHOULDER_PITCH_MIN = -3.0892
      SHOULDER_PITCH_MAX = 2.6704
      SHOULDER_SPEED = 1.5  # rad/s while D-pad is held

      # Shared joystick state (read by _JoystickViewer for HUD + nudge).
      js_state = {
        "absolute_velocity": True,
        "heading_align": False,
        "nudge_arms": False,
        "arm_mode": 0,  # index into _ARM_MODES
        "shoulder_pitch": 0.0,  # accumulated via D-pad up/down
        "waist_yaw": 0.0,  # accumulated via D-pad left/right (teleop tasks only)
      }
      _prev_arm_btn = False

      # Find nudge_arms* event indices in the event manager's interval terms.
      nudge_event_indices = []
      try:
        em = env.unwrapped.event_manager
        interval_names = em.active_terms.get("interval", [])
        for nudge_name in ("nudge_arms", "nudge_arms_position"):
          if nudge_name in interval_names:
            idx = interval_names.index(nudge_name)
            nudge_event_indices.append(idx)
            print(f"[Joystick] {nudge_name} event found (index {idx})")
        if not nudge_event_indices:
          print("[Joystick] No nudge_arms events registered in config")
      except Exception as e:
        print(f"[Joystick] Could not find nudge_arms events: {e}")

      # Detect height-aware command (4D command vector) and waist-yaw-aware
      # command (5D). The 5D variant is a subclass of the 4D one, so
      # has_height is True for both — order checks accordingly.
      from src.tasks.velocity.mdp.velocity_command import (
        UniformVelocityHeightCommandCfg,
        UniformVelocityHeightWaistCommandCfg,
      )
      has_height = isinstance(cmd_term.cfg, UniformVelocityHeightCommandCfg)
      has_waist_yaw = isinstance(cmd_term.cfg, UniformVelocityHeightWaistCommandCfg)
      if has_height:
        height_min, height_max = cmd_term.cfg.ranges.base_height
        height_default = cmd_term.cfg.default_height
        # ry=0 → default_height, ry=-1 → height_min, ry=+1 → height_max
        height_range_down = height_default - height_min  # how far stick-down can go
        height_range_up = height_max - height_default    # how far stick-up can go
      if has_waist_yaw:
        waist_yaw_min, waist_yaw_max = cmd_term.cfg.ranges.waist_yaw
        # Slightly widen the play range so the operator can reach a fuller
        # waist rotation than what training sampled. PD limits still apply.
        waist_yaw_min = min(waist_yaw_min, -1.4)
        waist_yaw_max = max(waist_yaw_max, 1.4)
        WAIST_YAW_SPEED = 1.5  # rad/s while D-pad held

      print("[Joystick] Controls:")
      print("  Left stick     : move (lin_vel_x / lin_vel_y)")
      print("  Right stick X  : rotate (ang_vel_z)")
      if has_height:
        print(f"  Right stick Y  : height ({height_min:.2f}–{height_max:.2f}m)")
      print("  D-pad up/down  : shoulder pitch (hold to move)")
      if has_waist_yaw:
        print(f"  D-pad left/right : waist yaw ({waist_yaw_min:.2f}–{waist_yaw_max:.2f} rad)")
      print("  Circle         : toggle absolute/relative velocity")
      print("  Cross (X)      : toggle heading alignment (absolute mode only)")
      print("  Square         : toggle arm nudge")
      print("  Triangle       : cycle arm pose mode")

      def _joystick_loop():
        while True:
          try:
            lx, ly, rz, ry = reader.get_values()
          except Exception:
            time.sleep(0.05)
            continue

          # Check button toggles.
          for btn, key, label in [
            (BTN_ABS_VELOCITY, "absolute_velocity", "Absolute velocity"),
            (BTN_HEADING_ALIGN, "heading_align", "Heading alignment"),
            (BTN_NUDGE_ARMS, "nudge_arms", "Arm nudge"),
          ]:
            new = reader.get_button_toggle(btn)
            if new != js_state[key]:
              js_state[key] = new
              print(f"[Joystick] {label}: {'ON' if new else 'OFF'}")

          # Cycle arm mode on Triangle press (edge-detect via toggle).
          nonlocal _prev_arm_btn
          arm_toggle = reader.get_button_toggle(BTN_ARM_MODE)
          if arm_toggle != _prev_arm_btn:
            _prev_arm_btn = arm_toggle
            js_state["arm_mode"] = (js_state["arm_mode"] + 1) % len(_ARM_MODES)
            mode = _ARM_MODES[js_state["arm_mode"]]
            print(f"[Joystick] Arm mode: {mode['name']}")

          try:
            if js_state["absolute_velocity"]:
              # Rotate world-frame joystick input into robot body frame.
              yaw = robot.data.heading_w  # (num_envs,)
              cos_yaw = torch.cos(yaw)
              sin_yaw = torch.sin(yaw)
              cmd_term.vel_command_b[:, 0] = cos_yaw * lx + sin_yaw * ly
              cmd_term.vel_command_b[:, 1] = -sin_yaw * lx + cos_yaw * ly

              # Heading alignment: auto-rotate to face velocity direction.
              if js_state["heading_align"] and (abs(lx) > 0 or abs(ly) > 0):
                desired_yaw = math.atan2(ly, lx)
                heading_error = desired_yaw - yaw
                heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
                cmd_term.vel_command_b[:, 2] = torch.clamp(
                  HEADING_ALIGN_GAIN * heading_error, -1.0, 1.0
                )
              else:
                cmd_term.vel_command_b[:, 2] = rz
            else:
              cmd_term.vel_command_b[:, 0] = lx
              cmd_term.vel_command_b[:, 1] = ly
              cmd_term.vel_command_b[:, 2] = rz

            # Map right stick Y to target height for height-aware tasks.
            if has_height:
              if ry < 0:
                cmd_term.vel_command_b[:, 3] = height_default + ry * height_range_down
              else:
                cmd_term.vel_command_b[:, 3] = height_default + ry * height_range_up

            cmd_term.is_standing_env[:] = False
            cmd_term.is_heading_env[:] = False
          except Exception:
            pass

          # D-pad up/down → accumulate shoulder_pitch.
          dpad_y = reader.get_dpad_y()
          if dpad_y != 0:
            sp = js_state["shoulder_pitch"] + dpad_y * SHOULDER_SPEED * 0.02
            js_state["shoulder_pitch"] = max(SHOULDER_PITCH_MIN, min(SHOULDER_PITCH_MAX, sp))

          # D-pad left/right → accumulate waist_yaw target (5D command only).
          if has_waist_yaw:
            try:
              dpad_x = reader.get_dpad_x()
            except AttributeError:
              dpad_x = 0.0
            if dpad_x != 0:
              wy = js_state["waist_yaw"] + dpad_x * WAIST_YAW_SPEED * 0.02
              js_state["waist_yaw"] = max(waist_yaw_min, min(waist_yaw_max, wy))
            try:
              cmd_term.vel_command_b[:, 4] = js_state["waist_yaw"]
            except Exception:
              pass

          time.sleep(0.02)

      t = threading.Thread(target=_joystick_loop, daemon=True)
      t.start()
      _js_viewer_kwargs = dict(
        js_state=js_state, cmd_term=cmd_term, nudge_event_indices=nudge_event_indices,
        has_height=has_height, has_waist_yaw=has_waist_yaw,
      )
    except Exception as e:
      print(f"[WARN] Joystick requested but failed to start: {e}")
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  # For the balance_weight task: instantiate a keyboard-driven weight
  # controller. It writes user-set masses into sim.model.body_mass every
  # step so the robot's payload is whatever the user typed.
  weight_ctrl: _WeightController | None = None
  if task_id == "Unitree-G1-Flat-Balance-Weight":
    try:
      weight_ctrl = _WeightController(env)
    except Exception as e:
      print(f"[WARN] Failed to set up weight controller: {e}")
      weight_ctrl = None

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if weight_ctrl is not None and resolved_viewer == "native":
    weight_ctrl.start()

  try:
    if resolved_viewer == "native":
      if _js_viewer_kwargs is not None:
        viewer = _JoystickViewer(env, policy, weight_ctrl=weight_ctrl, **_js_viewer_kwargs)
      else:
        viewer = _TermLoggingViewer(env, policy, weight_ctrl=weight_ctrl)
      viewer.run()
    elif resolved_viewer == "viser":
      ViserPlayViewer(env, policy).run()
    else:
      raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
  except KeyboardInterrupt:
    print("\n[INFO] Interrupted, exiting...")
  finally:
    if weight_ctrl is not None:
      weight_ctrl.stop()
    try:
      env.close()
    except Exception:
      pass
    os._exit(0)


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
