"""Script to play RL agent with RSL-RL."""

import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
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


def _reconcile_obs_with_checkpoint(env_cfg, checkpoint_path: Path):
  """Adjust env_cfg observations to match a checkpoint's saved config.

  Looks for params/env.yaml next to the checkpoint. If found, removes
  observation terms that weren't present during training so the model's
  expected input dimensions match.
  """
  env_yaml = checkpoint_path.parent / "params" / "env.yaml"
  if not env_yaml.exists():
    return
  saved_actor_terms = _parse_obs_terms_from_yaml(env_yaml, "actor")
  if saved_actor_terms is None:
    return
  current_actor_terms = list(env_cfg.observations["actor"].terms.keys())
  if saved_actor_terms == current_actor_terms:
    return
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


class _TermLoggingViewer(NativeMujocoViewer):
  """NativeMujocoViewer that prints termination reasons to console."""

  def _execute_step(self) -> bool:
    result = super()._execute_step()
    if result:
      _log_terminations(self.env)
    return result


class _JoystickViewer(_TermLoggingViewer):
  """NativeMujocoViewer with joystick HUD overlay and arm nudge event toggle."""

  def __init__(self, env, policy, *, js_state, cmd_term, nudge_event_idx, **kwargs):
    super().__init__(env, policy, **kwargs)
    self._js = js_state
    self._cmd_term = cmd_term
    self._nudge_idx = nudge_event_idx
    self._nudge_was_on = False
    # Disable nudge event initially (set timer to large value).
    if self._nudge_idx is not None:
      em = self.env.unwrapped.event_manager
      em._interval_term_time_left[self._nudge_idx][:] = 1e9

  def _execute_step(self) -> bool:
    # Toggle nudge_arms event via event manager interval timer.
    if self._nudge_idx is not None:
      nudge_on = self._js["nudge_arms"]
      if nudge_on != self._nudge_was_on:
        self._nudge_was_on = nudge_on
        em = self.env.unwrapped.event_manager
        if nudge_on:
          em._interval_term_time_left[self._nudge_idx][:] = 0.0
        else:
          em._interval_term_time_left[self._nudge_idx][:] = 1e9
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

    def _patched_set_texts(overlay):
      font, pos, text_1, text_2 = overlay
      text_1 += (
        "\n \nVelocity\nHeading\nNudge"
        "\n \nCmd Vel\nCur Vel"
      )
      text_2 += (
        f"\n \n"
        f"{'ABSOLUTE' if s['absolute_velocity'] else 'RELATIVE'}\n"
        f"{'ON' if s['heading_align'] else 'OFF'}\n"
        f"{'ON' if s['nudge_arms'] else 'OFF'}"
        f"\n \n"
        f"({cmd[0]:.2f}, {cmd[1]:.2f}, {cmd[2]:.2f})\n"
        f"({vel[0]:.2f}, {vel[1]:.2f}, {ang:.2f})"
      )
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
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
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
  if cfg.js:
    try:
      # When running `python scripts/play.py`, the `scripts/` dir is
      # usually on `sys.path` as the script directory, so import the
      # local module as `joystick`. Fall back to `scripts.joystick` if
      # that fails (e.g., running from package context).
      try:
        from joystick import JoystickReader
      except Exception:
        from scripts.joystick import JoystickReader
      import math, threading, time

      reader = JoystickReader()
      cmd_term = env.unwrapped.command_manager.get_term("twist")
      robot = cmd_term.robot

      # P-gain for heading alignment (ang_vel_z = gain * heading_error).
      HEADING_ALIGN_GAIN = 2.0

      # PS4 buttons: 0=Cross(X), 1=Circle, 2=Square, 3=Triangle
      BTN_HEADING_ALIGN = 0   # Cross (X)
      BTN_ABS_VELOCITY = 1    # Circle
      BTN_NUDGE_ARMS = 2      # Square

      # Shared joystick state (read by _JoystickViewer for HUD + nudge).
      js_state = {
        "absolute_velocity": True,
        "heading_align": False,
        "nudge_arms": False,
      }

      # Find nudge_arms event index in the event manager's interval terms.
      nudge_event_idx = None
      try:
        em = env.unwrapped.event_manager
        interval_names = em.active_terms.get("interval", [])
        if "nudge_arms" in interval_names:
          nudge_event_idx = interval_names.index("nudge_arms")
          print(f"[Joystick] nudge_arms event found (index {nudge_event_idx})")
        else:
          print("[Joystick] nudge_arms event not registered in config")
      except Exception as e:
        print(f"[Joystick] Could not find nudge_arms event: {e}")

      print("[Joystick] Controls:")
      print("  Left stick     : move (lin_vel_x / lin_vel_y)")
      print("  Right stick X  : rotate (ang_vel_z)")
      print("  Circle         : toggle absolute/relative velocity")
      print("  Cross (X)      : toggle heading alignment (absolute mode only)")
      print("  Square         : toggle arm nudge")

      def _joystick_loop():
        while True:
          try:
            lx, ly, rz = reader.get_values()
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

            cmd_term.is_standing_env[:] = False
            cmd_term.is_heading_env[:] = False
          except Exception:
            pass
          time.sleep(0.02)

      t = threading.Thread(target=_joystick_loop, daemon=True)
      t.start()
      _js_viewer_kwargs = dict(
        js_state=js_state, cmd_term=cmd_term, nudge_event_idx=nudge_event_idx,
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

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    if _js_viewer_kwargs is not None:
      viewer = _JoystickViewer(env, policy, **_js_viewer_kwargs)
    else:
      viewer = _TermLoggingViewer(env, policy)
    viewer.run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


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
