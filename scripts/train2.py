"""Script to train RL agent with RSL-RL — with native MuJoCo viewer."""

import logging
import os
import signal
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import mujoco
import mujoco.viewer
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


# Visual DR fields that need syncing from GPU model to CPU MjModel.
_VISUAL_FIELDS = frozenset({
    "geom_rgba", "geom_size", "geom_pos", "geom_quat",
    "mat_rgba", "site_pos", "site_quat",
    "body_pos", "body_quat", "body_ipos", "body_inertia", "body_iquat", "body_mass",
    "cam_pos", "cam_quat", "cam_fovy", "cam_intrinsic",
    "light_pos", "light_dir",
})


class NativeTrainViewer:
  """Wrapper that displays a live native MuJoCo viewer during training.

  Unlike play's NativeMujocoViewer which owns the step loop, this wrapper
  is passive: the runner controls stepping, and we just sync & render after
  each env.step().

  Ghost environments are rendered as semi-transparent copies using
  mjv_addGeoms, same technique as the play viewer.
  """

  def __init__(self, env, interval: int = 4, env_idx: int = 0, num_ghosts: int = 15):
    self.env = env
    self.interval = interval
    self.env_idx = env_idx
    self.num_ghosts = num_ghosts
    self._step_count = 0
    self._viewer_handle = None
    self._mjm = None
    self._mjd = None
    self._vd = None  # Secondary MjData for ghost env rendering.
    self._setup_viewer()

  def _setup_viewer(self):
    """Launch the passive MuJoCo viewer window."""
    sim = self.env.unwrapped.sim
    self._mjm = sim.mj_model
    self._mjd = sim.mj_data
    assert self._mjm is not None and self._mjd is not None

    num_envs = self.env.unwrapped.num_envs
    if num_envs > 1:
      self._vd = mujoco.MjData(self._mjm)

    self._viewer_handle = mujoco.viewer.launch_passive(
      self._mjm,
      self._mjd,
      show_left_ui=False,
      show_right_ui=False,
    )

    # Pre-compute ghost env indices and reusable rendering objects.
    self._ghost_indices = self._compute_ghost_indices()
    self._ghost_vopt = mujoco.MjvOption()
    self._ghost_pert = mujoco.MjvPerturb()

    ghost_str = f", ghosts={len(self._ghost_indices)}" if self._ghost_indices else ""
    print(f"[INFO] Native MuJoCo viewer opened (env_idx={self.env_idx}, interval={self.interval}{ghost_str})")

  def _sync_env_state_to_mjdata(self, target_data, sim_data, env_idx):
    """Copy one environment's state from GPU tensors to a CPU MjData."""
    if self._mjm.nq > 0:
      target_data.qpos[:] = sim_data.qpos[env_idx].cpu().numpy()
      target_data.qvel[:] = sim_data.qvel[env_idx].cpu().numpy()
    if self._mjm.nmocap > 0:
      target_data.mocap_pos[:] = sim_data.mocap_pos[env_idx].cpu().numpy()
      target_data.mocap_quat[:] = sim_data.mocap_quat[env_idx].cpu().numpy()
    target_data.xfrc_applied[:] = sim_data.xfrc_applied[env_idx].cpu().numpy()

  def _sync_model_fields(self, sim, env_idx):
    """Sync visual DR fields from GPU model to CPU MjModel for one env."""
    for field_name in sim.expanded_fields & _VISUAL_FIELDS:
      src = getattr(sim.model, field_name)[env_idx].cpu().numpy()
      dst = getattr(self._mjm, field_name)
      dst[:] = src.reshape(dst.shape)

  def _compute_ghost_indices(self):
    """Pre-compute which env indices to render as ghosts."""
    num_envs = self.env.unwrapped.num_envs
    if num_envs <= 1 or self.num_ghosts <= 0:
      return []
    if num_envs - 1 <= self.num_ghosts:
      return [i for i in range(num_envs) if i != self.env_idx]
    step = max(1, num_envs // self.num_ghosts)
    indices = []
    for i in range(0, num_envs, step):
      if i != self.env_idx:
        indices.append(i)
      if len(indices) >= self.num_ghosts:
        break
    return indices

  def _render_ghost_envs(self, v, sim, sim_data):
    """Render a subset of other environments as ghost geoms."""
    if self._vd is None or not self._ghost_indices:
      return

    for i in self._ghost_indices:
      self._sync_env_state_to_mjdata(self._vd, sim_data, i)
      self._sync_model_fields(sim, i)
      mujoco.mj_forward(self._mjm, self._vd)
      mujoco.mjv_addGeoms(self._mjm, self._vd, self._ghost_vopt, self._ghost_pert, mujoco.mjtCatBit.mjCAT_ALL, v.user_scn)

    # Restore primary env's model fields so the main robot looks correct.
    self._sync_model_fields(sim, self.env_idx)

  def _sync_and_render(self):
    """Copy GPU sim state to CPU MjData, add ghost envs, and render."""
    v = self._viewer_handle
    if v is None or not v.is_running():
      return

    sim = self.env.unwrapped.sim
    sim_data = sim.data

    with v.lock():
      # Sync primary environment.
      self._sync_env_state_to_mjdata(self._mjd, sim_data, self.env_idx)
      self._sync_model_fields(sim, self.env_idx)
      mujoco.mj_forward(self._mjm, self._mjd)

      # Clear previous ghost geoms, then add fresh ones.
      v.user_scn.ngeom = 0
      self._render_ghost_envs(v, sim, sim_data)

    # Push to viewer (outside lock).
    has_visual_dr = bool(sim.expanded_fields & _VISUAL_FIELDS)
    v.sync(state_only=not has_visual_dr)

  def __getattr__(self, name: str):
    return getattr(self.env, name)

  def step(self, actions):
    result = self.env.step(actions)
    self._step_count += 1
    if self._step_count % self.interval == 0:
      self._sync_and_render()
    return result

  def reset(self, *args, **kwargs):
    return self.env.reset(*args, **kwargs)

  def close(self):
    if self._viewer_handle is not None:
      try:
        self._viewer_handle.close()
      except Exception:
        pass
      self._viewer_handle = None
    self.env.close()


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  motion_file: str | None = None
  checkpoint: str | None = None
  """Path to a .pt checkpoint file to resume from. If not set, resumes from the latest checkpoint when --agent.resume is true."""
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  render: bool = True
  render_interval: int = 4
  render_ghosts: int = 15
  """Number of other environments to show as ghost copies in the viewer."""
  enable_nan_guard: bool = False
  torchrunx_log_dir: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task:
    if not cfg.motion_file:
      raise ValueError("For tracking tasks, --motion-file must be set ...")
    motion_path = Path(cfg.motion_file).expanduser().resolve()
    if not motion_path.exists():
      raise FileNotFoundError(f"Motion file not found: {motion_path}")
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = str(motion_path)
    print(f"[INFO] Using motion file: {motion_cmd.motion_file}")

    if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
      print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  # Native viewer needs render_mode=None; video needs "rgb_array".
  render_mode = "rgb_array" if cfg.video else None
  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode=render_mode
  )

  log_root_path = log_dir.parent

  resume_path: Path | None = None
  if cfg.checkpoint is not None:
    resume_path = Path(cfg.checkpoint).expanduser().resolve()
    if not resume_path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
    print(f"[INFO] Resuming from checkpoint: {resume_path}")
  elif cfg.agent.resume:
    print(f"[INFO] Resuming training from latest checkpoint for experiment: {cfg.agent.experiment_name}")
    resume_path = get_checkpoint_path(
      log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
    )
    print(f"[INFO] Found checkpoint to resume from: {resume_path}")

  # Set up native MuJoCo viewer on rank 0.
  if cfg.render and rank == 0:
    env = NativeTrainViewer(env, interval=cfg.render_interval, num_ghosts=cfg.render_ghosts)

  # Only record videos on rank 0.
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  runner_kwargs = {}
  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  # Only write config files from rank 0.
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  if resume_path is not None:
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)

  # Create log directory once before launching workers.
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))

  # Use GLX for native viewer (not EGL which is headless).
  if args.render:
    os.environ["MUJOCO_GL"] = "glx"
  else:
    os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    run_train(task_id, args, log_dir)
  else:
    import torchrunx

    logging.basicConfig(level=logging.INFO)

    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


_TYRO_FLAGS = (
  tyro.conf.AvoidSubcommands,
  tyro.conf.UsePythonSyntaxForLiteralCollections,
)


def main():
  # Parse first argument to choose the task.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=_TYRO_FLAGS,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=_TYRO_FLAGS,
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
