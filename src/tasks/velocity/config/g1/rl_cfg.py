"""RL configuration for Unitree G1 velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_balance_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 balance/teleoperation task."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_balance_velocity"
  cfg.logger = "tensorboard"
  return cfg


def unitree_g1_balance_height_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 balance + height control task."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_balance_height_velocity"
  cfg.actor.hidden_dims = (512, 512, 256)
  cfg.critic.hidden_dims = (512, 512, 256)
  cfg.max_iterations = 25001
  cfg.logger = "tensorboard"
  return cfg


def unitree_g1_balance_height_waist_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for G1 balance + height + waist_yaw teleop task.

  Uses a larger network than the standing config (3x512 layers vs the
  default 512→256→128) because the policy has more obs (5D command, history)
  and must learn standing + 6 walking directions + thermal awareness.
  ~25k iterations matches the multi-phase velocity curriculum (forward →
  reverse → side → rotation → combined) defined in the env config.
  """
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_balance_height_waist_velocity"
  cfg.actor.hidden_dims = (512, 512, 512)
  cfg.critic.hidden_dims = (512, 512, 512)
  cfg.max_iterations = 25001
  cfg.logger = "tensorboard"
  return cfg


def unitree_g1_balance_standing_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 standing-only balance task."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_balance_standing"
  cfg.logger = "tensorboard"
  return cfg


def unitree_g1_balance_weight_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 balance with payload weights."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_balance_weight_velocity"
  cfg.actor.hidden_dims = (512, 512, 256)
  cfg.critic.hidden_dims = (512, 512, 256)
  cfg.max_iterations = 25001
  cfg.logger = "tensorboard"
  return cfg


def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_velocity",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )
