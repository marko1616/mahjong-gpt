import torch
import datetime
from dataclasses import dataclass, field
from schedulers import Scheduler, SchedulerConfig, ConstantScheduler
from env.constants import PAD_ID


@dataclass
class TrainingConfig:
    """Training hyperparameters"""

    replay_buffer_size: int = 1600
    epoches_per_update: int = 2
    max_update_kl: float = 1
    clip_epsilon: float = 0.4
    weight_policy: float = 1
    weight_value: float = 1
    sample_trail_count: int = 1600
    batch_size: int = 8
    grad_accum_steps: int = 32
    episodes: int = 100
    beta: float = 0.2
    lr_value: float = 1e-7
    lr_policy: float = 1e-7

    action_epsilon_config: SchedulerConfig = field(
        default_factory=lambda: SchedulerConfig(
            scheduler_type="linear", init=0.1, minimum=0.0
        )
    )

    _action_epsilon_scheduler: Scheduler | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def action_epsilon_scheduler(self) -> Scheduler:
        if self._action_epsilon_scheduler is None:
            self._action_epsilon_scheduler = self.action_epsilon_config.build(
                self.episodes
            )
        return self._action_epsilon_scheduler

    def reset_scheduler(self, episodes: int | None = None) -> None:
        """Reset the scheduler with optional new episode count"""
        ep = episodes if episodes is not None else self.episodes
        self._action_epsilon_scheduler = self.action_epsilon_config.build(ep)


@dataclass
class EnvConfig:
    """Environment hyperparameters"""

    stable_seed_steps: int = 4
    init_env_seed: int = 31887
    env_num: int = 10


@dataclass
class TargetConfig:
    """Target hyperparameters"""

    target: str = "N_step_TD"
    lambd: float = 0.1
    gamma: float = 0.9

    alpha_config: SchedulerConfig = field(
        default_factory=lambda: SchedulerConfig(
            scheduler_type="linear", init=0.1, minimum=0.03
        )
    )

    n_td_config: SchedulerConfig = field(
        default_factory=lambda: SchedulerConfig(
            scheduler_type="linear", init=2.0, minimum=1.0
        )
    )

    _alpha_scheduler: Scheduler | None = field(default=None, repr=False, compare=False)
    _n_td_scheduler: Scheduler | None = field(default=None, repr=False, compare=False)

    @property
    def alpha_scheduler(self) -> Scheduler:
        if self._alpha_scheduler is None:
            self._alpha_scheduler = self.alpha_config.build()
        return self._alpha_scheduler

    @property
    def n_td_scheduler(self) -> Scheduler:
        if self._n_td_scheduler is None:
            self._n_td_scheduler = self.n_td_config.build()
        return self._n_td_scheduler

    def reset_schedulers(self, episodes: int) -> None:
        """Reset all schedulers with new episode count"""
        self._alpha_scheduler = self.alpha_config.build(episodes)
        self._n_td_scheduler = self.n_td_config.build(episodes)


@dataclass
class ModelConfig:
    """Model hyperparameters"""

    layer_policy: int = 6
    layer_value: int = 6
    dim_feedforward: int = 1024
    enable_compile: bool = False
    pad_token_id: int = PAD_ID
    max_seq_len: int = 512
    vocab_size: int = 46 + 34 + 4 + 1 + 1 # action, hand, player, [SEP], [PAD]
    max_norm: float = 0.5
    d_model: int = 1024
    dropout: float = 0.1
    nhead: int = 8


@dataclass
class SystemConfig:
    """Compute, save, and display parameters"""

    verbose_positive_done_reward: bool = True
    replay_buffer_file: str = "replay.json"
    log_dir: str = field(
        default_factory=lambda: f"runs/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    device: str = "cuda:0"
    dtype: torch.dtype = torch.float32
    amp_enable: bool = False
    amp_device_type: str = "cuda"
    amp_dtype: torch.dtype = torch.float16
    path_max: str = "./max"
    num_workers: int = 4


@dataclass
class EvalConfig:
    """Evaluation mode configuration"""

    eval_mode: bool = False
    path: str = "./mahjong"
    memget_num_per_update: int = 1600
    action_temperature: float = 0.01
    n_display: int = 16

    ci_enabled: bool = True
    ci_confidence: float = 0.95


@dataclass
class Config:
    """Main configuration combining all sub-configs"""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    evalu: EvalConfig = field(default_factory=EvalConfig)
    verbose_first: bool = True

    def __post_init__(self) -> None:
        self._sync_episodes()
        self._apply_eval_mode()

    def _sync_episodes(self) -> None:
        """Sync episode count across all schedulers"""
        episodes = self.training.episodes
        self.training.reset_scheduler(episodes)
        self.target.reset_schedulers(episodes)

    def _apply_eval_mode(self) -> None:
        """Apply eval mode settings if enabled"""
        if self.evalu.eval_mode:
            self.evalu.path = self.system.path_max
            self.evalu.memget_num_per_update = 1000
            self.evalu.action_temperature = 0.1
            self.evalu.n_display = 50
            self.training._action_epsilon_scheduler = ConstantScheduler(0.0)


def get_default_config() -> Config:
    """Get default training configuration"""
    return Config()


def get_eval_config() -> Config:
    """Get evaluation mode configuration"""
    config = Config(eval=EvalConfig(eval_mode=True))
    return config


def get_custom_config(
    episodes: int = 175,
    lr: float = 0.6e-6,
    gamma: float = 0.10,
    batch_size: int = 100,
    eval_mode: bool = False,
    device: str = "cuda:0",
    action_epsilon_scheduler: SchedulerConfig | None = None,
    alpha_scheduler: SchedulerConfig | None = None,
    n_td_scheduler: SchedulerConfig | None = None,
) -> Config:
    """Get a customized configuration"""
    training = TrainingConfig(
        episodes=episodes,
        lr_value=lr,
        lr_policy=lr,
        batch_size=batch_size,
    )
    if action_epsilon_scheduler is not None:
        training.action_epsilon_config = action_epsilon_scheduler

    target = TargetConfig(gamma=gamma)
    if alpha_scheduler is not None:
        target.alpha_config = alpha_scheduler
    if n_td_scheduler is not None:
        target.n_td_config = n_td_scheduler

    return Config(
        training=training,
        target=target,
        system=SystemConfig(device=device),
        eval=EvalConfig(eval_mode=eval_mode),
    )
