import json
from typing import Any, Dict, Optional

import torch
from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    PrivateAttr,
    field_serializer,
    field_validator,
)

from .schedulers import Scheduler, SchedulerConfig, ConstantScheduler
from .env.constants import PAD_ID


class RewardConfig(BaseModel):
    """
    All rewards/penalties for the environment.
    """

    reward_weight_shanten: float = Field(default=30.0)
    penalty_ava_num: float = Field(default=1.2)
    score_weight: float = Field(default=0.1)
    reward_riichi: float = Field(default=10.0)
    reward_open_tanyao: float = Field(default=-5.0)


class TrainingConfig(BaseModel):
    """Training hyperparameters"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    replay_buffer_size: int = 1600
    epochs_per_update: int = 2
    max_update_kl: float = 1
    clip_epsilon: float = 0.4
    weight_policy: float = 1
    weight_value: float = 1
    sample_trail_count: int = 1600
    batch_size: int = 4
    grad_accum_steps: int = 32

    # episodes split for resumability
    pass_n_episodes: int = Field(default=100)

    beta: float = 0.1
    lr_value: float = 1e-7
    lr_policy: float = 1e-7

    action_epsilon_config: SchedulerConfig = Field(
        default_factory=lambda: SchedulerConfig(
            scheduler_type="linear", init=0.1, minimum=0.01
        )
    )

    # JSON-serializable scheduler state (suffix `_state` for explicit serialization fields)
    action_epsilon_state: Optional[Dict[str, Any]] = None

    # runtime-only scheduler instance
    _action_epsilon_scheduler: Optional[Scheduler] = PrivateAttr(default=None)

    @property
    def action_epsilon_scheduler(self) -> Scheduler:
        if self._action_epsilon_scheduler is None:
            self._action_epsilon_scheduler = self.action_epsilon_config.build(
                episodes_override=self.pass_n_episodes,
                state=self.action_epsilon_state,
            )
        return self._action_epsilon_scheduler

    def reset_scheduler(self, episodes: Optional[int] = None) -> None:
        """Reset the scheduler with optional new episode count"""
        ep = int(episodes) if episodes is not None else self.pass_n_episodes
        self._action_epsilon_scheduler = self.action_epsilon_config.build(ep)
        # reset serialized state too
        self.action_epsilon_state = self._action_epsilon_scheduler.to_state()

    @field_serializer("action_epsilon_state")
    def _ser_action_epsilon_state(
        self, v: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        # Always prefer the live scheduler if it exists so JSON includes latest t/value.
        if self._action_epsilon_scheduler is not None:
            return self._action_epsilon_scheduler.to_state()
        return v


class EnvConfig(BaseModel):
    """Environment hyperparameters"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    stable_seed_steps: int = 4
    init_env_seed: int = 31887
    env_num: int = 10


class TargetConfig(BaseModel):
    """Target hyperparameters"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    target: str = "N_step_TD"
    lambd: float = 0.50
    gamma: float = 0.95

    alpha_config: SchedulerConfig = Field(
        default_factory=lambda: SchedulerConfig(
            scheduler_type="linear", init=0.05, minimum=0.01
        )
    )
    n_td_config: SchedulerConfig = Field(
        default_factory=lambda: SchedulerConfig(
            scheduler_type="constant",
            init=1.0,
        )
    )

    alpha_state: Optional[Dict[str, Any]] = None
    n_td_state: Optional[Dict[str, Any]] = None

    _alpha_scheduler: Optional[Scheduler] = PrivateAttr(default=None)
    _n_td_scheduler: Optional[Scheduler] = PrivateAttr(default=None)

    @property
    def alpha_scheduler(self) -> Scheduler:
        if self._alpha_scheduler is None:
            # Episodes override is supplied by Config._sync_episodes() via reset_schedulers().
            self._alpha_scheduler = self.alpha_config.build(state=self.alpha_state)
        return self._alpha_scheduler

    @property
    def n_td_scheduler(self) -> Scheduler:
        if self._n_td_scheduler is None:
            self._n_td_scheduler = self.n_td_config.build(state=self.n_td_state)
        return self._n_td_scheduler

    def reset_schedulers(self, episodes: int) -> None:
        """Reset all schedulers with new episode count"""
        self._alpha_scheduler = self.alpha_config.build(episodes_override=episodes)
        self._n_td_scheduler = self.n_td_config.build(episodes_override=episodes)
        self.alpha_state = self._alpha_scheduler.to_state()
        self.n_td_state = self._n_td_scheduler.to_state()

    @field_serializer("alpha_state")
    def _ser_alpha_state(self, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if self._alpha_scheduler is not None:
            return self._alpha_scheduler.to_state()
        return v

    @field_serializer("n_td_state")
    def _ser_n_td_state(self, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if self._n_td_scheduler is not None:
            return self._n_td_scheduler.to_state()
        return v


class ModelConfig(BaseModel):
    """Model hyperparameters"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    layer_policy: int = 6
    layer_value: int = 6
    dim_feedforward: int = 1024
    pad_token_id: int = PAD_ID
    max_seq_len: int = 512
    vocab_size: int = 46 + 34 + 4 + 1 + 1  # action, hand, player, [SEP], [PAD]
    max_norm: float = 0.5
    d_model: int = 1024
    dropout: float = 0.1
    nhead: int = 8


class SystemConfig(BaseModel):
    """Compute, save, and display parameters"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    verbose_positive_done_reward: bool = True
    replay_buffer_file: str = "replay.json"
    device: str = "cuda"

    # Use suffix fields for JSON-serializable dtype representation.
    dtype_: str = "float32"
    amp_enable: bool = False
    amp_device_type: str = "cuda"
    amp_dtype_: str = "float16"

    num_workers: int = 4

    @property
    def dtype(self) -> torch.dtype:
        """Backward-compatible torch.dtype accessor."""
        return getattr(torch, self.dtype_)

    @dtype.setter
    def dtype(self, v: torch.dtype) -> None:
        self.dtype_ = str(v).replace("torch.", "")

    @property
    def amp_dtype(self) -> torch.dtype:
        """Backward-compatible torch.dtype accessor."""
        return getattr(torch, self.amp_dtype_)

    @amp_dtype.setter
    def amp_dtype(self, v: torch.dtype) -> None:
        self.amp_dtype_ = str(v).replace("torch.", "")

    @field_validator("dtype_", "amp_dtype_")
    @classmethod
    def _check_dtype_name(cls, v: str) -> str:
        # Keep it permissive but ensure it's a torch attribute.
        if not hasattr(torch, v):
            raise ValueError(f"Unknown torch dtype name: {v}")
        return v


class EvalConfig(BaseModel):
    """Evaluation mode configuration"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    eval_mode: bool = False
    memget_num_per_update: int = 1600
    action_temperature: float = 0.01
    n_display: int = 16

    ci_enabled: bool = True
    ci_confidence: float = 0.95


class Config(BaseModel):
    """Main configuration combining all sub-configs"""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )

    version: str = "0.1"
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    env: EnvConfig = Field(default_factory=EnvConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)

    # Keep your original name `evalu`, but accept `eval=...` for compatibility.
    evalu: EvalConfig = Field(default_factory=EvalConfig, alias="eval")

    verbose_first: bool = True

    def model_post_init(self, __context: Any) -> None:
        self._sync_episodes()
        self._apply_eval_mode()

    def _sync_episodes(self) -> None:
        """Sync episode count across all schedulers"""
        episodes = self.training.pass_n_episodes
        self.training.reset_scheduler(episodes)
        self.target.reset_schedulers(episodes)

    def _apply_eval_mode(self) -> None:
        """Apply eval mode settings if enabled"""
        if self.evalu.eval_mode:
            self.evalu.memget_num_per_update = 1000
            self.evalu.action_temperature = 0.1
            self.evalu.n_display = 50

            # Force epsilon to 0 in eval by swapping scheduler to constant(0.0).
            self.training._action_epsilon_scheduler = ConstantScheduler(0.0)
            self.training.action_epsilon_state = (
                self.training._action_epsilon_scheduler.to_state()
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Export config to a JSON-serializable dict.
        """
        data = self.model_dump(mode="python", by_alias=True)
        return data

    def to_json(
        self,
        indent: int = 2,
        ensure_ascii: bool = False,
    ) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            ensure_ascii=ensure_ascii,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Config:
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, s: str) -> Config:
        return cls.from_dict(json.loads(s))


def get_default_config() -> Config:
    """Get default training configuration"""
    return Config()


def get_eval_config() -> Config:
    """Get evaluation mode configuration"""
    return Config(eval=EvalConfig(eval_mode=True))
