from pathlib import Path

from typing import Optional
from pydantic import BaseModel
from rich import print

from torch.utils.tensorboard import SummaryWriter

from .utils import RunningStats, z_from_confidence, normal_mean_bounds
from .config import Config


class StepMetrics(BaseModel):
    """
    Data model for recording metrics at a specific training step.
    """

    episode: int
    forward_pass_step: int
    loss: float
    value_loss: float
    policy_loss: float


class TrailMetrics(BaseModel):
    """
    Data model for recording metrics of a specific trail (step-wise rewards).
    """

    episode: int
    trail_idx: int
    rewards: list[float]
    trail_reward: float


class EpisodeMetrics(BaseModel):
    """
    Data model for recording aggregated metrics at the end of an episode.
    """

    episode: int
    avr_reward: float
    max_reward: float
    avr_reward_per_trail: float
    max_reward_per_trail: float
    # Optional fields for Confidence Intervals
    avr_reward_low: Optional[float] = None
    avr_reward_high: Optional[float] = None
    avr_trail_reward_low: Optional[float] = None
    avr_trail_reward_high: Optional[float] = None


class Recorder:
    """
    Handles logging of training statistics to console, TensorBoard, and JSON files.
    Tracks both high-level episode stats and granular step-wise data.
    """

    def __init__(self, config: Config, log_dir: str | Path | None = None) -> None:
        self.config = config

        # Internal buffers
        self.rewards: list[float] = []
        self.trail_rewards: list[float] = []
        self.losses: list[float] = []
        self.value_losses: list[float] = []
        self.policy_losses: list[float] = []

        # Aggregated stats
        self.avr_reward = 0.0
        self.max_reward = 0.0
        self.avr_reward_per_trail = 0.0
        self.max_reward_per_trail = 0.0
        self.reward_update_time = 0

        # Counters
        self.episode_count = 0
        self.forward_pass_step = 0

        # Logging setup
        self.writer = SummaryWriter(log_dir)

        # JSON Logging Paths
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.episode_log_path = self.log_dir / "episode_metrics.jsonl"
        self._set_episode_paths(self.episode_count)

        # Statistics helpers
        self.reward_stats = RunningStats()
        self.trail_reward_stats = RunningStats()
        self.ci_z = (
            z_from_confidence(config.evalu.ci_confidence)
            if config.evalu.ci_enabled
            else 0.0
        )

        self.avr_reward_low = 0.0
        self.avr_reward_high = 0.0
        self.avr_trail_reward_low = 0.0
        self.avr_trail_reward_high = 0.0

        self.last_episode_metrics: EpisodeMetrics | None = None

    def _set_episode_paths(self, episode_idx0: int) -> None:
        self.step_log_path = self.log_dir / f"step_metrics_episode_{episode_idx0}.jsonl"
        self.trail_log_path = (
            self.log_dir / f"trail_metrics_episode_{episode_idx0}.jsonl"
        )

    def _log_to_jsonl(self, file_path: Path, data: BaseModel) -> None:
        """
        Helper to write a Pydantic model as a JSON line to a file.
        """
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(data.model_dump_json() + "\n")

    def reward_update(self, reward_trail: list[float]) -> None:
        """
        Updates reward statistics with a new trail of rewards.
        Calculates means, maxes, and confidence intervals if enabled.
        """
        for r in reward_trail:
            r = float(r)
            self.rewards.append(r)
            self.reward_stats.update(r)

        trail_sum = float(sum(reward_trail))
        self.trail_rewards.append(trail_sum)
        self.trail_reward_stats.update(trail_sum)

        self.reward_update_time += 1

        # Log trail metrics to JSONL
        trail_data = TrailMetrics(
            episode=self.episode_count,
            trail_idx=self.reward_update_time,
            rewards=[float(r) for r in reward_trail],
            trail_reward=trail_sum,
        )
        self._log_to_jsonl(self.trail_log_path, trail_data)

        self.avr_reward = self.reward_stats.mean
        self.max_reward = (
            max(self.max_reward, max(reward_trail)) if self.rewards else 0.0
        )
        self.avr_reward_per_trail = self.trail_reward_stats.mean
        self.max_reward_per_trail = (
            max(self.max_reward_per_trail, trail_sum) if self.trail_rewards else 0.0
        )

        if self.config.evalu.ci_enabled:
            self.avr_reward_low, self.avr_reward_high, _ = normal_mean_bounds(
                self.avr_reward, self.reward_stats.var, self.reward_stats.n, self.ci_z
            )
            self.avr_trail_reward_low, self.avr_trail_reward_high, _ = (
                normal_mean_bounds(
                    self.avr_reward_per_trail,
                    self.trail_reward_stats.var,
                    self.trail_reward_stats.n,
                    self.ci_z,
                )
            )

        # Print to console based on display interval
        display_interval = int(
            self.config.evalu.memget_num_per_update / self.config.evalu.n_display
        )
        # Avoid division by zero if display_interval is 0
        if display_interval > 0 and self.reward_update_time % display_interval == 0:
            self._print_stats()

    def _print_stats(self) -> None:
        """Helper to print current statistics to stdout."""
        header = f"Collected {self.reward_update_time} trails in episode {self.episode_count}\n"

        if self.config.evalu.ci_enabled:
            stats = (
                f"AVR reward: {self.avr_reward:.4f}  "
                f"[{self.avr_reward_low:.4f}, {self.avr_reward_high:.4f}] (z={self.ci_z:.3f})\n"
                f"MAX reward: {self.max_reward:.4f}\n"
                f"AVR trail reward: {self.avr_reward_per_trail:.4f}  "
                f"[{self.avr_trail_reward_low:.4f}, {self.avr_trail_reward_high:.4f}] (z={self.ci_z:.3f})\n"
                f"MAX trail reward: {self.max_reward_per_trail:.4f}"
            )
        else:
            stats = (
                f"AVR reward: {self.avr_reward:.4f}\n"
                f"MAX reward: {self.max_reward:.4f}\n"
                f"AVR trail reward: {self.avr_reward_per_trail:.4f}\n"
                f"MAX trail reward: {self.max_reward_per_trail:.4f}"
            )
        print(header + stats)

    def loss_update(self, loss: float, value_loss: float, policy_loss: float) -> None:
        """
        Updates loss lists and logs step-wise metrics to JSON.
        """
        self.losses.append(loss)
        self.value_losses.append(value_loss)
        self.policy_losses.append(policy_loss)

        self.forward_pass_step += 1

        # Create Pydantic model for step data
        step_data = StepMetrics(
            episode=self.episode_count,
            forward_pass_step=self.forward_pass_step,
            loss=loss,
            value_loss=value_loss,
            policy_loss=policy_loss,
        )

        # Log granular step data to JSONL file
        self._log_to_jsonl(self.step_log_path, step_data)

    def seek(
        self, start_episode_idx0: int, *, overwrite_current_episode_files: bool = False
    ) -> None:
        """Align recoder"""
        self.episode_count = start_episode_idx0

        self.forward_pass_step = 0
        self.reward_update_time = 0

        self.rewards = []
        self.trail_rewards = []
        self.losses = []
        self.value_losses = []
        self.policy_losses = []

        self.avr_reward = 0.0
        self.max_reward = 0.0
        self.avr_reward_per_trail = 0.0
        self.max_reward_per_trail = 0.0

        self.reward_stats.reset()
        self.trail_reward_stats.reset()

        self._set_episode_paths(self.episode_count)

        if overwrite_current_episode_files:
            for p in (self.step_log_path, self.trail_log_path):
                if p.exists():
                    p.unlink()

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()

    def reset(self) -> EpisodeMetrics:
        """
        Called at the end of an episode/update cycle.
        Aggregates statistics, writes to TensorBoard, and logs episode metrics to JSON.
        Resets internal buffers for the next episode.
        """
        if self.losses:
            self.writer.add_scalar(
                "Loss/loss", sum(self.losses) / len(self.losses), self.episode_count
            )
            self.writer.add_scalar(
                "Loss/value loss",
                sum(self.value_losses) / len(self.value_losses),
                self.episode_count,
            )
            self.writer.add_scalar(
                "Loss/policy loss",
                sum(self.policy_losses) / len(self.policy_losses),
                self.episode_count,
            )

        self.writer.add_scalar("AVR Reward", self.avr_reward, self.episode_count)
        self.writer.add_scalar(
            "AVR Trail Reward", self.avr_reward_per_trail, self.episode_count
        )

        ci_data = {}
        if self.config.evalu.ci_enabled:
            low, high, se = normal_mean_bounds(
                self.avr_reward, self.reward_stats.var, self.reward_stats.n, self.ci_z
            )
            self.writer.add_scalar("AVR Reward CI/low", low, self.episode_count)
            self.writer.add_scalar("AVR Reward CI/high", high, self.episode_count)
            self.writer.add_scalar("AVR Reward CI/se", se, self.episode_count)

            tlow, thigh, tse = normal_mean_bounds(
                self.avr_reward_per_trail,
                self.trail_reward_stats.var,
                self.trail_reward_stats.n,
                self.ci_z,
            )
            self.writer.add_scalar("AVR Trail Reward CI/low", tlow, self.episode_count)
            self.writer.add_scalar(
                "AVR Trail Reward CI/high", thigh, self.episode_count
            )
            self.writer.add_scalar("AVR Trail Reward CI/se", tse, self.episode_count)

            ci_data = {
                "avr_reward_low": low,
                "avr_reward_high": high,
                "avr_trail_reward_low": tlow,
                "avr_trail_reward_high": thigh,
            }

        episode_data = EpisodeMetrics(
            episode=self.episode_count,
            avr_reward=self.avr_reward,
            max_reward=self.max_reward,
            avr_reward_per_trail=self.avr_reward_per_trail,
            max_reward_per_trail=self.max_reward_per_trail,
            **ci_data,
        )
        self.last_episode_metrics = episode_data
        self._log_to_jsonl(self.episode_log_path, episode_data)

        self.episode_count += 1
        self._set_episode_paths(self.episode_count)

        self.rewards = []
        self.trail_rewards = []
        self.losses = []
        self.value_losses = []
        self.policy_losses = []

        self.avr_reward = 0.0
        self.max_reward = 0.0
        self.avr_reward_per_trail = 0.0
        self.max_reward_per_trail = 0.0
        self.reward_update_time = 0
        self.forward_pass_step = 0

        self.reward_stats.reset()
        self.trail_reward_stats.reset()

        return episode_data
