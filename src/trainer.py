from pathlib import Path
from typing import Optional

import torch

from .ckpt_manager import CkptManager, RunManifest, PassSpec
from .utils.ckpt_utils import (
    EpisodeCheckpointWriter,
    export_scheduler_states,
    get_rng_state,
    set_rng_state,
)

from .agent import Agent
from .recorder import Recorder
from .config import get_default_config


class TrainerRunner:
    """High-level runner that executes passes and supports episode-wise resume."""

    def __init__(self, store: CkptManager) -> None:
        self.store = store

    def _find_latest_checkpoint_episode(self, pass_id: int) -> Optional[int]:
        """Return the latest checkpoint episode number if any."""
        ckpt_root = self.store.checkpoints_dir(pass_id)
        if not ckpt_root.exists():
            return None
        latest = None
        for d in ckpt_root.glob("ep_*"):
            if d.is_dir():
                try:
                    ep = int(d.name.split("_")[1])
                    latest = ep if latest is None else max(latest, ep)
                except Exception:
                    continue
        return latest

    def run(self, manifest: RunManifest) -> None:
        """Run starting from the active pass until completion."""
        while True:
            active_id = manifest.active_pass_id
            rec = next(
                (p for p in manifest.passes if p.spec.pass_id == active_id), None
            )
            if rec is None:
                return

            if rec.state.status in ("completed", "failed"):
                return

            self._run_single_pass(manifest, active_id)

            # Stop after one pass by default. You may auto-append next pass here.
            return

    def _run_single_pass(self, manifest: RunManifest, pass_id: int) -> None:
        """Execute one pass with checkpoint-based resume."""
        rec = next(p for p in manifest.passes if p.spec.pass_id == pass_id)
        spec = rec.spec
        state = rec.state
        state.status = "running"

        # Resume point: prefer state's last checkpoint, otherwise scan.
        latest_ep = self._find_latest_checkpoint_episode(pass_id)
        start_episode = 0 if latest_ep is None else (latest_ep + 1)

        config = spec.config

        agent = Agent(config)
        recorder = Recorder(config, self.store.logs_dir(pass_id))

        # Restore checkpoint if exists.
        if latest_ep is not None:
            ckpt_dir = self.store.episode_ckpt_dir(pass_id, latest_ep)
            meta = self._load_json(ckpt_dir / "meta.json")
            rng = torch.load(ckpt_dir / "rng.pt", weights_only=False)
            set_rng_state(rng)

            agent.load_weights(ckpt_dir)
            agent.load_optim_state(
                torch.load(ckpt_dir / "optim.pt", weights_only=False)
            )
            agent.replay_buffer.load(str(ckpt_dir / "replay.json"))
            agent.restore_scheduler_states(meta["scheduler_states"])
            recorder.seek(latest_ep, overwrite_current_episode_files=True)

        # Training loop.
        for ep in range(start_episode, spec.total_episodes):
            state.curr_episode = ep

            agent.get_memory_batch(recorder.reward_update)

            if not config.evalu.eval_mode:
                batch = agent.replay_buffer.sample(config.training.sample_trail_count)
                agent.update(batch, recorder.loss_update)

            agent.action_epsilon_scheduler.step()
            agent.n_td_scheduler.step()

            episode_metrics = recorder.reset()
            state.best_metric = max(
                state.best_metric, episode_metrics.avr_reward_per_trail
            )
            if (ep % spec.save_every) == 0:
                self._save_episode_checkpoint(
                    pass_id=pass_id,
                    episode=ep,
                    agent=agent,
                    episode_metrics=episode_metrics,
                )
                state.last_checkpoint_dir = str(
                    self.store.episode_ckpt_dir(pass_id, ep)
                )

            self.store.update_pass_state(manifest, pass_id, state)

        state.status = "completed"
        self.store.update_pass_state(manifest, pass_id, state)

    def _save_episode_checkpoint(
        self, pass_id: int, episode: int, agent, episode_metrics
    ) -> None:
        """Save a fully resumable episode checkpoint."""
        ckpt_dir = self.store.episode_ckpt_dir(pass_id, episode)
        w = EpisodeCheckpointWriter(ckpt_dir)
        w.begin()

        scheduler_states = export_scheduler_states(
            {
                "action_epsilon": agent.action_epsilon_scheduler,
                "alpha": agent.alpha_scheduler,
                "n_td": agent.n_td_scheduler,
            }
        )

        meta = {
            "pass_id": pass_id,
            "episode": episode,
            "scheduler_states": scheduler_states,
            "metrics": episode_metrics.model_dump(),
        }

        w.write_json("meta.json", meta)
        w.write_torch("rng.pt", get_rng_state())
        w.write_torch("optim.pt", agent.export_optim_state())

        agent.save_weights(w.tmp_dir)
        agent.replay_buffer.save(str(w.tmp_dir / "replay.json"))

        w.finalize()

    @staticmethod
    def _load_json(path: Path) -> dict:
        """Load a JSON file."""
        import json

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def main() -> None:
    ckpt_manager = CkptManager(Path("./runs"), "model-00")
    config = get_default_config()
    total_episodes = config.training.pass_n_episodes
    train_pass = PassSpec(
        pass_id=0,
        name="main",
        config=config,
        total_episodes=total_episodes,
        save_every=10,
        eval_every=0,
        notes="First pass",
    )
    manifest = ckpt_manager.init_new(train_pass, notes="First run")
    trainer = TrainerRunner(ckpt_manager)
    trainer.run(manifest)


if __name__ == "__main__":
    main()
