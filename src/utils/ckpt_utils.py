from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import torch

from schedulers import Scheduler


def get_rng_state() -> Dict[str, Any]:
    """Collect RNG states required for deterministic resume."""
    state: Dict[str, Any] = {
        "python_random": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG states collected by get_rng_state()."""
    import random as _random

    _random.setstate(state["python_random"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_mkdir(dir_path: Path) -> None:
    """Create a directory atomically by using a temp name and rename."""
    dir_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dir_path.with_name(dir_path.name + ".tmp")
    if tmp.exists():
        # Best-effort cleanup from a previous crash.
        for p in tmp.rglob("*"):
            if p.is_file():
                p.unlink()
        tmp.rmdir()
    tmp.mkdir(parents=True, exist_ok=False)
    tmp.replace(dir_path)


class EpisodeCheckpointWriter:
    """Write an episode checkpoint directory with atomic finalize."""

    def __init__(self, ckpt_dir: Path) -> None:
        self.ckpt_dir = ckpt_dir
        self.tmp_dir = ckpt_dir.with_name(ckpt_dir.name + ".tmp")

    def begin(self) -> None:
        """Create a temporary checkpoint directory."""
        if self.tmp_dir.exists():
            # Cleanup if exists.
            for p in self.tmp_dir.rglob("*"):
                if p.is_file():
                    p.unlink()
            for p in sorted(self.tmp_dir.rglob("*"), reverse=True):
                if p.is_dir():
                    p.rmdir()
            self.tmp_dir.rmdir()
        self.tmp_dir.mkdir(parents=True, exist_ok=False)

    def write_json(self, name: str, data: Dict[str, Any]) -> None:
        """Write a JSON file under the temporary checkpoint directory."""
        path = self.tmp_dir / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())

    def write_torch(self, name: str, obj: Any) -> None:
        """Write a torch-serialized file under the temporary checkpoint directory."""
        path = self.tmp_dir / name
        torch.save(obj, path)

    def finalize(self) -> None:
        """Atomically rename temp dir to final checkpoint dir."""
        if self.ckpt_dir.exists():
            # Keep existing checkpoint; do not overwrite.
            return
        self.tmp_dir.replace(self.ckpt_dir)


def export_scheduler_states(
    schedulers: Dict[str, Scheduler],
) -> Dict[str, Dict[str, Any]]:
    """Export states of multiple schedulers."""
    return {k: v.to_state() for k, v in schedulers.items()}
