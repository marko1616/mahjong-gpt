import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .config import Config


class PassSpec(BaseModel):
    """Immutable specification for a training pass.

    This object must never be mutated once the pass starts, so results remain reproducible.
    """

    init_from_pass_id: Optional[int] = None
    init_from_episode: Optional[int] = None
    init_mode: Literal["full", "weights_only"] = "full"

    model_config = ConfigDict(extra="forbid")

    pass_id: int
    name: str = ""
    config: Config
    total_episodes: int = Field(ge=1)
    save_every: int = Field(default=1, ge=1)
    eval_every: int = Field(default=0, ge=0)
    notes: str = ""


class PassState(BaseModel):
    """Mutable state for a training pass.

    This tracks progress and bookkeeping pointers, but it is not sufficient to resume training.
    Resumability relies on EpisodeCheckpoint content.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["pending", "running", "completed", "failed"] = "pending"
    curr_episode: int = 0
    best_metric: float = float("-inf")
    last_checkpoint_dir: Optional[str] = None
    updated_at_unix: float = 0.0


class PassRecord(BaseModel):
    """A pass entry inside the run manifest."""

    model_config = ConfigDict(extra="forbid")

    spec: PassSpec
    state: PassState = Field(default_factory=PassState)


class RunManifest(BaseModel):
    """Top-level manifest for a run, containing all passes."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    created_at_unix: float
    active_pass_id: int = 0
    passes: List[PassRecord] = Field(default_factory=list)
    notes: str = ""


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write JSON atomically to avoid corrupt files on crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


class CkptManager:
    """Filesystem-backed run storage with atomic manifest/state updates."""

    def __init__(self, root_dir: Path, run_id: str) -> None:
        self.root_dir = root_dir
        self.run_id = run_id
        self.run_dir = root_dir / run_id
        self.manifest_path = self.run_dir / "manifest.json"

    def exists(self) -> bool:
        """Return True if manifest exists."""
        return self.manifest_path.exists()

    def load_manifest(self) -> RunManifest:
        """Load the run manifest from disk."""
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            return RunManifest.model_validate(json.load(f))

    def save_manifest(self, manifest: RunManifest) -> None:
        """Persist manifest atomically."""
        _atomic_write_json(self.manifest_path, manifest.model_dump(mode="python"))

    def init_new(self, first_pass: PassSpec, *, notes: str = "") -> RunManifest:
        """Create a new run directory and initial manifest."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        m = RunManifest(
            run_id=self.run_id,
            created_at_unix=time.time(),
            active_pass_id=first_pass.pass_id,
            passes=[PassRecord(spec=first_pass)],
            notes=notes,
        )
        self.save_manifest(m)
        self._write_pass_spec(first_pass)
        self._write_pass_state(first_pass.pass_id, m.passes[0].state)
        return m

    def pass_dir(self, pass_id: int) -> Path:
        """Return pass directory path."""
        return self.run_dir / f"pass_{pass_id:04d}"

    def checkpoints_dir(self, pass_id: int) -> Path:
        """Return checkpoint root directory for a pass."""
        return self.pass_dir(pass_id) / "checkpoints"

    def logs_dir(self, pass_id: int) -> Path:
        """Return log directory for a pass."""
        return self.pass_dir(pass_id) / "logs"

    def episode_ckpt_dir(self, pass_id: int, episode: int) -> Path:
        """Return directory for an episode checkpoint."""
        return self.checkpoints_dir(pass_id) / f"ep_{episode:06d}"

    def _write_pass_spec(self, spec: PassSpec) -> None:
        """Write spec.json once for a pass."""
        pdir = self.pass_dir(spec.pass_id)
        pdir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(pdir / "spec.json", spec.model_dump(mode="python"))

    def _write_pass_state(self, pass_id: int, state: PassState) -> None:
        """Write state.json frequently for a pass."""
        pdir = self.pass_dir(pass_id)
        pdir.mkdir(parents=True, exist_ok=True)
        state.updated_at_unix = time.time()
        _atomic_write_json(pdir / "state.json", state.model_dump(mode="python"))

    def update_pass_state(
        self, manifest: RunManifest, pass_id: int, new_state: PassState
    ) -> None:
        """Update pass state in both manifest and pass/state.json."""
        for rec in manifest.passes:
            if rec.spec.pass_id == pass_id:
                rec.state = new_state
                break
        self.save_manifest(manifest)
        self._write_pass_state(pass_id, new_state)

    def append_pass(self, manifest: RunManifest, new_pass: PassSpec) -> RunManifest:
        """Append a new pass to existing run and make it active."""
        manifest.passes.append(PassRecord(spec=new_pass))
        manifest.active_pass_id = new_pass.pass_id
        self.save_manifest(manifest)
        self._write_pass_spec(new_pass)
        self._write_pass_state(new_pass.pass_id, manifest.passes[-1].state)
        return manifest
