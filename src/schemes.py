from pydantic import BaseModel, Field
from collections import deque
from pathlib import Path
import gzip
import random


from env.tokens import (
    TokenList,
)


class StepInfo(BaseModel):
    """Single step info with optional fields."""

    reward_update: float = 0.0
    extra: dict = Field(default_factory=dict)

    class Config:
        extra = "allow"


class Trail(BaseModel):
    """A single-player trajectory with Pydantic validation."""

    states: list[TokenList] = Field(default_factory=list)
    rewards: list[float] = Field(default_factory=list)
    dones: list[bool] = Field(default_factory=list)
    info: list[dict] = Field(default_factory=list)
    actions: list[int] = Field(default_factory=list)
    action_masks: list[list[int]] = Field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.actions)

    @property
    def total_reward(self) -> float:
        return sum(self.rewards)


class ReplayBufferData(BaseModel):
    """Serializable container for ReplayBuffer."""
    buffer_size: int
    trails: list[Trail] = Field(default_factory=list)


class ReplayBuffer:
    """Replay buffer with Pydantic-based serialization."""

    def __init__(self, buffer_size: int) -> None:
        self.buffer_size = buffer_size
        self.buffer: deque[Trail] = deque(maxlen=buffer_size)

    def add(self, trail: Trail) -> None:
        if not isinstance(trail, Trail):
            trail = (
                Trail(**trail)
                if isinstance(trail, dict)
                else Trail.model_validate(trail)
            )
        self.buffer.append(trail)

    def sample(self, batch_size: int) -> list[Trail]:
        sample_size = min(len(self.buffer), batch_size)
        return random.sample(list(self.buffer), sample_size)

    def __len__(self) -> int:
        return len(self.buffer)


    def to_json(self, pretty: bool = False) -> str:
        data = ReplayBufferData(
            buffer_size=self.buffer_size,
            trails=list(self.buffer),
        )
        return data.model_dump_json(indent=2 if pretty else None)

    @classmethod
    def from_json(cls, json_str: str) -> "ReplayBuffer":
        """Deserialize from JSON string."""
        data = ReplayBufferData.model_validate_json(json_str)
        buffer = cls(buffer_size=data.buffer_size)
        for trail in data.trails:
            buffer.buffer.append(trail)
        return buffer

    def save(
        self,
        path: str | Path,
        compress: bool = False,
        pretty: bool = True,
    ) -> None:
        path = Path(path)
        json_str = self.to_json(pretty=pretty)

        if compress:
            path = path.with_suffix(".json.gz")
            with gzip.open(path, "wt", encoding="utf-8") as f:
                f.write(json_str)
        else:
            path = path.with_suffix(".json")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)

        print(f"Saved {len(self.buffer)} trails to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        path = Path(path)

        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            json_str = f.read()

        data = ReplayBufferData.model_validate_json(json_str)
        buffer = cls(buffer_size=data.buffer_size)
        for trail in data.trails:
            buffer.buffer.append(trail)

        print(f"Loaded {len(buffer)} trails from {path}")
        return buffer