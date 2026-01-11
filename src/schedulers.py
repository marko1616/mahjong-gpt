from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


class Scheduler(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def step(self) -> float:
        pass

    @abstractmethod
    def get(self) -> float:
        pass


class LinearScheduler(Scheduler):
    def __init__(self, to_min_step: int, init: float, minimum: float) -> None:
        self.param = init
        self.minimum = minimum
        self.k = (minimum - init) / to_min_step

    def step(self) -> float:
        if self.param <= self.minimum:
            return self.minimum

        self.param += self.k
        return self.param

    def get(self) -> float:
        if self.param <= self.minimum:
            return self.minimum
        return self.param


class ConstantScheduler(Scheduler):
    def __init__(self, value: float) -> None:
        self.value = value

    def step(self) -> float:
        return self.value

    def get(self) -> float:
        return self.value


class ExponentialScheduler(Scheduler):
    def __init__(self, init: float, decay: float, minimum: float) -> None:
        self.param = init
        self.decay = decay
        self.minimum = minimum

    def step(self) -> float:
        if self.param <= self.minimum:
            return self.minimum
        self.param *= self.decay
        return max(self.param, self.minimum)

    def get(self) -> float:
        return max(self.param, self.minimum)


@dataclass
class SchedulerConfig:
    """Configuration for creating a scheduler"""

    scheduler_type: Literal["linear", "constant", "exponential"] = "linear"
    init: float = 1.0
    minimum: float = 0.0
    to_min_step: int = 100
    decay: float = 0.99

    def build(self, episodes_override: int | None = None) -> Scheduler:
        """Build a scheduler instance from this config"""
        steps = episodes_override if episodes_override is not None else self.to_min_step

        if self.scheduler_type == "linear":
            return LinearScheduler(steps, self.init, self.minimum)
        elif self.scheduler_type == "constant":
            return ConstantScheduler(self.init)
        elif self.scheduler_type == "exponential":
            return ExponentialScheduler(self.init, self.decay, self.minimum)
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
