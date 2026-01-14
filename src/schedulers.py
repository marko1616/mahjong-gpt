from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Literal, Optional, Type

from pydantic import BaseModel, ConfigDict, field_validator


class SchedulerType(str, Enum):
    """Scheduler type enum for stable JSON serialization."""

    linear = "linear"
    constant = "constant"
    exponential = "exponential"


class Scheduler(ABC):
    """Base scheduler interface with state export/import."""

    @abstractmethod
    def step(self) -> float:
        """Advance one step and return current value."""
        raise NotImplementedError

    @abstractmethod
    def get(self) -> float:
        """Get current value without stepping."""
        raise NotImplementedError

    @abstractmethod
    def to_state(self) -> Dict[str, Any]:
        """Export a JSON-serializable state dict (must include scheduler_type and t)."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_state(cls, state: Dict[str, Any]) -> "Scheduler":
        """Restore scheduler from a state dict created by to_state()."""
        raise NotImplementedError


class _BaseSchedulerState(BaseModel):
    """Base state model shared by all schedulers."""

    model_config = ConfigDict(extra="forbid")

    scheduler_type: SchedulerType
    t: int = 0  # number of step() calls already taken


class LinearSchedulerState(_BaseSchedulerState):
    scheduler_type: SchedulerType = SchedulerType.linear
    init: float
    minimum: float
    to_min_step: int
    param: float


class ConstantSchedulerState(_BaseSchedulerState):
    scheduler_type: SchedulerType = SchedulerType.constant
    value: float


class ExponentialSchedulerState(_BaseSchedulerState):
    scheduler_type: SchedulerType = SchedulerType.exponential
    init: float
    decay: float
    minimum: float
    param: float


_STATE_MODEL_BY_TYPE: Dict[SchedulerType, Type[_BaseSchedulerState]] = {
    SchedulerType.linear: LinearSchedulerState,
    SchedulerType.constant: ConstantSchedulerState,
    SchedulerType.exponential: ExponentialSchedulerState,
}


def scheduler_from_state(state: Dict[str, Any]) -> Scheduler:
    """Factory: build a scheduler instance from a state dict."""
    if "scheduler_type" not in state:
        raise ValueError("Scheduler state missing 'scheduler_type'.")

    stype = SchedulerType(state["scheduler_type"])
    model_cls = _STATE_MODEL_BY_TYPE.get(stype)
    if model_cls is None:
        raise ValueError(f"Unknown scheduler_type in state: {state['scheduler_type']}")

    parsed = model_cls.model_validate(state)

    if stype == SchedulerType.linear:
        return LinearScheduler.from_state(parsed.model_dump())
    if stype == SchedulerType.constant:
        return ConstantScheduler.from_state(parsed.model_dump())
    if stype == SchedulerType.exponential:
        return ExponentialScheduler.from_state(parsed.model_dump())

    raise ValueError(f"Unsupported scheduler_type: {stype}")


class LinearScheduler(Scheduler):
    """Linear scheduler: param moves from init to minimum over to_min_step steps."""

    def __init__(
        self,
        to_min_step: int,
        init: float,
        minimum: float,
        *,
        t: int = 0,
        param: Optional[float] = None,
    ) -> None:
        self.to_min_step = int(max(to_min_step, 1))
        self.init = float(init)
        self.minimum = float(minimum)

        # slope per step
        self.k = (self.minimum - self.init) / self.to_min_step

        self.t = int(max(t, 0))
        self.param = float(param) if param is not None else self.init

    def step(self) -> float:
        self.t += 1
        if self.param <= self.minimum:
            self.param = self.minimum
            return self.minimum

        self.param += self.k
        if self.param <= self.minimum:
            self.param = self.minimum
        return self.param

    def get(self) -> float:
        return self.minimum if self.param <= self.minimum else self.param

    def to_state(self) -> Dict[str, Any]:
        state = LinearSchedulerState(
            t=self.t,
            init=self.init,
            minimum=self.minimum,
            to_min_step=self.to_min_step,
            param=self.param,
        )
        return state.model_dump()

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "LinearScheduler":
        s = LinearSchedulerState.model_validate(state)
        return cls(
            to_min_step=s.to_min_step,
            init=s.init,
            minimum=s.minimum,
            t=s.t,
            param=s.param,
        )


class ConstantScheduler(Scheduler):
    """Constant scheduler: always returns a fixed value."""

    def __init__(self, value: float, *, t: int = 0) -> None:
        self.value = float(value)
        self.t = int(max(t, 0))

    def step(self) -> float:
        self.t += 1
        return self.value

    def get(self) -> float:
        return self.value

    def to_state(self) -> Dict[str, Any]:
        state = ConstantSchedulerState(t=self.t, value=self.value)
        return state.model_dump()

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "ConstantScheduler":
        s = ConstantSchedulerState.model_validate(state)
        return cls(value=s.value, t=s.t)


class ExponentialScheduler(Scheduler):
    """Exponential scheduler: param *= decay until minimum."""

    def __init__(
        self,
        init: float,
        decay: float,
        minimum: float,
        *,
        t: int = 0,
        param: Optional[float] = None,
    ) -> None:
        self.init = float(init)
        self.decay = float(decay)
        self.minimum = float(minimum)

        self.t = int(max(t, 0))
        self.param = float(param) if param is not None else self.init

    def step(self) -> float:
        self.t += 1
        if self.param <= self.minimum:
            self.param = self.minimum
            return self.minimum
        self.param *= self.decay
        if self.param <= self.minimum:
            self.param = self.minimum
        return self.param

    def get(self) -> float:
        return max(self.param, self.minimum)

    def to_state(self) -> Dict[str, Any]:
        state = ExponentialSchedulerState(
            t=self.t,
            init=self.init,
            decay=self.decay,
            minimum=self.minimum,
            param=self.param,
        )
        return state.model_dump()

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "ExponentialScheduler":
        s = ExponentialSchedulerState.model_validate(state)
        return cls(
            init=s.init,
            decay=s.decay,
            minimum=s.minimum,
            t=s.t,
            param=s.param,
        )


class SchedulerConfig(BaseModel):
    """Configuration for creating a scheduler."""

    model_config = ConfigDict(extra="forbid")

    scheduler_type: Literal["linear", "constant", "exponential"] = "linear"
    init: float = 1.0
    minimum: float = 0.0
    to_min_step: int = 100
    decay: float = 0.99

    @field_validator("to_min_step")
    @classmethod
    def _check_to_min_step(cls, v: int) -> int:
        return int(max(v, 1))

    def build(
        self,
        episodes_override: Optional[int] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> Scheduler:
        """
        Build a scheduler instance from this config.

        - If state is provided, it will restore the scheduler from state (preferred for resuming).
        - episodes_override overrides the number of steps for linear schedule if creating fresh.
        """
        if state is not None:
            # Restore from state (must include scheduler_type).
            return scheduler_from_state(state)

        steps = (
            int(episodes_override)
            if episodes_override is not None
            else int(self.to_min_step)
        )
        stype = SchedulerType(self.scheduler_type)

        if stype == SchedulerType.linear:
            return LinearScheduler(steps, self.init, self.minimum)
        if stype == SchedulerType.constant:
            # Constant uses init as its value for compatibility with your old config.
            return ConstantScheduler(self.init)
        if stype == SchedulerType.exponential:
            return ExponentialScheduler(self.init, self.decay, self.minimum)

        raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
