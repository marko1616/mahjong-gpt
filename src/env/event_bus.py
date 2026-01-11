from dataclasses import dataclass
from typing import Any, Callable, DefaultDict, Dict
from collections import defaultdict


@dataclass(frozen=True)
class Event:
    """A simple event message."""

    name: str
    payload: Dict[str, Any]


Handler = Callable[[Event], None]


class EventBus:
    """Synchronous pub-sub event bus."""

    def __init__(self) -> None:
        self._handlers: DefaultDict[str, list[Handler]] = defaultdict(list)

    def subscribe(self, name: str, handler: Handler) -> None:
        """Subscribe a handler to an event name."""
        self._handlers[name].append(handler)

    def publish(self, name: str, **payload: Any) -> None:
        """Publish an event with payload."""
        evt = Event(name=name, payload=dict(payload))
        for h in list(self._handlers.get(name, [])):
            h(evt)
