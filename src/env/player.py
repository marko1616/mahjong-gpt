from dataclasses import dataclass
from typing import Optional

from .event_bus import EventBus
from .hand import MahjongHand


@dataclass
class PlayerState:
    """Per-player flags and reward-shaping memory."""

    riichi: bool = False
    riichi_lock: bool = (
        False  # after riichi declaration turn, future discards are locked to drawn tile
    )
    has_open_tanyao: bool = False
    last_drawn: Optional[int] = None  # tile34
    first_round: bool = True
    last_shanten: Optional[int] = None
    last_available: Optional[int] = None


class MahjongPlayer:
    """
    A player entity containing concealed hand, melds, discards, and simple state.
    """

    def __init__(self, seat: int, bus: EventBus) -> None:
        self.seat = seat
        self.bus = bus
        self.hand = MahjongHand(seat=seat, bus=bus)
        self.discards: list[int] = []  # tile34 discards not called-away
        self.state = PlayerState()

    def reset(self) -> None:
        self.hand = MahjongHand(seat=self.seat, bus=self.bus)
        self.discards = []
        self.state = PlayerState()

    def draw(self, tile34: int) -> None:
        self.hand.add(tile34, 1)
        self.state.last_drawn = tile34
        self.bus.publish("tile_drawn", seat=self.seat, tile34=tile34)

    def discard(self, tile34: int) -> None:
        self.hand.remove(tile34, 1)
        self.discards.append(tile34)
        self.bus.publish("tile_discarded", seat=self.seat, tile34=tile34)

    def pop_last_discard_for_call(self, expected_tile34: int) -> None:
        """
        Remove the last discard (when it is called by another player).
        This avoids double-counting tiles between melds and discards.
        """
        if not self.discards:
            raise RuntimeError("no discard to call")
        last = self.discards[-1]
        if last != expected_tile34:
            raise RuntimeError(
                f"discard mismatch: last={last}, expected={expected_tile34}"
            )
        self.discards.pop()

    @property
    def melds(self):
        return self.hand.melds
