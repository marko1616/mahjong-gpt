from collections import deque
from typing import Deque
import random

from mahjong.tile import TilesConverter


class MahjongWall:
    """
    A standard 136-tile wall.

    We store tiles as tile34 indices (0..33) internally.
    """

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self.live: Deque[int] = deque()
        self.dora_indicators: list[int] = []
        self.rinshan: Deque[int] = deque()

    def reset(self) -> None:
        """Build and shuffle a full wall, then carve out dora indicators and rinshan."""
        tiles = [i for i in range(34) for _ in range(4)]
        self._rng.shuffle(tiles)

        # match the original behavior: take first 10 as dora indicators, next 4 as rinshan
        self.dora_indicators = tiles[:10]
        rinshan_list = tiles[10:14]
        live_list = tiles[14:]

        self.rinshan = deque(rinshan_list)
        self.live = deque(live_list)

    def remaining_live(self) -> int:
        return len(self.live)

    def draw(self) -> int:
        """Draw from live wall."""
        if not self.live:
            raise RuntimeError("live wall is empty")
        return self.live.popleft()

    def draw_rinshan(self) -> int:
        """Draw from rinshan (after kan)."""
        if not self.rinshan:
            raise RuntimeError("rinshan is empty")
        return self.rinshan.popleft()

    def dora_indicators_136(
        self, kang_count: int, end: bool, riichi: bool
    ) -> list[int]:
        """
        Return dora indicators as 136 array.
        - omote: first 5
        - ura: last 5 (only revealed at end if riichi)
        """
        inds: list[int] = []
        for i in range(kang_count + 1):
            inds.append(self.dora_indicators[i])
        if end and riichi:
            for i in range(kang_count + 1):
                inds.append(self.dora_indicators[5 + i])

        counts = [0] * 34
        for t in inds:
            counts[t] += 1
        return TilesConverter.to_136_array(counts)

    def remaining_tile34_list(self) -> list[int]:
        """Return remaining live tiles as a list (for reward shaping)."""
        return list(self.live)
