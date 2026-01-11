from typing import Optional, Sequence, Tuple

from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.meld import Meld

from .event_bus import EventBus
from .tiles import (
    ChiOption,
    chi_option_from_sequence,
    iter_chi_sequences_containing,
    is_suit,
)


class MahjongHand:
    """
    A concealed hand represented by tile34 counts.

    This class does not know about turn order or claims; it only handles tiles,
    shanten and hand value estimation.
    """

    def __init__(
        self, seat: Optional[int] = None, bus: Optional[EventBus] = None
    ) -> None:
        self._counts: list[int] = [0] * 34
        self.melds: list[Meld] = []
        self.has_open_tanyao: bool = False
        self._seat: Optional[int] = seat
        self._bus: Optional[EventBus] = bus

    def copy(self) -> "MahjongHand":
        h = MahjongHand()
        h._counts = self._counts.copy()
        h.melds = self.melds.copy()
        h.has_open_tanyao = self.has_open_tanyao
        return h

    def as_tile34(self) -> list[int]:
        """Return a copy of tile34 counts."""
        return self._counts.copy()

    def total_tiles(self) -> int:
        """Total tile count in concealed hand."""
        return sum(self._counts)

    def count(self, tile34: int) -> int:
        return self._counts[tile34]

    def add(self, tile34: int, n: int = 1) -> None:
        """Add tiles into concealed hand."""
        if n < 0:
            raise ValueError("n must be non-negative")
        if self._counts[tile34] + n > 4:
            raise ValueError("tile count would exceed 4")
        self._counts[tile34] += n

    def remove(self, tile34: int, n: int = 1) -> None:
        """Remove tiles from concealed hand."""
        if n < 0:
            raise ValueError("n must be non-negative")
        if self._counts[tile34] - n < 0:
            raise ValueError("tile count would go negative")
        self._counts[tile34] -= n

    def shanten(self, shanten_calc: Shanten) -> int:
        """Compute shanten number."""
        return shanten_calc.calculate_shanten(self.as_tile34())

    def estimate_value(
        self,
        calculator: HandCalculator,
        win_tile34: int,
        melds,
        dora_indicators_136: list[int],
        is_riichi: bool,
        has_open_tanyao: bool,
    ):
        """Estimate hand value using 'mahjong' library calculator."""
        win_tile_136 = TilesConverter.to_136_array(self._one_tile34(win_tile34))[0]
        tiles_34 = self.as_tile34()
        # Add meld to tiles_136, refer to https://github.com/MahjongRepository/mahjong/issues/54
        for meld in self.melds:
            for tile in meld.tiles_34:
                tiles_34[tile] += 1
        tiles_136 = TilesConverter.to_136_array(tiles_34)
        return calculator.estimate_hand_value(
            tiles_136,
            win_tile_136,
            melds=melds,
            dora_indicators=dora_indicators_136,
            config=HandConfig(
                is_riichi=is_riichi,
                options=OptionalRules(has_open_tanyao),
            ),
        )

    def possible_chi(self, discard_tile34: int) -> list[ChiOption]:
        """
        Return all chi options for a given discard tile (sliding-window implementation).
        Caller must still enforce "only from left player" rule at environment level.
        """
        if not is_suit(discard_tile34):
            return []

        options: list[ChiOption] = []
        for seq in iter_chi_sequences_containing(discard_tile34):
            opt = chi_option_from_sequence(discard_tile34, seq)
            a, b = opt.need_remove
            if self._counts[a] > 0 and self._counts[b] > 0:
                options.append(opt)
        return options

    def available_improvement_count(
        self, shanten_calc: Shanten, remaining_tile34: Sequence[int]
    ) -> int:
        """
        Compute the number of remaining tiles that can reduce shanten after discarding one tile.
        This preserves the spirit of the original implementation, but uses tile34 indices directly.

        Note: this is expensive (O(34 * unique_remaining)) and mainly intended for reward shaping.
        """
        tiles = self.as_tile34()
        now_shanten = shanten_calc.calculate_shanten(tiles)

        unique_remaining = set(remaining_tile34)
        improving_draws = set()

        for idx, cnt in enumerate(tiles):
            if cnt <= 0:
                continue
            tmp = tiles.copy()
            tmp[idx] -= 1
            for draw in unique_remaining:
                tmp2 = tmp.copy()
                tmp2[draw] += 1
                if shanten_calc.calculate_shanten(tmp2) < now_shanten:
                    improving_draws.add(draw)

        total = 0
        # count duplicates in remaining wall
        for t in improving_draws:
            total += remaining_tile34.count(t)
        return total

    @staticmethod
    def _one_tile34(tile34: int) -> list[int]:
        a = [0] * 34
        a[tile34] = 1
        return a

    def add_open_meld_chi(self, seq_tile34: Tuple[int, int, int]) -> None:
        counts = [0] * 34
        for t in seq_tile34:
            counts[t] += 1
        tiles136 = TilesConverter.to_136_array(counts)
        self.melds.append(Meld(meld_type=Meld.CHI, tiles=tiles136, opened=True))
        self.has_open_tanyao = True
        if self._bus is not None and self._seat is not None:
            self._bus.publish(
                "meld_made", seat=self._seat, kind="chi", tiles34=list(seq_tile34)
            )

    def add_open_meld_pon(self, tile34: int) -> None:
        counts = [0] * 34
        counts[tile34] = 3
        tiles136 = TilesConverter.to_136_array(counts)
        self.melds.append(Meld(meld_type=Meld.PON, tiles=tiles136, opened=True))
        self.has_open_tanyao = True
        if self._bus is not None and self._seat is not None:
            self._bus.publish(
                "meld_made", seat=self._seat, kind="pon", tiles34=[tile34] * 3
            )

    def add_open_meld_kan(self, tile34: int) -> None:
        counts = [0] * 34
        counts[tile34] = 4
        tiles136 = TilesConverter.to_136_array(counts)
        self.melds.append(Meld(meld_type=Meld.KAN, tiles=tiles136, opened=True))
        self.has_open_tanyao = True
        if self._bus is not None and self._seat is not None:
            self._bus.publish(
                "meld_made", seat=self._seat, kind="minkan", tiles34=[tile34] * 4
            )

    def add_closed_kan(self, tile34: int) -> None:
        counts = [0] * 34
        counts[tile34] = 4
        tiles136 = TilesConverter.to_136_array(counts)
        self.melds.append(Meld(meld_type=Meld.KAN, tiles=tiles136, opened=False))
        if self._bus is not None and self._seat is not None:
            self._bus.publish(
                "meld_made", seat=self._seat, kind="ankan", tiles34=[tile34] * 4
            )

    def has_open_pon(self, tile34: int) -> bool:
        pon_counts = [0] * 34
        pon_counts[tile34] = 3
        pon136 = TilesConverter.to_136_array(pon_counts)
        target = str(Meld(meld_type=Meld.PON, tiles=pon136, opened=True))
        for m in self.melds:
            if str(m) == target:
                return True
        return False

    def upgrade_pon_to_kan(self, tile34: int) -> None:
        """
        Upgrade an existing pon meld into kan meld (kakan).
        """
        # find matching pon
        pon_counts = [0] * 34
        pon_counts[tile34] = 3
        pon136 = TilesConverter.to_136_array(pon_counts)
        target = str(Meld(meld_type=Meld.PON, tiles=pon136, opened=True))

        idx = None
        for i, m in enumerate(self.melds):
            if str(m) == target:
                idx = i
                break
        if idx is None:
            raise RuntimeError("no matching pon meld to upgrade")

        del self.melds[idx]
        self.add_open_meld_kan(tile34)
        if self._bus is not None and self._seat is not None:
            self._bus.publish(
                "meld_upgraded", seat=self._seat, kind="kakan", tile34=tile34
            )
