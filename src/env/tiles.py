from dataclasses import dataclass
from typing import Iterable, Tuple

from .constants import CHI_UP, CHI_MID, CHI_DOWN


def tile34_to_str(tile34: int) -> str:
    """Convert tile34 index (0..33) to compact string like '1m', '9s', '7z'."""
    if not (0 <= tile34 <= 33):
        raise ValueError(f"tile34 out of range: {tile34}")

    if tile34 <= 26:
        suit = tile34 // 9  # 0 m, 1 p, 2 s
        num = (tile34 % 9) + 1
        return f"{num}{'mps'[suit]}"
    else:
        return f"{(tile34 - 27) + 1}z"


def tile34_from_str(tile: str) -> int:
    """Convert compact string like '5m' into tile34 index (0..33)."""
    if len(tile) != 2:
        raise ValueError(f"invalid tile string: {tile}")
    num = int(tile[0])
    suit = tile[1]
    if suit in ("m", "p", "s"):
        if not (1 <= num <= 9):
            raise ValueError(f"invalid suit tile: {tile}")
        base = {"m": 0, "p": 9, "s": 18}[suit]
        return base + (num - 1)
    if suit == "z":
        if not (1 <= num <= 7):
            raise ValueError(f"invalid honor tile: {tile}")
        return 27 + (num - 1)
    raise ValueError(f"invalid tile string: {tile}")


def is_suit(tile34: int) -> bool:
    """Return True if tile is a suited tile (m/p/s)."""
    return 0 <= tile34 <= 26


def suit_base(tile34: int) -> int:
    """Return suit base offset (0, 9, 18) for suited tiles."""
    if not is_suit(tile34):
        raise ValueError(f"not a suited tile: {tile34}")
    return (tile34 // 9) * 9


def iter_chi_sequences_containing(tile34: int) -> Iterable[Tuple[int, int, int]]:
    """
    Sliding-window iterator for chi sequences that contain the given tile.
    Example: for a suited tile in a suit, possible sequences are:
    (n-2,n-1,n), (n-1,n,n+1), (n,n+1,n+2) bounded within 1..9.
    """
    if not is_suit(tile34):
        return

    base = suit_base(tile34)
    offset = tile34 - base  # 0..8

    for start in (offset - 2, offset - 1, offset):
        if 0 <= start <= 6:
            yield (base + start, base + start + 1, base + start + 2)


@dataclass(frozen=True)
class ChiOption:
    """A chi option for a discard."""

    action: int  # CHI_UP / CHI_MID / CHI_DOWN
    sequence: Tuple[int, int, int]
    need_remove: Tuple[int, int]  # the two tiles to remove from hand


def chi_option_from_sequence(
    discard_tile34: int, seq: Tuple[int, int, int]
) -> ChiOption:
    """Build ChiOption (action id + removal tiles) from a sequence and discard."""
    if discard_tile34 not in seq:
        raise ValueError("discard not in sequence")

    if discard_tile34 == seq[0]:
        action = CHI_UP
        need = (seq[1], seq[2])
    elif discard_tile34 == seq[1]:
        action = CHI_MID
        need = (seq[0], seq[2])
    else:
        action = CHI_DOWN
        need = (seq[0], seq[1])

    return ChiOption(action=action, sequence=seq, need_remove=need)
