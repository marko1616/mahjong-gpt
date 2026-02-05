"""
Constants for the Mahjong environment.

Vocab layout (total = 46 + 34 + 4 + 2 = 86):
 0..45        : ACTION tokens (0-based local action ids)
     0..33    : discard tile34
     34..36   : CHI_UP / CHI_MID / CHI_DOWN
     37       : PON
     38..40   : KAN_OPEN / KAN_ADD / KAN_CLOSED
     41       : PEI (3p reserved)
     42       : RIICHI
     43       : RON
     44       : TSUMO
     45       : PASS
 46..79       : HAND tile tokens
 80..83       : PLAYER tokens
 84           : [SEP]
 85           : [PAD]
"""

from pydantic import BaseModel, Field, model_validator, model_serializer
from typing import Iterator, Any
from collections.abc import Iterable
from .tiles import tile34_from_str, tile34_to_str
from .constants import (
    DISCARD_MIN,
    DISCARD_MAX,
    SPEC_MIN,
    SPEC_MAX,
    HAND_MIN,
    HAND_MAX,
    PLAYER_MIN,
    PLAYER_MAX,
    SEP_ID,
    PAD_ID,
)

PLAYER_NAMES = ["P0", "P1", "P2", "P3"]

SPEC_MAP = [
    "CHI_UP",  # 34
    "CHI_MID",  # 35
    "CHI_DOWN",  # 36
    "PON",  # 37
    "KAN_OPEN",  # 38
    "KAN_ADD",  # 39
    "KAN_CLOSED",  # 40
    "PEI",  # 41
    "RIICHI",  # 42
    "RON",  # 43
    "TSUMO",  # 44
    "PASS",  # 45
]


class TokenList(BaseModel):
    """Token sequence as a Pydantic model."""

    token_ids: list[int] = Field(default_factory=list)

    @model_serializer(mode="plain")
    def serialize(self) -> str:
        """Serialize to human-readable string."""
        return self.to_human()

    @model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: Any) -> dict[str, Any]:
        """Accept various input formats."""
        if isinstance(data, dict):
            return data
        if isinstance(data, TokenList):
            return {"token_ids": data.token_ids}
        if isinstance(data, str):
            return {"token_ids": cls._parse_human(data)}
        if isinstance(data, (list, tuple)):
            token_ids = []
            for item in data:
                if isinstance(item, int):
                    token_ids.append(item)
                elif isinstance(item, dict):
                    token_ids.append(item.get("token_id"))
                else:
                    raise ValueError(f"Invalid token type: {type(item)}")
            return {"token_ids": token_ids}
        raise ValueError(f"Cannot convert {type(data)} to TokenList")

    @classmethod
    def _parse_human(cls, s: str) -> list[int]:
        """Parse human-readable string to token list."""
        token_ids = []
        parts = s.replace("<SEP>", " <SEP> ").replace("<PAD>", " <PAD> ").split()
        for part in parts:
            if not part:
                continue
            if part in ("<SEP>", "<PAD>"):
                token_ids.append(cls._parse_token(part))
            elif part.startswith("[") and part.endswith("]"):
                token_ids.append(cls._parse_token(part))
            elif part.startswith("D"):
                token_ids.append(cls._parse_token(part))
            else:
                # parse tile sequence like "1m2m3p"
                token_ids.extend(cls._parse_tiles(part))
        return token_ids

    @staticmethod
    def _parse_token(s: str) -> int:
        """Convert token string to token ID."""
        if s == "<PAD>":
            return PAD_ID
        if s == "<SEP>":
            return SEP_ID
        if s.startswith("<P") and s.endswith(">"):
            idx = int(s[2:-1])
            return PLAYER_MIN + idx
        if s.startswith("D"):
            return DISCARD_MIN + tile34_from_str(s[1:])
        if s.startswith("[") and s.endswith("]"):
            action_name = s[1:-1]
            return SPEC_MIN + SPEC_MAP.index(action_name)
        # Parse tile string like '1m2m3p'
        return HAND_MIN + tile34_from_str(s)

    @staticmethod
    def _parse_tiles(s: str) -> list[int]:
        """Parse tile string like '1m2m3p' to token IDs."""
        token_ids = []
        i = 0
        while i < len(s):
            if i + 1 < len(s) and s[i + 1] in "mpsz":
                token_ids.append(HAND_MIN + tile34_from_str(s[i : i + 2]))
                i += 2
            else:
                i += 1
        return token_ids

    def to_ids(self) -> list[int]:
        """Convert to list of integer token IDs."""
        return self.token_ids

    def to_human(self) -> str:
        """Convert to human-readable string."""
        parts = []
        current_tiles = []

        def flush_tiles():
            nonlocal current_tiles
            if current_tiles:
                parts.append("".join(current_tiles))
                current_tiles = []

        for tid in self.token_ids:
            if tid == SEP_ID:
                flush_tiles()
                parts.append("<SEP>")
                continue
            if HAND_MIN <= tid <= HAND_MAX:
                current_tiles.append(tile34_to_str(tid - HAND_MIN))
                continue

            flush_tiles()
            parts.append(self._token_to_human(tid))

        flush_tiles()
        return " ".join(parts)

    @staticmethod
    def _token_to_human(tid: int) -> str:
        """Convert token ID to human-readable string."""
        if tid == PAD_ID:
            return "<PAD>"
        if tid == SEP_ID:
            return "<SEP>"
        if PLAYER_MIN <= tid <= PLAYER_MAX:
            return f"<{PLAYER_NAMES[tid - PLAYER_MIN]}>"
        if DISCARD_MIN <= tid <= DISCARD_MAX:
            return f"D{tile34_to_str(tid - DISCARD_MIN)}"
        if SPEC_MIN <= tid <= SPEC_MAX:
            return f"[{SPEC_MAP[tid - SPEC_MIN]}]"
        return tile34_to_str(tid - HAND_MIN)

    @classmethod
    def from_human(cls, s: str) -> TokenList:
        return cls.model_validate(s)

    def append(self, token_id: int) -> None:
        """Append a token ID."""
        self.token_ids.append(token_id)

    def extend(self, token_ids: Iterable[int]) -> None:
        """Extend with multiple token IDs."""
        self.token_ids.extend(token_ids)

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, idx: int | slice) -> int | list[int]:
        return self.token_ids[idx]

    def __iter__(self) -> Iterator[int]:
        return iter(self.token_ids)

    def __contains__(self, item: int) -> bool:
        return item in self.token_ids

    def copy(self) -> TokenList:
        """Create a deep copy."""
        return TokenList(token_ids=self.token_ids.copy())

    def get_segments(self) -> list[list[int]]:
        """Split by SEP token into segments."""
        segments = []
        current = []
        for tid in self.token_ids:
            if tid == SEP_ID:
                if current:
                    segments.append(current)
                    current = []
            else:
                current.append(tid)
        if current:
            segments.append(current)
        return segments

    def get_player_segment(self, player_id: int) -> list[int]:
        """Get tokens for a specific player."""
        segments = self.get_segments()
        if 0 <= player_id < len(segments):
            return segments[player_id]
        return []

    @classmethod
    def empty(cls) -> TokenList:
        """Create empty TokenList."""
        return cls(token_ids=[])

    @classmethod
    def from_ids(cls, ids: list[int]) -> TokenList:
        """Create from list of token IDs."""
        return cls(token_ids=ids)

    def __repr__(self) -> str:
        return f"TokenList({self.to_human()})"

    def __str__(self) -> str:
        return self.to_human()
