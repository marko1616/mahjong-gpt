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

from __future__ import annotations
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
    "CHI_UP",  # 35
    "CHI_MID",  # 36
    "CHI_DOWN",  # 37
    "PON",  # 38
    "KAN_OPEN",  # 39
    "KAN_ADD",  # 40
    "KAN_CLOSED",  # 41
    "PEI",  # 42
    "RIICHI",  # 43
    "RON",  # 44
    "TSUMO",  # 45
    "PASS",  # 46
]


class Token(BaseModel):
    """Single token with Pydantic support."""

    token_id: int = Field(ge=0, le=PAD_ID)

    def to_human(self) -> str:
        tid = self.token_id
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
        if HAND_MIN <= tid <= HAND_MAX:
            return tile34_to_str(tid - HAND_MIN)
        return f"<UNK:{tid}>"

    @classmethod
    def from_human(cls, s: str) -> Token:
        s = s.strip()
        if s == "<PAD>":
            return cls(token_id=PAD_ID)
        if s == "<SEP>":
            return cls(token_id=SEP_ID)
        if s.startswith("<P") and s.endswith(">"):
            idx = int(s[2:-1])
            return cls(token_id=PLAYER_MIN + idx)
        if s.startswith("D"):
            return cls(token_id=DISCARD_MIN + tile34_from_str(s[1:]))
        if s.startswith("[") and s.endswith("]"):
            action_name = s[1:-1]
            return cls(token_id=SPEC_MIN + SPEC_MAP.index(action_name))
        return cls(token_id=HAND_MIN + tile34_from_str(s))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Token):
            return self.token_id == other.token_id
        if isinstance(other, int):
            return self.token_id == other
        return False

    def __hash__(self) -> int:
        return hash(self.token_id)


class TokenList(BaseModel):
    """Token sequence as a Pydantic model."""

    tokens: list[Token] = Field(default_factory=list)

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
            return {"tokens": data.tokens}
        if isinstance(data, str):
            return {"tokens": cls._parse_human(data)}
        if isinstance(data, (list, tuple)):
            tokens = []
            for item in data:
                if isinstance(item, Token):
                    tokens.append(item)
                elif isinstance(item, int):
                    tokens.append(Token(token_id=item))
                elif isinstance(item, dict):
                    tokens.append(Token(**item))
                else:
                    raise ValueError(f"Invalid token type: {type(item)}")
            return {"tokens": tokens}
        raise ValueError(f"Cannot convert {type(data)} to TokenList")

    @classmethod
    def _parse_human(cls, s: str) -> list[Token]:
        """Parse human-readable string to token list."""
        tokens = []
        parts = s.replace("<SEP>", " <SEP> ").replace("<PAD>", " <PAD> ").split()
        for part in parts:
            if not part:
                continue
            if part in ("<SEP>", "<PAD>"):
                tokens.append(Token.from_human(part))
            elif part.startswith("[") and part.endswith("]"):
                tokens.append(Token.from_human(part))
            elif part.startswith("D"):
                tokens.append(Token.from_human(part))
            else:
                # parse tile sequence like "1m2m3p"
                tokens.extend(cls._parse_tiles(part))
        return tokens

    @staticmethod
    def _parse_tiles(s: str) -> list[Token]:
        """Parse tile string like '1m2m3p' to tokens."""
        tokens = []
        i = 0
        while i < len(s):
            if i + 1 < len(s) and s[i + 1] in "mpsz":
                tokens.append(Token(token_id=HAND_MIN + tile34_from_str(s[i : i + 2])))
                i += 2
            else:
                i += 1
        return tokens

    def to_ids(self) -> list[int]:
        """Convert to list of integer token IDs."""
        return [t.token_id for t in self.tokens]

    def to_human(self) -> str:
        """Convert to human-readable string."""
        parts = []
        current_tiles = []

        def flush_tiles():
            nonlocal current_tiles
            if current_tiles:
                parts.append("".join(current_tiles))
                current_tiles = []

        for tok in self.tokens:
            tid = tok.token_id
            if tid == SEP_ID:
                flush_tiles()
                parts.append("<SEP>")
                continue
            if HAND_MIN <= tid <= HAND_MAX:
                current_tiles.append(tile34_to_str(tid - HAND_MIN))
                continue

            flush_tiles()
            parts.append(tok.to_human())

        flush_tiles()
        return " ".join(parts)

    @classmethod
    def from_human(cls, s: str) -> TokenList:
        return cls.model_validate(s)

    def append(self, token: Token | int) -> None:
        """Append a token."""
        if isinstance(token, int):
            token = Token(token_id=token)
        self.tokens.append(token)

    def extend(self, tokens: Iterable[Token | int]) -> None:
        """Extend with multiple tokens."""
        for t in tokens:
            self.append(t)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int | slice) -> Token | list[Token]:
        return self.tokens[idx]

    def __iter__(self) -> Iterator[Token]:
        return iter(self.tokens)

    def __contains__(self, item: Token | int) -> bool:
        if isinstance(item, int):
            return any(t.token_id == item for t in self.tokens)
        return item in self.tokens

    def iter_tokens(self) -> Iterator[Token]:
        """Iterate over tokens."""
        return iter(self.tokens)

    def copy(self) -> TokenList:
        """Create a deep copy."""
        return TokenList(tokens=[Token(token_id=t.token_id) for t in self.tokens])

    def get_segments(self) -> list[list[Token]]:
        """Split by SEP token into segments."""
        segments = []
        current = []
        for tok in self.tokens:
            if tok.token_id == SEP_ID:
                if current:
                    segments.append(current)
                    current = []
            else:
                current.append(tok)
        if current:
            segments.append(current)
        return segments

    def get_player_segment(self, player_id: int) -> list[Token]:
        """Get tokens for a specific player."""
        segments = self.get_segments()
        if 0 <= player_id < len(segments):
            return segments[player_id]
        return []

    @classmethod
    def empty(cls) -> TokenList:
        """Create empty TokenList."""
        return cls(tokens=[])

    @classmethod
    def from_ids(cls, ids: list[int]) -> TokenList:
        """Create from list of token IDs."""
        return cls(tokens=[Token(token_id=i) for i in ids])

    def __repr__(self) -> str:
        return f"TokenList({self.to_human()})"

    def __str__(self) -> str:
        return self.to_human()
