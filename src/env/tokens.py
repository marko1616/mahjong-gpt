"""
Token primitives for MahjongEnv history.

Vocab layout (total = 1 + 46 + 34 + 1 = 82):
  0            : [SEP]
  1..46        : ACTION tokens (exactly env action_id space)
      1..34    : discard tile34 (tile34 = id-1)
      35..37   : CHI_UP / CHI_MID / CHI_DOWN
      38       : PON
      39..41   : KAN_OPEN / KAN_ADD / KAN_CLOSED
      42       : PEI (3p reserved)
      43       : RIICHI
      44       : RON
      45       : TSUMO
      46       : PASS
  47..80       : HAND tile tokens (tile34 = id-47)   # env doesn't emit, model may use
  81           : [PAD]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union
import json

from .tiles import tile34_to_str


# ---- vocab ids (hard-coded, no constants import) ----

SEP_ID = 0

ACTION_MIN = 1
ACTION_MAX = 46
NUM_ACTIONS = ACTION_MAX - ACTION_MIN + 1  # 46

DISCARD_MIN = 1
DISCARD_MAX = 34  # discard id 1..34 => tile34 = id-1

HAND_BASE = ACTION_MAX + 1  # 47
HAND_MIN = HAND_BASE  # 47
HAND_MAX = HAND_BASE + 34 - 1  # 80
NUM_HAND_TOKENS = 34

PAD_ID = HAND_MAX + 1  # 81

VOCAB_SIZE = PAD_ID + 1  # 82


# action ids (still hard-coded, matching your old layout)
CHI_UP = 35
CHI_MID = 36
CHI_DOWN = 37
PON = 38
KAN_OPEN = 39
KAN_ADD = 40
KAN_CLOSED = 41
PEI = 42
RIICHI = 43
RON = 44
TSUMO = 45
PASS = 46


_ACTION_NAME: Dict[int, str] = {
    CHI_UP: "CHI_UP",
    CHI_MID: "CHI_MID",
    CHI_DOWN: "CHI_DOWN",
    PON: "PON",
    KAN_OPEN: "KAN_OPEN",
    KAN_ADD: "KAN_ADD",
    KAN_CLOSED: "KAN_CLOSED",
    PEI: "PEI",
    RIICHI: "RIICHI",
    RON: "RON",
    TSUMO: "TSUMO",
    PASS: "PASS",
}


# ---- Token types ----


class Token(ABC):
    @property
    @abstractmethod
    def token_id(self) -> int: ...

    @property
    def id(self) -> int:
        # action_id == token_id in this env vocabulary
        return int(self.token_id)

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]: ...

    @abstractmethod
    def to_human(self) -> str: ...

    def __int__(self) -> int:
        return int(self.token_id)

    def __index__(self) -> int:
        return int(self.token_id)

    def __str__(self) -> str:
        return self.to_human()


@dataclass(frozen=True, slots=True)
class SpecialToken(Token):
    _id: int
    name: str

    @property
    def token_id(self) -> int:
        return int(self._id)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "special", "name": self.name, "id": int(self.token_id)}

    def to_human(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class DiscardActionToken(Token):
    """ACTION id 1..34, tile34 = id-1"""

    tile34: int  # 0..33

    @property
    def token_id(self) -> int:
        return int(self.tile34) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "action_discard",
            "tile34": int(self.tile34),
            "id": int(self.token_id),
        }

    def to_human(self) -> str:
        return f"DISCARD {tile34_to_str(self.tile34)}"


@dataclass(frozen=True, slots=True)
class NamedActionToken(Token):
    """ACTION id 35..46"""

    _id: int

    @property
    def token_id(self) -> int:
        return int(self._id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "action",
            "id": int(self.token_id),
            "name": _ACTION_NAME.get(int(self._id), "UNKNOWN"),
        }

    def to_human(self) -> str:
        return _ACTION_NAME.get(int(self._id), f"ACTION({int(self._id)})")


@dataclass(frozen=True, slots=True)
class HandToken(Token):
    """HAND token id 47..80, tile34 = id-47"""

    tile34: int

    @property
    def token_id(self) -> int:
        return HAND_BASE + int(self.tile34)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "hand", "tile34": int(self.tile34), "id": int(self.token_id)}

    def to_human(self) -> str:
        return f"HAND {tile34_to_str(self.tile34)}"


# ---- cached singletons ----

TOK_SEP = SpecialToken(SEP_ID, "[SEP]")
TOK_PAD = SpecialToken(PAD_ID, "[PAD]")

_DISCARD_CACHE: tuple[DiscardActionToken, ...] = tuple(
    DiscardActionToken(t) for t in range(34)
)
_HAND_CACHE: tuple[HandToken, ...] = tuple(HandToken(t) for t in range(34))

# named action singletons
_NAMED_ACTION_CACHE: Dict[int, NamedActionToken] = {
    i: NamedActionToken(i) for i in range(35, 47)
}


def token_for_id(token_id: int) -> Token:
    """Decode any vocab id into a semantic Token."""
    tid = int(token_id)

    if tid == SEP_ID:
        return TOK_SEP
    if tid == PAD_ID:
        return TOK_PAD

    if DISCARD_MIN <= tid <= DISCARD_MAX:
        return _DISCARD_CACHE[tid - 1]

    if 35 <= tid <= 46:
        return _NAMED_ACTION_CACHE[tid]

    if HAND_MIN <= tid <= HAND_MAX:
        return _HAND_CACHE[tid - HAND_BASE]

    raise ValueError(f"unknown token id: {tid}")


def action_token(action_id: int) -> Token:
    """Action ids are 1..46 (same as vocab ids)."""
    aid = int(action_id)
    if not (ACTION_MIN <= aid <= ACTION_MAX):
        raise ValueError(f"action_id out of range: {aid}")
    return token_for_id(aid)


def hand_token(tile34: int) -> HandToken:
    t = int(tile34)
    if not (0 <= t < 34):
        raise ValueError(f"tile34 out of range: {t}")
    return _HAND_CACHE[t]


def hand_tokens_from_counts(tile34_counts: Sequence[int]) -> List[int]:
    """
    Expand a 34-length tile34 count vector into HAND token ids,
    repeating each tile token by its count.
    """
    if len(tile34_counts) != 34:
        raise ValueError(f"expected 34 counts, got {len(tile34_counts)}")
    out: List[int] = []
    for t, c in enumerate(tile34_counts):
        if c:
            out.extend([HAND_BASE + t] * int(c))
    return out


def token_from_dict(data: Dict[str, Any]) -> Token:
    t = data.get("type")
    if t == "special":
        return token_for_id(int(data["id"]))
    if t == "action_discard":
        return _DISCARD_CACHE[int(data["tile34"])]
    if t == "action":
        return token_for_id(int(data["id"]))
    if t == "hand":
        return _HAND_CACHE[int(data["tile34"])]
    raise ValueError(f"unknown token dict type: {t!r}")


class TokenList:
    """
    Stores semantic Token objects.
    Iteration yields legacy ints so `list(TokenList)` => list[int].
    """

    def __init__(self, init: Optional[Iterable[Union[int, Token]]] = None) -> None:
        self._tokens: List[Token] = []
        if init is not None:
            self.extend(init)

    def clear(self) -> None:
        self._tokens.clear()

    def append(self, item: Union[int, Token]) -> None:
        if isinstance(item, Token):
            self._tokens.append(item)
        else:
            self._tokens.append(token_for_id(int(item)))

    def extend(self, items: Iterable[Union[int, Token]]) -> None:
        for it in items:
            self.append(it)

    def iter_tokens(self) -> Iterator[Token]:
        return iter(self._tokens)

    def to_int_list(self) -> List[int]:
        return [t.token_id for t in self._tokens]

    def to_json_obj(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self._tokens]

    def to_json(self) -> str:
        return json.dumps(self.to_json_obj(), ensure_ascii=False)

    @classmethod
    def from_int_list(cls, ids: Sequence[int]) -> "TokenList":
        return cls(ids)

    @classmethod
    def from_json(cls, payload: Union[str, Sequence[Dict[str, Any]]]) -> "TokenList":
        data = json.loads(payload) if isinstance(payload, str) else payload
        out = cls()
        for d in data:
            out.append(token_from_dict(dict(d)))
        return out

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, idx: int) -> Token:
        return self._tokens[idx]

    def __iter__(self) -> Iterator[int]:
        for t in self._tokens:
            yield t.token_id

    def __str__(self) -> str:
        return " ".join(t.to_human() for t in self._tokens)

    def __repr__(self) -> str:
        return f"TokenList(n={len(self._tokens)})"
