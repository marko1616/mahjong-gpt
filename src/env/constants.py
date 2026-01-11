"""
Constants for the Mahjong environment.
"""

ACTIONS_PER_SEAT = 46
NUM_SEATS = 4

# Local action ids (1-based within a seat)
DISCARD_MIN = 1
DISCARD_MAX = 34

CHI_UP = 35  # take lowest tile (e.g., 2-3 eat 1)
CHI_MID = 36  # take middle tile (e.g., 1-3 eat 2)
CHI_DOWN = 37  # take highest tile (e.g., 1-2 eat 3)

PON = 38

KAN_OPEN = 39  # minkan (open kan from discard)
KAN_ADD = 40  # kakan (add kan to an existing pon)
KAN_CLOSED = 41  # ankan

PEI = 42  # reserved (3-player)
RIICHI = 43
RON = 44
TSUMO = 45
PASS = 46

PLAYER0 = 47
PLAYER1 = 48
PLAYER2 = 49
PLAYER3 = 50

PLAYER_TOKENS = [PLAYER0, PLAYER1, PLAYER2, PLAYER3]
