from pydantic import BaseModel, Field


class RewardConfig(BaseModel):
    """
    All rewards/penalties for the environment.

    Keep this centralized so tuning does not require touching game logic.
    """

    reward_weight_shanten: float = Field(default=30.0)
    penalty_ava_num: float = Field(default=1.2)
    score_weight: float = Field(default=0.1)
    reward_riichi: float = Field(default=6.0)
    reward_no_yaku: float = Field(default=-20.0)
    reward_open_tanyao: float = Field(default=-5.0)
