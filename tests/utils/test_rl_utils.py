import torch
import pytest
import math
from src.utils.rl_utils import get_gaes, get_n_tds, masked_logprob


@pytest.fixture
def sample_data():
    rewards = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    values = torch.tensor([0.5, 0.2, 0.8], dtype=torch.float32)
    dones = torch.tensor([False, False, True], dtype=torch.bool)
    return rewards, values, dones


def test_get_gae(sample_data):
    rewards, values, dones = sample_data
    gamma = 0.9
    lambd = 0.8

    # Manual Calculation:
    # T=2 (Done):
    #   delta_2 = r_2 + g*v_3*m_2 - v_2 = 1.0 + 0 - 0.8 = 0.2
    #   gae_2 = delta_2 = 0.2
    # T=1:
    #   delta_1 = r_1 + g*v_2*m_1 - v_1 = 0.0 + 0.9*0.8*1 - 0.2 = 0.52
    #   gae_1 = delta_1 + g*l*gae_2*m_1 = 0.52 + 0.9*0.8*0.2*1 = 0.52 + 0.144 = 0.664
    # T=0:
    #   delta_0 = r_0 + g*v_1*m_0 - v_0 = 1.0 + 0.9*0.2*1 - 0.5 = 1.0 + 0.18 - 0.5 = 0.68
    #   gae_0 = delta_0 + g*l*gae_1*m_0 = 0.68 + 0.9*0.8*0.664 = 0.68 + 0.47808 = 1.15808

    advs = get_gaes(rewards, values, dones, gamma, lambd)

    expected = torch.tensor([1.15808, 0.664, 0.2], dtype=torch.float32)
    assert torch.allclose(advs, expected, atol=1e-4)


def test_get_n_tds(sample_data):
    rewards, values, dones = sample_data
    # rewards: [1.0, 0.0, 1.0]
    # values:  [0.5, 0.2, 0.8]
    # dones:   [0,   0,   1]
    gamma = 0.9
    n = 2

    # Manual Calculation (n=2):
    # T=0: r_0 + g*r_1 + g^2 * v_2 (since done at T=2 is irrelevant for T=0's 2-step lookahead landing on T=2?)
    #      Wait, logic says:
    #      k=0: ret += 1.0 * 1.0 = 1.0
    #      k=1: ret += 0.9 * 0.0 = 1.0. done[1] is False. discount becomes 0.81.
    #      boot_index = 0 + 2 = 2.
    #      ret += 0.81 * v_2 = 1.0 + 0.81 * 0.8 = 1.0 + 0.648 = 1.648

    # T=1:
    #      k=0: ret += 1.0 * 0.0 = 0.0
    #      k=1: ret += 0.9 * 1.0 = 0.9. done[2] is True -> discount becomes 0.
    #      boot_index = 1 + 2 = 3 (Out of bounds).
    #      ret = 0.9

    # T=2:
    #      k=0: ret += 1.0 * 1.0 = 1.0. done[2] is True -> discount becomes 0.
    #      boot_index = 4 (Out of bounds).
    #      ret = 1.0

    tds = get_n_tds(rewards, values, dones, gamma, n)
    expected = torch.tensor([1.648, 0.9, 1.0], dtype=torch.float32)
    assert torch.allclose(tds, expected, atol=1e-4)


def test_masked_logprob():
    # Batch size 2, 3 actions
    logits = torch.tensor([[10.0, 10.0, 0.0], [5.0, 5.0, 5.0]])
    actions = torch.tensor([0, 2])
    # Mask out the middle action for first item, and first action for second item
    mask = torch.tensor([[1, 0, 1], [0, 1, 1]])

    # Item 0: mask [1, 0, 1]. logits become [10, -inf, 0].
    # Softmax: exp(10) / (exp(10) + exp(0)) ~= 1.0 / 1.0 ~= 1 (approx)
    # exact: log(e^10 / (e^10 + 1)) = 10 - log(e^10 + 1)

    # Item 1: mask [0, 1, 1]. logits become [-inf, 5, 5].
    # Softmax: exp(5)/(exp(5)+exp(5)) = 0.5. log(0.5) = -0.6931...

    log_probs = masked_logprob(logits, actions, mask)

    val0 = 10.0 - math.log(math.exp(10.0) + math.exp(0.0))
    val1 = math.log(0.5)

    expected = torch.tensor([val0, val1])
    assert torch.allclose(log_probs, expected, atol=1e-4)


def test_masked_logprob_shapes():
    B, Out = 4, 10
    logits = torch.randn(B, Out)
    actions = torch.randint(0, Out, (B,))
    mask = torch.ones(B, Out)

    log_probs = masked_logprob(logits, actions, mask)
    assert log_probs.shape == (B,)
