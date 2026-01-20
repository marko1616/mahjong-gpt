import torch


def get_gaes(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lambd: float,
) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Tensor of shape [T] containing rewards.
        values: Tensor of shape [T] containing value estimates.
        dones: Tensor of shape [T] indicating terminal states.
        gamma: Discount factor.
        lambd: GAE smoothing parameter.

    Returns:
        advs: Tensor of shape [T] containing advantage estimates.
    """
    T = rewards.shape[0]
    device, dtype = rewards.device, rewards.dtype

    # Create masks: 0 if done, 1 otherwise
    masks = (1.0 - dones.to(dtype)).to(device)

    # Next values shifted by 1, padded with 0 at the end
    values_next = torch.cat(
        [values[1:], torch.zeros(1, device=device, dtype=dtype)], dim=0
    )

    # TD errors (deltas)
    deltas = rewards + gamma * values_next * masks - values

    advs = torch.zeros(T, device=device, dtype=dtype)
    gae = torch.zeros(1, device=device, dtype=dtype)

    # Calculate GAE in reverse order
    for t in reversed(range(T)):
        gae = deltas[t] + gamma * lambd * gae * masks[t]
        advs[t] = gae

    return advs


def get_n_tds(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    n: int,
) -> torch.Tensor:
    """
    Compute n-step Temporal Difference returns.

    Args:
        rewards: Tensor of shape [T].
        values: Tensor of shape [T].
        dones: Tensor of shape [T].
        gamma: Discount factor.
        n: Number of steps for TD.

    Returns:
        out: Tensor of shape [T] containing n-step returns.
    """
    T = rewards.shape[0]
    device, dtype = rewards.device, rewards.dtype

    out = torch.zeros(T, device=device, dtype=dtype)

    for t in range(T):
        ret = torch.zeros(1, device=device, dtype=dtype)
        discount = torch.ones(1, device=device, dtype=dtype)

        # Accumulate discounted rewards for n steps
        for k in range(n):
            if t + k >= T:
                break
            ret = ret + discount * rewards[t + k]
            if dones[t + k]:
                discount = torch.zeros(1, device=device, dtype=dtype)
                break
            discount = discount * gamma

        # Add bootstrapped value if not terminal and within bounds
        boot_index = t + n
        if boot_index < T and discount.item() != 0.0:
            ret = ret + discount * values[boot_index]

        out[t] = ret

    return out


def masked_logprob(
    logits: torch.Tensor, actions: torch.Tensor, action_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute log probability of selected actions with masking.

    Args:
        logits: Tensor of shape [B, Out].
        actions: Tensor of shape [B] containing action indices.
        action_mask: Tensor of shape [B, Out] (1 for valid, 0 for invalid).

    Returns:
        logp_all: Tensor of shape [B] containing log probabilities of selected actions.
    """
    logits = logits.float()

    # Apply mask: fill invalid actions with a very small number
    invalid = ~action_mask.to(torch.bool)
    logits = logits.masked_fill(invalid, -1e9)

    logp_all = torch.log_softmax(logits, dim=-1)

    # Gather log probs corresponding to the selected actions
    return logp_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
