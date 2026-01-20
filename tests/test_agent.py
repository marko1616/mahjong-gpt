import pytest
import torch
import random

from src.config import Config
from src.schemes import Trail, ReplayBuffer
from src.env.env import MahjongEnv


@pytest.fixture
def light_config():
    """Create a minimal lightweight configuration for fast testing (CPU only)."""
    conf = Config()

    conf.system.device = "cpu"
    conf.system.amp_enable = False
    conf.system.num_workers = 2

    conf.model.layer_policy = 1
    conf.model.layer_value = 1
    conf.model.d_model = 16
    conf.model.nhead = 2
    conf.model.vocab_size = 100

    conf.training.batch_size = 2
    conf.training.sample_trail_count = 4
    conf.training.epochs_per_update = 1
    conf.training.replay_buffer_size = 100

    conf.env.env_num = 2
    conf.evalu.memget_num_per_update = 8

    return conf


def pick_random_legal_action(rng: random.Random, action_mask):
    """Pick a random legal action from the action mask."""
    legal = [i for i, v in enumerate(action_mask) if v == 1]
    return rng.choice(legal)


def create_trails_from_env(
    count: int, seed: int = 42, max_steps: int = 1024
) -> list[Trail]:
    """Create trails by running actual games with fixed seed.

    This function simulates the behavior of get_memory_async by creating
    separate trails for each of the 4 players in a mahjong game.
    """
    rng = random.Random(seed)
    trails = []

    games_needed = (count + 3) // 4  # Each game produces up to 4 trails

    for game_idx in range(games_needed):
        game_seed = seed + game_idx
        env = MahjongEnv(seed=game_seed)
        obs, reward, done, info = env.reset()

        # Create separate trails for each of the 4 players (like get_memory_async)
        memories = [Trail() for _ in range(4)]
        last_hands = [None] * 4
        no_memory_index = []

        steps = 0
        while not done and steps < max_steps:
            player_index = obs["seat"]
            hand = obs["hand"]
            history = obs["tokens"]
            action_mask = info["action_mask"]

            # Track last hand for terminal state
            last_hands[player_index] = hand

            # Store state for this player (like get_memory_async)
            memories[player_index].states.append(history)
            memories[player_index].action_masks.append(list(action_mask))

            # Select action
            action = pick_random_legal_action(rng, action_mask)

            # Step environment
            next_obs, reward, done, next_info = env.step(action)

            # Record action, reward, done for this player
            memories[player_index].actions.append(int(action))
            memories[player_index].rewards.append(float(reward))
            memories[player_index].dones.append(bool(done))
            memories[player_index].info.append(dict(next_info) if next_info else {})

            # Apply reward_update to previous step (like get_memory_async)
            if len(memories[player_index].rewards) >= 2:
                memories[player_index].rewards[-2] += float(
                    next_info.get("reward_update", 0.0)
                )

            obs, info = next_obs, next_info
            steps += 1

        # Handle terminal state for all players (like get_memory_async)
        if done:
            terminal_hist = obs["tokens"]
            for idx, memory in enumerate(memories):
                if len(memory.rewards) == 0:
                    no_memory_index.append(idx)
                    continue
                # Add terminal state
                memory.states.append(terminal_hist)
                # Mark last done as True
                memory.dones[-1] = True

        # Collect valid trails from this game
        for idx, memory in enumerate(memories):
            if idx not in no_memory_index and memory.length > 0:
                trails.append(memory)
                if len(trails) >= count:
                    return trails

    return trails


def test_agent_init(light_config):
    """Test Agent initialization."""
    from src.agent import Agent

    agent = Agent(light_config)

    assert agent.policy_model is not None
    assert agent.value_model is not None
    assert len(agent.workers) == light_config.system.num_workers
    assert agent.optimizer_policy is not None
    assert agent.optimizer_value is not None

    del agent


@pytest.mark.asyncio
async def test_agent_get_memory_async(light_config):
    """Test asynchronous memory collection with real environment."""
    from src.agent import Agent

    light_config.system.num_workers = 1
    agent = Agent(light_config)

    initial_buffer_len = len(agent.replay_buffer)

    rewards_collected = []

    def callback(rewards):
        rewards_collected.append(rewards)

    await agent._worker_task(0, agent.workers[0], count=1, call_back=callback)

    assert len(agent.replay_buffer) >= initial_buffer_len

    del agent


def test_agent_update(light_config):
    """Test PPO update loop with trails from real environment."""
    from src.agent import Agent

    agent = Agent(light_config)

    # Create trails from real environment
    trails = create_trails_from_env(4, seed=42)

    initial_policy_params = [p.clone() for p in agent.policy_model.parameters()]
    initial_value_params = [p.clone() for p in agent.value_model.parameters()]

    agent.update(trails)

    policy_changed = any(
        not torch.equal(p1, p2)
        for p1, p2 in zip(initial_policy_params, agent.policy_model.parameters())
    )
    value_changed = any(
        not torch.equal(p1, p2)
        for p1, p2 in zip(initial_value_params, agent.value_model.parameters())
    )

    assert policy_changed or value_changed

    del agent


def test_save_weights(light_config, tmp_path):
    """Test model weights saving."""
    from src.agent import Agent

    agent = Agent(light_config)
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()

    agent.save_weights(ckpt_dir)

    assert (ckpt_dir / "policy.safetensors").exists()
    assert (ckpt_dir / "value.safetensors").exists()

    del agent


def test_load_weights(light_config, tmp_path):
    """Test model weights loading."""
    from src.agent import Agent

    agent = Agent(light_config)
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()

    agent.save_weights(ckpt_dir)

    original_policy_state = {
        k: v.clone() for k, v in agent.policy_model.state_dict().items()
    }
    original_value_state = {
        k: v.clone() for k, v in agent.value_model.state_dict().items()
    }

    agent2 = Agent(light_config)
    agent2.load_weights(ckpt_dir)

    for key in original_policy_state:
        assert torch.equal(
            original_policy_state[key], agent2.policy_model.state_dict()[key]
        )
    for key in original_value_state:
        assert torch.equal(
            original_value_state[key], agent2.value_model.state_dict()[key]
        )

    del agent
    del agent2


def test_save_replay(light_config, tmp_path):
    """Test replay buffer saving with trails from real environment."""
    from src.agent import Agent

    agent = Agent(light_config)
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()

    trails = create_trails_from_env(3, seed=42)
    for trail in trails:
        agent.replay_buffer.add(trail)

    agent.replay_buffer.save(ckpt_dir / "replay.json")

    assert (ckpt_dir / "replay.json").exists()

    del agent


def test_load_replay(light_config, tmp_path):
    """Test replay buffer loading."""
    from src.agent import Agent

    agent = Agent(light_config)
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()

    trails = create_trails_from_env(3, seed=42)
    for trail in trails:
        agent.replay_buffer.add(trail)

    original_len = len(agent.replay_buffer)
    agent.replay_buffer.save(ckpt_dir / "replay.json")

    loaded_buffer = ReplayBuffer.load(ckpt_dir / "replay.json")

    assert len(loaded_buffer) == original_len

    del agent


def test_export_optim_state(light_config):
    """Test optimizer state export."""
    from src.agent import Agent

    agent = Agent(light_config)

    state = agent.export_optim_state()

    assert "optimizer_policy" in state
    assert "optimizer_value" in state

    del agent


def test_export_agent_rng_state(light_config):
    """Test agent RNG state export."""
    from src.agent import Agent

    agent = Agent(light_config)

    state = agent.export_agent_rng_state()

    assert "python_rng_state" in state
    assert "current_base_seed" in state
    assert "seed_count" in state

    del agent


def test_export_worker_seeds(light_config):
    """Test worker seeds export."""
    from src.agent import Agent

    agent = Agent(light_config)

    seeds = agent.export_worker_seeds()

    assert len(seeds) == light_config.system.num_workers

    del agent
