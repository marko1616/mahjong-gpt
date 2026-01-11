# Riichi Mahjong Bot Based on GPT Model

[ English | [中文](README_zh.md) ]

This project implements a Riichi Mahjong intelligent agent based on reinforcement learning, using deep policy gradient methods (PPO algorithm) and GPT models for decision learning.

## Requirements

- Python 3.11 or higher
- PyTorch 2.0 or higher

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/marko1616/mahjong_DRL
   ```

2. Enter the project directory:
   ```bash
   cd mahjong_DRL
   ```

3. Install dependencies:
   ```bash
   # Install PyTorch according to your OS and CUDA version
   # See: https://pytorch.org/get-started/locally/
   
   pip install -r requirements.txt
   ```

## Usage

1. Start training:
   ```bash
   python src/ppo.py
   ```
   This will start the agent training process. Training logs and model weights will be automatically saved to the specified directories.

2. Monitor training:
   Use TensorBoard to view training progress and performance:
   ```bash
   tensorboard --logdir=runs
   ```

3. Configuration:
   Modify `src/config.py` to adjust hyperparameters. The configuration uses dataclasses for type-safe settings:
   ```python
   from config import get_default_config, get_custom_config
   
   # Use default configuration
   config = get_default_config()
   
   # Or customize
   config = get_custom_config(
       episodes=200,
       lr=1e-6,
       batch_size=16,
       device="cuda:0"
   )
   ```

## Project Structure

```
src/
├── ppo.py              # PPO algorithm implementation and training entry point
├── model.py            # GPT model definition (based on minGPT)
├── config.py           # Hyperparameter configuration (dataclass-based)
├── schedulers.py       # Learning rate and parameter schedulers
├── utils/
│   └── stats_utils.py  # Statistical utilities (CI bounds, running stats)
└── env/
    ├── __init__.py
    ├── env.py          # Main Mahjong environment implementation
    ├── constants.py    # Action space constants
    ├── tiles.py        # Tile conversion utilities
    ├── tokens.py       # Token vocabulary and TokenList class
    ├── hand.py         # Hand management with shanten calculation
    ├── player.py       # Player state management
    ├── wall.py         # Tile wall distribution
    ├── event_bus.py    # Pub-sub event system
    ├── reward_config.py# Reward configuration (pydantic-based)
    └── worker.py       # Async multiprocessing environment wrapper
```

## Action Space

The action space consists of 47 actions (indices 0-46, where 0 is unused padding):

| Action ID | Description |
|-----------|-------------|
| 1-34      | Discard tile (tile34 = id - 1) |
| 35-37     | Chi (upper/middle/lower) |
| 38        | Pon |
| 39-41     | Kan (open/add/closed) |
| 42        | Pei (3-player, reserved) |
| 43        | Riichi |
| 44        | Ron |
| 45        | Tsumo |
| 46        | Pass |
