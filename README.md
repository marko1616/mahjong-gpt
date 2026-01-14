# Riichi Mahjong Bot Based on GPT Model

[ English | [中文](README_zh.md) ]

This project implements a Riichi Mahjong intelligent agent based on reinforcement learning, using deep policy gradient methods (PPO algorithm) and GPT models for decision learning.

## Requirements

- Python 3.14 or higher
- PyTorch 2.9 or higher

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
   python -m src.trainer
   ```
   This will start the agent training process. Training logs and model weights will be automatically saved to the specified directories.

2. Monitor training:
   Use TensorBoard to view training progress and performance:
   ```bash
   tensorboard --logdir=runs
   ```

3. Configuration:
   Modify `src/config.py` to adjust hyperparameters. The configuration uses Pydantic for type-safe settings:
   ```python
   from src.config import get_default_config, get_eval_config, Config
   
   # Use default configuration
   config = get_default_config()
   
   # Get evaluation mode configuration
   config = get_eval_config()
   
   # Or load custom configuration from JSON
   config = Config.from_json(json_string)
   ```

## Project Structure

```
src/
├── trainer.py          # Training entry point
├── agent.py            # PPO algorithm agent implementation
├── model.py            # GPT model definition (based on minGPT)
├── config.py           # Hyperparameter configuration (Pydantic-based)
├── schedulers.py       # Learning rate and parameter schedulers
├── schemes.py          # Data structures (Trail, ReplayBuffer, etc.)
├── recorder.py         # Training metrics recording and logging
├── ckpt_manager.py     # Checkpoint manager (supports resumable training)
├── utils/
│   ├── ckpt_utils.py   # Checkpoint utilities (RNG state, atomic writes)
│   └── stats_utils.py  # Statistical utilities (CI bounds, running stats)
└── env/
    ├── env.py          # Main Mahjong environment implementation
    ├── constants.py    # Action space constants
    ├── tiles.py        # Tile conversion utilities
    ├── tokens.py       # Token vocabulary and TokenList class
    ├── hand.py         # Hand management with shanten calculation
    ├── player.py       # Player state management
    ├── wall.py         # Tile wall distribution
    ├── event_bus.py    # Pub-sub event system
    └── worker.py       # Async multiprocessing environment wrapper
```

## Checkpointing & Resumable Training

The project supports a complete checkpointing mechanism that allows resuming training from any checkpoint:

- Checkpoints save model weights, optimizer states, RNG states, and scheduler states
- Use `CkptManager` to manage multi-pass training
- Supports atomic writes to prevent checkpoint corruption from crashes

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