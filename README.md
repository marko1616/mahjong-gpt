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

## CLI Tool

The `cli.py` at the project root provides an interactive command-line tool for managing multi-pass training workflows.

### Core Concepts

- **Run**: A complete training experiment identified by `run_id`, containing one or more Passes
- **Pass**: A training phase with its own config, checkpoints, and state. Can start from scratch or inherit weights from another Pass's checkpoint
- **Manifest**: Metadata file tracking all Passes' configurations and states within a Run

### Starting the CLI

```bash
python cli.py interactive
```

You'll be prompted for `root_dir` (model storage root) and `run_id` (run identifier). These can also be preset via environment variables:

```bash
export MAHJONG_GPT_ROOT=/mnt/models/mahjong-gpt
export MAHJONG_GPT_RUN_ID=model-00
python cli.py interactive
```

### Available Tasks

| Task | Description |
|------|-------------|
| `status` | Show current Run status (Pass progress, active Pass, etc.) |
| `init-run` | Initialize a new Run with Manifest and first Pass |
| `set-active` | Set the active Pass (TrainerRunner starts here) |
| `append-pass` | Add a new Pass, optionally bootstrapping from existing checkpoint |
| `edit-config` | Edit Pass config in `$EDITOR` (Pydantic validation) |
| `reset-pass-state` | Reset Pass state to pending (checkpoints preserved) |
| `run` | Launch TrainerRunner on the active Pass |

### Usage Examples

**Scenario: Train from scratch, then continue with new hyperparameters**

```
# 1. Initialize Run and pass-0
Choose a task: init-run
Run notes: First experiment
First pass name: pass-0
total_episodes for pass-0: 1000
→ Initialized run.

# 2. Start training
Choose a task: run
Run active pass 0 now? Yes
→ TrainerRunner begins pass-0

# 3. After pass-0 completes, add pass-1 inheriting weights from pass-0
Choose a task: append-pass
Config source: Edit JSON in $EDITOR  # modify learning rate, etc.
New pass name: pass-1-finetune
Bootstrap from existing pass checkpoint? Yes
Select source pass: 0 - pass-0 (completed)
Source episode (blank = latest): [Enter]
init_mode: weights_only
→ Appended pass 1 and set active.

# 4. Continue training
Choose a task: run
→ TrainerRunner loads pass-0 weights, executes pass-1
```

**Scenario: Check training status**

```
Choose a task: status

╭─────────────────────────────────────────────────────────╮
│ Run  run_id=model-00  active_pass_id=1  passes=2        │
╰─────────────────────────────────────────────────────────╯
┌─────────┬────────────────┬───────────┬─────────┬──────────┬─────────────┬──────────────────────┬───────────────────────┐
│ pass_id │ name           │ status    │ curr_ep │ total_ep │ best_metric │ last_ckpt_dir        │ init_from             │
├─────────┼────────────────┼───────────┼─────────┼──────────┼─────────────┼──────────────────────┼───────────────────────┤
│ 0       │ pass-0         │ completed │ 1000    │ 1000     │ 0.4521      │ pass_0/ckpt_ep1000   │ -                     │
│ 1       │ pass-1-finetune│ running   │ 350     │ 500      │ 0.4892      │ pass_1/ckpt_ep350    │ 0:latest (weights_only)│
└─────────┴────────────────┴───────────┴─────────┴──────────┴─────────────┴──────────────────────┴───────────────────────┘
```

## Project Structure

```
cli.py                  # interactive cli tool
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