# 基于GPT模型的立直麻将Bot

[ [English](README.md) | 中文 ]

本项目实现了一个基于强化学习的立直麻将智能体，使用深度策略梯度方法（PPO算法）和GPT模型来进行决策学习。

## 环境需求

- Python 3.14 或更高版本
- PyTorch 2.9 或更高版本

## 安装

1. 克隆仓库到本地：
   ```bash
   git clone https://github.com/marko1616/mahjong_DRL
   ```

2. 进入项目目录：
   ```bash
   cd mahjong_DRL
   ```

3. 安装依赖：
   ```bash
   # 根据你的操作系统和CUDA版本安装PyTorch
   # 参见：https://pytorch.org/get-started/locally/
   
   pip install -r requirements.txt
   ```

## CLI 工具

项目根目录下的 `cli.py` 提供了一个交互式命令行工具，用于管理多阶段训练流程。

### 核心概念

- **Run（训练运行）**：一次完整的训练实验，由 `run_id` 标识，包含一个或多个 Pass
- **Pass（训练阶段）**：训练的一个阶段，每个 Pass 有独立的配置、检查点和状态。可以从零开始，也可以从其他 Pass 的检查点继承权重
- **Manifest（清单）**：记录整个 Run 的元信息，包括所有 Pass 的配置和状态

### 启动 CLI

```bash
python cli.py interactive
```

启动后会提示输入 `root_dir`（模型存储根目录）和 `run_id`（运行标识符），也可以通过环境变量预设：

```bash
export MAHJONG_GPT_ROOT=/mnt/models/mahjong-gpt
export MAHJONG_GPT_RUN_ID=model-00
python cli.py interactive
```

### 可用任务

| 任务 | 说明 |
|------|------|
| `status` | 显示当前 Run 的状态（各 Pass 进度、活跃 Pass 等） |
| `init-run` | 初始化新的 Run，创建 Manifest 和第一个 Pass |
| `set-active` | 设置活跃 Pass（TrainerRunner 从这里开始） |
| `append-pass` | 添加新 Pass，可选择从现有 Pass 的检查点继承 |
| `edit-config` | 在 `$EDITOR` 中编辑 Pass 配置（Pydantic 校验） |
| `reset-pass-state` | 重置 Pass 状态为 pending（不删除检查点） |
| `run` | 启动 TrainerRunner 执行活跃 Pass |

### 使用示例

**场景：从零开始训练，完成后用新超参继续训练**

```
# 1. 初始化 Run 和 pass-0
Choose a task: init-run
Run notes: First experiment
First pass name: pass-0
total_episodes for pass-0: 1000
→ Initialized run.

# 2. 启动训练
Choose a task: run
Run active pass 0 now? Yes
→ TrainerRunner 开始执行 pass-0

# 3. pass-0 完成后，添加 pass-1 并从 pass-0 继承权重
Choose a task: append-pass
Config source: Edit JSON in $EDITOR  # 修改学习率等参数
New pass name: pass-1-finetune
Bootstrap from existing pass checkpoint? Yes
Select source pass: 0 - pass-0 (completed)
Source episode (blank = latest): [回车]
init_mode: weights_only
→ Appended pass 1 and set active.

# 4. 继续训练
Choose a task: run
→ TrainerRunner 加载 pass-0 权重，执行 pass-1
```

**场景：查看训练状态**

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

## 项目结构

```
cli.py                  # 交互式工具
src/
├── trainer.py          # 训练入口
├── agent.py            # PPO算法智能体实现
├── model.py            # GPT模型定义（基于minGPT）
├── config.py           # 超参数配置（基于Pydantic）
├── schedulers.py       # 学习率和参数调度器
├── schemes.py          # 数据结构定义（Trail、ReplayBuffer等）
├── recorder.py         # 训练指标记录与日志
├── ckpt_manager.py     # 检查点管理器（支持断点续训）
├── utils/
│   ├── ckpt_utils.py   # 检查点工具（RNG状态、原子写入）
│   └── stats_utils.py  # 统计工具（置信区间、滚动统计）
└── env/
    ├── env.py          # 麻将环境主实现
    ├── constants.py    # 动作空间常量
    ├── tiles.py        # 牌面转换工具
    ├── tokens.py       # 词表和TokenList类
    ├── hand.py         # 手牌管理与向听数计算
    ├── player.py       # 玩家状态管理
    ├── wall.py         # 牌山分发
    ├── event_bus.py    # 发布-订阅事件系统
    └── worker.py       # 异步多进程环境封装
```

## 检查点与断点续训

项目支持完整的检查点机制，可以在训练中断后从任意检查点恢复：

- 检查点保存模型权重、优化器状态、RNG状态和调度器状态
- 使用 `CkptManager` 管理多轮训练（Pass）
- 支持原子写入，防止因崩溃导致的检查点损坏

## 动作空间

动作空间包含47个动作（索引0-46，其中0为未使用的填充位）：

| 动作ID | 描述 |
|--------|------|
| 1-34   | 打牌（tile34 = id - 1） |
| 35-37  | 吃（上吃/中吃/下吃） |
| 38     | 碰 |
| 39-41  | 杠（明杠/加杠/暗杠） |
| 42     | 拔北（三人麻将，保留） |
| 43     | 立直 |
| 44     | 荣和 |
| 45     | 自摸 |
| 46     | 跳过 |