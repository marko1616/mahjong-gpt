# 基于GPT模型的立直麻将Bot

[ [English](README.md) | 中文 ]

本项目实现了一个基于强化学习的立直麻将智能体，使用深度策略梯度方法（PPO算法）和GPT模型来进行决策学习。

## 环境需求

- Python 3.11 或更高版本
- PyTorch 2.0 或更高版本

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

## 使用说明

1. 启动训练：
   ```bash
   python src/ppo.py
   ```
   这将启动代理的训练过程，训练日志和模型权重会自动保存在指定的目录中。

2. 监控训练：
   使用TensorBoard查看训练进度和性能：
   ```bash
   tensorboard --logdir=runs
   ```

3. 配置参数：
   修改 `src/config.py` 来调整超参数。配置使用 dataclass 实现类型安全的设置：
   ```python
   from config import get_default_config, get_custom_config
   
   # 使用默认配置
   config = get_default_config()
   
   # 或自定义配置
   config = get_custom_config(
       episodes=200,
       lr=1e-6,
       batch_size=16,
       device="cuda:0"
   )
   ```

## 项目结构

```
src/
├── ppo.py              # PPO算法实现和训练入口
├── model.py            # GPT模型定义（基于minGPT）
├── config.py           # 超参数配置（基于dataclass）
├── schedulers.py       # 学习率和参数调度器
├── utils/
│   └── stats_utils.py  # 统计工具（置信区间、滚动统计）
└── env/
    ├── __init__.py
    ├── env.py          # 麻将环境主实现
    ├── constants.py    # 动作空间常量
    ├── tiles.py        # 牌面转换工具
    ├── tokens.py       # 词表和TokenList类
    ├── hand.py         # 手牌管理与向听数计算
    ├── player.py       # 玩家状态管理
    ├── wall.py         # 牌山分发
    ├── event_bus.py    # 发布-订阅事件系统
    ├── reward_config.py# 奖励配置（基于pydantic）
    └── worker.py       # 异步多进程环境封装
```

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
