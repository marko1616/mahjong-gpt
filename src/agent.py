import math
import random
import asyncio
from typing import Callable, Iterable, Any
from pathlib import Path

from rich import print
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
)

import torch
from torch import optim
from torch import nn
from torch.amp import autocast, GradScaler


from safetensors.torch import load_file, save_file

from .env.worker import AsyncMahjongEnv

from .env.tokens import TokenList, SEP_ID, HAND_MIN

from .model import GPTModel
from .config import Config
from .schedulers import Scheduler
from .schemes import Trail, ReplayBuffer


class Agent:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.tc = config.training
        self.ec = config.env
        self.tgc = config.target
        self.mc = config.model
        self.sc = config.system
        self.evc = config.evalu

        self.gamma = self.tgc.gamma
        self.eps_clip = self.tc.clip_epsilon
        self.K_epochs = self.tc.epochs_per_update
        self.weight_policy = self.tc.weight_policy
        self.weight_value = self.tc.weight_value
        self.beta = self.tc.beta
        self.lambd = self.tgc.lambd

        self.current_base_seed = self.ec.init_env_seed
        self.rng = random.Random(self.current_base_seed)

        self.seed_count = 0
        self.max_trail_len = 0

        self.amp_ctx = autocast(
            device_type=self.sc.amp_device_type,
            enabled=self.sc.amp_enable,
            dtype=self.sc.amp_dtype,
        )
        self.amp_scaler = GradScaler(enabled=self.sc.amp_enable)

        pconfig = GPTModel.get_default_config()
        pconfig.n_layer = self.mc.layer_policy
        pconfig.n_head = self.mc.nhead
        pconfig.n_embd = self.mc.d_model
        pconfig.vocab_size = self.mc.vocab_size
        pconfig.out_size = 46
        pconfig.block_size = 512

        self.policy_model = GPTModel(pconfig)

        self.policy_model_old = GPTModel(pconfig)
        self.policy_model_old.load_state_dict(self.policy_model.state_dict())
        self.policy_model_old.eval()

        self.optimizer_policy = optim.AdamW(
            self.policy_model.parameters(), lr=self.tc.lr_policy
        )

        vconfig = GPTModel.get_default_config()
        vconfig.n_layer = self.mc.layer_value
        vconfig.n_head = self.mc.nhead
        vconfig.n_embd = self.mc.d_model
        vconfig.vocab_size = self.mc.vocab_size
        vconfig.out_size = 1
        vconfig.block_size = 512

        self.value_model = GPTModel(vconfig)

        self.value_model_old = GPTModel(vconfig)
        self.value_model_old.load_state_dict(self.value_model.state_dict())
        self.value_model_old.eval()

        self.optimizer_value = optim.AdamW(
            self.value_model.parameters(), lr=self.tc.lr_value
        )
        self.MseLoss = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(self.tc.replay_buffer_size)

        self.policy_model.to(self.sc.device, self.sc.dtype)
        self.policy_model_old.to(self.sc.device, self.sc.dtype)
        self.value_model.to(self.sc.device, self.sc.dtype)
        self.value_model_old.to(self.sc.device, self.sc.dtype)

        self.worker_count = self.sc.num_workers
        self.worker_seeds = [
            self.rng.randint(0, 1_000_000_000) for i in range(self.worker_count)
        ]
        self.workers = [
            AsyncMahjongEnv(seed=self.worker_seeds[i], reward_config=self.config.reward)
            for i in range(self.worker_count)
        ]
        print(f"Initialized {self.worker_count} async environment workers.")

    def save_weights(self, ckpt_dir: Path) -> None:
        """Save model weights needed for exact resume into a checkpoint directory."""
        save_file(
            self.policy_model.state_dict(),
            str(ckpt_dir / "policy.safetensors"),
        )
        save_file(
            self.value_model.state_dict(),
            str(ckpt_dir / "value.safetensors"),
        )

    def load_weights(self, ckpt_dir: Path, *, strict: bool = True) -> None:
        """Load model weights from a checkpoint directory."""
        p_old = ckpt_dir / "policy.safetensors"
        v_old = ckpt_dir / "value.safetensors"

        print(f"Loading policy weights from {p_old}")
        if p_old.exists():
            self.policy_model_old.load_state_dict(load_file(str(p_old)), strict=strict)
            # Keep live model consistent unless you explicitly want divergence.
            self.policy_model.load_state_dict(
                self.policy_model_old.state_dict(), strict=strict
            )

        print(f"Loading value weights from {v_old}")
        if v_old.exists():
            self.value_model_old.load_state_dict(load_file(str(v_old)), strict=strict)
            self.value_model.load_state_dict(
                self.value_model_old.state_dict(), strict=strict
            )

    def export_optim_state(self) -> dict[str, Any]:
        """Export optimizer states for exact resume."""
        return {
            "optimizer_policy": self.optimizer_policy.state_dict(),
            "optimizer_value": self.optimizer_value.state_dict(),
        }

    def load_optim_state(self, state: dict[str, Any]) -> None:
        """Restore optimizer states exported by export_optim_state()."""
        print("Restoring optimizer states.")
        if "optimizer_policy" in state:
            self.optimizer_policy.load_state_dict(state["optimizer_policy"])
        if "optimizer_value" in state:
            self.optimizer_value.load_state_dict(state["optimizer_value"])

    def export_amp_state(self) -> dict[str, Any]:
        """Export AMP scaler state (empty if AMP is disabled)."""
        if getattr(self.config.system, "amp_enable", False):
            return {"amp_scaler": self.amp_scaler.state_dict()}
        return {}

    def load_amp_state(self, state: dict[str, Any]) -> None:
        """Restore AMP scaler state if present."""
        if "amp_scaler" in state and getattr(self.config.system, "amp_enable", False):
            self.amp_scaler.load_state_dict(state["amp_scaler"])

    def export_scheduler_states(self) -> dict[str, dict[str, Any]]:
        """Export all scheduler states required for episode-wise resume."""
        return {
            "action_epsilon": self.config.training.action_epsilon_scheduler.to_state(),
            "alpha": self.config.target.alpha_scheduler.to_state(),
            "n_td": self.config.target.n_td_scheduler.to_state(),
        }

    def restore_scheduler_states(self, states: dict[str, dict[str, Any]]) -> None:
        """Restore scheduler states and rebuild live scheduler instances."""
        tc = self.config.training
        tgc = self.config.target

        if "action_epsilon" in states:
            tc.action_epsilon_state = states["action_epsilon"]
            tc._action_epsilon_scheduler = tc.action_epsilon_config.build(
                episodes_override=tc.pass_n_episodes,
                state=tc.action_epsilon_state,
            )

        if "alpha" in states:
            tgc.alpha_state = states["alpha"]
            tgc._alpha_scheduler = tgc.alpha_config.build(state=tgc.alpha_state)

        if "n_td" in states:
            tgc.n_td_state = states["n_td"]
            tgc._n_td_scheduler = tgc.n_td_config.build(state=tgc.n_td_state)

    def export_agent_rng_state(self) -> dict[str, Any]:
        """Export agent-local RNG and counters."""
        return {
            "python_rng_state": self.rng.getstate(),
            "current_base_seed": getattr(self, "current_base_seed", None),
            "seed_count": getattr(self, "seed_count", 0),
        }

    def restore_agent_rng_state(self, state: dict[str, Any]) -> None:
        """Restore agent-local RNG and counters."""
        if "python_rng_state" in state:
            self.rng.setstate(state["python_rng_state"])
        if "current_base_seed" in state and state["current_base_seed"] is not None:
            self.current_base_seed = state["current_base_seed"]
        if "seed_count" in state:
            self.seed_count = state["seed_count"]

    def export_worker_seeds(self) -> list[int]:
        """Export the per-worker seeds used to construct async env workers."""
        return self.worker_seeds

    def restore_workers_from_seeds(self, seeds: list[int]) -> None:
        """Recreate async environment workers from saved seeds."""
        self.worker_seeds = list(seeds)
        self.workers = [
            AsyncMahjongEnv(seed=s, reward_config=self.config.reward)
            for s in self.worker_seeds
        ]

    def save_replay(self, ckpt_dir: Path) -> None:
        """Save replay buffer into checkpoint directory."""
        self.replay_buffer.save(str(ckpt_dir / "replay.json"))

    def load_replay(self, ckpt_dir: Path) -> None:
        """Load replay buffer from checkpoint directory if present."""
        p = ckpt_dir / "replay.json"
        if p.exists():
            self.replay_buffer = ReplayBuffer.load(str(p))

    @property
    def action_epsilon_scheduler(self) -> Scheduler:
        return self.tc.action_epsilon_scheduler

    @property
    def alpha_scheduler(self) -> Scheduler:
        return self.tgc.alpha_scheduler

    @property
    def n_td_scheduler(self) -> Scheduler:
        return self.tgc.n_td_scheduler

    def _gather_last_nonpad(
        self, x: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Extracts the last non-padding token's embedding or value from the sequence.

        Args:
            x: Tensor of shape [B, L, D] or [B, L]
            input_ids: Tensor of shape [B, L] indicating token IDs

        Returns:
            Tensor of shape [B, D] or [B]
        """
        B, L = input_ids.shape
        device = input_ids.device

        lengths = (input_ids != self.mc.pad_token_id).sum(dim=1)  # [B]
        last_idx = (lengths - 1).clamp(min=0)  # [B]

        batch_idx = torch.arange(B, device=device)  # [B]

        if x.dim() == 3:  # [B, L, D] -> [B, D]
            return x[batch_idx, last_idx, :]
        elif x.dim() == 2:  # [B, L] -> [B]
            return x[batch_idx, last_idx]
        else:
            raise ValueError(f"Unexpected x.dim={x.dim()}, expected 2 or 3.")

    def _pad_list_to_length(
        self,
        lst: list[int],
        target_length: int | None = None,
        pad_token_id: int | None = None,
    ) -> list[int]:
        if target_length is None:
            target_length = self.mc.max_seq_len
        if pad_token_id is None:
            pad_token_id = self.mc.pad_token_id
        if len(lst) >= target_length:
            raise ValueError("Value mc.max_seq_len is too low.")
        return lst + [pad_token_id] * (target_length - len(lst))

    def _state_to_ids(self, s: Any) -> list[int]:
        if isinstance(s, TokenList):
            return s.to_ids()
        if hasattr(s, "to_ids"):
            return list(s.to_ids())
        return list(s)

    def _ids_to_tokenlist(self, ids: list[int]) -> TokenList:
        return TokenList.from_ids(ids)

    def _random_one_index(self, multihot: list[int]) -> int:
        one_indices = [i for i, v in enumerate(multihot) if v == 1]
        if not one_indices:
            raise RuntimeError("Action mask error: no valid actions")
        return random.choice(one_indices)

    def _get_pad_tokens(self, hand: Any, history: Any) -> TokenList:
        hand_tokens: list[int] = []
        for index, num in enumerate(hand):
            if num:
                hand_tokens += [HAND_MIN + index] * int(num)

        hist_ids = self._state_to_ids(history)
        input_ids = hand_tokens + [SEP_ID] + hist_ids
        input_ids = self._pad_list_to_length(input_ids)
        return self._ids_to_tokenlist(input_ids)

    def _truncate_to_done(
        self,
        rewards: list[float],
        actions: list[int],
        dones: list[bool],
        states: list[Any],
        action_masks: list[list[int]],
    ) -> tuple[list[float], list[int], list[bool], list[Any], list[list[int]]]:
        if any(dones):
            t_end = dones.index(True) + 1
            rewards = rewards[:t_end]
            actions = actions[:t_end]
            dones = dones[:t_end]
            action_masks = action_masks[:t_end]
            states = states[: t_end + 1]
        return rewards, actions, dones, states, action_masks

    def _get_GAEs(
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: [T]
            values: [T]
            dones: [T]
        Returns:
            advs: [T]
        """
        T = rewards.shape[0]
        device, dtype = rewards.device, rewards.dtype

        masks = (1.0 - dones.to(dtype)).to(device)  # [T]
        # values_next: [T] (shifted and zero-padded)
        values_next = torch.cat(
            [values[1:], torch.zeros(1, device=device, dtype=dtype)], dim=0
        )
        # deltas: [T]
        deltas = rewards + self.gamma * values_next * masks - values

        advs = torch.zeros(T, device=device, dtype=dtype)
        gae = torch.zeros(1, device=device, dtype=dtype)
        for t in reversed(range(T)):
            gae = deltas[t] + self.gamma * self.lambd * gae * masks[t]
            advs[t] = gae
        return advs  # [T]

    def _get_nTDs(
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute n-step Temporal Difference returns.

        Args:
            rewards: [T]
            values: [T]
            dones: [T]
        Returns:
            out: [T]
        """
        n = int(self.n_td_scheduler.get())
        T = rewards.shape[0]
        device, dtype = rewards.device, rewards.dtype

        out = torch.zeros(T, device=device, dtype=dtype)
        for t in range(T):
            ret = torch.zeros(1, device=device, dtype=dtype)
            discount = torch.ones(1, device=device, dtype=dtype)
            for k in range(n):
                if t + k >= T:
                    break
                ret = ret + discount * rewards[t + k]
                if dones[t + k]:
                    discount = torch.zeros(1, device=device, dtype=dtype)
                    break
                discount = discount * self.gamma
            boot_index = t + n
            if boot_index < T and discount.item() != 0.0:
                ret = ret + discount * values[boot_index]
            out[t] = ret
        return out  # [T]

    def _masked_logprob(
        self, logits: torch.Tensor, actions: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probability of selected actions with masking.

        Args:
            logits: [B, Out]
            actions: [B]
            action_mask: [B, Out]
        Returns:
            logp_all: [B]
        """
        logits = logits.float()  # [B, Out]
        invalid = ~action_mask.to(torch.bool)  # [B, Out]
        logits = logits.masked_fill(invalid, -1e9)
        logp_all = torch.log_softmax(logits, dim=-1)  # [B, Out]
        return logp_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # [B]

    def _select_action(self, input_state: Any, action_mask: list[int]) -> int:
        with torch.no_grad():
            ids = self._state_to_ids(input_state)
            input_tensor = torch.tensor([ids], device=self.sc.device)  # [1, L]

            with self.amp_ctx:
                logits_all = self.policy_model_old(input_tensor)  # [1, L, Out]
            logits = self._gather_last_nonpad(
                logits_all, input_tensor
            ).float()  # [1, Out]

            mask_tensor = torch.tensor(
                action_mask, device=self.sc.device, dtype=torch.bool
            )  # [Out] -> [1, Out] broadcast in masked_fill if needed, but here usually 1D
            logits = logits.masked_fill(~mask_tensor, -1e9)

            probs = torch.softmax(logits, dim=-1)  # [1, Out]
            action = torch.multinomial(probs, num_samples=1).item()
        return int(action)

    def update(
        self,
        memories: Iterable[Trail],
        call_back: Callable[[float, float, float], None] | None = None,
    ) -> None:
        """PPO update using true minibatch size (timesteps) and gradient accumulation."""
        memories = list(memories)
        if not memories:
            return

        self.policy_model.train()
        self.value_model.train()

        minibatch_size = int(self.tc.batch_size)
        grad_accum_steps = int(getattr(self.tc, "grad_accum_steps", 1))
        grad_accum_steps = max(1, grad_accum_steps)

        mix = float(self.alpha_scheduler.step())

        states_buf: list[torch.Tensor] = []
        actions_buf: list[torch.Tensor] = []
        masks_buf: list[torch.Tensor] = []
        old_logp_buf: list[torch.Tensor] = []
        adv_buf: list[torch.Tensor] = []
        target_v_buf: list[torch.Tensor] = []

        kls: list[float] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                "[cyan]Computing old values...", total=len(memories)
            )

            for memory in memories:
                rewards = list(memory.rewards)
                actions = list(memory.actions)

                dones = list(getattr(memory, "dones", []))
                if not dones and hasattr(memory, "is_terminals"):
                    dones = list(memory.is_terminals)

                action_masks = list(getattr(memory, "action_masks", []))
                if not action_masks:
                    action_masks = [[1] * 46 for _ in range(len(actions))]

                states = list(memory.states)

                rewards, actions, dones, states, action_masks = self._truncate_to_done(
                    rewards, actions, dones, states, action_masks
                )

                T = len(rewards)
                if T == 0:
                    continue
                if len(states) != T + 1:
                    continue

                states_ids = [self._state_to_ids(s) for s in states]

                # s_t: [T, L] (truncated to T, excluding terminal state for policy eval usually,
                # but here s_t includes only T steps for forward pass matching rewards)
                s_t = torch.tensor(
                    states_ids[:-1], device=self.sc.device, dtype=torch.long
                )
                a_t = torch.tensor(
                    actions, device=self.sc.device, dtype=torch.long
                )  # [T]
                r_t = torch.tensor(
                    rewards, device=self.sc.device, dtype=torch.float32
                )  # [T]
                d_t = torch.tensor(
                    dones, device=self.sc.device, dtype=torch.bool
                )  # [T]
                m_t = torch.tensor(
                    action_masks, device=self.sc.device, dtype=torch.bool
                )  # [T, Out]

                with torch.no_grad():
                    with self.amp_ctx:
                        # old_logits: [T, Out]
                        old_logits = self._gather_last_nonpad(
                            self.policy_model_old(s_t), s_t
                        )
                        # old_logp: [T]
                        old_logp = self._masked_logprob(old_logits, a_t, m_t).detach()

                        # old_v: [T]
                        old_v = self._gather_last_nonpad(
                            self.value_model_old(s_t).squeeze(-1).float(), s_t
                        ).detach()

                    # adv: [T]
                    adv = self._get_GAEs(r_t, old_v, d_t).float().detach()
                    # ntd: [T]
                    ntd = self._get_nTDs(r_t, old_v, d_t).float().detach()
                    # target_v: [T]
                    target_v = (old_v + (ntd - old_v) * mix).detach()

                states_buf.append(s_t)
                actions_buf.append(a_t)
                masks_buf.append(m_t)
                old_logp_buf.append(old_logp)
                adv_buf.append(adv)
                target_v_buf.append(target_v)

                progress.advance(task)

        if not states_buf:
            return

        states_all = torch.cat(states_buf, dim=0)  # [N_total, L]
        actions_all = torch.cat(actions_buf, dim=0)  # [N_total]
        masks_all = torch.cat(masks_buf, dim=0)  # [N_total, Out]
        old_logp_all = torch.cat(old_logp_buf, dim=0)  # [N_total]
        adv_all = torch.cat(adv_buf, dim=0)  # [N_total]
        target_v_all = torch.cat(target_v_buf, dim=0)  # [N_total]

        adv_all = (adv_all - adv_all.mean()) / (adv_all.std(unbiased=False) + 1e-8)

        N = int(states_all.shape[0])
        if N == 0:
            return

        self.optimizer_policy.zero_grad(set_to_none=True)
        self.optimizer_value.zero_grad(set_to_none=True)

        micro_step = 0
        early_stop = False

        print("Old value compute completed")

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
        ) as progress:
            epoch_task = progress.add_task(
                "[yellow]Training epochs", total=int(self.K_epochs)
            )
            batch_task = progress.add_task(
                "[magenta]Batches", total=math.ceil(N / minibatch_size)
            )

            for epoch in range(int(self.K_epochs)):
                perm = torch.randperm(N, device=self.sc.device)

                progress.reset(batch_task)
                progress.update(
                    batch_task, description=f"[magenta]Epoch {epoch + 1} batches"
                )
                for start in range(0, N, minibatch_size):
                    end = min(start + minibatch_size, N)
                    idx = perm[start:end]

                    mb_states = states_all[idx]  # [B, L]
                    mb_actions = actions_all[idx]  # [B]
                    mb_masks = masks_all[idx]  # [B, Out]
                    mb_old_logp = old_logp_all[idx]  # [B]
                    mb_adv = adv_all[idx]  # [B]
                    mb_target_v = target_v_all[idx]  # [B]

                    with self.amp_ctx:
                        # new_logits: [B, Out]
                        new_logits = self._gather_last_nonpad(
                            self.policy_model(mb_states), mb_states
                        )
                        # new_logp: [B]
                        new_logp = self._masked_logprob(
                            new_logits, mb_actions, mb_masks
                        )

                        # values: [B]
                        values = self._gather_last_nonpad(
                            self.value_model(mb_states).squeeze(-1).float(), mb_states
                        )

                    # ratios: [B]
                    ratios = torch.exp(new_logp - mb_old_logp)
                    approx_kl = (mb_old_logp - new_logp).mean()
                    kls.append(float(approx_kl.detach().abs().item()))

                    if approx_kl.detach().abs() > self.tc.max_update_kl:
                        print(f"KL break | approx_kl={approx_kl.item():.6f}")
                        early_stop = True
                        break

                    loss_policy = -(ratios * mb_adv).mean() + (
                        self.beta * approx_kl.abs()
                    )
                    loss_value = self.MseLoss(values, mb_target_v)

                    loss_policy_w = self.weight_policy * loss_policy
                    loss_value_w = self.weight_value * loss_value
                    loss_total = loss_policy_w + loss_value_w

                    if self.sc.amp_enable:
                        self.amp_scaler.scale(
                            loss_total / float(grad_accum_steps)
                        ).backward()
                    else:
                        (loss_total / float(grad_accum_steps)).backward()
                    micro_step += 1

                    if call_back is not None:
                        call_back(
                            float(loss_total.detach().item()),
                            float(loss_value_w.detach().item()),
                            float(loss_policy_w.detach().item()),
                        )

                    do_step = (micro_step % grad_accum_steps) == 0
                    is_last_micro = (epoch == int(self.K_epochs) - 1) and (end == N)

                    if do_step or is_last_micro:
                        for name, param in self.value_model.named_parameters():
                            if param.grad is not None and torch.isnan(param.grad).any():
                                print(f"NaN gradient in value: {name}")
                        for name, param in self.policy_model.named_parameters():
                            if param.grad is not None and torch.isnan(param.grad).any():
                                print(f"NaN gradient in policy: {name}")

                        if self.sc.amp_enable:
                            self.amp_scaler.step(self.optimizer_policy)
                            self.amp_scaler.step(self.optimizer_value)
                            self.amp_scaler.update()
                            self.optimizer_policy.zero_grad(set_to_none=True)
                            self.optimizer_value.zero_grad(set_to_none=True)
                        else:
                            self.optimizer_policy.step()
                            self.optimizer_value.step()
                            self.optimizer_policy.zero_grad(set_to_none=True)
                            self.optimizer_value.zero_grad(set_to_none=True)

                    progress.advance(batch_task)

                progress.advance(epoch_task)

                if early_stop:
                    break

        self.policy_model_old.load_state_dict(self.policy_model.state_dict())
        self.value_model_old.load_state_dict(self.value_model.state_dict())
        self.policy_model_old.eval()
        self.value_model_old.eval()

        if kls:
            print(f"AVR approx KL: {sum(kls) / len(kls):.6f}")

    async def get_memory_async(
        self,
        worker_idx: int,
        env: AsyncMahjongEnv,
        call_back: Callable[[list[float]], None] | None = None,
    ) -> None:
        """Collect one episode of trajectories asynchronously."""
        memories = [Trail() for _ in range(4)]

        # Await the environment response
        new_seed = self.rng.randint(0, 1_000_000_000)
        self.worker_seeds[worker_idx] = new_seed
        state, reward, done, info = await env.reset(new_seed)

        no_memory_index = []
        last_hands = [None] * 4

        while True:
            action_mask = info["action_mask"]
            player_index = state["seat"]
            history = state["tokens"]
            hand = state["hand"]
            last_hands[player_index] = hand

            tokens = self._get_pad_tokens(hand, history)
            memories[player_index].states.append(tokens)
            memories[player_index].action_masks.append(list(action_mask))

            # This block runs on main process (Model), blocking the loop briefly
            # which is the desired behavior (Model Inference Authority)
            if random.random() > self.action_epsilon_scheduler.get():
                action = self._select_action(tokens, action_mask)
            else:
                action = self._random_one_index(action_mask)

            # Await step response
            next_state, reward, done, next_info = await env.step(action)

            memories[player_index].actions.append(int(action))
            memories[player_index].rewards.append(float(reward))
            memories[player_index].dones.append(bool(done))
            memories[player_index].info.append(next_info)

            if len(memories[player_index].rewards) >= 2:
                memories[player_index].rewards[-2] += float(
                    next_info.get("reward_update", 0.0)
                )

            state, info = next_state, next_info

            if done:
                if self.sc.verbose_positive_done_reward and reward > 5:
                    print(f"Terminal reward: {reward}")
                terminal_hist = state["tokens"]
                for idx, memory in enumerate(memories):
                    if len(memory.rewards) == 0:
                        no_memory_index.append(idx)
                        continue
                    hand = last_hands[idx] if last_hands[idx] is not None else [0] * 34
                    memory.states.append(self._get_pad_tokens(hand, terminal_hist))
                    memory.dones[-1] = True
                break

        if call_back is not None:
            for idx, memory in enumerate(memories):
                if idx not in no_memory_index:
                    call_back(memory.rewards)

        for idx, memory in enumerate(memories):
            if idx not in no_memory_index:
                if len(memory.states) > self.max_trail_len:
                    self.max_trail_len = len(memory.states)
                self.replay_buffer.add(memory)

    async def _worker_task(
        self,
        worker_idx: int,
        worker: AsyncMahjongEnv,
        count: int,
        call_back: Callable | None,
    ):
        """Helper to let one worker run multiple games sequentially."""
        for _ in range(count):
            await self.get_memory_async(worker_idx, worker, call_back)

    def get_memory_batch(
        self, call_back: Callable[[list[float]], None] | None = None
    ) -> None:
        """Collect multiple episodes in parallel."""
        total_games = int(self.evc.memget_num_per_update / 4)

        # Distribute games among workers
        base_count = total_games // self.worker_count
        remainder = total_games % self.worker_count

        tasks = []
        for i, worker in enumerate(self.workers):
            count = base_count + (1 if i < remainder else 0)
            if count > 0:
                tasks.append(self._worker_task(i, worker, count, call_back))

        async def _batch_runner():
            await asyncio.gather(*tasks)

        asyncio.run(_batch_runner())

    def __del__(self):
        # Cleanup workers
        if hasattr(self, "workers"):
            for w in self.workers:
                w.close()
