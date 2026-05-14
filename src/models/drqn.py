from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Obs:
    cache_feat: np.ndarray
    global_feat: np.ndarray
    valid_mask: np.ndarray


@dataclass
class AR:
    action: int
    reward: float
    done: bool




class RunningScale:
    """EMA-based dynamic normalizer for non-negative features."""

    def __init__(self, init_scale: float, alpha: float, eps: float, min_scale: float, percentile: float):
        self.scale = max(float(init_scale), float(min_scale), float(eps))
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.min_scale = float(min_scale)
        self.percentile = float(percentile)

    def update(self, values: List[float]):
        if not values:
            return
        arr = np.asarray(values, dtype=np.float32)
        arr = arr[np.isfinite(arr) & (arr > 0.0)]
        if arr.size == 0:
            return
        stat = float(np.percentile(arr, self.percentile))
        stat = max(stat, self.min_scale, self.eps)
        self.scale = (1.0 - self.alpha) * self.scale + self.alpha * stat

    def normalize(self, value: float) -> float:
        denom = max(self.scale, self.min_scale) + self.eps
        return float(np.clip(float(value) / denom, 0.0, 1.0))


class CacheEnv:
    """
    action:
      0: NOOP (hit OR empty)
      1..S: evict slot i (miss+full only)
    """

    def __init__(self, cache_size: int, use_global: bool, invalid_penalty: bool, config: Dict[str, object]):
        self.cache_size = cache_size
        self.use_global = use_global
        self.invalid_penalty = invalid_penalty

        self.rec_denom = float(config["RECENCY_DENOM"])
        self.freq_denom = float(config["FREQ_DENOM"])
        self.hit_ema_alpha = float(config["HIT_EMA_ALPHA"])
        self.miss_streak_clip = int(config["MISS_STREAK_CLIP"])

        self.feature_scaling_mode = str(config.get("FEATURE_SCALING_MODE", "static")).lower()
        self.reward_mode = str(config.get("REWARD_MODE", "binary")).lower()
        self.scaler_alpha = float(config.get("SCALER_EMA_ALPHA", 0.02))
        self.scaler_eps = float(config.get("SCALER_EPS", 1e-6))
        self.scaler_min_scale = float(config.get("SCALER_MIN_SCALE", 1.0))
        self.scaler_percentile = float(config.get("SCALER_PERCENTILE", 90.0))
        self.reward_norm_eps = float(config.get("REWARD_NORM_EPS", 1e-6))
        self.reward_clip = float(config.get("REWARD_CLIP", 2.0))
        self.invalid_action_reward = float(config.get("INVALID_ACTION_REWARD", -1.0))
        self.reset()

    def reset(self):
        self.cache_slots = [0] * self.cache_size
        self.t = 0
        self.access_history = {}
        self.frequency = {}
        self.hit_ema = 0.0
        self.miss_streak = 0
        self.recency_scale = RunningScale(self.rec_denom, self.scaler_alpha, self.scaler_eps, self.scaler_min_scale, self.scaler_percentile)
        self.freq_scale = RunningScale(self.freq_denom, self.scaler_alpha, self.scaler_eps, self.scaler_min_scale, self.scaler_percentile)
        self.miss_scale = RunningScale(self.miss_streak_clip, self.scaler_alpha, self.scaler_eps, self.scaler_min_scale, self.scaler_percentile)

    def _update_stats(self, req_id: int):
        self.access_history[req_id] = self.t
        self.frequency[req_id] = self.frequency.get(req_id, 0) + 1

    def _has_item(self, req_id: int) -> bool:
        return req_id in self.cache_slots

    def _has_empty(self) -> bool:
        return 0 in self.cache_slots

    def _fill_empty(self, req_id: int):
        idx = self.cache_slots.index(0)
        self.cache_slots[idx] = req_id

    def _evict_into(self, slot_idx: int, req_id: int):
        self.cache_slots[slot_idx] = req_id

    def get_cache_features(self) -> np.ndarray:
        feats = np.ones((self.cache_size, 2), dtype=np.float32)
        feats[:, 1] = 0.0

        gaps, freqs = [], []
        for item in self.cache_slots:
            if item == 0:
                continue
            gaps.append(float(self.t - self.access_history.get(item, 0)))
            freqs.append(float(self.frequency.get(item, 0)))

        if self.feature_scaling_mode == "dynamic":
            # Dynamic mode adapts feature scale to current workload and locality.
            self.recency_scale.update(gaps)
            self.freq_scale.update(freqs)

        for i, item in enumerate(self.cache_slots):
            if item == 0:
                continue
            gap = float(self.t - self.access_history.get(item, 0))
            freq = float(self.frequency.get(item, 0))
            if self.feature_scaling_mode == "dynamic":
                rec = self.recency_scale.normalize(gap)
                frq = self.freq_scale.normalize(freq)
            else:
                rec = min(gap / self.rec_denom, 1.0)
                frq = min(freq / self.freq_denom, 1.0)
            feats[i, 0] = rec
            feats[i, 1] = frq
        return feats

    def get_global_features(self) -> np.ndarray:
        if not self.use_global:
            return np.zeros((3,), dtype=np.float32)
        occupancy = 1.0 - (self.cache_slots.count(0) / float(self.cache_size))
        if self.feature_scaling_mode == "dynamic":
            self.miss_scale.update([float(self.miss_streak)])
            miss_norm = self.miss_scale.normalize(float(self.miss_streak))
        else:
            miss_norm = min(self.miss_streak / float(self.miss_streak_clip), 1.0)
        return np.asarray([occupancy, self.hit_ema, miss_norm], dtype=np.float32)

    def valid_action_mask(self, req_id: int) -> np.ndarray:
        mask = np.zeros((self.cache_size + 1,), dtype=np.bool_)
        hit = self._has_item(req_id)
        empty = self._has_empty()
        if hit or empty:
            mask[0] = True
        else:
            mask[1:] = True
        return mask

    def _update_global(self, hit: bool):
        x = 1.0 if hit else 0.0
        self.hit_ema = (1.0 - self.hit_ema_alpha) * self.hit_ema + self.hit_ema_alpha * x
        if hit:
            self.miss_streak = 0
        else:
            self.miss_streak = min(self.miss_streak + 1, self.miss_streak_clip)

    def _reward_from_hit(self, hit: bool) -> float:
        x = 1.0 if hit else 0.0
        if self.reward_mode == "adaptive":
            # Advantage-like shaping against previous running hit expectation.
            p = float(np.clip(self.hit_ema, 0.0, 1.0))
            std = float(np.sqrt(p * (1.0 - p) + self.reward_norm_eps))
            rew = (x - p) / std
            return float(np.clip(rew, -self.reward_clip, self.reward_clip))
        return x

    def step(self, req_id: int, action: int) -> Tuple[float, bool]:
        self.t += 1
        self._update_stats(req_id)

        hit = self._has_item(req_id)
        if hit:
            reward = self._reward_from_hit(True)
            self._update_global(True)
            return reward, True

        reward = self._reward_from_hit(False)

        if self._has_empty():
            self._fill_empty(req_id)
            self._update_global(False)
            return reward, False

        if action <= 0 or action > self.cache_size:
            if self.invalid_penalty:
                reward = min(reward, self.invalid_action_reward)
            slot_idx = random.randrange(self.cache_size)
        else:
            slot_idx = action - 1

        self._evict_into(slot_idx, req_id)
        self._update_global(False)
        return reward, False


class EpisodeReplay:
    def __init__(self, max_episodes: int):
        self.episodes = deque(maxlen=max_episodes)

    def __len__(self):
        return len(self.episodes)

    def add_episode(self, obs_list: List[Obs], ar_list: List[AR]):
        self.episodes.append((obs_list, ar_list))

    def sample_batch(self, batch_size: int, seq_len: int):
        batch_obs, batch_ar = [], []
        for _ in range(batch_size):
            obs_list, ar_list = random.choice(self.episodes)
            T = len(ar_list)
            while T < seq_len + 1:
                obs_list, ar_list = random.choice(self.episodes)
                T = len(ar_list)
            start = random.randint(0, T - (seq_len + 1))
            batch_obs.append(obs_list[start : start + seq_len + 1])
            batch_ar.append(ar_list[start : start + seq_len])
        return batch_obs, batch_ar


CACHE_KEY_DIM = 32
META_DIM = 2
GLOBAL_DIM = 3
LSTM_INPUT_DIM = 128
HIDDEN_DIM = 128


def pool_meanmaxmin(slot_emb: torch.Tensor) -> torch.Tensor:
    mean = slot_emb.mean(dim=1)
    mx = slot_emb.max(dim=1).values
    mn = slot_emb.min(dim=1).values
    return torch.cat([mean, mx, mn], dim=1)


class PerSlotHead(nn.Module):
    def __init__(self, cache_size: int, K: int, H: int):
        super().__init__()
        self.cache_size = cache_size
        self.noop_head = nn.Sequential(nn.Linear(H, 128), nn.ReLU(), nn.Linear(128, 1))
        self.slot_head = nn.Sequential(nn.Linear(K + H, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, slot_emb: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        q_noop = self.noop_head(ctx)
        ctx_exp = ctx.unsqueeze(1).expand(-1, self.cache_size, -1)
        slot_in = torch.cat([slot_emb, ctx_exp], dim=-1)
        q_slots = self.slot_head(slot_in).squeeze(-1)
        return torch.cat([q_noop, q_slots], dim=1)


class DRQN_PerSlot(nn.Module):
    def __init__(self, cache_size: int, use_global: bool, device: torch.device):
        super().__init__()
        self.cache_size = cache_size
        self.use_global = use_global
        self.device = device

        self.slot_proj = nn.Sequential(
            nn.Linear(META_DIM, CACHE_KEY_DIM),
            nn.ReLU(),
            nn.Linear(CACHE_KEY_DIM, CACHE_KEY_DIM),
            nn.ReLU(),
        )

        in_dim = 3 * CACHE_KEY_DIM + (GLOBAL_DIM if use_global else 0)
        self.in_proj = nn.Sequential(nn.Linear(in_dim, LSTM_INPUT_DIM), nn.ReLU())
        self.lstm = nn.LSTM(LSTM_INPUT_DIM, HIDDEN_DIM, batch_first=True)
        self.head = PerSlotHead(cache_size, CACHE_KEY_DIM, HIDDEN_DIM)

    def init_hidden(self, B: int):
        h = torch.zeros(1, B, HIDDEN_DIM, device=self.device)
        c = torch.zeros(1, B, HIDDEN_DIM, device=self.device)
        return (h, c)

    def forward_step(self, cache_feat: torch.Tensor, global_feat: torch.Tensor, hidden):
        slot_emb = self.slot_proj(cache_feat)
        pooled = pool_meanmaxmin(slot_emb)
        x = pooled if not self.use_global else torch.cat([pooled, global_feat], dim=1)
        x = self.in_proj(x).unsqueeze(1)
        out, hidden = self.lstm(x, hidden)
        ctx = out[:, -1, :]
        q = self.head(slot_emb, ctx)
        return q, hidden


class PoolingQNet(nn.Module):
    def __init__(self, cache_size: int, use_global: bool, device: torch.device):
        super().__init__()
        self.cache_size = cache_size
        self.use_global = use_global
        self.device = device

        self.slot_proj = nn.Sequential(
            nn.Linear(META_DIM, CACHE_KEY_DIM),
            nn.ReLU(),
            nn.Linear(CACHE_KEY_DIM, CACHE_KEY_DIM),
            nn.ReLU(),
        )
        in_dim = 3 * CACHE_KEY_DIM + (GLOBAL_DIM if use_global else 0)

        self.in_proj = nn.Sequential(nn.Linear(in_dim, LSTM_INPUT_DIM), nn.ReLU())
        self.lstm = nn.LSTM(LSTM_INPUT_DIM, HIDDEN_DIM, batch_first=True)
        self.out = nn.Sequential(nn.Linear(HIDDEN_DIM, 128), nn.ReLU(), nn.Linear(128, cache_size + 1))

    def init_hidden(self, B: int):
        h = torch.zeros(1, B, HIDDEN_DIM, device=self.device)
        c = torch.zeros(1, B, HIDDEN_DIM, device=self.device)
        return (h, c)

    def forward_step(self, cache_feat: torch.Tensor, global_feat: torch.Tensor, hidden):
        slot_emb = self.slot_proj(cache_feat)
        pooled = pool_meanmaxmin(slot_emb)
        x = pooled if not self.use_global else torch.cat([pooled, global_feat], dim=1)
        x = self.in_proj(x).unsqueeze(1)
        out, hidden = self.lstm(x, hidden)
        q = self.out(out[:, -1, :])
        return q, hidden


def build_models(cache_size: int, s, device: torch.device):
    if s.algo == "drqn_perslot":
        online = DRQN_PerSlot(cache_size, s.use_global, device).to(device)
        target = DRQN_PerSlot(cache_size, s.use_global, device).to(device)
    elif s.algo == "pooling_lstm":
        online = PoolingQNet(cache_size, s.use_global, device=device).to(device)
        target = PoolingQNet(cache_size, s.use_global, device=device).to(device)
    else:
        raise ValueError(s.algo)

    target.load_state_dict(online.state_dict())
    return online, target


@torch.no_grad()
def select_action(model, obs: Obs, hidden, eps: float, device: torch.device) -> Tuple[int, object]:
    cf = torch.from_numpy(obs.cache_feat[None, :, :]).to(device)
    gf = torch.from_numpy(obs.global_feat[None, :]).to(device)
    mk = torch.from_numpy(obs.valid_mask[None, :]).to(device)

    if random.random() < eps:
        valid = torch.nonzero(mk[0], as_tuple=False).view(-1).tolist()
        a = int(random.choice(valid))
        _, hidden2 = model.forward_step(cf, gf, hidden)
        return a, hidden2

    q, hidden2 = model.forward_step(cf, gf, hidden)
    q = q.masked_fill(~mk, float("-inf"))
    a = int(torch.argmax(q, dim=1).item())
    return a, hidden2


def train_step(online, target, optimizer, replay: EpisodeReplay, cache_size: int, config: Dict[str, object], device: torch.device) -> float:
    if len(replay) < int(config["START_TRAIN_AFTER_EPISODES"]):
        return 0.0

    B = int(config["BATCH_SIZE"])
    L = int(config["BURN_IN"] + config["UNROLL"])
    A = cache_size + 1

    batch_obs, batch_ar = replay.sample_batch(B, L)

    cf_np = np.zeros((B, L + 1, cache_size, 2), dtype=np.float32)
    gf_np = np.zeros((B, L + 1, 3), dtype=np.float32)
    mk_np = np.zeros((B, L + 1, A), dtype=np.bool_)
    act_np = np.zeros((B, L), dtype=np.int64)
    rew_np = np.zeros((B, L), dtype=np.float32)
    done_np = np.zeros((B, L), dtype=np.bool_)

    for b in range(B):
        obs_seq = batch_obs[b]
        ar_seq = batch_ar[b]
        for t in range(L + 1):
            cf_np[b, t] = obs_seq[t].cache_feat
            gf_np[b, t] = obs_seq[t].global_feat
            mk_np[b, t] = obs_seq[t].valid_mask
        for t in range(L):
            act_np[b, t] = ar_seq[t].action
            rew_np[b, t] = ar_seq[t].reward
            done_np[b, t] = ar_seq[t].done

    cf = torch.from_numpy(cf_np).to(device)
    gf = torch.from_numpy(gf_np).to(device)
    mk = torch.from_numpy(mk_np).to(device)
    act = torch.from_numpy(act_np).to(device)
    rew = torch.from_numpy(rew_np).to(device)
    done = torch.from_numpy(done_np).to(device)

    h_on = online.init_hidden(B)
    h_tg = target.init_hidden(B)

    q_on_all, q_tg_all = [], []
    for t in range(L + 1):
        q_on, h_on = online.forward_step(cf[:, t], gf[:, t], h_on)
        with torch.no_grad():
            q_tg, h_tg = target.forward_step(cf[:, t], gf[:, t], h_tg)
        q_on_all.append(q_on)
        q_tg_all.append(q_tg)

    q_on_all = torch.stack(q_on_all, dim=1)
    q_tg_all = torch.stack(q_tg_all, dim=1)

    losses = []
    burn = int(config["BURN_IN"])
    gamma = float(config["GAMMA"])
    for t in range(burn, L):
        q_t = q_on_all[:, t, :].masked_fill(~mk[:, t, :], float("-inf"))
        q_sa = q_t.gather(1, act[:, t].unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next_online = q_on_all[:, t + 1, :].detach().masked_fill(~mk[:, t + 1, :], float("-inf"))
            best_a = q_next_online.argmax(dim=1)
            q_next_target = q_tg_all[:, t + 1, :].masked_fill(~mk[:, t + 1, :], float("-inf"))
            q_next = q_next_target.gather(1, best_a.unsqueeze(1)).squeeze(1)
            td = rew[:, t] + (~done[:, t]).float() * gamma * q_next

        losses.append(F.smooth_l1_loss(q_sa, td))

    loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), float(config["GRAD_CLIP"]))
    optimizer.step()
    return float(loss.item())


def rollout_episode(model, req_stream: List[int], start: int, length: int, cache_size: int, s, eps: float, config: Dict[str, object], device: torch.device) -> Tuple[List[Obs], List[AR], Dict[str, float]]:
    env = CacheEnv(cache_size, use_global=s.use_global, invalid_penalty=s.invalid_penalty, config=config)
    hidden = model.init_hidden(1)
    model.eval()

    obs_list, ar_list = [], []
    total_rew = 0.0
    hit_count = 0
    miss_count = 0

    end = min(start + length, len(req_stream))
    T = end - start
    if T < 2:
        return [], [], {"total_reward": 0.0, "hit_count": 0, "miss_count": 0}

    for i in range(T):
        req = req_stream[start + i]
        obs = Obs(
            cache_feat=env.get_cache_features(),
            global_feat=env.get_global_features(),
            valid_mask=env.valid_action_mask(req),
        )
        obs_list.append(obs)
        a, hidden = select_action(model, obs, hidden, eps, device)
        r, hit = env.step(req, a)
        total_rew += float(r)
        if hit:
            hit_count += 1
        else:
            miss_count += 1
        ar_list.append(AR(action=a, reward=float(r), done=(i == T - 1)))

    last_req = req_stream[end - 1]
    obs_list.append(
        Obs(
            cache_feat=env.get_cache_features(),
            global_feat=env.get_global_features(),
            valid_mask=env.valid_action_mask(last_req),
        )
    )
    return obs_list, ar_list, {"total_reward": total_rew, "hit_count": hit_count, "miss_count": miss_count}


def make_env_for_eval(cache_size: int, s, config: Dict[str, object]) -> CacheEnv:
    return CacheEnv(cache_size, use_global=s.use_global, invalid_penalty=s.invalid_penalty, config=config)


def make_obs_for_eval(env: CacheEnv, req: int) -> Obs:
    return Obs(
        cache_feat=env.get_cache_features(),
        global_feat=env.get_global_features(),
        valid_mask=env.valid_action_mask(req),
    )
