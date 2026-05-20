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
    req_feat: np.ndarray
    valid_mask: np.ndarray


@dataclass
class AR:
    action: int
    reward: float
    done: bool


class CacheEnv:
    def __init__(self, cache_size: int, use_global: bool, invalid_penalty: bool, config: Dict[str, object]):
        self.cache_size = cache_size
        self.use_global = use_global
        self.invalid_penalty = invalid_penalty

        self.rec_denom = float(config["RECENCY_DENOM"])
        self.freq_denom = float(config["FREQ_DENOM"])
        self.hit_ema_alpha = float(config["HIT_EMA_ALPHA"])
        self.miss_streak_clip = int(config["MISS_STREAK_CLIP"])
        self.use_two_stage_tinylfu = bool(config.get("USE_TWO_STAGE_TINYLFU", config.get("USE_TINYLFU_ADMISSION", False)))
        self.use_admission_heuristic_mask = bool(config.get("USE_ADMISSION_HEURISTIC_MASK", False))
        self.admission_features = bool(config.get("ADMISSION_FEATURES", True))
        self.recent_window_size = int(config.get("RECENT_WINDOW_SIZE", 1000))
        self.tinylfu_min_admit_count = int(config.get("TINYLFU_MIN_ADMIT_COUNT", 2))
        self.bypass_reward = float(config.get("BYPASS_REWARD", 0.0))
        self.req_freq_denom = float(config.get("REQ_FREQ_DENOM", self.freq_denom))
        self.recent_freq_denom = float(config.get("RECENT_FREQ_DENOM", max(1.0, np.sqrt(self.recent_window_size))))
        self.reset()

    def reset(self):
        self.cache_slots = [0] * self.cache_size
        self.t = 0
        self.access_history = {}
        self.frequency = {}
        self.recent_window = deque()
        self.recent_counts = {}
        self.hit_ema = 0.0
        self.miss_streak = 0
        self.last_step_info = {}

    def _update_stats(self, req_id: int):
        self.access_history[req_id] = self.t
        self.frequency[req_id] = self.frequency.get(req_id, 0) + 1
        self.recent_window.append(req_id)
        self.recent_counts[req_id] = self.recent_counts.get(req_id, 0) + 1
        while len(self.recent_window) > self.recent_window_size:
            old = self.recent_window.popleft()
            old_count = self.recent_counts.get(old, 0) - 1
            if old_count <= 0:
                self.recent_counts.pop(old, None)
            else:
                self.recent_counts[old] = old_count

    def _recent_count(self, req_id: int) -> int:
        return int(self.recent_counts.get(req_id, 0))

    def _total_count(self, req_id: int) -> int:
        return int(self.frequency.get(req_id, 0))

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
        feats = np.zeros((self.cache_size, META_DIM), dtype=np.float32)
        feats[:, 0] = 1.0
        feats[:, 3] = 1.0
        for i, item in enumerate(self.cache_slots):
            if item == 0:
                continue
            gap = self.t - self.access_history.get(item, 0)
            feats[i, 0] = min(gap / self.rec_denom, 1.0)
            feats[i, 1] = min(self._total_count(item) / self.freq_denom, 1.0)
            feats[i, 2] = min(self._recent_count(item) / self.recent_freq_denom, 1.0)
            feats[i, 3] = 1.0 if self._total_count(item) < self.tinylfu_min_admit_count else 0.0
        return feats

    def should_admit(self, req_id: int) -> bool:
        if not self.use_two_stage_tinylfu:
            return True
        if self._has_item(req_id) or self._has_empty():
            return True
        req_recent = self._recent_count(req_id)
        req_total = self._total_count(req_id)
        cache_items = [it for it in self.cache_slots if it != 0]
        if not cache_items:
            return True
        min_cache_recent = min(self._recent_count(it) for it in cache_items)
        min_cache_total = min(self._total_count(it) for it in cache_items)
        return (
            req_recent >= min_cache_recent
            or req_total >= self.tinylfu_min_admit_count
            or req_total >= min_cache_total
        )

    def get_request_features(self, req_id: int) -> np.ndarray:
        seen = 1.0 if req_id in self.access_history else 0.0
        if seen > 0.0:
            gap = self.t - self.access_history[req_id]
            req_recency = min(gap / self.rec_denom, 1.0)
        else:
            req_recency = 1.0
        total = self._total_count(req_id)
        recent = self._recent_count(req_id)
        req_total_freq = min(total / self.req_freq_denom, 1.0)
        req_recent_freq = min(recent / self.recent_freq_denom, 1.0)
        req_is_cold = 1.0 if total < self.tinylfu_min_admit_count else 0.0
        cache_items = [it for it in self.cache_slots if it != 0]
        if not cache_items:
            hotness_vs_cache = 0.5
        else:
            min_cache_recent = min(self._recent_count(it) for it in cache_items)
            raw = np.clip((recent - min_cache_recent) / self.recent_freq_denom, -1.0, 1.0)
            hotness_vs_cache = (raw + 1.0) / 2.0
        feat = np.asarray([seen, req_recency, req_total_freq, req_recent_freq, req_is_cold, hotness_vs_cache], dtype=np.float32)
        feat = np.clip(np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        if not self.admission_features:
            feat[:] = 0.0
        return feat

    def get_global_features(self) -> np.ndarray:
        if not self.use_global:
            return np.zeros((3,), dtype=np.float32)
        occupancy = 1.0 - (self.cache_slots.count(0) / float(self.cache_size))
        miss_norm = min(self.miss_streak / float(self.miss_streak_clip), 1.0)
        return np.asarray([occupancy, self.hit_ema, miss_norm], dtype=np.float32)

    def valid_action_mask(self, req_id: int) -> np.ndarray:
        mask = np.zeros((self.cache_size + 1,), dtype=np.bool_)
        hit = self._has_item(req_id)
        empty = self._has_empty()
        if hit or empty:
            mask[0] = True
        else:
            if not self.use_two_stage_tinylfu:
                mask[1:] = True
            elif self.should_admit(req_id):
                mask[1:] = True
            else:
                mask[0] = True
        return mask

    def _update_global(self, hit: bool):
        x = 1.0 if hit else 0.0
        self.hit_ema = (1.0 - self.hit_ema_alpha) * self.hit_ema + self.hit_ema_alpha * x
        if hit:
            self.miss_streak = 0
        else:
            self.miss_streak = min(self.miss_streak + 1, self.miss_streak_clip)

    def step(self, req_id: int, action: int) -> Tuple[float, bool]:
        self.t += 1
        if self._has_item(req_id):
            self._update_stats(req_id)
            self._update_global(True)
            self.last_step_info = {"hit": True, "miss": False, "admit": True, "bypass": False, "insert": False, "evict": False}
            return 1.0, True

        reward = 0.0
        if self._has_empty():
            self._update_stats(req_id)
            self._fill_empty(req_id)
            self._update_global(False)
            self.last_step_info = {"hit": False, "miss": True, "admit": True, "bypass": False, "insert": True, "evict": False}
            return reward, False

        admit = True
        if self.use_two_stage_tinylfu:
            admit = self.should_admit(req_id)

        if not admit:
            self._update_stats(req_id)
            self._update_global(False)
            self.last_step_info = {"hit": False, "miss": True, "admit": False, "bypass": True, "insert": False, "evict": False}
            return self.bypass_reward, False

        invalid = False
        if action <= 0 or action > self.cache_size:
            if self.invalid_penalty:
                reward = -1.0
            slot_idx = random.randrange(self.cache_size)
            invalid = True
        else:
            slot_idx = action - 1
        self._update_stats(req_id)
        self._evict_into(slot_idx, req_id)
        self._update_global(False)
        self.last_step_info = {
            "hit": False, "miss": True, "admit": True, "bypass": False, "insert": False, "evict": True,
            "invalid": invalid, "evicted_slot": slot_idx
        }
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
META_DIM = 4
GLOBAL_DIM = 3
REQ_DIM = 6
REQ_EMB_DIM = 32
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
        self.slot_proj = nn.Sequential(nn.Linear(META_DIM, CACHE_KEY_DIM), nn.ReLU(), nn.Linear(CACHE_KEY_DIM, CACHE_KEY_DIM), nn.ReLU())
        self.req_proj = nn.Sequential(nn.Linear(REQ_DIM, REQ_EMB_DIM), nn.ReLU(), nn.Linear(REQ_EMB_DIM, REQ_EMB_DIM), nn.ReLU())
        in_dim = 3 * CACHE_KEY_DIM + REQ_EMB_DIM + (GLOBAL_DIM if use_global else 0)
        self.in_proj = nn.Sequential(nn.Linear(in_dim, LSTM_INPUT_DIM), nn.ReLU())
        self.lstm = nn.LSTM(LSTM_INPUT_DIM, HIDDEN_DIM, batch_first=True)
        self.head = PerSlotHead(cache_size, CACHE_KEY_DIM, HIDDEN_DIM)

    def init_hidden(self, B: int):
        h = torch.zeros(1, B, HIDDEN_DIM, device=self.device)
        c = torch.zeros(1, B, HIDDEN_DIM, device=self.device)
        return (h, c)

    def forward_step(self, cache_feat: torch.Tensor, global_feat: torch.Tensor, req_feat: torch.Tensor, hidden):
        slot_emb = self.slot_proj(cache_feat)
        pooled = pool_meanmaxmin(slot_emb)
        req_emb = self.req_proj(req_feat)
        x = torch.cat([pooled, req_emb], dim=1)
        if self.use_global:
            x = torch.cat([x, global_feat], dim=1)
        x = self.in_proj(x).unsqueeze(1)
        out, hidden = self.lstm(x, hidden)
        q = self.head(slot_emb, out[:, -1, :])
        return q, hidden




def build_models(cache_size: int, s, device: torch.device):
    if s.algo != "drqn_perslot":
        raise ValueError(s.algo)
    online = DRQN_PerSlot(cache_size, s.use_global, device).to(device)
    target = DRQN_PerSlot(cache_size, s.use_global, device).to(device)
    target.load_state_dict(online.state_dict())
    return online, target

@torch.no_grad()
def select_action(model, obs: Obs, hidden, eps: float, device: torch.device) -> Tuple[int, object]:
    cf = torch.from_numpy(obs.cache_feat[None, :, :]).to(device)
    gf = torch.from_numpy(obs.global_feat[None, :]).to(device)
    rf = torch.from_numpy(obs.req_feat[None, :]).to(device)
    mk = torch.from_numpy(obs.valid_mask[None, :]).to(device)
    if random.random() < eps:
        a = int(random.choice(torch.nonzero(mk[0], as_tuple=False).view(-1).tolist()))
        _, hidden2 = model.forward_step(cf, gf, rf, hidden)
        return a, hidden2
    q, hidden2 = model.forward_step(cf, gf, rf, hidden)
    q = q.masked_fill(~mk, float("-inf"))
    return int(torch.argmax(q, dim=1).item()), hidden2


def train_step(online, target, optimizer, replay: EpisodeReplay, cache_size: int, config: Dict[str, object], device: torch.device) -> float:
    if len(replay) < int(config["START_TRAIN_AFTER_EPISODES"]):
        return 0.0
    B = int(config["BATCH_SIZE"])
    L = int(config["BURN_IN"] + config["UNROLL"])
    A = cache_size + 1
    batch_obs, batch_ar = replay.sample_batch(B, L)
    cf_np = np.zeros((B, L + 1, cache_size, META_DIM), dtype=np.float32)
    gf_np = np.zeros((B, L + 1, 3), dtype=np.float32)
    rf_np = np.zeros((B, L + 1, REQ_DIM), dtype=np.float32)
    mk_np = np.zeros((B, L + 1, A), dtype=np.bool_)
    act_np = np.zeros((B, L), dtype=np.int64)
    rew_np = np.zeros((B, L), dtype=np.float32)
    done_np = np.zeros((B, L), dtype=np.bool_)
    for b in range(B):
        for t in range(L + 1):
            cf_np[b, t] = batch_obs[b][t].cache_feat
            gf_np[b, t] = batch_obs[b][t].global_feat
            rf_np[b, t] = batch_obs[b][t].req_feat
            mk_np[b, t] = batch_obs[b][t].valid_mask
        for t in range(L):
            act_np[b, t] = batch_ar[b][t].action
            rew_np[b, t] = batch_ar[b][t].reward
            done_np[b, t] = batch_ar[b][t].done
    cf, gf, rf, mk = torch.from_numpy(cf_np).to(device), torch.from_numpy(gf_np).to(device), torch.from_numpy(rf_np).to(device), torch.from_numpy(mk_np).to(device)
    act, rew, done = torch.from_numpy(act_np).to(device), torch.from_numpy(rew_np).to(device), torch.from_numpy(done_np).to(device)
    h_on, h_tg = online.init_hidden(B), target.init_hidden(B)
    q_on_all, q_tg_all = [], []
    for t in range(L + 1):
        q_on, h_on = online.forward_step(cf[:, t], gf[:, t], rf[:, t], h_on)
        with torch.no_grad():
            q_tg, h_tg = target.forward_step(cf[:, t], gf[:, t], rf[:, t], h_tg)
        q_on_all.append(q_on)
        q_tg_all.append(q_tg)
    q_on_all = torch.stack(q_on_all, dim=1)
    q_tg_all = torch.stack(q_tg_all, dim=1)
    losses, burn, gamma = [], int(config["BURN_IN"]), float(config["GAMMA"])
    for t in range(burn, L):
        q_t = q_on_all[:, t, :].masked_fill(~mk[:, t, :], float("-inf"))
        q_sa = q_t.gather(1, act[:, t].unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_a = q_on_all[:, t + 1, :].detach().masked_fill(~mk[:, t + 1, :], float("-inf")).argmax(dim=1)
            q_next = q_tg_all[:, t + 1, :].masked_fill(~mk[:, t + 1, :], float("-inf")).gather(1, best_a.unsqueeze(1)).squeeze(1)
            td = rew[:, t] + (~done[:, t]).float() * gamma * q_next
        losses.append(F.smooth_l1_loss(q_sa, td))
    loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), float(config["GRAD_CLIP"]))
    optimizer.step()
    return float(loss.item())


def rollout_episode(model, req_stream: List[int], start: int, length: int, cache_size: int, s, eps: float, config: Dict[str, object], device: torch.device):
    env = CacheEnv(cache_size, use_global=s.use_global, invalid_penalty=s.invalid_penalty, config=config)
    hidden = model.init_hidden(1)
    model.eval()
    obs_list, ar_list = [], []
    stats = {
        "total_reward": 0.0, "hit_count": 0, "miss_count": 0, "bypass_count": 0, "insert_count": 0, "eviction_count": 0,
        "admit_count": 0, "reject_count": 0
    }
    end = min(start + length, len(req_stream))
    T = end - start
    if T < 2:
        return [], [], stats
    for i in range(T):
        req = req_stream[start + i]
        obs = Obs(env.get_cache_features(), env.get_global_features(), env.get_request_features(req), env.valid_action_mask(req))
        obs_list.append(obs)
        a, hidden = select_action(model, obs, hidden, eps, device)
        r, hit = env.step(req, a)
        stats["total_reward"] += float(r)
        step_info = env.last_step_info
        if step_info.get("hit", hit):
            stats["hit_count"] += 1
        else:
            stats["miss_count"] += 1
            if step_info.get("admit", False):
                stats["admit_count"] += 1
            else:
                stats["reject_count"] += 1
            if step_info.get("insert", False):
                stats["insert_count"] += 1
            elif step_info.get("bypass", False):
                stats["bypass_count"] += 1
            elif step_info.get("evict", False):
                stats["eviction_count"] += 1
        ar_list.append(AR(action=a, reward=float(r), done=(i == T - 1)))
    last_req = req_stream[end - 1]
    obs_list.append(Obs(env.get_cache_features(), env.get_global_features(), env.get_request_features(last_req), env.valid_action_mask(last_req)))
    return obs_list, ar_list, stats


def make_env_for_eval(cache_size: int, s, config: Dict[str, object]) -> CacheEnv:
    return CacheEnv(cache_size, use_global=s.use_global, invalid_penalty=s.invalid_penalty, config=config)


def make_obs_for_eval(env: CacheEnv, req: int) -> Obs:
    return Obs(env.get_cache_features(), env.get_global_features(), env.get_request_features(req), env.valid_action_mask(req))
