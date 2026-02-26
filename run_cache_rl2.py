# ============================================================
# VSCode-friendly Paper-grade Experiment Runner (FAST BASELINES) - NO HMIX
# + Quick preset (--use_quick_preset) with practical FULL eval
#
# ✅ NO HMIX: HMIX 관련 코드/파라미터/결과 컬럼 전부 제거
# ✅ alpha: 1.3, 1.4, 1.5, 1.6, 1.7, 1.8   (요청 반영)
#
# Run (paper-grade default):
#   python run_cache_rl_vscode_nohmix_quick.py --out_dir out --device cuda
#
# Run (quick preset):
#   python run_cache_rl_vscode_nohmix_quick.py --out_dir out --device cuda --use_quick_preset
#
# Outputs:
#   out/results.csv   (run-level)
#   out/summary.csv   (aggregated)
#   out/logs/*.jsonl  (train logs)
#   out/ckpt/*.pt     (resume checkpoint)
# ============================================================

from __future__ import annotations

import os
import json
import csv
import time
import random
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict
from collections import deque

import numpy as np
from src.baselines.factory import build_baselines
from src.evaluation.evaluator import evaluate_policy_with_baselines, BaselineKey
from src.workload.builder import build_trace
from src.workload.diagnostics import workload_diagnostics

# ---- Dependency check ----
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
except Exception as e:
    raise RuntimeError(
        "PyTorch가 필요함. 예) pip install torch\n"
        f"원인: {repr(e)}"
    )

try:
    from tqdm.auto import tqdm
except Exception as e:
    raise RuntimeError(
        "tqdm가 필요함. 예) pip install tqdm\n"
        f"원인: {repr(e)}"
    )


# =========================
# 0) CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="out", help="결과 저장 폴더 (기본: ./out)")
    p.add_argument("--device", type=str, default=None, help="cpu | cuda | cuda:0 등")
    p.add_argument("--use_quick_preset", action="store_true", help="빠른 실험용 축소 프리셋 적용")
    return p.parse_args()

ARGS = parse_args()
OUT_DIR = ARGS.out_dir

if ARGS.device is not None:
    DEVICE = torch.device(ARGS.device)
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)


# =========================
# 1) GLOBAL CONFIG (paper-grade default)
# =========================
CONFIG = {
    "OUT_DIR": OUT_DIR,

    # workload
    "VOCAB_SIZE": 10_000,
    "NUM_REQUESTS": 1_000_000,
    "TRAIN_RATIO": 0.8,

    # ✅ alpha list (요청)
    "ZIPF_ALPHAS": [1.3, 1.4, 1.5, 1.6, 1.7, 1.8],

    # cache sizes / scenarios
    "CACHE_SIZES": [16, 64],
    "SCENARIOS": ["zipf", "hotshift"],

    # baseline list (Step A multi-baseline-ready schema)
    "BASELINES": ["lru"],

    # --- hotshift 강화 ---
    "HOTSHIFT_PHASES": 12,
    "HOTSHIFT_OFFSET_STEP_MODE": "coprime_stride",
    "HOTSHIFT_OFFSET_STEP_CUSTOM": 5003,
    "HOTSHIFT_PHASE_SKEW": "1.25,1.35,1.45,1.55,1.65,1.75",
    "HOTSHIFT_MIX_RATIO": 0.8,
    "HOTSHIFT_TRANSITION": "smooth",

    # training
    "EPISODE_LEN": 4000,
    "MAX_TRAIN_EPISODES": 400,
    "REPLAY_MAX_EPISODES": 800,
    "BATCH_SIZE": 32,
    "BURN_IN": 20,
    "UNROLL": 40,
    "START_TRAIN_AFTER_EPISODES": 20,
    "UPDATES_PER_EPISODE": 20,
    "TARGET_UPDATE_EVERY_UPDATES": 500,

    "GAMMA": 0.99,
    "LR": 2e-4,
    "GRAD_CLIP": 1.0,

    # exploration
    "EPSILON_START": 1.0,
    "EPSILON_END": 0.05,
    "EPSILON_DECAY_STEPS": 300_000,

    # feature scaling (RL)
    "RECENCY_DENOM": 2000.0,
    "FREQ_DENOM": 200.0,

    # global features
    "HIT_EMA_ALPHA": 0.01,
    "MISS_STREAK_CLIP": 200,

    # eval schedule
    "FAST_EVAL_EVERY_EP": 5,
    "FAST_EVAL_STEPS": 20_000,
    "FULL_EVAL_EVERY_EP": 50,
    "FULL_EVAL_STEPS": 200_000,

    # seeds
    "SEEDS": [0],

    # checkpoint / resume
    "SAVE_CKPT": True,
    "SAVE_CKPT_EVERY_EP": 5,
}

# =========================
# 1.1) QUICK PRESET
# =========================
def apply_quick_preset():
    # 목적: “대략 돌아가는지 / 트렌드 확인”용
    CONFIG.update({
        "NUM_REQUESTS": 250_000,

        # alpha는 요청대로 1.3~1.8 유지
        "ZIPF_ALPHAS": [1.3, 1.4, 1.5, 1.6, 1.7, 1.8],

        "CACHE_SIZES": [16, 64],
        "SCENARIOS": ["zipf", "hotshift"],
        "SEEDS": [0, 1],

        # shorter training
        "EPISODE_LEN": 2000,
        "MAX_TRAIN_EPISODES": 80,
        "REPLAY_MAX_EPISODES": 300,
        "START_TRAIN_AFTER_EPISODES": 8,
        "UPDATES_PER_EPISODE": 8,

        # shorter eval
        "FAST_EVAL_EVERY_EP": 20,
        "FAST_EVAL_STEPS": 5_000,

        # ✅ 학습 중 full eval이 최소 1번은 찍히도록 현실적으로 조정
        "FULL_EVAL_EVERY_EP": 40,
        "FULL_EVAL_STEPS": 20_000,
    })

if ARGS.use_quick_preset:
    apply_quick_preset()
    print("[CONFIG] Quick preset enabled.")


# =========================
# 2) ABLATION MATRIX (NO HMIX)
# =========================
@dataclass(frozen=True)
class Setting:
    # algo: "drqn_perslot", "dqn_perslot", "pooling_lstm", "pooling_ff"
    algo: str
    use_global: bool
    invalid_penalty: bool

def setting_name(s: Setting) -> str:
    return f"{s.algo}|G{int(s.use_global)}|P{int(s.invalid_penalty)}"

SETTINGS: List[Setting] = [
    Setting("drqn_perslot", True,  True),   # main
    Setting("drqn_perslot", False, True),   # global ablation
    Setting("drqn_perslot", True,  False),  # penalty ablation
    Setting("dqn_perslot",  True,  True),   # memory(LSTM) ablation
    Setting("pooling_lstm", True,  True),
    Setting("pooling_ff",   True,  True),
]


# =========================
# 3) I/O helpers
# =========================
def run_id(scenario: str, alpha: float, cache_size: int, seed: int, s: Setting) -> str:
    return f"{scenario}_a{alpha}_S{cache_size}_seed{seed}_{setting_name(s)}"

def safe_filename(s: str) -> str:
    bad = '<>:"/\\|?*'
    for ch in bad:
        s = s.replace(ch, "_")
    return "".join(c if c.isprintable() else "_" for c in s)

def ensure_dirs():
    os.makedirs(CONFIG["OUT_DIR"], exist_ok=True)
    os.makedirs(os.path.join(CONFIG["OUT_DIR"], "logs"), exist_ok=True)
    os.makedirs(os.path.join(CONFIG["OUT_DIR"], "ckpt"), exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def epsilon_by_step(step: int) -> float:
    s = float(CONFIG["EPSILON_START"])
    e = float(CONFIG["EPSILON_END"])
    d = float(CONFIG["EPSILON_DECAY_STEPS"])
    return float(e + (s - e) * np.exp(-step / d))

def ckpt_path(rid: str) -> str:
    return os.path.join(CONFIG["OUT_DIR"], "ckpt", f"{safe_filename(rid)}.pt")

def log_path(rid: str) -> str:
    return os.path.join(CONFIG["OUT_DIR"], "logs", f"{safe_filename(rid)}.jsonl")

def results_csv_path() -> str:
    return os.path.join(CONFIG["OUT_DIR"], "results.csv")

def summary_csv_path() -> str:
    return os.path.join(CONFIG["OUT_DIR"], "summary.csv")

def diagnostics_csv_path() -> str:
    return os.path.join(CONFIG["OUT_DIR"], "workload_diagnostics.csv")

def write_row_csv(path: str, row: dict):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)

def load_done_ids() -> set:
    p = results_csv_path()
    if not os.path.exists(p):
        return set()
    done = set()
    with open(p, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            done.add(row["run_id"])
    return done


# =========================
# 4) Workload generators (modularized in src/workload)
# =========================

# =========================
# 6) Environment
# =========================
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

class CacheEnv:
    """
    action:
      0: NOOP (hit OR empty)
      1..S: evict slot i (miss+full only)
    """
    def __init__(self, cache_size: int, use_global: bool, invalid_penalty: bool):
        self.cache_size = cache_size
        self.use_global = use_global
        self.invalid_penalty = invalid_penalty

        self.rec_denom = float(CONFIG["RECENCY_DENOM"])
        self.freq_denom = float(CONFIG["FREQ_DENOM"])
        self.hit_ema_alpha = float(CONFIG["HIT_EMA_ALPHA"])
        self.miss_streak_clip = int(CONFIG["MISS_STREAK_CLIP"])
        self.reset()

    def reset(self):
        self.cache_slots = [0] * self.cache_size
        self.t = 0
        self.access_history = {}
        self.frequency = {}
        self.hit_ema = 0.0
        self.miss_streak = 0

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
        for i, item in enumerate(self.cache_slots):
            if item == 0:
                continue
            gap = self.t - self.access_history.get(item, 0)
            rec = min(gap / self.rec_denom, 1.0)
            frq = min(self.frequency.get(item, 0) / self.freq_denom, 1.0)
            feats[i, 0] = rec
            feats[i, 1] = frq
        return feats

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
            mask[1:] = True
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
        self._update_stats(req_id)

        hit = self._has_item(req_id)
        if hit:
            self._update_global(True)
            return 1.0, True

        reward = 0.0

        if self._has_empty():
            self._fill_empty(req_id)
            self._update_global(False)
            return reward, False

        # miss+full
        if action <= 0 or action > self.cache_size:
            if self.invalid_penalty:
                reward = -1.0
            slot_idx = random.randrange(self.cache_size)
        else:
            slot_idx = action - 1

        self._evict_into(slot_idx, req_id)
        self._update_global(False)
        return reward, False


# =========================
# 7) Replay
# =========================
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
            batch_obs.append(obs_list[start:start + seq_len + 1])
            batch_ar.append(ar_list[start:start + seq_len])
        return batch_obs, batch_ar


# =========================
# 8) Models
# =========================
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
        q_noop = self.noop_head(ctx)  # (B,1)
        ctx_exp = ctx.unsqueeze(1).expand(-1, self.cache_size, -1)
        slot_in = torch.cat([slot_emb, ctx_exp], dim=-1)  # (B,S,K+H)
        q_slots = self.slot_head(slot_in).squeeze(-1)     # (B,S)
        return torch.cat([q_noop, q_slots], dim=1)        # (B,1+S)

class DRQN_PerSlot(nn.Module):
    def __init__(self, cache_size: int, use_global: bool):
        super().__init__()
        self.cache_size = cache_size
        self.use_global = use_global

        self.slot_proj = nn.Sequential(
            nn.Linear(META_DIM, CACHE_KEY_DIM), nn.ReLU(),
            nn.Linear(CACHE_KEY_DIM, CACHE_KEY_DIM), nn.ReLU()
        )

        in_dim = 3 * CACHE_KEY_DIM + (GLOBAL_DIM if use_global else 0)
        self.in_proj = nn.Sequential(nn.Linear(in_dim, LSTM_INPUT_DIM), nn.ReLU())
        self.lstm = nn.LSTM(LSTM_INPUT_DIM, HIDDEN_DIM, batch_first=True)
        self.head = PerSlotHead(cache_size, CACHE_KEY_DIM, HIDDEN_DIM)

    def init_hidden(self, B: int):
        h = torch.zeros(1, B, HIDDEN_DIM, device=DEVICE)
        c = torch.zeros(1, B, HIDDEN_DIM, device=DEVICE)
        return (h, c)

    def forward_step(self, cache_feat: torch.Tensor, global_feat: torch.Tensor, hidden):
        slot_emb = self.slot_proj(cache_feat)   # (B,S,K)
        pooled = pool_meanmaxmin(slot_emb)      # (B,3K)
        x = pooled if not self.use_global else torch.cat([pooled, global_feat], dim=1)
        x = self.in_proj(x).unsqueeze(1)        # (B,1,I)
        out, hidden = self.lstm(x, hidden)
        ctx = out[:, -1, :]
        q = self.head(slot_emb, ctx)
        return q, hidden

class DQN_PerSlot(nn.Module):
    def __init__(self, cache_size: int, use_global: bool):
        super().__init__()
        self.cache_size = cache_size
        self.use_global = use_global

        self.slot_proj = nn.Sequential(
            nn.Linear(META_DIM, CACHE_KEY_DIM), nn.ReLU(),
            nn.Linear(CACHE_KEY_DIM, CACHE_KEY_DIM), nn.ReLU()
        )

        in_dim = 3 * CACHE_KEY_DIM + (GLOBAL_DIM if use_global else 0)
        self.ctx_mlp = nn.Sequential(
            nn.Linear(in_dim, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
        )
        self.head = PerSlotHead(cache_size, CACHE_KEY_DIM, HIDDEN_DIM)

    def init_hidden(self, B: int):
        return None

    def forward_step(self, cache_feat: torch.Tensor, global_feat: torch.Tensor, _hidden=None):
        slot_emb = self.slot_proj(cache_feat)
        pooled = pool_meanmaxmin(slot_emb)
        x = pooled if not self.use_global else torch.cat([pooled, global_feat], dim=1)
        ctx = self.ctx_mlp(x)
        q = self.head(slot_emb, ctx)
        return q, None

class PoolingQNet(nn.Module):
    def __init__(self, cache_size: int, use_global: bool, use_lstm: bool):
        super().__init__()
        self.cache_size = cache_size
        self.use_global = use_global
        self.use_lstm = use_lstm

        self.slot_proj = nn.Sequential(
            nn.Linear(META_DIM, CACHE_KEY_DIM), nn.ReLU(),
            nn.Linear(CACHE_KEY_DIM, CACHE_KEY_DIM), nn.ReLU()
        )
        in_dim = 3 * CACHE_KEY_DIM + (GLOBAL_DIM if use_global else 0)

        if use_lstm:
            self.in_proj = nn.Sequential(nn.Linear(in_dim, LSTM_INPUT_DIM), nn.ReLU())
            self.lstm = nn.LSTM(LSTM_INPUT_DIM, HIDDEN_DIM, batch_first=True)
            self.out = nn.Sequential(nn.Linear(HIDDEN_DIM, 128), nn.ReLU(), nn.Linear(128, cache_size + 1))
        else:
            self.ff = nn.Sequential(
                nn.Linear(in_dim, HIDDEN_DIM), nn.ReLU(),
                nn.Linear(HIDDEN_DIM, cache_size + 1)
            )

    def init_hidden(self, B: int):
        if not self.use_lstm:
            return None
        h = torch.zeros(1, B, HIDDEN_DIM, device=DEVICE)
        c = torch.zeros(1, B, HIDDEN_DIM, device=DEVICE)
        return (h, c)

    def forward_step(self, cache_feat: torch.Tensor, global_feat: torch.Tensor, hidden):
        slot_emb = self.slot_proj(cache_feat)
        pooled = pool_meanmaxmin(slot_emb)
        x = pooled if not self.use_global else torch.cat([pooled, global_feat], dim=1)
        if self.use_lstm:
            x = self.in_proj(x).unsqueeze(1)
            out, hidden = self.lstm(x, hidden)
            q = self.out(out[:, -1, :])
            return q, hidden
        q = self.ff(x)
        return q, None

def build_models(cache_size: int, s: Setting):
    if s.algo == "drqn_perslot":
        online = DRQN_PerSlot(cache_size, s.use_global).to(DEVICE)
        target = DRQN_PerSlot(cache_size, s.use_global).to(DEVICE)
    elif s.algo == "dqn_perslot":
        online = DQN_PerSlot(cache_size, s.use_global).to(DEVICE)
        target = DQN_PerSlot(cache_size, s.use_global).to(DEVICE)
    elif s.algo == "pooling_lstm":
        online = PoolingQNet(cache_size, s.use_global, use_lstm=True).to(DEVICE)
        target = PoolingQNet(cache_size, s.use_global, use_lstm=True).to(DEVICE)
    elif s.algo == "pooling_ff":
        online = PoolingQNet(cache_size, s.use_global, use_lstm=False).to(DEVICE)
        target = PoolingQNet(cache_size, s.use_global, use_lstm=False).to(DEVICE)
    else:
        raise ValueError(s.algo)

    target.load_state_dict(online.state_dict())
    return online, target


# =========================
# 9) RL core (masked action + Double DQN)
# =========================
@torch.no_grad()
def select_action(model, obs: Obs, hidden, eps: float) -> Tuple[int, object]:
    cf = torch.from_numpy(obs.cache_feat[None, :, :]).to(DEVICE)
    gf = torch.from_numpy(obs.global_feat[None, :]).to(DEVICE)
    mk = torch.from_numpy(obs.valid_mask[None, :]).to(DEVICE)

    if random.random() < eps:
        valid = torch.nonzero(mk[0], as_tuple=False).view(-1).tolist()
        a = int(random.choice(valid))
        _, hidden2 = model.forward_step(cf, gf, hidden)
        return a, hidden2

    q, hidden2 = model.forward_step(cf, gf, hidden)
    q = q.masked_fill(~mk, float("-inf"))
    a = int(torch.argmax(q, dim=1).item())
    return a, hidden2

def train_step(online, target, optimizer, replay: EpisodeReplay, cache_size: int) -> float:
    if len(replay) < int(CONFIG["START_TRAIN_AFTER_EPISODES"]):
        return 0.0

    B = int(CONFIG["BATCH_SIZE"])
    L = int(CONFIG["BURN_IN"] + CONFIG["UNROLL"])
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

    cf = torch.from_numpy(cf_np).to(DEVICE)
    gf = torch.from_numpy(gf_np).to(DEVICE)
    mk = torch.from_numpy(mk_np).to(DEVICE)
    act = torch.from_numpy(act_np).to(DEVICE)
    rew = torch.from_numpy(rew_np).to(DEVICE)
    done = torch.from_numpy(done_np).to(DEVICE)

    h_on = online.init_hidden(B)
    h_tg = target.init_hidden(B)

    q_on_all, q_tg_all = [], []
    for t in range(L + 1):
        q_on, h_on = online.forward_step(cf[:, t], gf[:, t], h_on)
        with torch.no_grad():
            q_tg, h_tg = target.forward_step(cf[:, t], gf[:, t], h_tg)
        q_on_all.append(q_on)
        q_tg_all.append(q_tg)

    q_on_all = torch.stack(q_on_all, dim=1)  # (B,L+1,A)
    q_tg_all = torch.stack(q_tg_all, dim=1)

    losses = []
    burn = int(CONFIG["BURN_IN"])
    gamma = float(CONFIG["GAMMA"])
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
    nn.utils.clip_grad_norm_(online.parameters(), float(CONFIG["GRAD_CLIP"]))
    optimizer.step()
    return float(loss.item())


# =========================
# 10) Rollout + Eval (modular baselines)
# =========================
def rollout_episode(model, req_stream: List[int], start: int, length: int,
                    cache_size: int, s: Setting, eps: float) -> Tuple[List[Obs], List[AR], float]:
    env = CacheEnv(cache_size, use_global=s.use_global, invalid_penalty=s.invalid_penalty)
    hidden = model.init_hidden(1)
    model.eval()

    obs_list, ar_list = [], []
    total_rew = 0.0

    end = min(start + length, len(req_stream))
    T = end - start
    if T < 2:
        return [], [], 0.0

    for i in range(T):
        req = req_stream[start + i]
        obs = Obs(
            cache_feat=env.get_cache_features(),
            global_feat=env.get_global_features(),
            valid_mask=env.valid_action_mask(req),
        )
        obs_list.append(obs)
        a, hidden = select_action(model, obs, hidden, eps)
        r, _ = env.step(req, a)
        total_rew += float(r)
        ar_list.append(AR(action=a, reward=float(r), done=(i == T - 1)))

    last_req = req_stream[end - 1]
    obs_list.append(Obs(
        cache_feat=env.get_cache_features(),
        global_feat=env.get_global_features(),
        valid_mask=env.valid_action_mask(last_req),
    ))
    return obs_list, ar_list, total_rew

BASELINE_CACHE: Dict[BaselineKey, Dict[str, float]] = {}


def make_env_for_eval(cache_size: int, s: Setting) -> CacheEnv:
    return CacheEnv(cache_size, use_global=s.use_global, invalid_penalty=s.invalid_penalty)


def make_obs_for_eval(env: CacheEnv, req: int) -> Obs:
    return Obs(
        cache_feat=env.get_cache_features(),
        global_feat=env.get_global_features(),
        valid_mask=env.valid_action_mask(req),
    )


def eval_policy(
    model,
    scenario: str,
    alpha: float,
    test_stream: List[int],
    cache_size: int,
    s: Setting,
    eval_kind: str,
) -> Dict[str, float]:
    baseline_names = list(CONFIG["BASELINES"])
    return evaluate_policy_with_baselines(
        model=model,
        scenario=scenario,
        alpha=alpha,
        test_stream=test_stream,
        cache_size=cache_size,
        s=s,
        eval_kind=eval_kind,
        baseline_names=baseline_names,
        baseline_cache=BASELINE_CACHE,
        build_baselines_fn=build_baselines,
        make_env_fn=make_env_for_eval,
        make_obs_fn=make_obs_for_eval,
        select_action_fn=select_action,
    )


# =========================
# 11) Checkpoint / Resume
# =========================
@dataclass
class TrainState:
    rid: str
    scenario: str
    alpha: float
    cache_size: int
    seed: int
    setting: str
    ep_done: int
    global_step: int
    train_cursor: int
    total_updates: int
    loss_tail: List[float]

def save_ckpt(rid: str, online, target, optimizer, replay: EpisodeReplay, st: TrainState):
    torch.save({
        "config": CONFIG,
        "state": asdict(st),
        "online": online.state_dict(),
        "target": target.state_dict(),
        "opt": optimizer.state_dict(),
        "replay": replay.episodes,
    }, ckpt_path(rid))

def load_ckpt(rid: str):
    p = ckpt_path(rid)
    if not os.path.exists(p):
        return None
    return torch.load(p, map_location=DEVICE)


# =========================
# 12) Train one run
# =========================
def train_one_run(
    scenario: str,
    alpha: float,
    cache_size: int,
    seed: int,
    s: Setting,
    train_ids: List[int],
    test_fast: List[int],
    test_full: List[int],
):
    ensure_dirs()
    rid = run_id(scenario, alpha, cache_size, seed, s)

    done = load_done_ids()
    if rid in done:
        return

    set_seed(seed)

    loaded = load_ckpt(rid)
    if loaded is not None:
        st = TrainState(**loaded["state"])
        online, target = build_models(cache_size, s)
        online.load_state_dict(loaded["online"])
        target.load_state_dict(loaded["target"])
        optimizer = optim.Adam(online.parameters(), lr=float(CONFIG["LR"]))
        optimizer.load_state_dict(loaded["opt"])
        replay = EpisodeReplay(int(CONFIG["REPLAY_MAX_EPISODES"]))
        replay.episodes = loaded["replay"]
        print(f"[RESUME] {rid} from ep={st.ep_done} cursor={st.train_cursor} upd={st.total_updates}", flush=True)
    else:
        online, target = build_models(cache_size, s)
        optimizer = optim.Adam(online.parameters(), lr=float(CONFIG["LR"]))
        replay = EpisodeReplay(int(CONFIG["REPLAY_MAX_EPISODES"]))
        st = TrainState(
            rid=rid, scenario=scenario, alpha=float(alpha), cache_size=int(cache_size),
            seed=int(seed), setting=setting_name(s),
            ep_done=0, global_step=0, train_cursor=0, total_updates=0, loss_tail=[]
        )

    lp = log_path(rid)
    flog = open(lp, "a", encoding="utf-8")

    ep_len = int(CONFIG["EPISODE_LEN"])
    max_eps = int(CONFIG["MAX_TRAIN_EPISODES"])
    upd_per_ep = int(CONFIG["UPDATES_PER_EPISODE"])
    target_every = int(CONFIG["TARGET_UPDATE_EVERY_UPDATES"])

    t0 = time.time()
    pbar = tqdm(range(st.ep_done + 1, max_eps + 1), desc=f"TRAIN {rid}", leave=True)

    try:
        for ep in pbar:
            if st.train_cursor + ep_len >= len(train_ids):
                break

            eps = epsilon_by_step(st.global_step)
            obs_list, ar_list, total_rew = rollout_episode(
                online, train_ids, st.train_cursor, ep_len, cache_size, s, eps
            )
            st.train_cursor += ep_len
            st.global_step += ep_len
            st.ep_done = ep

            L = int(CONFIG["BURN_IN"] + CONFIG["UNROLL"])
            if len(ar_list) >= (L + 2):
                replay.add_episode(obs_list, ar_list)

            online.train()
            avg_loss = 0.0
            for _ in range(upd_per_ep):
                loss = train_step(online, target, optimizer, replay, cache_size)
                st.total_updates += 1
                st.loss_tail.append(float(loss))
                if len(st.loss_tail) > 2000:
                    st.loss_tail = st.loss_tail[-2000:]
                avg_loss += float(loss)

                if st.total_updates % target_every == 0:
                    target.load_state_dict(online.state_dict())

            avg_loss /= max(1, upd_per_ep)
            hit_proxy = (total_rew / ep_len) * 100.0

            eval_rec = {}
            if ep % int(CONFIG["FAST_EVAL_EVERY_EP"]) == 0:
                res = eval_policy(online, scenario, alpha, test_fast, cache_size, s, eval_kind="fast")
                eval_rec.update({f"fast_{k}": v for k, v in res.items()})

            if ep % int(CONFIG["FULL_EVAL_EVERY_EP"]) == 0:
                res = eval_policy(online, scenario, alpha, test_full, cache_size, s, eval_kind="full")
                eval_rec.update({f"full_{k}": v for k, v in res.items()})

            pbar.set_postfix({
                "eps": f"{eps:.3f}",
                "hit%": f"{hit_proxy:.2f}",
                "loss": f"{avg_loss:.4f}",
                "replay": len(replay),
                "upd": st.total_updates,
            })

            rec = {
                "run_id": rid,
                "scenario": scenario,
                "alpha": alpha,
                "cache_size": cache_size,
                "seed": seed,
                "setting": setting_name(s),
                "episode": ep,
                "global_step": st.global_step,
                "epsilon": eps,
                "train_hit_proxy": hit_proxy,
                "avg_loss_ep": avg_loss,
                "replay_episodes": len(replay),
                "total_updates": st.total_updates,
                "wall_sec": time.time() - t0,
            }
            rec.update(eval_rec)
            flog.write(json.dumps(rec) + "\n")
            flog.flush()

            if CONFIG["SAVE_CKPT"] and (ep % int(CONFIG["SAVE_CKPT_EVERY_EP"]) == 0):
                save_ckpt(rid, online, target, optimizer, replay, st)

    finally:
        flog.close()

    # final eval: full
    final = eval_policy(online, scenario, alpha, test_full, cache_size, s, eval_kind="full")

    row = {
        "run_id": rid,
        "scenario": scenario,
        "alpha": float(alpha),
        "cache_size": int(cache_size),
        "seed": int(seed),
        "setting": setting_name(s),
        "algo": s.algo,
        "use_global": int(s.use_global),
        "invalid_penalty": int(s.invalid_penalty),

        "train_episodes": int(st.ep_done),
        "total_updates": int(st.total_updates),
        "final_loss_tail_mean": float(np.mean(st.loss_tail[-1000:])) if st.loss_tail else 0.0,

        "rl_hit": float(final["rl_hit"]),
        "wall_sec": float(time.time() - t0),
    }

    for name in CONFIG["BASELINES"]:
        row[f"baseline_hit_{name}"] = float(final.get(f"baseline_hit_{name}", 0.0))
        row[f"rl_minus_baseline_{name}"] = float(final.get(f"rl_minus_baseline_{name}", 0.0))
    write_row_csv(results_csv_path(), row)

    if CONFIG["SAVE_CKPT"]:
        save_ckpt(rid, online, target, optimizer, replay, st)

    baseline_msg = ' '.join([f"{str(name).upper()} {row.get('baseline_hit_' + str(name), 0.0):.2f}" for name in CONFIG['BASELINES']])
    print(f"\n[DONE] {rid} | RL {row['rl_hit']:.2f}  {baseline_msg}", flush=True)


# =========================
# 13) Summary aggregation
# =========================
def mean_std(xs: List[float]) -> Tuple[float, float]:
    xs = np.asarray(xs, dtype=np.float64)
    return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0

def build_summary():
    p = results_csv_path()
    if not os.path.exists(p):
        return

    rows = []
    with open(p, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    group: Dict[Tuple[str, str, str, str], List[dict]] = {}
    for row in rows:
        k = (row["scenario"], row["alpha"], row["cache_size"], row["setting"])
        group.setdefault(k, []).append(row)

    out = []
    for (scenario, alpha, cache_size, setting), rs in sorted(
        group.items(),
        key=lambda x: (x[0][0], float(x[0][1]), int(x[0][2]), x[0][3])
    ):
        rl = [float(x["rl_hit"]) for x in rs]
        rl_m, rl_s = mean_std(rl)

        row_out = {
            "scenario": scenario,
            "alpha": float(alpha),
            "cache_size": int(cache_size),
            "setting": setting,
            "n": len(rs),
            "rl_mean": rl_m,
            "rl_std": rl_s,
        }

        for baseline_name in CONFIG["BASELINES"]:
            baseline_col = f"baseline_hit_{baseline_name}"
            delta_col = f"rl_minus_baseline_{baseline_name}"
            baseline_vals = [float(x.get(baseline_col, 0.0)) for x in rs]
            delta_vals = [float(x.get(delta_col, 0.0)) for x in rs]
            b_m, b_s = mean_std(baseline_vals)
            d_m, d_s = mean_std(delta_vals)
            row_out[f"{baseline_name}_mean"] = b_m
            row_out[f"{baseline_name}_std"] = b_s
            row_out[f"rl_minus_{baseline_name}_mean"] = d_m
            row_out[f"rl_minus_{baseline_name}_std"] = d_s

        out.append(row_out)

    if out:
        with open(summary_csv_path(), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader()
            for row in out:
                w.writerow(row)


# =========================
# 14) Master loop (trace shared per (scenario,alpha,seed))
# =========================
def run_all():
    ensure_dirs()
    done = load_done_ids()

    tasks = []
    for scenario in CONFIG["SCENARIOS"]:
        for alpha in CONFIG["ZIPF_ALPHAS"]:
            for cache_size in CONFIG["CACHE_SIZES"]:
                for seed in CONFIG["SEEDS"]:
                    for s in SETTINGS:
                        rid = run_id(scenario, float(alpha), int(cache_size), int(seed), s)
                        if rid in done:
                            continue
                        tasks.append((scenario, float(alpha), int(cache_size), int(seed), s))

    print(f"Remaining runs: {len(tasks)}")

    # trace cache: per (scenario,alpha,seed) 공유
    stream_cache: Dict[Tuple[str, float, int], Dict[str, List[int]]] = {}

    master = tqdm(tasks, desc="ALL RUNS", leave=True)
    for scenario, alpha, cache_size, seed, s in master:
        master.set_postfix({
            "scenario": scenario, "alpha": alpha, "S": cache_size, "seed": seed, "setting": setting_name(s)
        })

        gk = (scenario, float(alpha), int(seed))
        if gk not in stream_cache:
            print(f"\n[TRACE] building trace for scenario={scenario}, alpha={alpha}, seed={seed} ...", flush=True)
            req_stream = build_trace(CONFIG, scenario, alpha, seed, set_seed)
            phases = int(CONFIG["HOTSHIFT_PHASES"]) if scenario == "hotshift" else 1
            diag = workload_diagnostics(
                trace=req_stream,
                vocab_size=int(CONFIG["VOCAB_SIZE"]),
                phases=phases,
                top_k=min(100, int(CONFIG["VOCAB_SIZE"])),
            )
            diag_row = {
                "scenario": scenario,
                "alpha": float(alpha),
                "seed": int(seed),
                **diag,
            }
            write_row_csv(diagnostics_csv_path(), diag_row)
            print(
                "[DIAG]"
                f" scenario={scenario} alpha={alpha} seed={seed}"
                f" js_mean={diag['diag_js_mean']:.4f}"
                f" topk_overlap={diag['diag_topk_overlap_mean']:.4f}",
                flush=True,
            )

            split = int(len(req_stream) * float(CONFIG["TRAIN_RATIO"]))
            stream_cache[gk] = {
                "train": req_stream[:split],
                "fast": req_stream[split: split + int(CONFIG["FAST_EVAL_STEPS"])],
                "full": req_stream[split: split + int(CONFIG["FULL_EVAL_STEPS"])],
            }

        train_ids = stream_cache[gk]["train"]
        test_fast = stream_cache[gk]["fast"]
        test_full = stream_cache[gk]["full"]

        train_one_run(
            scenario=scenario,
            alpha=alpha,
            cache_size=cache_size,
            seed=seed,
            s=s,
            train_ids=train_ids,
            test_fast=test_fast,
            test_full=test_full,
        )

        build_summary()

    print("\nSaved:", results_csv_path())
    print("Saved:", summary_csv_path())
    print("Saved:", diagnostics_csv_path())
    print("Logs:", os.path.join(CONFIG["OUT_DIR"], "logs"))
    print("Ckpts:", os.path.join(CONFIG["OUT_DIR"], "ckpt"))


if __name__ == "__main__":
    run_all()
