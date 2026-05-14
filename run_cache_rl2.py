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

import numpy as np
import optuna
from src.baselines.factory import build_baselines
from src.evaluation.evaluator import evaluate_policy_with_baselines, BaselineKey
from src.models.drqn import (
    EpisodeReplay,
    Obs,
    build_models,
    make_env_for_eval,
    make_obs_for_eval,
    rollout_episode,
    select_action,
    train_step,
)
from src.workload.builder import build_trace

# ---- Dependency check ----
try:
    import torch
    import torch.optim as optim
except Exception as e:
    raise RuntimeError(
        "PyTorch가 필요함. 예) pip install torch\n"
        f"원인: {repr(e)}"
    )

try:
    from optuna.exceptions import TrialPruned
except Exception as e:
    raise RuntimeError(
        "Optuna가 필요함. 예) pip install optuna\n"
        f"원인: {repr(e)}"
    )


# =========================
# 0) CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="out", help="결과 저장 폴더 (기본: ./out)")
    p.add_argument("--device", type=str, default="cuda", help="cpu | cuda | cuda:0 등 (기본: cuda)")
    p.add_argument("--use_quick_preset", action="store_true", help="빠른 실험용 축소 프리셋 적용")
    p.add_argument("--optuna_trials", type=int, default=40, help="Optuna trial 횟수 (기본: 40)")
    return p.parse_args()

ARGS = parse_args()
OUT_DIR = ARGS.out_dir

def configure_torch_runtime(device_arg: str) -> torch.device:
    requested = (device_arg or "cuda").lower().strip()
    wants_cuda = requested.startswith("cuda")

    if wants_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA 장치를 요청했지만 torch.cuda.is_available()가 False입니다. "
            "CUDA 지원 PyTorch 설치/드라이버 점검 후 다시 실행하세요."
        )

    device = torch.device(requested)

    if device.type == "cuda":
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"요청한 CUDA 디바이스 인덱스({device.index})가 유효하지 않습니다. "
                f"사용 가능한 GPU 개수: {torch.cuda.device_count()}"
            )

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        print(f"DEVICE: {device} ({torch.cuda.get_device_name(dev_idx)})")
    else:
        print(f"DEVICE: {device} (경고: GPU 가속 비활성화)")

    return device


DEVICE = configure_torch_runtime(ARGS.device)


# =========================
# 1) GLOBAL CONFIG (paper-grade default)
# =========================
CONFIG = {
    "OUT_DIR": OUT_DIR,
    "EXPERIMENT_TAG": "full",

    # workload
    "VOCAB_SIZE": 10_000,
    "NUM_REQUESTS": 1_000_000,
    "TRAIN_RATIO": 0.8,

    # ✅ alpha list (요청)
    "ZIPF_ALPHAS": [1.3, 1.4, 1.5, 1.6, 1.7, 1.8],

    # cache sizes / scenarios
    "CACHE_SIZES": [16, 64],
    "SCENARIOS": ["zipf"],

    # baseline list (Step A multi-baseline-ready schema)
    "BASELINES": ["lru", "arc"],


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

    # feature scaling (RL): static fallback + dynamic default
    "RECENCY_DENOM": 2000.0,
    "FREQ_DENOM": 200.0,
    "FEATURE_SCALING_MODE": "dynamic",
    "SCALER_EMA_ALPHA": 0.02,
    "SCALER_EPS": 1e-6,
    "SCALER_MIN_SCALE": 1.0,
    "SCALER_PERCENTILE": 90.0,

    # global features + reward shaping
    "HIT_EMA_ALPHA": 0.01,
    "MISS_STREAK_CLIP": 200,
    "REWARD_MODE": "adaptive",
    "REWARD_NORM_EPS": 1e-6,
    "REWARD_CLIP": 2.0,
    "INVALID_ACTION_REWARD": -1.0,

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
        "EXPERIMENT_TAG": "quick",
        "NUM_REQUESTS": 250_000,

        # alpha는 요청대로 1.3~1.8 유지
        "ZIPF_ALPHAS": [1.3, 1.4, 1.5, 1.6, 1.7, 1.8],

        "CACHE_SIZES": [16, 64],
        "SCENARIOS": ["zipf"],
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
    # algo: "drqn_perslot", "pooling_lstm"
    algo: str
    use_global: bool
    invalid_penalty: bool

def setting_name(s: Setting) -> str:
    return f"{s.algo}|G{int(s.use_global)}|P{int(s.invalid_penalty)}"

SETTINGS: List[Setting] = [
    Setting("drqn_perslot", True, False),
    Setting("pooling_lstm", True, True),
]


# =========================
# 3) I/O helpers
# =========================
def run_id(scenario: str, alpha: float, cache_size: int, seed: int, s: Setting) -> str:
    tag = str(CONFIG.get("EXPERIMENT_TAG", "default"))
    return f"{tag}_{scenario}_a{alpha}_S{cache_size}_seed{seed}_{setting_name(s)}"

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
# 6-10) DRQN model/env/replay (modularized in src/models/drqn.py)
# =========================
BASELINE_CACHE: Dict[BaselineKey, Dict[str, float]] = {}


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
        make_env_fn=lambda cs, setting: make_env_for_eval(cs, setting, CONFIG),
        make_obs_fn=make_obs_for_eval,
        select_action_fn=lambda m, o, h, eps: select_action(m, o, h, eps, DEVICE),
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
    trial: optuna.trial.Trial | None = None,
    persist_result: bool = True,
    run_suffix: str | None = None,
):
    ensure_dirs()
    rid = run_id(scenario, alpha, cache_size, seed, s)
    if run_suffix:
        rid = f"{rid}_{run_suffix}"

    if persist_result:
        done = load_done_ids()
        if rid in done:
            return None

    set_seed(seed)

    loaded = load_ckpt(rid)
    if loaded is not None:
        st = TrainState(**loaded["state"])
        online, target = build_models(cache_size, s, DEVICE)
        online.load_state_dict(loaded["online"])
        target.load_state_dict(loaded["target"])
        optimizer = optim.Adam(online.parameters(), lr=float(CONFIG["LR"]))
        optimizer.load_state_dict(loaded["opt"])
        replay = EpisodeReplay(int(CONFIG["REPLAY_MAX_EPISODES"]))
        replay.episodes = loaded["replay"]
        print(f"[RESUME] {rid} from ep={st.ep_done} cursor={st.train_cursor} upd={st.total_updates}", flush=True)
    else:
        online, target = build_models(cache_size, s, DEVICE)
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
    print(f"[RUN START] {rid} | scenario={scenario} alpha={alpha} cache={cache_size} seed={seed}", flush=True)

    interrupted = False
    try:
        for ep in range(st.ep_done + 1, max_eps + 1):
            if st.train_cursor + ep_len >= len(train_ids):
                break

            eps = epsilon_by_step(st.global_step)
            obs_list, ar_list, rollout_stats = rollout_episode(
                online, train_ids, st.train_cursor, ep_len, cache_size, s, eps, CONFIG, DEVICE
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
                loss = train_step(online, target, optimizer, replay, cache_size, CONFIG, DEVICE)
                st.total_updates += 1
                st.loss_tail.append(float(loss))
                if len(st.loss_tail) > 2000:
                    st.loss_tail = st.loss_tail[-2000:]
                avg_loss += float(loss)

                if st.total_updates % target_every == 0:
                    target.load_state_dict(online.state_dict())

            avg_loss /= max(1, upd_per_ep)
            total_rew = float(rollout_stats["total_reward"])
            hit_count = float(rollout_stats["hit_count"])
            hit_proxy = (hit_count / ep_len) * 100.0

            eval_rec = {}
            if ep % int(CONFIG["FAST_EVAL_EVERY_EP"]) == 0:
                res = eval_policy(online, scenario, alpha, test_fast, cache_size, s, eval_kind="fast")
                eval_rec.update({f"fast_{k}": v for k, v in res.items()})
                if trial is not None:
                    trial.report(hit_proxy, ep)
                    if trial.should_prune():
                        raise TrialPruned(f"Pruned at ep={ep}, hit_proxy={hit_proxy:.4f}")

            if ep % int(CONFIG["FULL_EVAL_EVERY_EP"]) == 0:
                res = eval_policy(online, scenario, alpha, test_full, cache_size, s, eval_kind="full")
                eval_rec.update({f"full_{k}": v for k, v in res.items()})


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
                "train_total_reward": total_rew,
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

    except KeyboardInterrupt:
        if CONFIG["SAVE_CKPT"]:
            save_ckpt(rid, online, target, optimizer, replay, st)
        print(
            f"\n[INTERRUPTED] {rid} | checkpoint saved at ep={st.ep_done}, "
            f"step={st.global_step}, cursor={st.train_cursor}",
            flush=True,
        )
        raise SystemExit(130)

    finally:
        flog.close()

    if interrupted:
        return None

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
    if persist_result:
        write_row_csv(results_csv_path(), row)

    if CONFIG["SAVE_CKPT"]:
        save_ckpt(rid, online, target, optimizer, replay, st)

    baseline_msg = ' '.join([f"{str(name).upper()} {row.get('baseline_hit_' + str(name), 0.0):.2f}" for name in CONFIG['BASELINES']])
    print(f"\n[DONE] {rid} | RL {row['rl_hit']:.2f}  {baseline_msg}", flush=True)
    return row


def build_stream_cache() -> Dict[Tuple[str, float, int], Dict[str, List[int]]]:
    stream_cache: Dict[Tuple[str, float, int], Dict[str, List[int]]] = {}
    for scenario in CONFIG["SCENARIOS"]:
        for alpha in CONFIG["ZIPF_ALPHAS"]:
            for seed in CONFIG["SEEDS"]:
                gk = (scenario, float(alpha), int(seed))
                req_stream = build_trace(CONFIG, scenario, alpha, seed, set_seed)
                split = int(len(req_stream) * float(CONFIG["TRAIN_RATIO"]))
                stream_cache[gk] = {
                    "train": req_stream[:split],
                    "fast": req_stream[split: split + int(CONFIG["FAST_EVAL_STEPS"])],
                    "full": req_stream[split: split + int(CONFIG["FULL_EVAL_STEPS"])],
                }
    return stream_cache


def objective(trial: optuna.trial.Trial, stream_cache: Dict[Tuple[str, float, int], Dict[str, List[int]]]) -> float:
    CONFIG["LR"] = trial.suggest_float("LR", 1e-5, 1e-3, log=True)
    CONFIG["GAMMA"] = trial.suggest_float("GAMMA", 0.9, 0.999)
    CONFIG["UNROLL"] = trial.suggest_int("UNROLL", 20, 80, step=10)
    CONFIG["BATCH_SIZE"] = trial.suggest_categorical("BATCH_SIZE", [16, 32, 64])

    scenario_grid = [
        ("zipf", 1.3, 16),
        ("zipf", 1.3, 64),
        ("zipf", 1.8, 16),
        ("zipf", 1.8, 64),
    ]

    seed = int(CONFIG["SEEDS"][0])
    setting = SETTINGS[0]
    scores: List[float] = []

    for idx, (scenario, alpha, cache_size) in enumerate(scenario_grid):
        gk = (scenario, float(alpha), seed)
        streams = stream_cache[gk]
        row = train_one_run(
            scenario=scenario,
            alpha=alpha,
            cache_size=cache_size,
            seed=seed,
            s=setting,
            train_ids=streams["train"],
            test_fast=streams["fast"],
            test_full=streams["full"],
            trial=trial,
            persist_result=False,
            run_suffix=f"optuna_t{trial.number}_s{idx}",
        )
        if row is None:
            return 0.0
        scores.append(float(row["rl_hit"]))
        trial.report(float(np.mean(scores)), idx + 1)
        if trial.should_prune():
            raise TrialPruned(f"Pruned at scenario #{idx + 1} with mean={np.mean(scores):.4f}")

    return float(np.mean(scores)) if scores else 0.0


def optimize_hparams(stream_cache: Dict[Tuple[str, float, int], Dict[str, List[int]]]) -> Dict[str, float]:
    print("\n[OPTUNA] Starting hyperparameter optimization...")
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        storage="sqlite:///optuna_study.db",
        study_name="drqn_cache_tuning",
        load_if_exists=True,
    )
    study.optimize(lambda trial: objective(trial, stream_cache), n_trials=int(ARGS.optuna_trials))
    best_params = dict(study.best_params)
    with open(os.path.join(CONFIG["OUT_DIR"], "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)
    print(f"[OPTUNA] Best params: {best_params}")
    return best_params


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

    # Step 0: stream cache 생성(재사용)
    stream_cache = build_stream_cache()

    # Step 1: Optuna 최적화(딱 1회)
    best_params = optimize_hparams(stream_cache)
    CONFIG.update(best_params)
    print(f"[OPTUNA] Applied best params to CONFIG: {best_params}")

    # Step 2: 기존 실험 매트릭스 실행
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

    for scenario, alpha, cache_size, seed, s in tasks:

        gk = (scenario, float(alpha), int(seed))
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
    print("Logs:", os.path.join(CONFIG["OUT_DIR"], "logs"))
    print("Ckpts:", os.path.join(CONFIG["OUT_DIR"], "ckpt"))


if __name__ == "__main__":
    run_all()
