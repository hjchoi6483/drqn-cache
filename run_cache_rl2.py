# Main experiment runner for DRQN cache experiments.
# This script keeps a single entry point while separating:
# CLI, config presets/filters, training/eval, Optuna, and summaries.

from __future__ import annotations

import os
import subprocess
import json
import csv
import time
import random
import argparse
import shutil
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any

import numpy as np
import optuna
from src.baselines.factory import build_baselines
from src.evaluation.evaluator import evaluate_policy_with_baselines, BaselineKey
from src.models.drqn import (
    EpisodeReplay,
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
        "PyTorch is required. Install it with: pip install torch\n"
        f"Cause: {repr(e)}"
    )

try:
    from optuna.exceptions import TrialPruned
    from optuna.trial import TrialState
except Exception as e:
    raise RuntimeError(
        "Optuna is required. Install it with: pip install optuna\n"
        f"Cause: {repr(e)}"
    )


# =========================
# 2) CLI parsing
# =========================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="out", help="Directory where experiment outputs are written (default: ./out)")
    p.add_argument("--device", type=str, default="cuda", help="Execution device, e.g., cpu, cuda, or cuda:0 (default: cuda)")
    p.add_argument("--preset", type=str, choices=["quick", "paper_opt", "full"], default="full")
    p.add_argument("--skip_optuna", action="store_true")
    p.add_argument("--best_params_path", type=str, default=None)
    p.add_argument("--tuning_profile", type=str, choices=["quick", "paper", "robust"], default="paper")
    p.add_argument("--only_algo", type=str, default=None)
    p.add_argument("--only_alpha", type=str, default=None)
    p.add_argument("--only_cache", type=str, default=None)
    p.add_argument("--seeds", type=str, default=None)
    p.add_argument("--baseline_set", type=str, choices=["minimal", "diverse", "paper"], default="paper")
    p.add_argument("--study_name", type=str, default=None)
    p.add_argument("--optuna_storage", type=str, default=None)
    p.add_argument("--optuna_trials", type=int, default=40, help="Optuna trial budget or target number of finished trials (COMPLETE+PRUNED; default: 40)")
    p.add_argument(
        "--optuna_trials_mode",
        type=str,
        choices=["target_total", "additional"],
        default="target_total",
        help=(
            "Optuna trial semantics: target_total sets the study-wide target number of finished "
            "trials (COMPLETE+PRUNED, default); additional runs this many new trials in this process."
        ),
    )
    p.add_argument(
        "--resume_mode",
        type=str,
        choices=["rerun_incomplete", "checkpoint"],
        default="rerun_incomplete",
        help=(
            "Resume policy: rerun_incomplete restarts unfinished runs missing from results.csv (default); "
            "checkpoint resumes training from an existing checkpoint when available."
        ),
    )
    return p.parse_args()

ARGS = parse_args()
OUT_DIR = ARGS.out_dir
OPTUNA_RUN_INFO: Dict[str, Any] = {
    "optuna_completed_trials_before_run": None,
    "optuna_pruned_trials_before_run": None,
    "optuna_finished_trials_before_run": None,
    "optuna_remaining_trials_requested": None,
    "optuna_storage": None,
    "study_name": None,
}

def configure_torch_runtime(device_arg: str) -> torch.device:
    requested = (device_arg or "cuda").lower().strip()
    wants_cuda = requested.startswith("cuda")

    if wants_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA device was requested, but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build and verify the GPU driver before rerunning."
        )

    device = torch.device(requested)

    if device.type == "cuda":
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"The requested CUDA device index ({device.index}) is invalid. "
                f"Available GPU count: {torch.cuda.device_count()}"
            )

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        print(f"DEVICE: {device} ({torch.cuda.get_device_name(dev_idx)})")
    else:
        print(f"DEVICE: {device} (warning: GPU acceleration is disabled)")

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

    # Zipf alpha grid used by the reported experiments
    "ZIPF_ALPHAS": [1.3, 1.4, 1.5, 1.6, 1.7, 1.8],

    # cache sizes / scenarios
    "CACHE_SIZES": [16, 64],
    "SCENARIOS": ["zipf"],

    # paper-grade baseline set for reporting/comparison
    "BASELINES": ["lru", "lfu", "lruk", "2q", "arc", "tinylfu", "wtinylfu", "belady"],


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
    "USE_TWO_STAGE_TINYLFU": True,
    "RECENT_WINDOW_SIZE": 1000,
    "TINYLFU_COUNTER_DECAY": 0.99,
    "TINYLFU_MIN_ADMIT_COUNT": 2,
    "BYPASS_REWARD": 0.0,
    "ADMISSION_FEATURES": True,
    "REQ_FREQ_DENOM": 50.0,
    "RECENT_FREQ_DENOM": 20.0,
    "USE_ADMISSION_HEURISTIC_MASK": False,

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
# 5) Preset application
# =========================
def apply_quick_preset(config: Dict[str, Any]):
    config.update({
        "EXPERIMENT_TAG": "quick",
        "NUM_REQUESTS": 250_000,

        # Keep the same alpha grid as the larger experiments
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

        # Ensure at least one full evaluation is emitted during quick training
        "FULL_EVAL_EVERY_EP": 40,
        "FULL_EVAL_STEPS": 20_000,
    })

def apply_paper_opt_preset(config: Dict[str, Any]):
    config.update({
        "EXPERIMENT_TAG": "paper_opt",
        "NUM_REQUESTS": 500_000,
        "TRAIN_RATIO": 0.8,
        "ZIPF_ALPHAS": [1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
        "CACHE_SIZES": [16, 64],
        "SEEDS": [0, 1, 2, 3, 4],
        "EPISODE_LEN": 2500,
        "MAX_TRAIN_EPISODES": 160,
        "REPLAY_MAX_EPISODES": 500,
        "START_TRAIN_AFTER_EPISODES": 10,
        "UPDATES_PER_EPISODE": 12,
        "FAST_EVAL_EVERY_EP": 40,
        "FAST_EVAL_STEPS": 10_000,
        "FULL_EVAL_EVERY_EP": 80,
        "FULL_EVAL_STEPS": 100_000,
        "SAVE_CKPT": True,
        "SAVE_CKPT_EVERY_EP": 5,
    })


def apply_preset(config: Dict[str, Any], preset_name: str):
    if preset_name == "quick":
        apply_quick_preset(config)
    elif preset_name == "paper_opt":
        apply_paper_opt_preset(config)
    elif preset_name == "full":
        config["EXPERIMENT_TAG"] = "full"
    else:
        raise ValueError(f"Unknown preset: {preset_name}")


# =========================
# 2) ABLATION MATRIX (NO HMIX)
# =========================
@dataclass(frozen=True)
class Setting:
    # algo: "drqn_perslot"
    algo: str
    use_global: bool
    invalid_penalty: bool

def setting_name(s: Setting) -> str:
    return f"{s.algo}|G{int(s.use_global)}|P{int(s.invalid_penalty)}"

SETTINGS: List[Setting] = [
    Setting("drqn_perslot", True, False),
]


def _parse_csv_numbers(raw: str, cast, label: str) -> List[Any]:
    vals: List[Any] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(cast(tok))
        except ValueError as e:
            raise ValueError(f"Invalid {label} value '{tok}'.") from e
    if not vals:
        raise ValueError(f"{label} cannot be empty.")
    return vals


def apply_baseline_set(config: Dict[str, Any], baseline_set: str):
    baseline_sets = {
        "minimal": ["lru", "arc"],
        "diverse": ["lru", "lfu", "lruk", "2q", "arc", "tinylfu", "belady"],
        "paper": ["lru", "lfu", "lruk", "2q", "arc", "tinylfu", "wtinylfu", "belady"],
    }
    if baseline_set not in baseline_sets:
        raise ValueError(f"Unknown baseline set: {baseline_set}")
    config["BASELINES"] = list(baseline_sets[baseline_set])


def apply_cli_filters(config: Dict[str, Any], args: argparse.Namespace):
    apply_preset(config, args.preset)
    apply_baseline_set(config, args.baseline_set)

    if args.only_alpha is not None:
        config["ZIPF_ALPHAS"] = _parse_csv_numbers(args.only_alpha, float, "alpha")
    if args.only_cache is not None:
        config["CACHE_SIZES"] = _parse_csv_numbers(args.only_cache, int, "cache_size")
    if args.seeds is not None:
        config["SEEDS"] = _parse_csv_numbers(args.seeds, int, "seed")

    global SETTINGS
    if args.only_algo is not None:
        SETTINGS = [s for s in SETTINGS if s.algo == args.only_algo]
        if not SETTINGS:
            raise ValueError(f"No SETTINGS matched --only_algo={args.only_algo}")


apply_cli_filters(CONFIG, ARGS)


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
def experiment_config_path() -> str:
    return os.path.join(CONFIG["OUT_DIR"], "experiment_config.json")


def get_optuna_storage_path() -> str:
    return ARGS.optuna_storage or f"sqlite:///{CONFIG['OUT_DIR']}/optuna_{CONFIG['EXPERIMENT_TAG']}_{ARGS.tuning_profile}.db"


def get_study_name() -> str:
    algo_name = SETTINGS[0].algo if SETTINGS else "none"
    twostage = "twostage" if CONFIG.get("USE_TWO_STAGE_TINYLFU", False) else "nostage"
    default_study_name = f"{CONFIG['EXPERIMENT_TAG']}_{ARGS.tuning_profile}_{algo_name}_{twostage}"
    return ARGS.study_name or default_study_name


def count_trials_by_state(study: optuna.study.Study) -> Dict[TrialState, int]:
    counts = {state: 0 for state in TrialState}
    for trial in study.trials:
        counts[trial.state] = counts.get(trial.state, 0) + 1
    return counts


def count_finished_optuna_trials(trial_counts: Dict[TrialState, int]) -> int:
    """Count trials that should satisfy the resume target.

    Optuna marks pruned trials as terminal, so resume should not request
    replacements for them when --optuna_trials_mode target_total is used.
    """
    return trial_counts.get(TrialState.COMPLETE, 0) + trial_counts.get(TrialState.PRUNED, 0)


def compute_remaining_optuna_trials(finished_trials: int, requested_trials: int, mode: str = "target_total") -> int:
    if finished_trials < 0:
        raise ValueError(f"finished_trials must be non-negative, got {finished_trials}")
    if requested_trials < 0:
        raise ValueError(f"requested_trials must be non-negative, got {requested_trials}")
    if mode == "target_total":
        return max(0, requested_trials - finished_trials)
    if mode == "additional":
        return requested_trials
    raise ValueError(f"Unknown optuna_trials_mode: {mode}")


def update_optuna_run_info(
    completed_before: int | None,
    pruned_before: int | None,
    finished_before: int | None,
    remaining_requested: int | None,
    storage: str | None,
    study_name: str | None,
) -> None:
    OPTUNA_RUN_INFO.update({
        "optuna_completed_trials_before_run": completed_before,
        "optuna_pruned_trials_before_run": pruned_before,
        "optuna_finished_trials_before_run": finished_before,
        "optuna_remaining_trials_requested": remaining_requested,
        "optuna_storage": storage,
        "study_name": study_name,
    })


def save_experiment_config(config: Dict[str, Any], args: argparse.Namespace, settings: List[Setting], used_best_params_path: str | None):
    branch = commit = None
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        pass
    with open(experiment_config_path(), "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "args": vars(args),
            "final_config": config,
            "git_branch": branch,
            "git_commit": commit,
            "baseline_list": config["BASELINES"],
            "settings_list": [setting_name(s) for s in settings],
            "optuna_used": not args.skip_optuna,
            "optuna_trials": args.optuna_trials,
            "optuna_trials_mode": args.optuna_trials_mode,
            "optuna_completed_trials_before_run": OPTUNA_RUN_INFO.get("optuna_completed_trials_before_run"),
            "optuna_pruned_trials_before_run": OPTUNA_RUN_INFO.get("optuna_pruned_trials_before_run"),
            "optuna_finished_trials_before_run": OPTUNA_RUN_INFO.get("optuna_finished_trials_before_run"),
            "optuna_remaining_trials_requested": OPTUNA_RUN_INFO.get("optuna_remaining_trials_requested"),
            "optuna_storage": OPTUNA_RUN_INFO.get("optuna_storage") or get_optuna_storage_path(),
            "study_name": OPTUNA_RUN_INFO.get("study_name") or get_study_name(),
            "best_params_path": used_best_params_path,
            "tuning_profile": args.tuning_profile,
            "resume_mode": args.resume_mode,
        }, f, ensure_ascii=False, indent=2)

def write_row_csv(path: str, row: dict):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)

COMPLETION_REQUIRED_COLUMNS = ("run_id", "rl_hit", "train_episodes", "total_updates")


def is_completed_result_row(row: Dict[str, Any]) -> bool:
    """Return True only for result rows that look like finalized experiments."""
    if not row or any(not str(row.get(col, "")).strip() for col in COMPLETION_REQUIRED_COLUMNS):
        return False
    try:
        float(row["rl_hit"])
        int(float(row["train_episodes"]))
        int(float(row["total_updates"]))
    except (TypeError, ValueError):
        return False
    return True


def load_done_ids() -> set:
    p = results_csv_path()
    if not os.path.exists(p):
        return set()
    done = set()
    with open(p, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if is_completed_result_row(row):
                done.add(row["run_id"])
    return done


def incomplete_archive_dir(rid: str) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return os.path.join(CONFIG["OUT_DIR"], "incomplete_archive", f"{safe_filename(rid)}_{stamp}")


def archive_incomplete_artifacts(rid: str) -> List[str]:
    """Move stale checkpoint/log files aside before rerunning an unfinished set."""
    moved: List[str] = []
    paths = [ckpt_path(rid), log_path(rid)]
    existing = [path for path in paths if os.path.exists(path)]
    if not existing:
        return moved

    archive_dir = incomplete_archive_dir(rid)
    os.makedirs(archive_dir, exist_ok=True)
    for path in existing:
        dst = os.path.join(archive_dir, os.path.basename(path))
        shutil.move(path, dst)
        moved.append(dst)
    return moved


def prepare_incomplete_restart(rid: str, persist_result: bool) -> Any | None:
    """Load or reset an unfinished run according to --resume_mode."""
    if persist_result and ARGS.resume_mode == "rerun_incomplete":
        moved = archive_incomplete_artifacts(rid)
        if moved:
            print(
                f"[RERUN INCOMPLETE] {rid} has no completed result row; "
                f"archived stale artifacts and restarting from episode 0: {moved}",
                flush=True,
            )
        return None

    loaded = load_ckpt(rid)
    if loaded is not None:
        print(f"[CHECKPOINT FOUND] {rid} ({ARGS.resume_mode})", flush=True)
    return loaded


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
    baseline_names: List[str] | None = None,
) -> Dict[str, float]:
    if baseline_names is None:
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
            print(f"[SKIP DONE] {rid} already has a completed result row.", flush=True)
            return None

    set_seed(seed)

    loaded = prepare_incomplete_restart(rid, persist_result)
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
            obs_list, ar_list, ep_stats = rollout_episode(
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
            hit_proxy = (ep_stats["hit_count"] / max(1, ep_len)) * 100.0

            eval_rec = {}
            if ep % int(CONFIG["FAST_EVAL_EVERY_EP"]) == 0:
                res = eval_policy(online, scenario, alpha, test_fast, cache_size, s, eval_kind="fast")
                eval_rec.update({f"fast_{k}": v for k, v in res.items()})
                if trial is not None and "fast_rl_hit" in eval_rec:
                    trial.report(float(eval_rec["fast_rl_hit"]), ep)
                    if trial.should_prune():
                        raise TrialPruned(f"Pruned at ep={ep}, fast_rl_hit={float(eval_rec['fast_rl_hit']):.4f}")

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
                "train_total_reward": float(ep_stats["total_reward"]),
                "bypass_count": int(ep_stats["bypass_count"]),
                "admit_count": int(ep_stats["admit_count"]),
                "reject_count": int(ep_stats["reject_count"]),
                "insert_count": int(ep_stats["insert_count"]),
                "eviction_count": int(ep_stats["eviction_count"]),
                "bypass_rate": float(ep_stats["bypass_count"]) / max(1, int(ep_stats["miss_count"])),
                "admit_rate": float(ep_stats["admit_count"]) / max(1, int(ep_stats["miss_count"])),
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


def get_tuning_grid(config: Dict[str, Any], profile: str) -> List[Tuple[str, float, int, int]]:
    alphas = [1.3, 1.8] if profile == "quick" else [1.3, 1.5, 1.8]
    caches = [16, 64]
    seeds = config["SEEDS"][:1] if profile in {"quick", "paper"} else (config["SEEDS"][:2] if len(config["SEEDS"]) >= 2 else config["SEEDS"][:1])
    out = []
    for a in alphas:
        for c in caches:
            for seed in seeds:
                out.append(("zipf", float(a), int(c), int(seed)))
    return out


def suggest_hparams(trial: optuna.trial.Trial, config: Dict[str, Any]) -> None:
    config["LR"] = trial.suggest_float("LR", 3e-5, 8e-4, log=True)
    config["GAMMA"] = trial.suggest_float("GAMMA", 0.92, 0.995)
    config["UNROLL"] = trial.suggest_categorical("UNROLL", [30, 40, 60, 80])
    config["BATCH_SIZE"] = trial.suggest_categorical("BATCH_SIZE", [16, 32, 64])
    config["TARGET_UPDATE_EVERY_UPDATES"] = trial.suggest_categorical("TARGET_UPDATE_EVERY_UPDATES", [200, 500, 1000])
    config["UPDATES_PER_EPISODE"] = trial.suggest_categorical("UPDATES_PER_EPISODE", [8, 12, 16])
    config["EPSILON_DECAY_STEPS"] = trial.suggest_categorical("EPSILON_DECAY_STEPS", [100_000, 200_000, 300_000])
    if config["USE_TWO_STAGE_TINYLFU"]:
        config["RECENT_WINDOW_SIZE"] = trial.suggest_categorical("RECENT_WINDOW_SIZE", [500, 1000, 5000])
        config["TINYLFU_MIN_ADMIT_COUNT"] = trial.suggest_categorical("TINYLFU_MIN_ADMIT_COUNT", [1, 2, 3])
        config["RECENT_FREQ_DENOM"] = trial.suggest_categorical("RECENT_FREQ_DENOM", [10.0, 20.0, 50.0])


def compute_objective_score(scores: List[float]) -> float:
    if not scores:
        return 0.0
    return float(0.8 * np.mean(scores) + 0.2 * np.min(scores))


def objective(trial: optuna.trial.Trial, stream_cache: Dict[Tuple[str, float, int], Dict[str, List[int]]]) -> float:
    suggest_hparams(trial, CONFIG)

    scenario_grid = get_tuning_grid(CONFIG, ARGS.tuning_profile)
    setting = SETTINGS[0]
    scores: List[float] = []
    scenario_scores: List[Dict[str, Any]] = []

    for idx, (scenario, alpha, cache_size, seed) in enumerate(scenario_grid):
        gk = (scenario, float(alpha), int(seed))
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
        score = float(row["rl_hit"])
        scores.append(score)
        scenario_scores.append({"scenario": scenario, "alpha": alpha, "cache_size": cache_size, "seed": seed, "score": score})
        trial.report(float(np.mean(scores)), idx + 1)
        if trial.should_prune():
            raise TrialPruned(f"Pruned at scenario #{idx + 1} with mean={np.mean(scores):.4f}")

    if not scores:
        return 0.0
    mean_score = float(np.mean(scores))
    min_score = float(np.min(scores))
    std_score = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    hard_scores = [x["score"] for x in scenario_scores if x["alpha"] == 1.3 and x["cache_size"] == 16]
    hard_score = float(np.mean(hard_scores)) if hard_scores else min_score
    objective_score = compute_objective_score(scores)
    trial.set_user_attr("mean_score", mean_score)
    trial.set_user_attr("min_score", min_score)
    trial.set_user_attr("std_score", std_score)
    trial.set_user_attr("hard_score", hard_score)
    trial.set_user_attr("scenario_scores", scenario_scores)
    return float(objective_score)


def optimize_hparams(stream_cache: Dict[Tuple[str, float, int], Dict[str, List[int]]]) -> Dict[str, float]:
    print("\n[OPTUNA] Starting hyperparameter optimization...")
    study_name = get_study_name()
    storage = get_optuna_storage_path()
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )

    trial_counts = count_trials_by_state(study)
    completed_trials = trial_counts.get(TrialState.COMPLETE, 0)
    pruned_trials = trial_counts.get(TrialState.PRUNED, 0)
    finished_trials = count_finished_optuna_trials(trial_counts)
    running_trials = trial_counts.get(TrialState.RUNNING, 0)
    remaining_trials = compute_remaining_optuna_trials(
        finished_trials=finished_trials,
        requested_trials=int(ARGS.optuna_trials),
        mode=ARGS.optuna_trials_mode,
    )
    update_optuna_run_info(completed_trials, pruned_trials, finished_trials, remaining_trials, storage, study_name)

    if running_trials:
        print(
            f"[OPTUNA RESUME WARNING] study={study_name} has {running_trials} existing RUNNING trial(s); "
            "they are not counted toward the finished target.",
            flush=True,
        )

    if ARGS.optuna_trials_mode == "target_total":
        if remaining_trials == 0:
            print(
                f"[OPTUNA RESUME] study={study_name} finished={finished_trials} "
                f"(complete={completed_trials}, pruned={pruned_trials}) "
                f"target={int(ARGS.optuna_trials)} remaining=0; skipping optimization.",
                flush=True,
            )
        else:
            print(
                f"[OPTUNA RESUME] study={study_name} finished={finished_trials} "
                f"(complete={completed_trials}, pruned={pruned_trials}) "
                f"target={int(ARGS.optuna_trials)} remaining={remaining_trials}",
                flush=True,
            )
    else:
        print(
            f"[OPTUNA RESUME] study={study_name} finished={finished_trials} "
            f"(complete={completed_trials}, pruned={pruned_trials}) "
            f"mode=additional requested={int(ARGS.optuna_trials)} remaining={remaining_trials}",
            flush=True,
        )

    if remaining_trials > 0:
        study.optimize(lambda trial: objective(trial, stream_cache), n_trials=remaining_trials)

    final_counts = count_trials_by_state(study)
    final_completed_trials = final_counts.get(TrialState.COMPLETE, 0)
    final_pruned_trials = final_counts.get(TrialState.PRUNED, 0)
    final_finished_trials = count_finished_optuna_trials(final_counts)
    if final_completed_trials == 0:
        raise ValueError(
            "No completed Optuna trials exist for this study; cannot load best_params. "
            f"complete=0, pruned={final_pruned_trials}, finished={final_finished_trials}, "
            f"requested={int(ARGS.optuna_trials)}, mode={ARGS.optuna_trials_mode}, "
            f"remaining={remaining_trials}. Pruned trials count toward the resume target but Optuna "
            "still needs at least one COMPLETE trial to provide best_params; increase --optuna_trials, "
            "use --optuna_trials_mode additional, or relax pruning to run new trials."
        )

    best_params = dict(study.best_params)
    with open(os.path.join(CONFIG["OUT_DIR"], "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)
    print(f"[OPTUNA] Best params: {best_params}")
    return best_params


def load_best_params(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"best_params JSON must be object: {path}")
    return data


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

    if not out:
        return
    with open(summary_csv_path(), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        for row in out:
            w.writerow(row)

    def write_grouped(path: str, grouped_rows: Dict[Tuple[Any, ...], List[dict]], keys: List[str]):
        rows_out = []
        for k, rs in grouped_rows.items():
            rr = {kk: kv for kk, kv in zip(keys, k)}
            rl = [float(x["rl_hit"]) for x in rs]
            rr["n"] = len(rs)
            rr["rl_mean"], rr["rl_std"] = mean_std(rl)
            rr["rl_min"], rr["rl_max"] = float(np.min(rl)), float(np.max(rl))
            rr["wall_sec_mean"] = float(np.mean([float(x["wall_sec"]) for x in rs]))
            rr["final_loss_tail_mean_mean"] = float(np.mean([float(x["final_loss_tail_mean"]) for x in rs]))
            for b in CONFIG["BASELINES"]:
                bcol = f"baseline_hit_{b}"
                dcol = f"rl_minus_baseline_{b}"
                bvals = [float(x.get(bcol, 0.0)) for x in rs]
                dvals = [float(x.get(dcol, 0.0)) for x in rs]
                rr[f"{b}_mean"] = float(np.mean(bvals))
                rr[f"rl_minus_{b}_mean"] = float(np.mean(dvals))
                wins = sum(1 for dv in dvals if dv > 0.0)
                rr[f"win_count_vs_{b}"] = wins
                rr[f"win_rate_vs_{b}"] = wins / max(1, len(dvals))
            rows_out.append(rr)
        if rows_out:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
                w.writeheader()
                for r in rows_out:
                    w.writerow(r)

    overall = {}
    by_cache = {}
    by_alpha = {}
    for row in rows:
        overall.setdefault((row["setting"], row["algo"]), []).append(row)
        by_cache.setdefault((row["setting"], int(row["cache_size"])), []).append(row)
        by_alpha.setdefault((row["setting"], float(row["alpha"])), []).append(row)
    write_grouped(os.path.join(CONFIG["OUT_DIR"], "summary_overall.csv"), overall, ["setting", "algo"])
    write_grouped(os.path.join(CONFIG["OUT_DIR"], "summary_by_cache.csv"), by_cache, ["setting", "cache_size"])
    write_grouped(os.path.join(CONFIG["OUT_DIR"], "summary_by_alpha.csv"), by_alpha, ["setting", "alpha"])
    hard = [r for r in rows if float(r["alpha"]) in {1.3, 1.4} and int(r["cache_size"]) == 16]
    if hard:
        write_grouped(os.path.join(CONFIG["OUT_DIR"], "summary_hard_conditions.csv"), {("hard",): hard}, ["group"])
    if "belady" in CONFIG["BASELINES"]:
        belady = {}
        for row in rows:
            belady.setdefault((row["setting"], row["algo"]), []).append(row)
        belady_rows = []
        for k, rs in belady.items():
            rl = np.mean([float(x["rl_hit"]) for x in rs])
            b = np.mean([float(x.get("baseline_hit_belady", 0.0)) for x in rs])
            belady_rows.append({"setting": k[0], "algo": k[1], "belady_mean": b, "rl_mean": rl, "rl_minus_belady_mean": rl - b, "gap_to_belady": b - rl})
        with open(os.path.join(CONFIG["OUT_DIR"], "summary_belady_gap.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(belady_rows[0].keys()))
            w.writeheader()
            for r in belady_rows:
                w.writerow(r)


# =========================
# 14) Master loop (trace shared per (scenario,alpha,seed))
# =========================
def run_all():
    ensure_dirs()

    # Step 0: build reusable stream cache
    stream_cache = build_stream_cache()

    print(f"[BASELINES] {CONFIG['BASELINES']}")

    # Step 1: experiment mode
    used_best_params_path = None
    if ARGS.skip_optuna:
        if ARGS.best_params_path:
            print("[MODE] Fixed best_params evaluation")
            best_params = load_best_params(ARGS.best_params_path)
            CONFIG.update(best_params)
            used_best_params_path = ARGS.best_params_path
            print(f"[PARAMS] Loaded best params from {ARGS.best_params_path}: {best_params}")
        else:
            print("[MODE] Default CONFIG evaluation")
            print("[PARAMS] --skip_optuna set. Using CONFIG defaults.")
    else:
        print("[MODE] Optuna tuning enabled")
        best_params = optimize_hparams(stream_cache)
        CONFIG.update(best_params)
        print(f"[OPTUNA] Applied best params to CONFIG: {best_params}")

    save_experiment_config(CONFIG, ARGS, SETTINGS, used_best_params_path)

    # Step 2: run the configured experiment matrix
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
