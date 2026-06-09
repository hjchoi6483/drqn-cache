# DRQN Cache Replacement Experiments

This repository contains the code used to train and evaluate a Deep Recurrent Q-Network (DRQN) cache replacement policy on synthetic Zipf request traces. The runner compares the learned policy against standard cache replacement baselines under the same traces, cache capacities, and random seeds.

The repository is intended to be cited from a paper as a reproducible artifact: it provides a single experiment entry point, fixed presets, resumable Optuna tuning, checkpointed training, and CSV summaries for downstream analysis.

## Contents

- [Overview](#overview)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Experiment presets](#experiment-presets)
- [Extended workloads (non-stationary and YCSB)](#extended-workloads-non-stationary-and-ycsb)
- [Baselines](#baselines)
- [Hyperparameter tuning](#hyperparameter-tuning)
- [Outputs](#outputs)
- [Resuming interrupted runs](#resuming-interrupted-runs)
- [Reproducing paper-style experiments](#reproducing-paper-style-experiments)
- [Testing](#testing)
- [Notes for reviewers](#notes-for-reviewers)

## Overview

The main workflow is implemented in `run_cache_rl2.py`:

1. Generate Zipf-distributed request streams.
2. Split each stream into training, fast-evaluation, and full-evaluation segments.
3. Optionally tune DRQN hyperparameters with Optuna.
4. Train the recurrent cache policy for every selected scenario, Zipf alpha, cache size, seed, and model setting.
5. Evaluate the learned policy and all selected baselines on identical request streams.
6. Write per-run results, aggregate summaries, logs, checkpoints, and experiment metadata.

The current final-release model setting is:

- `drqn_perslot|G1|P0`: a per-cache-slot DRQN policy with global features enabled and invalid-action penalties disabled.

## Repository structure

```text
.
├── README.md
├── requirements-colab.txt
├── run_cache_rl2.py                # Main experiment runner
├── scripts/
│   └── compare_results.py          # Paired comparison utility for result CSV files
├── src/
│   ├── baselines/                  # Classical cache replacement policies
│   ├── evaluation/                 # Shared RL/baseline evaluation logic
│   ├── models/                     # DRQN environment, replay, networks, and training step
│   └── workload/                   # Synthetic workload generation (zipf, nonstationary, ycsb)
└── tests/
    ├── test_baselines.py           # Lightweight baseline correctness checks
    └── test_workloads.py           # Workload generator sanity checks
```

## Installation

### Local environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch
pip install -r requirements-colab.txt
```

Install a CUDA-enabled PyTorch build if you plan to run on a GPU. The requirements file intentionally does not pin PyTorch because the correct package often depends on the CUDA runtime available on the target machine.

### Google Colab or managed GPU environment

Colab usually provides PyTorch already. In that case, install only the additional dependencies:

```bash
pip install -r requirements-colab.txt
```

## Quick start

Run a small CPU job to verify that the pipeline starts correctly:

```bash
python run_cache_rl2.py \
  --out_dir out_quick \
  --device cpu \
  --preset quick \
  --skip_optuna \
  --baseline_set minimal \
  --only_alpha 1.3 \
  --only_cache 16 \
  --seeds 0
```

For GPU execution, use `--device cuda` or a specific device such as `--device cuda:0`.

## Experiment presets

The runner supports four presets through `--preset`.

| Preset | Intended use | Request count | Seeds | Training episodes | Evaluation size |
| --- | --- | ---: | --- | ---: | --- |
| `quick` | Fast development and sanity checks | 250,000 | `0,1` | 80 | reduced fast/full evaluation |
| `paper_opt` | Practical paper-oriented sweep | 500,000 | `0,1,2,3,4` | 160 | intermediate evaluation |
| `paper_ext` | Extended workloads (non-stationary + YCSB) | 500,000 | `0,1,2,3,4` | 160 | intermediate evaluation |
| `full` | Largest default run | 1,000,000 | `0` | 400 | largest default evaluation |

The default workload grid uses:

- Scenario: `zipf`
- Zipf alpha values: `1.3, 1.4, 1.5, 1.6, 1.7, 1.8`
- Cache sizes: `16, 64`
- Vocabulary size: `10,000`
- Train/test split: `80% / 20%`

The `paper_ext` preset instead runs the extended scenarios `shift`, `hotshift`,
`ycsb_a`, `ycsb_b`, `ycsb_c`, and `ycsb_d` (see
[Extended workloads](#extended-workloads-non-stationary-and-ycsb)).

You can restrict a run with:

```bash
--only_alpha 1.3,1.8
--only_cache 16
--seeds 0,1,2
--only_algo drqn_perslot
--only_scenario ycsb_d        # restrict the scenario grid (replaces preset SCENARIOS)
```

## Extended workloads (non-stationary and YCSB)

Beyond the stationary `zipf` scenario, the runner provides two additional
workload families so the learned policy can be evaluated outside a single
stationary synthetic distribution. All extended scenarios emit the same
fixed-size integer key stream (ids in `1..VOCAB_SIZE`, never `0`) consumed by the
existing simulator, and every trace is a deterministic function of
`(scenario, alpha, seed)`.

### Experiment C — non-stationary synthetic workloads

| Scenario | Description | Alpha slot |
| --- | --- | --- |
| `shift` | The request distribution's Zipf skew changes once, partway through the stream. The early stationary regime covers training; the skew change lands inside the evaluation portion. | start skew (e.g. `1.3`); the end skew is `SHIFT_ALPHA_TO` |
| `hotshift` | The Zipf shape is held fixed, but the rank→key mapping is rotated by a fresh permutation every `HOTSHIFT_PERIOD` requests, so *which* concrete keys are hot drifts over time (a changing working set). | Zipf shape skew (e.g. `1.3`) |

The relevant `CONFIG` knobs are `SHIFT_ALPHA_TO` (default `1.8`), `SHIFT_FRAC`
(default `0.7`, chosen so that with `TRAIN_RATIO=0.8` the shift falls inside the
evaluation portion), and `HOTSHIFT_PERIOD` (default `50_000`).

### Experiment A — YCSB-style workloads

| Scenario | Description | Alpha slot |
| --- | --- | --- |
| `ycsb_a`, `ycsb_b`, `ycsb_c` | Keys drawn from a bounded Zipfian over `1..VOCAB_SIZE` with constant `zipf_const` (YCSB default `~0.99`). | Zipfian constant (e.g. `0.99`) |
| `ycsb_d` | Read-latest, non-stationary: the active population grows by inserting new key ids over time and reads concentrate (Zipfian) on the most-recently-inserted end, so the popular set drifts toward newer keys. | Zipfian constant (e.g. `0.99`) |

YCSB traces are generated synthetically (no external download) with a properly
bounded inverse-CDF Zipf sampler rather than `numpy.random.zipf` (which is
unbounded and, when modded, distorts the tail). The default Zipfian constant is
`YCSB_ZIPF_CONST` (`0.99`).

**Page-access simplification.** The cache simulator consumes only a sequence of
touched key ids. A YCSB read or update both touch exactly one page, and the
read/write distinction does not change *which* page is touched, so every YCSB
operation is collapsed to a single key access and the read/update ratio is
ignored. Under this page-access model, `ycsb_a` (50/50), `ycsb_b` (95/5), and
`ycsb_c` (100/0) are statistically equivalent access-key sequences; they are
still generated as separate scenarios so they can be reported distinctly and so
future write-aware extensions slot in cleanly.

### Per-scenario alpha slots (`SCENARIO_ALPHAS`)

The experiment loop pairs every scenario with the alpha grid. Because the
extended scenarios reuse the single `alpha` float slot to mean different things
(start/shape skew for non-stationary, Zipfian constant for YCSB), the
`paper_ext` preset defines an optional `CONFIG["SCENARIO_ALPHAS"]` map from
scenario name to its list of alpha-slot values:

```python
"SCENARIO_ALPHAS": {
    "shift":    [1.3],   # start skew 1.3 -> SHIFT_ALPHA_TO (1.8)
    "hotshift": [1.3],   # Zipf shape 1.3
    "ycsb_a":   [0.99], "ycsb_b": [0.99], "ycsb_c": [0.99], "ycsb_d": [0.99],
}
```

Alpha-grid iteration is routed through a single `alphas_for(scenario)` helper
(used by both `build_stream_cache` and the main task loop), which returns the
per-scenario list when present and otherwise falls back to `ZIPF_ALPHAS`. The
`zipf` scenario is absent from the map, so it always uses `ZIPF_ALPHAS` and its
behavior is unchanged. Passing `--only_alpha` overrides `alphas_for` for **all**
scenarios.

### Selecting scenarios

Use `--only_scenario` (comma-separated) to restrict the run to specific
scenarios; it replaces the preset `SCENARIOS` list, mirroring `--only_alpha` /
`--only_cache`:

```bash
--only_scenario shift,hotshift
--only_scenario ycsb_d
```

The `quick` preset also defines `SCENARIO_ALPHAS` for the extended scenarios, so
small CPU smoke runs such as
`--preset quick --only_scenario ycsb_d --only_cache 16 --seeds 0` resolve to a
single clean alpha slot without further flags.

### Notes

- **Non-stationary scenarios are not retuned.** `shift` and `hotshift` reuse the
  Zipf-tuned `best_params.json` (run with `--skip_optuna --best_params_path ...`)
  precisely to test generalization to distributions the policy did not train on.
- **YCSB is separately tuned** via the `ycsb` tuning profile (see
  [Hyperparameter tuning](#hyperparameter-tuning)).
- **Belady remains a valid offline upper bound** for every scenario: it is
  computed offline on the full evaluation trace, which is correct for
  non-stationary and YCSB traces too.

## Baselines

Baselines are selected with `--baseline_set`.

| Set | Policies |
| --- | --- |
| `minimal` | `lru`, `arc` |
| `diverse` | `lru`, `lfu`, `lruk`, `2q`, `arc`, `tinylfu`, `belady` |
| `paper` | `lru`, `lfu`, `lruk`, `2q`, `arc`, `tinylfu`, `belady` |

Implemented policies:

- Least Recently Used (`lru`)
- Least Frequently Used (`lfu`)
- LRU-K with `k=2` (`lruk`)
- 2Q (`2q`)
- Adaptive Replacement Cache (`arc`)
- TinyLFU (`tinylfu`)
- Belady's optimal offline policy (`belady`)

Belady receives the full evaluation trace, so it should be interpreted as an offline upper-bound reference rather than an online deployable policy.

## Hyperparameter tuning

Optuna tuning is enabled by default. Disable it with `--skip_optuna` when you want to use the configuration values already in `run_cache_rl2.py` or load a parameter JSON file.

```bash
python run_cache_rl2.py \
  --out_dir out_paper \
  --device cuda \
  --preset paper_opt \
  --tuning_profile paper \
  --optuna_trials 40 \
  --baseline_set paper
```

Tuning profiles define the representative grid used inside the Optuna objective:

| Profile | Scenarios | Alpha slot | Cache sizes | Seeds |
| --- | --- | --- | --- | --- |
| `quick` | `zipf` | `1.3, 1.8` | `16, 64` | first configured seed |
| `paper` | `zipf` | `1.3, 1.5, 1.8` | `16, 64` | first configured seed |
| `robust` | `zipf` | `1.3, 1.5, 1.8` | `16, 64` | up to two configured seeds |
| `ycsb` | `ycsb_a`, `ycsb_d` | `0.99` (Zipfian const) | `16, 64` | first configured seed |

The `ycsb` profile retunes on a representative stationary (`ycsb_a`) plus
non-stationary (`ycsb_d`) YCSB pair and must be run with the YCSB scenarios in
the grid (e.g. `--preset paper_ext`, optionally narrowed with `--only_scenario`).
The resulting `best_params.json` can then be loaded for a YCSB-only evaluation
run with `--skip_optuna --best_params_path ...`.

The objective balances average and difficult-case performance:

```text
objective = 0.8 * mean(scores) + 0.2 * min(scores)
```

By default, Optuna storage is written to:

```text
{OUT_DIR}/optuna_{EXPERIMENT_TAG}_{tuning_profile}.db
```

The corresponding best parameters are saved as:

```text
{OUT_DIR}/best_params.json
```

To reuse parameters without running Optuna:

```bash
python run_cache_rl2.py \
  --out_dir out_eval \
  --device cuda \
  --preset paper_opt \
  --skip_optuna \
  --best_params_path out_paper/best_params.json
```

## Outputs

Each run writes the following files under `--out_dir`:

| Path | Description |
| --- | --- |
| `results.csv` | One row per completed experiment run, including RL hit rate and baseline hit rates. |
| `summary.csv` | Aggregated means and standard deviations grouped by scenario, alpha, cache size, and setting. |
| `summary_overall.csv` | Overall aggregation by experiment setting and algorithm. |
| `summary_by_cache.csv` | Aggregation by cache size. |
| `summary_by_alpha.csv` | Aggregation by Zipf alpha. |
| `summary_hard_conditions.csv` | Aggregation for harder conditions, currently low alpha and small cache. |
| `summary_belady_gap.csv` | Gap between the learned policy and Belady when Belady is enabled. |
| `experiment_config.json` | Final configuration, CLI arguments, Git branch/commit, Optuna metadata, and timestamp. |
| `best_params.json` | Best Optuna parameters when tuning is enabled. |
| `logs/` | Per-run training logs. |
| `ckpt/` | Per-run checkpoints. |
| `incomplete_archive/` | Archived stale artifacts when interrupted runs are restarted. |

A separate paired-comparison utility is available for comparing two result files:

```bash
python scripts/compare_results.py \
  --a path/to/run_a/results.csv \
  --b path/to/run_b/results.csv \
  --metric rl_hit
```

## Resuming interrupted runs

Completed runs are detected from finalized rows in `results.csv`. If a run is missing a completed result row, the default resume behavior is to archive stale log/checkpoint artifacts and restart that run from episode 0:

```bash
python run_cache_rl2.py \
  --out_dir out_paper \
  --device cuda \
  --preset paper_opt \
  --resume_mode rerun_incomplete
```

To continue from an existing checkpoint instead, use:

```bash
python run_cache_rl2.py \
  --out_dir out_paper \
  --device cuda \
  --preset paper_opt \
  --resume_mode checkpoint
```

Optuna also supports resumable studies. With the default `--optuna_trials_mode target_total`, `--optuna_trials` means the target number of finished trials in the study, counting both COMPLETE and PRUNED trials. For example, if 25 trials are already finished and you rerun with `--optuna_trials 40`, the runner requests only 15 additional trials. Use `--optuna_trials_mode additional` if every invocation should launch the requested number of new trials regardless of existing study state.

## Reproducing paper-style experiments

A practical paper-style command is:

```bash
python run_cache_rl2.py \
  --out_dir out_paper_opt \
  --device cuda \
  --preset paper_opt \
  --tuning_profile paper \
  --optuna_trials 40 \
  --baseline_set paper \
  --resume_mode rerun_incomplete
```

For deterministic evaluation using a previously tuned configuration:

```bash
python run_cache_rl2.py \
  --out_dir out_paper_eval \
  --device cuda \
  --preset paper_opt \
  --skip_optuna \
  --best_params_path out_paper_opt/best_params.json \
  --baseline_set paper \
  --resume_mode rerun_incomplete
```

For a smaller reviewer-run subset:

```bash
python run_cache_rl2.py \
  --out_dir out_reviewer_subset \
  --device cpu \
  --preset quick \
  --skip_optuna \
  --baseline_set diverse \
  --only_alpha 1.3,1.8 \
  --only_cache 16 \
  --seeds 0
```

### Extended workloads (non-stationary + YCSB)

The extended experiments use the `paper_ext` preset. Non-stationary scenarios
reuse the Zipf-tuned parameters, while YCSB is retuned separately.

```bash
# A) Non-stationary + YCSB-stationary evaluation reusing the Zipf-tuned params (no tuning)
python run_cache_rl2.py --out_dir out_ext --device cuda --preset paper_ext \
  --skip_optuna --best_params_path out_paper_opt/best_params.json \
  --baseline_set paper --only_scenario shift,hotshift,ycsb_a,ycsb_b,ycsb_c \
  --resume_mode rerun_incomplete

# B) YCSB retuning (separate study), then YCSB evaluation with the tuned params
python run_cache_rl2.py --out_dir out_ycsb_tune --device cuda --preset paper_ext \
  --tuning_profile ycsb --optuna_trials 20 --baseline_set paper \
  --only_scenario ycsb_a,ycsb_b,ycsb_c,ycsb_d --resume_mode rerun_incomplete

# (If you prefer to evaluate all YCSB with the retuned params explicitly:)
python run_cache_rl2.py --out_dir out_ycsb_eval --device cuda --preset paper_ext \
  --skip_optuna --best_params_path out_ycsb_tune/best_params.json \
  --baseline_set paper --only_scenario ycsb_a,ycsb_b,ycsb_c,ycsb_d \
  --resume_mode rerun_incomplete
```

These runs are resumable: completed rows in `results.csv` are detected via
`load_done_ids()` inside the main loop, so an interrupted Colab session resumes
where it left off when rerun with the same command.

## Testing

Run the lightweight baseline checks with:

```bash
python tests/test_baselines.py
```

Run the workload generator sanity checks (validity, determinism, distribution
shift / hot-key rotation / YCSB-D temporal drift, and `build_trace` dispatch)
with:

```bash
python tests/test_workloads.py
```

Run a syntax check for all Python modules with:

```bash
python -m compileall -q .
```

## Notes for reviewers

- The main reported metric is hit rate percentage. For each run, `results.csv` includes `rl_hit`, `baseline_hit_{name}`, and `rl_minus_baseline_{name}` columns.
- Belady is computed offline with access to the full evaluation stream; it is included as an upper-bound reference and remains valid for the non-stationary and YCSB scenarios.
- The workload generator supports stationary Zipf, non-stationary (`shift`, `hotshift`), and YCSB (`ycsb_a`–`ycsb_d`) traces. The modular workload interface is in `src/workload/builder.py` (`zipf.py`, `nonstationary.py`, `ycsb.py`) if additional scenarios are needed.
- Checkpoints include model weights, optimizer state, replay contents, and training state so that long runs can be resumed.
- `experiment_config.json` records the Git branch and commit, making it easier to match results to the exact code revision used for an experiment.
