# DRQN Cache Replacement Experiments

This repository contains the code used to train and evaluate a Deep Recurrent Q-Network (DRQN) cache replacement policy on synthetic Zipf request traces. The runner compares the learned policy against standard cache replacement baselines under the same traces, cache capacities, and random seeds.

The repository is intended to be cited from a paper as a reproducible artifact: it provides a single experiment entry point, fixed presets, resumable Optuna tuning, checkpointed training, and CSV summaries for downstream analysis.

## Contents

- [Overview](#overview)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Experiment presets](#experiment-presets)
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
│   └── workload/                   # Synthetic workload generation
└── tests/
    └── test_baselines.py           # Lightweight baseline correctness checks
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

The runner supports three presets through `--preset`.

| Preset | Intended use | Request count | Seeds | Training episodes | Evaluation size |
| --- | --- | ---: | --- | ---: | --- |
| `quick` | Fast development and sanity checks | 250,000 | `0,1` | 80 | reduced fast/full evaluation |
| `paper_opt` | Practical paper-oriented sweep | 500,000 | `0,1,2,3,4` | 160 | intermediate evaluation |
| `full` | Largest default run | 1,000,000 | `0` | 400 | largest default evaluation |

The default workload grid uses:

- Scenario: `zipf`
- Zipf alpha values: `1.3, 1.4, 1.5, 1.6, 1.7, 1.8`
- Cache sizes: `16, 64`
- Vocabulary size: `10,000`
- Train/test split: `80% / 20%`

You can restrict a run with:

```bash
--only_alpha 1.3,1.8
--only_cache 16
--seeds 0,1,2
--only_algo drqn_perslot
```

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

| Profile | Alpha values | Cache sizes | Seeds |
| --- | --- | --- | --- |
| `quick` | `1.3, 1.8` | `16, 64` | first configured seed |
| `paper` | `1.3, 1.5, 1.8` | `16, 64` | first configured seed |
| `robust` | `1.3, 1.5, 1.8` | `16, 64` | up to two configured seeds |

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

## Testing

Run the lightweight baseline checks with:

```bash
python tests/test_baselines.py
```

Run a syntax check for all Python modules with:

```bash
python -m compileall -q .
```

## Notes for reviewers

- The main reported metric is hit rate percentage. For each run, `results.csv` includes `rl_hit`, `baseline_hit_{name}`, and `rl_minus_baseline_{name}` columns.
- Belady is computed offline with access to the full evaluation stream; it is included as an upper-bound reference.
- The workload generator currently supports Zipf traces only. The modular workload interface is in `src/workload/builder.py` if additional scenarios are needed.
- Checkpoints include model weights, optimizer state, replay contents, and training state so that long runs can be resumed.
- `experiment_config.json` records the Git branch and commit, making it easier to match results to the exact code revision used for an experiment.
