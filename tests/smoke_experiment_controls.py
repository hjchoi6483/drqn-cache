import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.argv = ['run_cache_rl2.py', '--device', 'cpu', '--skip_optuna']
r = importlib.import_module('run_cache_rl2')

assert r._parse_csv_numbers('1.3,1.4', float, 'alpha') == [1.3, 1.4]
assert r._parse_csv_numbers('16,64', int, 'cache') == [16, 64]
assert r._parse_csv_numbers('0,1,2', int, 'seed') == [0, 1, 2]
assert r.compute_remaining_optuna_trials(25, 40, 'target_total') == 15
assert r.compute_remaining_optuna_trials(25 + 5, 40, 'target_total') == 10
assert r.compute_remaining_optuna_trials(40, 40, 'target_total') == 0
assert r.compute_remaining_optuna_trials(45, 40, 'target_total') == 0
assert r.compute_remaining_optuna_trials(25, 40, 'additional') == 40
assert r.count_finished_optuna_trials({r.TrialState.COMPLETE: 25, r.TrialState.PRUNED: 5}) == 30
r.apply_baseline_set(r.CONFIG, 'minimal')
assert r.CONFIG['BASELINES'] == ['lru', 'arc']

assert r.is_completed_result_row({
    'run_id': 'done',
    'rl_hit': '12.5',
    'train_episodes': '3',
    'total_updates': '4',
})
assert not r.is_completed_result_row({'run_id': 'partial', 'rl_hit': '', 'train_episodes': '3', 'total_updates': '4'})

old_out_dir = r.CONFIG['OUT_DIR']
with tempfile.TemporaryDirectory() as td:
    r.CONFIG['OUT_DIR'] = td
    r.ensure_dirs()
    rid = 'quick_zipf_a1.3_S16_seed0_drqn_perslot|G1|P0'
    Path(r.ckpt_path(rid)).write_text('checkpoint')
    Path(r.log_path(rid)).write_text('log')
    moved = r.archive_incomplete_artifacts(rid)
    assert len(moved) == 2
    assert not Path(r.ckpt_path(rid)).exists()
    assert not Path(r.log_path(rid)).exists()
    assert all(Path(path).exists() for path in moved)
r.CONFIG['OUT_DIR'] = old_out_dir

p = Path('tmp_best_params.json')
p.write_text(json.dumps({'LR': 1e-4, 'GAMMA': 0.95}))
data = r.load_best_params(str(p))
assert data['LR'] == 1e-4
r.apply_paper_opt_preset(r.CONFIG)
assert r.CONFIG['EXPERIMENT_TAG'] == 'paper_opt'
assert r.CONFIG['MAX_TRAIN_EPISODES'] == 160
p.unlink()
print('smoke_experiment_controls passed')
