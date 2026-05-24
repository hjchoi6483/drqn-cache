import importlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.argv = ['run_cache_rl2.py', '--device', 'cpu', '--skip_optuna']
r = importlib.import_module('run_cache_rl2')

assert r._parse_csv_numbers('1.3,1.4', float, 'alpha') == [1.3, 1.4]
assert r._parse_csv_numbers('16,64', int, 'cache') == [16, 64]
assert r._parse_csv_numbers('0,1,2', int, 'seed') == [0, 1, 2]
r.apply_baseline_set(r.CONFIG, 'minimal')
assert r.CONFIG['BASELINES'] == ['lru', 'arc']

p = Path('tmp_best_params.json')
p.write_text(json.dumps({'LR': 1e-4, 'GAMMA': 0.95}))
data = r.load_best_params(str(p))
assert data['LR'] == 1e-4
r.apply_paper_opt_preset(r.CONFIG)
assert r.CONFIG['EXPERIMENT_TAG'] == 'paper_opt'
assert r.CONFIG['MAX_TRAIN_EPISODES'] == 160
p.unlink()
print('smoke_experiment_controls passed')
