# How to Run

## 1) 환경 준비
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch tqdm numpy
```

## 2) 빠른 실행(권장)
```bash
python run_cache_rl2.py --out_dir out_stepb_quick --device cpu --use_quick_preset
```

## 3) 기본(긴) 실행
```bash
python run_cache_rl2.py --out_dir out_stepb_full --device cpu
```

## 4) 결과 확인
- `out_*/results.csv`: run-level RL/baseline 성능
- `out_*/summary.csv`: 집계 결과
