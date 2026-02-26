# How to Run

## 1) 환경 준비
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch tqdm numpy
```

## 2) 빠른 실행(권장)
Step B 결과가 정상 동작하는지 빠르게 확인합니다.
```bash
python run_cache_rl2.py --out_dir out_stepb_quick --device cpu --use_quick_preset
```

## 3) 기본(긴) 실행
```bash
python run_cache_rl2.py --out_dir out_stepb_full --device cpu
```

## 4) Step B에서 추가된 핵심 결과 확인
실행 후 아래 파일을 확인하면 zipf vs hotshift 차이를 수치로 바로 확인할 수 있습니다.

- `out_*/workload_diagnostics.csv`
  - `diag_js_mean`: phase 간 Jensen-Shannon divergence 평균
  - `diag_topk_overlap_mean`: phase 간 top-k 겹침 비율 평균
  - hotshift는 zipf보다 JS divergence가 높고 overlap이 더 낮게 나오는 것이 정상입니다.

- `out_*/results.csv`: run-level RL/baseline 성능
- `out_*/summary.csv`: 집계 결과

## 5) hotshift 강도 조절 파라미터
`run_cache_rl2.py`의 CONFIG에서 조절:
- `HOTSHIFT_PHASES`
- `HOTSHIFT_OFFSET_STEP_MODE` (`coprime_stride`, `random_stride`, `half_plus_one`, `custom`)
- `HOTSHIFT_OFFSET_STEP_CUSTOM`
- `HOTSHIFT_PHASE_SKEW`
- `HOTSHIFT_MIX_RATIO`
- `HOTSHIFT_TRANSITION` (`abrupt`, `smooth`)
