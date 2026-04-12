# drqn-cache

캐시 교체(Cache Replacement) 문제를 위한 DRQN 기반 실험 러너입니다.  
현재 코드는 **2개 RL 설정**과 **2개 고전 베이스라인(LRU, ARC)**을 동일 워크로드에서 비교하도록 구성되어 있습니다.

## 핵심 특징

- **모델 설정 2종 고정**
  - `drqn_perslot|G1|P0`
  - `pooling_lstm|G1|P1`
- **워크로드 시나리오 3종 지원**
  - `zipf_static`
  - `zipf_phase_shift`
  - `zipf_random_jump`
- **Optuna 하이퍼파라미터 최적화 내장**
  - 실행 시작 시 1회 수행 후 `best_params.json` 저장
- **평가 시 RL vs LRU vs ARC 동시 비교**
  - `results.csv`/`summary.csv`에 baseline별 점수와 RL 대비 차이 기록
- **재시작(Resume) 지원**
  - run_id 기준 체크포인트(`out/ckpt/*.pt`)를 읽어 이어서 학습

---

## 프로젝트 구조

- `run_cache_rl2.py`
  - 전체 실험 오케스트레이션
  - Optuna → 본 실험 매트릭스 순서 실행
  - 결과/요약/로그/체크포인트 저장
- `src/models/drqn.py`
  - `CacheEnv`, replay buffer, DRQN 모델, action 선택/학습 유틸
- `src/workload/builder.py`
  - 시나리오별 요청 스트림 생성(`zipf_static`, `zipf_phase_shift`, `zipf_random_jump`)
- `src/evaluation/evaluator.py`
  - RL 정책 평가 + baseline 점수 캐시/계산
- `src/baselines/lru.py`
  - LRU 시뮬레이터
- `src/baselines/arc.py`
  - ARC 시뮬레이터
- `src/baselines/factory.py`
  - baseline 이름 기반 생성 팩토리

---

## 실행 방법

## 1) 환경 준비

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-colab.txt
```

## 2) Quick preset (짧은 검증용)

```bash
python run_cache_rl2.py --out_dir out_quick --device cpu --use_quick_preset --optuna_trials 30
```

Quick preset에서 주요 축소 항목:
- `NUM_REQUESTS`: 250,000
- `MAX_TRAIN_EPISODES`: 80
- `SEEDS`: `[0, 1]`
- `FULL_EVAL_STEPS`: 20,000

## 3) Full run (기본 실험)

```bash
python run_cache_rl2.py --out_dir out_full --device cpu --optuna_trials 40
```

기본(full) 설정 예:
- `NUM_REQUESTS`: 1,000,000
- `CACHE_SIZES`: `[16, 64]`
- `SCENARIOS`: `zipf_static`, `zipf_phase_shift`, `zipf_random_jump`
- `SEEDS`: `[0]`

> GPU를 쓰려면 `--device cuda` (또는 `cuda:0`)를 지정하세요.  
> CUDA를 지정했는데 장치가 없으면 즉시 에러를 발생시키도록 되어 있습니다.

---

## Optuna 최적화 방식

실행 흐름은 아래와 같습니다.

1. `build_stream_cache()`로 `(scenario, alpha, seed)`별 스트림을 미리 생성/캐시
2. `optimize_hparams()`에서 study 실행
3. best params를 `OUT_DIR/best_params.json`에 저장
4. best params를 현재 `CONFIG`에 반영 후 본 실험 시작

탐색 파라미터:
- `LR`: `1e-5 ~ 1e-3` (log)
- `GAMMA`: `0.9 ~ 0.999`
- `UNROLL`: `20 ~ 80` (step=10)
- `BATCH_SIZE`: `[16, 32, 64]`

Objective 내부 대표 시나리오 그리드:
- `("zipf_static", 0.99, 16)`
- `("zipf_static", 1.6, 64)`
- `("zipf_phase_shift", -1.0, 16)`
- `("zipf_random_jump", -1.0, 64)`

각 trial은 위 4개를 순차 학습/평가하고 `rl_hit` 평균을 최종 점수로 사용합니다.

---

## 결과 산출물

`--out_dir` 기준으로 아래 파일/폴더가 생성됩니다.

- `best_params.json`: Optuna best params
- `results.csv`: run-level 최종 결과
  - `rl_hit`
  - `baseline_hit_lru`, `baseline_hit_arc`
  - `rl_minus_baseline_lru`, `rl_minus_baseline_arc`
- `summary.csv`: `(scenario, alpha, cache_size, setting)` 그룹 집계(mean/std)
- `logs/*.jsonl`: 에피소드 단위 학습/평가 로그
- `logs/workload_*.json`: 생성한 워크로드 메타데이터
- `ckpt/*.pt`: 재시작용 체크포인트

---

## 로그 형태

실행 중 콘솔에는 핵심 이벤트만 출력됩니다.

- 시작: `[RUN START] ...`
- 재시작: `[RESUME] ...`
- 완료: `[DONE] ... | RL xx.xx LRU xx.xx ARC xx.xx`
- Optuna: `[OPTUNA] ...`

---

## 참고

- `how_to_run.md`, `core_structure_report.md`는 현재 README로 통합되어 있습니다.
- 대규모 실험은 I/O가 많으므로 빠른 디스크(SSD) 사용을 권장합니다.
