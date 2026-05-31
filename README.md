# drqn-cache

DRQN 기반 캐시 교체 정책을 학습하고, 고전 알고리즘들과 동일한 요청 트레이스에서 비교 평가하는 실험 러너입니다.

## 핵심 특징

- **RL 모델 2종 비교**
  - `drqn_perslot|G1|P0`
  - `pooling_lstm|G1|P1`
- **베이스라인 세트 선택**
  - `minimal`: `lru`, `arc`
  - `diverse`: `lru`, `lfu`, `lruk`, `2q`, `arc`, `tinylfu`, `belady`
  - `paper`: `lru`, `lfu`, `lruk`, `2q`, `arc`, `tinylfu`, `wtinylfu`, `belady`
- **Optuna 하이퍼파라미터 탐색**
  - 단일 환경 과적합을 피하기 위해 4개 대표 시나리오 평균 점수 사용
  - 기본적으로 기존 SQLite study를 불러와 COMPLETE trial 총수가 `--optuna_trials`에 도달할 때까지만 자동 재개
- **결과 자동 저장**
  - `results.csv`, `summary.csv`, `best_params.json`, 로그/체크포인트 파일

---

## 저장소 구조

- `run_cache_rl2.py`
  - 전체 실험 오케스트레이션
  - Optuna 실행, 학습/평가 루프, 결과 저장
- `src/models/drqn.py`
  - `CacheEnv`, Replay 버퍼, DRQN/PoolingLSTM 모델, 학습 유틸
- `src/workload/zipf.py`
  - Zipf 요청 트레이스 생성
- `src/workload/builder.py`
  - 시나리오별 트레이스 빌더 (`zipf` 지원)
- `src/baselines/lru.py`
  - LRU 시뮬레이터
- `src/baselines/arc.py`
  - ARC 시뮬레이터 (`T1/T2/B1/B2`, 적응 파라미터 `p`)
- `src/baselines/factory.py`
  - 문자열 이름 기반 베이스라인 팩토리
- `src/evaluation/evaluator.py`
  - RL/베이스라인 공통 평가 및 캐시

---

## 실행 환경

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-colab.txt
```

---

## 실행 방법

### 1) Quick preset

```bash
python run_cache_rl2.py --out_dir out_quick --device cpu --preset quick --optuna_trials 30
```

### 2) Paper-opt preset

```bash
python run_cache_rl2.py --out_dir out_paper --device cpu --preset paper_opt --optuna_trials 40
```

### 3) Full run

```bash
python run_cache_rl2.py --out_dir out_full --device cpu --preset full --optuna_trials 40
```

> GPU 사용 시 `--device cuda` 또는 `--device cuda:0` 지정.

### 중단 후 재시작

기본 재시작 방식은 `--resume_mode rerun_incomplete`입니다. 같은 `--out_dir`로 다시 실행하면 `results.csv`에 최종 결과 행이 있는 완료 set은 건너뛰고, 결과 행 없이 중간에 멈춘 set은 기존 체크포인트/로그를 `incomplete_archive/`로 옮긴 뒤 처음부터 다시 실행합니다.

```bash
python run_cache_rl2.py \
  --out_dir out_full \
  --device cpu \
  --preset full \
  --skip_optuna \
  --resume_mode rerun_incomplete
```

중간 체크포인트에서 이어서 학습하고 싶다면 `--resume_mode checkpoint`를 사용합니다.

```bash
python run_cache_rl2.py \
  --out_dir out_full \
  --device cpu \
  --preset full \
  --skip_optuna \
  --resume_mode checkpoint
```

---

## 기본 설정(코드 기준)

`run_cache_rl2.py`의 기본값은 아래와 같습니다.

- 요청 수: `1,000,000`
- Zipf alpha: `[1.3, 1.4, 1.5, 1.6, 1.7, 1.8]`
- 캐시 크기: `[16, 64]`
- 시나리오: `zipf`
- seed: `[0]`
- 학습 에피소드: 최대 `400`

Quick preset(`--preset quick`, legacy `--use_quick_preset`) 적용 시:

- 요청 수: `250,000`
- seed: `[0, 1]`
- 학습/평가 스텝 축소 (빠른 smoke + 트렌드 확인용)

Paper-opt preset(`--preset paper_opt`) 적용 시:

- 요청 수: `500,000`
- seed: `[0, 1, 2, 3, 4]`
- 학습/평가를 full보다 가볍게 유지하면서 quick보다 안정적인 비교용 설정

---

## Optuna objective 설계

튜닝 프로파일(`--tuning_profile`)에 따라 대표 그리드를 다르게 사용합니다.

- `quick`: alpha `[1.3, 1.8]`, cache `[16, 64]`, seed 1개
- `paper`: alpha `[1.3, 1.5, 1.8]`, cache `[16, 64]`, seed 1개
- `robust`: alpha `[1.3, 1.5, 1.8]`, cache `[16, 64]`, seed 최대 2개

objective는 `0.8 * mean(scores) + 0.2 * min(scores)`를 사용해 평균 성능과 hard case 안정성을 함께 반영합니다.

### Optuna 자동 재개 동작

기본 Optuna storage는 기존과 동일하게 `sqlite:///{OUT_DIR}/optuna_{EXPERIMENT_TAG}_{tuning_profile}.db`이며, study는 `load_if_exists=True`로 불러옵니다. 따라서 같은 `--out_dir`, `--preset`, `--tuning_profile`, `--only_algo`/stage 조합을 다시 실행하면 이전 study를 이어서 사용합니다.

기본값인 `--optuna_trials_mode target_total`에서는 `--optuna_trials`가 **이번 실행에서 새로 추가할 trial 수**가 아니라 **study 안의 목표 COMPLETE trial 총수**를 의미합니다. 예를 들어 첫 실행이 25개의 COMPLETE trial 이후 중단되었고 같은 명령을 `--optuna_trials 40`으로 다시 실행하면, 러너는 자동으로 남은 15개만 요청해 총 40개 COMPLETE trial까지 채웁니다. 이미 40개 이상의 COMPLETE trial이 있으면 optimization을 건너뛰고 기존 `study.best_params`를 사용합니다. PRUNED/FAIL/RUNNING trial은 목표 COMPLETE trial 수에 포함하지 않으며, 기존 RUNNING trial이 있으면 경고만 출력하고 삭제하지 않습니다.

의도적으로 매 실행마다 trial을 더 추가하고 싶다면 예전 동작과 같은 `--optuna_trials_mode additional`을 지정합니다. 이 모드에서 `--optuna_trials 40`은 현재 COMPLETE trial 수와 무관하게 이번 프로세스에서 40개 trial을 추가로 요청합니다.

Colab에서 중단 후 자동 재개를 활용하는 예시:

```bash
python run_cache_rl2.py \
  --out_dir /content/drive/MyDrive/drqn-cache-results/out_paper_opt_resume \
  --device cuda \
  --preset paper_opt \
  --tuning_profile paper \
  --optuna_trials 40 \
  --baseline_set minimal \
  --only_algo drqn_perslot \
  --resume_mode rerun_incomplete
```

탐색 파라미터:

- `LR`: `3e-5 ~ 8e-4` (log)
- `GAMMA`: `0.92 ~ 0.995`
- `UNROLL`: `[30, 40, 60, 80]`
- `BATCH_SIZE`: `[16, 32, 64]`
- `TARGET_UPDATE_EVERY_UPDATES`: `[200, 500, 1000]`
- `UPDATES_PER_EPISODE`: `[8, 12, 16]`
- `EPSILON_DECAY_STEPS`: `[100000, 200000, 300000]`

---

## 출력 파일

- `OUT_DIR/best_params.json`: Optuna 최고 파라미터
- `OUT_DIR/experiment_config.json`: 실행 메타데이터/최종 CONFIG/CLI 인자/깃 정보 및 Optuna 재개 메타데이터(`optuna_trials`, `optuna_trials_mode`, `optuna_completed_trials_before_run`, `optuna_remaining_trials_requested`, `optuna_storage`, `study_name`)
- `OUT_DIR/results.csv`: run-level 결과
- `OUT_DIR/summary.csv`: 그룹 집계 결과
- `OUT_DIR/summary_overall.csv`
- `OUT_DIR/summary_by_cache.csv`
- `OUT_DIR/summary_by_alpha.csv`
- `OUT_DIR/summary_hard_conditions.csv`
- `OUT_DIR/summary_belady_gap.csv`
- `OUT_DIR/logs/*.jsonl`: 에피소드 로그
- `OUT_DIR/ckpt/*.pt`: 모델 체크포인트

---

## 코드 리뷰 기준 확인 포인트

전체 코드 기준으로 아래 사항이 보장되도록 구성되어 있습니다.

1. **베이스라인 공정성**
   - RL과 LRU/ARC 모두 동일 `test_stream`으로 평가
2. **중복 계산 최소화**
   - 베이스라인 결과는 `(scenario, alpha, cache_size, eval_kind, names)` 키로 캐시
3. **안전한 디바이스 설정**
   - CUDA 요청 시 사용 가능 여부/인덱스 유효성 검사
4. **행동 제약 일관성**
   - `valid_action_mask`로 hit/empty 상태에서 `NOOP`만 유효하게 처리

## Experiment control quick guide

### Quick smoke run
```bash
python run_cache_rl2.py \
  --preset quick \
  --only_alpha 1.3 \
  --only_cache 16 \
  --seeds 0 \
  --only_algo drqn_perslot \
  --baseline_set minimal \
  --skip_optuna \
  --out_dir out_smoke_cleanup \
  --device cpu
```

### Lightweight Optuna tuning run
```bash
python run_cache_rl2.py \
  --out_dir /content/out_tune_light_two_stage \
  --device cuda \
  --preset quick \
  --tuning_profile quick \
  --optuna_trials 10 \
  --baseline_set minimal \
  --only_algo drqn_perslot
```

### Fixed best_params paper_opt run
```bash
python run_cache_rl2.py \
  --out_dir /content/drive/MyDrive/drqn-cache-results/out_paper_two_stage_fixed \
  --device cuda \
  --preset paper_opt \
  --skip_optuna \
  --best_params_path /content/drive/MyDrive/drqn-cache-results/out_tune_light_two_stage/best_params.json \
  --baseline_set paper \
  --only_algo drqn_perslot \
  --seeds 0,1,2,3,4
```

### Notes
- Prefer local `/content` during Optuna tuning to reduce Google Drive I/O overhead.
- Copy `best_params.json` to Drive after tuning.
- Use Drive `out_dir` for long persistence-focused final runs.
