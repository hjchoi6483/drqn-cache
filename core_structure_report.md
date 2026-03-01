# Core Structure Report (현재 코드 기준)

## 1) 문서 목적
이 문서는 **현재 저장소의 실제 코드 상태**를 기준으로, `run_cache_rl2.py` 실행 흐름과 `src/*` 모듈 간 역할 분리를 정확히 설명합니다. 특히 요청하신 대로 **quick preset 동작과 전체 실험 워크플로우**를 중심으로 정리했습니다.

---

## 2) 현재 아키텍처 요약

현재 구조는 "실험 오케스트레이션(상단 스크립트) + 기능 모듈(src)" 형태입니다.

- 최상위 실행기
  - `run_cache_rl2.py`: CLI 파싱, 설정(CONFIG), 실험 태스크 생성/반복, 체크포인트 재개, 결과 저장/집계 담당.
- 기능 모듈
  - `src/workload/*`: 요청 스트림(현재 zipf) 생성.
  - `src/models/drqn.py`: 환경(CacheEnv), 관측 생성, 모델 정의(DRQN/DQN/Pooling), 액션 선택, 에피소드 롤아웃, 학습 스텝.
  - `src/evaluation/evaluator.py`: RL 정책과 baseline을 동일 스트림에서 평가하고 차이값 계산.
  - `src/baselines/*`: baseline 팩토리(현재 LRU).

즉, 실행 제어는 `run_cache_rl2.py`에 있고, 학습/평가 핵심 로직은 `src`로 분리되어 있습니다.

---

## 3) quick preset 문제 및 수정 사항

### 문제 현상
기존에는 실행 완료 이력이 `results.csv`에 남아 있으면 `run_all()`에서 태스크가 모두 제외되어, quick preset 실행 시
`Remaining runs: 0`으로 시작해 **처음부터 다시 실행되지 않는 것처럼 보이는 문제**가 있었습니다.

### 원인
`run_id`가 `scenario/alpha/cache_size/seed/setting`만으로 구성되어 있어, full 실험과 quick 실험이 같은 ID 공간을 공유했습니다.

### 수정 내용
- CONFIG에 `EXPERIMENT_TAG`를 도입했습니다.
  - 기본 실행: `full`
  - quick preset: `quick`
- `run_id()`가 ID 앞에 `EXPERIMENT_TAG`를 포함하도록 변경했습니다.

결과적으로 quick/full 실행 결과가 서로 다른 run_id를 사용하므로, 한쪽 결과가 다른 쪽의 재실행을 막지 않습니다.

---

## 4) 실행 워크플로우 (정밀 설명)

아래는 실제 함수 호출 순서 기준 워크플로우입니다.

### 4.1 진입 및 설정
1. `parse_args()`로 `--out_dir`, `--device`, `--use_quick_preset`을 읽습니다.
2. 기본 `CONFIG`를 구성합니다.
3. `--use_quick_preset`이 켜져 있으면 `apply_quick_preset()`으로 학습/평가 길이를 줄이고, 실험 태그를 `quick`으로 바꿉니다.
4. `if __name__ == "__main__": run_all()`로 메인 루프를 시작합니다.

### 4.2 전체 태스크 생성 (`run_all`)
1. `ensure_dirs()`로 출력 디렉터리를 준비합니다.
2. `load_done_ids()`로 기존 `results.csv`의 run_id를 읽습니다.
3. `SCENARIOS × ZIPF_ALPHAS × CACHE_SIZES × SEEDS × SETTINGS` 조합을 순회해 미완료 태스크만 수집합니다.
4. 각 `(scenario, alpha, seed)` 조합마다 trace를 한 번만 만들고(`stream_cache`), 이를 cache_size/setting 실행에서 재사용합니다.
   - `build_trace()` → 현재는 `trace_zipf()` 사용.
   - `TRAIN_RATIO` 기준으로 train/fast/full 평가 스트림을 분리합니다.
5. 태스크마다 `train_one_run()`을 호출하고, 매 실행 후 `build_summary()`를 갱신합니다.

### 4.3 단일 실행 학습 (`train_one_run`)
1. `run_id()` 생성 후 완료 여부를 다시 확인합니다.
2. 같은 run_id ckpt가 있으면 모델/옵티마/리플레이/상태를 복구하고, 없으면 새로 초기화합니다.
3. 에피소드 루프:
   - `rollout_episode()`로 `EPISODE_LEN` 길이 샘플을 진행(탐험률 `epsilon_by_step` 적용).
   - 조건을 만족하면 시퀀스를 `EpisodeReplay`에 저장.
   - `train_step()`을 `UPDATES_PER_EPISODE`만큼 반복.
   - `TARGET_UPDATE_EVERY_UPDATES`마다 타깃 네트워크 동기화.
   - 주기적으로 fast/full 평가를 수행 (`eval_policy` → evaluator 모듈).
   - 로그 JSONL 기록, ckpt 저장.
4. 학습 종료 후 full 평가 1회를 수행하고 `results.csv`에 최종 row를 기록합니다.

### 4.4 평가 파이프라인 (`eval_policy` / evaluator)
1. RL 정책 평가:
   - `make_env_for_eval`로 환경 생성.
   - test_stream 요청을 순회하며 `make_obs_for_eval` → `select_action(eps=0)` → `env.step` 수행.
   - RL hit rate 계산.
2. baseline 평가:
   - `compute_baselines_once()`가 `(scenario, alpha, cache, eval_kind, baseline_names)` 키로 캐시 확인.
   - 없으면 baseline 시뮬레이터를 만들어 동일 stream으로 hit rate 계산 후 캐시.
3. 출력:
   - `rl_hit`
   - `baseline_hit_*`
   - `rl_minus_baseline_*`

### 4.5 집계 (`build_summary`)
1. `results.csv`를 읽습니다.
2. `(scenario, alpha, cache_size, setting)` 기준으로 그룹화합니다.
3. RL 및 baseline 평균/표준편차를 계산해 `summary.csv`를 생성합니다.

---

## 5) 모듈별 책임

### `src/workload`
- `builder.py`: scenario 라우팅(`zipf` 처리).
- `zipf.py`: `np.random.zipf` 기반 요청 ID 시퀀스 생성.

### `src/models/drqn.py`
- `CacheEnv`: 캐시 상태, 통계(hit_ema, miss_streak 등), 보상 계산.
- 관측/행동 관련 함수: `make_obs_for_eval`, `select_action`.
- 네트워크: `DRQN_PerSlot`, `DQN_PerSlot`, `PoolingQNet`.
- 학습 유틸: `EpisodeReplay`, `rollout_episode`, `train_step`.

### `src/evaluation/evaluator.py`
- RL 평가 루프 + baseline 공통 계산.
- baseline 결과 캐시로 중복 평가 비용 절감.

### `src/baselines`
- `lru.py`: OrderedDict 기반 LRU 캐시 시뮬레이터.
- `factory.py`: 이름 기반 baseline 생성(현재 lru).

---

## 6) 데이터/산출물 흐름

- 입력(내부 생성)
  - `build_trace`가 request stream 생성.
- 중간 산출물
  - `out_dir/logs/*.jsonl`: 에피소드별 학습 로그.
  - `out_dir/ckpt/*.pt`: 재개용 학습 상태.
- 최종 산출물
  - `out_dir/results.csv`: run-level 성능.
  - `out_dir/summary.csv`: 그룹 집계 성능.

---

## 7) 유지보수 관점 체크포인트

1. run_id 체계
   - 현재 quick/full 분리를 위해 `EXPERIMENT_TAG`가 포함되므로, 다른 실험군 추가 시 태그 정책을 먼저 정의하는 것이 안전합니다.
2. baseline 확장
   - `build_baselines`에 구현체만 추가하면 evaluator는 그대로 재사용 가능합니다.
3. scenario 확장
   - `build_trace` 분기와 설정값만 추가하면 상위 루프는 변경 없이 확장 가능합니다.
4. 재현성
   - trace 생성 전 `set_seed` 호출, torch/numpy/random 고정이 적용되어 있습니다.

---

## 8) 결론

현재 코드는 "단일 실행 스크립트 + 핵심 모듈 분리" 단계로 정리되어 있으며, 이번 수정으로 quick preset이 기존 full 결과와 충돌하지 않고 독립적으로 실행됩니다. 또한 워크플로우는 `run_all -> train_one_run -> eval_policy/build_summary`로 명확히 분리되어 있어, 이후 baseline/scenario 확장도 구조적으로 수월한 상태입니다.
