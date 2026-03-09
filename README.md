# drqn-cache

캐시 교체 강화학습 실험을 위한 러너입니다. 현재 코드는 **2개 RL 모델(최고 성능 조합만 유지)** + **2개 고전 베이스라인(LRU, ARC)** 비교에 최적화되어 있습니다.

## 이번 리팩토링 핵심 변경점

1. **모델 축소/정리**
   - 유지 모델:
     - `drqn_perslot|G1|P0` (Global feature 사용, Invalid penalty 미사용)
     - `pooling_lstm|G1|P1`
   - 제거 모델:
     - `dqn_perslot`
     - `pooling_ff`
     - 기타 DRQN ablation 조합 (`G0`, `P1` 등)

2. **Optuna objective 범용화 (Option 2)**
   - 단일 환경(`alpha=1.3, cache=16`) 최적화 방식 제거
   - 하나의 trial에서 대표 시나리오를 모두 짧게 평가 후 평균 성능을 objective로 반환
   - 현재 trial 내부 평가 그리드:
     - `(zipf, 1.3, 16)`
     - `(zipf, 1.3, 64)`
     - `(zipf, 1.8, 16)`
     - `(zipf, 1.8, 64)`

3. **ARC baseline 추가**
   - `src/baselines/arc.py`에 ARC(T1/T2/B1/B2 + `p` 적응 로직) 구현
   - baseline factory와 실험 러너에 `arc` 연동
   - 결과 CSV/요약에서 RL vs LRU vs ARC 비교 가능

4. **로그 출력 단순화**
   - tqdm 기반 반복 진행바 출력 제거
   - 각 run에 대해 **시작 시점**과 **완료 시점(최종 성능)**만 출력
   - Colab/CLI 로그 길이 과다 문제 완화

---

## 프로젝트 구조

- `run_cache_rl2.py`
  - 전체 실험 오케스트레이션
  - Optuna 최적화 + 매트릭스 실행
  - run 결과 저장(`results.csv`) 및 집계(`summary.csv`)
- `src/models/drqn.py`
  - CacheEnv, replay, 학습 루프 유틸
  - 유지 모델 2종(`DRQN_PerSlot`, `PoolingQNet(LSTM)`)
- `src/baselines/lru.py`
  - LRU 시뮬레이터
- `src/baselines/arc.py`
  - ARC 시뮬레이터
- `src/baselines/factory.py`
  - baseline name → simulator 생성
- `src/evaluation/evaluator.py`
  - RL 정책 및 baseline 공통 평가

---

## 실행 방법

### 1) 환경 준비

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-colab.txt
```

### 2) Quick preset

```bash
python run_cache_rl2.py --out_dir out_quick --device cpu --use_quick_preset --optuna_trials 30
```

### 3) Full run

```bash
python run_cache_rl2.py --out_dir out_full --device cpu --optuna_trials 40
```

---

## Optuna 동작 방식

`run_all()` 시작 시 study를 1회 실행하고 best param을 `best_params.json`에 저장합니다.

- 탐색 파라미터
  - `LR`: `1e-5 ~ 1e-3` (log)
  - `GAMMA`: `0.9 ~ 0.999`
  - `UNROLL`: `20 ~ 80` (step=10)
  - `BATCH_SIZE`: `[16, 32, 64]`
- objective
  - trial 1개당 대표 4개 시나리오를 순차 학습/평가
  - `rl_hit` 평균값을 최종 objective로 사용
  - 중간 평균 기반 pruning 적용

이 방식으로 특정 단일 환경에 과적합되는 위험을 줄이고, 더 robust한 파라미터를 찾도록 했습니다.

---

## 결과 파일

- `OUT_DIR/best_params.json`: Optuna 최고 파라미터
- `OUT_DIR/results.csv`: run-level 결과
- `OUT_DIR/summary.csv`: 그룹 집계 결과
- `OUT_DIR/logs/*.jsonl`: 에피소드 로그
- `OUT_DIR/ckpt/*.pt`: 체크포인트

---

## 출력 로그 예시

- Run 시작 시:
  - `[RUN START] ...`
- Run 종료 시:
  - `[DONE] ... | RL xx.xx  LRU xx.xx ARC xx.xx`

진행률 바를 제거해 노이즈를 줄이고 핵심 정보만 남겼습니다.
