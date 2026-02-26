# Core Structure Report (업데이트)

## 목표 기반 업데이트
요청한 3개 목표를 기준으로, 기존 구조 검토를 **실행 가능한 모듈화 계획 + 실험 설계 변경안**으로 재정리했습니다.

### 목표 1) 제시된 리팩터링 로드맵에 따라 모듈화
### 목표 2) hotshift 변수 재조정 (zipf와 분리되는 동적 분포 반영)
### 목표 3) LRU/DRQN 외 TinyLFU, ARC 비교 추가 (모듈화 연장선)

---

## 1. 현재 구조 진단 (핵심)
현재 `run_cache_rl2.py` 단일 파일 구조는 다음 문제가 있습니다.
- 실험 설정/워크로드/환경/모델/학습/평가/집계가 1개 파일에 결합.
- baseline 구현이 사실상 LRU 1종 중심이라 비교 확장성 낮음.
- hotshift가 분포 이동(phase drift)을 충분히 만들지 못하면 zipf와 유사한 결과로 수렴.

따라서, 이번 목표는 단순 문서화가 아니라 **비교 실험 확장 가능한 코드 아키텍처**로 전환하는 것입니다.

---

## 2. 모듈화 타겟 구조 (권장 디렉터리)

```text
src/
  config/
    schema.py            # dataclass/pydantic 기반 설정 모델
    defaults.py          # 기본 설정 + quick preset
  workload/
    zipf.py              # zipf trace
    hotshift.py          # hotshift trace + drift 파라미터
    builder.py           # scenario 라우팅
  env/
    cache_env.py         # RL용 환경
    features.py          # cache/global feature 생성
  agents/
    drqn.py
    dqn.py
    pooling.py
    factory.py           # algo 이름 -> 모델 생성
  baselines/
    lru.py
    tinylfu.py
    arc.py
    factory.py           # baseline 라우팅
  training/
    replay.py
    trainer.py           # train_one_run 분해(초기화/에피소드/업데이트)
    checkpoint.py
  evaluation/
    evaluator.py         # RL + baseline 공통 평가 루프
    metrics.py
  io/
    logging.py
    results.py
  app/
    run_experiment.py    # 기존 run_all 역할
```

### 분리 원칙
- `workload`와 `baselines`를 독립시켜, 알고리즘 추가 시 학습 코드 수정 최소화.
- `evaluation/evaluator.py`에서 "RL vs baseline N종" 비교를 공통 인터페이스로 처리.
- `config/schema.py`에서 hotshift 관련 파라미터 검증(범위/상호 제약) 강제.

---

## 3. hotshift 재조정 계획 (zipf와 결과 분리)

현재 hotshift가 zipf와 유사해지는 원인은 보통 아래 조합에서 발생합니다.
- phase 수가 적고,
- 이동 폭(offset step)이 작거나 주기성이 높고,
- 이동 시 hot set의 농도가 유지되어도 실제 cache pressure 변화가 작을 때.

### 3.1 파라미터 재설계 제안
`hotshift.py`에서 아래 파라미터를 명시화:
- `hotshift_phases`: 기본 4 -> **8~16** 실험
- `hotshift_offset_mode`: `half_plus_one` 외 `coprime_stride`, `random_stride`
- `hotshift_offset_step`: `vocab_size`와 서로소(coprime) 보장 옵션
- `hotshift_phase_skew`: phase별 alpha 가중치(예: 1.2 -> 1.8 순환)
- `hotshift_mix_ratio`: hotset/coldset 혼합비 (예: 0.8/0.2)
- `hotshift_transition`: abrupt vs smooth 전환(선형/코사인 블렌드)

### 3.2 검증 지표 추가
hotshift가 zipf와 다른 분포임을 검증하려면 학습 전에 아래를 산출:
- phase별 top-k overlap
- phase 간 Jensen-Shannon divergence
- 이동 전후 cache miss burst 길이

위 3개를 `workload diagnostics` CSV로 저장해, "hotshift가 실제로 이동했는지" 사전 검증.

### 3.3 즉시 적용 권장 기본값
- `phases=12`
- `offset_mode=coprime_stride`
- `offset_step=5003` (vocab=10000 기준 예시)
- `transition=smooth`
- `phase_skew=[1.25,1.35,1.45,1.55,1.65,1.75]` 순환

---

## 4. Baseline 확장 계획: LRU + TinyLFU + ARC

## 4.1 공통 인터페이스
`baselines/factory.py`에서 아래 프로토콜 통일:
- `reset(capacity)`
- `access(key) -> hit: bool`
- `stats() -> dict`

이 인터페이스로 evaluator는 baseline 종류와 무관하게 동일 루프 사용.

## 4.2 TinyLFU
- 최소 요구: frequency sketch + admission policy + victim 비교.
- 단순 구현 시작점:
  - Count-Min Sketch 기반 추정 빈도
  - 후보(new key) vs victim(old key) 빈도 비교 후 admit 결정
  - eviction policy는 SLRU/LRU 기반 선택

## 4.3 ARC
- 최소 요구: T1/T2 + B1/B2(ghost) + 적응 파라미터 `p`.
- miss/hit 패턴에 따라 recent/frequent 비중 자동 조정.

## 4.4 결과 스키마 확장
`results.csv`/`summary.csv`에 아래 필드 추가:
- `baseline_name`
- `baseline_hit`
- `rl_minus_baseline`
- `scenario_drift_score` (hotshift 진단 지표 요약)

---

## 5. 구현 순서 (실행 우선순위)

### Step A (1차)
- `workload`, `baselines(lru)` 분리
- evaluator 공통 루프 도입
- 결과 포맷을 "다중 baseline" 대응으로 변경

### Step B (2차)
- `hotshift.py` 재작성 + diagnostics 추가
- zipf/hotshift 분포 차이 자동 리포트

### Step C (3차)
- TinyLFU/ARC baseline 추가
- summary에서 DRQN vs {LRU, TinyLFU, ARC} 동시 비교

### Step D (4차)
- `train_one_run` 분해 + smoke test
- 최소 CI: quick preset + 1 seed + 1 alpha + 1 cache size

---

## 6. 완료 기준 (Definition of Done)
아래를 만족하면 목표 달성으로 판단:
1. 코드가 모듈 구조(`workload/env/agents/baselines/evaluation`)로 분리됨.
2. hotshift diagnostics에서 zipf 대비 분포 차이가 수치로 확인됨(JS divergence 등).
3. 결과 리포트에 LRU/TinyLFU/ARC 비교 컬럼이 포함됨.
4. 단일 커맨드로 전체 실험 실행 가능하며 기존 체크포인트/로그 기능 유지.

---

## 7. 최종 코멘트
이번 요청의 핵심은 "리포트 작성"이 아니라 "바로 구현 가능한 아키텍처 변경 지시서"입니다.
따라서 위 구조를 기준으로 다음 작업은 실제 코드 분리와 baseline 추가를 순차 반영하는 방식으로 진행하는 것이 가장 안전합니다.
