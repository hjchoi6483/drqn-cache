# drqn-cache

DRQN 기반 캐시 교체 실험 러너입니다. 현재 코드는 **2단계 워크플로우**를 지원합니다.

1. **Optuna 하이퍼파라미터 최적화 (1회)**
   - 대표 난이도 시나리오 `scenario=zipf, alpha=1.3, cache_size=16`에서 탐색
   - 탐색 대상: `LR`, `GAMMA`, `UNROLL`, `BATCH_SIZE`
   - pruning 적용으로 비효율 trial 조기 중단
   - 결과를 `OUT_DIR/best_params.json`에 저장
2. **실험 매트릭스 실행**
   - Step 1에서 얻은 `best_params`를 `CONFIG`에 반영
   - 기존 매트릭스(`alpha 1.3~1.8`, `cache_size 16/64`, 설정 조합)를 순차 실행
   - 결과를 기존과 동일하게 `results.csv`, `summary.csv`에 기록

---

## 핵심 구조 (Core Structure)

- `run_cache_rl2.py`
  - CLI/CONFIG 로딩
  - Stream cache 생성 및 재사용
  - Optuna objective/study 실행
  - 단일 학습 실행(`train_one_run`) 및 결과 저장
  - 집계(`build_summary`)
- `src/workload/*`
  - 요청 스트림 생성 (`zipf`)
- `src/models/drqn.py`
  - DRQN/DQN/Pooling 모델, CacheEnv, replay, rollout, train_step
- `src/evaluation/evaluator.py`
  - RL + baseline 공통 평가
- `src/baselines/*`
  - baseline 시뮬레이터 (현재 LRU)

---

## 2단계 워크플로우 상세

### Step 1) Optimization (Optuna)

`run_all()` 시작 시 1회 수행:

- `study = optuna.create_study(direction="maximize", pruner=MedianPruner, storage="sqlite:///optuna_study.db", study_name="drqn_cache_tuning", load_if_exists=True)`
- `n_trials` 기본값 40 (`--optuna_trials`로 조정)
- objective 내부 대표 시나리오 고정:
  - `scenario='zipf'`
  - `alpha=1.3`
  - `cache_size=16`
- 탐색 공간:
  - `LR`: `1e-5 ~ 1e-3` (`log=True`)
  - `GAMMA`: `0.9 ~ 0.999`
  - `UNROLL`: `20 ~ 80` (`step=10`)
  - `BATCH_SIZE`: `[16, 32, 64]`
- pruning:
  - 학습 루프에서 `FAST_EVAL_EVERY_EP`마다 `trial.report(hit_proxy, ep)`
  - `trial.should_prune()`가 `True`면 `TrialPruned` 발생

최적 파라미터는 `best_params.json`으로 저장됩니다.

### Step 2) Matrix Run

- `CONFIG.update(best_params)`로 전역 적용
- 기존 매트릭스 루프 그대로 실행
- `(scenario, alpha, seed)` 단위 `stream_cache` 재사용으로 trace 중복 생성 방지
- 모든 결과는 기존과 동일하게 아래 파일에 누적:
  - `results.csv`
  - `summary.csv`

---

## 실행 방법 (How to Run)

### 1) 환경 준비

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch tqdm numpy optuna
```

### 2) 빠른 실행(권장)

```bash
python run_cache_rl2.py --out_dir out_quick --device cpu --use_quick_preset --optuna_trials 30
```

### 3) 기본 실행

```bash
python run_cache_rl2.py --out_dir out_full --device cpu --optuna_trials 40
```

---


## Colab Pro+에서 실행 (GitHub 코드 그대로, 기능 유지)

아래 절차를 따르면 **GitHub에 올린 현재 코드 그대로** Colab Pro+ GPU에서 실행할 수 있습니다.
(Optuna 2단계 워크플로우, 체크포인트 재개, 결과 집계 기능 유지)

### 0) (선택) Colab에서 GitHub 노트북으로 바로 열기

리포지토리에 notebook을 추가했다면 아래 형식으로 바로 열 수 있습니다.

```text
https://colab.research.google.com/github/<GITHUB_OWNER>/<REPO_NAME>/blob/<BRANCH>/notebooks/<NOTEBOOK>.ipynb
```

노트북이 없다면 아래 `git clone` 방식으로 실행하면 됩니다.

### 1) Colab 런타임 설정

- Colab 메뉴: `런타임` → `런타임 유형 변경`
- Hardware accelerator: `GPU` (Pro+ 권장)

### 2) Google Drive 마운트 (결과/체크포인트 영속화)

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3) GitHub에서 코드 가져오기

```bash
%cd /content
!git clone https://github.com/<GITHUB_OWNER>/<REPO_NAME>.git drqn-cache
%cd /content/drqn-cache
```

- 브랜치를 지정하려면:

```bash
!git checkout <BRANCH_NAME>
```

### 4) 의존성 설치

Colab 기본 PyTorch를 우선 사용하고, 프로젝트 필수 패키지만 설치합니다.

```bash
!pip install -U pip
!pip install -r requirements-colab.txt
```

필요할 때만 PyTorch를 재설치하세요.

```bash
# 예시 (CUDA 12.1)
# !pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

### 5) 실행 (Quick / Full)

> 핵심: `--out_dir`를 Drive 경로로 두면, 런타임이 끊겨도 결과/체크포인트/Optuna DB를 유지할 수 있습니다.

Quick (권장 시작):

```bash
!python run_cache_rl2.py \
  --out_dir /content/drive/MyDrive/drqn-cache-out/quick \
  --device cuda \
  --use_quick_preset \
  --optuna_trials 30
```

Full:

```bash
!python run_cache_rl2.py \
  --out_dir /content/drive/MyDrive/drqn-cache-out/full \
  --device cuda \
  --optuna_trials 40
```

### 6) 재시작/이어달리기

- 같은 `--out_dir`로 재실행하면 체크포인트(`ckpt/*.pt`)에서 자동 재개됩니다.
- Optuna는 실행 경로의 `optuna_study.db`를 사용해 trial 기록을 이어갑니다.
- 런타임 재할당이 잦다면 프로젝트 폴더 자체를 Drive에 clone해서 실행해도 됩니다.

### 7) 자주 겪는 문제

- `ModuleNotFoundError` 발생 시: `!pip install -r requirements-colab.txt` 재실행
- GPU 미사용 시: 런타임 GPU 설정 후 `--device cuda`로 재실행
- 새 실험으로 완전 초기화: 해당 `out_dir` + `optuna_study.db` 삭제 후 실행


## 출력 파일

- `OUT_DIR/best_params.json`: Optuna 최적 하이퍼파라미터
- `OUT_DIR/results.csv`: run-level 결과
- `OUT_DIR/summary.csv`: 그룹 집계 결과
- `OUT_DIR/logs/*.jsonl`: 학습 로그
- `OUT_DIR/ckpt/*.pt`: 체크포인트

---


## 실험 중단/재시작 가이드 (Windows CMD 포함)

- **중단 방법 (Graceful Exit):** 실행 중인 CMD 창에서 `Ctrl+C`를 누르면 현재 학습 상태를 체크포인트(`OUT_DIR/ckpt/*.pt`)로 즉시 저장하고, 프로세스를 즉시 종료합니다(종료코드 130, 다음 trial/실험으로 자동 진행하지 않음).
- **재개 방법:** 같은 명령어를 다시 실행하면 기존 체크포인트(`ep_done`, `global_step`, `train_cursor`, `replay`, optimizer/model 상태)를 자동 복구하여 이어서 학습합니다.
- **Optuna 재개:** Optuna 탐색 이력은 로컬 `optuna_study.db`(SQLite)에 저장되며, 동일 study 이름으로 자동 이어서 실행됩니다.
- **초기화 방법:** 완전히 새 실험을 시작하려면 `out/` 폴더(또는 지정한 `OUT_DIR`)와 `optuna_study.db` 파일을 삭제한 뒤 다시 실행하세요.

---

## 참고

- quick/full 분리는 `EXPERIMENT_TAG`가 run_id에 반영되어 서로 결과 충돌을 방지합니다.
- baseline/시나리오 확장 시 `src/baselines`, `src/workload`만 확장하면 상위 러너 구조는 유지 가능합니다.
