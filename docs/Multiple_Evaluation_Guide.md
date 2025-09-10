# 여러 평가 실행 및 결과 통합 가이드

이 가이드는 여러 모델, task, 설정으로 평가를 자동화하고 결과를 통합하는 방법을 설명합니다.

## 스크립트 소개

### 1. `run_multiple_evals.sh` - 평가 자동화 스크립트 (Linux 전용)
여러 모델과 설정으로 평가를 자동으로 실행하는 쉘 스크립트입니다.

### 2. `aggregate_results.py` - 결과 통합 스크립트
여러 평가 결과를 수집하여 보기 좋게 정리하는 Python 스크립트입니다.

## 설치 및 준비

### 스크립트 실행 권한 설정 (Linux)
```bash
chmod +x scripts/run_multiple_evals.sh
```

**참고**: 스크립트는 실행할 때마다 자동으로 lm-evaluation-harness를 설치/업데이트합니다.
- 기본적으로 스크립트가 있는 디렉토리의 상위 디렉토리를 프로젝트 루트로 사용
- `LM_EVAL_ROOT` 환경 변수로 다른 경로 지정 가능

```bash
# 커스텀 경로 사용
LM_EVAL_ROOT=/path/to/lm-evaluation-harness ./scripts/run_multiple_evals.sh
```

## 사용법

### 기본 사용법

#### 1. 단순 실행
```bash
# 기본 설정으로 실행 (gpt2 모델, 3개 task, 0/5-shot)
./scripts/run_multiple_evals.sh
```

#### 2. 로컬 모델 평가
```bash
# pythia-160m 모델로 빠른 테스트
MODELS="/workspace/models/pythia-160m" \
TASKS="hellaswag" \
FEWSHOTS="0" \
LIMIT=10 \
./scripts/run_multiple_evals.sh
```

#### 3. 여러 모델 비교
```bash
MODELS="gpt2 gpt2-medium /path/to/local/model" \
TASKS="hellaswag arc_easy winogrande" \
FEWSHOTS="0 5" \
./scripts/run_multiple_evals.sh
```

### 환경 변수 설정

| 변수 | 설명 | 기본값 | 예시 |
|------|------|--------|------|
| `MODELS` | 평가할 모델 목록 (공백 구분) | `"gpt2"` | `"gpt2 gpt2-medium /path/to/model"` |
| `TASKS` | 평가할 task 목록 (공백 구분) | `"hellaswag arc_easy winogrande"` | `"mmlu arc_challenge"` |
| `FEWSHOTS` | Few-shot 설정 (공백 구분) | `"0 5"` | `"0 1 5 10"` |
| `BATCH_SIZES` | 배치 크기 설정 | `"auto"` | `"8 16 32"` |
| `DEVICE` | GPU 디바이스 | `"cuda:0"` | `"cuda:1"` 또는 `"cpu"` |
| `LIMIT` | 샘플 수 제한 (테스트용) | (없음) | `"100"` |
| `OUTPUT_BASE` | 결과 저장 디렉토리 | `"results/experiments_YYYYMMDD_HHMMSS"` | `"results/my_experiment"` |

### 실제 사용 예시

#### 예시 1: 빠른 테스트
```bash
# 10개 샘플로 빠르게 테스트
MODELS="/workspace/models/pythia-160m" \
TASKS="hellaswag" \
FEWSHOTS="0" \
LIMIT=10 \
OUTPUT_BASE="results/quick_test" \
./scripts/run_multiple_evals.sh
```

#### 예시 2: 전체 평가
```bash
# 전체 데이터셋으로 평가
MODELS="/workspace/models/pythia-160m" \
TASKS="hellaswag arc_easy winogrande piqa" \
FEWSHOTS="0 5" \
DEVICE="cuda:0" \
OUTPUT_BASE="results/full_evaluation" \
./scripts/run_multiple_evals.sh
```

#### 예시 3: 모델 비교
```bash
# 여러 모델 비교
MODELS="gpt2 /workspace/models/pythia-160m /workspace/models/my_finetuned_model" \
TASKS="hellaswag arc_easy" \
FEWSHOTS="0 5" \
OUTPUT_BASE="results/model_comparison" \
./scripts/run_multiple_evals.sh
```

#### 예시 4: 배치 크기 실험
```bash
# 다양한 배치 크기로 테스트
MODELS="/workspace/models/pythia-160m" \
TASKS="hellaswag" \
FEWSHOTS="0" \
BATCH_SIZES="4 8 16 auto" \
LIMIT=100 \
OUTPUT_BASE="results/batch_size_test" \
./scripts/run_multiple_evals.sh
```

## 결과 확인

### 생성되는 파일 구조
```
results/experiments_20240101_120000/
├── evaluation.log                    # 실행 로그
├── summary.md                        # 통합 결과 (Markdown)
├── summary.csv                       # 통합 결과 (CSV)
├── gpt2/
│   ├── hellaswag_0shot/
│   │   └── results_2024-01-01T12-00-00.json
│   └── hellaswag_5shot/
│       └── results_2024-01-01T12-05-00.json
└── pythia-160m/
    ├── hellaswag_0shot/
    │   └── results_2024-01-01T12-10-00.json
    └── hellaswag_5shot/
        └── results_2024-01-01T12-15-00.json
```

### 결과 통합하기

#### 자동 통합 (스크립트 실행 후 자동)
`run_multiple_evals.sh` 스크립트는 평가 완료 후 자동으로 결과를 통합합니다.

#### 수동 통합
```bash
# Markdown 형식으로 출력
python scripts/aggregate_results.py \
    --results-dir results/experiments_20240101_120000 \
    --output comparison.md

# CSV 형식으로 출력
python scripts/aggregate_results.py \
    --results-dir results/experiments_20240101_120000 \
    --format csv \
    --output results.csv

# LaTeX 테이블로 출력
python scripts/aggregate_results.py \
    --results-dir results/experiments_20240101_120000 \
    --format latex \
    --output table.tex
```

### 결과 파일 설명

#### summary.md 내용
- **Summary Statistics**: 모델별 평균 성능
- **Detailed Results**: 모든 task와 설정의 상세 결과
- **Performance Matrix**: Task x Model 피벗 테이블
- **Source Files**: 사용된 JSON 파일 목록

#### summary.csv 내용
- 모든 결과를 표 형식으로 정리
- Excel이나 다른 도구에서 분석 가능

## 고급 사용법

### 1. 커스텀 평가 스크립트 작성
```bash
#!/bin/bash
# my_evaluation.sh

# 특정 실험 설정
MODELS="/workspace/models/pythia-160m"
TASKS="hellaswag arc_easy winogrande piqa boolq"
FEWSHOTS="0 1 5"
LIMIT=""  # 전체 평가
OUTPUT_BASE="results/pythia_full_eval"

# 실행
./scripts/run_multiple_evals.sh
```

### 2. 결과 모니터링
```bash
# 실시간 로그 확인
tail -f results/experiments_*/evaluation.log

# GPU 사용률 모니터링
watch -n 1 nvidia-smi
```

### 3. 병렬 실행 (고급)
```bash
# 여러 GPU에서 동시 실행
DEVICE="cuda:0" TASKS="hellaswag" ./scripts/run_multiple_evals.sh &
DEVICE="cuda:1" TASKS="arc_easy" ./scripts/run_multiple_evals.sh &
wait
```

## 문제 해결

### 1. 메모리 부족
```bash
# 배치 크기 줄이기
BATCH_SIZES="4" ./scripts/run_multiple_evals.sh

# 또는 샘플 수 제한
LIMIT=100 ./scripts/run_multiple_evals.sh
```

### 2. 스크립트 실행 권한 오류
```bash
chmod +x scripts/run_multiple_evals.sh
```

### 3. Python 패키지 오류
```bash
pip install pandas numpy
```

### 4. 로컬 모델 경로 오류
```bash
# 절대 경로 사용
MODELS="/home/user/models/my_model"  # 좋음
MODELS="~/models/my_model"           # 피하세요
MODELS="../models/my_model"          # 피하세요
```

## 팁과 모범 사례

1. **테스트 먼저**: 전체 평가 전에 `LIMIT=10`으로 빠른 테스트
2. **로그 확인**: `evaluation.log` 파일에서 오류 확인
3. **디스크 공간**: 결과 파일이 많이 생성되므로 충분한 공간 확보
4. **체계적 명명**: `OUTPUT_BASE`를 의미 있게 설정하여 실험 구분
5. **증분 실행**: 큰 실험은 작은 단위로 나누어 실행

## 예상 실행 시간

| 설정 | 예상 시간 |
|------|-----------|
| 1 모델, 1 task, 100 샘플 | ~1-2분 |
| 1 모델, 1 task, 전체 | ~10-30분 |
| 3 모델, 3 tasks, 2 few-shots, 전체 | ~3-5시간 |

실제 시간은 모델 크기, GPU 성능, 데이터셋 크기에 따라 다릅니다.

## 추가 자료

- [lm-evaluation-harness 문서](../README.md)
- [Multi-GPU 가이드](./MultiGPU_Guide.md)
- [Task 목록](../lm_eval/tasks/README.md)