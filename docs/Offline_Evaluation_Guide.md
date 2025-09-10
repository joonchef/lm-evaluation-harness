# 오프라인 환경 평가 가이드

이 가이드는 인터넷 연결이 없는 환경에서 lm-evaluation-harness를 사용하는 방법을 설명합니다.

## 개요

오프라인 환경에서 평가를 수행하려면:
1. 온라인 환경에서 모든 데이터셋을 사전 다운로드
2. 오프라인 환경으로 데이터 전송
3. 환경 변수 설정 후 평가 실행

## 1단계: 데이터셋 준비 (온라인 환경)

### 방법 1: 자동화 스크립트 사용 (권장)

```bash
# 실행 권한 부여
chmod +x scripts/prepare_offline_cache.sh

# 모든 데이터셋 다운로드 및 압축
./scripts/prepare_offline_cache.sh /path/to/cache

# 또는 기본 경로 사용
./scripts/prepare_offline_cache.sh
```

이 스크립트는:
- 모든 task의 데이터셋을 자동으로 다운로드
- tar.gz 형식으로 압축
- 설정 스크립트와 문서 생성

### 방법 2: 수동으로 특정 데이터셋만 다운로드

```bash
# 특정 task의 데이터셋만 다운로드
python scripts/download_all_datasets.py \
    --cache-dir /path/to/cache \
    --tasks hellaswag,arc_easy,winogrande

# task 목록 파일 사용
echo "hellaswag" > tasks.txt
echo "arc_easy" >> tasks.txt
python scripts/download_all_datasets.py \
    --cache-dir /path/to/cache \
    --task-file tasks.txt
```

### 방법 3: Python으로 직접 다운로드

```python
# download_datasets.py
import datasets
from pathlib import Path

CACHE_DIR = "/path/to/offline/cache"
DATASETS_TO_DOWNLOAD = [
    ("Rowan/hellaswag", None),
    ("allenai/ai2_arc", "ARC-Easy"),
    ("allenai/ai2_arc", "ARC-Challenge"),
    ("allenai/winogrande", "winogrande_xl"),
    ("EleutherAI/piqa", None),
    ("Rowan/hellaswag", None),
    ("cais/mmlu", "all"),
    ("gsm8k", "main"),
]

for dataset_path, config_name in DATASETS_TO_DOWNLOAD:
    print(f"Downloading {dataset_path}/{config_name}...")
    try:
        dataset = datasets.load_dataset(
            dataset_path,
            config_name,
            cache_dir=CACHE_DIR,
            trust_remote_code=True
        )
        print(f"✓ Downloaded {dataset_path}")
    except Exception as e:
        print(f"✗ Failed to download {dataset_path}: {e}")

print(f"\nDatasets cached at: {CACHE_DIR}")
```

## 2단계: 데이터 전송

### 압축 및 전송

```bash
# 캐시 디렉토리 압축 (온라인 환경)
tar -czf lm_eval_datasets.tar.gz -C /path/to offline_cache/

# USB, 네트워크 드라이브 등으로 전송
cp lm_eval_datasets.tar.gz /mnt/usb/

# 오프라인 환경에서 압축 해제
tar -xzf lm_eval_datasets.tar.gz -C /workspace/
```

### 전송할 파일 목록
- `lm_eval_datasets.tar.gz` - 데이터셋 캐시
- `setup_offline_cache.sh` - 설정 스크립트 (선택)
- 평가할 모델 파일들

## 3단계: 오프라인 환경 설정

### 환경 변수 설정

```bash
# 필수 환경 변수
export HF_DATASETS_CACHE="/workspace/offline_cache"
export HF_HOME="/workspace/offline_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 확인
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE"
```

### 영구 설정 (.bashrc)

```bash
# ~/.bashrc 또는 ~/.bash_profile에 추가
cat >> ~/.bashrc << 'EOF'
# 오프라인 평가 환경
export HF_DATASETS_CACHE="/workspace/offline_cache"
export HF_HOME="/workspace/offline_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
EOF

# 적용
source ~/.bashrc
```

## 4단계: 오프라인 평가 실행

### 기본 실행

```bash
# 로컬 모델과 캐시된 데이터로 평가
lm_eval --model hf \
    --model_args pretrained=/workspace/models/pythia-160m \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```

### 여러 task 평가

```bash
# 환경 변수가 설정된 상태에서
MODELS="/workspace/models/pythia-160m" \
TASKS="hellaswag arc_easy winogrande" \
FEWSHOTS="0 5" \
./scripts/run_multiple_evals.sh
```

## 데이터셋 캐시 구조

```
offline_cache/
├── downloads/
│   ├── extracted/
│   └── *.lock files
├── Rowan___hellaswag/
│   └── default/
│       └── 1.1.0/
│           ├── dataset_info.json
│           └── *.arrow files
├── allenai___ai2_arc/
│   ├── ARC-Challenge/
│   └── ARC-Easy/
└── download_summary.json  # 다운로드 요약
```

## 문제 해결

### 1. "Dataset not found" 오류

```bash
# 캐시 디렉토리 확인
ls -la $HF_DATASETS_CACHE

# 환경 변수 확인
env | grep HF_

# 특정 데이터셋 확인
find $HF_DATASETS_CACHE -name "*hellaswag*"
```

### 2. "Connection error" 오류

```bash
# 오프라인 모드 강제
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# requests 라이브러리 오프라인 설정
export REQUESTS_CA_BUNDLE=""
export CURL_CA_BUNDLE=""
```

### 3. 캐시 버전 불일치

```python
# 캐시 메타데이터 확인
import json
from pathlib import Path

cache_dir = Path("/workspace/offline_cache")
for metadata_file in cache_dir.rglob("dataset_info.json"):
    with open(metadata_file) as f:
        info = json.load(f)
        print(f"{metadata_file.parent}: version {info.get('version')}")
```

### 4. 권한 문제

```bash
# 캐시 디렉토리 권한 수정
chmod -R 755 $HF_DATASETS_CACHE
chown -R $USER:$USER $HF_DATASETS_CACHE
```

## 캐시 검증

### 다운로드 완성도 확인

```python
# check_cache.py
import json
from pathlib import Path

def check_cache(cache_dir):
    cache_path = Path(cache_dir)
    
    # download_summary.json 확인
    summary_file = cache_path / "download_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
            print(f"Downloaded: {summary['total_downloaded']} datasets")
            print(f"Failed: {summary['failed_count']} datasets")
            
            if summary['failed_datasets']:
                print("\nFailed datasets:")
                for item in summary['failed_datasets']:
                    print(f"  - {item['dataset']}")
    
    # 실제 캐시 파일 확인
    arrow_files = list(cache_path.rglob("*.arrow"))
    print(f"\nFound {len(arrow_files)} arrow files")
    
    # 데이터셋별 정리
    datasets = {}
    for arrow_file in arrow_files:
        dataset_name = arrow_file.parts[-4]  # 보통 4단계 상위가 데이터셋 이름
        if dataset_name not in datasets:
            datasets[dataset_name] = []
        datasets[dataset_name].append(arrow_file.name)
    
    print(f"\nCached datasets ({len(datasets)}):")
    for name in sorted(datasets.keys()):
        print(f"  - {name}: {len(datasets[name])} files")

check_cache("/workspace/offline_cache")
```

## 주요 데이터셋 목록

자주 사용되는 데이터셋과 그 크기 (참고용):

| Task | Dataset | 예상 크기 |
|------|---------|----------|
| hellaswag | Rowan/hellaswag | ~50MB |
| arc_easy | allenai/ai2_arc (ARC-Easy) | ~5MB |
| arc_challenge | allenai/ai2_arc (ARC-Challenge) | ~5MB |
| winogrande | allenai/winogrande | ~10MB |
| piqa | EleutherAI/piqa | ~10MB |
| mmlu | cais/mmlu | ~500MB |
| gsm8k | gsm8k | ~10MB |
| truthfulqa | truthful_qa | ~5MB |
| boolq | boolq | ~10MB |

전체 캐시 크기는 모든 task를 포함할 경우 약 5-10GB 정도입니다.

## 자동화 예시

### 전체 오프라인 평가 워크플로우

```bash
#!/bin/bash
# offline_evaluation.sh

# 1. 환경 설정
export HF_DATASETS_CACHE="/workspace/offline_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 2. 모델 경로 설정
MODEL_PATH="/workspace/models/pythia-160m"

# 3. 평가 실행
echo "Starting offline evaluation..."
lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH \
    --tasks hellaswag,arc_easy,winogrande,piqa \
    --num_fewshot 0 \
    --device cuda:0 \
    --batch_size auto \
    --output_path results/

echo "Evaluation complete!"
```

## 팁과 모범 사례

1. **충분한 디스크 공간 확보**: 전체 데이터셋 캐시는 10GB 이상 필요할 수 있음
2. **버전 일치**: 온라인에서 다운로드한 datasets 라이브러리 버전과 오프라인 환경 버전 일치 필요
3. **증분 다운로드**: 모든 task가 필요하지 않다면 필요한 것만 선택적으로 다운로드
4. **캐시 재사용**: 한 번 준비한 캐시는 여러 평가에 재사용 가능
5. **로그 확인**: 오프라인 모드에서는 더 자세한 로그 확인이 중요

## 추가 자료

- [HuggingFace Datasets 오프라인 모드](https://huggingface.co/docs/datasets/loading#offline)
- [환경 변수 설정 가이드](https://huggingface.co/docs/datasets/package_reference/environment_variables)
- [lm-evaluation-harness 문서](../README.md)