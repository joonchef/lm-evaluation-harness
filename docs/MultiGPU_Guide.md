# Multi-GPU 평가 가이드

이 문서는 lm-evaluation-harness에서 여러 GPU를 활용하여 평가 속도를 향상시키는 방법을 설명합니다.

## 목차
- [개요](#개요)
- [Data Parallelism (데이터 병렬 처리)](#data-parallelism-데이터-병렬-처리)
- [Model Parallelism (모델 병렬 처리)](#model-parallelism-모델-병렬-처리)
- [혼합 병렬 처리](#혼합-병렬-처리)
- [vLLM 백엔드](#vllm-백엔드)
- [SGLang 백엔드](#sglang-백엔드)
- [모델 크기별 최적 전략](#모델-크기별-최적-전략)
- [성능 최적화 팁](#성능-최적화-팁)
- [트러블슈팅](#트러블슈팅)

## 개요

Multi-GPU를 활용하면 평가 시간을 크게 단축할 수 있습니다. 주요 방법:
- **Data Parallelism**: 각 GPU가 모델 전체를 로드하고 데이터를 분산 처리
- **Model Parallelism**: 모델을 여러 GPU에 분할하여 로드
- **Hybrid Parallelism**: 데이터와 모델 병렬 처리를 동시에 활용

## Data Parallelism (데이터 병렬 처리)

### 기본 사용법

각 GPU가 독립적인 모델 복사본을 로드하고 데이터를 나눠서 처리합니다.

```bash
# accelerate를 사용한 데이터 병렬 처리
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=EleutherAI/pythia-1.4b \
    --tasks lambada_openai,arc_easy \
    --batch_size 16
```

### 프로세스 수 지정

```bash
# 4개 GPU 사용
accelerate launch --num_processes 4 -m lm_eval \
    --model hf \
    --model_args pretrained=gpt2-large \
    --tasks hellaswag \
    --batch_size 32
```

### 장점
- **선형적 속도 향상**: GPU 수에 비례하여 속도 증가
- **구현이 간단**: 추가 설정 최소화
- **메모리 효율적**: 각 GPU가 독립적으로 작동

### 단점
- 각 GPU가 전체 모델을 로드할 수 있어야 함
- 모델이 단일 GPU 메모리보다 크면 사용 불가

## Model Parallelism (모델 병렬 처리)

### 기본 사용법

대형 모델을 여러 GPU에 분산하여 로드합니다.

```bash
# parallelize=True로 자동 분산
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-70b-hf,parallelize=True \
    --tasks hellaswag \
    --batch_size 8
```

### 고급 설정

```bash
# 세부 메모리 관리
lm_eval --model hf \
    --model_args "pretrained=large_model,parallelize=True,device_map_option=auto,max_memory_per_gpu=40GB,max_cpu_memory=100GB,offload_folder=./offload" \
    --tasks mmlu
```

### 파라미터 설명
- `device_map_option`: 모델 가중치 분산 방식 (default: "auto")
- `max_memory_per_gpu`: GPU당 최대 메모리 사용량
- `max_cpu_memory`: CPU RAM 오프로딩 최대량
- `offload_folder`: 디스크 오프로딩 폴더

## 혼합 병렬 처리

데이터와 모델 병렬 처리를 동시에 활용합니다.

```bash
# 2개 모델 복사본, 각각 2개 GPU 사용 (총 4 GPU)
accelerate launch --multi_gpu --num_processes 2 \
    -m lm_eval --model hf \
    --model_args pretrained=very_large_model,parallelize=True \
    --tasks lambada_openai,arc_easy \
    --batch_size 16
```

## vLLM 백엔드

vLLM은 최적화된 추론 엔진으로 더 빠른 평가를 제공합니다.

### 설치

```bash
pip install lm_eval[vllm]
```

### Tensor Parallelism

```bash
# 4개 GPU에 모델 분산
lm_eval --model vllm \
    --model_args pretrained=meta-llama/Llama-2-13b-hf,tensor_parallel_size=4,dtype=auto \
    --tasks gsm8k \
    --batch_size auto
```

### Data Parallelism

```bash
# 4개 모델 복사본 실행
lm_eval --model vllm \
    --model_args pretrained=model_name,data_parallel_size=4 \
    --tasks hellaswag \
    --batch_size auto
```

### 혼합 사용

```bash
# TP=2, DP=2 (총 4 GPU)
lm_eval --model vllm \
    --model_args pretrained=large_model,tensor_parallel_size=2,data_parallel_size=2,gpu_memory_utilization=0.8 \
    --tasks mmlu \
    --batch_size auto
```

### vLLM 최적화 파라미터
- `gpu_memory_utilization`: GPU 메모리 사용률 (0.8 = 80%)
- `max_model_len`: 최대 시퀀스 길이 제한
- `dtype`: 데이터 타입 (auto, float16, bfloat16)

## SGLang 백엔드

SGLang은 최신 배치 추론 최적화를 제공합니다.

### 설치

```bash
# SGLang 문서 참조하여 별도 설치 필요
# https://docs.sglang.ai/start/install.html
```

### 사용 예시

```bash
# Data Parallel with SGLang
lm_eval --model sglang \
    --model_args pretrained=model_name,dp_size=4,tp_size=1,dtype=auto \
    --tasks gsm8k_cot \
    --batch_size auto
```

### 메모리 관리

```bash
# OOM 방지를 위한 메모리 제한
lm_eval --model sglang \
    --model_args pretrained=model_name,tp_size=4,mem_fraction_static=0.7 \
    --tasks multiple_choice_task \
    --batch_size 16  # auto 대신 수동 설정
```

## 모델 크기별 최적 전략

### 소형 모델 (< 7B)

```bash
# Data Parallel 추천
accelerate launch --num_processes 4 -m lm_eval \
    --model hf \
    --model_args pretrained=EleutherAI/pythia-2.8b \
    --tasks arc_easy,hellaswag,winogrande \
    --batch_size 32
```

### 중형 모델 (7B-13B)

```bash
# vLLM Data Parallel 추천
lm_eval --model vllm \
    --model_args pretrained=meta-llama/Llama-2-7b-hf,data_parallel_size=4 \
    --tasks mmlu \
    --batch_size auto
```

### 대형 모델 (30B-70B)

```bash
# vLLM Tensor Parallel 필수
lm_eval --model vllm \
    --model_args pretrained=meta-llama/Llama-2-70b-hf,tensor_parallel_size=8 \
    --tasks hellaswag \
    --batch_size auto
```

### 초대형 모델 (>70B)

```bash
# Tensor + Data Parallel 조합
lm_eval --model vllm \
    --model_args pretrained=very_large_model,tensor_parallel_size=4,data_parallel_size=2 \
    --tasks lambada_openai \
    --batch_size auto
```

## 성능 최적화 팁

### 1. 자동 배치 크기 조정

```bash
# GPU 메모리 최대 활용
lm_eval --model hf \
    --model_args pretrained=model_name \
    --tasks task_name \
    --batch_size auto:4  # 4번 재계산하여 최적화
```

### 2. Mixed Precision

```bash
# FP16으로 메모리 절약 및 속도 향상
lm_eval --model hf \
    --model_args pretrained=model_name,dtype=float16 \
    --tasks hellaswag \
    --device cuda:0
```

### 3. 캐싱 활용

```bash
# 중단된 평가 재개 시 유용
lm_eval --model hf \
    --model_args pretrained=model_name \
    --tasks mmlu \
    --use_cache ./cache_dir \
    --cache_requests
```

### 4. Flash Attention (지원 모델)

```bash
# HuggingFace 모델에서 Flash Attention 활성화
lm_eval --model hf \
    --model_args pretrained=model_name,attn_implementation=flash_attention_2 \
    --tasks task_name
```

## 성능 비교

### 예상 속도 향상 (참고용)

| 설정 | 상대 속도 | 사용 사례 |
|------|-----------|-----------|
| 1 GPU (baseline) | 1.0x | 기준 |
| 4 GPU (Data Parallel) | ~3.5-3.8x | 소형 모델 |
| 4 GPU (vLLM DP) | ~3.8-4.0x | 중형 모델 |
| 8 GPU (vLLM TP) | 필수 | 70B+ 모델 |
| 4 GPU (vLLM TP+DP) | 최적 | 대형 모델 대량 평가 |

### 실제 측정 예시

```bash
# 성능 측정을 위한 테스트
time lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks lambada_openai \
    --limit 1000 \
    --device cuda:0

# vs

time accelerate launch --num_processes 4 -m lm_eval \
    --model hf \
    --model_args pretrained=gpt2 \
    --tasks lambada_openai \
    --limit 1000
```

## 트러블슈팅

### CUDA Out of Memory

```bash
# 배치 크기 감소
--batch_size 4

# 또는 gradient checkpointing 활성화 (지원 모델)
--model_args "pretrained=model,gradient_checkpointing=True"

# 또는 모델 병렬화
--model_args "pretrained=model,parallelize=True"
```

### 불균등한 GPU 사용률

```bash
# device_map 수동 지정
--model_args "pretrained=model,device_map={'': 0, 'lm_head': 1}"
```

### vLLM 초기화 실패

```bash
# 최대 모델 길이 제한
--model_args "pretrained=model,max_model_len=4096"

# 메모리 사용률 조정
--model_args "pretrained=model,gpu_memory_utilization=0.7"
```

### accelerate 설정 충돌

```bash
# FSDP 비활성화 확인
accelerate config

# 또는 직접 실행
ACCELERATE_USE_FSDP=0 accelerate launch ...
```

## 모니터링

### GPU 사용률 확인

```bash
# 실시간 모니터링
watch -n 1 nvidia-smi

# 또는 상세 정보
nvidia-smi dmon -i 0,1,2,3
```

### 메모리 프로파일링

```bash
# PyTorch 메모리 프로파일링 활성화
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 lm_eval ...
```

## 권장 워크플로우

1. **초기 테스트**: 작은 limit으로 설정 확인
   ```bash
   lm_eval --model hf --limit 10 ...
   ```

2. **배치 크기 최적화**: auto 사용
   ```bash
   --batch_size auto
   ```

3. **병렬화 방식 선택**: 모델 크기에 따라 결정

4. **전체 평가 실행**: 최적 설정으로 실행

## 참고 자료

- [HuggingFace Accelerate 문서](https://huggingface.co/docs/accelerate)
- [vLLM 문서](https://docs.vllm.ai/)
- [SGLang 문서](https://docs.sglang.ai/)
- [lm-evaluation-harness README](../README.md)