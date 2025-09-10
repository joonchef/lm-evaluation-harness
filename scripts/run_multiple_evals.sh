#!/bin/bash
#
# 여러 모델, task, 설정으로 평가를 자동화하는 스크립트 (Linux 전용)
#
# Usage:
#   chmod +x run_multiple_evals.sh
#   ./run_multiple_evals.sh
#   
# 또는 특정 설정으로:
#   MODELS="gpt2 /path/to/local/model" TASKS="hellaswag arc_easy" ./run_multiple_evals.sh

set -euo pipefail  # 엄격한 오류 처리

# ===== 설정 변수 =====
# 환경 변수로 덮어쓸 수 있음

# 평가할 모델 목록
MODELS=${MODELS:-"gpt2"}

# 평가할 task 목록
TASKS=${TASKS:-"hellaswag arc_easy winogrande"}

# Few-shot 설정
FEWSHOTS=${FEWSHOTS:-"0 5"}

# 배치 크기 설정
BATCH_SIZES=${BATCH_SIZES:-"auto"}

# GPU 설정
DEVICE=${DEVICE:-"cuda:0"}

# 샘플 수 제한 (빠른 테스트용, 전체 평가는 주석 처리)
LIMIT=${LIMIT:-""}  # 비어있으면 전체 평가

# 결과 저장 기본 디렉토리
OUTPUT_BASE=${OUTPUT_BASE:-"results/experiments_$(date +%Y%m%d_%H%M%S)"}

# 로그 파일
LOG_FILE="${OUTPUT_BASE}/evaluation.log"

# ===== 함수 정의 =====

# 로깅 함수
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 디렉토리 생성
setup_directories() {
    mkdir -p "$OUTPUT_BASE"
    log_message "Created output directory: $OUTPUT_BASE"
}

# 설정 요약 출력
print_configuration() {
    log_message "========== Evaluation Configuration =========="
    log_message "Models: $MODELS"
    log_message "Tasks: $TASKS"
    log_message "Few-shot settings: $FEWSHOTS"
    log_message "Batch sizes: $BATCH_SIZES"
    log_message "Device: $DEVICE"
    log_message "Output directory: $OUTPUT_BASE"
    if [ -n "${LIMIT}" ]; then
        log_message "Sample limit: $LIMIT"
    fi
    log_message "=============================================="
}

# 단일 평가 실행
run_single_evaluation() {
    local model=$1
    local task=$2
    local fewshot=$3
    local batch_size=$4
    
    # 출력 디렉토리 설정
    local output_dir="${OUTPUT_BASE}/${model}/${task}_${fewshot}shot"
    if [ "$batch_size" != "auto" ]; then
        output_dir="${output_dir}_batch${batch_size}"
    fi
    
    mkdir -p "$output_dir"
    
    # 실행 명령 구성
    local cmd="lm_eval --model hf"
    cmd="$cmd --model_args pretrained=$model"
    cmd="$cmd --tasks $task"
    cmd="$cmd --num_fewshot $fewshot"
    cmd="$cmd --batch_size $batch_size"
    cmd="$cmd --device $DEVICE"
    cmd="$cmd --output_path $output_dir"
    
    if [ -n "${LIMIT}" ]; then
        cmd="$cmd --limit $LIMIT"
    fi
    
    log_message "Running: $model | $task | ${fewshot}-shot | batch_size=$batch_size"
    log_message "Command: $cmd"
    
    # 평가 실행
    if $cmd 2>&1 | tee -a "$LOG_FILE"; then
        log_message "✓ Completed: $model | $task | ${fewshot}-shot"
    else
        log_message "✗ Failed: $model | $task | ${fewshot}-shot"
        return 1
    fi
}

# 모든 평가 실행
run_all_evaluations() {
    local total_runs=0
    local successful_runs=0
    local failed_runs=0
    
    # 모델 배열로 변환
    IFS=' ' read -ra MODEL_ARRAY <<< "$MODELS"
    IFS=' ' read -ra TASK_ARRAY <<< "$TASKS"
    IFS=' ' read -ra FEWSHOT_ARRAY <<< "$FEWSHOTS"
    IFS=' ' read -ra BATCH_SIZE_ARRAY <<< "$BATCH_SIZES"
    
    # 총 실행 수 계산
    total_runs=$((${#MODEL_ARRAY[@]} * ${#TASK_ARRAY[@]} * ${#FEWSHOT_ARRAY[@]} * ${#BATCH_SIZE_ARRAY[@]}))
    log_message "Total evaluations to run: $total_runs"
    
    local current_run=0
    
    # 중첩 루프로 모든 조합 실행
    for model in "${MODEL_ARRAY[@]}"; do
        for task in "${TASK_ARRAY[@]}"; do
            for fewshot in "${FEWSHOT_ARRAY[@]}"; do
                for batch_size in "${BATCH_SIZE_ARRAY[@]}"; do
                    current_run=$((current_run + 1))
                    log_message "[$current_run/$total_runs] Starting evaluation..."
                    
                    if run_single_evaluation "$model" "$task" "$fewshot" "$batch_size"; then
                        successful_runs=$((successful_runs + 1))
                    else
                        failed_runs=$((failed_runs + 1))
                    fi
                    
                    # 진행 상황 출력
                    log_message "Progress: $current_run/$total_runs (Success: $successful_runs, Failed: $failed_runs)"
                done
            done
        done
    done
    
    log_message "========== Evaluation Summary =========="
    log_message "Total: $total_runs"
    log_message "Successful: $successful_runs"
    log_message "Failed: $failed_runs"
    log_message "========================================"
}

# 결과 집계
aggregate_results() {
    log_message "Aggregating results..."
    
    # Python 스크립트 존재 확인
    if [ -f "scripts/aggregate_results.py" ]; then
        # Markdown 보고서 생성
        python3 scripts/aggregate_results.py \
            --results-dir "$OUTPUT_BASE" \
            --output "${OUTPUT_BASE}/summary.md" \
            --format markdown
        
        # CSV 파일도 생성
        python3 scripts/aggregate_results.py \
            --results-dir "$OUTPUT_BASE" \
            --output "${OUTPUT_BASE}/summary.csv" \
            --format csv
        
        log_message "✓ Results aggregated: ${OUTPUT_BASE}/summary.md and summary.csv"
    else
        log_message "⚠ aggregate_results.py not found, skipping aggregation"
    fi
}

# 실행 시간 측정
measure_time() {
    local start_time=$1
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    log_message "Total execution time: ${hours}h ${minutes}m ${seconds}s"
}

# ===== 메인 실행 =====

main() {
    # 스크립트 실행 권한 확인
    if [ ! -x "$0" ]; then
        echo "Warning: Script is not executable. Run: chmod +x $0"
    fi
    
    # LM_EVAL_ROOT 환경 변수 확인 또는 자동 설정
    if [ -z "${LM_EVAL_ROOT:-}" ]; then
        # 환경 변수가 없으면 스크립트 위치에서 프로젝트 루트 찾기
        SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
        LM_EVAL_ROOT="$(dirname "$SCRIPT_DIR")"
        echo "LM_EVAL_ROOT not set. Using: $LM_EVAL_ROOT"
    else
        echo "Using LM_EVAL_ROOT: $LM_EVAL_ROOT"
    fi
    
    # 프로젝트 파일 존재 확인
    if [ ! -f "$LM_EVAL_ROOT/setup.py" ] && [ ! -f "$LM_EVAL_ROOT/pyproject.toml" ]; then
        echo "Error: Invalid LM_EVAL_ROOT path: $LM_EVAL_ROOT"
        echo "Cannot find setup.py or pyproject.toml"
        exit 1
    fi
    
    # lm-evaluation-harness 강제 설치
    echo "Installing lm-evaluation-harness from: $LM_EVAL_ROOT"
    pip install -e "$LM_EVAL_ROOT" --quiet --upgrade
    
    # 설치 확인
    if command -v lm_eval &> /dev/null; then
        echo "✓ lm-evaluation-harness ready"
    else
        echo "Error: Installation failed. Trying with pip3..."
        pip3 install -e "$LM_EVAL_ROOT" --upgrade
        
        if ! command -v lm_eval &> /dev/null; then
            echo "Error: Could not install lm_eval. Please check your Python environment."
            exit 1
        fi
    fi
    
    # Python 확인
    if ! command -v python3 &> /dev/null; then
        echo "Error: python3 is not installed"
        exit 1
    fi
    
    local start_time=$(date +%s)
    
    # 디렉토리 설정
    setup_directories
    
    # 설정 출력
    print_configuration
    
    # 모든 평가 실행
    run_all_evaluations
    
    # 결과 집계
    aggregate_results
    
    # 실행 시간 출력
    measure_time $start_time
    
    log_message "All evaluations completed. Results saved in: $OUTPUT_BASE"
}

# 스크립트 실행
main "$@"