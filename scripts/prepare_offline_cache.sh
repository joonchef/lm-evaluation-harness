#!/bin/bash
#
# 오프라인 평가를 위한 데이터셋 캐시 준비 스크립트
#
# Usage:
#   ./prepare_offline_cache.sh
#   ./prepare_offline_cache.sh /custom/cache/dir

set -euo pipefail

# ===== 설정 =====
CACHE_DIR=${1:-"./offline_cache"}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_NAME="lm_eval_datasets_${TIMESTAMP}.tar.gz"

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ===== 함수 =====
print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    print_step "Checking requirements..."
    
    # Python 확인
    if ! command -v python3 &> /dev/null; then
        print_error "python3 is not installed"
        exit 1
    fi
    
    # 필요한 패키지 확인
    python3 -c "import datasets" 2>/dev/null || {
        print_warning "datasets package not found. Installing..."
        pip install datasets tqdm pyyaml
    }
    
    echo "✓ All requirements met"
}

prepare_cache_dir() {
    print_step "Preparing cache directory: $CACHE_DIR"
    
    mkdir -p "$CACHE_DIR"
    CACHE_DIR=$(realpath "$CACHE_DIR")
    echo "Cache directory: $CACHE_DIR"
}

download_datasets() {
    print_step "Downloading all datasets..."
    
    # 스크립트 위치 찾기
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    DOWNLOAD_SCRIPT="$SCRIPT_DIR/download_all_datasets.py"
    
    if [ ! -f "$DOWNLOAD_SCRIPT" ]; then
        print_error "download_all_datasets.py not found at $DOWNLOAD_SCRIPT"
        exit 1
    fi
    
    # 다운로드 실행
    python3 "$DOWNLOAD_SCRIPT" \
        --cache-dir "$CACHE_DIR" \
        --tasks all \
        --verbose
    
    if [ $? -ne 0 ]; then
        print_error "Dataset download failed"
        exit 1
    fi
    
    echo "✓ Datasets downloaded successfully"
}

create_archive() {
    print_step "Creating archive..."
    
    # 캐시 크기 확인
    CACHE_SIZE=$(du -sh "$CACHE_DIR" | cut -f1)
    echo "Cache size: $CACHE_SIZE"
    
    # 압축
    print_step "Compressing cache directory..."
    tar -czf "$ARCHIVE_NAME" -C "$(dirname "$CACHE_DIR")" "$(basename "$CACHE_DIR")"
    
    ARCHIVE_SIZE=$(du -sh "$ARCHIVE_NAME" | cut -f1)
    echo "✓ Archive created: $ARCHIVE_NAME (Size: $ARCHIVE_SIZE)"
}

create_transfer_script() {
    print_step "Creating transfer script..."
    
    cat > "setup_offline_cache.sh" << 'EOF'
#!/bin/bash
#
# 오프라인 환경에서 캐시 설정 스크립트
#
# Usage:
#   ./setup_offline_cache.sh lm_eval_datasets_*.tar.gz

set -euo pipefail

ARCHIVE_FILE=$1
CACHE_DIR=${2:-"/workspace/datasets"}

if [ ! -f "$ARCHIVE_FILE" ]; then
    echo "Error: Archive file not found: $ARCHIVE_FILE"
    exit 1
fi

echo "Extracting datasets to $CACHE_DIR..."
mkdir -p "$(dirname "$CACHE_DIR")"
tar -xzf "$ARCHIVE_FILE" -C "$(dirname "$CACHE_DIR")"

# 환경 변수 설정 파일 생성
cat > "offline_env.sh" << 'ENV_EOF'
# 오프라인 환경 설정
export HF_DATASETS_CACHE="$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Offline environment configured:"
echo "  HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "  HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE"
ENV_EOF

echo "✓ Cache extracted successfully"
echo ""
echo "To use the offline cache, run:"
echo "  source offline_env.sh"
echo ""
echo "Or set environment variables:"
echo "  export HF_DATASETS_CACHE=$CACHE_DIR"
echo "  export HF_DATASETS_OFFLINE=1"
EOF
    
    chmod +x setup_offline_cache.sh
    echo "✓ Created setup_offline_cache.sh"
}

create_documentation() {
    print_step "Creating documentation..."
    
    cat > "OFFLINE_CACHE_README.md" << EOF
# 오프라인 데이터셋 캐시

생성 시간: $(date)
캐시 디렉토리: $CACHE_DIR
아카이브 파일: $ARCHIVE_NAME

## 사용 방법

### 1. 파일 전송
오프라인 환경으로 다음 파일들을 전송하세요:
- $ARCHIVE_NAME (데이터셋 아카이브)
- setup_offline_cache.sh (설정 스크립트)

### 2. 오프라인 환경에서 설정
\`\`\`bash
# 아카이브 압축 해제 및 설정
./setup_offline_cache.sh $ARCHIVE_NAME

# 환경 변수 설정
source offline_env.sh
\`\`\`

### 3. 평가 실행
\`\`\`bash
# 오프라인 모드로 평가 실행
lm_eval --model hf \\
    --model_args pretrained=/path/to/model \\
    --tasks hellaswag \\
    --device cuda:0
\`\`\`

## 포함된 데이터셋
$(ls -la "$CACHE_DIR" 2>/dev/null | head -20 || echo "Cache directory listing not available")

## 문제 해결

### 데이터셋을 찾을 수 없는 경우
1. 환경 변수가 올바르게 설정되었는지 확인:
   \`echo \$HF_DATASETS_CACHE\`
   \`echo \$HF_DATASETS_OFFLINE\`

2. 캐시 디렉토리가 올바른 위치에 있는지 확인

3. 권한 문제가 없는지 확인:
   \`ls -la \$HF_DATASETS_CACHE\`

### 특정 데이터셋이 없는 경우
일부 데이터셋은 다운로드에 실패했을 수 있습니다.
download_summary.json 파일을 확인하세요.
EOF
    
    echo "✓ Created OFFLINE_CACHE_README.md"
}

print_summary() {
    echo ""
    echo "========================================"
    echo "오프라인 캐시 준비 완료!"
    echo "========================================"
    echo ""
    echo "생성된 파일:"
    echo "  - $ARCHIVE_NAME (데이터셋 아카이브)"
    echo "  - setup_offline_cache.sh (설정 스크립트)"
    echo "  - OFFLINE_CACHE_README.md (사용 설명서)"
    echo ""
    echo "다음 단계:"
    echo "1. 위 파일들을 오프라인 환경으로 전송"
    echo "2. setup_offline_cache.sh 실행하여 설정"
    echo "3. source offline_env.sh로 환경 변수 설정"
    echo ""
    echo "캐시 크기: $CACHE_SIZE"
    echo "아카이브 크기: $ARCHIVE_SIZE"
}

# ===== 메인 실행 =====
main() {
    echo "======================================"
    echo "오프라인 평가를 위한 데이터셋 캐시 준비"
    echo "======================================"
    echo ""
    
    check_requirements
    prepare_cache_dir
    download_datasets
    create_archive
    create_transfer_script
    create_documentation
    print_summary
}

# 스크립트 실행
main