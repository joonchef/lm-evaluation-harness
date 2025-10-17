# Domain Eval Integration Plan

## 개요

이 문서는 `../domain_eval` 프로젝트(공공, 금융, 국방 도메인 평가)를 `lm-evaluation-harness` 프로젝트에 통합하기 위한 상세 구현 계획입니다. 컨텍스트 초기화 후 이 문서만으로 즉시 구현을 시작할 수 있도록 작성되었습니다.

## 배경 정보

### 원본 프로젝트 (domain_eval)
- **위치**: `../domain_eval/`
- **데이터셋**: `domain_mc_eval_dataset_251014.csv` (약 6,719개 문항)
- **평가 도메인**: 공공(public), 금융(finance), 국방(defense)
- **모델**: VLLM 기반, GPT-OSS 추론 모델 지원
- **특징**: `<|start|>assistant<|channel|>final<|message|>` 마커 처리

### 통합 목표
- lm-evaluation-harness의 표준 Task 시스템으로 통합
- YAML 기반 설정으로 재구성
- 기존 VLLM 지원 및 `think_end_token` 활용
- 메타데이터 기반 집계 함수 구현 (format, lang, sub_domain)

## 핵심 기술 결정사항

### 1. GPT-OSS 모델 지원
**결정**: lm-eval의 기존 `think_end_token` 파라미터 사용
- `lm_eval/models/vllm_causallms.py:139` - `think_end_token` 파라미터 존재
- `lm_eval/models/utils.py:857-883` - `postprocess_generated_text()` 함수가 자동 처리
- domain_eval의 커스텀 마커 처리 로직 불필요

**사용법**:
```bash
lm_eval --model vllm \
    --model_args pretrained=<model>,think_end_token="<|message|>" \
    --tasks domain_eval_public
```

### 2. 태스크 네이밍
**결정**: `domain_eval_{domain}` 형식 사용 (korean 접두사 제거)
- ✅ `domain_eval_public`
- ✅ `domain_eval_finance`
- ✅ `domain_eval_defense`
- ❌ ~~`korean_domain_eval_public`~~

### 3. 메타데이터 필터링 전략
**결정**: Aggregation 함수 방식 채택 (Filter 방식 대신)

**이유**:
- Filter 방식: 각 조합마다 별도 태스크 생성 필요 (3 domains × 2 formats × 2 langs × N subdomains = 과다한 태스크)
- Aggregation 방식: 단일 태스크에서 메타데이터별 세분화된 집계 제공

**구현**:
- `utils.py`에 커스텀 집계 함수 구현:
  - `aggregate_by_format()` - text/md_table별 정확도
  - `aggregate_by_lang()` - ko/en별 정확도
  - `aggregate_by_subdomain()` - 세부 도메인별 정확도

### 4. 데이터 구조 변경 (KMMLU 스타일)
**결정**: A, B, C, D, E 컬럼 방식 채택 (KMMLU 표준과 동일)

**이유**:
- lm-eval 표준 워크플로우와 완벽한 호환
- 프롬프트 템플릿 자유로운 커스터마이징
- 선택지 개수 동적 처리 (2지/4지/5지선다 모두 지원)
- O/X 문제도 A=○, B=× 형태로 자연스럽게 통합

**변경 사항**:
- CSV 구조: `question` + 선택지 포함 → `question` + `A,B,C,D,E` 컬럼 분리
- answer 형식: ①②③④⑤ → A/B/C/D/E 통일
- 프롬프트 생성: utils.py에서 A,B,C,D,E를 ①②③④⑤로 렌더링

## Phase 0: CSV 데이터 전처리

원본 CSV의 `question` 컬럼에는 선택지가 포함되어 있습니다. KMMLU 스타일로 변환하려면 전처리가 필요합니다.

### 전처리 스크립트 (`preprocess_csv.py`)

```python
"""
Domain Eval CSV 전처리 스크립트
원본 형식 → KMMLU 스타일 (A, B, C, D, E 컬럼)
"""
import pandas as pd
import re

def parse_question_and_choices(question_text):
    """
    question 텍스트에서 순수 질문과 선택지 분리

    예시 입력:
    "다음 중 옳은 것은?\n①선택지1\n②선택지2\n③선택지3\n④선택지4"

    예시 출력:
    question: "다음 중 옳은 것은?"
    choices: ["선택지1", "선택지2", "선택지3", "선택지4"]
    """
    # 원형 번호 패턴으로 분리
    pattern = r'[①②③④⑤○×]'

    # 첫 번째 원형 번호 위치 찾기
    match = re.search(pattern, question_text)

    if not match:
        # 선택지 없는 경우 (이미 분리된 경우)
        return question_text.strip(), []

    # 순수 질문 부분
    question = question_text[:match.start()].strip()

    # 선택지 부분
    choices_text = question_text[match.start():].strip()

    # 원형 번호로 분리
    choice_lines = re.split(r'[①②③④⑤○×]', choices_text)
    choices = [c.strip() for c in choice_lines if c.strip()]

    return question, choices

def convert_answer_to_letter(answer_text, num_choices):
    """
    한글 답안 형식을 A/B/C/D/E로 변환

    ①②③④⑤ → A/B/C/D/E
    ○× → A/B
    """
    answer_map = {
        '①': 'A', '②': 'B', '③': 'C', '④': 'D', '⑤': 'E',
        '○': 'A', '×': 'B',
        '1': 'A', '2': 'B', '3': 'C', '4': 'D', '5': 'E'
    }

    answer = answer_text.strip()

    if answer in answer_map:
        return answer_map[answer]

    # 숫자로 시도
    try:
        idx = int(answer) - 1
        return chr(ord('A') + idx)
    except:
        return 'A'  # 기본값

def preprocess_csv(input_path, output_path):
    """CSV 전처리 메인 함수"""
    df = pd.read_csv(input_path, encoding='utf-8-sig')

    # 새로운 컬럼 추가
    df['A'] = ''
    df['B'] = ''
    df['C'] = ''
    df['D'] = ''
    df['E'] = ''

    for idx, row in df.iterrows():
        # question에서 순수 질문과 선택지 분리
        question, choices = parse_question_and_choices(row['question'])

        # 순수 질문으로 업데이트
        df.at[idx, 'question'] = question

        # A, B, C, D, E 컬럼에 선택지 할당
        choice_cols = ['A', 'B', 'C', 'D', 'E']
        for i, choice in enumerate(choices):
            if i < len(choice_cols):
                df.at[idx, choice_cols[i]] = choice

        # answer를 A/B/C/D/E로 변환
        original_answer = row['answer']
        df.at[idx, 'answer'] = convert_answer_to_letter(original_answer, len(choices))

    # 저장
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"전처리 완료: {output_path}")
    print(f"총 {len(df)}개 문항")

    # 통계 출력
    print("\n선택지 개수 분포:")
    choice_counts = []
    for _, row in df.iterrows():
        count = sum(1 for col in ['A', 'B', 'C', 'D', 'E'] if row[col])
        choice_counts.append(count)

    from collections import Counter
    for num, count in sorted(Counter(choice_counts).items()):
        print(f"  {num}지선다: {count}개")

if __name__ == "__main__":
    input_csv = "../domain_eval/domain_mc_eval_dataset_251014.csv"
    output_csv = "lm_eval/tasks/domain_eval/data/domain_mc_eval_dataset.csv"

    preprocess_csv(input_csv, output_csv)
```

### 실행 방법

```bash
python preprocess_csv.py
```

## Phase 1: 디렉토리 구조 생성

```bash
mkdir -p lm_eval/tasks/domain_eval/data
```

**생성할 파일 목록**:
```
lm_eval/tasks/domain_eval/
├── _domain_eval.yaml          # 그룹 정의
├── _default_yaml               # 공통 설정
├── domain_eval_public.yaml    # 공공 도메인
├── domain_eval_finance.yaml   # 금융 도메인
├── domain_eval_defense.yaml   # 국방 도메인
├── utils.py                    # 커스텀 함수들
├── README.md                   # 문서
└── data/
    └── domain_mc_eval_dataset.csv  # 전처리된 CSV
```

## Phase 2: 데이터 준비

### 2.1. CSV 전처리 실행

```bash
# Phase 0의 전처리 스크립트 실행
python preprocess_csv.py
```

### 2.2. 전처리된 CSV 구조

**컬럼 구성** (KMMLU 스타일):
- `id`: 문항 ID
- `question`: 순수 질문 텍스트 (선택지 제거됨)
- `A`, `B`, `C`, `D`, `E`: 각 선택지 (없으면 빈 문자열)
- `answer`: 정답 (A/B/C/D/E)
- `domain`: public/finance/defense
- `sub_domain`: 세부 도메인
- `format`: text/md_table
- `lang`: ko/en

**CSV 예시**:
```csv
id,question,A,B,C,D,E,answer,domain,sub_domain,format,lang
1,"다음 중 헌법기관이 아닌 것은?",국회,대통령,헌법재판소,감사원,,C,공공,헌법,text,ko
2,"국회의원의 임기는 4년이다",○,×,,,,A,공공,행정법,text,ko
3,"다음 중 옳은 것은?",선택지1,선택지2,선택지3,선택지4,선택지5,B,금융,금융규제,md_table,ko
```

## Phase 3: YAML 파일 작성

### 3.1. `_domain_eval.yaml` (그룹 정의)

```yaml
group: domain_eval
task:
  - domain_eval_public
  - domain_eval_finance
  - domain_eval_defense
aggregate_metric_list:
  - metric: acc
    weight_by_size: true
metadata:
  version: 1.0
```

### 3.2. `_default_yaml` (공통 설정)

```yaml
dataset_path: csv
dataset_kwargs:
  data_files:
    test: domain_eval/data/domain_mc_eval_dataset.csv
output_type: multiple_choice
test_split: test
doc_to_text: !function utils.doc_to_text
doc_to_choice: !function utils.doc_to_choice
doc_to_target: !function utils.doc_to_target
process_docs: !function utils.process_docs
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
  - metric: acc_by_format
    aggregation: !function utils.aggregate_by_format
    higher_is_better: true
  - metric: acc_by_lang
    aggregation: !function utils.aggregate_by_lang
    higher_is_better: true
  - metric: acc_by_subdomain
    aggregation: !function utils.aggregate_by_subdomain
    higher_is_better: true
metadata:
  version: 1.0
```

**핵심 포인트**:
- A, B, C, D, E 컬럼에서 동적으로 선택지 추출
- utils.py의 함수들이 프롬프트 생성 및 변환 처리
- KMMLU와 유사한 구조

### 3.3. 도메인별 YAML 파일

**`domain_eval_public.yaml`**:
```yaml
include: _default_yaml
task: domain_eval_public
dataset_name: Public Domain Evaluation
doc_to_domain: "public"
tag:
  - domain_eval
  - public
```

**`domain_eval_finance.yaml`**:
```yaml
include: _default_yaml
task: domain_eval_finance
dataset_name: Finance Domain Evaluation
doc_to_domain: "finance"
tag:
  - domain_eval
  - finance
```

**`domain_eval_defense.yaml`**:
```yaml
include: _default_yaml
task: domain_eval_defense
dataset_name: Defense Domain Evaluation
doc_to_domain: "defense"
tag:
  - domain_eval
  - defense
```

## Phase 4: `utils.py` 완전 구현

```python
"""
Domain Evaluation Task Utilities (KMMLU Style)

이 모듈은 공공, 금융, 국방 도메인 평가를 위한 유틸리티 함수를 제공합니다.
A, B, C, D, E 컬럼 방식을 사용하여 KMMLU와 동일한 구조를 따릅니다.
"""

from typing import Any, Dict, List
from collections import defaultdict


# ============================================================================
# 문서 전처리 함수
# ============================================================================

def process_docs(dataset, domain: str):
    """
    데이터셋을 특정 도메인으로 필터링

    Args:
        dataset: HuggingFace Dataset 객체
        domain: 'public', 'finance', 'defense' 중 하나

    Returns:
        필터링된 데이터셋
    """
    def _filter_domain(doc):
        return doc.get("domain", "").lower() == domain.lower()

    return dataset.filter(_filter_domain)


# ============================================================================
# 프롬프트 및 선택지 생성 함수들 (KMMLU 스타일)
# ============================================================================

def doc_to_text(doc: Dict[str, Any]) -> str:
    """
    문서에서 프롬프트 텍스트 생성
    A, B, C, D, E 컬럼을 ①②③④⑤로 변환하여 표시

    Args:
        doc: 문서 딕셔너리 (question, A, B, C, D, E 등 포함)

    Returns:
        프롬프트 문자열

    Example:
        Input: {"question": "다음 중 옳은 것은?", "A": "선택1", "B": "선택2", "C": "", "D": "", "E": ""}
        Output:
        '''
        다음 중 옳은 것은?
        ①선택1
        ②선택2
        정답：
        '''
    """
    text = doc['question'].strip() + "\n"

    circle_nums = ['①', '②', '③', '④', '⑤']
    choice_cols = ['A', 'B', 'C', 'D', 'E']

    idx = 0
    for col in choice_cols:
        choice = doc.get(col, '')
        if choice and str(choice).strip():
            text += f"{circle_nums[idx]}{choice}\n"
            idx += 1

    text += "정답："
    return text


def doc_to_choice(doc: Dict[str, Any]) -> List[str]:
    """
    문서에서 선택지 추출 (원형 번호 형식)
    실제 존재하는 A, B, C, D, E 컬럼만 반환

    Args:
        doc: 문서 딕셔너리

    Returns:
        원형 번호 리스트 (예: ['①', '②'] 또는 ['①', '②', '③', '④', '⑤'])

    Example:
        - O/X 문제: ['①', '②']
        - 4지선다: ['①', '②', '③', '④']
        - 5지선다: ['①', '②', '③', '④', '⑤']
    """
    circle_nums = ['①', '②', '③', '④', '⑤']
    choices = []

    for i, col in enumerate(['A', 'B', 'C', 'D', 'E']):
        choice = doc.get(col, '')
        if choice and str(choice).strip():
            choices.append(circle_nums[i])

    return choices if choices else ['①', '②', '③', '④', '⑤']


def doc_to_target(doc: Dict[str, Any]) -> int:
    """
    정답 인덱스 반환 (0-based)
    A/B/C/D/E → 0/1/2/3/4

    Args:
        doc: 문서 딕셔너리

    Returns:
        0-based 정답 인덱스

    Example:
        answer='A' → 0
        answer='B' → 1
        answer='C' → 2
    """
    answer = str(doc.get('answer', 'A')).strip().upper()

    # A, B, C, D, E → 0, 1, 2, 3, 4
    choice_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

    if answer in choice_map:
        return choice_map[answer]

    # Fallback: 첫 번째 선택지
    return 0


# ============================================================================
# 커스텀 집계 함수들
# ============================================================================

def aggregate_by_format(items: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    format 메타데이터별 정확도 계산

    Args:
        items: 평가 결과 아이템 리스트 (각 아이템은 doc과 acc 포함)

    Returns:
        {"acc_format_text": 0.85, "acc_format_md_table": 0.78} 형식의 딕셔너리
    """
    format_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for item in items:
        doc = item.get("doc", {})
        is_correct = item.get("acc", 0)  # 0 or 1
        fmt = doc.get("format", "unknown")

        format_stats[fmt]["correct"] += is_correct
        format_stats[fmt]["total"] += 1

    # 정확도 계산
    result = {}
    for fmt, stats in format_stats.items():
        if stats["total"] > 0:
            result[f"acc_format_{fmt}"] = stats["correct"] / stats["total"]

    return result


def aggregate_by_lang(items: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    lang 메타데이터별 정확도 계산

    Args:
        items: 평가 결과 아이템 리스트

    Returns:
        {"acc_lang_ko": 0.82, "acc_lang_en": 0.76} 형식의 딕셔너리
    """
    lang_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for item in items:
        doc = item.get("doc", {})
        is_correct = item.get("acc", 0)
        lang = doc.get("lang", "unknown")

        lang_stats[lang]["correct"] += is_correct
        lang_stats[lang]["total"] += 1

    # 정확도 계산
    result = {}
    for lang, stats in lang_stats.items():
        if stats["total"] > 0:
            result[f"acc_lang_{lang}"] = stats["correct"] / stats["total"]

    return result


def aggregate_by_subdomain(items: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    sub_domain 메타데이터별 정확도 계산

    Args:
        items: 평가 결과 아이템 리스트

    Returns:
        {"acc_subdomain_행정법": 0.88, "acc_subdomain_헌법": 0.79, ...} 형식의 딕셔너리
    """
    subdomain_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for item in items:
        doc = item.get("doc", {})
        is_correct = item.get("acc", 0)
        subdomain = doc.get("sub_domain", "unknown")

        subdomain_stats[subdomain]["correct"] += is_correct
        subdomain_stats[subdomain]["total"] += 1

    # 정확도 계산
    result = {}
    for subdomain, stats in subdomain_stats.items():
        if stats["total"] > 0:
            # 서브도메인 이름을 safe한 문자열로 변환 (공백 -> 언더스코어)
            safe_name = subdomain.replace(" ", "_").replace("/", "_")
            result[f"acc_subdomain_{safe_name}"] = stats["correct"] / stats["total"]

    return result
```

**주요 구현 포인트**:
1. **`doc_to_text()`**: A, B, C, D, E 컬럼을 읽어 ①②③④⑤로 변환하여 프롬프트 생성
2. **`doc_to_choice()`**: 실제 존재하는 선택지만 동적으로 반환 (2지/4지/5지 자동 감지)
3. **`doc_to_target()`**: A/B/C/D/E 문자를 0-based 인덱스로 변환
4. **집계 함수들**: 각 메타데이터 필드별로 세분화된 정확도 반환
5. **KMMLU 호환**: KMMLU와 동일한 구조로 다른 벤치마크와 일관성 유지

## Phase 5: `README.md` 작성

```markdown
# Domain Evaluation Tasks

## 개요

한국어 도메인별 평가 태스크로, 공공, 금융, 국방 분야의 전문 지식을 평가합니다.

### 데이터셋 정보

- **출처**: 자체 제작 (domain_mc_eval_dataset_251014.csv)
- **문항 수**: 약 6,719개
- **도메인**: 공공(public), 금융(finance), 국방(defense)
- **형식**: 객관식 (4-5지선다형, O/X)
- **언어**: 한국어(ko), 영어(en)
- **답안 형식**: ①②③④⑤, ○×, 1-5

### 메타데이터

각 문항은 다음 메타데이터를 포함합니다:
- `domain`: 주 도메인 (public/finance/defense)
- `sub_domain`: 세부 도메인 (예: 행정법, 금융규제, 군사전략 등)
- `format`: 문제 형식 (text: 일반 텍스트, md_table: 마크다운 테이블)
- `lang`: 언어 (ko: 한국어, en: 영어)

## 태스크 목록

### 개별 도메인 태스크

- **`domain_eval_public`**: 공공 도메인 평가
- **`domain_eval_finance`**: 금융 도메인 평가
- **`domain_eval_defense`**: 국방 도메인 평가

### 통합 그룹

- **`domain_eval`**: 전체 도메인 통합 평가 (가중 평균)

## 사용 방법

### 기본 평가

```bash
# 단일 도메인 평가
lm_eval --model hf \
    --model_args pretrained=beomi/Llama-3-Open-Ko-8B \
    --tasks domain_eval_public \
    --batch_size 8

# 전체 도메인 평가
lm_eval --model hf \
    --model_args pretrained=beomi/Llama-3-Open-Ko-8B \
    --tasks domain_eval \
    --batch_size 8
```

### GPT-OSS 추론 모델 평가

GPT-OSS 스타일 추론 모델의 경우 `think_end_token` 파라미터를 사용하세요:

```bash
lm_eval --model vllm \
    --model_args pretrained=your-gpt-oss-model,think_end_token="<|message|>",tensor_parallel_size=2 \
    --tasks domain_eval \
    --batch_size auto
```

**지원 마커**:
- `<|start|>assistant<|channel|>final<|message|>` 형식의 추론 과정 자동 제거
- `think_end_token` 파라미터로 마지막 토큰 지정

### 멀티 GPU 평가

```bash
# 데이터 병렬화
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=beomi/Llama-3-Open-Ko-8B \
    --tasks domain_eval \
    --batch_size 8

# 모델 병렬화 (vLLM)
lm_eval --model vllm \
    --model_args pretrained=large-model,tensor_parallel_size=4 \
    --tasks domain_eval \
    --batch_size auto
```

## 평가 지표

### 기본 지표

- **`acc`**: 전체 정확도

### 세분화 지표

평가 완료 후 다음 세분화 지표가 자동으로 계산됩니다:

- **`acc_format_text`**: 텍스트 형식 문제 정확도
- **`acc_format_md_table`**: 테이블 형식 문제 정확도
- **`acc_lang_ko`**: 한국어 문제 정확도
- **`acc_lang_en`**: 영어 문제 정확도
- **`acc_subdomain_*`**: 세부 도메인별 정확도

### 결과 예시

```json
{
  "results": {
    "domain_eval_public": {
      "acc": 0.75,
      "acc_format_text": 0.78,
      "acc_format_md_table": 0.68,
      "acc_lang_ko": 0.76,
      "acc_lang_en": 0.72,
      "acc_subdomain_행정법": 0.82,
      "acc_subdomain_헌법": 0.79,
      ...
    }
  }
}
```

## 확장 가능성

### 새로운 도메인 추가

1. 새로운 YAML 파일 생성: `domain_eval_{new_domain}.yaml`
2. `_domain_eval.yaml`의 task 리스트에 추가
3. CSV에 해당 도메인 데이터 추가

### 커스텀 집계 추가

`utils.py`에 새로운 집계 함수를 추가하고 `_default_yaml`의 `metric_list`에 등록:

```python
def aggregate_by_custom(items):
    # 커스텀 로직
    return {"custom_metric": value}
```

## 참고 사항

- 한국어 답안 형식(①②③④⑤, ○×)이 자동으로 처리됩니다
- 테이블 형식 문제는 마크다운으로 렌더링됩니다
- Few-shot 예시는 현재 지원하지 않습니다 (0-shot 평가)

## 문제 해결

### CSV 로딩 실패
- 경로 확인: `lm_eval/tasks/domain_eval/data/domain_mc_eval_dataset_251014.csv`
- 파일 인코딩: UTF-8 확인

### 답안 파싱 오류
- `utils.py`의 `extract_korean_choice()` 함수 디버깅
- 지원 형식: ①②③④⑤, ○×, 1-5

### 메타데이터 집계 누락
- `_default_yaml`의 `metric_list` 확인
- `utils.py`의 집계 함수 반환값 형식 확인
```

## 구현 단계별 가이드

### Step 1: 환경 준비
```bash
cd C:\workspace\lm-evaluation-harness
```

### Step 2: 디렉토리 및 데이터 준비
```bash
mkdir -p lm_eval/tasks/domain_eval/data
cp ../domain_eval/domain_mc_eval_dataset_251014.csv lm_eval/tasks/domain_eval/data/
```

### Step 3: 파일 생성 순서
1. `utils.py` (다른 파일들이 이를 참조)
2. `_default_yaml` (공통 설정)
3. `_domain_eval.yaml` (그룹 정의)
4. `domain_eval_public.yaml`
5. `domain_eval_finance.yaml`
6. `domain_eval_defense.yaml`
7. `README.md`

### Step 4: 테스트

**태스크 목록 확인**:
```bash
lm_eval --tasks list | grep domain_eval
```

**단일 도메인 테스트**:
```bash
lm_eval --model hf \
    --model_args pretrained=skt/kogpt2-base-v2 \
    --tasks domain_eval_public \
    --limit 10 \
    --batch_size 2
```

**전체 도메인 테스트**:
```bash
lm_eval --model hf \
    --model_args pretrained=skt/kogpt2-base-v2 \
    --tasks domain_eval \
    --limit 30 \
    --batch_size 2
```

### Step 5: 검증 체크리스트

- [ ] 태스크가 `lm_eval --tasks list`에 표시되는가?
- [ ] 각 도메인별로 올바른 개수의 문항이 로드되는가?
- [ ] 한국어 답안 형식(①②③④⑤)이 올바르게 파싱되는가?
- [ ] 평가 결과에 세분화 지표(format, lang, subdomain)가 포함되는가?
- [ ] GPT-OSS 모델에서 `think_end_token`이 작동하는가?

## 트러블슈팅 가이드

### 문제 1: YAML 파일 파싱 에러
**증상**: `yaml.scanner.ScannerError` 또는 유사 에러

**해결**:
- YAML 문법 검증 (들여쓰기, 콜론 뒤 공백)
- `!function` 태그 앞에 공백 없는지 확인
- Jinja2 템플릿 문법 검증

### 문제 2: utils.py 함수 import 실패
**증상**: `ModuleNotFoundError` 또는 `AttributeError`

**해결**:
- `lm_eval/tasks/domain_eval/utils.py` 경로 확인
- 함수명이 YAML의 `!function` 선언과 일치하는지 확인
- Python 문법 에러 체크: `python -m py_compile lm_eval/tasks/domain_eval/utils.py`

### 문제 3: CSV 데이터 로딩 실패
**증상**: `FileNotFoundError` 또는 빈 데이터셋

**해결**:
- 상대 경로 확인: `domain_eval/data/domain_mc_eval_dataset_251014.csv`
- CSV 인코딩 확인: UTF-8 with BOM 제거
- CSV 구조 확인: 필수 컬럼(question, choices, answer, domain) 존재 여부

### 문제 4: 집계 함수 결과 누락
**증상**: 기본 `acc`만 출력되고 세분화 지표 없음

**해결**:
- 집계 함수의 반환값 형식 확인: `Dict[str, float]`
- items 파라미터 구조 확인: `[{"doc": {...}, "acc": 0 or 1}, ...]`
- 메타데이터 필드명 확인: `doc.get("format")`, `doc.get("lang")` 등

### 문제 5: 한국어 답안 파싱 오류
**증상**: 모든 답안이 0번으로 파싱됨

**해결**:
- `extract_korean_choice()` 함수 디버깅
- 입력 텍스트 인코딩 확인 (UTF-8)
- 정규식 패턴 테스트

### 문제 6: GPT-OSS 마커 제거 안됨
**증상**: 출력에 `<|start|>assistant...` 등이 남아있음

**해결**:
- `think_end_token` 파라미터 정확히 지정: `"<|message|>"`
- VLLM 모델 사용 확인: `--model vllm`
- 토큰 존재 여부 확인: 모델 출력에 실제로 해당 토큰이 있는지

## 성능 최적화 팁

### 배치 크기 자동 조정
```bash
--batch_size auto
```

### 멀티 GPU 활용
```bash
# vLLM Tensor Parallelism
--model_args tensor_parallel_size=4

# Accelerate Data Parallelism
accelerate launch -m lm_eval ...
```

### 결과 캐싱
```bash
--cache_requests true
```

## 확장 및 수정 가이드

### 새로운 메타데이터 집계 추가

1. **`utils.py`에 함수 추가**:
```python
def aggregate_by_difficulty(items):
    """난이도별 집계"""
    difficulty_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for item in items:
        difficulty = item["doc"].get("difficulty", "unknown")
        difficulty_stats[difficulty]["correct"] += item.get("acc", 0)
        difficulty_stats[difficulty]["total"] += 1

    return {
        f"acc_difficulty_{d}": s["correct"] / s["total"]
        for d, s in difficulty_stats.items() if s["total"] > 0
    }
```

2. **`_default_yaml`에 메트릭 등록**:
```yaml
metric_list:
  - metric: acc_by_difficulty
    aggregation: !function utils.aggregate_by_difficulty
    higher_is_better: true
```

### 프롬프트 수정

`_default_yaml`의 `doc_to_text` 수정:

```yaml
# 영어 프롬프트
doc_to_text: "Question: {{question.strip()}}\nOptions:\n{% for i, choice in enumerate(choices) %}{{loop.index}}. {{choice}}\n{% endfor %}Answer: "

# Few-shot 지원 (추후)
fewshot_config:
  sampler: first_n  # 또는 random
doc_to_text: "{{question}}\n{% for ex in fewshot_examples %}Example: {{ex.question}} -> {{ex.answer}}\n{% endfor %}정답: "
```

### 새로운 도메인 추가

1. CSV에 새 도메인 데이터 추가 (domain 컬럼)
2. `domain_eval_newdomain.yaml` 생성:
```yaml
include: _default_yaml
task: domain_eval_newdomain
dataset_name: New Domain Evaluation
doc_to_domain: "newdomain"
tag:
  - domain_eval
  - newdomain
```
3. `_domain_eval.yaml`에 등록:
```yaml
task:
  - domain_eval_public
  - domain_eval_finance
  - domain_eval_defense
  - domain_eval_newdomain  # 추가
```

## 검증 스크립트

평가 결과 검증을 위한 Python 스크립트:

```python
import json

# 결과 파일 로드
with open("results.json") as f:
    results = json.load(f)

# 기본 지표 확인
task_results = results["results"]["domain_eval_public"]
print(f"Overall Accuracy: {task_results['acc']:.2%}")

# 세분화 지표 확인
for key, value in task_results.items():
    if key.startswith("acc_"):
        print(f"{key}: {value:.2%}")

# 도메인별 비교
for domain in ["public", "finance", "defense"]:
    task_name = f"domain_eval_{domain}"
    if task_name in results["results"]:
        acc = results["results"][task_name]["acc"]
        print(f"{domain.capitalize()}: {acc:.2%}")
```

## 다음 단계

이 문서를 기반으로 다음 순서로 진행하세요:

1. **Phase 1-2**: 디렉토리 생성 및 데이터 복사
2. **Phase 4**: `utils.py` 작성 (가장 중요)
3. **Phase 3**: YAML 파일들 작성
4. **Phase 5**: `README.md` 작성
5. **Step 4**: 테스트 및 검증
6. **Step 5**: 체크리스트 완료 확인

문제 발생 시 트러블슈팅 가이드를 참조하거나, lm-evaluation-harness 공식 문서를 확인하세요:
- https://github.com/EleutherAI/lm-evaluation-harness
- `lm_eval/tasks/README.md`

---

**문서 버전**: 1.0
**작성일**: 2025-10-17
**목적**: 컨텍스트 초기화 후 즉시 구현 가능한 완전한 통합 계획 제공
