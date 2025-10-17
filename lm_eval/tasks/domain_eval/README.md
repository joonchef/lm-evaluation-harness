# Domain Evaluation Tasks

## 개요

한국어 도메인별 평가 태스크로, 공공, 금융, 국방 분야의 전문 지식을 평가합니다.

### 데이터셋 정보

- **출처**: 자체 제작
- **문항 수**: 약 6,719개
- **도메인**: 공공(public), 금융(finance), 국방(defense)
- **형식**: 객관식 (2-5지선다형, O/X 포함)
- **언어**: 한국어(ko), 영어(en)
- **프롬프트 형식**: ①②③④⑤ (원형 번호)

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
      "acc_subdomain_헌법": 0.79
    }
  }
}
```

## 데이터 준비

### CSV 전처리

원본 CSV 파일이 question 컬럼에 선택지를 포함하고 있는 경우, KMMLU 스타일(A, B, C, D, E 컬럼)로 변환해야 합니다:

```bash
# 프로젝트 루트에서 실행
python preprocess_csv.py
```

전처리 스크립트는 다음 작업을 수행합니다:
- question에서 순수 질문과 선택지 분리
- A, B, C, D, E 컬럼으로 재구성
- answer를 ①②③④⑤에서 A/B/C/D/E로 변환

### 전처리된 CSV 구조

```csv
id,question,A,B,C,D,E,answer,domain,sub_domain,format,lang
1,"다음 중 헌법기관이 아닌 것은?",국회,대통령,헌법재판소,감사원,,C,공공,헌법,text,ko
2,"국회의원의 임기는 4년이다",○,×,,,,A,공공,행정법,text,ko
```

## 확장 가능성

### 새로운 도메인 추가

1. 새로운 YAML 파일 생성: `domain_eval_{new_domain}.yaml`
2. `_domain_eval.yaml`의 task 리스트에 추가
3. CSV에 해당 도메인 데이터 추가

### 커스텀 집계 추가

`utils.py`에 새로운 집계 함수를 추가하고 `_default_yaml`의 `metric_list`에 등록:

```python
def aggregate_by_difficulty(items):
    # 커스텀 로직
    return {"acc_difficulty_hard": value}
```

## 참고 사항

- KMMLU와 동일한 구조로 설계되어 다른 벤치마크와 일관성 유지
- 선택지 개수(2지/4지/5지선다)가 동적으로 감지됨
- O/X 문제도 자연스럽게 통합 (A=○, B=×)
- Few-shot 예시는 현재 지원하지 않습니다 (0-shot 평가)

## 문제 해결

### CSV 로딩 실패
- 경로 확인: `lm_eval/tasks/domain_eval/data/domain_mc_eval_dataset.csv`
- 파일 인코딩: UTF-8 확인
- CSV 구조 확인: A, B, C, D, E, answer 컬럼 존재 확인

### 태스크 목록에 표시 안됨
```bash
lm_eval --tasks list | grep domain_eval
```
- YAML 파일 문법 검증
- `!function` 태그 확인

### 메타데이터 집계 누락
- `_default_yaml`의 `metric_list` 확인
- `utils.py`의 집계 함수 반환값 형식 확인

## 버전 정보

- **버전**: 1.0
- **데이터 구조**: KMMLU 스타일 (A, B, C, D, E 컬럼)
- **프롬프트 형식**: 한국어 원형 번호 (①②③④⑤)
