● Domain Evaluation Task 통합 구현 계획서

  개요

  ../domain_eval 프로젝트의 한국어 도메인별 평가(공공, 금융, 국방)를 lm-evaluation-harness에 통합합니다.

  배경 정보

  domain_eval 프로젝트 구조

  - 평가 스크립트: domain_eval.py (VLLM 기반 독립 실행)
  - 데이터셋: domain_mc_eval_dataset_251014.csv (~6,719 문제)
    - 공공: ~3,374 문제
    - 금융: ~2,379 문제
    - 국방: ~966 문제
  - 데이터 구조:
    - question: 문제 텍스트
    - answer: 정답 (①②③④⑤ 또는 ○×)
    - domain: 도메인 (공공/금융/국방)
    - sub_domain: 하위 도메인 (선택적)
    - format: 문제 형식 (text/md_table)
    - lang: 언어 (ko/en)
  - 특수 기능:
    - GPT-OSS 모델 마커 처리: <|start|>assistant<|channel|>final<|message|> → lm-eval의 think_end_token으로
  대체 가능
    - 한국어 객관식 답변 추출: ①②③④⑤, ○×, 1-5 숫자

  lm-evaluation-harness 구조

  - Task 정의: YAML 기반 (lm_eval/tasks/)
  - 모델 지원: VLLM 이미 지원됨 (lm_eval/models/vllm_causallms.py)
  - 평가 타입: multiple_choice (객관식)
  - 집계 시스템: 커스텀 집계 함수 등록 가능

  ---
  구현 계획

  Phase 1: 디렉토리 구조 생성

  lm_eval/tasks/domain_eval/
  ├── _domain_eval.yaml              # 그룹 task 정의
  ├── _default_yaml                  # 공통 설정 (프롬프트, 메트릭)
  ├── utils.py                       # 커스텀 필터 & 집계 함수
  ├── README.md                      # 사용법 문서
  ├── domain_eval_public.yaml        # 공공 도메인
  ├── domain_eval_finance.yaml       # 금융 도메인
  └── domain_eval_defense.yaml       # 국방 도메인

  Phase 2: 데이터 준비

  Option A: CSV 파일 복사 (권장)

  # CSV 파일을 lm_eval 프로젝트 내부로 복사
  mkdir -p lm_eval/tasks/domain_eval/data
  cp ../domain_eval/domain_mc_eval_dataset_251014.csv lm_eval/tasks/domain_eval/data/

  Option B: 커스텀 데이터셋 로더

  _default_yaml에서 custom_dataset 함수 정의하여 CSV 직접 로드

  Phase 3: YAML 파일 작성

  3.1 _domain_eval.yaml (그룹 정의)

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
    description: "한국어 도메인별 객관식 평가 (공공, 금융, 국방)"

  3.2 _default_yaml (공통 설정)

  dataset_path: csv
  dataset_kwargs:
    data_files:
      test: "lm_eval/tasks/domain_eval/data/domain_mc_eval_dataset_251014.csv"
  output_type: multiple_choice
  test_split: test
  doc_to_text: "주어지는 객관식 문제에 대한 정답을 하나만 골라 해당 번호를 출력하세요.\n\n{{question}}"
  doc_to_choice: !function utils.extract_choices
  doc_to_target: !function utils.normalize_answer
  process_results: !function utils.process_domain_results
  metric_list:
    - metric: acc
      aggregation: mean
      higher_is_better: true
  filter_list:
    - name: "domain_filter"
      filter:
        - function: "regex"
          regex_pattern: "①|②|③|④|⑤|○|×|1|2|3|4|5"
          group_select: 0
          fallback: ""
  metadata:
    version: 1.0

  3.3 domain_eval_public.yaml

  dataset_name: 공공
  include: _default_yaml
  task: domain_eval_public
  tag: domain_eval_tasks
  process_docs: !function utils.filter_domain_public

  3.4 domain_eval_finance.yaml

  dataset_name: 금융
  include: _default_yaml
  task: domain_eval_finance
  tag: domain_eval_tasks
  process_docs: !function utils.filter_domain_finance

  3.5 domain_eval_defense.yaml

  dataset_name: 국방
  include: _default_yaml
  task: domain_eval_defense
  tag: domain_eval_tasks
  process_docs: !function utils.filter_domain_defense

  Phase 4: utils.py 구현

  """
  Domain Evaluation Task Utilities
  한국어 도메인별 평가를 위한 커스텀 함수들
  """

  import re
  import sys
  from typing import Optional, List, Dict, Any
  from collections import defaultdict


  # ==================== 데이터 필터링 ====================

  def filter_domain_public(dataset):
      """공공 도메인 문제만 필터링"""
      return dataset.filter(lambda x: x['domain'] == '공공')


  def filter_domain_finance(dataset):
      """금융 도메인 문제만 필터링"""
      return dataset.filter(lambda x: x['domain'] == '금융')


  def filter_domain_defense(dataset):
      """국방 도메인 문제만 필터링"""
      return dataset.filter(lambda x: x['domain'] == '국방')


  # ==================== 답변 처리 ====================

  def extract_choices(doc: Dict[str, Any]) -> List[str]:
      """
      문제에서 선택지 추출

      Args:
          doc: 문제 문서 (question, answer, domain 등 포함)

      Returns:
          선택지 리스트 (①②③④⑤ 또는 ["○", "×"])
      """
      question = doc.get('question', '')

      # O/X 문제 감지
      if '○' in question or '×' in question:
          return ['○', '×']

      # 일반 객관식 (5지선다)
      return ['①', '②', '③', '④', '⑤']


  def normalize_answer(doc: Dict[str, Any]) -> str:
      """
      정답을 표준 형식으로 변환

      Args:
          doc: 문제 문서

      Returns:
          정규화된 정답 (①②③④⑤ 또는 ○×)
      """
      answer = str(doc.get('answer', '')).strip()

      # 이미 올바른 형식이면 그대로 반환
      if answer in ['①', '②', '③', '④', '⑤', '○', '×']:
          return answer

      # 숫자 → 원 숫자 변환
      number_to_circle = {
          '1': '①', '2': '②', '3': '③', '4': '④', '5': '⑤'
      }
      if answer in number_to_circle:
          return number_to_circle[answer]

      # O/X → ○/× 변환
      if answer.upper() in ['O', 'Ο', '0']:  # 영문 O, 그리스 문자 Ο, 숫자 0
          return '○'
      if answer.upper() in ['X', '×']:
          return '×'

      return answer


  def extract_korean_choice(generated_text: str) -> Optional[str]:
      """
      생성된 텍스트에서 한국어 객관식 답변 추출
      domain_eval.py의 extract_choice 로직 이식

      Args:
          generated_text: 모델이 생성한 텍스트

      Returns:
          추출된 답변 (①②③④⑤ 또는 ○×) 또는 None
      """
      if not generated_text:
          return None

      # O/X 문제 우선 처리
      if '○' in generated_text:
          return '○'
      elif '×' in generated_text:
          return '×'

      # 원 숫자 객관식 (①②③④⑤)
      options = ["①", "②", "③", "④", "⑤"]
      indices = [generated_text.find(option) for option in options]
      positive_indices = [idx for idx in indices if idx >= 0]

      if positive_indices:
          choice_idx = indices.index(min(positive_indices))
          return options[choice_idx]

      # Fallback: 일반 숫자 (1-5)
      sec_options = ["1", "2", "3", "4", "5"]
      indices = [
          generated_text.find(option) if generated_text.find(option) >= 0 else sys.maxsize
          for option in sec_options
      ]

      if min(indices) == sys.maxsize:
          # 추출 실패
          return None

      choice_idx = indices.index(min(indices))
      return options[choice_idx]


  # ==================== 결과 처리 ====================

  def process_domain_results(doc: Dict[str, Any], results: List) -> Dict[str, Any]:
      """
      도메인 평가 결과 처리

      Args:
          doc: 문제 문서
          results: 모델 응답 결과 리스트

      Returns:
          메트릭 딕셔너리 (acc, acc_by_format, acc_by_lang, acc_by_subdomain)
      """
      # multiple_choice는 loglikelihood 결과 리스트 [(ll, is_greedy), ...]
      lls = [res[0] for res in results]
      pred_idx = lls.index(max(lls))

      # 정답 인덱스
      choices = extract_choices(doc)
      gold_answer = normalize_answer(doc)
      gold_idx = choices.index(gold_answer) if gold_answer in choices else -1

      # 기본 정확도
      acc = 1.0 if pred_idx == gold_idx else 0.0

      # 메타데이터 수집 (집계용)
      result = {
          'acc': acc,
          'format': doc.get('format', 'unknown'),
          'lang': doc.get('lang', 'unknown'),
          'sub_domain': doc.get('sub_domain', '(미분류)'),
      }

      return result


  # ==================== 집계 함수 ====================

  def aggregate_by_format(items: List[Dict]) -> Dict[str, float]:
      """
      format (text/md_table)별로 정확도 집계

      Args:
          items: process_results에서 반환된 결과 리스트

      Returns:
          format별 정확도 딕셔너리
      """
      by_format = defaultdict(list)

      for item in items:
          format_type = item.get('format', 'unknown')
          by_format[format_type].append(item['acc'])

      results = {}
      for fmt, accs in by_format.items():
          if accs:
              results[f'acc_format_{fmt}'] = sum(accs) / len(accs)

      return results


  def aggregate_by_lang(items: List[Dict]) -> Dict[str, float]:
      """
      lang (ko/en)별로 정확도 집계

      Args:
          items: process_results에서 반환된 결과 리스트

      Returns:
          언어별 정확도 딕셔너리
      """
      by_lang = defaultdict(list)

      for item in items:
          lang = item.get('lang', 'unknown')
          by_lang[lang].append(item['acc'])

      results = {}
      for lng, accs in by_lang.items():
          if accs:
              results[f'acc_lang_{lng}'] = sum(accs) / len(accs)

      return results


  def aggregate_by_subdomain(items: List[Dict]) -> Dict[str, float]:
      """
      sub_domain별로 정확도 집계

      Args:
          items: process_results에서 반환된 결과 리스트

      Returns:
          sub_domain별 정확도 딕셔너리
      """
      by_subdomain = defaultdict(list)

      for item in items:
          subdomain = item.get('sub_domain', '(미분류)')
          by_subdomain[subdomain].append(item['acc'])

      results = {}
      for sd, accs in by_subdomain.items():
          if accs:
              # 서브도메인 이름을 메트릭 키로 사용 (공백을 언더스코어로)
              safe_name = sd.replace(' ', '_').replace('/', '_')
              results[f'acc_subdomain_{safe_name}'] = sum(accs) / len(accs)

      return results

  Phase 5: README.md 작성

  # Domain Evaluation Tasks

  한국어 도메인별 객관식 평가 태스크 (공공, 금융, 국방)

  ## 데이터셋

  - **출처**: domain_mc_eval_dataset_251014.csv
  - **총 문제 수**: ~6,719
    - 공공: ~3,374
    - 금융: ~2,379
    - 국방: ~966
  - **형식**: text (92%), md_table (8%)
  - **언어**: ko (93%), en (7%)

  ## 사용법

  ### 전체 도메인 평가

  ```bash
  lm_eval --model vllm \
      --model_args pretrained=/path/to/model,tensor_parallel_size=2 \
      --tasks domain_eval \
      --batch_size auto \
      --output_path results/

  특정 도메인 평가

  # 공공 도메인만
  lm_eval --model vllm \
      --model_args pretrained=/path/to/model \
      --tasks domain_eval_public

  # 금융 + 국방
  lm_eval --model vllm \
      --model_args pretrained=/path/to/model \
      --tasks domain_eval_finance,domain_eval_defense

  GPT-OSS 모델 평가

  lm_eval --model vllm \
      --model_args pretrained=/path/to/gpt-oss-model,\
  tensor_parallel_size=2,\
  think_end_token='<|start|>assistant<|channel|>final<|message|>' \
      --tasks domain_eval \
      --apply_chat_template true

  결과 출력

  Results:
    domain_eval_public:
      acc: 0.7532                          # 전체 정확도
      acc_format_text: 0.7612              # text 형식 정확도
      acc_format_md_table: 0.7216          # table 형식 정확도
      acc_lang_ko: 0.7589                  # 한국어 정확도
      acc_lang_en: 0.5862                  # 영어 정확도
      acc_subdomain_행정안전부: 0.7845     # 서브도메인별 정확도
      acc_subdomain_(미분류): 0.7321

  특징

  - 자동 답변 추출: ①②③④⑤, ○×, 1-5 숫자 자동 인식
  - 다층 집계: domain, sub_domain, format, lang별 자동 집계
  - GPT-OSS 지원: think_end_token 파라미터로 마커 자동 제거
  - VLLM 최적화: 배치 처리 및 자동 배치 크기 탐지

  Metrics

  | Metric               | Description              |
  |----------------------|--------------------------|
  | acc                  | 전체 정확도                   |
  | acc_format_{format}  | 형식별 정확도 (text, md_table) |
  | acc_lang_{lang}      | 언어별 정확도 (ko, en)         |
  | acc_subdomain_{name} | 서브도메인별 정확도               |

  Citation

  @misc{domain_eval_2024,
    title={Korean Domain-Specific LLM Evaluation},
    author={domain_eval contributors},
    year={2024},
    howpublished={\url{https://github.com/joonchef/domain_eval}}
  }

  ---

  ## 구현 순서

  ### Step 1: 디렉토리 및 파일 생성
  ```bash
  cd lm_eval/tasks
  mkdir -p domain_eval/data
  touch domain_eval/_domain_eval.yaml
  touch domain_eval/_default_yaml
  touch domain_eval/utils.py
  touch domain_eval/README.md
  touch domain_eval/domain_eval_public.yaml
  touch domain_eval/domain_eval_finance.yaml
  touch domain_eval/domain_eval_defense.yaml

  Step 2: CSV 데이터 복사

  cp ../../../domain_eval/domain_mc_eval_dataset_251014.csv \
     lm_eval/tasks/domain_eval/data/

  Step 3: YAML 파일 작성

  위 Phase 3의 내용대로 각 YAML 파일 작성

  Step 4: utils.py 구현

  위 Phase 4의 코드 작성

  Step 5: README.md 작성

  위 Phase 5의 내용 작성

  Step 6: 테스트

  # Task 목록 확인
  lm_eval --tasks list | grep domain_eval

  # 샘플 데이터로 테스트 (limit 사용)
  lm_eval --model vllm \
      --model_args pretrained=/path/to/model \
      --tasks domain_eval_public \
      --limit 10

  ---
  주의사항

  CSV 인코딩

  - domain_mc_eval_dataset_251014.csv가 UTF-8-sig 인코딩인지 확인
  - 필요시 dataset_kwargs에 encoding: utf-8-sig 추가

  YAML 함수 참조

  - !function utils.extract_choices는 lm_eval/tasks/domain_eval/utils.py의 함수를 참조
  - lm-eval이 자동으로 task 디렉토리의 utils.py를 로드함

  메트릭 집계

  - process_results가 반환하는 딕셔너리에 메타데이터 포함 필요
  - 집계 함수는 metric_list의 aggregation으로 등록

  Filter vs Aggregation

  - Filter: 모델 응답 후처리 (regex, take_first 등)
  - Aggregation: 메트릭 집계 (평균, 합계 등)
  - format/lang/subdomain은 Aggregation으로 처리

  ---
  확장 가능성

  추가 도메인

  새 도메인 추가 시:
  1. domain_eval_{new_domain}.yaml 생성
  2. utils.py에 filter_domain_{new_domain} 함수 추가
  3. _domain_eval.yaml의 task 리스트에 추가

  추가 메트릭

  - F1 score: 다중 정답 지원 시
  - Calibration metrics: 확률 예측 평가
  - Per-difficulty metrics: 난이도별 집계

  다국어 확장

  - 영어 프롬프트 버전 추가: doc_to_text_en
  - 프롬프트 A/B 테스트: 여러 프롬프트 변형 비교

  ---
  트러블슈팅

  Task not found

  # Task 등록 확인
  python -c "from lm_eval.tasks import TaskManager; tm = TaskManager(); print(tm.task_index.keys())"

  CSV 로딩 실패

  # 수동으로 CSV 로드 테스트
  import datasets
  ds = datasets.load_dataset('csv', data_files={'test': 'path/to/csv'})
  print(ds['test'][0])

  답변 추출 실패

  # utils.extract_korean_choice 단위 테스트
  from lm_eval.tasks.domain_eval.utils import extract_korean_choice
  test_cases = ["정답은 ①입니다", "답: 3", "○가 맞습니다"]
  for case in test_cases:
      print(f"{case} -> {extract_korean_choice(case)}")

  ---
  이 문서를 따라 구현하면 domain_eval을 lm-evaluation-harness에 완전히 통합할 수 있습니다.