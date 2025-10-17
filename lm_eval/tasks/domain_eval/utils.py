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
