"""
Domain Eval 데이터셋 정규화 스크립트
다양한 선택지 형식을 ①②③④⑤ (흰 원 숫자)로 통일
"""
import pandas as pd
import re
import sys
from pathlib import Path
from collections import Counter


def normalize_choices(text: str) -> str:
    """
    선택지 형식을 ①②③④⑤로 통일하고 줄바꿈 추가

    처리 순서:
    1. 한글 보기 (줄바꿈 포함) → 원 숫자
    2. 검은 원 숫자 → 흰 원 숫자
    3. 숫자 보기 → 원 숫자 (줄바꿈 추가)
    """
    if not isinstance(text, str):
        return text

    original = text

    # 1. 한글 보기 → 원 숫자 (줄바꿈 포함하여 매칭)
    text = re.sub(r'\n가\.', '\n①', text)
    text = re.sub(r'\n나\.', '\n②', text)
    text = re.sub(r'\n다\.', '\n③', text)
    text = re.sub(r'\n라\.', '\n④', text)
    text = re.sub(r'\n마\.', '\n⑤', text)

    # 2. 검은 원 숫자 → 흰 원 숫자
    replacements = {
        '➀': '①',
        '➁': '②',
        '➂': '③',
        '➃': '④',
        '➄': '⑤'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # 3. 숫자 보기 처리
    # 3-1. 숫자 + (마침표/콜론/괄호) 앞에 줄바꿈 추가 (이미 있으면 중복 방지)
    text = re.sub(r'(?<!\n)([1-5][.:\)])', r'\n\1', text)

    # 3-2. 콜론과 괄호를 마침표로 통일
    text = re.sub(r'\n([1-5]):', r'\n\1.', text)  # 1: → 1.
    text = re.sub(r'\n([1-5])\)', r'\n\1.', text)  # 1) → 1.

    # 3-3. 줄바꿈 + 숫자 + 마침표를 원 숫자로 변환
    text = re.sub(r'\n1\.', '\n①', text)
    text = re.sub(r'\n2\.', '\n②', text)
    text = re.sub(r'\n3\.', '\n③', text)
    text = re.sub(r'\n4\.', '\n④', text)
    text = re.sub(r'\n5\.', '\n⑤', text)

    return text


def analyze_patterns(text: str) -> dict:
    """텍스트에서 발견된 선택지 패턴 분석"""
    patterns = {
        'hangul': len(re.findall(r'\n[가나다라마]\.', text)),
        'black_circle': sum([text.count(c) for c in ['➀', '➁', '➂', '➃', '➄']]),
        'white_circle': sum([text.count(c) for c in ['①', '②', '③', '④', '⑤']]),
        'number_dot': len(re.findall(r'\n[1-5]\.', text)),
        'number_colon': len(re.findall(r'\n[1-5]:', text)),
        'number_paren': len(re.findall(r'\n[1-5]\)', text)),
    }
    return patterns


def normalize_dataset(input_path: str, output_path: str):
    """데이터셋 정규화 메인 함수"""

    print(f"입력 파일: {input_path}")

    if not Path(input_path).exists():
        print(f"오류: 입력 파일을 찾을 수 없습니다: {input_path}")
        return False

    # CSV 로드
    try:
        df = pd.read_csv(input_path, encoding='utf-8-sig')
        print(f"✓ 로드 완료: {len(df)}개 문항")
    except UnicodeDecodeError:
        print("UTF-8-sig 인코딩 실패, UTF-8로 재시도...")
        df = pd.read_csv(input_path, encoding='utf-8')
        print(f"✓ 로드 완료: {len(df)}개 문항")

    print(f"컬럼: {df.columns.tolist()}\n")

    # 처리 전 패턴 분석
    print("=" * 60)
    print("처리 전 패턴 분석")
    print("=" * 60)

    total_patterns = Counter()
    for text in df['question']:
        if isinstance(text, str):
            patterns = analyze_patterns(text)
            total_patterns.update(patterns)

    print(f"한글 보기 (가나다라마): {total_patterns['hangul']}개")
    print(f"검은 원 숫자 (➀➁➂➃➄): {total_patterns['black_circle']}개")
    print(f"흰 원 숫자 (①②③④⑤): {total_patterns['white_circle']}개")
    print(f"숫자+마침표 (1.2.3.): {total_patterns['number_dot']}개")
    print(f"숫자+콜론 (1:2:3:): {total_patterns['number_colon']}개")
    print(f"숫자+괄호 (1)2)3)): {total_patterns['number_paren']}개")
    print()

    # 처리 전 샘플 출력
    print("=" * 60)
    print("처리 전 샘플 (첫 3개)")
    print("=" * 60)
    for idx in range(min(3, len(df))):
        question = df.iloc[idx]['question']
        if isinstance(question, str):
            print(f"\n[{idx+1}] {question[:200]}")
            if len(question) > 200:
                print("  ...")
    print()

    # question 컬럼 정규화
    print("=" * 60)
    print("정규화 진행 중...")
    print("=" * 60)

    df['question'] = df['question'].apply(normalize_choices)

    # 처리 후 패턴 분석
    print("\n처리 후 패턴 분석")
    print("=" * 60)

    total_patterns_after = Counter()
    for text in df['question']:
        if isinstance(text, str):
            patterns = analyze_patterns(text)
            total_patterns_after.update(patterns)

    print(f"한글 보기 (가나다라마): {total_patterns_after['hangul']}개")
    print(f"검은 원 숫자 (➀➁➂➃➄): {total_patterns_after['black_circle']}개")
    print(f"흰 원 숫자 (①②③④⑤): {total_patterns_after['white_circle']}개")
    print(f"숫자+마침표 (1.2.3.): {total_patterns_after['number_dot']}개")
    print(f"숫자+콜론 (1:2:3:): {total_patterns_after['number_colon']}개")
    print(f"숫자+괄호 (1)2)3)): {total_patterns_after['number_paren']}개")
    print()

    # 처리 후 샘플 출력
    print("=" * 60)
    print("처리 후 샘플 (첫 3개)")
    print("=" * 60)
    for idx in range(min(3, len(df))):
        question = df.iloc[idx]['question']
        if isinstance(question, str):
            print(f"\n[{idx+1}] {question[:200]}")
            if len(question) > 200:
                print("  ...")
    print()

    # 출력 디렉토리 생성
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 저장
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 정규화 완료: {output_path}")
    print(f"  총 {len(df)}개 문항 처리")

    # 변환 통계
    print("\n" + "=" * 60)
    print("변환 통계")
    print("=" * 60)
    print(f"한글 보기 → 원 숫자: {total_patterns['hangul'] - total_patterns_after['hangul']}개 변환")
    print(f"검은 원 → 흰 원: {total_patterns['black_circle'] - total_patterns_after['black_circle']}개 변환")
    print(f"숫자+마침표 → 원 숫자: {total_patterns['number_dot'] - total_patterns_after['number_dot']}개 변환")
    print(f"숫자+콜론 → 원 숫자: {total_patterns['number_colon'] - total_patterns_after['number_colon']}개 변환")
    print(f"숫자+괄호 → 원 숫자: {total_patterns['number_paren'] - total_patterns_after['number_paren']}개 변환")
    print(f"최종 원 숫자 총계: {total_patterns_after['white_circle']}개")

    return True


if __name__ == "__main__":
    # 기본 경로 설정
    input_csv = "../domain_eval/domain_mc_eval_dataset_251014.csv"
    output_csv = "../domain_eval/domain_mc_eval_dataset_251014_normalized.csv"

    # 명령줄 인자로 경로 변경 가능
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
    if len(sys.argv) > 2:
        output_csv = sys.argv[2]

    success = normalize_dataset(input_csv, output_csv)

    if success:
        print("\n✓ 정규화 성공!")
        print(f"\n다음 단계:")
        print(f"  python preprocess_csv.py {output_csv}")
    else:
        print("\n✗ 정규화 실패!")
        sys.exit(1)
