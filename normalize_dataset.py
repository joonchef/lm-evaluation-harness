"""
Domain Eval 데이터셋 정규화 스크립트
다양한 선택지 형식을 ①②③④⑤ (흰 원 숫자)로 통일
"""
import pandas as pd
import re
import sys
import io
from pathlib import Path
from collections import Counter

# UTF-8 출력 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def is_compound_structure(text: str) -> bool:
    """
    복합 선택지 구조 감지

    조건:
    - 한글 보기(가나다라마) 3개 이상 있고
    - 원 숫자(①②③④⑤) 3개 이상 있고
    - 한글 보기가 원 숫자보다 먼저 나타나면
    → 복합 구조! (1차 보기: 가나다, 2차 선택지: ①②③④)

    Returns:
        True: 복합 구조 (한글 보기 유지해야 함)
        False: 단순 구조 (한글 보기 변환 가능)
    """
    if not isinstance(text, str):
        return False

    # 한글 보기 개수
    hangul_count = len(re.findall(r'\n[가나다라마]\.\s', text))
    # 원 숫자 개수
    circle_count = sum([text.count(c) for c in ['①', '②', '③', '④', '⑤']])

    # 둘 다 3개 이상이어야 복합 구조 가능
    if hangul_count >= 3 and circle_count >= 3:
        # 첫 번째 한글 보기와 첫 번째 원 숫자의 위치 비교
        hangul_match = re.search(r'\n[가나다라마]\.', text)
        circle_match = re.search(r'[①②③④⑤]', text)

        if hangul_match and circle_match:
            # 한글이 원 숫자보다 먼저 나타나면 복합 구조
            return hangul_match.start() < circle_match.start()

    return False


def normalize_choices(text: str) -> tuple:
    """
    선택지 형식을 ①②③④⑤로 통일하고 줄바꿈 추가

    처리 순서 (우선순위: 원숫자 > 숫자 > 한글):
    1. 보기X 형식 정규화
    2. 검은 원 숫자 → 흰 원 숫자
    3. 숫자 보기 → 원 숫자 (줄바꿈 기반)
    4. 한글 보기 → 원 숫자 (줄바꿈 기반)

    Returns:
        (normalized_text, changes_list): 정규화된 텍스트와 적용된 변경사항 리스트
    """
    if not isinstance(text, str):
        return text, []

    changes = []  # 적용된 변경사항 기록

    # 0. 보기X 형식 전처리
    # "보기1." 또는 "보기:1." → "1."로 변환
    if re.search(r'보기:?\s*[1-5]', text):  # 콜론 선택적 매칭
        text = re.sub(r'보기:?\s*([1-5])\.?\s*', r'\n\1. ', text)
        changes.append("보기X 제거")

    # 1. 검은 원 숫자 → 흰 원 숫자
    black_to_white = {
        '➀': '①', '➁': '②', '➂': '③', '➃': '④', '➄': '⑤'
    }
    for old, new in black_to_white.items():
        if old in text:
            text = text.replace(old, new)
            if "검은 원 → 흰 원" not in changes:
                changes.append("검은 원 → 흰 원")

    # 2. 숫자 보기 처리
    # 2-0. 괄호 → 마침표 변환 (줄바꿈 없는 경우 포함, 먼저 처리)
    for i in range(1, 6):
        text = re.sub(rf'{i}\)', rf'{i}.', text)

    # 2-1. 콜론 → 마침표 변환 (줄바꿈 기반)
    if re.search(r'\n[1-5]:', text):
        text = re.sub(r'\n([1-5]):\s*', r'\n\1. ', text)
        changes.append("숫자+콜론 → 숫자+마침표")

    # 2-2. 숫자 보기 줄바꿈 추가 (보기 유무와 관계없이 항상 실행)
    # "1. [내용]2. [내용]" → "1. [내용]\n2. [내용]"
    for i in range(1, 6):  # 1, 2, 3, 4, 5 처리
        # 케이스 1: 줄바꿈 없이 숫자+마침표 앞에 오는 모든 문자 ("①1.5%2." → "①1.5%\n2.")
        # 숫자+마침표 앞에 줄바꿈이 없으면 무조건 줄바꿈 추가 ("?1. [내용]" → "?\n1. [내용]")
        text = re.sub(rf'(?<!\n)([^\n]){i}\.', rf'\1\n{i}.', text)

    # 마침표 뒤 공백 없으면 추가 ("1.[내용]" → "1. [내용]")
    text = re.sub(r'\n([1-5])\.(?=[^\s])', r'\n\1. ', text)

    # 2-3. 줄바꿈 + 숫자 + 마침표 → 원 숫자 변환
    if re.search(r'\n[1-5]\.', text):
        text = re.sub(r'\n1\.\s*', '\n①', text)
        text = re.sub(r'\n2\.\s*', '\n②', text)
        text = re.sub(r'\n3\.\s*', '\n③', text)
        text = re.sub(r'\n4\.\s*', '\n④', text)
        text = re.sub(r'\n5\.\s*', '\n⑤', text)
        changes.append("숫자 보기 → 원 숫자")

    # 3. 한글 보기 처리 (복합 구조 감지)
    if re.search(r'\n[가나다라마]\.\s', text):
        # 복합 구조 감지
        if is_compound_structure(text):
            # 복합 구조: 한글 보기 유지 (1차 보기: 가나다, 2차 선택지: ①②③④)
            changes.append("복합 구조 (한글 보기 유지)")
        else:
            # 단순 구조: 한글 보기 → 원 숫자 변환
            text = re.sub(r'\n가\.\s*', '\n①', text)
            text = re.sub(r'\n나\.\s*', '\n②', text)
            text = re.sub(r'\n다\.\s*', '\n③', text)
            text = re.sub(r'\n라\.\s*', '\n④', text)
            text = re.sub(r'\n마\.\s*', '\n⑤', text)
            changes.append("한글 보기 → 원 숫자")

    return text, changes


def analyze_patterns(text: str) -> dict:
    """텍스트에서 발견된 선택지 패턴 분석 (줄바꿈 기반)"""
    patterns = {
        # 줄바꿈 기반 패턴 (실제 선택지만)
        'hangul': len(re.findall(r'\n[가나다라마]\.\s', text)),
        'black_circle': sum([text.count(c) for c in ['➀', '➁', '➂', '➃', '➄']]),
        'white_circle': sum([text.count(c) for c in ['①', '②', '③', '④', '⑤']]),
        'number_dot': len(re.findall(r'\n[1-5]\.', text)),
        'number_colon': len(re.findall(r'\n[1-5]:', text)),
        'number_paren': len(re.findall(r'\n[1-5]\)', text)),
        'bogi_pattern': len(re.findall(r'보기\s*[1-5]', text)),
    }
    return patterns


def convert_answer_to_number(answer: str) -> int:
    """
    정답을 숫자로 변환

    Args:
        answer: 정답 문자열 (①, A, 1 등)

    Returns:
        1-based 숫자 (①/A/1 → 1, ②/B/2 → 2, ...)
    """
    if not isinstance(answer, str):
        return 0

    answer = answer.strip().upper()

    # 원 숫자
    circle_map = {'①': 1, '②': 2, '③': 3, '④': 4, '⑤': 5}
    if answer in circle_map:
        return circle_map[answer]

    # 알파벳
    if answer in ['A', 'B', 'C', 'D', 'E']:
        return ord(answer) - ord('A') + 1

    # 숫자
    try:
        return int(answer)
    except:
        return 0


def validate_question_answer(question: str, answer: str) -> tuple:
    """
    질문과 정답의 일관성 검증

    Args:
        question: 정규화된 질문 텍스트
        answer: 정답 문자열

    Returns:
        (is_valid, error_message, choice_count, answer_num)
    """
    if not isinstance(question, str):
        return False, "질문이 문자열이 아님", 0, 0

    # 원 숫자 개수 세기 (정규화 후이므로 원 숫자만 체크)
    circle_count = sum([question.count(c) for c in ['①', '②', '③', '④', '⑤']])

    # O/X 문제 체크
    ox_count = question.count('○') + question.count('×')
    if ox_count >= 2:
        # O/X 문제는 검증 통과
        return True, "정상(O/X)", 2, convert_answer_to_number(answer)

    # 정답 번호 변환
    answer_num = convert_answer_to_number(answer)

    if answer_num == 0:
        return False, "정답을 숫자로 변환 불가", circle_count, 0

    # 선택지가 없는 경우
    if circle_count == 0:
        return False, "선택지 없음", 0, answer_num

    # 정답이 보기 개수보다 큰 경우
    if answer_num > circle_count:
        return False, f"정답({answer_num})이 보기개수({circle_count})보다 큼", circle_count, answer_num

    return True, "정상", circle_count, answer_num


def classify_question_pattern(text: str) -> str:
    """
    문제의 선택지 패턴 분류 (우선순위 기반)

    우선순위: 원 숫자 > 숫자 보기 > 한글 보기
    - 원 숫자가 있으면 다른 패턴 무시 (오탐지 가능성 높음)
    - 실제 복합 구조만 mixed로 분류

    Returns:
        패턴 타입 문자열
        - 'hangul': 한글 보기
        - 'black_circle': 검은 원 숫자
        - 'white_circle': 흰 원 숫자
        - 'number_dot': 숫자+마침표
        - 'number_colon': 숫자+콜론
        - 'number_paren': 숫자+괄호
        - 'bogi_pattern': 보기X 형식
        - 'ox': O/X 문제
        - 'mixed': 여러 패턴 혼용 (진짜 복합 구조)
        - 'no_pattern': 패턴 없음
    """
    if not isinstance(text, str):
        return 'unknown'

    # 각 패턴의 개수를 세기 (줄바꿈 기반 엄격 매칭)
    hangul_count = len(re.findall(r'\n[가나다라마]\.\s', text))
    black_circle_count = sum([text.count(c) for c in ['➀', '➁', '➂', '➃', '➄']])
    white_circle_count = sum([text.count(c) for c in ['①', '②', '③', '④', '⑤']])
    number_dot_count = len(re.findall(r'\n[1-5]\.', text))
    number_colon_count = len(re.findall(r'\n[1-5]:', text))
    number_paren_count = len(re.findall(r'\n[1-5]\)', text))
    bogi_count = len(re.findall(r'보기\s*[1-5]', text))
    ox_count = text.count('○') + text.count('×')

    # 카운트 딕셔너리
    counts = {
        'white_circle': white_circle_count,
        'black_circle': black_circle_count,
        'number_dot': number_dot_count,
        'number_colon': number_colon_count,
        'number_paren': number_paren_count,
        'bogi_pattern': bogi_count,
        'hangul': hangul_count,
        'ox': ox_count
    }

    # 우선순위 1: 원 숫자 (3개 이상)
    if white_circle_count >= 3:
        # 복합 구조 체크 (한글 보기 + 원 숫자)
        if hangul_count >= 3 and is_compound_structure(text):
            return 'mixed:hangul+white_circle'

        # 다른 패턴들의 합계
        other_counts = hangul_count + number_dot_count + number_colon_count + number_paren_count

        # 원 숫자가 70% 이상 우세하면 단일 패턴
        if other_counts == 0 or white_circle_count >= other_counts * 0.7:
            return 'white_circle'

        # 그 외는 원숫자 우선
        return 'white_circle'

    # 우선순위 2: 검은 원 숫자
    if black_circle_count >= 3:
        return 'black_circle'

    # 우선순위 3: 숫자 보기 (보기X 포함)
    total_number = number_dot_count + number_colon_count + number_paren_count + bogi_count
    if total_number >= 3:
        if bogi_count > 0:
            return 'bogi_pattern'
        elif number_dot_count >= max(number_colon_count, number_paren_count):
            return 'number_dot'
        elif number_colon_count > number_paren_count:
            return 'number_colon'
        else:
            return 'number_paren'

    # 우선순위 4: 한글 보기
    if hangul_count >= 3:
        return 'hangul'

    # 우선순위 5: O/X 문제
    if ox_count >= 2:
        return 'ox'

    # 패턴 없음
    return 'no_pattern'


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

    # 케이스별 문제 수 분석
    print("케이스별 문제 수 (처리 전)")
    print("=" * 60)

    pattern_counts_before = Counter()
    for text in df['question']:
        if isinstance(text, str):
            pattern = classify_question_pattern(text)
            pattern_counts_before[pattern] += 1

    pattern_names = {
        'hangul': '한글 보기 (가나다라마)',
        'black_circle': '검은 원 숫자 (➀➁➂➃➄)',
        'white_circle': '흰 원 숫자 (①②③④⑤)',
        'number_dot': '숫자+마침표 (1.2.3.)',
        'number_colon': '숫자+콜론 (1:2:3:)',
        'number_paren': '숫자+괄호 (1)2)3))',
        'bogi_pattern': '보기X 형식 (보기1.보기2.)',
        'ox': 'O/X 문제 (○×)',
        'mixed:hangul+white_circle': '복합 구조 (한글+원숫자)',
        'no_pattern': '패턴 없음',
        'unknown': '알 수 없음',
    }

    # 모든 발견된 패턴 출력 (동적)
    for pattern, count in sorted(pattern_counts_before.items(), key=lambda x: -x[1]):
        if count > 0:
            pattern_label = pattern_names.get(pattern, pattern)
            print(f"{pattern_label}: {count}개 ({count/len(df)*100:.1f}%)")
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

    # 변경사항 추적용 리스트 (변경된 것만)
    normalization_changes = []
    # 전체 데이터 추적용 리스트 (모든 행 포함)
    all_data_changes = []

    normalized_questions = []
    for idx, row in df.iterrows():
        text = row['question']
        normalized_text, changes = normalize_choices(text)
        normalized_questions.append(normalized_text)

        # 패턴 분류
        pattern_before = classify_question_pattern(text)
        pattern_after = classify_question_pattern(normalized_text)
        has_changes = bool(changes) and text != normalized_text

        # 전체 데이터 기록 (모든 행)
        all_data_changes.append({
            'index': idx,
            'changed': has_changes,
            'pattern_before': pattern_before,
            'pattern_after': pattern_after,
            'changes': changes if changes else [],
            'original': text,
            'normalized': normalized_text,
            'domain': row.get('domain', ''),
            'sub_domain': row.get('sub_domain', ''),
            'format': row.get('format', ''),
            'lang': row.get('lang', '')
        })

        # 변경사항이 있으면 별도 기록
        if has_changes:
            normalization_changes.append({
                'index': idx,
                'pattern_before': pattern_before,
                'pattern_after': pattern_after,
                'original': text,
                'normalized': normalized_text,
                'changes': changes
            })

    df['question'] = normalized_questions

    # 데이터 검증 (정규화 후)
    print("\n" + "=" * 60)
    print("데이터 검증 중...")
    print("=" * 60)

    validation_errors = []
    # 원본 질문도 저장하기 위해 all_data_changes에서 가져오기
    original_questions = {item['index']: item['original'] for item in all_data_changes}

    for idx, row in df.iterrows():
        normalized_text = normalized_questions[idx]
        original_text = original_questions.get(idx, '')
        answer = row.get('answer', '')

        is_valid, error_msg, choice_count, answer_num = validate_question_answer(
            normalized_text, answer
        )

        if not is_valid:
            validation_errors.append({
                'index': idx,
                'error_type': error_msg,
                'answer': answer,
                'answer_num': answer_num,
                'choice_count': choice_count,
                'original_question': original_text,
                'normalized_question': normalized_text,
                'domain': row.get('domain', ''),
                'sub_domain': row.get('sub_domain', '')
            })

    print(f"검증 완료: 전체 {len(df)}개")
    print(f"  정상: {len(df) - len(validation_errors)}개 ({(len(df) - len(validation_errors))/len(df)*100:.1f}%)")
    print(f"  오류: {len(validation_errors)}개 ({len(validation_errors)/len(df)*100:.1f}%)")

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

    # 케이스별 문제 수 분석 (처리 후)
    print("케이스별 문제 수 (처리 후)")
    print("=" * 60)

    pattern_counts_after = Counter()
    for text in df['question']:
        if isinstance(text, str):
            pattern = classify_question_pattern(text)
            pattern_counts_after[pattern] += 1

    # 모든 발견된 패턴 출력 (동적)
    for pattern, count in sorted(pattern_counts_after.items(), key=lambda x: -x[1]):
        if count > 0:
            pattern_label = pattern_names.get(pattern, pattern)
            print(f"{pattern_label}: {count}개 ({count/len(df)*100:.1f}%)")
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

    # 변경사항 파일 저장
    changes_output_path = output_path.replace('.csv', '_changes.txt')
    with open(changes_output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("데이터셋 정규화 변경사항 추적 파일\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"총 변경된 문항: {len(normalization_changes)}개 / {len(df)}개\n\n")

        for i, change in enumerate(normalization_changes, 1):
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"[변경 {i}] 행 {change['index']} | 패턴: {change['pattern_before']} → {change['pattern_after']}\n")
            f.write("=" * 80 + "\n\n")

            f.write("[기존 문제]\n")
            f.write("-" * 80 + "\n")
            f.write(change['original'] + "\n\n")

            f.write("[보완된 문제]\n")
            f.write("-" * 80 + "\n")
            f.write(change['normalized'] + "\n\n")

            f.write("[보완 내용]\n")
            f.write("-" * 80 + "\n")
            for change_detail in change['changes']:
                f.write(f"  • {change_detail}\n")
            f.write("\n")

    print(f"✓ 변경사항 추적 파일 생성: {changes_output_path}")
    print(f"  변경된 문항: {len(normalization_changes)}개")

    # 전체 데이터 CSV 파일 저장 (변경/미변경 모두 포함)
    changes_csv_path = output_path.replace('.csv', '_changes.csv')
    all_changes_df = pd.DataFrame(all_data_changes)
    # changes 리스트를 세미콜론으로 구분된 문자열로 변환
    all_changes_df['changes'] = all_changes_df['changes'].apply(
        lambda x: '; '.join(x) if x else '변경없음'
    )
    # 컬럼명 한글로 변경
    all_changes_df.columns = [
        '행번호', '변경여부', '정규화전_패턴', '정규화후_패턴',
        '보완내용', '기존문제', '보완된문제',
        '도메인', '세부도메인', '형식', '언어'
    ]
    all_changes_df.to_csv(changes_csv_path, index=False, encoding='utf-8-sig')

    changed_count = all_changes_df['변경여부'].sum()
    unchanged_count = len(all_changes_df) - changed_count

    print(f"✓ 전체 데이터 CSV 파일 생성: {changes_csv_path}")
    print(f"  전체: {len(all_changes_df)}개 (변경: {changed_count}개, 미변경: {unchanged_count}개)")

    # 검증 오류 CSV 파일 저장
    if validation_errors:
        validation_csv_path = output_path.replace('.csv', '_validation_errors.csv')
        validation_df = pd.DataFrame(validation_errors)
        validation_df.columns = [
            '행번호', '오류유형', '정답', '정답번호', '보기개수',
            '원본문제', '정규화문제', '도메인', '세부도메인'
        ]
        validation_df.to_csv(validation_csv_path, index=False, encoding='utf-8-sig')

        print(f"\n⚠ 검증 오류 CSV 파일 생성: {validation_csv_path}")
        print(f"  오류 문항: {len(validation_errors)}개")

        # 오류 유형별 통계
        error_type_counts = Counter([e['error_type'] for e in validation_errors])
        print("\n  오류 유형별 통계:")
        for error_type, count in error_type_counts.most_common():
            print(f"    - {error_type}: {count}개")
    else:
        print("\n✓ 검증 오류 없음: 모든 문항이 정상입니다.")

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

    # 케이스별 변환 성공률
    print("\n케이스별 변환 성공률")
    print("=" * 60)

    converted_patterns = ['hangul', 'black_circle', 'number_dot', 'number_colon', 'number_paren']
    for pattern in converted_patterns:
        before_count = pattern_counts_before.get(pattern, 0)
        after_count = pattern_counts_after.get(pattern, 0)
        if before_count > 0:
            success_rate = (before_count - after_count) / before_count * 100
            print(f"{pattern_names[pattern]}: {before_count}개 → {after_count}개 ({success_rate:.1f}% 변환)")

    # 최종 원 숫자 비율
    white_circle_count = pattern_counts_after.get('white_circle', 0)
    ox_count = pattern_counts_after.get('ox', 0)
    print(f"\n최종 결과:")
    print(f"  원 숫자로 통일: {white_circle_count}개 ({white_circle_count/len(df)*100:.1f}%)")
    print(f"  O/X 유지: {ox_count}개 ({ox_count/len(df)*100:.1f}%)")

    return True


if __name__ == "__main__":
    # 기본 경로 설정
    input_csv = "domain_mc_eval_dataset_251014.csv"
    output_csv = "domain_mc_eval_dataset_251014_normalized.csv"

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
