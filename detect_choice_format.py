"""
보기 형식 검출 스크립트
다양한 선택지 형식을 검출하고 분류
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


def detect_choice_format(text: str) -> str:
    """
    선택지 형식 검출

    검출 우선순위:
    1. 원 숫자 형식 (흰원, 검은원)
    2. 줄바꿈 기반 숫자 형식 (마침표, 콜론, 괄호)
    3. 보기X 형식
    4. O/X 문제
    5. 복구 필요 형식 (보기1234 마침표/괄호/콜론, 숫자1234)
    6. 기타 (원숫자불완전, 선택지없음)

    Returns:
        형식명 문자열
    """
    if not isinstance(text, str):
        return 'unknown'

    # 1. 흰 원 숫자 (①②③④⑤)
    has_white_circle_12345 = all(c in text for c in ['①', '②', '③', '④', '⑤'])
    has_white_circle_1234 = all(c in text for c in ['①', '②', '③', '④'])
    has_white_circle_123 = all(c in text for c in ['①', '②', '③']) and '④' not in text
    has_white_circle_12 = all(c in text for c in ['①', '②']) and \
                          not any(c in text for c in ['③', '④', '⑤'])

    if has_white_circle_12345:
        return '원숫자(5지선다)'
    elif has_white_circle_1234:
        return '원숫자(4지선다)'
    elif has_white_circle_123:
        return '원숫자(3지선다)'
    elif has_white_circle_12:
        return '원숫자(2지선다)'

    # 2. 검은 원 숫자 (➀➁➂➃➄)
    has_black_circle_12345 = all(c in text for c in ['➀', '➁', '➂', '➃', '➄'])
    has_black_circle_1234 = all(c in text for c in ['➀', '➁', '➂', '➃'])

    if has_black_circle_12345:
        return '검은원(5지선다)'
    elif has_black_circle_1234:
        return '검은원(4지선다)'

    # 3. 줄바꿈 기반 숫자+마침표 (\n1. \n2. \n3. \n4.)
    has_number_dot_12345 = all(re.search(rf'\n{i}\.', text) for i in [1, 2, 3, 4, 5])
    has_number_dot_1234 = all(re.search(rf'\n{i}\.', text) for i in [1, 2, 3, 4])

    if has_number_dot_12345:
        return '숫자마침표(5지선다)'
    elif has_number_dot_1234:
        return '숫자마침표(4지선다)'

    # 4. 줄바꿈 기반 숫자+콜론 (\n1: \n2: \n3: \n4:)
    has_number_colon_12345 = all(re.search(rf'\n{i}:', text) for i in [1, 2, 3, 4, 5])
    has_number_colon_1234 = all(re.search(rf'\n{i}:', text) for i in [1, 2, 3, 4])

    if has_number_colon_12345:
        return '숫자콜론(5지선다)'
    elif has_number_colon_1234:
        return '숫자콜론(4지선다)'

    # 5. 줄바꿈 기반 숫자+괄호 (\n1) \n2) \n3) \n4))
    has_number_paren_12345 = all(re.search(rf'\n{i}\)', text) for i in [1, 2, 3, 4, 5])
    has_number_paren_1234 = all(re.search(rf'\n{i}\)', text) for i in [1, 2, 3, 4])

    if has_number_paren_12345:
        return '숫자괄호(5지선다)'
    elif has_number_paren_1234:
        return '숫자괄호(4지선다)'

    # 6. 보기X 형식 (줄바꿈 기반)
    has_bogi_12345 = all(f'보기{i}' in text for i in [1, 2, 3, 4, 5])
    has_bogi_1234 = all(f'보기{i}' in text for i in [1, 2, 3, 4])

    if has_bogi_12345:
        return '보기X형식(5지선다)'
    elif has_bogi_1234:
        return '보기X형식(4지선다)'

    # 7. O/X 문제
    has_ox = '○' in text and '×' in text
    if has_ox:
        return 'O/X문제'

    # 8. 복구 필요 형식: 보기1234_줄바꿈없음 (마침표)
    # 보기1이 있고, 2. 3. 4. 포함 (하지만 줄바꿈 기반 보기X 형식은 아님)
    if '보기1' in text and all(f'{i}.' in text for i in [2, 3, 4]):
        return '보기1234_줄바꿈없음'

    # 9. 복구 필요 형식: 보기1234_괄호_줄바꿈없음 (괄호)
    # 보기1)이 있고, 2) 3) 4) 포함
    if '보기1)' in text and all(f'{i})' in text for i in [2, 3, 4]):
        return '보기1234_괄호_줄바꿈없음'

    # 10. 복구 필요 형식: 보기1234_콜론 (띄어쓰기 포함)
    # 보기 1: 보기 2: 보기 3: 보기 4: 모두 포함
    if all(f'보기 {i}:' in text for i in [1, 2, 3, 4]):
        return '보기1234_콜론'

    # 11. 복구 필요 형식: 숫자1234_줄바꿈없음
    # 1. 2. 3. 4. 모두 포함 (하지만 줄바꿈 기반 숫자 형식은 아님)
    if all(f'{i}.' in text for i in [1, 2, 3, 4]):
        return '숫자1234_줄바꿈없음'

    # 12. 원 숫자 불완전 (일부만 포함)
    circle_count = sum([text.count(c) for c in ['①', '②', '③', '④', '⑤']])
    if circle_count > 0:
        return f'원숫자불완전({circle_count}개)'

    # 13. 선택지 없음
    return '선택지없음'


def analyze_format_distribution(df: pd.DataFrame) -> dict:
    """형식별 분포 분석"""
    format_counts = Counter()

    for q in df['question']:
        fmt = detect_choice_format(q)
        format_counts[fmt] += 1

    return format_counts


def main():
    """메인 실행 함수"""

    # 입력 파일
    input_csv = "domain_mc_eval_dataset_251021.csv"

    # 명령줄 인자로 경로 변경 가능
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]

    print(f"입력 파일: {input_csv}")

    if not Path(input_csv).exists():
        print(f"오류: 입력 파일을 찾을 수 없습니다: {input_csv}")
        return False

    # CSV 로드
    try:
        df = pd.read_csv(input_csv, encoding='utf-8-sig')
        print(f"✓ 로드 완료: {len(df)}개 문항\n")
    except UnicodeDecodeError:
        print("UTF-8-sig 인코딩 실패, UTF-8로 재시도...")
        df = pd.read_csv(input_csv, encoding='utf-8')
        print(f"✓ 로드 완료: {len(df)}개 문항\n")

    # 형식 검출
    print("=" * 60)
    print("형식 검출 중...")
    print("=" * 60)

    choice_formats = []
    for q in df['question']:
        fmt = detect_choice_format(q)
        choice_formats.append(fmt)

    df['choice_format'] = choice_formats

    # 형식별 분포 분석
    format_counts = Counter(choice_formats)

    print(f"\n형식별 분포 ({len(format_counts)}가지 형식)")
    print("=" * 60)

    # 정상 형식과 복구 필요 형식 구분
    normal_formats = [
        '원숫자(5지선다)', '원숫자(4지선다)', '원숫자(3지선다)', '원숫자(2지선다)',
        '검은원(5지선다)', '검은원(4지선다)',
        '숫자마침표(5지선다)', '숫자마침표(4지선다)',
        '숫자콜론(5지선다)', '숫자콜론(4지선다)',
        '숫자괄호(5지선다)', '숫자괄호(4지선다)',
        '보기X형식(5지선다)', '보기X형식(4지선다)',
        'O/X문제'
    ]

    recovery_formats = [
        '보기1234_줄바꿈없음',
        '보기1234_괄호_줄바꿈없음',
        '보기1234_콜론',
        '숫자1234_줄바꿈없음'
    ]

    print("\n▶ 정상 형식 (복구 불필요):")
    normal_total = 0
    for fmt in normal_formats:
        count = format_counts.get(fmt, 0)
        if count > 0:
            pct = count / len(df) * 100
            print(f"  {fmt}: {count}개 ({pct:.1f}%)")
            normal_total += count

    print(f"\n▶ 복구 필요 형식:")
    recovery_total = 0
    for fmt in recovery_formats:
        count = format_counts.get(fmt, 0)
        if count > 0:
            pct = count / len(df) * 100
            print(f"  {fmt}: {count}개 ({pct:.1f}%)")
            recovery_total += count

    print(f"\n▶ 기타:")
    other_total = 0
    for fmt, count in sorted(format_counts.items(), key=lambda x: -x[1]):
        if fmt not in normal_formats and fmt not in recovery_formats:
            pct = count / len(df) * 100
            print(f"  {fmt}: {count}개 ({pct:.1f}%)")
            other_total += count

    # 통계 요약
    print(f"\n" + "=" * 60)
    print(f"통계 요약")
    print("=" * 60)
    print(f"정상 형식: {normal_total}개 ({normal_total/len(df)*100:.1f}%)")
    print(f"복구 필요: {recovery_total}개 ({recovery_total/len(df)*100:.1f}%)")
    print(f"기타: {other_total}개 ({other_total/len(df)*100:.1f}%)")
    print(f"총계: {len(df)}개")

    # 도메인별 형식 분포
    if 'domain' in df.columns:
        print(f"\n" + "=" * 60)
        print(f"도메인별 형식 분포")
        print("=" * 60)

        for domain in df['domain'].unique():
            domain_df = df[df['domain'] == domain]
            domain_format_counts = Counter(domain_df['choice_format'])

            print(f"\n▶ {domain} 도메인 ({len(domain_df)}개):")
            for fmt, count in sorted(domain_format_counts.items(), key=lambda x: -x[1])[:5]:
                pct = count / len(domain_df) * 100
                print(f"  {fmt}: {count}개 ({pct:.1f}%)")

    # 출력 파일 저장
    output_csv = input_csv.replace('.csv', '_format_detected.csv')
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print(f"\n" + "=" * 60)
    print(f"✓ 형식 검출 완료: {output_csv}")
    print(f"  총 {len(df)}개 문항 처리")
    print(f"  새 컬럼: choice_format")

    return True


if __name__ == "__main__":
    success = main()

    if success:
        print("\n✓ 형식 검출 성공!")
    else:
        print("\n✗ 형식 검출 실패!")
        sys.exit(1)
