"""
Question/Options 분리 스크립트
detect_choice_format.py의 출력을 받아 문제와 보기를 분리
"""
import pandas as pd
import re
import sys
import io
from pathlib import Path
from typing import Tuple, List, Optional

# UTF-8 출력 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def find_all_positions(text: str, pattern: str, is_regex: bool = False) -> List[int]:
    """텍스트에서 패턴의 모든 위치를 찾음"""
    positions = []
    if is_regex:
        for match in re.finditer(pattern, text):
            positions.append(match.start())
    else:
        start = 0
        while True:
            pos = text.find(pattern, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
    return positions


def split_by_format(text: str, choice_format: str) -> Tuple[str, str, str]:
    """
    choice_format에 따라 question과 options 분리

    Returns:
        (split_question, split_options, note)
    """
    if not isinstance(text, str) or not isinstance(choice_format, str):
        return text, "", "분리 실패: 잘못된 입력 타입"

    note = ""

    # 1. O/X문제
    if choice_format == 'O/X문제':
        return text, "", ""

    # 2. 보기_공백_숫자 (2~5지선다)
    if choice_format.startswith('보기_공백_숫자'):
        # "보기 1" 또는 "보기  1" 등의 패턴 찾기
        pattern = r'보기\s+1[\.:\)]'
        matches = list(re.finditer(pattern, text))

        if len(matches) == 0:
            return text, "", f"분리 실패: '보기 1' 패턴을 찾을 수 없음"
        elif len(matches) > 1:
            note = f"키워드 '보기 1' {len(matches)}회 발견 (위치: {', '.join(str(m.start()) for m in matches)})"

        split_pos = matches[0].start()
        return text[:split_pos].strip(), text[split_pos:].strip(), note

    # 3. 보기12_줄바꿈없음, 보기123_줄바꿈없음, 보기1234_줄바꿈없음
    if '보기' in choice_format and '줄바꿈없음' in choice_format and '괄호' not in choice_format and '콜론' not in choice_format:
        keyword = "보기1"
        positions = find_all_positions(text, keyword)

        if len(positions) == 0:
            return text, "", f"분리 실패: '{keyword}' 키워드를 찾을 수 없음"
        elif len(positions) > 1:
            note = f"키워드 '{keyword}' {len(positions)}회 발견 (위치: {', '.join(str(p) for p in positions)})"

        split_pos = positions[0]
        return text[:split_pos].strip(), text[split_pos:].strip(), note

    # 4. 보기1234_괄호_줄바꿈없음
    if choice_format == '보기1234_괄호_줄바꿈없음':
        keyword = "보기1)"
        positions = find_all_positions(text, keyword)

        if len(positions) == 0:
            return text, "", f"분리 실패: '{keyword}' 키워드를 찾을 수 없음"
        elif len(positions) > 1:
            note = f"키워드 '{keyword}' {len(positions)}회 발견 (위치: {', '.join(str(p) for p in positions)})"

        split_pos = positions[0]
        return text[:split_pos].strip(), text[split_pos:].strip(), note

    # 5. 보기1234_콜론
    if choice_format == '보기1234_콜론':
        keyword = "보기 1:"
        positions = find_all_positions(text, keyword)

        if len(positions) == 0:
            return text, "", f"분리 실패: '{keyword}' 키워드를 찾을 수 없음"
        elif len(positions) > 1:
            note = f"키워드 '{keyword}' {len(positions)}회 발견 (위치: {', '.join(str(p) for p in positions)})"

        split_pos = positions[0]
        return text[:split_pos].strip(), text[split_pos:].strip(), note

    # 6. 보기X형식(3지선다), 보기X형식(4지선다), 보기X형식(5지선다)
    if choice_format.startswith('보기X형식'):
        keyword = "보기1."
        positions = find_all_positions(text, keyword)

        if len(positions) == 0:
            return text, "", f"분리 실패: '{keyword}' 키워드를 찾을 수 없음"
        elif len(positions) > 1:
            note = f"키워드 '{keyword}' {len(positions)}회 발견 (위치: {', '.join(str(p) for p in positions)})"

        split_pos = positions[0]
        return text[:split_pos].strip(), text[split_pos:].strip(), note

    # 7. 선택지없음 - 제거 대상
    if choice_format == '선택지없음':
        return text, "", "제거 대상: 선택지없음"

    # 8. 숫자1234_줄바꿈없음
    if choice_format == '숫자1234_줄바꿈없음':
        # 8.1. "보기:1"이 있는 경우
        keyword1 = "보기:1"
        if keyword1 in text:
            positions = find_all_positions(text, keyword1)
            if len(positions) > 1:
                note = f"키워드 '{keyword1}' {len(positions)}회 발견 (위치: {', '.join(str(p) for p in positions)})"
            split_pos = positions[0]
            return text[:split_pos].strip(), text[split_pos:].strip(), note

        # 8.2. 없는 경우 - 여러 패턴 시도
        # 우선순위 1: 줄바꿈 기반 "\n1."
        pattern = r'\n1\.'
        matches = list(re.finditer(pattern, text))
        pattern_name = '\\n1.'

        # 우선순위 2: 문장부호 뒤의 "1." (줄바꿈 없는 케이스)
        if len(matches) == 0:
            pattern = r'[\?\.\!:]\s*1\.'
            matches = list(re.finditer(pattern, text))
            pattern_name = '문장부호+1.'

        # 우선순위 3: 단순 "1." 패턴
        if len(matches) == 0:
            pattern = r'1\.'
            matches = list(re.finditer(pattern, text))
            pattern_name = '1.'

        if len(matches) == 0:
            return text, "", "분리 실패: '1.' 패턴을 찾을 수 없음"

        # 매칭된 경우
        if len(matches) > 1:
            note = f"키워드 '{pattern_name}' {len(matches)}회 발견 (위치: {', '.join(str(m.start()) for m in matches)})"

        # 분리 위치 계산
        match = matches[0]
        # 줄바꿈 패턴의 경우 \n 다음부터, 나머지는 1. 부터
        if pattern_name == '\\n1.':
            split_pos = match.start() + 1  # \n 다음
            return text[:split_pos-1].strip(), text[split_pos:].strip(), note
        elif pattern_name == '문장부호+1.':
            # 문장부호는 question에 포함, 1.부터는 options
            split_pos = match.end() - 2  # "1." 앞까지
            return text[:split_pos].strip(), text[split_pos:].strip(), note
        else:
            # 단순 "1." 패턴
            split_pos = match.start()
            return text[:split_pos].strip(), text[split_pos:].strip(), note

    # 9. 원숫자(2지선다), 원숫자(3지선다), 원숫자(4지선다), 원숫자(5지선다)
    if choice_format.startswith('원숫자(') and '불완전' not in choice_format:
        keyword = "①"
        positions = find_all_positions(text, keyword)

        if len(positions) == 0:
            return text, "", f"분리 실패: '{keyword}' 키워드를 찾을 수 없음"
        elif len(positions) > 1:
            note = f"키워드 '{keyword}' {len(positions)}회 발견 (위치: {', '.join(str(p) for p in positions)})"

        split_pos = positions[-1]  # 마지막 출현 위치 사용 (문제 설명 내 ① 기호와 구분)
        return text[:split_pos].strip(), text[split_pos:].strip(), note

    # 10. 원숫자불완전 - 제거 대상
    if '원숫자불완전' in choice_format:
        return text, "", f"제거 대상: {choice_format}"

    # 11. 검은원 형식 (추가)
    if choice_format.startswith('검은원('):
        keyword = "➀"
        positions = find_all_positions(text, keyword)

        if len(positions) == 0:
            return text, "", f"분리 실패: '{keyword}' 키워드를 찾을 수 없음"
        elif len(positions) > 1:
            note = f"키워드 '{keyword}' {len(positions)}회 발견 (위치: {', '.join(str(p) for p in positions)})"

        split_pos = positions[0]
        return text[:split_pos].strip(), text[split_pos:].strip(), note

    # 12. 숫자마침표 형식 (추가)
    if choice_format.startswith('숫자마침표('):
        pattern = r'\n1\.'
        matches = list(re.finditer(pattern, text))

        if len(matches) == 0:
            return text, "", "분리 실패: '\\n1.' 패턴을 찾을 수 없음"
        elif len(matches) > 1:
            note = f"키워드 '\\n1.' {len(matches)}회 발견 (위치: {', '.join(str(m.start()) for m in matches)})"

        split_pos = matches[0].start() + 1  # \n 다음부터
        return text[:split_pos-1].strip(), text[split_pos:].strip(), note

    # 13. 숫자콜론 형식 (추가)
    if choice_format.startswith('숫자콜론('):
        pattern = r'\n1:'
        matches = list(re.finditer(pattern, text))

        if len(matches) == 0:
            return text, "", "분리 실패: '\\n1:' 패턴을 찾을 수 없음"
        elif len(matches) > 1:
            note = f"키워드 '\\n1:' {len(matches)}회 발견 (위치: {', '.join(str(m.start()) for m in matches)})"

        split_pos = matches[0].start() + 1  # \n 다음부터
        return text[:split_pos-1].strip(), text[split_pos:].strip(), note

    # 14. 숫자괄호 형식 (추가)
    if choice_format.startswith('숫자괄호('):
        pattern = r'\n1\)'
        matches = list(re.finditer(pattern, text))

        if len(matches) == 0:
            return text, "", "분리 실패: '\\n1)' 패턴을 찾을 수 없음"
        elif len(matches) > 1:
            note = f"키워드 '\\n1)' {len(matches)}회 발견 (위치: {', '.join(str(m.start()) for m in matches)})"

        split_pos = matches[0].start() + 1  # \n 다음부터
        return text[:split_pos-1].strip(), text[split_pos:].strip(), note

    # 처리되지 않은 형식
    return text, "", f"경고: 처리되지 않은 형식 '{choice_format}'"


def main():
    """메인 실행 함수"""

    # 입력 파일
    input_csv = "domain_mc_eval_dataset_251021_format_detected.csv"

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

    # choice_format 컬럼 확인
    if 'choice_format' not in df.columns:
        print("오류: 'choice_format' 컬럼이 없습니다. detect_choice_format.py를 먼저 실행하세요.")
        return False

    # 분리 처리
    print("=" * 60)
    print("Question/Options 분리 중...")
    print("=" * 60)

    split_questions = []
    split_options = []
    split_notes = []

    for idx, row in df.iterrows():
        question = row['question']
        choice_format = row['choice_format']

        q, opt, note = split_by_format(question, choice_format)

        split_questions.append(q)
        split_options.append(opt)
        split_notes.append(note)

        # 진행 상황 출력 (100개마다)
        if (idx + 1) % 100 == 0:
            print(f"  처리 중: {idx + 1}/{len(df)} 문항...")

    # 새 컬럼 추가
    df['split_question'] = split_questions
    df['split_options'] = split_options
    df['split_note'] = split_notes

    # 제거 대상 분리
    removed_mask = df['split_note'].str.contains('제거 대상:', na=False)
    df_removed = df[removed_mask].copy()
    df_kept = df[~removed_mask].copy()

    # 통계 수집
    print(f"\n" + "=" * 60)
    print("처리 결과 통계")
    print("=" * 60)

    total_count = len(df)
    kept_count = len(df_kept)
    removed_count = len(df_removed)

    # 비고별 통계
    note_with_warning = df_kept[df_kept['split_note'].str.len() > 0]
    warning_count = len(note_with_warning)
    success_count = kept_count - warning_count

    print(f"\n총 문항 수: {total_count}개")
    print(f"  ✓ 정상 분리: {success_count}개 ({success_count/total_count*100:.1f}%)")
    print(f"  ⚠ 경고 포함: {warning_count}개 ({warning_count/total_count*100:.1f}%)")
    print(f"  ✗ 제거됨: {removed_count}개 ({removed_count/total_count*100:.1f}%)")

    # 경고 유형별 통계
    if warning_count > 0:
        print(f"\n경고 유형별 분포:")
        warning_types = note_with_warning['split_note'].value_counts()
        for wtype, count in warning_types.head(10).items():
            short_type = wtype[:60] + '...' if len(wtype) > 60 else wtype
            print(f"  {short_type}: {count}개")

    # 제거 사유별 통계
    if removed_count > 0:
        print(f"\n제거 사유별 분포:")
        removal_reasons = df_removed['split_note'].value_counts()
        for reason, count in removal_reasons.items():
            print(f"  {reason}: {count}개")

    # 출력 파일 저장
    base_name = input_csv.replace('_format_detected.csv', '')
    if base_name == input_csv:  # _format_detected가 없는 경우
        base_name = input_csv.replace('.csv', '')

    output_csv = f"{base_name}_split.csv"
    removed_csv = f"{base_name}_split_removed.csv"
    report_txt = f"{base_name}_split_report.txt"

    # 유지 데이터 저장
    df_kept.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✓ 분리 완료: {output_csv}")
    print(f"  총 {kept_count}개 문항 저장")

    # 제거 데이터 저장
    if removed_count > 0:
        df_removed.to_csv(removed_csv, index=False, encoding='utf-8-sig')
        print(f"✓ 제거 데이터: {removed_csv}")
        print(f"  총 {removed_count}개 문항 저장")

    # 리포트 저장
    with open(report_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Question/Options 분리 처리 리포트\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"입력 파일: {input_csv}\n")
        f.write(f"출력 파일: {output_csv}\n")
        f.write(f"제거 파일: {removed_csv}\n\n")

        f.write(f"총 문항 수: {total_count}개\n")
        f.write(f"  ✓ 정상 분리: {success_count}개 ({success_count/total_count*100:.1f}%)\n")
        f.write(f"  ⚠ 경고 포함: {warning_count}개 ({warning_count/total_count*100:.1f}%)\n")
        f.write(f"  ✗ 제거됨: {removed_count}개 ({removed_count/total_count*100:.1f}%)\n\n")

        if warning_count > 0:
            f.write("경고 유형별 분포:\n")
            for wtype, count in warning_types.items():
                f.write(f"  {wtype}: {count}개\n")
            f.write("\n")

        if removed_count > 0:
            f.write("제거 사유별 분포:\n")
            for reason, count in removal_reasons.items():
                f.write(f"  {reason}: {count}개\n")
            f.write("\n")

        # choice_format별 통계
        f.write("\n" + "=" * 60 + "\n")
        f.write("형식별 분리 결과\n")
        f.write("=" * 60 + "\n\n")

        format_stats = df_kept.groupby('choice_format').agg({
            'split_question': 'count',
            'split_note': lambda x: (x.str.len() > 0).sum()
        }).rename(columns={'split_question': '총 개수', 'split_note': '경고 개수'})

        for fmt, row in format_stats.iterrows():
            total = row['총 개수']
            warnings = row['경고 개수']
            f.write(f"{fmt}:\n")
            f.write(f"  총 {total}개 (경고: {warnings}개)\n\n")

    print(f"✓ 리포트 저장: {report_txt}")

    print("\n" + "=" * 60)
    print("✓ 처리 완료!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = main()

    if not success:
        print("\n✗ 처리 실패!")
        sys.exit(1)
