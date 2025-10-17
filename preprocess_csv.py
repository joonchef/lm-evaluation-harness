"""
Domain Eval CSV 전처리 스크립트
원본 형식 → KMMLU 스타일 (A, B, C, D, E 컬럼)
"""
import pandas as pd
import re
import sys
from pathlib import Path

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

    answer = str(answer_text).strip()

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
    print(f"입력 파일: {input_path}")

    if not Path(input_path).exists():
        print(f"오류: 입력 파일을 찾을 수 없습니다: {input_path}")
        return False

    # CSV 로드
    try:
        df = pd.read_csv(input_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        print("UTF-8-sig 인코딩 실패, UTF-8로 재시도...")
        df = pd.read_csv(input_path, encoding='utf-8')

    print(f"로드된 데이터: {len(df)}개 문항")
    print(f"컬럼: {df.columns.tolist()}")

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

    # 출력 디렉토리 생성
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 저장
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n전처리 완료: {output_path}")
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

    # 샘플 출력
    print("\n샘플 데이터 (첫 3개):")
    for idx in range(min(3, len(df))):
        row = df.iloc[idx]
        print(f"\n[{idx+1}] {row['question'][:50]}...")
        for col in ['A', 'B', 'C', 'D', 'E']:
            if row[col]:
                print(f"  {col}. {row[col][:40]}...")
        print(f"  정답: {row['answer']}")

    return True

if __name__ == "__main__":
    # 기본 경로 설정
    input_csv = "../domain_eval/domain_mc_eval_dataset_251014.csv"
    output_csv = "lm_eval/tasks/domain_eval/data/domain_mc_eval_dataset.csv"

    # 명령줄 인자로 경로 변경 가능
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
    if len(sys.argv) > 2:
        output_csv = sys.argv[2]

    success = preprocess_csv(input_csv, output_csv)

    if success:
        print("\n전처리 성공!")
    else:
        print("\n전처리 실패!")
        sys.exit(1)
