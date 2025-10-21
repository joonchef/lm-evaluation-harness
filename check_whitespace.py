import pandas as pd

df = pd.read_csv('domain_mc_eval_dataset_251014_normalized_clean.csv', encoding='utf-8-sig')
questions = df['question']

leading_ws = sum(1 for q in questions if q != q.lstrip())
trailing_ws = sum(1 for q in questions if q != q.rstrip())

print(f'총 {len(df)}개 문항 중:')
print(f'  앞쪽 공백/줄바꿈 있음: {leading_ws}개')
print(f'  뒤쪽 공백/줄바꿈 있음: {trailing_ws}개')
print(f'\nstrip() 적용 확인: {"✓ 모두 제거됨" if leading_ws == 0 and trailing_ws == 0 else "✗ 아직 남아있음"}')

# 샘플 몇 개 출력 (repr로 공백 확인)
print('\n샘플 확인 (repr로 앞뒤 문자 확인):')
for i in range(min(3, len(questions))):
    q = questions.iloc[i]
    print(f'\n[{i+1}] 앞 30자: {repr(q[:30])}')
    print(f'    뒤 30자: {repr(q[-30:])}')
