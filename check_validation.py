import pandas as pd

df = pd.read_csv('domain_mc_eval_dataset_251014_normalized_validation_errors.csv', encoding='utf-8-sig')

print('컬럼:', df.columns.tolist())
print(f'\n총 오류 수: {len(df)}개')
print('\n첫 번째 오류:')
print(f"행번호: {df.iloc[0]['행번호']}")
print(f"오류유형: {df.iloc[0]['오류유형']}")
print(f"정답번호: {df.iloc[0]['정답번호']}")
print(f"보기개수: {df.iloc[0]['보기개수']}")
print(f"\n원본문제 (처음 300자):")
print(df.iloc[0]['원본문제'][:300])
print(f"\n정규화문제 (처음 300자):")
print(df.iloc[0]['정규화문제'][:300])
