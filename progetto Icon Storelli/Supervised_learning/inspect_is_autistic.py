import pandas as pd
feature = ["A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score","A9_Score","A10_Score","age","gender","ethnicity","jundice","is_autistic","screening_score","PDD_parent","Class/ASD"]
df = pd.read_csv("Ontologia/Autism-Dataset.csv", sep=",", names=['id']+feature, header=0,dtype=str)
df = df[df['Class/ASD'].notna() & (df['Class/ASD'].str.strip().str.lower() != 'class/asd')].reset_index(drop=True)
print('is_autistic unique values:')
print(df['is_autistic'].value_counts(dropna=False).head(20))
print('\nSome sample rows with is_autistic not in [0,1]:')
mask = ~df['is_autistic'].isin(['0','1'])
print(df.loc[mask, ['id','is_autistic','screening_score','age']].head(20).to_string(index=False))
print('\nColumns with prefix is_autistic in dummies:')
dummies = pd.get_dummies(df, columns=feature[:-1])
iso_cols = [c for c in dummies.columns if c.startswith('is_autistic_')]
print(iso_cols[:20])
print('\nCounts for these dummies:')
print(dummies[iso_cols].sum().sort_values(ascending=False).head(20))