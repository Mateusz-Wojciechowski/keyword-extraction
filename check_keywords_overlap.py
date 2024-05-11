import pandas as pd
import json

keywords_df = pd.read_csv('keywords_merged.csv')

keywords_df['BertKeyWords'] = keywords_df['BertKeyWords'].apply(json.loads)
keywords_df['YakeKeyWords'] = keywords_df['YakeKeyWords'].apply(json.loads)
keywords_df['TextRankKeywords'] = keywords_df['TextRankKeywords'].apply(json.loads)


textrank_matches = 0
yake_matches = 0

for i in range(len(keywords_df)):
    for word in keywords_df['BertKeyWords'][i]:
        if word in keywords_df['YakeKeyWords'][i]:
            yake_matches += 1
        if word in keywords_df['TextRankKeywords'][i]:
            textrank_matches += 1


total_keywords_amount = 10 * len(keywords_df)
print(total_keywords_amount)
print(f"Textrank matches: {textrank_matches}")
print(f"Yake matches: {yake_matches}")