import pandas as pd
import json

keywords_df = pd.read_csv('keywords/keywords_merged.csv')

keywords_df['BertKeyWords'] = keywords_df['BertKeyWords'].apply(json.loads)
keywords_df['YakeKeyWords'] = keywords_df['YakeKeyWords'].apply(json.loads)
keywords_df['TextRankKeywords'] = keywords_df['TextRankKeywords'].apply(json.loads)
keywords_df['RakeKeyWords'] = keywords_df['RakeKeyWords'].apply(json.loads)

textrank_matches = 0
yake_matches = 0
rake_matches = 0
textrank_yake_matches = 0
rake_yake_matches = 0
rake_textrank_matches = 0


for i in range(len(keywords_df)):
    for word in keywords_df['BertKeyWords'][i]:
        if word in keywords_df['YakeKeyWords'][i]:
            yake_matches += 1
        if word in keywords_df['TextRankKeywords'][i]:
            textrank_matches += 1
        if word in keywords_df['RakeKeyWords'][i]:
            rake_matches += 1

for i in range(len(keywords_df)):
    for word in keywords_df['YakeKeyWords'][i]:
        if word in keywords_df['TextRankKeywords'][i]:
            textrank_yake_matches += 1
        if word in keywords_df['RakeKeyWords'][i]:
            rake_yake_matches += 1

for i in range(len(keywords_df)):
    for word in keywords_df['TextRankKeywords'][i]:
        if word in keywords_df['RakeKeyWords'][i]:
            rake_textrank_matches += 1


total_keywords_amount = 10 * len(keywords_df)
print(total_keywords_amount)
print(f"Textrank-Bert matches: {textrank_matches}")
print(f"Yake-Bert matches: {yake_matches}")
print(f"Rake-Bert matches {rake_matches}")
print(f"Textrank-YAKE matches: {textrank_yake_matches}")
print(f"Rake-Yake matches: {rake_yake_matches}")
print(f"Rake-Textrank matches: {rake_textrank_matches}")


print(f"Textrank matches ratio: {textrank_matches/total_keywords_amount}")
print(f"Yake matches ratio: {yake_matches/total_keywords_amount}")
print(f"Rake matches ration: {rake_matches/total_keywords_amount}")
print(f"Textrank-Yake matches ratio: {textrank_yake_matches/total_keywords_amount}")
print(f"Rake-Yake matches ratio: {rake_yake_matches/total_keywords_amount}")
print(f"Rake-Textrank matches ratio: {rake_textrank_matches/total_keywords_amount}")