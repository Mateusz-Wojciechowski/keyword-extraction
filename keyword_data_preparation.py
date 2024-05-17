import pandas as pd
import ast
import json


def safely_eval_literal(column):
    try:
        return ast.literal_eval(column)
    except ValueError:
        return column


keywords_bert = pd.read_csv('keywords_bert.csv', index_col=None)
keywords_yake = pd.read_csv('keywords_yake.csv', index_col=None)
keywords_textrank = pd.read_csv('keywords_textrank.csv', index_col=None)
keywords_rake = pd.read_csv('keywords_rake.csv', index_col=None)

keywords_df = pd.concat([keywords_bert, keywords_yake, keywords_textrank, keywords_rake], axis=1)

keywords_df['BertKeyWords'] = keywords_df['BertKeyWords'].apply(safely_eval_literal)
keywords_df['YakeKeyWords'] = keywords_df['YakeKeyWords'].apply(safely_eval_literal)
keywords_df['TextRankKeywords'] = keywords_df['TextRankKeywords'].apply(safely_eval_literal)
keywords_df['RakeKeyWords'] = keywords_df['RakeKeyWords'].apply(safely_eval_literal)

keywords_df['BertKeyWords'] = keywords_df['BertKeyWords'].apply(json.dumps)
keywords_df['YakeKeyWords'] = keywords_df['YakeKeyWords'].apply(json.dumps)
keywords_df['TextRankKeywords'] = keywords_df['TextRankKeywords'].apply(json.dumps)
keywords_df['RakeKeyWords'] = keywords_df['RakeKeyWords'].apply(json.dumps)

keywords_df.to_csv('keywords_merged.csv', index=False)

