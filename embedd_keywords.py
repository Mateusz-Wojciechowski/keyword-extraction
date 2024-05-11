from transformers import AutoTokenizer, AutoModel
import pandas as pd
import json

keywords_df = pd.read_csv('keywords_merged.csv')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')


def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()


keywords_df['BertKeyWords'] = keywords_df['BertKeyWords'].apply(json.loads)
keywords_df['YakeKeyWords'] = keywords_df['YakeKeyWords'].apply(json.loads)
keywords_df['TextRankKeywords'] = keywords_df['TextRankKeywords'].apply(json.loads)

keywords_df['BertKeyWords_embeddings'] = keywords_df['BertKeyWords'].apply(lambda x: [get_bert_embedding(word) for word in x])
keywords_df['YakeKeyWords_embeddings'] = keywords_df['YakeKeyWords'].apply(lambda x: [get_bert_embedding(word) for word in x])
keywords_df['TextRankKeywords_embeddings'] = keywords_df['TextRankKeywords'].apply(lambda x: [get_bert_embedding(word) for word in x])

print(keywords_df['BertKeyWords_embeddings'])