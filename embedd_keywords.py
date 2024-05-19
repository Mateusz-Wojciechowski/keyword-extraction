from transformers import AutoTokenizer, AutoModel
import pandas as pd
import json
import torch
import pickle

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available. Device name: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA is not available, using CPU instead.")

keywords_df = pd.read_csv('keywords_data/keywords_merged.csv')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

model = model.to(device)


def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().detach().numpy()


keywords_df['BertKeyWords'] = keywords_df['BertKeyWords'].apply(json.loads)
keywords_df['YakeKeyWords'] = keywords_df['YakeKeyWords'].apply(json.loads)
keywords_df['TextRankKeywords'] = keywords_df['TextRankKeywords'].apply(json.loads)
keywords_df['RakeKeyWords'] = keywords_df['RakeKeyWords'].apply(json.loads)

bert_embeddings = []
yake_embeddings = []
textrank_embeddings = []
rake_embeddings = []

for i in range(len(keywords_df)):
    bert, yake, textrank, rake = [], [], [], []
    for word in keywords_df['BertKeyWords'][i]:
        bert.append(get_bert_embedding(word))
    for word in keywords_df['YakeKeyWords'][i]:
        yake.append(get_bert_embedding(word))
    for word in keywords_df['TextRankKeywords'][i]:
        textrank.append(get_bert_embedding(word))
    for word in keywords_df['RakeKeyWords'][i]:
        rake.append(get_bert_embedding(word))

    bert_embeddings.append(bert)
    yake_embeddings.append(yake)
    textrank_embeddings.append(textrank)
    rake_embeddings.append(rake)


with open('embeddings/bert_embeddings.pkl', 'wb') as file:
    pickle.dump(bert_embeddings, file)

with open('embeddings/yake_embeddings.pkl', 'wb') as file:
    pickle.dump(yake_embeddings, file)

with open('embeddings/textrank_embeddings.pkl', 'wb') as file:
    pickle.dump(textrank_embeddings, file)

with open('embeddings/rake_embeddings.pkl', 'wb') as file:
    pickle.dump(textrank_embeddings, file)