from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import torch


def keybert_extraction(text, n_gram_range, stopwords_lang, num_keywords, diversity):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    st_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    kw_model = KeyBERT(model=st_model)

    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=n_gram_range,
        stop_words=stopwords_lang,
        top_n=num_keywords,
        use_mmr=True,
        diversity=diversity
    )
    keywords_list = []
    for elem in keywords:
        keyword, _ = elem
        keywords_list.append(keyword)

    return keywords_list


