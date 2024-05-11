import yake


def yake_extraction(text, max_ngram, language, deduplication_threshold, deduplication_algo, window_size, num_keywords):
    custom_kw_extractor = yake.KeywordExtractor(
        lan=language,
        n=max_ngram,
        dedupLim=deduplication_threshold,
        dedupFunc=deduplication_algo,
        windowsSize=window_size,
        top=num_keywords,
        features=None
    )

    keywords = custom_kw_extractor.extract_keywords(text)
    keywords_list = []
    for elem in keywords:
        keyword, _ = elem
        keywords_list.append(keyword)

    return keywords_list

