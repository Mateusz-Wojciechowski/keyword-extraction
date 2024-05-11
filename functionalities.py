import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))


def create_phrase_list(words, stop_words, punctuation):
    phrases_list = []
    phrase = []
    for word in words:
        if word.lower() in stop_words or word in punctuation:
            if phrase:
                phrases_list.append(' '.join(phrase))
                phrase = []
        else:
            phrase.append(word)

    if phrase:
        phrases_list.append(' '.join(phrase))
    return phrases_list


def preprocess_text(text):
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.lower() not in stop_words and word not in string.punctuation]
    return filtered_words



