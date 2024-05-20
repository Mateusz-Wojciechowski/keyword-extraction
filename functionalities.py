import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.lower() not in stop_words and word not in string.punctuation]
    return filtered_words



