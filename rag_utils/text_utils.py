from typing import List

import nltk
from langdetect import LangDetectException, detect
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

stemmer = PorterStemmer()


def detect_language(sentence: str) -> str:
    try:
        language = detect(sentence)
    except LangDetectException:
        language = None
    return language


def preprocess_text(sentence: str) -> List[str]:
    language = detect_language(sentence)
    if language == "en":
        stop_words = set(stopwords.words("english"))
    elif language == "fr":
        stop_words = set(stopwords.words("french"))
    else:
        stop_words = set(stopwords.words("english")) | set(stopwords.words("french"))
    return [
        stemmer.stem(word.lower())
        for word in word_tokenize(sentence)
        if word.isalnum() and word.lower() not in stop_words
    ]
