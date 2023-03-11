import json
import os

STOPWORDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_corpora/stopwords.json")

with open(STOPWORDS_PATH, encoding="utf-8") as f:
    STOPWORDS = json.load(f)
