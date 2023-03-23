import json
import os

RAW_STOPWORDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_corpora/stopwords.json")
with open(RAW_STOPWORDS_PATH, encoding="utf-8") as f:
    RAW_STOPWORDS = json.load(f)

RAW_WORDNET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_corpora/wordnet.json")
with open(RAW_WORDNET_PATH, encoding="utf-8") as f:
    RAW_WORDNET = json.load(f)
