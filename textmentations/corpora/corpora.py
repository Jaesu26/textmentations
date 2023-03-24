import json
import os

STOPWORDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpora/stopwords.txt")
with open(STOPWORDS_PATH, encoding="utf-8") as f:
    STOPWORDS = frozenset(f.read().splitlines())

WORDNET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpora/wordnet.json")
with open(WORDNET_PATH, encoding="utf-8") as f:
    WORDNET = json.load(f)
