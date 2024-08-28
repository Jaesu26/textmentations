import json
import os
from typing import Dict, FrozenSet, List

from .types import Word


def get_stopwords() -> FrozenSet[Word]:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpora/stopwords.txt")
    with open(path, encoding="utf-8") as f:
        stopwords = frozenset(f.read().splitlines())
    return stopwords


def get_wordnet() -> Dict[Word, List[Word]]:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpora/wordnet.json")
    with open(path, encoding="utf-8") as f:
        wordnet = json.load(f)
    return wordnet


STOPWORDS = get_stopwords()
WORDNET = get_wordnet()
