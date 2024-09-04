from __future__ import annotations

import json
import os

from .types import Word


def get_stopwords() -> frozenset[Word]:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpora/stopwords.txt")
    with open(path, encoding="utf-8") as f:
        stopwords = frozenset(f.read().splitlines())
    return stopwords


def get_wordnet() -> dict[Word, list[Word]]:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpora/wordnet.json")
    with open(path, encoding="utf-8") as f:
        wordnet = json.load(f)
    return wordnet
