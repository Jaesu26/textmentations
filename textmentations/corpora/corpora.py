from __future__ import annotations

import json
import os

from textmentations.corpora.types import Word

_STOPWORDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpora/stopwords.txt")
_WORDNET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpora/wordnet.json")


def read_stopwords(path: str | os.PathLike) -> frozenset[Word]:
    with open(path, encoding="utf-8") as f:
        stopwords = frozenset(f.read().splitlines())
    return stopwords


def read_wordnet(path: str | os.PathLike) -> dict[Word, list[Word]]:
    with open(path, encoding="utf-8") as f:
        wordnet = json.load(f)
    return wordnet


_STOPWORDS = read_stopwords(_STOPWORDS_PATH)
_WORDNET = read_wordnet(_WORDNET_PATH)


def get_stopwords() -> frozenset[Word]:
    return _STOPWORDS


def get_wordnet() -> dict[Word, list[Word]]:
    return _WORDNET
