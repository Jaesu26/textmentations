from __future__ import annotations

import numpy as np

from textmentations.corpora.corpora import get_stopwords, get_wordnet
from textmentations.corpora.types import Word

_STOPWORDS = get_stopwords()
_WORDNET = get_wordnet()


def get_synonyms(word: Word) -> list[Word]:
    """Gets synonyms of the word from WordNet."""
    synonyms = _WORDNET.get(word, [])
    return synonyms


def get_random_synonym(word: Word, rng: np.random.Generator) -> Word:
    """Gets a random synonym for the word."""
    synonyms = get_synonyms(word)
    if synonyms:
        index = rng.integers(0, len(synonyms))
        return synonyms[index]
    return word


def is_stopword(word: Word) -> bool:
    """Checks whether the word is a stopword."""
    return word in _STOPWORDS
