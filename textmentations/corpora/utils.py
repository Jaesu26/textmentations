import random
from typing import List

from .corpora import STOPWORDS, WORDNET
from .types import Word


def get_synonyms(word: Word) -> List[Word]:
    """Gets synonyms of the word from WordNet."""
    synonyms = WORDNET.get(word, [])
    return synonyms


def get_random_synonym(word: Word) -> Word:
    """Gets a random synonym for the word."""
    synonyms = get_synonyms(word)
    if synonyms:
        synonym = random.choice(synonyms)
        return synonym
    return word


def is_stopword(word: Word) -> bool:
    """Checks whether the word is a stopword."""
    return word in STOPWORDS
