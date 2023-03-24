from typing import List

from .corpora import STOPWORDS, WORDNET
from .types import Word


def get_synonyms(word: Word) -> List[Word]:
    """Gets synonyms of the word from WordNet."""
    synonyms = WORDNET.get(word, [])
    return synonyms


def is_stopword(word: Word) -> bool:
    """Checks whether the word is a stopword."""
    return word in STOPWORDS
