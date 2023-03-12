from typing import List

from .corpus_types import Word
from .stopwords import STOPWORDS
from .wordnet import WORDNET


def get_stopwords() -> List[Word]:
    """get stopwords."""
    stopwords = STOPWORDS.get("stopwords", [])
    stopwords = set(stopwords)
    return stopwords


def get_synonyms(word: Word) -> List[Word]:
    """get synonyms of the word from WordNet."""
    synonyms = WORDNET.get(word, [])
    return synonyms
