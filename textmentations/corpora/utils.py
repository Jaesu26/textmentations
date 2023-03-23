from typing import List

from .corpora import STOPWORDS
from .corpus_types import Word
from .raw_corpora import RAW_WORDNET


# TODO: corpora 폴더 구조 최적화
def get_synonyms(word: Word) -> List[Word]:
    """Gets synonyms of the word from WordNet."""
    synonyms = RAW_WORDNET.get(word, [])
    return synonyms


def is_stopword(word: Word) -> bool:
    """Checks whether the word is a stopword."""
    return word in STOPWORDS
