from typing import List

from .corpus_types import Word
from .wordnet import WORDNET


def get_synonyms(word: Word) -> List[Word]:
    """get synonyms of the word from WordNet."""
    synonyms = WORDNET.get(word, [])
    return synonyms
