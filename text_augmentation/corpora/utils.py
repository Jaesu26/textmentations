from typing import List
from ..corpus_types import Word

import json

WORDNET_PATH = "wordnet.json"

with open(WORDNET_PATH, encoding="utf-8") as f:
    WORDNET = json.load(f)


def get_synonyms(word: Word) -> List[Word]:
    """get synonyms of the word from WordNet."""
    synonyms = WORDNET.get(word, [])
    return synonyms