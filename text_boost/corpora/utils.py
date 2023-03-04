import json
import os
from typing import List

from .corpus_types import Word

WORDNET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "wordnet.json"
)

with open(WORDNET_PATH, encoding="utf-8") as f:
    WORDNET = json.load(f)


def get_synonyms(word: Word) -> List[Word]:
    """get synonyms of the word from WordNet."""
    synonyms = WORDNET.get(word, [])
    return synonyms
