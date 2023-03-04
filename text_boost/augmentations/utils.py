import re
from typing import List, Sequence, TypeVar

from ..corpora.corpus_types import Word, Sentence, Text

__all__ = [
    "strip",
    "remove_empty_strings",
    "split_sentence",
    "split_text",
    "combine_words",
    "combine_sentences",
]

T = TypeVar("T", Word, Sentence)


def strip(strings: Sequence[T]) -> List[T]:
    """Remove leading and trailing whitespaces from each string in the sequence."""
    return [string.strip() for string in strings]


def remove_empty_strings(strings: Sequence[T]) -> List[T]:
    """Remove empty strings from the sequence of strings."""
    return [string for string in strings if string]


def split_sentence(sentence: Sentence) -> List[Word]:
    """Split the sentence into words."""
    words = sentence.split()
    words = strip(words)
    words = remove_empty_strings(words)
    return words

      
def split_text(text: Text) -> List[Sentence]:
    """Split the text into sentences."""
    sentences = re.split(r"[.?!]", text)
    sentences = strip(sentences)
    sentences = remove_empty_strings(sentences)
    return sentences 

  
def combine_words(words: List[Word]) -> Sentence:
    """Combine words into a sentence."""
    sentence = " ".join(words)
    return sentence

  
def combine_sentences(sentences: List[Sentence]) -> Text:
    """Combine sentences into a text."""
    text = ". ".join(sentences)
    if text:
        text = ".".join([text, ""])
    return text
