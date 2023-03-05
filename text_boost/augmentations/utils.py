import re
from typing import List, Optional, overload

from ..corpora.corpus_types import Word, Sentence, Text

__all__ = [
    "strip",
    "remove_empty_strings",
    "split_sentence",
    "split_text",
    "combine_words",
    "combine_sentences",
    "combine_sentences_with_text",
    "get_first_sentence",
    "get_nth_sentence",
    "remove_first_sentence",
    "remove_nth_sentence",
]


@overload
def strip(strings: List[Word]) -> List[Word]:
    ...


@overload
def strip(strings: List[Sentence]) -> List[Sentence]:
    ...


def strip(strings):
    """Removes leading and trailing whitespaces from each string in the list."""
    return [s.strip() for s in strings]


@overload
def remove_empty_strings(strings: List[Word]) -> List[Word]:
    ...


@overload
def remove_empty_strings(strings: List[Sentence]) -> List[Sentence]:
    ...


def remove_empty_strings(strings):
    """Removes empty strings from the list of strings."""
    return [s for s in strings if s]


def split_sentence(sentence: Sentence) -> List[Word]:
    """Splits the sentence into words."""
    words = sentence.split()
    words = strip(words)
    words = remove_empty_strings(words)
    return words

      
def split_text(text: Text) -> List[Sentence]:
    """Splits the text into sentences."""
    sentences = re.split(r"[.?!]", text)
    sentences = strip(sentences)
    sentences = remove_empty_strings(sentences)
    return sentences 

  
def combine_words(words: List[Word]) -> Sentence:
    """Combines words into a sentence."""
    sentence = " ".join(words)
    return sentence

  
def combine_sentences(sentences: List[Sentence]) -> Text:
    """Combines sentences into a text."""
    text = ". ".join(sentences)
    if text:
        text = ".".join([text, ""])
    return text


def combine_sentences_with_text(
    text: Text,
    *,
    prefix_sentences: Optional[List[Sentence]] = None,
    suffix_sentences: Optional[List[Sentence]] = None
) -> Text:
    """Combines prefix sentences, the input text, and suffix sentences into a single text.

    Args:
        text (Text): The input text.
        prefix_sentences (List[Sentence]): List of sentences to add at the beginning of the text.
        suffix_sentences (List[Sentence]): List of sentences to add at the end of the text.

    Returns:
        full_text (Text): The combined text.
    """
    prefix_text = combine_sentences(prefix_sentences) if prefix_sentences is not None else ""
    suffix_text = combine_sentences(suffix_sentences) if suffix_sentences is not None else ""
    full_text = " ".join([prefix_text, text, suffix_text]).strip()
    return full_text


def get_first_sentence(text: Text) -> Sentence:
    """gets the first sentence from the text"""
    return get_nth_sentence(text, 0)


def get_nth_sentence(text: Text, n: int) -> Text:
    """gets the nth sentence from the text"""
    sentences = split_text(text)
    return sentences[n]


def remove_first_sentence(text: Text) -> Text:
    """Removes the first sentence from the text"""
    return remove_nth_sentence(text, 0)


def remove_nth_sentence(text: Text, n: int) -> Text:
    """Removes the nth sentence from the text"""
    sentences = split_text(text)
    remaining_sentences = [s for idx, s in enumerate(sentences) if idx != n]
    text = combine_sentences(remaining_sentences)
    return text
