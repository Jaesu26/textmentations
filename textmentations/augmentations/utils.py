import re
from typing import List, Optional, overload

from ..corpora.corpus_types import Word, Sentence, Text

__all__ = [
    "strip",
    "remove_empty_strings",
    "split_sentence",
    "split_text",
    "join_words",
    "join_sentences",
    "extract_first_sentence",
    "extract_nth_sentence",
    "remove_first_sentence",
    "remove_nth_sentence",
    "wrap_text_with_sentences",
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


def join_words(words: List[Word]) -> Sentence:
    """joins words into a sentence."""
    sentence = " ".join(words)
    return sentence


def join_sentences(sentences: List[Sentence]) -> Text:
    """joins sentences into a text."""
    text = ". ".join(sentences)
    if text:
        text = ".".join([text, ""])
    return text


def extract_first_sentence(text: Text) -> Sentence:
    """extracts the first sentence from the text"""
    return extract_nth_sentence(text, 0)


def extract_nth_sentence(text: Text, n: int) -> Sentence:
    """extracts the nth sentence from the text"""
    sentences = split_text(text)
    try:
        return sentences[n]
    except IndexError:
        return ""


def remove_first_sentence(text: Text) -> Text:
    """Removes the first sentence from the text"""
    return remove_nth_sentence(text, 0)


def remove_nth_sentence(text: Text, n: int) -> Text:
    """Removes the nth sentence from the text"""
    sentences = split_text(text)
    try:
        del sentences[n]
    except IndexError:
        return text

    text = join_sentences(sentences)
    return text


def wrap_text_with_sentences(
    text: Text,
    *,
    prefix_sentences: Optional[List[Sentence]] = None,
    suffix_sentences: Optional[List[Sentence]] = None
) -> Text:
    """Wraps the input text with the specified prefix and suffix sentences.

    Args:
        text (Text): The text to wrap with sentences.
        prefix_sentences (List[Sentence]): List of sentences to add at the beginning of the text.
        suffix_sentences (List[Sentence]): List of sentences to add at the end of the text.

    Returns:
        wrapped_text (Text): The wrapped text.
    """
    prefix_text = join_sentences(prefix_sentences) if prefix_sentences is not None else ""
    suffix_text = join_sentences(suffix_sentences) if suffix_sentences is not None else ""
    wrapped_text = " ".join([prefix_text, text, suffix_text]).strip()
    return wrapped_text
