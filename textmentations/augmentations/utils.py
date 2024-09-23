from __future__ import annotations

import os
import re
from collections.abc import Callable
from functools import wraps
from typing import TypeVar

from deep_translator import GoogleTranslator
from transformers import AlbertForMaskedLM, BertTokenizerFast
from typing_extensions import Concatenate, ParamSpec

from textmentations.corpora.types import Corpus, Sentence, Text, Word

EMPTY_STRING = ""
SPACE = " "
_translator = GoogleTranslator(source="ko", target="en")
_P = ParamSpec("_P")
_T = TypeVar("_T")


def autopsy_sentence(
    func: Callable[Concatenate[list[Word], _P], list[Word]]
) -> Callable[Concatenate[Sentence, _P], Sentence]:
    """The decorator follows these steps:
        1. Splits the input sentence into words.
        2. Applies the `func` to the words.
        3. Joins the words returned by `func` into a sentence.

    Args:
        func: The function to be decorated.

    Returns:
        A wrapper function that performs the steps.

    Examples:
        >>> @autopsy_sentence
        ... def remove_second_word(words: list[Word]) -> list[Word]:
        ...    try:
        ...        del words[1]
        ...        return words
        ...    except IndexError:
        ...        return words
        ...
        >>> sentence = "짜장면을 맛있게 먹었다"
        >>> remove_second_word(sentence)
        '짜장면을 먹었다'
    """

    @wraps(func)
    def wrapped(sentence: Sentence, *args: _P.args, **kwargs: _P.kwargs) -> Sentence:
        words = split_sentence_into_words(sentence)
        augmented_words = func(words, *args, **kwargs)
        augmented_sentence = join_words_into_sentence(augmented_words)
        return augmented_sentence

    return wrapped


def autopsy_text(
    func: Callable[Concatenate[list[Sentence], _P], list[Sentence]]
) -> Callable[Concatenate[Text, _P], Text]:
    """The decorator follows these steps:
        1. Splits the input text into sentences.
        2. Applies the `func` to the sentences.
        3. Joins the sentences returned by `func` into a text.

    Args:
        func: The function to be decorated.

    Returns:
        A wrapper function that performs the steps.

    Examples:
        >>> @autopsy_text
        ... def remove_second_sentence(sentences: list[Sentence]) -> list[Sentence]:
        ...    try:
        ...        del sentences[1]
        ...        return sentences
        ...    except IndexError:
        ...        return sentences
        ...
        >>> text = "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다."
        >>> remove_second_sentence(text)
        '짜장면을 맛있게 먹었다.'
    """

    @wraps(func)
    def wrapped(text: Text, *args: _P.args, **kwargs: _P.kwargs) -> Text:
        sentences = split_text_into_sentences(text)
        augmented_sentences = func(sentences, *args, **kwargs)
        augmented_text = join_sentences_into_text(augmented_sentences)
        return augmented_text

    return wrapped


def split_sentence_into_words(sentence: Sentence) -> list[Word]:
    """Splits the sentence into words."""
    words = sentence.split()
    words = strip_v2(words)
    return words


def split_text_into_sentences(text: Text) -> list[Sentence]:
    """Splits the text into sentences."""
    sentences = re.split(r"[.]", text)
    sentences = strip_v2(sentences)
    return sentences


def strip_v2(strings: list[Corpus]) -> list[Corpus]:
    """Removes leading and trailing whitespace from each string and filters out any strings that are empty."""
    return [stripped for s in strings if (stripped := s.strip())]


def strip(strings: list[Corpus]) -> list[Corpus]:
    """Removes leading and trailing whitespaces from each string in the list."""
    return [s.strip() for s in strings]


def remove_empty_strings(strings: list[Corpus]) -> list[Corpus]:
    """Removes empty strings from the list of strings."""
    return [s for s in strings if s]


def join_words_into_sentence(words: list[Word]) -> Sentence:
    """Joins words into a sentence."""
    sentence = SPACE.join(words)
    return sentence


def join_sentences_into_text(sentences: list[Sentence]) -> Text:
    """Joins sentences into a text."""
    text = ". ".join(sentences)
    if text:
        text += "."
    return text


def extract_first_sentence(text: Text) -> Sentence:
    """Extracts the first sentence from the text."""
    return extract_nth_sentence(text, 0)


def extract_nth_sentence(text: Text, n: int) -> Sentence:
    """Extracts the nth sentence from the text."""
    sentences = split_text_into_sentences(text)
    try:
        return sentences[n]
    except IndexError:
        return EMPTY_STRING


def remove_first_sentence(text: Text) -> Text:
    """Removes the first sentence from the text."""
    return remove_nth_sentence(text, 0)


def remove_nth_sentence(text: Text, n: int) -> Text:
    """Removes the nth sentence from the text."""
    sentences = split_text_into_sentences(text)
    try:
        del sentences[n]
        return join_sentences_into_text(sentences)
    except IndexError:
        return text


def wrap_text_with_sentences(
    text: Text, *, prefix_sentences: list[Sentence] | None = None, suffix_sentences: list[Sentence] | None = None
) -> Text:
    """Wraps the text with the specified prefix and suffix sentences.

    Args:
        text: The input text to wrap with sentences.
        prefix_sentences: The list of sentences to add at the beginning of the text.
        suffix_sentences: The list of sentences to add at the end of the text.

    Returns:
        A wrapped text.
    """
    prefix_text = join_sentences_into_text(prefix_sentences) if prefix_sentences else EMPTY_STRING
    suffix_text = join_sentences_into_text(suffix_sentences) if suffix_sentences else EMPTY_STRING
    wrapped_text = SPACE.join([prefix_text, text, suffix_text]).strip()
    return wrapped_text


def with_cleared_double_hash_tokens(func: Callable[_P, Text]) -> Callable[_P, Text]:
    """Decorator that clears all double-hash ("##") subword tokens from the text returned by the decorated function.

    Args:
        func: The function to be decorated.

    Returns:
        A wrapper function.

    Examples:
        >>> @with_cleared_double_hash_tokens
        ... def create_example() -> Text:
        ...     return "나는 짬뽕 ##을 먹었다."
        ...
        >>> create_example()
        '나는 짬뽕을 먹었다.'
    """

    @wraps(func)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> Text:
        text = func(*args, **kwargs)
        cleared_text = clear_double_hash_tokens(text)
        return cleared_text

    return wrapped


def clear_double_hash_tokens(text: Text) -> Text:
    """Clears all double-hash ("##") subword tokens from the text."""
    double_hash_token_pattern = r"\s*##\b"
    cleared_text = re.sub(double_hash_token_pattern, EMPTY_STRING, text)
    return cleared_text


def pass_empty_text(func: Callable[Concatenate[Text, _P], Text]) -> Callable[Concatenate[Text, _P], Text]:
    """Returns the text directly if it is empty, otherwise calls the decorated function.

    Args:
        func: The function to be decorated.

    Returns:
        A wrapper function.
    """

    @wraps(func)
    def wrapped(text: Text, *args: _P.args, **kwargs: _P.kwargs) -> Text:
        if not text:
            return text
        return func(text, *args, **kwargs)

    return wrapped


def _squeeze_first(list_: list[_T]) -> _T:
    """Returns the first and only element in the list containing exactly one element.

    Args:
        list_: The list that is expected to contain exactly one element.

    Returns:
        The single element in the list.

    Raises:
        ValueError: If the list does not contain exactly one element.
    """
    if len(list_) != 1:
        raise ValueError("Input list must contain exactly one element.")
    return list_[0]


def get_translator() -> GoogleTranslator:
    """Returns an instance of GoogleTranslator class."""
    return _translator


def _get_albert_mlm(model_path: str | os.PathLike) -> AlbertForMaskedLM:
    """Gets an ALBERT model for masked language modeling."""
    model = AlbertForMaskedLM.from_pretrained(pretrained_model_name_or_path=model_path)
    return model


def _get_bert_tokenizer_fast(model_path: str | os.PathLike) -> BertTokenizerFast:
    """Gets a fast BERT tokenizer."""
    tokenizer = BertTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_path,
        clean_up_tokenization_spaces=True,
    )
    return tokenizer
