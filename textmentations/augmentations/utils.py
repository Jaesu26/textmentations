import re
from functools import wraps
from typing import Callable, List, Optional

from typing_extensions import Concatenate, ParamSpec

from ..corpora.corpus_types import Word, Sentence, Text, WS

_P = ParamSpec("_P")


def autopsy_sentence(
    func: Callable[Concatenate[List[Word], _P], List[Word]]
) -> Callable[Concatenate[Sentence, _P], Sentence]:
    """The decorator follows these steps:
        1. Splits the input sentence into words.
        2. Applies the `func` to the words.
        3. Joins the words returned by `func` into a sentence.

    Args:
        func (Callable[Concatenate[List[Word], _P], List[Word]]):
            The function to be decorated.

    Returns:
        Callable[Concatenate[Sentence, _P], Sentence]:
            A wrapper function that performs the steps.

    Examples:
        >>> @autopsy_sentence
        ... def remove_second_word(words):
        ...    try:
        ...        del words[1]
        ...        return words
        ...    except IndexError:
        ...        return words
        ...
        >>> sentence = "짜장면을 맛있게 먹었다"
        >>> remove_second_word(sentence)
        "짜장면을 먹었다"
    """
    @wraps(func)
    def wrapped(sentence: Sentence, *args: _P.args, **kwargs: _P.kwargs) -> Sentence:
        words = split_sentence(sentence)
        processed_words = func(words, *args, **kwargs)
        processed_sentence = join_words(processed_words)
        return processed_sentence
    return wrapped


def autopsy_text(
    func: Callable[Concatenate[List[Sentence], _P], List[Sentence]]
) -> Callable[Concatenate[Text, _P], Text]:
    """The decorator follows these steps:
        1. Splits the input text into sentences.
        2. Applies the `func` to the sentences.
        3. Joins the sentences returned by `func` into a text.

    Args:
        func (Callable[Concatenate[List[Sentence], _P], List[Sentence]]):
            The function to be decorated.

    Returns:
        Callable[Concatenate[Text, _P], Text]:
            A wrapper function that performs the steps.

    Examples:
        >>> @autopsy_text
        ... def remove_second_sentence(sentences):
        ...    try:
        ...        del sentences[1]
        ...        return sentences
        ...    except IndexError:
        ...        return sentences
        ...
        >>> text = "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다."
        >>> remove_second_sentence(text)
        "짜장면을 맛있게 먹었다."
    """
    @wraps(func)
    def wrapped(text: Text, *args: _P.args, **kwargs: _P.kwargs) -> Text:
        sentences = split_text(text)
        processed_sentences = func(sentences, *args, **kwargs)
        processed_text = join_sentences(processed_sentences)
        return processed_text
    return wrapped


def split_sentence(sentence: Sentence) -> List[Word]:
    """Splits the sentence into words."""
    words = sentence.split()
    words = strip(words)
    words = remove_empty_strings(words)
    return words


def split_text(text: Text) -> List[Sentence]:
    """Splits the text into sentences."""
    sentences = re.split(r"[.]", text)
    sentences = strip(sentences)
    sentences = remove_empty_strings(sentences)
    return sentences


def join_words(words: List[Word]) -> Sentence:
    """Joins words into a sentence."""
    sentence = " ".join(words)
    return sentence


def join_sentences(sentences: List[Sentence]) -> Text:
    """Joins sentences into a text."""
    text = ". ".join(sentences)
    if text:
        text = ".".join([text, ""])
    return text


def strip(strings: List[WS]) -> List[WS]:
    """Removes leading and trailing whitespaces from each string in the list."""
    return [s.strip() for s in strings]


def remove_empty_strings(strings: List[WS]) -> List[WS]:
    """Removes empty strings from the list of strings."""
    return [s for s in strings if s]


def extract_first_sentence(text: Text) -> Sentence:
    """Extracts the first sentence from the text."""
    return extract_nth_sentence(text, 0)


def extract_nth_sentence(text: Text, n: int) -> Sentence:
    """Extracts the nth sentence from the text."""
    sentences = split_text(text)
    try:
        nth_sentence = sentences[n]
        return nth_sentence
    except IndexError:
        return ""


def remove_first_sentence(text: Text) -> Text:
    """Removes the first sentence from the text."""
    return remove_nth_sentence(text, 0)


def remove_nth_sentence(text: Text, n: int) -> Text:
    """Removes the nth sentence from the text"""
    sentences = split_text(text)
    try:
        del sentences[n]
        text_without_nth_sentence = join_sentences(sentences)
        return text_without_nth_sentence
    except IndexError:
        return text


def pass_empty_text(func: Callable[Concatenate[Text, _P], Text]) -> Callable[Concatenate[Text, _P], Text]:
    """Returns the input text directly if it is empty, otherwise calls the decorated function.

    Args:
        func(Callable[Concatenate[Text, _P], Text]): The function to be decorated.

    Returns:
        A wrapper function.
    """
    @wraps(func)
    def wrapped(text: Text, *args: _P.args, **kwargs: _P.kwargs) -> Text:
        if not text:
            return text
        return func(text, *args, **kwargs)
    return wrapped


def wrap_text_with_sentences(
    text: Text,
    *,
    prefix_sentences: Optional[List[Sentence]] = None,
    suffix_sentences: Optional[List[Sentence]] = None
) -> Text:
    """Wraps the text with the specified prefix and suffix sentences.

    Args:
        text (Text): The input text to wrap with sentences.
        prefix_sentences (List[Sentence]): List of sentences to add at the beginning of the text.
        suffix_sentences (List[Sentence]): List of sentences to add at the end of the text.

    Returns:
        Text: The wrapped text.
    """
    prefix_text = join_sentences(prefix_sentences) if prefix_sentences else ""
    suffix_text = join_sentences(suffix_sentences) if suffix_sentences else ""
    wrapped_text = " ".join([prefix_text, text, suffix_text]).strip()
    return wrapped_text
