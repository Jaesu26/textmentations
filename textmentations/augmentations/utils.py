import re
from functools import wraps
from typing import Callable, List, Optional, overload

from typing_extensions import Concatenate, ParamSpec

from ..corpora.corpus_types import Word, Sentence, Text

__all__ = [
    "autopsy_sentence",
    "autopsy_text",
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

P = ParamSpec("P")


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


def autopsy_sentence(
    func: Callable[Concatenate[List[Word], P], List[Word]]
) -> Callable[Concatenate[Sentence, P], Sentence]:
    """The decorator follows this procedure:
        1. Splits the input sentence into words.
        2. Applies the `func` to the words.
        3. Joins the words returned by 'func' into a sentence.

    Args:
        func (Callable[Concatenate[List[Word], P], List[Word]]):
            The function to be decorated. It should take a list of words as its first argument.

    Returns:
        Callable[Concatenate[Sentence, P], Sentence]:
            A decorated function that performs the procedure.

    Example:
        >>> from typing import List, TypeVar
        ...
        >>> Word = TypeVar("Word", bound=str)
        ...
        ...
        >>> @autopsy_sentence
        ... def remove_second_word(words: List[Word]) -> List[Word]:
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
    def wrapped(sentence: Sentence, *args: P.args, **kwargs: P.kwargs) -> Sentence:
        words = split_sentence(sentence)
        words = func(words, *args, **kwargs)
        sentence = join_words(words)
        return sentence
    return wrapped


def autopsy_text(
    func: Callable[Concatenate[List[Sentence], P], List[Sentence]]
) -> Callable[Concatenate[Text, P], Text]:
    """The decorator follows this procedure:
        1. Splits the input text into sentences.
        2. Applies the `func` to the sentences.
        3. Joins the sentences returned by 'func' into a text.

    Args:
        func (Callable[Concatenate[List[Sentence], P], List[Sentence]]):
            The function to be decorated. It should take a list of sentences as its first argument.

    Returns:
        Callable[Concatenate[Text, P], Text]:
            A decorated function that performs the procedure.

    Example:
        >>> from typing import List, TypeVar
        ...
        >>> Sentence = TypeVar("Sentence", bound=str)
        ...
        ...
        >>> @autopsy_text
        ... def remove_second_sentence(sentences: List[Sentence]) -> List[Sentence]:
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
    def wrapped(text: Text, *args: P.args, **kwargs: P.kwargs) -> Text:
        sentences = split_text(text)
        sentences = func(sentences, *args, **kwargs)
        text = join_sentences(sentences)
        return text
    return wrapped


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
    """Joins words into a sentence."""
    sentence = " ".join(words)
    return sentence


def join_sentences(sentences: List[Sentence]) -> Text:
    """Joins sentences into a text."""
    text = ". ".join(sentences)
    if text:
        text = ".".join([text, ""])
    return text


def extract_first_sentence(text: Text) -> Sentence:
    """Extracts the first sentence from the text."""
    return extract_nth_sentence(text, 0)


def extract_nth_sentence(text: Text, n: int) -> Sentence:
    """Extracts the nth sentence from the text."""
    sentences = split_text(text)
    try:
        return sentences[n]
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
        text = join_sentences(sentences)
        return text
    except IndexError:
        return text


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
