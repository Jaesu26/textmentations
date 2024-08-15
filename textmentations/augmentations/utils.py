import re
from functools import wraps
from typing import Callable, List, Optional

from googletrans import Translator
from typing_extensions import Concatenate, ParamSpec

from ..corpora.types import Corpus, Sentence, Text, Word

TRANSLATOR = Translator()
_P = ParamSpec("_P")


def autopsy_sentence(
    func: Callable[Concatenate[List[Word], _P], List[Word]]
) -> Callable[Concatenate[Sentence, _P], Sentence]:
    """A decorator follows these steps:
        1. Splits the input sentence into words.
        2. Applies the `func` to the words.
        3. Joins the words returned by `func` into a sentence.

    Args:
        func: The function to be decorated.

    Returns:
        A wrapper function that performs the steps.

    Examples:
        >>> @autopsy_sentence
        ... def remove_second_word(words: List[Word]) -> Word:
        ...    try:
        ...        del words[1]
        ...        return words
        ...    except IndexError:
        ...        return words
        ...
        ...
        >>> sentence = "짜장면을 맛있게 먹었다"
        >>> remove_second_word(sentence)
        "짜장면을 먹었다"
    """

    @wraps(func)
    def wrapped(sentence: Sentence, *args: _P.args, **kwargs: _P.kwargs) -> Sentence:
        words = split_sentence_into_words(sentence)
        processed_words = func(words, *args, **kwargs)
        processed_sentence = join_words_into_sentence(processed_words)
        return processed_sentence

    return wrapped


def autopsy_text(
    func: Callable[Concatenate[List[Sentence], _P], List[Sentence]]
) -> Callable[Concatenate[Text, _P], Text]:
    """A decorator follows these steps:
        1. Splits the input text into sentences.
        2. Applies the `func` to the sentences.
        3. Joins the sentences returned by `func` into a text.

    Args:
        func: The function to be decorated.

    Returns:
        A wrapper function that performs the steps.

    Examples:
        >>> @autopsy_text
        ... def remove_second_sentence(sentences: List[Sentence]) -> Sentence:
        ...    try:
        ...        del sentences[1]
        ...        return sentences
        ...    except IndexError:
        ...        return sentences
        ...
        ...
        >>> text = "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다."
        >>> remove_second_sentence(text)
        "짜장면을 맛있게 먹었다."
    """

    @wraps(func)
    def wrapped(text: Text, *args: _P.args, **kwargs: _P.kwargs) -> Text:
        sentences = split_text_into_sentences(text)
        processed_sentences = func(sentences, *args, **kwargs)
        processed_text = join_sentences_into_text(processed_sentences)
        return processed_text

    return wrapped


def split_sentence_into_words(sentence: Sentence) -> List[Word]:
    """Splits the sentence into words."""
    words = sentence.split()
    words = strip(words)
    words = remove_empty_strings(words)
    return words


def split_text_into_sentences(text: Text) -> List[Sentence]:
    """Splits the text into sentences."""
    sentences = re.split(r"[.]", text)
    sentences = strip(sentences)
    sentences = remove_empty_strings(sentences)
    return sentences


# TODO: := 연산자를 사용하여 strip과 remove_empty_strings 함수를 strip_v2 함수로 대체 (Python >= 3.8)
def strip(strings: List[Corpus]) -> List[Corpus]:
    """Removes leading and trailing whitespaces from each string in the list."""
    return [s.strip() for s in strings]


def remove_empty_strings(strings: List[Corpus]) -> List[Corpus]:
    """Removes empty strings from the list of strings."""
    return [s for s in strings if s]


def join_words_into_sentence(words: List[Word]) -> Sentence:
    """Joins words into a sentence."""
    sentence = " ".join(words)
    return sentence


def join_sentences_into_text(sentences: List[Sentence]) -> Text:
    """Joins sentences into a text."""
    text = ". ".join(sentences)
    text = text + "." if text else text
    return text


def extract_first_sentence(text: Text) -> Sentence:
    """Extracts the first sentence from the text."""
    return extract_nth_sentence(text, 0)


def extract_nth_sentence(text: Text, n: int) -> Sentence:
    """Extracts the nth sentence from the text."""
    sentences = split_text_into_sentences(text)
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
    sentences = split_text_into_sentences(text)
    try:
        del sentences[n]
        text_without_nth_sentence = join_sentences_into_text(sentences)
        return text_without_nth_sentence
    except IndexError:
        return text


def wrap_text_with_sentences(
    text: Text, *, prefix_sentences: Optional[List[Sentence]] = None, suffix_sentences: Optional[List[Sentence]] = None
) -> Text:
    """Wraps the text with the specified prefix and suffix sentences.

    Args:
        text: The input text to wrap with sentences.
        prefix_sentences: List of sentences to add at the beginning of the text.
        suffix_sentences: List of sentences to add at the end of the text.

    Returns:
        A wrapped text.
    """
    prefix_text = join_sentences_into_text(prefix_sentences) if prefix_sentences else ""
    suffix_text = join_sentences_into_text(suffix_sentences) if suffix_sentences else ""
    wrapped_text = " ".join([prefix_text, text, suffix_text]).strip()
    return wrapped_text


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


def get_translator() -> Translator:
    """Returns an instance of Translator class"""
    return TRANSLATOR
