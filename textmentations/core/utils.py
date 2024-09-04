from __future__ import annotations

from typing import Any

from ..augmentations.utils import extract_first_sentence, remove_first_sentence, wrap_text_with_sentences
from ..corpora.types import Sentence, Text


def extract_first_sentence_by_key(key2text: dict[str, Text]) -> dict[str, Sentence]:
    """Extracts the first sentence from a dictionary of text inputs.

    Args:
        key2text: A dictionary with string keys and Text values.

    Returns:
        A dictionary with the same keys as `key2text`, but with the values replaced by their first sentence.
    """
    return {key: extract_first_sentence(text) for key, text in key2text.items()}


def remove_first_sentence_by_key(key2text: dict[str, Text]) -> dict[str, Text]:
    """Removes the first sentence from the dictionary of text inputs.

    Args:
        key2text: The dictionary with string keys and Text values.

    Returns:
        A dictionary with the same keys as `key2text`,
            but with the values replaced by the input text with the first sentence removed.
    """
    return {key: remove_first_sentence(text) for key, text in key2text.items()}


def wrap_text_with_first_sentence_by_key(
    key2text_without_first_sentence: dict[str, Text], key2first_sentence: dict[str, Sentence]
) -> dict[str, Text]:
    """Wraps text inputs with their corresponding first sentence.

    Args:
        key2text_without_first_sentence: The dictionary with string keys and Text values that
            represent text inputs with their first sentences removed.
        key2first_sentence: The dictionary with string keys and Sentence values that
            represent the first sentences of the original text inputs.

    Returns:
        A dictionary with the same keys as `key2text_without_first_sentence`,
            but with the values wrapped with their corresponding first sentence as a prefix.
    """
    return {
        key: wrap_text_with_sentences(text_without_first_sentence, prefix_sentences=[key2first_sentence[key]])
        for key, text_without_first_sentence in key2text_without_first_sentence.items()
    }


def get_shortest_class_fullname(cls: type[Any]) -> str:
    """Returns the shortened full name of the given class."""
    class_fullname = f"{cls.__module__}.{cls.__name__}"
    return shorten_class_name(class_fullname)


def shorten_class_name(class_fullname: str) -> str:
    """Shortens the given class full name."""
    split = class_fullname.split(".")
    if len(split) == 1:
        return class_fullname
    top_module, *_, class_name = split
    if top_module == "textmentations":
        return class_name
    return class_fullname
