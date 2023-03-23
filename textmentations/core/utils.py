from typing import Dict

from ..augmentations.utils import extract_first_sentence, remove_first_sentence, wrap_text_with_sentences
from ..corpora.corpus_types import Sentence, Text


def extract_first_sentence_by_key(key2text: Dict[str, Text]) -> Dict[str, Sentence]:
    """Extracts the first sentence from a dictionary of text inputs.

    Args:
        key2text: A dictionary with string keys and Text values.

    Returns:
        A dictionary with the same keys as `key2text`, but with the values replaced by their first sentence.
    """
    key2first_sentence = {}
    for key, text in key2text.items():
        if text is not None:
            key2first_sentence[key] = extract_first_sentence(text)
        else:
            key2first_sentence[key] = None
    return key2first_sentence


def remove_first_sentence_by_key(key2text: Dict[str, Text]) -> Dict[str, Text]:
    """Removes the first sentence from the dictionary of text inputs.

    Args:
        key2text: The dictionary with string keys and Text values.

    Returns:
        A dictionary with the same keys as `key2text`,
            but with the values replaced by the input text with the first sentence removed.
    """
    key2text_without_first_sentence = {}
    for key, text in key2text.items():
        if text is not None:
            key2text_without_first_sentence[key] = remove_first_sentence(text)
        else:
            key2text_without_first_sentence[key] = None
    return key2text_without_first_sentence


# TODO: wrap_text_with_first_sentence_by_key 함수의 일반화 버전 구현
def wrap_text_with_first_sentence_by_key(
    key2text_without_first_sentence: Dict[str, Text],
    key2first_sentence: Dict[str, Sentence]
) -> Dict[str, Text]:
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
    key2text = {}
    for key, text_without_first_sentence in key2text_without_first_sentence.items():
        if text_without_first_sentence is not None:
            first_sentence = key2first_sentence[key]
            key2text[key] = wrap_text_with_sentences(text_without_first_sentence, prefix_sentences=[first_sentence])
        else:
            key2text[key] = None
    return key2text
