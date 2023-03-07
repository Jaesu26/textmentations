from typing import Dict

from ..augmentations.utils import extract_first_sentence, remove_first_sentence, wrap_text_with_sentences
from ..corpora.corpus_types import Text


def extract_first_sentences_from_kwargs(kwargs: Dict[str, Text]) -> Dict[str, Text]:
    kwargs_first_sentences = {}
    for key, arg in kwargs.items():
        if arg is not None:
            text = kwargs[key]
            kwargs_first_sentences[key] = extract_first_sentence(text)
        else:
            kwargs_first_sentences[key] = None
    return kwargs_first_sentences


def remove_first_sentences_from_kwargs(kwargs: Dict[str, Text]) -> Dict[str, Text]:
    kwargs_without_first_sentences = {}
    for key, arg in kwargs.items():
        if arg is not None:
            text = kwargs[key]
            kwargs_without_first_sentences[key] = remove_first_sentence(text)
        else:
            kwargs_without_first_sentences[key] = None
    return kwargs_without_first_sentences


def wrap_augmented_kwargs_with_first_sentences(
    augmented_kwargs_without_first_sentences: Dict[str, Text],
    kwargs_first_sentences: Dict[str, Text]
) -> Dict[str, Text]:
    augmented_kwargs = {}
    for key, arg in augmented_kwargs_without_first_sentences.items():
        if arg is not None:
            augmented_text_without_first_sentence = augmented_kwargs_without_first_sentences[key]
            first_sentence = kwargs_first_sentences[key]
            augmented_text = wrap_text_with_sentences(
                augmented_text_without_first_sentence,
                prefix_sentences=[first_sentence]
            )
            augmented_kwargs[key] = augmented_text
        else:
            augmented_kwargs[key] = None
    return augmented_kwargs
