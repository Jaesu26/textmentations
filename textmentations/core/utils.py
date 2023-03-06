from typing import Dict

from ..augmentations.utils import combine_sentences_with_text, get_first_sentence, remove_first_sentence
from ..corpora.corpus_types import Text


def get_first_sentences_from_kwargs(kwargs: Dict[str, Text]) -> Dict[str, Text]:
    kwargs_first_sentences = {}
    for key, arg in kwargs.items():
        if arg is not None:
            text = kwargs[key]
            kwargs_first_sentences[key] = get_first_sentence(text)
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


def combine_augmented_kwargs_with_first_sentences(
    augmented_kwargs_without_first_sentences: Dict[str, Text],
    kwargs_first_sentences: Dict[str, Text]
) -> Dict[str, Text]:
    augmented_kwargs = {}
    for key, arg in augmented_kwargs_without_first_sentences.items():
        if arg is not None:
            augmented_text_without_first_sentence = augmented_kwargs_without_first_sentences[key]
            first_sentence = kwargs_first_sentences[key]
            augmented_text = combine_sentences_with_text(
                augmented_text_without_first_sentence,
                prefix_sentences=[first_sentence]
            )
            augmented_kwargs[key] = augmented_text
        else:
            augmented_kwargs[key] = None
    return augmented_kwargs
