import random
import warnings
from typing import Any, Dict, Tuple, Union

from albumentations.core.transforms_interface import to_tuple
from googletrans.constants import LANGUAGES
from typing_extensions import Literal

from ..core.transforms_interface import MultipleCorpusTypesTransform, SingleCorpusTypeTransform, TextTransform
from ..corpora.types import Language, Text
from . import functional as F


class AEDA(SingleCorpusTypeTransform):
    """Randomly inserts punctuations in the input text.

    Args:
        insertion_prob_limit: The probability of inserting a punctuation.
            If insertion_prob_limit is a float, the range will be (0.0, insertion_prob_limit).
        punctuations: Punctuations to be inserted at random.
        p: The probability of applying this transform.

    References:
        https://arxiv.org/pdf/2108.13230.pdf
    """

    def __init__(
        self,
        insertion_prob_limit: Union[float, Tuple[float, float]] = (0.0, 0.3),
        punctuations: Tuple[str, ...] = (".", ";", "?", ":", "!", ","),
        ignore_first: bool = False,
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super(AEDA, self).__init__(ignore_first, always_apply, p)
        self._validate_transform_init_args(insertion_prob_limit, punctuations)
        self.insertion_prob_limit = to_tuple(insertion_prob_limit, low=0.0)
        self.punctuations = punctuations

    def _validate_transform_init_args(
        self, insertion_prob_limit: Union[float, Tuple[float, float]], punctuations: Tuple[str, ...]
    ) -> None:
        if not isinstance(insertion_prob_limit, (float, int, tuple)):
            raise TypeError(
                "insertion_prob_limit must be a real number between 0 and 1 or a tuple with length 2. "
                f"Got: {type(insertion_prob_limit)}"
            )
        if isinstance(insertion_prob_limit, (float, int)):
            if not (0.0 <= insertion_prob_limit <= 1.0):
                raise ValueError(
                    "If insertion_prob_limit is a real number, "
                    f"it must be between 0 and 1. Got: {insertion_prob_limit}"
                )
        elif isinstance(insertion_prob_limit, tuple):
            if len(insertion_prob_limit) != 2:
                raise ValueError(
                    f"If insertion_prob_limit is a tuple, it's length must be 2. Got: {insertion_prob_limit}"
                )
            if not (0.0 <= insertion_prob_limit[0] <= insertion_prob_limit[1] <= 1.0):
                raise ValueError(f"insertion_prob_limit values must be between 0 and 1. Got: {insertion_prob_limit}")
        if not (isinstance(punctuations, tuple) and all(isinstance(punc, str) for punc in punctuations)):
            raise TypeError(f"punctuations must be a tuple and all elements must be strings. Got: {punctuations}")

    def apply(self, text: Text, insertion_prob: float = 0.3, **params: Any) -> Text:
        return F.insert_punctuations(text, insertion_prob, self.punctuations)

    def get_params(self) -> Dict[str, float]:
        return {"insertion_prob": random.uniform(self.insertion_prob_limit[0], self.insertion_prob_limit[1])}

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("insertion_prob_limit", "punctuations")


class BackTranslation(SingleCorpusTypeTransform):
    """Back-translates the input text by translating it to the target language and then back to the original.

    Args:
        from_lang: The language of the input text.
        to_lang: The language to which the input text will be translated.
        p: The probability of applying this transform.
    """

    def __init__(
        self,
        from_lang: Language = "ko",
        to_lang: Language = "en",
        ignore_first: bool = False,
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super(BackTranslation, self).__init__(ignore_first, always_apply, p)
        self._validate_transform_init_args(from_lang, to_lang)
        self.from_lang = from_lang
        self.to_lang = to_lang

    def _validate_transform_init_args(self, from_lang: Language, to_lang: Language) -> None:
        if from_lang not in LANGUAGES:
            raise ValueError(f"from_lang must be one of {list(LANGUAGES.keys())}. Got: {from_lang}")
        if to_lang not in LANGUAGES:
            raise ValueError(f"to_lang must be one of {list(LANGUAGES.keys())}. Got: {to_lang}")

    def apply(self, text: Text, **params: Any) -> Text:
        return F.back_translate(text, self.from_lang, self.to_lang)

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("from_lang", "to_lang")


class Cut(MultipleCorpusTypesTransform):
    """Cuts a portion of each word or each sentence or the text from the input text.

    Args:
        length: The length to cut each element in the input text.
        begin: Whether to cut each element at start or end.
        unit: Unit to which transform is to be applied.
        p: The probability of applying this transform.
    """

    def __init__(
        self,
        length: int = 512,
        begin: bool = True,
        unit: Literal["word", "sentence", "text"] = "text",
        ignore_first: bool = False,
        always_apply: bool = False,
        p: float = 1.0,
    ) -> None:
        super(Cut, self).__init__(unit, ignore_first, always_apply, p)
        if unit not in ["word", "sentence", "text"]:
            raise ValueError("unit must be one of ['word', 'sentence', 'text']")
        self._validate_transform_init_args(length, begin)
        self.length = length
        self.begin = begin

    def apply_to_words(self, text: Text, **params: Any) -> Text:
        return F.cut_words(text, self.length, self.begin)

    def apply_to_sentences(self, text: Text, **params: Any) -> Text:
        return F.cut_sentences(text, self.length, self.begin)

    def apply_to_text(self, text: Text, **params: Any) -> Text:
        return F.cut_text(text, self.length, self.begin)

    def _validate_transform_init_args(self, length: int, begin: bool) -> None:
        if not isinstance(length, int):
            raise TypeError(f"length must be a positive integer. Got: {length}")
        if length <= 0:
            raise ValueError(f"length must be positive. Got: {length}")
        if not isinstance(begin, bool):
            raise TypeError(f"begin must be boolean. Got: {type(begin)}")

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("length", "begin")


class RandomDeletion(MultipleCorpusTypesTransform):
    """Randomly deletes words or sentences in the input text.

    Args:
        deletion_prob: The probability of deleting an element.
        min_elements:
            If a `float`, it is the minimum proportion of elements to retain in the text.
            If an `int`, it is the minimum number of elements in the text.
        unit: Unit to which transform is to be applied.
        p: The probability of applying this transform.

    References:
        https://arxiv.org/pdf/1901.11196.pdf
    """

    def __init__(
        self,
        deletion_prob: float = 0.1,
        min_elements: Union[float, int] = 0.8,
        unit: Literal["word", "sentence"] = "word",
        ignore_first: bool = False,
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super(RandomDeletion, self).__init__(unit, ignore_first, always_apply, p)
        if unit not in ["word", "sentence"]:
            raise ValueError("unit must be one of ['word', 'sentence']")
        self._validate_transform_init_args(deletion_prob, min_elements)
        self.deletion_prob = deletion_prob
        self.min_elements = min_elements

    def _validate_transform_init_args(self, deletion_prob: float, min_elements: Union[float, int]) -> None:
        if not isinstance(deletion_prob, (float, int)):
            raise TypeError(f"deletion_prob must be a real number between 0 and 1. Got: {type(deletion_prob)}")
        if not (0.0 <= deletion_prob <= 1.0):
            raise ValueError(f"deletion_prob must be between 0 and 1. Got: {deletion_prob}")
        if not isinstance(min_elements, (float, int)):
            raise TypeError(f"min_elements must be either an int or a float. Got: {type(min_elements)}")
        if isinstance(min_elements, float):
            if not (0.0 <= min_elements <= 1.0):
                raise ValueError(f"If min_elements is a float, it must be between 0 and 1. Got: {min_elements}")
        elif isinstance(min_elements, int):
            if min_elements < 0:
                raise ValueError(f"If min_elements is an int, it must be non negative. Got: {min_elements}")

    def apply_to_words(self, text: Text, **params: Any) -> Text:
        return F.delete_words(text, self.deletion_prob, self.min_elements)

    def apply_to_sentences(self, text: Text, **params: Any) -> Text:
        return F.delete_sentences(text, self.deletion_prob, self.min_elements)

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("deletion_prob", "min_elements")


class RandomDeletionSentence(TextTransform):
    """Randomly deletes sentences in the input text.

    Args:
        deletion_prob: The probability of deleting a sentence.
        min_sentences:
            If a `float`, it is the minimum proportion of sentences to retain in the text.
            If an `int`, it is the minimum number of sentences in the text.
        p: The probability of applying this transform.
    """

    def __init__(
        self,
        deletion_prob: float = 0.1,
        min_sentences: Union[float, int] = 0.8,
        ignore_first: bool = False,
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super(RandomDeletionSentence, self).__init__(ignore_first, always_apply, p)
        warnings.warn(
            "This class has been deprecated. Please use RandomDeletion with unit='sentence'", DeprecationWarning
        )
        self._validate_transform_init_args(deletion_prob, min_sentences)
        self.deletion_prob = deletion_prob
        self.min_sentences = min_sentences

    def _validate_transform_init_args(self, deletion_prob: float, min_sentences: Union[float, int]) -> None:
        if not isinstance(deletion_prob, (float, int)):
            raise TypeError(f"deletion_prob must be a real number between 0 and 1. Got: {type(deletion_prob)}")
        if not (0.0 <= deletion_prob <= 1.0):
            raise ValueError(f"deletion_prob must be between 0 and 1. Got: {deletion_prob}")
        if not isinstance(min_sentences, (float, int)):
            raise TypeError(f"min_sentences must be either an int or a float. Got: {type(min_sentences)}")
        if isinstance(min_sentences, float):
            if not (0.0 <= min_sentences <= 1.0):
                raise ValueError(f"If min_sentences is a float, it must be between 0 and 1. Got: {min_sentences}")
        elif isinstance(min_sentences, int):
            if min_sentences < 0:
                raise ValueError(f"If min_sentences is an int, it must be non-negative. Got: {min_sentences}")

    def apply(self, text: Text, **params: Any) -> Text:
        return F.delete_sentences(text, self.deletion_prob, self.min_sentences)

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("deletion_prob", "min_sentences")


class RandomInsertion(SingleCorpusTypeTransform):
    """Randomly inserts synonyms in the input text n times.

    Args:
        insertion_prob: The probability of inserting a synonym.
        n_times: The number of times to repeat the process.
        p: The probability of applying this transform.

    References:
        https://arxiv.org/pdf/1901.11196.pdf
    """

    def __init__(
        self,
        insertion_prob: float = 0.2,
        n_times: int = 1,
        ignore_first: bool = False,
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super(RandomInsertion, self).__init__(ignore_first, always_apply, p)
        self._validate_transform_init_args(insertion_prob, n_times)
        self.insertion_prob = insertion_prob
        self.n_times = n_times

    def _validate_transform_init_args(self, insertion_prob: float, n_times: int) -> None:
        if not isinstance(insertion_prob, (float, int)):
            raise TypeError(f"insertion_prob must be a real number between 0 and 1. Got: {type(insertion_prob)}")
        if not (0.0 <= insertion_prob <= 1.0):
            raise ValueError(f"insertion_prob must be between 0 and 1. Got: {insertion_prob}")
        if not isinstance(n_times, int):
            raise TypeError(f"n_times must be a positive integer. Got: {type(n_times)}")
        if n_times <= 0:
            raise ValueError(f"n_times must be positive. Got: {n_times}")

    def apply(self, text: Text, **params: Any) -> Text:
        return F.insert_synonyms(text, self.insertion_prob, self.n_times)

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("insertion_prob", "n_times")


class RandomSwap(MultipleCorpusTypesTransform):
    """Randomly swaps two words or two sentences in the input text n times.

    Args:
        n_times: The number of times to repeat the process.
        unit: Unit to which transform is to be applied.
        p: The probability of applying this transform.

    References:
        https://arxiv.org/pdf/1901.11196.pdf
    """

    def __init__(
        self,
        n_times: int = 1,
        unit: Literal["word", "sentence"] = "word",
        ignore_first: bool = False,
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super(RandomSwap, self).__init__(unit, ignore_first, always_apply, p)
        if unit not in ["word", "sentence"]:
            raise ValueError("unit must be one of ['word', 'sentence']")
        self._validate_transform_init_args(n_times)
        self.n_times = n_times

    def _validate_transform_init_args(self, n_times: int) -> None:
        if not isinstance(n_times, int):
            raise TypeError(f"n_times must be a positive integer. Got: {type(n_times)}")
        if n_times <= 0:
            raise ValueError(f"n_times must be positive. Got: {n_times}")

    def apply_to_words(self, text: Text, **params: Any) -> Text:
        return F.swap_words(text, self.n_times)

    def apply_to_sentences(self, text: Text, **params: Any) -> Text:
        return F.swap_sentences(text, self.n_times)

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("n_times",)


class RandomSwapSentence(TextTransform):
    """Randomly swaps two sentences in the input text n times.

    Args:
        n_times: The number of times to repeat the process.
        p: The probability of applying this transform.
    """

    def __init__(
        self,
        n_times: int = 1,
        ignore_first: bool = False,
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super(RandomSwapSentence, self).__init__(ignore_first, always_apply, p)
        warnings.warn("This class has been deprecated. Please use RandomSwap with unit='sentence'", DeprecationWarning)
        self._validate_transform_init_args(n_times)
        self.n_times = n_times

    def _validate_transform_init_args(self, n_times: int) -> None:
        if not isinstance(n_times, int):
            raise TypeError(f"n_times must be a positive integer. Got: {type(n_times)}")
        if n_times <= 0:
            raise ValueError(f"n_times must be positive. Got: {n_times}")

    def apply(self, text: Text, **params: Any) -> Text:
        return F.swap_sentences(text, self.n_times)

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("n_times",)


class SynonymReplacement(SingleCorpusTypeTransform):
    """Randomly replaces words in the input text with synonyms.

    Args:
        replacement_prob: The probability of replacing a word with a synonym.
        p: The probability of applying this transform.

    References:
        https://arxiv.org/pdf/1901.11196.pdf
    """

    def __init__(
        self,
        replacement_prob: float = 0.2,
        ignore_first: bool = False,
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super(SynonymReplacement, self).__init__(ignore_first, always_apply, p)
        self._validate_transform_init_args(replacement_prob)
        self.replacement_prob = replacement_prob

    def _validate_transform_init_args(self, replacement_prob: float) -> None:
        if not isinstance(replacement_prob, (float, int)):
            raise TypeError(f"replacement_prob must be a real number between 0 and 1. Got: {type(replacement_prob)}")
        if not (0.0 <= replacement_prob <= 1.0):
            raise ValueError(f"replacement_prob must be between 0 and 1. Got: {replacement_prob}")

    def apply(self, text: Text, **params: Any) -> Text:
        return F.replace_synonyms(text, self.replacement_prob)

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("replacement_prob",)
