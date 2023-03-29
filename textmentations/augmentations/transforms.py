import random
from typing import Any, Dict, Tuple, Union

from albumentations.core.transforms_interface import to_tuple
from googletrans.constants import LANGUAGES

from ..core.transforms_interface import TextTransform
from ..corpora.types import Language, Text
from . import functional as F


class AEDA(TextTransform):
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


class BackTranslation(TextTransform):
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
            raise ValueError(f"from_lang must be one of ({list(LANGUAGES.keys())}). Got: {from_lang}")
        if to_lang not in LANGUAGES:
            raise ValueError(f"to_lang must be one of ({list(LANGUAGES.keys())}). Got: {to_lang}")

    def apply(self, text: Text, **params: Any) -> Text:
        return F.back_translate(text, self.from_lang, self.to_lang)

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("from_lang", "to_lang")


# TODO: cut 클래스에 docstring 추가
class CutTransform(TextTransform):
    def apply(self, text: Text, **params: Any) -> Text:
        raise NotImplementedError

    def _validate_transform_init_args(self, cutting_length: int, start_from_beginning: bool) -> None:
        if not isinstance(cutting_length, int):
            raise TypeError(f"cutting_length must be a positive integer. Got: {cutting_length}")
        if cutting_length <= 0:
            raise ValueError(f"cutting_length must be positive. Got: {cutting_length}")
        if not isinstance(start_from_beginning, bool):
            raise TypeError(f"start_from_beginning must be boolean. Got: {type(start_from_beginning)}")

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("cutting_length", "start_from_beginning")


class CutWord(CutTransform):
    def __init__(
        self,
        cutting_length: int = 8,
        start_from_beginning: bool = True,
        ignore_first: bool = False,
        always_apply: bool = False,
        p: float = 1.0,
    ) -> None:
        super(CutTransform, self).__init__(ignore_first, always_apply, p)
        self._validate_transform_init_args(cutting_length, start_from_beginning)
        self.cutting_length = cutting_length
        self.start_from_beginning = start_from_beginning

    def apply(self, text: Text, **params: Any) -> Text:
        return F.cut_words(text, self.cutting_length, self.start_from_beginning)


class CutSentence(CutTransform):
    def __init__(
        self,
        cutting_length: int = 128,
        start_from_beginning: bool = True,
        ignore_first: bool = False,
        always_apply: bool = False,
        p: float = 1.0,
    ) -> None:
        super(CutSentence, self).__init__(ignore_first, always_apply, p)
        self._validate_transform_init_args(cutting_length, start_from_beginning)
        self.cutting_length = cutting_length
        self.start_from_beginning = start_from_beginning

    def apply(self, text: Text, **params: Any) -> Text:
        return F.cut_sentences(text, self.cutting_length, self.start_from_beginning)


class CutText(CutTransform):
    def __init__(
        self,
        cutting_length: int = 512,
        start_from_beginning: bool = True,
        ignore_first: bool = False,
        always_apply: bool = False,
        p: float = 1.0,
    ) -> None:
        super(CutText, self).__init__(ignore_first, always_apply, p)
        self._validate_transform_init_args(cutting_length, start_from_beginning)
        self.cutting_length = cutting_length
        self.start_from_beginning = start_from_beginning

    def apply(self, text: Text, **params: Any) -> Text:
        return F.cut_text(text, self.cutting_length, self.start_from_beginning)


class RandomDeletion(TextTransform):
    """Randomly deletes words in the input text.

    Args:
        deletion_prob: The probability of deleting a word.
        min_words_each_sentence:
            If a `float`, it is the minimum proportion of words to retain in each sentence.
            If an `int`, it is the minimum number of words in each sentence.
        p: The probability of applying this transform.

    References:
        https://arxiv.org/pdf/1901.11196.pdf
    """

    def __init__(
        self,
        deletion_prob: float = 0.1,
        min_words_each_sentence: Union[float, int] = 0.8,
        ignore_first: bool = False,
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super(RandomDeletion, self).__init__(ignore_first, always_apply, p)
        self._validate_transform_init_args(deletion_prob, min_words_each_sentence)
        self.deletion_prob = deletion_prob
        self.min_words_each_sentence = min_words_each_sentence

    def _validate_transform_init_args(self, deletion_prob: float, min_words_each_sentence: Union[float, int]) -> None:
        if not isinstance(deletion_prob, (float, int)):
            raise TypeError(f"deletion_prob must be a real number between 0 and 1. Got: {type(deletion_prob)}")
        if not (0.0 <= deletion_prob <= 1.0):
            raise ValueError(f"deletion_prob must be between 0 and 1. Got: {deletion_prob}")
        if not isinstance(min_words_each_sentence, (float, int)):
            raise TypeError(
                f"min_words_each_sentence must be either an int or a float. Got: {type(min_words_each_sentence)}"
            )
        if isinstance(min_words_each_sentence, float):
            if not (0.0 <= min_words_each_sentence <= 1.0):
                raise ValueError(
                    f"If min_words_each_sentence is a float, it must be between 0 and 1. Got: {min_words_each_sentence}"
                )
        elif isinstance(min_words_each_sentence, int):
            if min_words_each_sentence < 0:
                raise ValueError(
                    f"If min_words_each_sentence is an int, it must be non negative. Got: {min_words_each_sentence}"
                )

    def apply(self, text: Text, **params: Any) -> Text:
        return F.delete_words(text, self.deletion_prob, self.min_words_each_sentence)

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("deletion_prob", "min_words_each_sentence")


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


class RandomInsertion(TextTransform):
    """Repeats n times the task of randomly inserting synonyms in the input text.

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


class RandomSwap(TextTransform):
    """Repeats n times the task of randomly swapping two words in a randomly selected sentence from the input text.

    Args:
        n_times: The number of times to repeat the process.
        p: The probability of applying this transform.

    References:
        https://arxiv.org/pdf/1901.11196.pdf
    """

    def __init__(
        self,
        n_times: int = 1,
        ignore_first: bool = False,
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super(RandomSwap, self).__init__(ignore_first, always_apply, p)
        self._validate_transform_init_args(n_times)
        self.n_times = n_times

    def _validate_transform_init_args(self, n_times: int) -> None:
        if not isinstance(n_times, int):
            raise TypeError(f"n_times must be a positive integer. Got: {type(n_times)}")
        if n_times <= 0:
            raise ValueError(f"n_times must be positive. Got: {n_times}")

    def apply(self, text: Text, **params: Any) -> Text:
        return F.swap_words(text, self.n_times)

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("n_times",)


class RandomSwapSentence(TextTransform):
    """Repeats n times the task of randomly swapping two sentences in the input text.

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


class SynonymReplacement(TextTransform):
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
