import numbers
from typing import Any, Dict, List, Tuple, Union

from ..core.transforms_interface import TextTransform
from ..corpora.corpus_types import Text
from .utils import split_text
from . import functional as F

__all__ = [
    "RandomDeletionWords",
    "RandomDeletionSentences",
    "RandomInsertion",
    "RandomSwapWords",
    "RandomSwapSentences",
    "SynonymsReplacement",
]


class RandomDeletionWords(TextTransform):
    """Randomly deletes words in the input text.
    
    Args:
        deletion_probability (float): The probability of deleting a word. Default 0.1.
        min_words_each_sentence (Union[float, int]):
            If a `float`, then it is the minimum proportion of words to retain in each sentence.
            If an `int`, then it is the minimum number of words in each sentence. Default 5.
        p (float): The probability of applying the transform. Default: 0.5.
    """

    def __init__(
        self,
        deletion_probability: float = 0.1,
        min_words_each_sentence: Union[float, int] = 5,
        ignore_first: bool = False,
        always_apply: bool = False, 
        p: float = 0.5
    ) -> None:
        super(RandomDeletionWords, self).__init__(ignore_first, always_apply, p)
        RandomDeletionWords._validate_params(
            deletion_probability=deletion_probability,
            min_words_each_sentence=min_words_each_sentence,
        )
        self.deletion_probability = deletion_probability
        self.min_words_each_sentence = min_words_each_sentence

    @staticmethod
    def _validate_params(
        *,
        deletion_probability: float,
        min_words_each_sentence: Union[float, int],
    ) -> None:
        if not isinstance(deletion_probability, numbers.Real):
            raise TypeError(f"deletion_probability must be a real number between 0 and 1. Got: {deletion_probability}")
        if not (0.0 <= deletion_probability <= 1.0):
            raise ValueError(f"deletion_probability must be between 0 and 1. Got: {deletion_probability}")
        if not isinstance(min_words_each_sentence, (float, int)):
            raise TypeError(
                f"min_words_each_sentence must be either an int or a float. Got: {type(min_words_each_sentence)}"
            )
        if isinstance(min_words_each_sentence, float) and not (0.0 <= min_words_each_sentence <= 1.0):
            raise ValueError(
                f"If min_words_each_sentence is a float, it must be between 0 and 1. Got: {min_words_each_sentence}"
            )
        if isinstance(min_words_each_sentence, int) and min_words_each_sentence < 0:
            raise ValueError(
                f"If min_words_each_sentence is an int, it must be a non-negative. Got: {min_words_each_sentence}"
            )

    def apply(self, text: Text, **params: Any) -> Text:
        return F.delete_words(text, self.deletion_probability, self.min_words_each_sentence)

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("deletion_probability", "min_words_each_sentence")
    
    
class RandomDeletionSentences(TextTransform):
    """Randomly deletes sentences in the input text.
    
    Args:
        deletion_probability (float): The probability of deleting a sentence. Default 0.1.
        min_sentences (Union[float, int]):
            If a `float`, then it is the minimum proportion of sentences to retain in the text.
            If an `int`, then it is the minimum number of sentences in the text. Default 3.
        p (float): The probability of applying the transform. Default: 0.5.
    """

    def __init__(
        self,
        deletion_probability: float = 0.1,
        min_sentences: Union[float, int] = 3,
        ignore_first: bool = False,
        always_apply: bool = False, 
        p: float = 0.5
    ) -> None:
        super(RandomDeletionSentences, self).__init__(ignore_first, always_apply, p)
        RandomDeletionSentences._validate_params(
            deletion_probability=deletion_probability,
            min_sentences=min_sentences,
        )
        self.deletion_probability = deletion_probability
        self.min_sentences = min_sentences

    @staticmethod
    def _validate_params(
        *,
        deletion_probability: float,
        min_sentences: Union[float, int],
    ) -> None:
        if not isinstance(deletion_probability, numbers.Real):
            raise TypeError(f"deletion_probability must be a real number between 0 and 1. Got: {deletion_probability}")
        if not (0.0 <= deletion_probability <= 1.0):
            raise ValueError(f"deletion_probability must be between 0 and 1. Got: {deletion_probability}")
        if not isinstance(min_sentences, (float, int)):
            raise TypeError(f"min_sentences must be either an int or a float. Got: {type(min_sentences)}")
        if isinstance(min_sentences, float) and not (0.0 <= min_sentences <= 1.0):
            raise ValueError(f"If min_sentences is a float, it must be between 0 and 1. Got: {min_sentences}")
        if isinstance(min_sentences, int) and min_sentences < 0:
            raise ValueError(f"If min_sentences is an int, it must be a non-negative. Got: {min_sentences}")

    def apply(self, text: Text, min_sentences: Union[float, int] = 3, **params: Any) -> Text:
        return F.delete_sentences(text, self.deletion_probability, min_sentences)

    @property
    def targets_as_params(self) -> List[str]:
        return ["text"]

    def get_params_dependent_on_targets(self, params: Dict[str, Text]) -> Dict[str, Union[float, int]]:
        if isinstance(self.min_sentences, int):
            return {"min_sentences": self.min_sentences - self.ignore_first}

        # n: Length of original sentences (>= 2)
        # p: `min_sentences` ([0, 1))
        # q: The minimum proportion of sentences to retain in the text after deletion if `ignore_first` is True
        # If not `ignore_first`: the minimum number of sentences after deleting is n * p
        # If `ignore_first`: the minimum number of sentences after deleting is 1 + (n - 1)*q
        # So, n * p == 1 + (n - 1)*q, ===> q = (n*p - 1) / (n - 1)

        text = params["text"]
        num_original_sentences = len(split_text(text)) + self.ignore_first
        if num_original_sentences < 2:
            return {"min_sentences": self.min_sentences}
        return {
            "min_sentences": (num_original_sentences * self.min_sentences - self.ignore_first)
            / (num_original_sentences - self.ignore_first)
        }

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("deletion_probability", "min_sentences")


class RandomInsertion(TextTransform):
    """Repeats n times the task of randomly inserting synonyms in the input text.

    Args:
        insertion_probability (float): The probability of inserting a synonym. Default 0.2.
        n_times (int): The number of times to repeat the operation. Default 1.
        p (float): The probability of applying the transform. Default: 0.5.
    """

    def __init__(
        self,
        insertion_probability: float = 0.2,
        n_times: int = 1,
        ignore_first: bool = False,
        always_apply: bool = False,
        p: float = 0.5
    ) -> None:
        super(RandomInsertion, self).__init__(ignore_first, always_apply, p)
        RandomInsertion._validate_params(
            insertion_probability=insertion_probability,
            n_times=n_times,
        )
        self.insertion_probability = insertion_probability
        self.n_times = n_times

    @staticmethod
    def _validate_params(
        *,
        insertion_probability: float,
        n_times: int,
    ) -> None:
        if not isinstance(insertion_probability, numbers.Real):
            raise TypeError(
                f"insertion_probability must be a real number between 0 and 1. Got: {insertion_probability}"
            )
        if not (0.0 <= insertion_probability <= 1.0):
            raise ValueError(f"insertion_probability must be between 0 and 1. Got: {insertion_probability}")
        if not isinstance(n_times, int) or n_times <= 0:
            raise ValueError(f"n_times must be a positive integer. Got: {n_times}")

    def apply(self, text: Text, **params: Any) -> Text:
        return F.insert_synonyms(text, self.insertion_probability, self.n_times)

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("insertion_probability", "n_times")


class RandomSwapWords(TextTransform):
    """Repeats n times the task of randomly swapping two words in a randomly selected sentence from the input text.
    
    Args:
        n_times (int): The number of times to repeat the operation. Default: 1.
        p (float): The probability of applying the transform. Default: 0.5.
    """

    def __init__(
        self,
        n_times: int = 1,
        ignore_first: bool = False,
        always_apply: bool = False,
        p: float = 0.5
    ) -> None:
        super(RandomSwapWords, self).__init__(ignore_first, always_apply, p)
        RandomSwapWords._validate_params(
            n_times=n_times,
        )
        self.n_times = n_times

    @staticmethod
    def _validate_params(
        *,
        n_times: int,
    ) -> None:
        if not isinstance(n_times, int):
            raise TypeError(f"n_times must be a positive integer. Got: {n_times}")
        if n_times <= 0:
            raise ValueError(f"n_times must be positive. Got: {n_times}")

    def apply(self, text: Text, **params: Any) -> Text:
        return F.swap_words(text, self.n_times)

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("n_times",)

    
class RandomSwapSentences(TextTransform):
    """Repeats n times the task of randomly swapping two sentences in the input text.
    
    Args:
        n_times (int): The number of times to repeat the operation. Default: 1.
        p (float): The probability of applying the transform. Default: 0.5.
    """

    def __init__(
        self,
        n_times: int = 1,
        ignore_first: bool = False,
        always_apply: bool = False,
        p: float = 0.5
    ) -> None:
        super(RandomSwapSentences, self).__init__(ignore_first, always_apply, p)
        RandomSwapSentences._validate_params(
            n_times=n_times,
        )
        self.n_times = n_times

    @staticmethod
    def _validate_params(
        *,
        n_times: int,
    ) -> None:
        if not isinstance(n_times, int):
            raise TypeError(f"n_times must be a positive integer. Got: {n_times}")
        if n_times <= 0:
            raise ValueError(f"n_times must be positive. Got: {n_times}")

    def apply(self, text: Text, **params: Any) -> Text:
        return F.swap_sentences(text, self.n_times)

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("n_times",)
    
    
class SynonymsReplacement(TextTransform):
    """Randomly replaces words in the input text with synonyms.
    
    Args:
        replacement_probability (float): The probability of replacing a word with a synonym. Default 0.2.
        p (float): The probability of applying the transform. Default: 0.5.
    """

    def __init__(
        self, 
        replacement_probability: float = 0.2,
        ignore_first: bool = False,
        always_apply: bool = False, 
        p: float = 0.5
    ) -> None:
        super(SynonymsReplacement, self).__init__(ignore_first, always_apply, p)
        SynonymsReplacement._validate_params(
            replacement_probability=replacement_probability,
        )
        self.replacement_probability = replacement_probability

    @staticmethod
    def _validate_params(
        *,
        replacement_probability: float,
    ) -> None:
        if not isinstance(replacement_probability, numbers.Real):
            raise TypeError(
                f"replacement_probability must be a real number between 0 and 1. Got: {replacement_probability}"
            )
        if not (0.0 <= replacement_probability <= 1.0):
            raise ValueError(f"replacement_probability must be between 0 and 1. Got: {replacement_probability}")

    def apply(self, text: Text, **params: Any) -> Text:
        return F.replace_synonyms(text, self.replacement_probability)

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("replacement_probability",)
