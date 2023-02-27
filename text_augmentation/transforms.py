from typing import Any, Callable, Dict, Tuple, Union

from albumentations.core.transforms_interface import BasicTransform

from .utils import Word, Sentence, Text
from . import functional as F

__all__ = [
    "TextTransform",
    "RandomSwapWords",
    "RandomSwapSentences",
    "RandomDeletionWords",
    "RandomDeletionSentences",
]


class TextTransform(BasicTransform):
    """Transform applied to a text"""

    @property
    def targets(self) -> Dict[str, Callable[[Text], Text]]:
        return {"text": self.apply}
      
    def update_params(self, params: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        return params 

    
class RandomSwapWords(TextTransform):
    """Randomly swap two words in a randomly selected sentence from the text"""
    
    def __init__(
        self, 
        ignore_first: bool = False,
        always_apply: bool = False, 
        p: float = 0.5
    ) -> None:
        super(RandomSwapWords, self).__init__(always_apply, p)
        self.ignore_first = ignore_first

    def apply(self, text: Text, **params: Any) -> Text:
        return F.swap_words(text, self.ignore_first)

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("ignore_first",)

    
class RandomSwapSentences(TextTransform):
    """Randomly swap two sentences in the text"""

    def __init__(
        self, 
        ignore_first: bool = False,
        always_apply: bool = False, 
        p: float = 0.5
    ) -> None:
        super(RandomSwapSentences, self).__init__(always_apply, p)
        self.ignore_first = ignore_first

    def apply(self, text: Text, **params: Any) -> Text:
        return F.swap_sentences(text, self.ignore_first)

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("ignore_first",)


class RandomDeletionWords(TextTransform):
    """Randomly delete words in the text"""

    def __init__(
        self, 
        min_words_each_sentence: Union[float, int] = 5,
        deletion_prob: float = 0.1, 
        ignore_first: bool = False,
        always_apply: bool = False, 
        p: float = 0.5
    ) -> None:
        super(RandomDeletionWords, self).__init__(always_apply, p)
        
        if not isinstance(min_words_each_sentence, (float, int)):
            raise TypeError(f"min_words_each_sentence must be either an integer or a float. Got: {type(min_words_each_sentence)} type")
        if isinstance(min_words_each_sentence, float) and not (0.0 <= min_words_each_sentence <= 1.0):
            raise ValueError(f"If min_words_each_sentence is a float, it must be between 0 and 1. Got: {min_words_each_sentence}")
        if isinstance(min_words_each_sentence, int) and min_words_each_sentence < 0:
            raise ValueError(f"If min_words_each_sentence is an integer, it must be a non-negative. Got: {min_words_each_sentence}")
        
        self.min_words_each_sentence = min_words_each_sentence
        self.deletion_prob = deletion_prob
        self.ignore_first = ignore_first

    def apply(self, text: Text, **params: Any) -> Text:
        return F.delete_words(text, self.min_words_each_sentence, self.deletion_prob, self.ignore_first)

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return ("min_words_each_sentence", "deletion_prob", "ignore_first")
    
    
class RandomDeletionSentences(TextTransform):
    """Randomly delete sentences in the text"""

    def __init__(
        self, 
        min_sentences: Union[float, int] = 3,
        deletion_prob: float = 0.1,
        ignore_first: bool = False,
        always_apply: bool = False, 
        p: float = 0.5
    ) -> None:
        super(RandomDeletionSentences, self).__init__(always_apply, p)
        
        if not isinstance(min_sentences, (float, int)):
            raise TypeError(f"min_sentences must be either an integer or a float. Got: {type(min_sentences)} type")
        if isinstance(min_sentences, float) and not (0.0 <= min_sentences <= 1.0):
            raise ValueError(f"If min_sentences is a float, it must be between 0 and 1. Got: {min_sentences}")
        if isinstance(min_sentences, int) and min_sentences < 0:
            raise ValueError(f"If min_sentences is an integer, it must be a non-negative. Got: {min_sentences}")
    
        self.min_sentences = min_sentences
        self.deletion_prob = deletion_prob
        self.ignore_first = ignore_first

    def apply(self, text: Text, **params: Any) -> Text:
        return F.delete_sentences(text, self.min_sentences, self.deletion_prob, self.ignore_first)

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return ("min_sentences", "deletion_prob", "ignore_first")