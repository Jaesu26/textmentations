from typing import Any, Callable, Dict, Tuple, Union
from ..corpus_types import Text

from albumentations.core.transforms_interface import BasicTransform

from . import functional as F

__all__ = [
    "TextTransform",
    "RandomSwapWords",
    "RandomSwapSentences",
    "RandomDeletionWords",
    "RandomDeletionSentences",
    "SynonymsReplacement",
]


class TextTransform(BasicTransform):
    """Transform applied to text.
    
    Args:
        always_apply (bool): whether the transform should be always applied. Default: False. 
        p (float): probability of applying the transform. Default: 0.5.
    """

    @property
    def targets(self) -> Dict[str, Callable[[Text], Text]]:
        return {"text": self.apply}
      
    def update_params(self, params: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        return params 

    
class RandomSwapWords(TextTransform):
    """Randomly swap two words in a randomly selected sentence from the input text.
    
    Args:
        ignore_first (bool): whether to ignore the first sentence when applying the transform. Default: False.
        p (float): probability of applying the transform. Default: 0.5.
        
    Targets:
        text
    """
    
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
    """Randomly swap two sentences in the input text.
    
    Args:
        ignore_first (bool): whether to ignore the first sentence when applying the transform. Default: False.
        p (float): probability of applying the transform. Default: 0.5.
        
    Targets:
        text
    """

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
    """Randomly delete words in the input text.
    
    Args:
        min_words_each_sentence (float or int): the minimum number of words in each sentence.
            If a `float`, then it's the proportion of words that should be kept in each sentence after deletion.
            If an `int`, then it's the minimum number of words in each sentence. Default 5.
        deletion_prob (float): probability of deleting a word. Default 0.1.
        ignore_first (bool): whether to ignore the first sentence when applying the transform. Default: False.
        p (float): probability of applying the transform. Default: 0.5.
        
    Targets:
        text
    """

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
            raise TypeError(f"min_words_each_sentence must be either an integer or a float. Got: {type(min_words_each_sentence)}")
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
    """Randomly delete sentences in the input text.
    
    Args:
        min_sentences (float or int): the minimum number of sentences in the text.
            If a `float`, then it's the proportion of sentences that should be kept in the text after deletion.
            If an `int`, then it's the minimum number of sentences in text. Default 3.
        deletion_prob (float): probability of deleting a sentence. Default 0.1.
        ignore_first (bool): whether to ignore the first sentence when applying the transform. Default: False.
        p (float): probability of applying the transform. Default: 0.5.
        
    Targets:
        text
    """

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
            raise TypeError(f"min_sentences must be either an integer or a float. Got: {type(min_sentences)}")
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
    
    
class SynonymsReplacement(TextTransform):
    """Randomly replace words in the input text with synonyms.
    
    Args:
        replacement_prob (float): probability of replacing a word with a synonym. Default 0.2.
        ignore_first (bool): whether to ignore the first sentence when applying the transform. Default: False.
        p (float): probability of applying the transform. Default: 0.5.
        
    Targets:
        text
    """

    def __init__(
        self, 
        replacement_prob: float = 0.2,
        ignore_first: bool = False,
        always_apply: bool = False, 
        p: float = 0.5
    ) -> None:
        super(SynonymsReplacement, self).__init__(always_apply, p)
        self.replacement_prob = replacement_prob
        self.ignore_first = ignore_first

    def apply(self, text: Text, **params: Any) -> Text:
        return F.replace_synonyms(text, self.replacement_prob, self.ignore_first)

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("replacement_prob", "ignore_first")