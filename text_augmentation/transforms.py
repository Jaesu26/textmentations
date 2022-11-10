from typing import Any, Callable, Dict, List, Tuple

from albumentations.core.transforms_interface import BasicTransform

from .utils import Word, Sentence, Text
from . import functional as F

__all__ = [
    "TextTransform",
    "RandomSwapWords",
    "RandomSwapSentences",
    "RandomDeletionWords",
    "RandomDeletionSentences",
    "DeletionFullstops",
    "DeletionLastFullstop",
]


class TextTransform(BasicTransform):
    """Transform applied to a text"""

    @property
    def targets(self) -> Dict[str, Callable[[Text], Text]]:
        return {"text": self.apply}
      
    def update_params(self, params: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        params.update({"len_text":len(kwargs["text"])})
        return params 

    
class RandomSwapWords(TextTransform):
    """Randomly swap two words in a random sentence in the text"""

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
        min_words_each_sentence: int = 5,
        deletion_prob: float = 0.1, 
        ignore_first: bool = False,
        always_apply: bool = False, 
        p: float = 0.5
    ) -> None:
        super(RandomDeletionWords, self).__init__(always_apply, p)
        
        if not isinstance(min_words_each_sentence, int) or min_words_each_sentence < 0:
            raise ValueError(f"min_words_each_sentence must be non negative integer. Got: {min_words_each_sentence}")
        
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
        min_sentences: int = 3,
        deletion_prob: float = 0.1,
        ignore_first: bool = False,
        always_apply: bool = False, 
        p: float = 0.5
    ) -> None:
        super(RandomDeletionSentences, self).__init__(always_apply, p)
        
        if not isinstance(min_sentences, int) or min_sentences < 0:
            raise ValueError(f"min_sentences must be non negative integer. Got: {min_sentences}")
    
        self.min_sentences = min_sentences
        self.deletion_prob = deletion_prob
        self.ignore_first = ignore_first

    def apply(self, text: Text, **params: Any) -> Text:
        return F.delete_sentences(text, self.min_sentences, self.deletion_prob, self.ignore_first)

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return ("min_sentences", "deletion_prob", "ignore_first")

    
class DeletionFullstops(TextTransform):
    """Delete full stops in the text"""

    def __init__(self, always_apply: bool = False, p: float = 1.0) -> None:
        super(DeletionFullstops, self).__init__(always_apply, p)

    def apply(self, text: Text, **params: Any) -> Text:
        return F.delete_fullstops(text)

    def get_transform_init_args_names(self) -> Tuple[()]:
        return ()

    
class DeletionLastFullstop(TextTransform):
    """Delete a full stop at the end of the text"""

    def __init__(self, always_apply: bool = False, p: float = 1.0) -> None:
        super(DeletionLastFullstop, self).__init__(always_apply, p)

    def apply(self, text: Text, **params: Any) -> Text:
        return F.delete_last_fullstop(text)

    def get_transform_init_args_names(self) -> Tuple[()]:
        return ()
