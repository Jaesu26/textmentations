from typing import Any, Callable, Dict, List, Tuple

from albumentations.augmentations.core.transforms_interface import BasicTransform

from .functional import Word, Sentence, Text
from . import functional as F

__all__ = [
    "TextTransform",
    "RandomSwapSentences",
    "RandomSwapWords",
    "RandomDeletionSentences",
    "RandomDeletionWords",
    "DeletionFullstops",
]


class TextTransform(BasicTransform):
    """Transform applied to a text"""

    @property
    def targets(self) -> Dict[str, Callable[[Text], Text]]:
        return {"text": self.apply}
      
    def update_params(self, params: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        return params 

    def get_words_from_sentence(self, sentence: Sentence) -> List[Word]:
        """Split the sentence to get words"""
        words = sentence.split()
        return words
    
    def get_sentences_from_text(self, text: Text) -> List[Sentence]:
        """Split the text to get sentences"""
        sentences = text.split(".")
        if text.endswith("."):
            return sentences[:-1]
        return sentences 
    
    def get_sentence_from_words(self, words: List[Word]) -> Sentence:
        """Combine words to get a sentence"""
        sentence = " ".join(words)
        return sentence
    
    def get_text_from_sentences(self, sentences: List[Sentence]) -> Text:
        """Combine sentences to get a text"""
        text = ".".join(sentences)
        return text

    
class RandomSwapWords(TextTransform):
    """Randomly swap two words in a random sentence"""

    def __init__(
        self, 
        ignore_first: bool = False,
        always_apply: bool = False, 
        p: float = 0.5
    ) -> None:
        super(RandomSwapWords, self).__init__(always_apply, p)
        self.ignore_first = ignore_first

    def apply(self, text: Text, **params: Any) -> Text:
        sentences = self.get_sentences_from_text(text)
        if len(sentences) <= self.ignore_first:
            return text

        idx = np.random.randint(self.ignore_first, len(sentences))
        words = self.get_words_from_sentence(sentences[idx])
        words = F.swap_words(words)
        sentence = self.get_sentence_from_words(words)
        sentences[idx] = sentence
        text = self.get_text_from_sentences(sentences)
        return text

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
        sentences = self.get_sentences_from_text(text)
        sentences = F.swap_sentences(sentences, self.ignore_first)
        text = self.get_text_from_sentences(sentences)
        return text

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
        sentences = self.get_sentences_from_text(text)
        new_sentences = [sentences[0]] if self.ignore_first else [] 
        
        for sentence in sentences[self.ignore_first:]:
            words = self.get_words_from_sentence(sentence)
            words = F.delete_words(words, self.min_words_each_sentence, self.deletion_prob)
            new_sentence = self.get_sentence_from_words(words)
            if new_sentence:
                new_sentences.append(new_sentence)

        text = self.get_text_from_sentences(new_sentences)
        return text

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
        sentences = self.get_sentences_from_text(text)
        sentences = F.delete_sentences(sentences, self.min_sentences, self.deletion_prob, self.ignore_first)
        text = self.get_text_from_sentences(sentences)
        return text

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return ("min_sentences", "deletion_prob", "ignore_first")

    
class DeletionFullstops(TextTransform):
    """Delete full stops in the text"""

    def __init__(self, always_apply: bool = False, p: float = 1.0) -> None:
        super(DeletionFullstops, self).__init__(always_apply, p)

    def apply(self, text: Text, **params: Any) -> Text:
        text = F.delete_fullstops(text)
        return text

    def get_transform_init_args_names(self) -> Tuple[()]:
        return ()
