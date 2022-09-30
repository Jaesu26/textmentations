from typing import Any, Callable, Dict, Tuple

from albumentations.augmentations.core.transforms_interface import BasicTransform

from .functions import Word, Words, Sentence, Sentences, Text
from . import functional as F

__all__ = [
    "TextTransform",
    "RandomSwapSentences",
    "RandomSwapWords",
    "RandomDeletionSentences",
    "RandomDeletionWords",
]


class TextTransform(BasicTransform):
    """Transform applied to text"""

    @property
    def targets(self) -> Dict[str, Callable]:
        return {"text": self.apply}
      
    def update_params(self, params: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        return params 

    def get_words_from_sentence(self, sentence: Sentence) -> Words:
        """Split a sentence to get words"""
        words = sentence.split()
        return words
    
    def get_sentences_from_text(self, text: Text) -> Sentences:
        """Split a text to get sentences"""
        sentences = text.split(".")
        return sentences[:-1] 
    
    def get_sentence_from_words(self, words: Words) -> Sentence:
        """Combine words to get a sentence"""
        sentence = " ".join(words)
        return sentence
    
    def get_text_from_sentences(self, sentences: Sentences) -> Text:
        """Combine sentences to get a text"""
        text = ".".join(sentences) + "."
        return text


class RandomSwapSentences(TextTransform):
    """Randomly swap two sentences in a text"""

    def __init__(self, always_apply: bool = False, p: float = 0.5) -> None:
        super(RandomSwapSentences, self).__init__(always_apply, p)

    def apply(self, text: Text, **params: Any) -> Text:
        sentences = self.get_senteces_from_text(text)
        sentences = F.swap_sentences(sentences)
        text = self.get_text_from_sentences(sentences)
        return text

    def get_transform_init_args_names(self) -> Tuple[()]:
        return ()


class RandomSwapWords(TextTransform):
    """Randomly swap two words in a random sentence"""

    def __init__(self, always_apply: bool = False, p: float = 0.5) -> None:
        super(RandomSwapWords, self).__init__(always_apply, p)

    def apply(self, text: Text, **params: Any) -> Text:
        sentences = self.get_sentences_from_text(text)
        idx = np.random.randint(len(sentences))
        words = self.get_words_from_sentence(sentences[idx])
        words = F.swap_words(words)
        sentence = self.get_sentence_from_words(words)
        sentences[idx] = sentence
        text = self.get_text_from_sentences(sentences)
        return text

    def get_transform_init_args_names(self) -> Tuple[()]:
        return ()


class RandomDeletionSentences(TextTransform):
    """Randomly delete sentences in a text"""

    def __init__(
        self, 
        min_sentences: int = 3,
        deletion_prob: float = 0.1, 
        always_apply: bool = False, 
        p: float = 0.5
    ) -> None:
        super(RandomDeletionSentences, self).__init__(always_apply, p)
        
        if not isinstance(min_sentences, int) or min_sentences < 0:
            raise ValueError(f"min_sentences must be non negative integer. Got: {min_sentence}")
    
        self.min_sentences = min_sentences
        self.deletion_prob = deletion_prob

    def apply(self, text: Text, **params: Any) -> Text:
        sentences = self.get_senteces_from_text(text)
        sentences = F.delete_sentences(sentences, self.min_sentences, self.deletion_prob)
        text = self.get_text_from_sentences(sentences)
        return text

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("min_sentences", "deletion_prob")


class RandomDeletionWords(TextTransform):
    """Randomly delete words in a text"""

    def __init__(
        self, 
        min_words_each_sentence: int = 3,
        deletion_prob: float = 0.1, 
        always_apply: bool = False, 
        p: float = 0.5
    ) -> None:
        super(RandomDeletionSentences, self).__init__(always_apply, p)
        
        if not isinstance(min_words_each_sentence, int) or min_words_each_sentence < 0:
            raise ValueError(f"min_words_each_sentence must be non negative integer. Got: {min_words_each_sentence}")
        
        self.min_words_each_sentence = min_words_each_sentence
        self.deletion_prob = deletion_prob

    def apply(self, text: Text, **params: Any) -> Text:
        new_sentences = []
        sentences = self.get_senteces_from_text(text)
        for sentence in sentences:
            words = self.get_words_from_sentence(sentence)
            words = F.delete_words(words, self.min_words_each_sentence, self.delete_prob)
            new_sentences.append(self.combine_words(words))
        text = self.get_text_from_sentences(new_sentences)
        return text

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("min_words_each_sentence", "delete_prob")
