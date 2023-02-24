from typing import List, Sequence

import re

__all__ = [
    "strip",
    "remove_empty_strings",
    "split_sentence",
    "split_text",
    "combine_words",
    "combine_sentences",
    "Word",
    "Sentence",
    "Text",
]

Word = str
Sentence = str
Text = str


def strip(strings: Sequence[str]) -> List[str]:
    """Remove leading and trailing whitespaces from each string in the sequence"""
    return [string.strip() for string in strings]


def remove_empty_strings(strings: Sequence[str]) -> List[str]:
    """Remove empty strings from a sequence of strings"""
    return [string for string in strings if string]

  
def split_sentence(sentence: Sentence) -> List[Word]:
    """Split the sentence to get words"""
    words = sentence.split()
    words = strip(words)
    words = remove_empty_strings(words)
    return words

      
def split_text(text: Text) -> List[Sentence]:
    """Split the text to get sentences"""
    sentences = re.split(r"[.?!]", text)
    sentences = strip(sentences)
    sentences = remove_empty_strings(sentences)
    return sentences 

  
def combine_words(words: List[Word]) -> Sentence:
    """Combine words to get a sentence"""
    sentence = " ".join(words)
    return sentence

  
def combine_sentences(sentences: List[Sentence]) -> Text:
    """Combine sentences to get a text"""
    text = ". ".join(sentences)
    if text:
        text = ".".join([text, ""])
    return text
