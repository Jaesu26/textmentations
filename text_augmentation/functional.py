from typing import List

import re
import numpy as np

__all__ = [
    "swap_sentence",
    "swap_words",
    "delete_sentences",
    "delete_words",
    "delete_fullstops",
    "Word",
    "Sentence",
    "Text",
]

Word = str
Sentence = str
Text = str


def swap_words(words: List[Word]) -> List[Word]:
    """Randomly swap two words"""
    if len(words) < 2:
        return words

    idx1, idx2 = np.random.choice(len(words), size=2, replace=False)
    words[idx1], words[idx2] = words[idx2], words[idx1]
    return words


def swap_sentences(sentences: List[Sentence], ignore_first: bool) -> List[Sentence]:
    """Randomly swap two sentences"""
    if len(sentences) < ignore_first + 2:
        return sentences

    idx1, idx2 = np.random.choice(np.arange(ignore_first, len(sentences)), size=2, replace=False)
    sentences[idx1], sentences[idx2] = sentences[idx2], sentences[idx1]
    return sentences


def delete_words(words: List[Word], min_words: int, deletion_prob: float) -> List[Word]:
    """Randomly delete words""" 
    if len(words) <= min_words:
        return words

    new_words = []
    deletion_max_counts = len(words) - min_words
    deletion_counts = 0
    
    for word in words:
        if np.random.random() < deletion_prob and deletion_counts < deletion_max_counts:
            deletion_counts += 1
            continue
            
        new_words.append(word)
    return new_words


def delete_sentences(
    sentences: List[Sentence], 
    min_sentences: int, 
    deletion_prob: float, 
    ignore_first: bool
) -> List[Sentence]:
    """Randomly delete sentences"""
    if len(sentences) <= min_sentences:
        return sentences

    new_sentences = [sentences[0]] if ignore_first else [] 
    deletion_max_counts = len(sentences) - min_sentences
    deletion_counts = 0
    
    for sentence in sentences[ignore_first:]:
        if np.random.random() < deletion_prob and deletion_counts < deletion_max_counts:
            deletion_counts += 1
            continue

        new_sentences.append(sentence)
    return new_sentences


def delete_fullstops(text: Text) -> Text:
    """Delete full stops in the text"""
    text = text.replace(".", " ")
    text = re.sub(r"\s+", " ", text) 
    text = text.strip()   
    return text
