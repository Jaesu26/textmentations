from typing import Any, List

import numpy as np

__all__ = [
    "swap_sentence"
    "swap_words"
    "delete_sentences",
    "delete_words",
]

Word = str
Words = List[str]
Sentence = str
Sentences = List[str]
Text = str


def swap_sentences(sentences: Sentences) -> Sentences:
    """swap two sentences"""
    if len(sentences) < 2:
        return sentences

    idx1, idx2 = np.random.choice(sentences, size=2, replace=False)
    sentences[idx1], sentences[idx2] = sentences[idx2], sentences[idx1]
    return sentences


def swap_words(words: Words) -> Words:
    """swap two words"""
    if len(words) < 2:
        return words

    idx1, idx2 = np.random.choice(words, size=2, replace=False)
    words[idx1], words[idx2] = words[idx2], words[idx1]
    return words


def delete_sentences(sentences: Sentences, min_sentences: int, deletion_prob: float) -> Sentences:
    """delete random sentences"""
    num_sentences = len(sentences)
    deletion_max_counts = num_sentences - min_sentences

    if not isinstance(min_sentences, int) or min_sentences < 0:
        raise ValueError('min_sentences must be positive integer')
    if num_sentences <= min_sentences:
        return sentences

    new_sentences = []
    deletion_counts = 0
    for sentence in sentences:
        if np.random.random() < deletion_prob and deletion_counts < deletion_max_counts:
            deletion_counts += 1
            continue

        new_sentences.append(sentence)
    return new_sentences


def delete_words(words: Words, min_words: int, deletion_prob: float) -> Words:
    """delete random words"""
    num_words = len(words)
    deletion_max_counts = num_words - min_words
    
    if not isinstance(min_words, int) or min_words < 0:
        raise ValueError('min_words must be positive integer')
    if num_words <= min_words:
        return words

    new_words = []
    deletion_counts = 0
    for word in words:
        if np.random.random() < deletion_prob and deletion_counts < deletion_max_counts:
            deletion_counts += 1
            continue
            
        new_words.append(word)
    return new_words
