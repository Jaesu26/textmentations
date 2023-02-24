from typing import List

import numpy as np

from .utils import (
    split_sentence,
    split_text,
    combine_words,
    combine_sentences,
    Word,
    Sentence,
    Text,
)

__all__ = [
    "swap_words",
    "swap_sentences",
    "delete_words",
    "delete_sentences",
]


def swap_words(text: Text, ignore_first: bool) -> Text:
    """Randomly swap two words in a random sentence in the text"""
    sentences = split_text(text)
    if len(sentences) <= ignore_first + 1:
        return text

    idx = np.random.randint(ignore_first, len(sentences))
    words = split_sentence(sentences[idx])
    if len(words) >= 2:    
        idx1, idx2 = np.random.choice(len(words), size=2, replace=False)
        words[idx1], words[idx2] = words[idx2], words[idx1]

    sentence = combine_words(words)
    sentences[idx] = sentence
    text = combine_sentences(sentences)
    return text


def swap_sentences(text: Text, ignore_first: bool) -> Text:
    """Randomly swap two sentences in the text"""
    sentences = split_text(text)
    if len(sentences) < ignore_first + 2:
        return text
        
    idx1, idx2 = np.random.choice(np.arange(ignore_first, len(sentences)), size=2, replace=False)
    sentences[idx1], sentences[idx2] = sentences[idx2], sentences[idx1]
    text = combine_sentences(sentences)
    return text


def delete_words(
    text: Text, 
    min_words_each_sentence: int, 
    deletion_prob: float, 
    ignore_first: bool
) -> Text:
    """Randomly delete words in the text"""
    sentences = split_text(text)
    new_sentences = [sentences[0]] if ignore_first else [] 
    
    for sentence in sentences[ignore_first:]:
        words = split_sentence(sentence)
        new_words = words

        if len(words) > min_words_each_sentence:
            new_words = []
            deletion_max_counts = len(words) - min_words_each_sentence
            deletion_counts = 0
            for word in words:
                if np.random.random() < deletion_prob and deletion_counts < deletion_max_counts:
                    deletion_counts += 1
                    continue
                new_words.append(word)

        new_sentence = combine_words(new_words)
        if new_sentence:
            new_sentences.append(new_sentence)

    text = combine_sentences(new_sentences)
    return text

    
def delete_sentences(
    text: Text, 
    min_sentences: int, 
    deletion_prob: float, 
    ignore_first: bool
) -> Text:
    """Randomly delete sentences in the text"""
    sentences = split_text(text)
    if len(sentences) <= min_sentences:
        return text

    new_sentences = [sentences[0]] if ignore_first else [] 
    deletion_max_counts = len(sentences) - min_sentences
    deletion_counts = 0
    
    for sentence in sentences[ignore_first:]:
        if np.random.random() < deletion_prob and deletion_counts < deletion_max_counts:
            deletion_counts += 1
            continue
        new_sentences.append(sentence)

    text = combine_sentences(new_sentences)
    return text
