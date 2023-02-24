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
    "swap_words_in_sentence",
    "swap_sentences",
    "delete_words",
    "delete_words_in_sentence",
    "delete_sentences",
]


def swap_words(text: Text, ignore_first: bool) -> Text:
    """Randomly swap two words in a randomly selected sentence from the text"""
    sentences = split_text(text)
    if len(sentences) <= ignore_first + 1:
        return text

    idx = np.random.randint(ignore_first, len(sentences))
    sentences[idx] = swap_words_in_sentence(sentences[idx])
    text = combine_sentences(sentences)
    return text


def swap_words_in_sentence(sentence: Sentence) -> Sentence:
    """Randomly swap two words in the sentence"""
    words = split_sentence(sentence)
    if len(words) < 2:
        return sentence
    
    idx1, idx2 = np.random.choice(len(words), size=2, replace=False)
    words[idx1], words[idx2] = words[idx2], words[idx1]
    sentence = combine_words(words)
    return sentence


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
        sentence = delete_words_in_sentence(sentence, min_words_each_sentence)
        if sentence:
            new_sentences.append(sentence)

    text = combine_sentences(new_sentences)
    return text


def delete_words_in_sentence(sentence: Sentence, min_words: int) -> Sentence:
    """Randomly delete words in the sentence"""
    words = split_sentence(sentence)
    if len(words) <= min_words:
        return sentence

    new_words = []
    max_deletion_counts = len(words) - min_words
    deleted_counts = 0
    
    for word in words:
        if np.random.random() < deletion_prob and deleted_counts < max_deletion_counts:
            deleted_counts += 1
            continue
        new_words.append(word)

    sentence = combine_words(new_words)
    return sentence


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
    max_deletion_counts = len(sentences) - min_sentences
    deleted_counts = 0
    
    for sentence in sentences[ignore_first:]:
        if np.random.random() < deletion_prob and deleted_counts < max_deletion_counts:
            deleted_counts += 1
            continue
        new_sentences.append(sentence)

    text = combine_sentences(new_sentences)
    return text
