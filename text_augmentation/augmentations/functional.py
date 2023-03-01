from typing import List, Union
from ..corpus_types import Word, Sentence, Text

import math
import random

from ..corpora.utils import get_synonyms

from .utils import (
    split_sentence,
    split_text,
    combine_words,
    combine_sentences,
)

__all__ = [
    "delete_words",
    "delete_sentences",
    "replace_synonyms",
    "swap_words",
    "swap_sentences",
]


def delete_words(
    text: Text, 
    min_words_each_sentence: Union[float, int], 
    deletion_prob: float, 
    ignore_first: bool
) -> Text:
    """Randomly delete words in the text."""
    if not text:
        return text
    
    sentences = split_text(text)
    new_sentences = [sentences[0]] if ignore_first else [] 
    
    for sentence in sentences[ignore_first:]:
        sentence = delete_words_in_sentence(sentence, min_words_each_sentence, deletion_prob)
        if sentence:
            new_sentences.append(sentence)

    text = combine_sentences(new_sentences)
    return text


def delete_words_in_sentence(sentence: Sentence, min_words: Union[float, int], deletion_prob: float) -> Sentence:
    """Randomly delete words in the sentence."""
    words = split_sentence(sentence)
    num_words = len(words)
    
    if isinstance(min_words, float):
        min_words = math.ceil(num_words * min_words)
    if num_words <= min_words:
        return sentence

    new_words = []
    max_deletion_counts = num_words - min_words
    deleted_counts = 0
    
    for word in words:
        if random.random() < deletion_prob and deleted_counts < max_deletion_counts:
            deleted_counts += 1
            continue
        new_words.append(word)

    sentence = combine_words(new_words)
    return sentence


def delete_sentences(
    text: Text, 
    min_sentences: Union[float, int], 
    deletion_prob: float, 
    ignore_first: bool
) -> Text:
    """Randomly delete sentences in the text."""
    if not text:
        return text
    
    sentences = split_text(text)
    num_sentences = len(sentences)
    
    if isinstance(min_sentences, float):
        min_sentences = math.ceil(num_sentences * min_sentences)    
    if num_sentences <= min_sentences:
        return text

    new_sentences = [sentences[0]] if ignore_first else [] 
    max_deletion_counts = num_sentences - min_sentences
    deleted_counts = 0
    
    for sentence in sentences[ignore_first:]:
        if random.random() < deletion_prob and deleted_counts < max_deletion_counts:
            deleted_counts += 1
            continue
        new_sentences.append(sentence)

    text = combine_sentences(new_sentences)
    return text


def replace_synonyms(text: Text, replacement_prob: float, ignore_first: bool) -> Text:
    """Randomly replace words in the text with synonyms."""
    if not text:
        return text
    
    sentences = split_text(text)
    new_sentences = [sentences[0]] if ignore_first else [] 
    
    for sentence in sentences[ignore_first:]:
        sentence = replace_synonyms_in_sentence(sentence, replacement_prob)
        new_sentences.append(sentence)
        
    text = combine_sentence(new_sentences)
    return text


def replace_synonyms_in_sentence(sentence: Sentence, replacement_prob: float) -> Sentence:
    """Randomly replace words in the sentence with synonyms."""
    words = split_sentence(sentence)
    new_words = []
    
    for word in words:
        if random.random() < replacement_prob:
            synonym = replace_word_with_synonym(word)
            new_words.append(synonym)

    sentence = combine_words(new_words)
    return sentence


def replace_word_with_synonym(word: Word) -> Word:
    """Replace the word with one of synonyms at random."""
    synonyms = get_synonyms(word)
    if not synonyms:
        return word
    
    synonym = random.choice(synonyms)
    return synonym


def swap_words(text: Text, ignore_first: bool) -> Text:
    """Randomly swap two words in a randomly selected sentence from the text."""
    sentences = split_text(text)
    num_sentences = len(sentences)
    
    if num_sentences < ignore_first + 1:
        return text

    idx = random.randrange(ignore_first, num_sentences)
    sentences[idx] = swap_words_in_sentence(sentences[idx])
    text = combine_sentences(sentences)
    return text


def swap_words_in_sentence(sentence: Sentence) -> Sentence:
    """Randomly swap two words in the sentence."""
    words = split_sentence(sentence)
    num_words = len(words)
    
    if num_words < 2:
        return sentence
    
    idx1, idx2 = random.sample(range(num_words), k=2)
    words[idx1], words[idx2] = words[idx2], words[idx1]
    sentence = combine_words(words)
    return sentence


def swap_sentences(text: Text, ignore_first: bool) -> Text:
    """Randomly swap two sentences in the text."""
    sentences = split_text(text)
    num_sentences = len(sentences)
    
    if num_sentences < ignore_first + 2:
        return text
        
    idx1, idx2 = random.sample(range(ignore_first, num_sentences), k=2)
    sentences[idx1], sentences[idx2] = sentences[idx2], sentences[idx1]
    text = combine_sentences(sentences)
    return text