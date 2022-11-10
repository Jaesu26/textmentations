from typing import List

import re
import numpy as np

from .utils import (
    strip,
    get_words_from_sentence,
    get_sentences_from_text,
    get_sentence_from_words,
    get_text_from_sentences,
    Word,
    Sentence,
    Text
)

__all__ = [
    "swap_words",
    "swap_sentences",
    "delete_words",
    "delete_sentences",
    "delete_fullstops",
    "delete_last_fullstop",
]


def swap_words(text: Text, ignore_first: bool) -> Text:
    """Randomly swap two words in a random sentence in the text"""
    sentences = get_sentences_from_text(text)
    if len(sentences) <= ignore_first + 1:
        return text

    idx = np.random.randint(ignore_first, len(sentences))
    words = get_words_from_sentence(sentences[idx])
    if len(words) >= 2:    
        idx1, idx2 = np.random.choice(len(words), size=2, replace=False)
        words[idx1], words[idx2] = words[idx2], words[idx1]

    sentence = get_sentence_from_words(words)
    sentences[idx] = sentence
    text = get_text_from_sentences(sentences)
    return text


def swap_sentences(text: Text, ignore_first: bool) -> Text:
    """Randomly swap two sentences in the text"""
    sentences = get_sentences_from_text(text)
    if len(sentences) < ignore_first + 2:
        return text
        
    idx1, idx2 = np.random.choice(np.arange(ignore_first, len(sentences)), size=2, replace=False)
    sentences[idx1], sentences[idx2] = sentences[idx2], sentences[idx1]
    text = get_text_from_sentences(sentences)
    return text


def delete_words(
    text: Text, 
    min_words: int, 
    deletion_prob: float, 
    ignore_first: bool
) -> Text:
    """Randomly delete words in the text"""
    sentences = get_sentences_from_text(text)
    new_sentences = [sentences[0]] if ignore_first else [] 
    
    for sentence in sentences[ignore_first:]:
        words = get_words_from_sentence(sentence)
        new_words = words

        if len(words) > min_words:
            new_words = []
            deletion_max_counts = len(words) - min_words
            deletion_counts = 0
            for word in words:
                if np.random.random() < deletion_prob and deletion_counts < deletion_max_counts:
                    deletion_counts += 1
                    continue
                new_words.append(word)

        new_sentence = get_sentence_from_words(new_words)
        if new_sentence:
            new_sentences.append(new_sentence)

    text = get_text_from_sentences(new_sentences)
    return text

    
def delete_sentences(
    text: Text, 
    min_sentences: int, 
    deletion_prob: float, 
    ignore_first: bool
) -> Text:
    """Randomly delete sentences in the text"""
    sentences = get_sentences_from_text(text)
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

    text = get_text_from_sentences(new_sentences)
    return text


def delete_fullstops(text: Text) -> Text:
    """Delete full stops in the text"""
    text = text.replace(".", " ")
    text = re.sub(r" +", " ", text) 
    text = text.strip()   
    return text


def delete_last_fullstop(text: Text) -> Text:
    """Delete a full stop at the end of the text"""
    if text.endswith("."):
        text = text[:-1]
    return text
