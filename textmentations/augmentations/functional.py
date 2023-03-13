# TODO: 변수명 명확하게 바꾸기, docstring 세세히 적기 (프로젝트 공통 적용)

import math
import random
from typing import Any, List, Union

from ..corpora.corpus_types import Word, Sentence, Text
from ..corpora.utils import get_stopwords, get_synonyms

from .utils import (
    autopsy_sentence,
    autopsy_text
)

__all__ = [
    "delete_words",
    "delete_sentences",
    "insert_synonyms",
    "replace_synonyms",
    "swap_words",
    "swap_sentences",
]


def delete_words(text: Text, deletion_probability: float, min_words_each_sentence: Union[float, int]) -> Text:
    """Randomly deletes words in the text.

    Args:
        text (Text): The input text.
        deletion_probability (float): The probability of deleting a word.
        min_words_each_sentence (float or int):
            If a `float`, then it is the minimum proportion of words to retain in each sentence after deletion.
            If an `int`, then it is the minimum number of words in each sentence.

    Returns:
        Text: A text with randomly deleted words each sentence.

    Examples:
        >>> text = "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."
        >>> deletion_probability = 0.7
        >>> min_words_each_sentence = 1
        >>> delete_words(text, deletion_probability, min_words_each_sentence)
        "짜장면을. 짬뽕도. 먹고 싶었다."
    """
    return _delete_words(text, deletion_probability, min_words_each_sentence)


@autopsy_text
def _delete_words(
    sentences: List[Sentence],
    deletion_probability: float,
    min_words_each_sentence: Union[float, int]
) -> List[Sentence]:
    """Randomly deletes words in the list of sentences. Decorated with `autopsy_text`."""
    new_sentences = []

    for sentence in sentences:
        sentence = delete_words_in_sentence(sentence, deletion_probability, min_words_each_sentence)
        if sentence:
            new_sentences.append(sentence)

    return new_sentences


@autopsy_sentence
def delete_words_in_sentence(
    words: List[Word],
    deletion_probability: float,
    min_words: Union[float, int]
) -> List[Word]:
    """Randomly deletes words in the list of words. Decorated with `autopsy_sentence`."""
    num_words = len(words)
    if isinstance(min_words, float):
        min_words = math.ceil(num_words * min_words)
    if num_words <= min_words:
        return words

    new_words = []
    max_deletion_counts = num_words - min_words
    deleted_counts = 0
    
    for word in words:
        if random.random() < deletion_probability and deleted_counts < max_deletion_counts:
            deleted_counts += 1
            continue
        new_words.append(word)

    return new_words


def delete_sentences(text: Text, deletion_probability: float, min_sentences: Union[float, int]) -> Text:
    """Randomly deletes sentences in the text.

    Args:
        text (Text): The input text.
        deletion_probability (float): The probability of deleting a sentence.
        min_sentences (float or int):
            If a `float`, then it is the minimum proportion of sentences to retain in the text after deletion.
            If an `int`, then it is the minimum number of sentences in the text.

    Returns:
        Text: A text with randomly deleted sentences.

    Examples:
        >>> text = "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."
        >>> deletion_probability = 0.5
        >>> min_sentences = 1
        >>> delete_sentences(text, deletion_probability, min_sentences)
        "짬짜면도 먹고 싶었다."
    """
    return _delete_sentences(text, deletion_probability, min_sentences)


@autopsy_text
def _delete_sentences(
    sentences: List[Sentence],
    deletion_probability: float,
    min_sentences: Union[float, int]
) -> List[Sentence]:
    """Randomly deletes sentences in the list of sentences. Decorated with `autopsy_text`."""
    num_sentences = len(sentences)
    if isinstance(min_sentences, float):
        min_sentences = math.ceil(num_sentences * min_sentences)    
    if num_sentences <= min_sentences:
        return sentences

    augmented_sentences = []
    max_deletion_counts = num_sentences - min_sentences
    deleted_counts = 0
    
    for sentence in sentences:
        if random.random() < deletion_probability and deleted_counts < max_deletion_counts:
            deleted_counts += 1
            continue
        augmented_sentences.append(sentence)

    return augmented_sentences


def insert_synonyms(text: Text, insertion_probability: float, n_times: int) -> Text:
    """Repeats n times the task of randomly inserting synonyms in the text.

    Args:
        text (Text): The input text.
        insertion_probability (float): The probability of inserting a synonym.
        n_times (int): The number of times to repeat the synonym-insertion process.

    Returns:
        Text: A text with randomly inserted synonyms each sentence.

    Examples:
        >>> text = "물 한잔만 주세요."
        >>> insertion_probability = 0.7
        >>> n_times = 2
        >>> insert_synonyms(text, insertion_probability, n_times)
        "음료 물 한잔만 상수도 주세요."
    """
    return _insert_synonyms(text, insertion_probability, n_times)


@autopsy_text
def _insert_synonyms(sentences: List[Sentence], insertion_probability: float, n_times: int) -> List[Sentence]:
    """Repeats n times the task of randomly inserting synonyms in the list of sentences.
    Decorated with `autopsy_text`.
    """
    new_sentences = []

    for sentence in sentences:
        sentence = insert_synonyms_in_sentence(sentence, insertion_probability, n_times)
        new_sentences.append(sentence)

    return new_sentences


@autopsy_sentence
def insert_synonyms_in_sentence(words: List[Word], insertion_probability: float, n_times: int) -> List[Word]:
    """Repeats n times the task of randomly inserting synonyms in the list of words.
    Decorated with `autopsy_sentence`.
    """
    augmented_words = words[:]
    for _ in range(n_times):
        augmented_words = insert_synonyms_into_another(words, augmented_words, insertion_probability)
    return augmented_words


def insert_synonyms_into_another(
    inserting_words: List[Word],
    target_words: List[Word],
    insertion_probability: float
) -> List[Word]:
    """Randomly inserts synonyms of `inserting_words` into `target_words` at random position.

    Args:
        inserting_words (List[Word]): The words whose synonyms will be inserted into `target_words`.
        target_words (List[Word]): The words into which the synonyms will be inserted.
        insertion_probability (float): The probability of inserting a synonym for each word in `inserting_word`.

    Returns:
        List[Word]: `target_words` with synonyms of `inserting_words` randomly inserted.
    """
    stopwords = get_stopwords()
    current_num_words = len(target_words)

    for inserting_word in inserting_words:
        if inserting_word not in stopwords and random.random() < insertion_probability:
            synonym = replace_word_with_synonym(inserting_word)
            if synonym == inserting_word:
                continue

            synonym_index = random.randint(0, current_num_words)
            target_words.insert(synonym_index, synonym)
            current_num_words += 1

    return target_words


def replace_word_with_synonym(word: Word) -> Word:
    """Replaces the word with one of synonyms at random."""
    synonyms = get_synonyms(word)
    if not synonyms:
        return word

    synonym = random.choice(synonyms)
    return synonym


def replace_synonyms(text: Text, replacement_probability: float) -> Text:
    """Randomly replaces words that are not stopwords in the text with synonyms.
    
    Args:
        text (Text): The input text.
        replacement_probability (float): The probability of replacing a word with a synonym.

    Returns:
        text: A text with random words replaced by synonyms.

    Examples:
        >>> text = "물 한잔만 주세요."
        >>> replacement_probability = 0.5
        >>> replace_synonyms(text, replacement_probability)
        "음료 한잔만 주세요."
    """
    return _replace_synonyms(text, replacement_probability)


@autopsy_text
def _replace_synonyms(sentences: List[Sentence], replacement_probability: float) -> List[Sentence]:
    """Randomly replaces words that are not stopwords in the list of sentences with synonyms.
    Decorated with `autopsy_text`.
    """
    new_sentences = []

    for sentence in sentences:
        sentence = replace_synonyms_in_sentence(sentence, replacement_probability)
        new_sentences.append(sentence)

    return new_sentences


@autopsy_sentence
def replace_synonyms_in_sentence(words: List[Word], replacement_probability: float) -> List[Word]:
    """Randomly replaces words that are not stopwords in the list of words with synonyms.
    Decorated with `autopsy_sentence`.
    """
    stopwords = get_stopwords()
    new_words = []

    for word in words:
        if word not in stopwords and random.random() < replacement_probability:
            synonym = replace_word_with_synonym(word)
            new_words.append(synonym)

    return new_words


def swap_words(text: Text, n_times: int) -> Text:
    """Repeats n times the task of randomly swapping two words in a randomly selected sentence from the text.

    Args:
        text (Text): The input text.
        n_times (int): The number of times to repeat the word-swapping process in a randomly selected sentence.

    Returns:
        Text: A text with randomly shuffled words each sentence.

    Examples:
        >>> text = "짜장면을 맛있게 먹었다. 짬뽕도 먹고 싶었다."
        >>> n_times = 2
        >>> swap_words(text, n_times)
        "맛있게 짜장면을 먹었다. 먹고 짬뽕도 싶었다."
    """
    return _swap_words(text, n_times)


@autopsy_text
def _swap_words(sentences: List[Sentence], n_times: int) -> Text:
    """Repeats n times the task of randomly swapping two words in a randomly selected sentence.
    Decorated with `autopsy_text`.
    """
    num_sentences = len(sentences)
    if num_sentences < 1:
        return sentences

    for _ in range(n_times):
        index = random.randrange(num_sentences)
        sentences[index] = swap_two_words_in_sentence(sentences[index])

    return sentences


@autopsy_sentence
def swap_two_words_in_sentence(words: List[Word]) -> List[Word]:
    """Randomly swaps two words in the list of words. Decorated with `autopsy_sentence`."""
    return swap_two_elements(words)


def swap_sentences(text: Text, n_times: int) -> Text:
    """Repeats n times the task of randomly swapping two sentences in the text.

    Args:
        text (Text): The input text.
        n_times (int): The number of times to repeat the sentence-swapping process.

    Returns:
        Text: A text with randomly shuffled sentences.

    Examples:
        >>> text = "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."
        >>> n_times = 1
        >>> swap_sentences(text, n_times)
        "짜장면을 맛있게 먹었다. 짬짜면도 먹고 싶었다. 짬뽕도 맛있게 먹었다."
    """
    return _swap_sentences(text, n_times)


@autopsy_text
def _swap_sentences(sentences: List[Sentence], n_times: int) -> List[Sentence]:
    """Repeats n times the task of randomly swapping two sentences in the list of sentences.
    Decorated with `autopsy_text`.
    """
    augmented_sentences = sentences
    for _ in range(n_times):
        augmented_sentences = swap_two_elements(augmented_sentences)
    return augmented_sentences


def swap_two_elements(elements: List[Any]) -> List[Any]:
    """Randomly swaps two elements in the list of elements."""
    num_elements = len(elements)
    if num_elements < 2:
        return elements

    index1, index2 = random.sample(range(num_elements), k=2)
    elements[index1], elements[index2] = elements[index2], elements[index1]
    return elements
