import math
import random
from typing import List, Union

from ..corpora.corpus_types import Word, Sentence, Text, WS
from ..corpora.utils import get_stopwords, get_synonyms
from .utils import autopsy_sentence, autopsy_text, pass_empty_text

__all__ = [
    "delete_words",
    "delete_sentences",
    "insert_synonyms",
    "replace_synonyms",
    "swap_words",
    "swap_sentences",
]


def delete_words(text: Text, deletion_prob: float, min_words_each_sentence: Union[float, int]) -> Text:
    """Randomly deletes words in the text.

    Args:
        text (Text): The input text.
        deletion_prob (float): The probability of deleting a word.
        min_words_each_sentence (Union[float, int]):
            If a `float`, then it is the minimum proportion of words to retain in each sentence.
            If an `int`, then it is the minimum number of words in each sentence.

    Returns:
        Text: A text with randomly deleted words.

    Examples:
        >>> text = "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."
        >>> deletion_prob = 0.7
        >>> min_words_each_sentence = 1
        >>> delete_words(text, deletion_prob, min_words_each_sentence)
        "짜장면을. 짬뽕도. 먹고 싶었다."
    """
    return _delete_words(text, deletion_prob, min_words_each_sentence)


@autopsy_text
def _delete_words(
    sentences: List[Sentence],
    deletion_prob: float,
    min_words_each_sentence: Union[float, int]
) -> List[Sentence]:
    """Randomly deletes words in each sentence. Decorated with `autopsy_text`."""
    augmented_sentences = []
    for sentence in sentences:
        augmented_sentence = delete_words_in_sentence(sentence, deletion_prob, min_words_each_sentence)
        if not augmented_sentence:
            continue
        augmented_sentences.append(augmented_sentence)
    return augmented_sentences


@autopsy_sentence
def delete_words_in_sentence(words: List[Word], deletion_prob: float, min_words: Union[float, int]) -> List[Word]:
    """Randomly deletes words in the list of words. Decorated with `autopsy_sentence`."""
    return delete_strings(words, deletion_prob, min_words)


def delete_strings(
    strings: List[WS],
    deletion_prob: float,
    min_strings: Union[float, int]
) -> List[WS]:
    """Randomly deletes strings in the list of strings."""
    num_strings = len(strings)
    if isinstance(min_strings, float):
        min_strings = math.ceil(num_strings * min_strings)
    if num_strings <= min_strings:
        return strings

    augmented_strings = []
    max_deletion_counts = num_strings - min_strings
    deleted_counts = 0
    for string in strings:
        if random.random() < deletion_prob and deleted_counts < max_deletion_counts:
            deleted_counts += 1
            continue
        augmented_strings.append(string)
    return augmented_strings


def delete_sentences(text: Text, deletion_prob: float, min_sentences: Union[float, int]) -> Text:
    """Randomly deletes sentences in the text.

    Args:
        text (Text): The input text.
        deletion_prob (float): The probability of deleting a sentence.
        min_sentences (Union[float, int]):
            If a `float`, then it is the minimum proportion of sentences to retain in the text.
            If an `int`, then it is the minimum number of sentences in the text.

    Returns:
        Text: A text with randomly deleted sentences.

    Examples:
        >>> text = "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."
        >>> deletion_prob = 0.5
        >>> min_sentences = 1
        >>> delete_sentences(text, deletion_prob, min_sentences)
        "짬짜면도 먹고 싶었다."
    """
    return _delete_sentences(text, deletion_prob, min_sentences)


@autopsy_text
def _delete_sentences(
    sentences: List[Sentence],
    deletion_prob: float,
    min_sentences: Union[float, int]
) -> List[Sentence]:
    """Randomly deletes sentences in the list of sentences. Decorated with `autopsy_text`."""
    return delete_strings(sentences, deletion_prob, min_sentences)


def insert_synonyms(text: Text, insertion_prob: float, n_times: int) -> Text:
    """Repeats n times the task of randomly inserting synonyms of words that are not stopwords in the text.

    Args:
        text (Text): The input text.
        insertion_prob (float): The probability of inserting a synonym.
        n_times (int): The number of times to repeat the synonym-insertion process.

    Returns:
        Text: A text with randomly inserted synonyms.

    Examples:
        >>> text = "물 한잔만 주세요."
        >>> insertion_prob = 0.7
        >>> n_times = 2
        >>> insert_synonyms(text, insertion_prob, n_times)
        "음료 물 한잔만 상수도 주세요."
    """
    return _insert_synonyms(text, insertion_prob, n_times)


@autopsy_text
def _insert_synonyms(sentences: List[Sentence], insertion_prob: float, n_times: int) -> List[Sentence]:
    """Repeats n times the task of randomly inserting synonyms of words that are not stopwords in each sentence.
    Decorated with `autopsy_text`.
    """
    augmented_sentences = []
    for sentence in sentences:
        augmented_sentence = insert_synonyms_in_sentence(sentence, insertion_prob, n_times)
        augmented_sentences.append(augmented_sentence)
    return augmented_sentences


@autopsy_sentence
def insert_synonyms_in_sentence(words: List[Word], insertion_prob: float, n_times: int) -> List[Word]:
    """Repeats n times the task of randomly inserting synonyms of words that are not stopwords in the list of words.
    Decorated with `autopsy_sentence`.
    """
    augmented_words = words[:]
    for _ in range(n_times):
        augmented_words = insert_synonyms_into_another(words, augmented_words, insertion_prob)
    return augmented_words


def insert_synonyms_into_another(
    inserting_words: List[Word],
    augmented_words: List[Word],
    insertion_prob: float
) -> List[Word]:
    """Randomly inserts synonyms of `inserting_words` that are not stopwords into `augmented_words` at random position.

    Args:
        inserting_words (List[Word]): The words whose synonyms will be inserted into `augmented_words`.
        augmented_words (List[Word]): The words into which the synonyms will be inserted.
        insertion_prob (float): The probability of inserting a synonym for each word in `inserting_word`.

    Returns:
        List[Word]: `augmented_words` with synonyms of `inserting_words` randomly inserted.
    """
    stopwords = get_stopwords()
    current_num_words = len(augmented_words)
    for inserting_word in inserting_words:
        if inserting_word not in stopwords and random.random() < insertion_prob:
            synonym = replace_word_with_synonym(inserting_word)
            if synonym == inserting_word:
                continue
            synonym_index = random.randint(0, current_num_words)
            augmented_words.insert(synonym_index, synonym)
            current_num_words += 1
    return augmented_words


def replace_word_with_synonym(word: Word) -> Word:
    """Replaces the word with one of synonyms at random."""
    synonyms = get_synonyms(word)
    if synonyms:
        synonym = random.choice(synonyms)
        return synonym
    return word


def replace_synonyms(text: Text, replacement_prob: float) -> Text:
    """Randomly replaces words that are not stopwords in the text with synonyms.
    
    Args:
        text (Text): The input text.
        replacement_prob (float): The probability of replacing a word with a synonym.

    Returns:
        Text: A text with random words replaced by synonyms.

    Examples:
        >>> text = "물 한잔만 주세요."
        >>> replacement_prob = 0.5
        >>> replace_synonyms(text, replacement_prob)
        "음료 한잔만 주세요."
    """
    return _replace_synonyms(text, replacement_prob)


@autopsy_text
def _replace_synonyms(sentences: List[Sentence], replacement_prob: float) -> List[Sentence]:
    """Randomly replaces words that are not stopwords in each sentence with synonyms.
    Decorated with `autopsy_text`.
    """
    augmented_sentences = []
    for sentence in sentences:
        sentence = replace_synonyms_in_sentence(sentence, replacement_prob)
        augmented_sentences.append(sentence)
    return augmented_sentences


@autopsy_sentence
def replace_synonyms_in_sentence(words: List[Word], replacement_prob: float) -> List[Word]:
    """Randomly replaces words that are not stopwords in the list of words with synonyms.
    Decorated with `autopsy_sentence`.
    """
    stopwords = get_stopwords()
    augmented_words = []
    for word in words:
        if word not in stopwords and random.random() < replacement_prob:
            synonym = replace_word_with_synonym(word)
            augmented_words.append(synonym)
    return augmented_words


@pass_empty_text
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
def _swap_words(sentences: List[Sentence], n_times: int) -> List[Sentence]:
    """Repeats n times the task of randomly swapping two words in a randomly selected sentence.
    Decorated with `autopsy_text`.
    """
    num_sentences = len(sentences)
    augmented_sentences = sentences
    for _ in range(n_times):
        index = random.randrange(num_sentences)
        augmented_sentences[index] = swap_two_words_in_sentence(augmented_sentences[index])
    return augmented_sentences


@autopsy_sentence
def swap_two_words_in_sentence(words: List[Word]) -> List[Word]:
    """Randomly swaps two words in the list of words. Decorated with `autopsy_sentence`."""
    return swap_two_strings(words)


def swap_two_strings(strings: List[WS]) -> List[WS]:
    """Randomly swaps two strings in the list of strings."""
    num_strings = len(strings)
    if num_strings >= 2:
        index1, index2 = random.sample(range(num_strings), k=2)
        strings[index1], strings[index2] = strings[index2], strings[index1]
    return strings


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
        augmented_sentences = swap_two_strings(augmented_sentences)
    return augmented_sentences


@autopsy_text
def _insert_punctuation(sentences: List[Sentence], n_times: int) -> List[Sentence]:
    """Repeats n times the task of randomly swapping two words in a randomly selected sentence.
    Decorated with `autopsy_text`.
    """
    num_sentences = len(sentences)
    if num_sentences < 1:
        return sentences

    augmented_sentences = sentences
    for _ in range(n_times):
        index = random.randrange(num_sentences)
        augmented_sentences[index] = swap_two_words_in_sentence(augmented_sentences[index])
    return augmented_sentences


# TODO: AEDA 함수 구현
@autopsy_sentence
def insert_punctuation_in_sentence(words, insertion_prob, punctuations):
    augmented_words = []
    for word in words:
        if random.random() < insertion_prob:
            punctuation = random.choice(punctuations)
            augmented_words.append(punctuation)
        augmented_words.append(word)
    return augmented_words
