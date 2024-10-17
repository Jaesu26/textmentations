from __future__ import annotations

import math
from typing import List, Literal, Set

import numpy as np
from kiwipiepy import Kiwi, Token

from textmentations.augmentations.utils import (
    Autopsy,
    _find_true_indices,
    _flatten,
    _generate_boolean_mask,
    autopsy_sentence,
    autopsy_text,
    check_rng,
    pass_empty_text,
)
from textmentations.corpora.types import Corpus, Morpheme, POSTag, Sentence, Text, Word
from textmentations.corpora.utils import choose_synonym, is_stopword

_ADDS_SPACE: Literal[False] = False
_NOUN_TAGS: Set[POSTag] = {"NNG", "NNP", "NNB", "NR", "NP"}
_morpheme_analyzer = Kiwi()


@pass_empty_text
def delete_words(
    text: Text,
    deletion_prob: float,
    min_words_per_sentence: float | int,
    *,
    seed: int | np.random.Generator | None = None,
) -> Text:
    """Randomly deletes words in the text.

    Args:
        text: The input text.
        deletion_prob: The probability of deleting a word.
        min_words_per_sentence:
            If a `float`, it is the minimum proportion of words to retain in each sentence.
            If an `int`, it is the minimum number of words in each sentence.
        seed: The seed for a random number generator. Can be None, an int, or an instance of np.random.Generator.
            If `None`, a new random number generator is created with a random seed.
            If an `int`, a generator is created using the seed.
            If an instance of `np.random.Generator`, it is used directly.

    Returns:
        A text with randomly deleted words.

    Examples:
        >>> import textmentations.augmentations.modification.functional as fm
        >>> text = "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."
        >>> deletion_prob = 0.1
        >>> min_words_per_sentence = 0.8
        >>> augmented_text = fm.delete_words(text, deletion_prob, min_words_per_sentence)
    """
    rng = check_rng(seed)
    return _delete_words(text, deletion_prob, min_words_per_sentence, rng)


@autopsy_text
def _delete_words(
    sentences: list[Sentence],
    deletion_prob: float,
    min_words_per_sentence: float | int,
    rng: np.random.Generator,
) -> list[Sentence]:
    """Randomly deletes words in each sentence."""
    return [
        augmented_sentence
        for sentence in sentences
        if (augmented_sentence := _delete_words_in_sentence(sentence, deletion_prob, min_words_per_sentence, rng))
    ]


@autopsy_sentence
def _delete_words_in_sentence(
    words: list[Word],
    deletion_prob: float,
    min_words: float | int,
    rng: np.random.Generator,
) -> list[Word]:
    """Randomly deletes words in the list of words."""
    min_words = math.ceil(len(words) * min_words) if isinstance(min_words, float) else min_words
    return _delete_strings(words, deletion_prob, min_words, rng)


def _delete_strings(
    strings: list[Corpus],
    deletion_prob: float,
    min_strings: int,
    rng: np.random.Generator,
) -> list[Corpus]:
    """Randomly deletes strings in the list of strings."""
    if (num_strings := len(strings)) <= min_strings:
        return strings
    deletion_mask = _generate_boolean_mask(num_strings, deletion_prob, rng)
    indices_to_delete = _find_true_indices(deletion_mask)
    if len(indices_to_delete) > (num_possible_deletions := num_strings - min_strings):
        indices_to_delete = rng.choice(indices_to_delete, size=num_possible_deletions, replace=False)
    indices_to_delete = set(indices_to_delete)
    augmented_strings = [string for index, string in enumerate(strings) if index not in indices_to_delete]
    return augmented_strings


@pass_empty_text
def delete_sentences(
    text: Text,
    deletion_prob: float,
    min_sentences: float | int,
    *,
    seed: int | np.random.Generator | None = None,
) -> Text:
    """Randomly deletes sentences in the text.

    Args:
        text: The input text.
        deletion_prob: The probability of deleting a sentence.
        min_sentences:
            If a `float`, it is the minimum proportion of sentences to retain in the text.
            If an `int`, it is the minimum number of sentences in the text.
        seed: The seed for a random number generator. Can be None, an int, or an instance of np.random.Generator.
            If `None`, a new random number generator is created with a random seed.
            If an `int`, a generator is created using the seed.
            If an instance of `np.random.Generator`, it is used directly.

    Returns:
        A text with randomly deleted sentences.

    Examples:
        >>> import textmentations.augmentations.modification.functional as fm
        >>> text = "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."
        >>> deletion_prob = 0.05
        >>> min_sentences = 0.9
        >>> augmented_text = fm.delete_sentences(text, deletion_prob, min_sentences)
    """
    rng = check_rng(seed)
    return _delete_sentences(text, deletion_prob, min_sentences, rng)


@autopsy_text
def _delete_sentences(
    sentences: list[Sentence],
    deletion_prob: float,
    min_sentences: float | int,
    rng: np.random.Generator,
) -> list[Sentence]:
    """Randomly deletes sentences in the list of sentences."""
    min_sentences = math.ceil(len(sentences) * min_sentences) if isinstance(min_sentences, float) else min_sentences
    return _delete_strings(sentences, deletion_prob, min_sentences, rng)


@pass_empty_text
def insert_synonyms(
    text: Text,
    insertion_prob: float,
    n_times: int,
    *,
    seed: int | np.random.Generator | None = None,
) -> Text:
    """Repeats n times the task of randomly inserting synonyms of words that are not stopwords in the text.

    Args:
        text: The input text.
        insertion_prob: The probability of inserting a synonym.
        n_times: The number of times to repeat the synonym-insertion process.
        seed: The seed for a random number generator. Can be None, an int, or an instance of np.random.Generator.
            If `None`, a new random number generator is created with a random seed.
            If an `int`, a generator is created using the seed.
            If an instance of `np.random.Generator`, it is used directly.

    Returns:
        A text with randomly inserted synonyms.

    Examples:
        >>> import textmentations.augmentations.modification.functional as fm
        >>> text = "물 한 잔만 주세요."
        >>> insertion_prob = 0.2
        >>> n_times = 1
        >>> augmented_text = fm.insert_synonyms(text, insertion_prob, n_times)
    """
    rng = check_rng(seed)
    return _insert_synonyms(text, insertion_prob, n_times, rng)


@autopsy_text
def _insert_synonyms(
    sentences: list[Sentence],
    insertion_prob: float,
    n_times: int,
    rng: np.random.Generator,
) -> list[Sentence]:
    """Repeats n times the task of randomly inserting synonyms of words that are not stopwords in each sentence."""
    return [_insert_synonyms_in_sentence(sentence, insertion_prob, n_times, rng) for sentence in sentences]


@autopsy_sentence
def _insert_synonyms_in_sentence(
    words: list[Word],
    insertion_prob: float,
    n_times: int,
    rng: np.random.Generator,
) -> list[Word]:
    """Repeats n times the task of randomly inserting synonyms of words that are not stopwords in the list of words."""
    for _ in range(n_times):
        words = _insert_synonyms_once_between_words(words, insertion_prob, rng)
    return words


def _insert_synonyms_once_between_words(
    words: list[Word],
    insertion_prob: float,
    rng: np.random.Generator,
) -> list[Word]:
    """Randomly inserts synonyms of words that are not stopwords in the list of words."""
    num_words = len(words)
    augmented_words: List[List[Word]] = [[]]  # To insert synonyms in front of the words
    augmented_words.extend([word] for word in words)
    choice_mask = _generate_boolean_mask(num_words, insertion_prob, rng)
    chosen_word_indices = _find_true_indices(choice_mask)
    chosen_word_indices = rng.permutation(chosen_word_indices).tolist()
    for index in chosen_word_indices:
        word = words[index]
        synonym = _replace_word_in_eojeol_with_synonym(word, rng)
        if synonym == word:
            continue
        insertion_index = rng.integers(num_words + 1)
        augmented_words[insertion_index].append(synonym)
    return _flatten(augmented_words)


@Autopsy(split_func=_morpheme_analyzer.tokenize, join_func=_morpheme_analyzer.join)
def _replace_word_in_eojeol_with_synonym(
    tokens: list[Token],
    rng: np.random.Generator,
) -> list[tuple[Morpheme, POSTag, Literal[False]]]:
    return [
        ((_replace_word_with_synonym(morph, rng), tag, _ADDS_SPACE) if tag in _NOUN_TAGS else (morph, tag, _ADDS_SPACE))
        for morph, tag, *_ in tokens
    ]


def _replace_word_with_synonym(word: Word, rng: np.random.Generator) -> Word:
    """Replaces word that is not stopword with synonym."""
    if not is_stopword(word):
        synonym = choose_synonym(word, rng)
        return synonym
    return word


@pass_empty_text
def insert_punctuation(
    text: Text,
    insertion_prob: float,
    punctuation: tuple[str, ...],
    *,
    seed: int | np.random.Generator | None = None,
) -> Text:
    """Randomly inserts punctuation in the text.

    Args:
        text: The input text.
        insertion_prob: The probability of inserting a punctuation mark.
        punctuation: The punctuation to be inserted at random.
        seed: The seed for a random number generator. Can be None, an int, or an instance of np.random.Generator.
            If `None`, a new random number generator is created with a random seed.
            If an `int`, a generator is created using the seed.
            If an instance of `np.random.Generator`, it is used directly.

    Returns:
        A text with randomly inserted punctuation.

    Examples:
        >>> import textmentations.augmentations.modification.functional as fm
        >>> text = "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."
        >>> insertion_prob = 0.2
        >>> punctuation = (".", ";", "?", ":", "!", ",")
        >>> augmented_text = fm.insert_punctuation(text, insertion_prob, punctuation)
    """
    rng = check_rng(seed)
    return _insert_punctuation(text, insertion_prob, punctuation, rng)


@autopsy_text
def _insert_punctuation(
    sentences: list[Sentence],
    insertion_prob: float,
    punctuation: tuple[str, ...],
    rng: np.random.Generator,
) -> list[Sentence]:
    """Randomly inserts punctuation in each sentence."""
    return [_insert_punctuation_in_sentence(sentence, insertion_prob, punctuation, rng) for sentence in sentences]


@autopsy_sentence
def _insert_punctuation_in_sentence(
    words: list[Word],
    insertion_prob: float,
    punctuation: tuple[str, ...],
    rng: np.random.Generator,
) -> list[Word]:
    """Randomly inserts punctuation in the list of words."""
    augmented_words: List[List[Word]] = [[]]  # To insert a punctuation mark in front of the words
    augmented_words.extend([word] for word in words)
    insertion_mask = _generate_boolean_mask(len(words) + 1, insertion_prob, rng)
    insertion_indices = _find_true_indices(insertion_mask).tolist()
    punctuation_indices = rng.integers(len(punctuation), size=len(insertion_indices)).tolist()
    for insertion_index, punctuation_index in zip(insertion_indices, punctuation_indices):
        augmented_words[insertion_index].append(punctuation[punctuation_index])
    return _flatten(augmented_words)


@pass_empty_text
def replace_synonyms(text: Text, replacement_prob: float, *, seed: int | np.random.Generator | None = None) -> Text:
    """Randomly replaces words that are not stopwords in the text with synonyms.

    Args:
        text: The input text.
        replacement_prob: The probability of replacing a word with a synonym.
        seed: The seed for a random number generator. Can be None, an int, or an instance of np.random.Generator.
            If `None`, a new random number generator is created with a random seed.
            If an `int`, a generator is created using the seed.
            If an instance of `np.random.Generator`, it is used directly.

    Returns:
        A text with random words replaced by synonyms.

    Examples:
        >>> import textmentations.augmentations.modification.functional as fm
        >>> text = "물 한 잔만 주세요."
        >>> replacement_prob = 0.2
        >>> augmented_text = fm.replace_synonyms(text, replacement_prob)
    """
    rng = check_rng(seed)
    return _replace_synonyms(text, replacement_prob, rng)


@autopsy_text
def _replace_synonyms(sentences: list[Sentence], replacement_prob: float, rng: np.random.Generator) -> list[Sentence]:
    """Randomly replaces words that are not stopwords in each sentence with synonyms."""
    return [_replace_synonyms_in_sentence(sentence, replacement_prob, rng) for sentence in sentences]


@Autopsy(split_func=_morpheme_analyzer.tokenize, join_func=_morpheme_analyzer.join)
def _replace_synonyms_in_sentence(
    tokens: list[Token],
    replacement_prob: float,
    rng: np.random.Generator,
) -> list[tuple[Morpheme, POSTag]]:
    """Randomly replaces words that are not stopwords in the list of words with synonyms."""
    replacement_mask = _generate_boolean_mask(len(tokens), replacement_prob, rng).tolist()
    return [
        (_replace_word_with_synonym(morph, rng), tag) if tag in _NOUN_TAGS and should_replace else (morph, tag)
        for (morph, tag, *_), should_replace in zip(tokens, replacement_mask)
    ]


@pass_empty_text
def swap_words(text: Text, alpha: float | int, *, seed: int | np.random.Generator | None = None) -> Text:
    """Repeats n times the task of randomly swapping two words in a randomly selected sentence from the text.

    Args:
        text: The input text.
        alpha:
            If a `float`, it is the number of times to repeat the process is calculated as `N = alpha * L`,
            where `L` is the length of the text.
            If an `int`, it is the number of times to repeat the process.
        seed: The seed for a random number generator. Can be None, an int, or an instance of np.random.Generator.
            If `None`, a new random number generator is created with a random seed.
            If an `int`, a generator is created using the seed.
            If an instance of `np.random.Generator`, it is used directly.

    Returns:
        A text with randomly shuffled words each sentence.

    Examples:
        >>> import textmentations.augmentations.modification.functional as fm
        >>> text = "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."
        >>> alpha = 0.1
        >>> augmented_text = fm.swap_words(text, alpha)
    """
    rng = check_rng(seed)
    n_times = math.ceil(len(text) * alpha) if isinstance(alpha, float) else alpha
    return _swap_words(text, n_times, rng)


@autopsy_text
def _swap_words(sentences: list[Sentence], n_times: int, rng: np.random.Generator) -> list[Sentence]:
    """Repeats n times the task of randomly swapping two words in a randomly selected sentence."""
    sentence_lengths = np.array([*map(len, sentences)])
    weights = sentence_lengths / np.sum(sentence_lengths)
    chosen_sentence_indices = rng.choice(len(sentences), replace=True, size=n_times, p=weights).tolist()
    for index in chosen_sentence_indices:
        sentences[index] = _swap_two_words_in_sentence(sentences[index], rng)
    return sentences


@autopsy_sentence
def _swap_two_words_in_sentence(words: list[Word], rng: np.random.Generator) -> list[Word]:
    """Randomly swaps two words in the list of words."""
    return _swap_two_strings(words, rng)


def _swap_two_strings(strings: list[Corpus], rng: np.random.Generator) -> list[Corpus]:
    """Randomly swaps two strings in the list of strings."""
    if (num_strings := len(strings)) >= 2:
        index1, index2 = rng.choice(num_strings, replace=False, size=2).tolist()
        strings[index1], strings[index2] = strings[index2], strings[index1]
    return strings


@pass_empty_text
def swap_sentences(text: Text, n_times: int, *, seed: int | np.random.Generator | None = None) -> Text:
    """Repeats n times the task of randomly swapping two sentences in the text.

    Args:
        text: The input text.
        n_times: The number of times to repeat the process.
        seed: The seed for a random number generator. Can be None, an int, or an instance of np.random.Generator.
            If `None`, a new random number generator is created with a random seed.
            If an `int`, a generator is created using the seed.
            If an instance of `np.random.Generator`, it is used directly.

    Returns:
        A text with randomly shuffled sentences.

    Examples:
        >>> import textmentations.augmentations.modification.functional as fm
        >>> text = "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."
        >>> n_times = 1
        >>> augmented_text = fm.swap_sentences(text, n_times)
    """
    rng = check_rng(seed)
    return _swap_sentences(text, n_times, rng)


@autopsy_text
def _swap_sentences(sentences: list[Sentence], n_times: int, rng: np.random.Generator) -> list[Sentence]:
    """Repeats n times the task of randomly swapping two sentences in the list of sentences."""
    for _ in range(n_times):
        sentences = _swap_two_strings(sentences, rng)
    return sentences
