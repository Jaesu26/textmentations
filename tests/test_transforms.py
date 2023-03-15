import pytest

import textmentations as T
import textmentations.augmentations.functional as F
from textmentations.augmentations.utils import split_text, extract_first_sentence


@pytest.mark.parametrize(
    ["deletion_probability", "min_words_each_sentence", "expected_text"],
    [
        (1.0, 0, ""),
        (1.0, 1, "먹었다. 먹었다. 싶었다."),
        (1.0, 0.0, ""),
        (1.0, 0.5, "맛있게 먹었다. 맛있게 먹었다. 먹고 싶었다."),
        (0.0, 1, "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."),
        (0.0, 0.5, "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."),
    ]
)
def test_random_deletion_words(text, deletion_probability, min_words_each_sentence, expected_text):
    augmenter = T.RandomDeletionWords(deletion_probability, min_words_each_sentence, p=1.0)
    data = augmenter(text=text)
    augmented_text = F.delete_words(text, deletion_probability, min_words_each_sentence)
    assert augmented_text == data["text"]
    assert augmented_text == expected_text


@pytest.mark.parametrize(
    ["deletion_probability", "min_sentences", "expected_text"],
    [
        (1.0, 0, ""),
        (1.0, 1, "짬짜면도 먹고 싶었다."),
        (1.0, 0.0, ""),
        (1.0, 0.5, "짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."),
        (0.0, 1, "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."),
        (0.0, 0.5, "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."),
    ]
)
def test_random_deletion_sentences(text, deletion_probability, min_sentences, expected_text):
    augmenter = T.RandomDeletionSentences(deletion_probability, min_sentences, p=1.0)
    data = augmenter(text=text)
    augmented_text = F.delete_sentences(text, deletion_probability, min_sentences)
    assert augmented_text == data["text"]
    assert augmented_text == expected_text


def test_random_insertion():
    text = "물 한잔만 주세요."
    insertion_probability = 1.0
    n_times = 1

    augmenter = T.RandomInsertion(insertion_probability, n_times, p=1.0)
    data = augmenter(text=text)
    assert data["text"] != text


def test_random_swap_words(text):
    original_sentences = split_text(text)
    n = len(original_sentences)
    n_times = 1

    augmenter = T.RandomSwapWords(n_times, p=1.0)
    data = augmenter(text=text)
    augmented_sentences = split_text(data["text"])
    assert sum([original == augmented for original, augmented in zip(original_sentences, augmented_sentences)]) == n - 1


def test_random_swap_sentences(text):
    original_sentences = split_text(text)
    n = len(original_sentences)
    n_times = 1

    augmenter = T.RandomSwapSentences(n_times, p=1.0)
    data = augmenter(text=text)
    augmented_sentences = split_text(data["text"])
    assert sum([original == augmented for original, augmented in zip(original_sentences, augmented_sentences)]) == n - 2


def test_synonyms_replacement():
    text = "물 한잔만 주세요."
    replacement_probability = 1.0

    augmenter = T.SynonymsReplacement(replacement_probability, p=1.0)
    data = augmenter(text=text)
    assert data["text"] != text


@pytest.mark.parametrize(
    "transform",
    [
        T.RandomDeletionWords,
        T.RandomDeletionSentences,
        T.RandomInsertion,
        T.RandomSwapWords,
        T.RandomSwapSentences,
        T.SynonymsReplacement,
    ]
)
def test_empty_input_text(transform):
    text = ""
    augmenter = transform(p=1.0)
    data = augmenter(text=text)
    assert data["text"] == ""


@pytest.mark.parametrize(
    "transform",
    [
        T.RandomDeletionWords,
        T.RandomDeletionSentences,
        T.RandomInsertion,
        T.RandomSwapWords,
        T.RandomSwapSentences,
        T.SynonymsReplacement,
    ]
)
def test_ignore_first(text, transform):
    augmenter = transform(ignore_first=True, p=1.0)
    data = augmenter(text=text)
    assert extract_first_sentence(data["text"]) == extract_first_sentence(text)
