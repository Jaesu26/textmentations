import pytest

import textmentations.augmentations.functional as F
from textmentations.augmentations.utils import split_text


@pytest.mark.parametrize(
    ["deletion_prob", "min_words_each_sentence", "expected_text"],
    [
        (0.0, 0.5, "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."),
        (0.0, 1, "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."),
        (1.0, 0.0, ""),
        (1.0, 0.5, "맛있게 먹었다. 맛있게 먹었다. 먹고 싶었다."),
        (1.0, 0, ""),
        (1.0, 1, "먹었다. 먹었다. 싶었다."),
        (True, False, ""),
        (True, True, "먹었다. 먹었다. 싶었다."),
    ]
)
def test_delete_words(text_without_synonyms, deletion_prob, min_words_each_sentence, expected_text):
    augmented_text = F.delete_words(text_without_synonyms, deletion_prob, min_words_each_sentence)
    assert augmented_text == expected_text


@pytest.mark.parametrize(
    ["deletion_prob", "min_sentences", "expected_text"],
    [
        (0.0, 0.5, "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."),
        (0.0, 1, "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."),
        (1.0, 0.0, ""),
        (1.0, 0.5, "짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."),
        (1.0, 0, ""),
        (1.0, 1, "짬짜면도 먹고 싶었다."),
        (True, False, ""),
        (True, True, "짬짜면도 먹고 싶었다."),
    ]
)
def test_delete_sentences(text_without_synonyms, deletion_prob, min_sentences, expected_text):
    augmented_text = F.delete_sentences(text_without_synonyms, deletion_prob, min_sentences)
    assert augmented_text == expected_text


@pytest.mark.parametrize(
    ["text", "is_same"],
    [
        ("text_with_synonyms", False),
        ("text_without_synonyms", True)
    ]
)
def test_insert_synonyms(text, is_same, request):
    text = request.getfixturevalue(text)
    insertion_prob = 1.0
    n_times = 1
    augmented_text = F.insert_synonyms(text, insertion_prob, n_times)
    assert (augmented_text == text) == is_same


@pytest.mark.parametrize(
    ["text", "is_same"],
    [
        ("text_with_synonyms", False),
        ("text_without_synonyms", True)
    ]
)
def test_replace_synonyms(text, is_same, request):
    text = request.getfixturevalue(text)
    replacement_prob = 1.0
    augmented_text = F.replace_synonyms(text, replacement_prob)
    assert (augmented_text == text) == is_same


def test_swap_words(text):
    original_sentences = split_text(text)
    n = len(original_sentences)
    n_times = 1
    augmented_text = F.swap_words(text, n_times)
    augmented_sentences = split_text(augmented_text)
    assert sum([original == augmented for original, augmented in zip(original_sentences, augmented_sentences)]) == n - 1


def test_swap_sentences(text):
    original_sentences = split_text(text)
    n = len(original_sentences)
    n_times = 1
    augmented_text = F.swap_sentences(text, n_times)
    augmented_sentences = split_text(augmented_text)
    assert sum([original == augmented for original, augmented in zip(original_sentences, augmented_sentences)]) == n - 2


def test_insert_punctuations(text_without_synonyms):
    insertion_prob = 1.0
    punctuations = (";",)
    augmented_text = F.insert_punctuations(text_without_synonyms, insertion_prob, punctuations)
    expected_text = "; 짜장면을 ; 맛있게 ; 먹었다. ; 짬뽕도 ; 맛있게 ; 먹었다. ; 짬짜면도 ; 먹고 ; 싶었다."
    assert augmented_text == expected_text
