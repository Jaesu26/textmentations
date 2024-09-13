import pytest

import textmentations.augmentations.generation.functional as fg
import textmentations.augmentations.modification.functional as fm
from textmentations.augmentations.generation.transforms import _albert_model, _albert_tokenizer
from textmentations.augmentations.utils import split_text_into_sentences


def test_back_translate():
    text = "나는 목이 말라서 물을 마셨다."
    augmented_text = fg.back_translate(text, from_lang="ko", to_lang="en")
    assert augmented_text != text


@pytest.mark.parametrize(
    ["deletion_prob", "min_words_per_sentence", "expected_text"],
    [
        (0.0, 0.5, "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."),
        (False, 1, "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."),
        (1.0, 0.0, ""),
        (1.0, 1.0, "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."),
        (1.0, 0, ""),
        (True, 3, "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."),
    ],
)
def test_delete_words(text_without_synonyms, deletion_prob, min_words_per_sentence, expected_text):
    augmented_text = fm.delete_words(text_without_synonyms, deletion_prob, min_words_per_sentence)
    assert augmented_text == expected_text


@pytest.mark.parametrize(
    ["deletion_prob", "min_sentences", "expected_text"],
    [
        (0.0, 0.5, "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."),
        (False, 1, "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."),
        (1.0, 0.0, ""),
        (1.0, 1.0, "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."),
        (1.0, 0, ""),
        (True, 3, "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."),
    ],
)
def test_delete_sentences(text_without_synonyms, deletion_prob, min_sentences, expected_text):
    augmented_text = fm.delete_sentences(text_without_synonyms, deletion_prob, min_sentences)
    assert augmented_text == expected_text


@pytest.mark.parametrize(["text", "is_same"], [("text_with_synonyms", False), ("text_without_synonyms", True)])
def test_insert_synonyms(text, is_same, request):
    text = request.getfixturevalue(text)
    insertion_prob = 1.0
    n_times = 1
    augmented_text = fm.insert_synonyms(text, insertion_prob, n_times)
    assert (augmented_text == text) == is_same


def test_insert_punctuation(text_without_synonyms):
    insertion_prob = 1.0
    punctuation = (";",)
    augmented_text = fm.insert_punctuation(text_without_synonyms, insertion_prob, punctuation)
    expected_text = "; 짜장면을 ; 맛있게 ; 먹었다. ; 짬뽕도 ; 맛있게 ; 먹었다. ; 짬짜면도 ; 먹고 ; 싶었다."
    assert augmented_text == expected_text


def test_iterative_mask_fill(text):
    original_sentences = split_text_into_sentences(text)
    augmented_text = fg.iterative_mask_fill(
        text, model=_albert_model, tokenizer=_albert_tokenizer, top_k=5, device="cpu"
    )
    augmented_sentences = split_text_into_sentences(augmented_text)
    assert sum([original != augmented for original, augmented in zip(original_sentences, augmented_sentences)]) == 1


@pytest.mark.parametrize(["text", "is_same"], [("text_with_synonyms", False), ("text_without_synonyms", True)])
def test_replace_synonyms(text, is_same, request):
    text = request.getfixturevalue(text)
    replacement_prob = 1.0
    augmented_text = fm.replace_synonyms(text, replacement_prob)
    assert (augmented_text == text) == is_same


def test_swap_words(text):
    original_sentences = split_text_into_sentences(text)
    alpha = 0.01
    augmented_text = fm.swap_words(text, alpha)
    augmented_sentences = split_text_into_sentences(augmented_text)
    assert sum([original != augmented for original, augmented in zip(original_sentences, augmented_sentences)]) == 1


def test_swap_sentences(text):
    original_sentences = split_text_into_sentences(text)
    n = len(original_sentences)
    assert n >= 2
    n_times = 1
    augmented_text = fm.swap_sentences(text, n_times)
    augmented_sentences = split_text_into_sentences(augmented_text)
    assert sum([original != augmented for original, augmented in zip(original_sentences, augmented_sentences)]) == 2
