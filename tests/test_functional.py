import textmentations.augmentations.functional as F
from textmentations.augmentations.utils import split_text


def test_delete_words():
    text = "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."
    expected_text = "먹었다. 먹었다. 싶었다."
    deletion_probability = 1.0
    min_words_each_sentence = 1
    augmented_text = F.delete_words(text, deletion_probability, min_words_each_sentence)
    assert augmented_text == expected_text


def test_delete_sentences():
    text = "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."
    expected_text = "짬짜면도 먹고 싶었다."
    deletion_probability = 1.0
    min_sentences = 1
    augmented_text = F.delete_sentences(text, deletion_probability, min_sentences)
    assert augmented_text == expected_text


def test_insert_synonyms():
    text = "물 한잔만 주세요."
    insertion_probability = 1.0
    n_times = 1
    augmented_text = F.insert_synonyms(text, insertion_probability, n_times)
    assert augmented_text != text


def test_replace_synonyms():
    text = "물 한잔만 주세요."
    replacement_probability = 1.0
    augmented_text = F.replace_synonyms(text, replacement_probability)
    assert augmented_text != text


def test_swap_words():
    text = "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."
    original_sentences = split_text(text)
    n = len(original_sentences)
    n_times = 1

    augmented_text = F.swap_words(text, n_times)
    augmented_sentences = split_text(augmented_text)
    assert sum([original == augment for original, augment in zip(original_sentences, augmented_sentences)]) == n - 1


def test_swap_sentences():
    text = "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."
    original_sentences = split_text(text)
    n = len(original_sentences)
    n_times = 1

    augmented_text = F.swap_sentences(text, n_times)
    augmented_sentences = split_text(augmented_text)
    assert sum([original == augment for original, augment in zip(original_sentences, augmented_sentences)]) == n - 2
