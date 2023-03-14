import textmentations as T
import textmentations.augmentations.functional as F
from textmentations.augmentations.utils import split_text


def test_random_deletion_words():
    text = "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."
    expected_text = "먹었다. 먹었다. 싶었다."
    deletion_probability = 1.0
    min_words_each_sentence = 1

    augment = T.RandomDeletionWords(deletion_probability, min_words_each_sentence, p=1.0)
    data = augment(text=text)
    augmented_text = F.delete_words(text, deletion_probability, min_words_each_sentence)
    assert augmented_text == data["text"]
    assert augmented_text == expected_text


def test_random_deletion_sentences():
    text = "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."
    expected_text = "짬짜면도 먹고 싶었다."
    deletion_probability = 1.0
    min_sentences = 1

    augment = T.RandomDeletionSentences(deletion_probability, min_sentences, p=1.0)
    data = augment(text=text)
    augmented_text = F.delete_sentences(text, deletion_probability, min_sentences)
    assert augmented_text == data["text"]
    assert augmented_text == expected_text


def test_random_insertion():
    text = "물 한잔만 주세요."
    insertion_probability = 1.0
    n_times = 1

    augment = T.RandomInsertion(insertion_probability, n_times, p=1.0)
    data = augment(text=text)
    assert data["text"] != text


def test_random_swap_words():
    text = "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."
    original_sentences = split_text(text)
    n = len(original_sentences)
    n_times = 1

    augment = T.RandomSwapWords(n_times, p=1.0)
    data = augment(text=text)
    augmented_sentences = split_text(data["text"])
    assert sum([original == augment for original, augment in zip(original_sentences, augmented_sentences)]) == n - 1


def test_random_swap_sentences():
    text = "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."
    original_sentences = split_text(text)
    n = len(original_sentences)
    n_times = 1

    augment = T.RandomSwapSentences(n_times, p=1.0)
    data = augment(text=text)
    augmented_sentences = split_text(data["text"])
    assert sum([original == augment for original, augment in zip(original_sentences, augmented_sentences)]) == n - 2


def test_synonyms_replacement():
    text = "물 한잔만 주세요."
    replacement_probability = 1.0

    augment = T.SynonymsReplacement(replacement_probability, p=1.0)
    data = augment(text=text)
    assert data["text"] != text
