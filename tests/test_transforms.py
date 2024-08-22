import re

import pytest

from textmentations import AEDA, RandomDeletion
from textmentations.augmentations.utils import extract_first_sentence


@pytest.mark.parametrize("ignore_first", [False, True])
def test_empty_input_text(augmentation, ignore_first):
    text = ""
    augment = augmentation(ignore_first=ignore_first, p=1.0)
    data = augment(text=text)
    assert data["text"] == ""


def test_ignore_first(text, augmentation):
    augment = augmentation(ignore_first=True, p=1.0)
    data = augment(text=text)
    assert extract_first_sentence(data["text"]) == extract_first_sentence(text)


@pytest.mark.parametrize("incorrect_n_times", [2j, 1.5, "0.0", None])
def test_incorrect_n_times_type(augmentation_with_n_times, incorrect_n_times):
    code_object = augmentation_with_n_times.__init__.__code__
    params_names = code_object.co_varnames[: code_object.co_argcount]
    n_times_params = {param: incorrect_n_times for param in params_names if param == "n_times"}
    expected_message = f"n_times must be a positive integer. Got: {type(incorrect_n_times)}"
    with pytest.raises(TypeError, match=expected_message):
        augmentation_with_n_times(**n_times_params)


@pytest.mark.parametrize("incorrect_n_times", [-1, 0])
def test_incorrect_n_times_value(augmentation_with_n_times, incorrect_n_times):
    code_object = augmentation_with_n_times.__init__.__code__
    params_names = code_object.co_varnames[: code_object.co_argcount]
    n_times_params = {param: incorrect_n_times for param in params_names if param == "n_times"}
    expected_message = f"n_times must be positive. Got: {incorrect_n_times}"
    with pytest.raises(ValueError, match=expected_message):
        augmentation_with_n_times(**n_times_params)


@pytest.mark.parametrize("incorrect_probability", [2j, "0.0", None])
def test_incorrect_probability_type(augmentation_with_probability, incorrect_probability):
    code_object = augmentation_with_probability.__init__.__code__
    params_names = code_object.co_varnames[: code_object.co_argcount]
    probability_params = {param: incorrect_probability for param in params_names if "prob" in param}
    expected_message = "must be a real number between 0 and 1."
    with pytest.raises(TypeError, match=expected_message):
        augmentation_with_probability(**probability_params)


@pytest.mark.parametrize("incorrect_probability", [-1.0, 2])
def test_incorrect_probability_value(augmentation_with_probability, incorrect_probability):
    code_object = augmentation_with_probability.__init__.__code__
    params_names = code_object.co_varnames[: code_object.co_argcount]
    probability_params = {param: incorrect_probability for param in params_names if "prob" in param}
    expected_message = "must be between 0 and 1."
    with pytest.raises(ValueError, match=expected_message):
        augmentation_with_probability(**probability_params)


@pytest.mark.parametrize("incorrect_punctuation", [0, ",", ["."], (), (",", ":", None)])
def test_incorrect_punctuation_type(incorrect_punctuation):
    with pytest.raises(TypeError):
        AEDA(punctuation=incorrect_punctuation)


def test_random_deletion_deprecation_warning():
    min_words_per_sentence = 0.2
    expected_message = (
        "min_words_each_sentence is deprecated. Use `min_words_per_sentence` instead."
        " self.min_words_per_sentence will be set to min_words_each_sentence."
    )
    with pytest.warns(DeprecationWarning, match=expected_message):
        rd = RandomDeletion(min_words_each_sentence=min_words_per_sentence)
    assert rd.min_words_per_sentence == min_words_per_sentence


def test_aeda_punctuations_deprecation_warning():
    punctuation = (".", ",", "123")
    expected_message = (
        "punctuations is deprecated. Use `punctuation` instead. self.punctuation will be set to punctuations."
    )
    with pytest.warns(DeprecationWarning, match=expected_message):
        aeda = AEDA(punctuations=punctuation)
    assert aeda.punctuation == punctuation


def test_aeda_insertion_prob_limit_deprecation_warning():
    insertion_prob_range = (0.0, 0.3)
    expected_message = (
        "insertion_prob_limit is deprecated."
        " Use `insertion_prob_range` as tuple (lower limit, insertion_prob_limit) instead."
        " self.insertion_prob_range will be set to insertion_prob_limit."
    )
    with pytest.warns(DeprecationWarning, match=re.escape(expected_message)):
        aeda = AEDA(insertion_prob_limit=insertion_prob_range)
    assert aeda.insertion_prob_range == insertion_prob_range


def test_aeda_insertion_prob_limit_non_tuple_deprecation_warning():
    insertion_prob_range = 0.3
    with pytest.warns(DeprecationWarning) as record:
        AEDA(insertion_prob_limit=insertion_prob_range)
    assert len(record) == 2


def test_aeda_insertion_prob_range_non_tuple_deprecation_warning():
    insertion_prob_range = 0.3
    expected_message = (
        "insertion_prob_range is should be a tuple with length 2."
        " The provided value will be automatically converted to a tuple (0, insertion_prob_range)."
    )
    with pytest.warns(DeprecationWarning, match=re.escape(expected_message)):
        aeda = AEDA(insertion_prob_range=insertion_prob_range)
    assert aeda.insertion_prob_range == (0.0, insertion_prob_range)
