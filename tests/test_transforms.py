import re

import pytest
from deep_translator.exceptions import LanguageNotSupportedException

from textmentations import AEDA, BackTranslation, RandomDeletion, RandomDeletionSentence
from textmentations.augmentations.transforms import LANGUAGES
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


@pytest.mark.parametrize(
    ["input_text", "ignore_first", "min_sentences", "expected_min_sentences"],
    [
        ("짬짜면도 먹고 싶었다.", True, 0, 0),
        ("짬짜면도 먹고 싶었다.", False, 0, 0),
        ("짬짜면도 먹고 싶었다.", True, 1, 0),
        ("짬짜면도 먹고 싶었다.", False, 1, 1),
        ("짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다.", False, 0.5, 0.5),
        ("", True, 0.5, 0.5),
        ("짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다.", True, 0.33, 0.33),
        ("짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다.", True, 0.5, 0.25),
    ],
)
def test_get_params_dependent_on_data_of_random_deletion_sentence(
    input_text, ignore_first, min_sentences, expected_min_sentences
):
    params = {}
    data = {"text": input_text}
    rds = RandomDeletionSentence(min_sentences=min_sentences, ignore_first=ignore_first)
    params_dependent_on_data = rds.get_params_dependent_on_data(params=params, data=data)
    assert params_dependent_on_data["min_sentences"] == expected_min_sentences


def test_albumentations_compatibility(text):
    rds1 = RandomDeletionSentence(min_sentences=3, ignore_first=True, p=1.0)
    rds2 = RandomDeletionSentence(min_sentences=0.1, ignore_first=True, p=1.0)
    rds1(text=text)
    rds2(text=text)


@pytest.mark.parametrize(
    "incorrect_lang_param",
    [
        {"from_lang": ""},
        {"from_lang": "123"},
        {"from_lang": "korean"},
        {"to_lang": ""},
        {"to_lang": "kor"},
        {"to_lang": "english"},
    ],
)
def test_incorrect_language(incorrect_lang_param):
    expected_message = f"must be one of {LANGUAGES}."
    with pytest.raises(LanguageNotSupportedException, match=re.escape(expected_message)):
        BackTranslation(**incorrect_lang_param)


@pytest.mark.parametrize(
    ["cls", "incorrect_minimum_param"],
    [
        (RandomDeletion, {"min_words_per_sentence": True}),
        (RandomDeletion, {"min_words_per_sentence": "1"}),
        (RandomDeletionSentence, {"min_sentences": False}),
        (RandomDeletionSentence, {"min_sentences": "0.5"}),
    ],
)
def test_incorrect_minimum_type(cls, incorrect_minimum_param):
    with pytest.raises(TypeError):
        cls(**incorrect_minimum_param)


@pytest.mark.parametrize(
    ["cls", "incorrect_minimum_param"],
    [
        (RandomDeletion, {"min_words_per_sentence": 1.5}),
        (RandomDeletion, {"min_words_per_sentence": -2}),
        (RandomDeletionSentence, {"min_sentences": -0.8}),
        (RandomDeletionSentence, {"min_sentences": -1}),
    ],
)
def test_incorrect_minimum_value(cls, incorrect_minimum_param):
    with pytest.raises(ValueError):
        cls(**incorrect_minimum_param)


@pytest.mark.parametrize("incorrect_n_times", [2j, 1.5, "0.0", None])
def test_incorrect_n_times_type(augmentation_with_n_times, incorrect_n_times):
    code_object = augmentation_with_n_times.__init__.__code__
    param_names = code_object.co_varnames[: code_object.co_argcount]
    incorrect_n_times_param = {name: incorrect_n_times for name in param_names if name == "n_times"}
    expected_message = f"n_times must be a positive integer. Got: {type(incorrect_n_times)}"
    with pytest.raises(TypeError, match=expected_message):
        augmentation_with_n_times(**incorrect_n_times_param)


@pytest.mark.parametrize("incorrect_n_times", [-1, 0])
def test_incorrect_n_times_value(augmentation_with_n_times, incorrect_n_times):
    code_object = augmentation_with_n_times.__init__.__code__
    param_names = code_object.co_varnames[: code_object.co_argcount]
    incorrect_n_times_param = {name: incorrect_n_times for name in param_names if name == "n_times"}
    expected_message = f"n_times must be positive. Got: {incorrect_n_times}"
    with pytest.raises(ValueError, match=expected_message):
        augmentation_with_n_times(**incorrect_n_times_param)


@pytest.mark.parametrize("incorrect_probability", [2j, "0.0", None])
def test_incorrect_probability_type(augmentation_with_probability, incorrect_probability):
    code_object = augmentation_with_probability.__init__.__code__
    param_names = code_object.co_varnames[: code_object.co_argcount]
    incorrect_probability_param = {name: incorrect_probability for name in param_names if name.endswith("prob")}
    expected_message = "must be a real number between 0 and 1."
    with pytest.raises(TypeError, match=expected_message):
        augmentation_with_probability(**incorrect_probability_param)


@pytest.mark.parametrize("incorrect_probability", [-1.0, 2])
def test_incorrect_probability_value(augmentation_with_probability, incorrect_probability):
    code_object = augmentation_with_probability.__init__.__code__
    param_names = code_object.co_varnames[: code_object.co_argcount]
    incorrect_probability_param = {name: incorrect_probability for name in param_names if name.endswith("prob")}
    expected_message = "must be between 0 and 1."
    with pytest.raises(ValueError, match=expected_message):
        augmentation_with_probability(**incorrect_probability_param)


@pytest.mark.parametrize("incorrect_prob_range", [("0.0", "0.5"), None])
def test_incorrect_prob_range_type(augmentation_with_prob_range, incorrect_prob_range):
    code_object = augmentation_with_prob_range.__init__.__code__
    param_names = code_object.co_varnames[: code_object.co_argcount]
    incorrect_prob_range_param = {name: incorrect_prob_range for name in param_names if name.endswith("prob_range")}
    with pytest.raises(TypeError):
        augmentation_with_prob_range(**incorrect_prob_range_param)


@pytest.mark.parametrize(
    "incorrect_prob_range",
    [(0, 0.5, 1), (True, False), (0.3, 0.1), (-1, 0.5), (0.2, 1.5), (-1, 1.5)],
)
def test_incorrect_prob_range_value(augmentation_with_prob_range, incorrect_prob_range):
    code_object = augmentation_with_prob_range.__init__.__code__
    param_names = code_object.co_varnames[: code_object.co_argcount]
    incorrect_prob_range_param = {name: incorrect_prob_range for name in param_names if name.endswith("prob_range")}
    with pytest.raises(ValueError):
        augmentation_with_prob_range(**incorrect_prob_range_param)


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
