import pytest

from textmentations import AEDA
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
    params_names = augmentation_with_n_times.__init__.__code__.co_varnames
    n_times_params = {param: incorrect_n_times for param in params_names if param == "n_times"}
    expected_message = f"n_times must be a positive integer. Got: {type(incorrect_n_times)}"
    with pytest.raises(TypeError, match=expected_message):
        augmentation_with_n_times(**n_times_params)


@pytest.mark.parametrize("incorrect_n_times", [-1, 0])
def test_incorrect_n_times_value(augmentation_with_n_times, incorrect_n_times):
    params_names = augmentation_with_n_times.__init__.__code__.co_varnames
    n_times_params = {param: incorrect_n_times for param in params_names if param == "n_times"}
    expected_message = f"n_times must be positive. Got: {incorrect_n_times}"
    with pytest.raises(ValueError, match=expected_message):
        augmentation_with_n_times(**n_times_params)


@pytest.mark.parametrize("incorrect_probability", [2j, "0.0", None])
def test_incorrect_probability_type(augmentation_with_probability, incorrect_probability):
    params_names = augmentation_with_probability.__init__.__code__.co_varnames
    probability_params = {param: incorrect_probability for param in params_names if "prob" in param}
    expected_message = "must be a real number between 0 and 1."
    with pytest.raises(TypeError, match=expected_message):
        augmentation_with_probability(**probability_params)


@pytest.mark.parametrize("incorrect_probability", [-1.0, 2])
def test_incorrect_probability_value(augmentation_with_probability, incorrect_probability):
    params_names = augmentation_with_probability.__init__.__code__.co_varnames
    probability_params = {param: incorrect_probability for param in params_names if "prob" in param}
    expected_message = "must be between 0 and 1."
    with pytest.raises(ValueError, match=expected_message):
        augmentation_with_probability(**probability_params)


@pytest.mark.parametrize("incorrect_punctuation", [0, ",", ["."], (), (",", ":", None)])
def test_incorrect_punctuation_type(incorrect_punctuation):
    with pytest.raises(TypeError):
        AEDA(punctuation=incorrect_punctuation)


def test_aeda_deprecation_warning():
    expected_message = (
        "punctuations is deprecated. Use `punctuation` instead. self.punctuation will be set to punctuations."
    )
    with pytest.warns(DeprecationWarning, match=expected_message):
        punctuation = (".", ",")
        AEDA(punctuations=punctuation)
