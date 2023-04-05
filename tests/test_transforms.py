import pytest

from textmentations.augmentations.utils import extract_first_sentence


# TODO: SingleCorupsTypeTransform과 MultipleCorpusTypesTransform으로 나누어 test하기
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


@pytest.mark.parametrize("incorrect_unit", [0, "token"])
def test_incorrect_unit_value(augmentation_multiple, incorrect_unit):
    augmentation = augmentation_multiple
    params_names = augmentation.__init__.__code__.co_varnames
    probability_params = {param: incorrect_unit for param in params_names if param == "unit"}
    with pytest.raises(ValueError) as error_info:
        augmentation(**probability_params)
    assert "unit must be one of" in str(error_info.value)


@pytest.mark.parametrize("incorrect_n_times", [2j, 1.5, "0.0", None])
def test_incorrect_n_times_type(augmentation_with_n_times, incorrect_n_times):
    augmentation = augmentation_with_n_times
    params_names = augmentation.__init__.__code__.co_varnames
    n_times_params = {param: incorrect_n_times for param in params_names if param == "n_times"}
    with pytest.raises(TypeError) as error_info:
        augmentation(**n_times_params)
    assert str(error_info.value) == f"n_times must be a positive integer. Got: {type(incorrect_n_times)}"


@pytest.mark.parametrize("incorrect_n_times", [-1, 0])
def test_incorrect_n_times_value(augmentation_with_n_times, incorrect_n_times):
    augmentation = augmentation_with_n_times
    params_names = augmentation.__init__.__code__.co_varnames
    n_times_params = {param: incorrect_n_times for param in params_names if param == "n_times"}
    with pytest.raises(ValueError) as error_info:
        augmentation(**n_times_params)
    assert str(error_info.value) == f"n_times must be positive. Got: {incorrect_n_times}"


@pytest.mark.parametrize("incorrect_probability", [2j, "0.0", None])
def test_incorrect_probability_type(augmentation_with_probability, incorrect_probability):
    augmentation = augmentation_with_probability
    params_names = augmentation.__init__.__code__.co_varnames
    probability_params = {param: incorrect_probability for param in params_names if "prob" in param}
    with pytest.raises(TypeError) as error_info:
        augmentation(**probability_params)
    assert "must be a real number between 0 and 1" in str(error_info.value)


@pytest.mark.parametrize("incorrect_probability", [-1.0, 2])
def test_incorrect_probability_value(augmentation_with_probability, incorrect_probability):
    augmentation = augmentation_with_probability
    params_names = augmentation.__init__.__code__.co_varnames
    probability_params = {param: incorrect_probability for param in params_names if "prob" in param}
    with pytest.raises(ValueError) as error_info:
        augmentation(**probability_params)
    assert "must be between 0 and 1" in str(error_info.value)
