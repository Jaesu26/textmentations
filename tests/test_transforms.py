import pytest

from textmentations.augmentations.utils import extract_first_sentence


def test_empty_input_text(augmentation):
    text = ""
    augment = augmentation(p=1.0)
    data = augment(text=text)
    assert data["text"] == ""


@pytest.mark.parametrize("incorrect_probability", [2j, "0.0", None])
def test_incorrect_probability_type(augmentation_with_probability, incorrect_probability):
    augmentation = augmentation_with_probability
    param_names = augmentation.__init__.__code__.co_varnames
    probability_params = {name: incorrect_probability for name in param_names if 'prob' in name}
    with pytest.raises(TypeError) as error_info:
        augmentation(**probability_params)
    assert "must be a real number between 0 and 1." in str(error_info.value)


@pytest.mark.parametrize("incorrect_probability", [-1.0, 2])
def test_incorrect_probability_value(augmentation_with_probability, incorrect_probability):
    augmentation = augmentation_with_probability
    param_names = augmentation.__init__.__code__.co_varnames
    probability_params = {name: incorrect_probability for name in param_names if 'prob' in name}
    with pytest.raises(ValueError) as error_info:
        augmentation(**probability_params)
    assert "must be between 0 and 1" in str(error_info.value)


def test_ignore_first(text, augmentation):
    augment = augmentation(ignore_first=True, p=1.0)
    data = augment(text=text)
    assert extract_first_sentence(data["text"]) == extract_first_sentence(text)
