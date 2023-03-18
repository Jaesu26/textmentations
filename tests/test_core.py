import pytest

from textmentations.core.transforms_interface import TextTransform


@pytest.mark.parametrize("incorrect_ignore_first", [2j, "0.0", None])
def test_incorrect_ignore_first(text, incorrect_ignore_first):
    with pytest.raises(TypeError) as error_info:
        TextTransform(ignore_first=incorrect_ignore_first)
    assert str(error_info.value) == f"ignore_first must be boolean. Got: {type(incorrect_ignore_first)}"


@pytest.mark.parametrize("incorrect_always_apply", [2j, "0.0", None])
def test_incorrect_always_apply(text, incorrect_always_apply):
    with pytest.raises(TypeError) as error_info:
        TextTransform(always_apply=incorrect_always_apply)
    assert str(error_info.value) == f"always_apply must be boolean. Got: {type(incorrect_always_apply)}"


@pytest.mark.parametrize("incorrect_p", [2j, "0.0", None])
def test_incorrect_p_type(text, incorrect_p):
    with pytest.raises(TypeError) as error_info:
        TextTransform(p=incorrect_p)
    assert str(error_info.value) == f"p must be a real number between 0 and 1. Got: {type(incorrect_p)}"


@pytest.mark.parametrize("incorrect_p", [-1.0, 2])
def test_incorrect_p_value(text, incorrect_p):
    with pytest.raises(ValueError) as error_info:
        TextTransform(p=incorrect_p)
    assert str(error_info.value) == f"p must be between 0 and 1. Got: {incorrect_p}"
