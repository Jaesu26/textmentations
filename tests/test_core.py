import re

import pytest

import textmentations as T
from textmentations.core.transforms_interface import TextTransform


@pytest.mark.parametrize("incorrect_ignore_first", [2j, 1.0, "0.0", None])
def test_incorrect_ignore_first_type(text, incorrect_ignore_first):
    expected_message = f"ignore_first must be boolean. Got: {type(incorrect_ignore_first)}"
    with pytest.raises(TypeError, match=expected_message):
        TextTransform(ignore_first=incorrect_ignore_first)


@pytest.mark.parametrize("incorrect_always_apply", [2j, 1.0, "0.0", None])
def test_incorrect_always_apply_type(text, incorrect_always_apply):
    expected_message = f"always_apply must be boolean. Got: {type(incorrect_always_apply)}"
    with pytest.raises(TypeError, match=expected_message):
        TextTransform(always_apply=incorrect_always_apply)


@pytest.mark.parametrize("incorrect_p", [2j, "0.0", None])
def test_incorrect_p_type(text, incorrect_p):
    expected_message = f"p must be a real number between 0 and 1. Got: {type(incorrect_p)}"
    with pytest.raises(TypeError, match=expected_message):
        TextTransform(p=incorrect_p)


@pytest.mark.parametrize("incorrect_p", [-1.0, 2])
def test_incorrect_p_value(text, incorrect_p):
    expected_message = f"p must be between 0 and 1. Got: {incorrect_p}"
    with pytest.raises(ValueError, match=expected_message):
        TextTransform(p=incorrect_p)


def test_named_args():
    text = "짜장면을 맛있게 먹었다."
    augment = T.RandomSwap(p=1.0)
    expected_message = "You have to pass data to augmentations as named arguments, for example: augment(text=text)"
    with pytest.raises(KeyError, match=re.escape(expected_message)):
        augment(text)


def test_incorrect_target_type():
    incorrect_text = 123456789
    augment = T.RandomSwap(p=1.0)
    expected_message = "You have to pass string data to augmentations."
    with pytest.raises(TypeError, match=expected_message):
        augment(text=incorrect_text)
