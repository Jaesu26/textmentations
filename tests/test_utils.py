import pytest

from textmentations.augmentations.utils import clear_double_hash_tokens


@pytest.mark.parametrize(
    ["input_text", "expected_text"],
    [
        ("짬뽕을 맛있게 먹었다.", "짬뽕을 맛있게 먹었다."),
        ("짬뽕 ##을 맛있게 먹었다.", "짬뽕을 맛있게 먹었다."),
        ("짬뽕 ####을 맛있게 먹었다.", "짬뽕 ##을 맛있게 먹었다."),
        ("## 짬뽕을 맛있##  ##게 먹었##다.", "## 짬뽕을 맛있##게 먹었다."),
    ],
)
def test_clear_double_hash_tokens(input_text, expected_text):
    assert clear_double_hash_tokens(input_text) == expected_text
