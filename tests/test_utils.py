import pytest

from textmentations.augmentations.utils import clear_double_hash_tokens, extract_nth_sentence, remove_nth_sentence


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


@pytest.mark.parametrize(
    ["input_text", "n", "expected_text"],
    [
        ("짬뽕을. 맛있게. 먹었다.", 0, "짬뽕을"),
        ("짬뽕을. 맛있게. 먹었다.", -1, "먹었다"),
        ("짬뽕을. 맛있게. 먹었다.", 3, ""),
    ],
)
def test_extract_nth_sentence(input_text, n, expected_text):
    assert extract_nth_sentence(input_text, n) == expected_text


@pytest.mark.parametrize(
    ["input_text", "n", "expected_text"],
    [
        ("짬뽕을. 맛있게. 먹었다.", 0, "맛있게. 먹었다."),
        ("짬뽕을. 맛있게. 먹었다.", -1, "짬뽕을. 맛있게."),
        ("짬뽕을. 맛있게. 먹었다.", 3, "짬뽕을. 맛있게. 먹었다."),
    ],
)
def test_remove_nth_sentence(input_text, n, expected_text):
    assert remove_nth_sentence(input_text, n) == expected_text
