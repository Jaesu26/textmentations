import pytest

import textmentations as T
from textmentations.augmentations.utils import extract_first_sentence


@pytest.mark.parametrize(
    "augmentation",
    [
        T.RandomDeletionWords,
        T.RandomDeletionSentences,
        T.RandomInsertion,
        T.SynonymsReplacement,
    ]
)
def test_incorrect_probability(augmentation):
    incorrect_probability = -1.0
    param_names = augmentation.__init__.__code__.co_varnames
    probability_params = {name: incorrect_probability for name in param_names if 'probability' in name}
    with pytest.raises(ValueError) as error_info:
        augmentation(**probability_params)
    assert "must be between 0 and 1" in str(error_info.value)


def test_empty_input_text(augmentation):
    text = ""
    augment = augmentation(p=1.0)
    data = augment(text=text)
    assert data["text"] == ""


def test_ignore_first(text, augmentation):
    augment = augmentation(ignore_first=True, p=1.0)
    data = augment(text=text)
    assert extract_first_sentence(data["text"]) == extract_first_sentence(text)
