import albumentations as A
import pytest

from .utils import set_seed


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("ignore_first", [False, True])
@pytest.mark.parametrize("always_apply", [False, True])
@pytest.mark.parametrize("p", [0.5, 1.0])
def test_augmentations_serialization(
    long_text,
    augmentation,
    seed,
    ignore_first,
    always_apply,
    p,
):
    augment = augmentation(ignore_first=ignore_first, always_apply=always_apply, p=p)
    serialized_augment = A.to_dict(augment)
    deserialized_augment = A.from_dict(serialized_augment)
    set_seed(seed)
    augmented_data = augment(text=long_text)
    set_seed(seed)
    deserialized_augmented_data = deserialized_augment(text=long_text)
    assert augmented_data["text"] == deserialized_augmented_data["text"]
