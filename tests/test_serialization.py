import albumentations as A
import pytest

from .utils import get_units, set_seed


@pytest.mark.parametrize("seed", [23, 42])
@pytest.mark.parametrize("ignore_first", [False, True])
@pytest.mark.parametrize("always_apply", [False, True])
@pytest.mark.parametrize("p", [0.5, 1.0])
def test_augmentation_serialization(
    long_text,
    augmentation,
    seed,
    ignore_first,
    always_apply,
    p,
):
    if hasattr(augmentation, "_units"):
        units = get_units(augmentation)
        for unit in units:
            augment = augmentation(unit=unit, ignore_first=ignore_first, always_apply=always_apply, p=p)
            _test_augmentation_serialization(long_text, augment, seed)
    else:
        augment = augmentation(ignore_first=ignore_first, always_apply=always_apply, p=p)
        _test_augmentation_serialization(long_text, augment, seed)


def _test_augmentation_serialization(text, augment, seed):
    serialized_augment = A.to_dict(augment)
    deserialized_augment = A.from_dict(serialized_augment)
    set_seed(seed)
    augmented_data = augment(text=text)
    set_seed(seed)
    deserialized_augmented_data = deserialized_augment(text=text)
    assert augmented_data["text"] == deserialized_augmented_data["text"]
