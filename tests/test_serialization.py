import albumentations as A
import pytest

from .utils import get_units, set_seed


@pytest.mark.parametrize("seed", [23, 42])
@pytest.mark.parametrize("ignore_first", [False, True])
@pytest.mark.parametrize("always_apply", [False, True])
@pytest.mark.parametrize("p", [0.5, 1.0])
def test_augmentation_single_serialization(
    long_text,
    augmentation_single,
    seed,
    ignore_first,
    always_apply,
    p,
):
    augment = augmentation_single(ignore_first=ignore_first, always_apply=always_apply, p=p)
    _test_augmentation_serialization(long_text, augment, seed)


def _test_augmentation_serialization(input_text, augment, seed):
    serialized_augment = A.to_dict(augment)
    deserialized_augment = A.from_dict(serialized_augment)
    set_seed(seed)
    augmented_data = augment(text=input_text)
    set_seed(seed)
    deserialized_augmented_data = deserialized_augment(text=input_text)
    assert augmented_data["text"] == deserialized_augmented_data["text"]


@pytest.mark.parametrize("seed", [23, 42])
@pytest.mark.parametrize("ignore_first", [False, True])
@pytest.mark.parametrize("always_apply", [False, True])
@pytest.mark.parametrize("p", [0.5, 1.0])
def test_augmentation_multiple_serialization(
    long_text,
    augmentation_multiple,
    seed,
    ignore_first,
    always_apply,
    p,
):
    units = get_units(augmentation_multiple)
    for unit in units:
        augment = augmentation_multiple(unit=unit, ignore_first=ignore_first, always_apply=always_apply, p=p)
        _test_augmentation_serialization(long_text, augment, seed)
