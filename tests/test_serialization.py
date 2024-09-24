import albumentations as A
import pytest
from albumentations.core.serialization import SERIALIZABLE_REGISTRY

from tests.conftest import AUGMENTATIONS
from tests.utils import set_seed
from textmentations import Compose


def test_class_fullname(augmentation):
    class_fullname = augmentation.get_class_fullname()
    assert class_fullname == augmentation.__name__


def test_to_dict(augmentation):
    augment = augmentation()
    serialized_dict = augment.to_dict()
    assert serialized_dict["__version__"] == "1.3.2"


def test_serializable_registry(augmentation):
    class_fullname = augmentation.__name__
    assert class_fullname in SERIALIZABLE_REGISTRY


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("ignore_first", [False, True])
@pytest.mark.parametrize("p", [0.5, 1.0])
def test_augmentations_serialization(long_text, augmentation, seed, ignore_first, p):
    augment = augmentation(ignore_first=ignore_first, p=p)
    serialized_dict = A.to_dict(augment)
    deserialized_augment = A.from_dict(serialized_dict)
    set_seed(seed)
    augmented_data = augment(text=long_text)
    set_seed(seed)
    deserialized_augmented_data = deserialized_augment(text=long_text)
    assert augmented_data["text"] == deserialized_augmented_data["text"]


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("ignore_first", [False, True])
@pytest.mark.parametrize("p", [0.5, 1.0])
def test_compose_serialization(long_text, seed, ignore_first, p):
    augment = Compose([aug(ignore_first=ignore_first, p=p) for aug in AUGMENTATIONS])
    serialized_dict = A.to_dict(augment)
    deserialized_augment = A.from_dict(serialized_dict)
    set_seed(seed)
    augmented_data = augment(text=long_text)
    set_seed(seed)
    deserialized_augmented_data = deserialized_augment(text=long_text)
    assert augmented_data["text"] == deserialized_augmented_data["text"]
