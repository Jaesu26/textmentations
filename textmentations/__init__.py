from .augmentations.transforms import (
    AEDA,
    BackTranslation,
    RandomDeletion,
    RandomDeletionSentence,
    RandomInsertion,
    RandomSwap,
    RandomSwapSentence,
    SynonymReplacement,
)
from .core.composition import BaseCompose, Compose, OneOf, OneOrOther, ReplayCompose, Sequential, SomeOf
from .core.transforms_interface import TextTransform

__version__ = "1.2.0"

__all__ = [
    "AEDA",
    "BackTranslation",
    "RandomDeletion",
    "RandomDeletionSentence",
    "RandomInsertion",
    "RandomSwap",
    "RandomSwapSentence",
    "SynonymReplacement",
    "BaseCompose",
    "Compose",
    "OneOf",
    "OneOrOther",
    "ReplayCompose",
    "Sequential",
    "SomeOf",
    "TextTransform",
]
