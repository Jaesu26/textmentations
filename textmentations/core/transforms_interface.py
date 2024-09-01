from __future__ import annotations

from collections.abc import Callable
from typing import Any, NoReturn
from warnings import warn

from albumentations.core.transforms_interface import BasicTransform

from textmentations import __version__

from ..corpora.types import Text
from .utils import (
    extract_first_sentence_by_key,
    get_shortest_class_fullname,
    remove_first_sentence_by_key,
    wrap_text_with_first_sentence_by_key,
)


class TextTransform(BasicTransform):
    """Transform applied to text.

    Args:
        ignore_first: Whether to ignore the first sentence when applying this transform.
            It is useful when the main idea of the text is expressed in the first sentence.
        always_apply: Whether to always apply this transform.
        p: The probability of applying this transform.
    """

    def __init__(self, ignore_first: bool = False, always_apply: bool | None = None, p: float = 0.5) -> None:
        if always_apply is not None:
            if not isinstance(always_apply, bool):
                raise TypeError(f"always_apply must be boolean. Got: {type(always_apply)}")
            if always_apply:
                warn(
                    "always_apply is deprecated. Use `p=1` if you want to always apply the transform."
                    " self.p will be set to 1.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                p = 1.0
            else:
                warn("always_apply is deprecated.", DeprecationWarning, stacklevel=2)
        self._validate_base_init_args(ignore_first=ignore_first, p=p)
        self.ignore_first = ignore_first
        super().__init__(p=p)

    def _validate_base_init_args(self, *, ignore_first: bool, p: float) -> None:
        if not isinstance(ignore_first, bool):
            raise TypeError(f"ignore_first must be boolean. Got: {type(ignore_first)}")
        if not isinstance(p, (float, int)):
            raise TypeError(f"p must be a real number between 0 and 1. Got: {type(p)}")
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be between 0 and 1. Got: {p}")

    def __call__(self, *args: Any, force_apply: bool = False, **kwargs: Text) -> dict[str, Text]:
        if args:
            raise KeyError("You have to pass data to augmentations as named arguments, for example: augment(text=text)")
        if not all(isinstance(text, str) for _, text in kwargs.items()):
            raise TypeError("You have to pass string data to augmentations.")
        if not self.ignore_first:
            return super().__call__(force_apply=force_apply, **kwargs)
        return self.apply_without_first(force_apply=force_apply, **kwargs)

    def apply_without_first(self, force_apply: bool = False, **kwargs: Text) -> dict[str, Text]:
        key2first_sentence = extract_first_sentence_by_key(kwargs)
        key2text_without_first_sentence = remove_first_sentence_by_key(kwargs)
        key2augmented_text_without_first_sentence = super().__call__(
            force_apply=force_apply, **key2text_without_first_sentence
        )
        key2augmented_text = wrap_text_with_first_sentence_by_key(
            key2augmented_text_without_first_sentence, key2first_sentence
        )
        return key2augmented_text

    def apply(self, text: Text, *args: Any, **params: Any) -> Text:
        raise NotImplementedError

    @property
    def targets(self) -> dict[str, Callable[..., Text]]:
        return {"text": self.apply}

    def update_params_shape(self, params: dict[str, Any], data: dict[str, Text]) -> dict[str, Any]:
        return params

    def update_params(self, params: dict[str, Any], **kwargs: Text) -> dict[str, Any]:
        return params

    def add_targets(self, additional_targets: dict[str, str]) -> NoReturn:
        raise AttributeError("add_targets method is not available in TextTransform as it is not applicable to text.")

    def get_params_dependent_on_targets(self, params: dict[str, Text]) -> dict[str, Any]:
        return {}

    @classmethod
    def get_class_fullname(cls) -> str:
        return get_shortest_class_fullname(cls)

    def get_base_init_args(self) -> dict[str, Any]:
        return {"ignore_first": self.ignore_first, **super().get_base_init_args()}

    def to_dict(self, on_not_implemented_error: str = "raise") -> dict[str, Any]:
        serialized_dict = super().to_dict(on_not_implemented_error)
        serialized_dict["__version__"] = __version__
        return serialized_dict
