from typing import Any, Callable, Dict

from albumentations.core.transforms_interface import BasicTransform

from ..corpora.corpus_types import Text
from .utils import (
    get_first_sentences_from_kwargs,
    remove_first_sentences_from_kwargs,
    combine_augmented_kwargs_with_first_sentences
)

__all__ = [
    "TextTransform",
]


class TextTransform(BasicTransform):
    """Transform applied to text.

    Args:
        ignore_first (bool): Whether to ignore the first sentence when applying the transform.
            It is useful when the main idea of the text is expressed in the first sentence. Default: False.
        always_apply (bool): Whether the transform should be always applied. Default: False.
        p (float): Probability of applying the transform. Default: 0.5.
    """

    def __init__(self, ignore_first: bool = False, always_apply: bool = False, p: float = 0.5):
        super(TextTransform, self).__init__(always_apply, p)
        self.ignore_first = ignore_first

    def __call__(self, *args: Any, force_apply: bool = False, **kwargs: Any) -> Dict[str, Text]:
        if not self.ignore_first:
            return super(TextTransform, self).__call__(self, *args, force_apply, **kwargs)
        return self.apply_without_first(self, *args, force_apply, **kwargs)

    def apply_without_first(self, *args: Any, force_apply: bool = False, **kwargs: Any) -> Dict[str, Text]:
        kwargs_with_first_sentences = get_first_sentences_from_kwargs(kwargs)
        kwargs_without_first_sentences = remove_first_sentences_from_kwargs(kwargs)
        augmented_kwargs_without_first_sentences = super(TextTransform, self).__call__(
            self, *args, force_apply, **kwargs_without_first_sentences
        )
        augmented_kwargs = combine_augmented_kwargs_with_first_sentences(
            augmented_kwargs_without_first_sentences,
            kwargs_with_first_sentences
        )
        return augmented_kwargs

    @property
    def targets(self) -> Dict[str, Callable[[Text, Dict[str, Any]], Text]]:
        return {"text": self.apply}

    def update_params(self, params: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        return params

    def get_base_init_args(self) -> Dict[str, Any]:
        return {"ignore_first": self.ignore_first, "always_apply": self.always_apply, "p": self.p}
