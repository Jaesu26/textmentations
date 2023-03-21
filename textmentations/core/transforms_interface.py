from typing import Any, Callable, Dict

from albumentations.core.transforms_interface import BasicTransform

from ..corpora.corpus_types import Text
from .utils import extract_first_sentence_by_key, remove_first_sentence_by_key, wrap_text_with_first_sentence_by_key


class TextTransform(BasicTransform):
    """Transform applied to text.

    Args:
        ignore_first (bool): Whether to ignore the first sentence when applying the transform.
            It is useful when the main idea of the text is expressed in the first sentence. Default: False.
        always_apply (bool): Whether the transform should be always applied. Default: False.
        p (float): The probability of applying the transform. Default: 0.5.
    """

    def __init__(self, ignore_first: bool = False, always_apply: bool = False, p: float = 0.5) -> None:
        super(TextTransform, self).__init__(always_apply, p)
        self._validate_base_init_args(ignore_first, always_apply, p)
        self.ignore_first = ignore_first

    def _validate_base_init_args(self, ignore_first: bool, always_apply: bool, p: float) -> None:
        if not isinstance(ignore_first, bool):
            raise TypeError(f"ignore_first must be boolean. Got: {type(ignore_first)}")
        if not isinstance(always_apply, bool):
            raise TypeError(f"always_apply must be boolean. Got: {type(always_apply)}")
        if not isinstance(p, (float, int)):
            raise TypeError(f"p must be a real number between 0 and 1. Got: {type(p)}")
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be between 0 and 1. Got: {p}")

    def __call__(self, *args: Any, force_apply: bool = False, **kwargs: Text) -> Dict[str, Text]:
        if args:
            raise KeyError("You have to pass data to augmentations as named arguments, for example: augment(text=text)")
        if not all(isinstance(text, str) for _, text in kwargs.items()):
            raise TypeError("You have to pass string data to augmentations.")
        if not self.ignore_first:
            return super(TextTransform, self).__call__(force_apply=force_apply, **kwargs)
        return self.apply_without_first(force_apply=force_apply, **kwargs)

    def apply_without_first(self, force_apply: bool = False, **kwargs: Text) -> Dict[str, Text]:
        key2first_sentence = extract_first_sentence_by_key(kwargs)
        key2text_without_first_sentence = remove_first_sentence_by_key(kwargs)
        key2augmented_text_without_first_sentence = super(TextTransform, self).__call__(
            force_apply=force_apply, **key2text_without_first_sentence
        )
        key2augmented_text = wrap_text_with_first_sentence_by_key(
            key2augmented_text_without_first_sentence,
            key2first_sentence
        )
        return key2augmented_text

    def apply(self, text: Text, **params: Any) -> Text:
        raise NotImplementedError

    @property
    def targets(self) -> Dict[str, Callable[..., Text]]:
        return {"text": self.apply}

    def update_params(self, params: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        return params

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def get_base_init_args(self) -> Dict[str, Any]:
        return {"ignore_first": self.ignore_first, "always_apply": self.always_apply, "p": self.p}
