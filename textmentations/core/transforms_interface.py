from typing import Any, Callable, Dict, Tuple

from albumentations.core.transforms_interface import BasicTransform
from typing_extensions import Literal

from ..corpora.types import Text
from .utils import extract_first_sentence_by_key, remove_first_sentence_by_key, wrap_text_with_first_sentence_by_key


class TextTransform(BasicTransform):
    """Transform applied to text.

    Args:
        ignore_first: Whether to ignore the first sentence when applying this transform.
            If `ignore_first` is True, this transform will be applied to the text without the first sentence.
            It is useful when the main idea of the text is expressed in the first sentence.
        always_apply: Whether to always apply this transform.
        p: The probability of applying this transform.
    """

    def __init__(self, ignore_first: bool = False, always_apply: bool = False, p: float = 0.5) -> None:
        super().__init__(always_apply, p)
        self.ignore_first = ignore_first

    def _validate_base_init_args(self, **params: Any) -> None:
        ignore_first = params.pop("ignore_first")
        always_apply = params.pop("always_apply")
        p = params.pop("p")
        if not isinstance(ignore_first, bool):
            raise TypeError(f"ignore_first must be boolean. Got: {type(ignore_first)}")
        if not isinstance(always_apply, bool):
            raise TypeError(f"always_apply must be boolean. Got: {type(always_apply)}")
        if not isinstance(p, (float, int)):
            raise TypeError(f"p must be a real number between 0 and 1. Got: {type(p)}")
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be between 0 and 1. Got: {p}")

    def _validate_transform_init_args(self, *args: Any, **params: Any) -> None:
        raise NotImplementedError

    def __call__(self, *args: Any, force_apply: bool = False, **kwargs: Text) -> Dict[str, Text]:
        if args:
            raise KeyError("You have to pass data to augmentations as named arguments, for example: augment(text=text)")
        if not all(isinstance(text, str) for _, text in kwargs.items()):
            raise TypeError("You have to pass string data to augmentations.")
        if not self.ignore_first:
            return super().__call__(force_apply=force_apply, **kwargs)
        return self.apply_without_first(force_apply=force_apply, **kwargs)

    def apply_without_first(self, force_apply: bool = False, **kwargs: Text) -> Dict[str, Text]:
        key2first_sentence = extract_first_sentence_by_key(kwargs)
        key2text_without_first_sentence = remove_first_sentence_by_key(kwargs)
        key2augmented_text_without_first_sentence = super().__call__(
            force_apply=force_apply, **key2text_without_first_sentence
        )
        key2augmented_text = wrap_text_with_first_sentence_by_key(
            key2augmented_text_without_first_sentence, key2first_sentence
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


class SingleCorpusTypeTransform(TextTransform):
    """Transform applied to single text component unit."""

    def __init__(
        self,
        ignore_first: bool = False,
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(ignore_first, always_apply, p)
        self._validate_base_init_args(ignore_first=ignore_first, always_apply=always_apply, p=p)


# TODO: MultipleCorpusTypesTransform을 상속받는 경우
#  WordBaseTransform, SentenceBaseTransform, TextBaseTransform 중에서 알맞은 클래스를 상속받도록 하자
#  또는 SingleCorpusTypeTrnaform에 _unit 클래스 변수를 추가해도 된다
#  이렇게 할 경우 WordBaseTransform, SentenceBaseTransform, TextBaseTransform만 사용하고 TextTransform을 리팩토링 하면 됨
class MultipleCorpusTypesTransform(TextTransform):
    """Transform applied to multiple text component units.

    Args:
        unit: Unit to which transform is to be applied.
        ignore_first: Whether to ignore the first sentence when applying this transform.
            If `ignore_first` is True, this transform will be applied to the text without the first sentence.
            It is useful when the main idea of the text is expressed in the first sentence.
        always_apply: Whether to always apply this transform.
        p: The probability of applying this transform.
    """

    def __init__(
        self,
        unit: Literal["word", "sentence", "text"] = "word",
        ignore_first: bool = False,
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(ignore_first, always_apply, p)
        self._validate_base_init_args(unit=unit, ignore_first=ignore_first, always_apply=always_apply, p=p)
        self.unit = unit

    def _validate_base_init_args(self, **params: Any) -> None:
        unit = params.pop("unit")
        possible_units = self.get_possible_units_names()
        if unit not in possible_units:
            raise ValueError(f"unit must be one of {possible_units}.")
        super()._validate_base_init_args(**params)

    def apply(self, text: Text, **params: Any) -> Text:
        return self.units.get(self.unit, lambda x, **p: x)(text, **params)

    @property
    def units(self) -> Dict[str, Callable[..., Text]]:
        return {"word": self.apply_to_words, "sentence": self.apply_to_sentences, "text": self.apply_to_text}

    def apply_to_words(self, text: Text, **params: Any) -> Text:
        raise NotImplementedError

    def apply_to_sentences(self, text: Text, **params: Any) -> Text:
        raise NotImplementedError

    def apply_to_text(self, text: Text, **params: Any) -> Text:
        raise NotImplementedError

    def get_possible_units_names(self) -> Tuple[str, ...]:
        raise NotImplementedError

    def get_base_init_args(self) -> Dict[str, Any]:
        return {"unit": self.unit, "ignore_first": self.ignore_first, "always_apply": self.always_apply, "p": self.p}
