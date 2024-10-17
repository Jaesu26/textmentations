from __future__ import annotations

from pathlib import Path
from typing import Any, List

import torch
from deep_translator.constants import GOOGLE_LANGUAGES_TO_CODES as CODE_BY_GOOGLE_LANGUAGE
from deep_translator.exceptions import LanguageNotSupportedException as LanguageNotSupportedError

import textmentations.augmentations.generation.functional as fg
from textmentations.augmentations.utils import _read_pretrained_albert_mlm, _read_pretrained_bert_tokenizer_fast
from textmentations.core.transforms_interface import TextTransform
from textmentations.corpora.types import Language, Text

LANGUAGES: List[Language] = sorted(CODE_BY_GOOGLE_LANGUAGE.values())
_ALBERT_MODEL_PATH = Path(__file__).resolve().parent / "_models" / "kykim-albert-kor-base"
_albert_model = _read_pretrained_albert_mlm(model_path=_ALBERT_MODEL_PATH).eval()
_albert_tokenizer = _read_pretrained_bert_tokenizer_fast(model_path=_ALBERT_MODEL_PATH)


class BackTranslation(TextTransform):
    """Back-translates the input text by translating it to the target language and then back to the original.

    Args:
        from_lang: The language of the input text.
        to_lang: The language to which the input text will be translated.
        ignore_first: Whether to ignore the first sentence when applying this transform.
            It is useful when the main idea of the text is in the first sentence.
        p: The probability of applying this transform.

    Examples:
        >>> import textmentations as T
        >>> text = "어제 식당에 갔다. 목이 너무 말랐다. 먼저 물 한 잔을 마셨다. 그리고 탕수육을 맛있게 먹었다."
        >>> bt = T.BackTranslation(from_lang="ko", to_lang="en", p=1.0)
        >>> augmented_text = bt(text=text)["text"]

    References:
        https://arxiv.org/pdf/1808.09381
    """

    def __init__(
        self,
        from_lang: Language = "ko",
        to_lang: Language = "en",
        ignore_first: bool = False,
        always_apply: bool | None = None,
        p: float = 0.5,
    ) -> None:
        super().__init__(ignore_first=ignore_first, always_apply=always_apply, p=p)
        self._validate_transform_init_args(from_lang=from_lang, to_lang=to_lang)
        self.from_lang = from_lang
        self.to_lang = to_lang

    def _validate_transform_init_args(self, *, from_lang: Language, to_lang: Language) -> None:
        if from_lang not in LANGUAGES:
            raise LanguageNotSupportedError(f"from_lang must be one of {LANGUAGES}. Got: {from_lang}")
        if to_lang not in LANGUAGES:
            raise LanguageNotSupportedError(f"to_lang must be one of {LANGUAGES}. Got: {to_lang}")

    def apply(self, text: Text, **params: Any) -> Text:
        return fg.back_translate(text, self.from_lang, self.to_lang)

    def get_transform_init_args_names(self) -> tuple[str, str]:
        return "from_lang", "to_lang"


class ContextualInsertion(TextTransform):
    """Randomly inserts mask tokens in the input text and fills them with language model predictions.

    Args:
        insertion_prob: The probability of inserting a word.
        top_k: The number of candidate words to replace the masked word at each iteration
        device: The device to use for computation (e.g., "cpu", "cuda:1", torch.device("cuda")).
        ignore_first: Whether to ignore the first sentence when applying this transform.
            It is useful when the main idea of the text is in the first sentence.
        p: The probability of applying this transform.

    Examples:
        >>> import textmentations as T
        >>> text = "어제 식당에 갔다. 목이 너무 말랐다. 먼저 물 한 잔을 마셨다. 그리고 탕수육을 맛있게 먹었다."
        >>> ci = T.ContextualInsertion(isnertion_prob=0.15, top_k=5, device="cuda:0", p=1.0)
        >>> augmented_text = ci(text=text)["text"]
    """

    _model = _albert_model
    _tokenizer = _albert_tokenizer
    _vocab_size = _tokenizer.vocab_size

    def __init__(
        self,
        insertion_prob: float = 0.15,
        top_k: int = 5,
        device: str | torch.device = "cpu",
        ignore_first: bool = False,
        always_apply: bool | None = None,
        p: float = 0.5,
    ) -> None:
        super().__init__(ignore_first=ignore_first, always_apply=always_apply, p=p)
        self._validate_transform_init_args(insertion_prob=insertion_prob, top_k=top_k, device=device)
        self.insertion_prob = insertion_prob
        self.top_k = top_k
        self.device = device

    def _validate_transform_init_args(self, *, insertion_prob: float, top_k: int, device: str | torch.device) -> None:
        if not isinstance(insertion_prob, (float, int)):
            raise TypeError(f"insertion_prob must be a real number between 0 and 1. Got: {type(insertion_prob)}")
        if not (0.0 <= insertion_prob <= 1.0):
            raise ValueError(f"insertion_prob must be between 0 and 1. Got: {insertion_prob}")
        if type(top_k) is not int:
            raise TypeError(f"top_k must be a positive integer. Got: {type(top_k)}")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive. Got: {top_k}")
        if top_k > self._vocab_size:
            raise ValueError(
                f"top_k exceeds the tokenizer's vocabulary size. Maximum allowed: {self._vocab_size}. Got: {top_k}"
            )
        torch.device(device)  # Checks if the device is valid

    def apply(self, text: Text, *args: Any, **params: Any) -> Text:
        return fg.insert_contextual_words(
            text, self._model, self._tokenizer, self.insertion_prob, self.top_k, self.device
        )

    def get_transform_init_args_names(self) -> tuple[str, str, str]:
        return "insertion_prob", "top_k", "device"


class ContextualReplacement(TextTransform):
    """Randomly replaces words in the input text with mask tokens and fills them with language model predictions.

    Args:
        masking_prob: The probability of masking a word.
        top_k: The number of candidate words to replace the masked word at each iteration
        device: The device to use for computation (e.g., "cpu", "cuda:1", torch.device("cuda")).
        ignore_first: Whether to ignore the first sentence when applying this transform.
            It is useful when the main idea of the text is in the first sentence.
        p: The probability of applying this transform.

    Examples:
        >>> import textmentations as T
        >>> text = "어제 식당에 갔다. 목이 너무 말랐다. 먼저 물 한 잔을 마셨다. 그리고 탕수육을 맛있게 먹었다."
        >>> cr = T.ContextualReplacement(masking_prob=0.15, top_k=5, device="cuda:0", p=1.0)
        >>> augmented_text = cr(text=text)["text"]

    References:
        https://arxiv.org/pdf/1805.06201
        https://arxiv.org/pdf/1812.06705
    """

    _model = _albert_model
    _tokenizer = _albert_tokenizer
    _vocab_size = _tokenizer.vocab_size

    def __init__(
        self,
        masking_prob: float = 0.15,
        top_k: int = 5,
        device: str | torch.device = "cpu",
        ignore_first: bool = False,
        always_apply: bool | None = None,
        p: float = 0.5,
    ) -> None:
        super().__init__(ignore_first=ignore_first, always_apply=always_apply, p=p)
        self._validate_transform_init_args(masking_prob=masking_prob, top_k=top_k, device=device)
        self.masking_prob = masking_prob
        self.top_k = top_k
        self.device = device

    def _validate_transform_init_args(self, *, masking_prob: float, top_k: int, device: str | torch.device) -> None:
        if not isinstance(masking_prob, (float, int)):
            raise TypeError(f"masking_prob must be a real number between 0 and 1. Got: {type(masking_prob)}")
        if not (0.0 <= masking_prob <= 1.0):
            raise ValueError(f"masking_prob must be between 0 and 1. Got: {masking_prob}")
        if type(top_k) is not int:
            raise TypeError(f"top_k must be a positive integer. Got: {type(top_k)}")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive. Got: {top_k}")
        if top_k > self._vocab_size:
            raise ValueError(
                f"top_k exceeds the tokenizer's vocabulary size. Maximum allowed: {self._vocab_size}. Got: {top_k}"
            )
        torch.device(device)  # Checks if the device is valid

    def apply(self, text: Text, *args: Any, **params: Any) -> Text:
        return fg.replace_contextual_words(
            text, self._model, self._tokenizer, self.masking_prob, self.top_k, self.device
        )

    def get_transform_init_args_names(self) -> tuple[str, str, str]:
        return "masking_prob", "top_k", "device"


class IterativeMaskFilling(TextTransform):
    """Iteratively masks words in a randomly selected sentence and replaces them with language model predictions.

    Args:
        top_k: The number of candidate words to replace the masked word at each iteration
        device: The device to use for computation (e.g., "cpu", "cuda:1", torch.device("cuda")).
        ignore_first: Whether to ignore the first sentence when applying this transform.
            It is useful when the main idea of the text is in the first sentence.
        p: The probability of applying this transform.

    Examples:
        >>> import textmentations as T
        >>> text = "어제 식당에 갔다. 목이 너무 말랐다. 먼저 물 한 잔을 마셨다. 그리고 탕수육을 맛있게 먹었다."
        >>> imf = T.IterativeMaskFilling(top_k=5, device="cuda:0", p=1.0)
        >>> augmented_text = imf(text=text)["text"]

    References:
        https://arxiv.org/pdf/2401.01830
    """

    _model = _albert_model
    _tokenizer = _albert_tokenizer
    _vocab_size = _tokenizer.vocab_size

    def __init__(
        self,
        top_k: int = 5,
        device: str | torch.device = "cpu",
        ignore_first: bool = False,
        always_apply: bool | None = None,
        p: float = 0.5,
    ) -> None:
        super().__init__(ignore_first=ignore_first, always_apply=always_apply, p=p)
        self._validate_transform_init_args(top_k=top_k, device=device)
        self.top_k = top_k
        self.device = device

    def _validate_transform_init_args(self, *, top_k: int, device: str | torch.device) -> None:
        if type(top_k) is not int:
            raise TypeError(f"top_k must be a positive integer. Got: {type(top_k)}")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive. Got: {top_k}")
        if top_k > self._vocab_size:
            raise ValueError(
                f"top_k exceeds the tokenizer's vocabulary size. Maximum allowed: {self._vocab_size}. Got: {top_k}"
            )
        torch.device(device)  # Checks if the device is valid

    def apply(self, text: Text, **params: Any) -> Text:
        return fg.iterative_mask_fill(text, self._model, self._tokenizer, self.top_k, self.device)

    def get_transform_init_args_names(self) -> tuple[str, str]:
        return "top_k", "device"
