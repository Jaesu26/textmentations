from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from deep_translator.constants import GOOGLE_LANGUAGES_TO_CODES
from deep_translator.exceptions import LanguageNotSupportedException

import textmentations.augmentations.generation.functional as fg
from textmentations.augmentations.utils import _get_albert_mlm, _get_bert_tokenizer_fast
from textmentations.core.transforms_interface import TextTransform
from textmentations.corpora.types import Language, Text

LANGUAGES = sorted(GOOGLE_LANGUAGES_TO_CODES.values())
_ALBERT_MODEL_PATH = Path(__file__).resolve().parent / "_models" / "kykim-albert-kor-base"
_albert_model = _get_albert_mlm(model_path=_ALBERT_MODEL_PATH).eval()
_albert_tokenizer = _get_bert_tokenizer_fast(model_path=_ALBERT_MODEL_PATH)


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
            raise LanguageNotSupportedException(f"from_lang must be one of {LANGUAGES}. Got: {from_lang}")
        if to_lang not in LANGUAGES:
            raise LanguageNotSupportedException(f"to_lang must be one of {LANGUAGES}. Got: {to_lang}")

    def apply(self, text: Text, **params: Any) -> Text:
        return fg.back_translate(text, self.from_lang, self.to_lang)

    def get_transform_init_args_names(self) -> tuple[str, str]:
        return ("from_lang", "to_lang")


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
        return ("top_k", "device")
