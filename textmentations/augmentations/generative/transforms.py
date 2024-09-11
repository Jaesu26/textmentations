from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ...core.transforms_interface import TextTransform
from ...corpora.types import Text
from ..utils import _get_albert_mlm, _get_bert_tokenizer_fast
from . import functional as F

_ALBERT_MODEL_PATH = Path(__file__).resolve().parent / "_models" / "kykim-albert-kor-base"
_albert_model = _get_albert_mlm(model_path=_ALBERT_MODEL_PATH).eval()
_albert_tokenizer = _get_bert_tokenizer_fast(model_path=_ALBERT_MODEL_PATH)


class IterativeMaskFilling(TextTransform):
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
        dummy_tensor = torch.tensor([0])
        dummy_tensor = dummy_tensor.to(device).cpu()  # Raise original error

    def apply(self, text: Text, **params: Any) -> Text:
        return F.iterative_mask_fill(text, self._model, self._tokenizer, self.top_k, self.device)

    def get_transform_init_args_names(self) -> tuple[str, str]:
        return ("top_k", "device")
