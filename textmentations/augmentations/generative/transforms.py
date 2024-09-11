from __future__ import annotations

from pathlib import Path
from typing import Any

from ...core.transforms_interface import TextTransform
from ...corpora.types import Text
from ..utils import _get_albert_mlm, _get_bert_tokenizer_fast

_ALBERT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "kykim-albert-kor-base"
_kykim_albert_model = _get_albert_mlm(model_path=_ALBERT_MODEL_PATH).eval()
_kykim_albert_tokenizer = _get_bert_tokenizer_fast(model_path=_ALBERT_MODEL_PATH)


class IterativeMaskFilling(TextTransform):
    _model = _kykim_albert_model
    _tokenizer = _kykim_albert_tokenizer

    def apply(self, text: Text, *args: Any, **params: Any) -> Text:
        return text
