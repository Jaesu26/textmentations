"""Corpus types.
Word -> (), Sentence -> {}, Text -> [], then it's expressed as follows.
[{(아침에는) (짜장면을) (맛있게) (먹었다)}. {(점심에는) (짬뽕을) (맛있게) (먹었다)}. {(저녁에는) (짬짜면을) (맛있게) (먹었다)}.]
"""

from typing import TypeVar

from typing_extensions import TypeAlias

Word: TypeAlias = str
Sentence: TypeAlias = str
Text: TypeAlias = str
WS = TypeVar("WS", Word, Sentence)
