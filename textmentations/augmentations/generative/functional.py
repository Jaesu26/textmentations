from __future__ import annotations

import random
import re
from typing import Any

import torch

from ...corpora.types import Sentence, Text, Word
from ..utils import _squeeze_first, autopsy_sentence, autopsy_text, join_words_into_sentence, pass_empty_text


@pass_empty_text
def iterative_mask_fill(text: Text, model: Any, tokenizer: Any, top_k: int, device: str | torch.device) -> Text:
    model.to(device)
    augmented_text = _iterative_mask_fill(text, model, tokenizer, top_k, device)
    augmented_text = re.sub(r"\s*##\b", "", augmented_text)  # ex) 나는 짬뽕 ##을 먹었다. -> 나는 짬뽕을 먹었다.
    return augmented_text


@autopsy_text
def _iterative_mask_fill(
    sentences: list[Sentence],
    model: Any,
    tokenizer: Any,
    top_k: int,
    device: str | torch.device,
) -> list[Sentence]:
    num_sentences = len(sentences)
    index = random.randrange(0, num_sentences)
    sentence = sentences[index]
    sentences[index] = _iterative_mask_fill_in_sentence(sentence, model, tokenizer, top_k, device)
    return sentences


@autopsy_sentence
def _iterative_mask_fill_in_sentence(
    words: list[Word],
    model: Any,
    tokenizer: Any,
    top_k: int,
    device: str | torch.device,
) -> list[Word]:
    for masking_index, word in enumerate(words):
        words[masking_index] = tokenizer.mask_token
        sentence = join_words_into_sentence(words)
        plausible_words = _predict_mask(sentence, model, tokenizer, top_k, device)
        if not plausible_words:
            words[masking_index] = word
            continue
        plausible_word = _squeeze_first(plausible_words)
        words[masking_index] = plausible_word
    return words


def _predict_mask(sentence: Sentence, model: Any, tokenizer: Any, top_k: int, device: str | torch.device) -> list[Word]:
    mask_token_id = tokenizer.mask_token_id
    input_ids = tokenizer.encode(sentence, truncation=True, return_tensors="pt")
    if mask_token_id not in input_ids:
        return []
    input_ids = input_ids.to(device)
    with torch.no_grad():
        mlm_output = model(input_ids)
    mask_tokens_index = torch.where(input_ids.eq(mask_token_id))
    topk = mlm_output.logits[mask_tokens_index].cpu().topk(top_k)
    token_ids_list, scores_list = topk.indices.tolist(), topk.values.tolist()
    token_ids_to_decode = []
    for token_ids, scores in zip(token_ids_list, scores_list):
        token_id = _squeeze_first(random.choices(token_ids, weights=scores, k=1))
        token_ids_to_decode.append(token_id)
    plausible_words = [tokenizer.decode(token_id) for token_id in token_ids_to_decode]
    return plausible_words
