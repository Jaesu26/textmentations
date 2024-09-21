from __future__ import annotations

import random
import re
from typing import Any

import torch
from deep_translator.exceptions import NotValidLength, RequestError, TooManyRequests, TranslationNotFound

from textmentations.augmentations.utils import (
    _squeeze_first,
    autopsy_sentence,
    autopsy_text,
    get_translator,
    join_words_into_sentence,
    pass_empty_text,
    remove_empty_strings,
)
from textmentations.corpora.types import Language, Sentence, Text, Word


@pass_empty_text
def back_translate(text: Text, from_lang: Language, to_lang: Language) -> Text:
    """Back-translates the text by translating it to the target language and then back to the original.

    Args:
        text: The input text.
        from_lang: The language of the input text.
        to_lang: The language to which the input text will be translated.

    Returns:
        A back-translated text.

    Examples:
        >>> import textmentations.augmentations.generation.functional as fg
        >>> text = "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."
        >>> from_lang = "ko"
        >>> to_lang = "en"
        >>> augmented_text = fg.back_translate(text, from_lang, to_lang)
    """
    translator = get_translator()
    try:
        translator.source, translator.target = from_lang, to_lang
        translated_text = translator.translate(text)
        translator.source, translator.target = to_lang, from_lang
        back_translated_text = translator.translate(translated_text)
        return back_translated_text
    except (NotValidLength, RequestError, TooManyRequests, TranslationNotFound):
        return text


@pass_empty_text
def iterative_mask_fill(text: Text, model: Any, tokenizer: Any, top_k: int, device: str | torch.device) -> Text:
    """Iteratively masks words in a randomly selected sentence and replaces them with language model predictions.

    Args:
        text: The input text.
        model: The masked language model used for making predictions.
        tokenizer: The tokenizer that will be used to encode text for the model and decode the model's output.
        top_k: The number of candidate words to replace the masked word at each iteration
        device: The device to use for computation (e.g., "cpu", "cuda:1", torch.device("cuda")).

    Returns:
        A augmented text.

    Examples:
        >>> import textmentations.augmentations.generation.functional as fg
        >>> from transformers import AutoModelForMaskedLM, AutoTokenizer
        >>> text = "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."
        >>> pretrained_model_name_or_path = "Pre-trained huggingface masked language model name or path you want to use"
        >>> model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path)
        >>> tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        >>> top_k = 5
        >>> device = "cuda:0"
        >>> augmented_text = fg.iterative_mask_fill(text, model, tokenizer, top_k, device)
    """
    model.to(device)
    augmented_text = _iterative_mask_fill(text, model, tokenizer, top_k, device)
    augmented_text = re.sub(r"\s*##\b", "", augmented_text)  # e.g., 나는 짬뽕 ##을 먹었다. -> 나는 짬뽕을 먹었다.
    return augmented_text


@autopsy_text
def _iterative_mask_fill(
    sentences: list[Sentence],
    model: Any,
    tokenizer: Any,
    top_k: int,
    device: str | torch.device,
) -> list[Sentence]:
    """Iteratively masks words in a randomly selected sentence and replaces them with language model predictions."""
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
    """Iteratively masks each word in the list of words and replaces it with language model predictions."""
    for masking_index, word in enumerate(words):
        words[masking_index] = tokenizer.mask_token
        sentence_with_masking = join_words_into_sentence(words)
        plausible_words = _predict_masks(sentence_with_masking, model, tokenizer, top_k, device)
        plausible_word_iter = iter(plausible_words)
        plausible_word = next(plausible_word_iter, word)
        words[masking_index] = plausible_word
    return words


def _predict_masks(
    sentence: Sentence,
    model: Any,
    tokenizer: Any,
    top_k: int,
    device: str | torch.device,
) -> list[Word]:
    """Predicts plausible words to replace mask tokens in the sentence using the masked language model."""
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


@pass_empty_text
def replace_contextual_words(
    text: Text,
    model: Any,
    tokenizer: Any,
    masking_prob: float,
    top_k: int,
    device: str | torch.device,
) -> Text:
    """Randomly replaces words in the text with mask tokens and fills them with language model predictions.

    Args:
        text: The input text.
        model: The masked language model used for making predictions.
        tokenizer: The tokenizer that will be used to encode text for the model and decode the model's output.
        masking_prob: The probability of masking a word.
        top_k: The number of candidate words to replace the masked word at each iteration
        device: The device to use for computation (e.g., "cpu", "cuda:1", torch.device("cuda")).

    Examples:
        >>> import textmentations.augmentations.generation.functional as fg
        >>> from transformers import AutoModelForMaskedLM, AutoTokenizer
        >>> text = "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."
        >>> pretrained_model_name_or_path = "Pre-trained huggingface masked language model name or path you want to use"
        >>> model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path)
        >>> tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        >>> masking_prob = 0.15
        >>> top_k = 5
        >>> device = "cuda:0"
        >>> augmented_text = fg.replace_contextual_words(text, model, tokenizer, masking_prob, top_k, device)
    """
    model.to(device)
    augmented_text = _replace_contextual_words(text, model, tokenizer, masking_prob, top_k, device)
    augmented_text = re.sub(r"\s*##\b", "", augmented_text)  # e.g., 나는 짬뽕 ##을 먹었다. -> 나는 짬뽕을 먹었다.
    return augmented_text


@autopsy_text
def _replace_contextual_words(
    sentences: list[Sentence],
    model: Any,
    tokenizer: Any,
    masking_prob: float,
    top_k: int,
    device: str | torch.device,
) -> list[Sentence]:
    """Randomly replaces words in each sentence with mask tokens and fills them with language model predictions."""
    return [
        _replace_contextual_words_in_sentence(sentence, model, tokenizer, masking_prob, top_k, device)
        for sentence in sentences
    ]


@autopsy_sentence
def _replace_contextual_words_in_sentence(
    words: list[Word],
    model: Any,
    tokenizer: Any,
    masking_prob: float,
    top_k: int,
    device: str | torch.device,
) -> list[Word]:
    """Randomly replaces words in the list of words with mask tokens and fills them with language model predictions."""
    mask_token = tokenizer.mask_token
    masked_words = []
    words_with_masking = []
    for word in words:
        if random.random() < masking_prob:
            masked_words.append(word)
            words_with_masking.append(mask_token)
            continue
        words_with_masking.append(word)
    sentence_with_masking = join_words_into_sentence(words_with_masking)
    plausible_words = _predict_masks(sentence_with_masking, model, tokenizer, top_k, device)
    plausible_word_iter = iter(plausible_words)
    masked_words_iter = iter(masked_words)
    augmented_words = []
    for word in words_with_masking:
        if word != mask_token:
            augmented_words.append(word)
            continue
        unmasked_word = next(masked_words_iter)
        plausible_word = next(plausible_word_iter, unmasked_word)
        augmented_words.append(plausible_word)
    return augmented_words


@pass_empty_text
def insert_contextual_words(
    text: Text,
    model: Any,
    tokenizer: Any,
    insertion_prob: float,
    top_k: int,
    device: str | torch.device,
) -> Text:
    """Randomly inserts mask tokens in the text and fills them with language model predictions.

    Args:
        text: The input text.
        model: The masked language model used for making predictions.
        tokenizer: The tokenizer that will be used to encode text for the model and decode the model's output.
        insertion_prob: The probability of inserting a mask token.
        top_k: The number of candidate words to replace the masked word at each iteration
        device: The device to use for computation (e.g., "cpu", "cuda:1", torch.device("cuda")).

    Examples:
        >>> import textmentations.augmentations.generation.functional as fg
        >>> from transformers import AutoModelForMaskedLM, AutoTokenizer
        >>> text = "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."
        >>> pretrained_model_name_or_path = "Pre-trained huggingface masked language model name or path you want to use"
        >>> model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path)
        >>> tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        >>> insertion_prob = 0.15
        >>> top_k = 5
        >>> device = "cuda:0"
        >>> augmented_text = fg.insert_contextual_words(text, model, tokenizer, insertion_prob, top_k, device)
    """
    model.to(device)
    augmented_text = _insert_contextual_words(text, model, tokenizer, insertion_prob, top_k, device)
    augmented_text = re.sub(r"\s*##\b", "", augmented_text)  # e.g., 나는 짬뽕 ##을 먹었다. -> 나는 짬뽕을 먹었다.
    return augmented_text


@autopsy_text
def _insert_contextual_words(
    sentences: list[Sentence],
    model: Any,
    tokenizer: Any,
    insertion_prob: float,
    top_k: int,
    device: str | torch.device,
) -> list[Sentence]:
    """Randomly inserts mask tokens in each sentence and fills them with language model predictions."""
    return [
        _insert_contextual_words_in_sentence(sentence, model, tokenizer, insertion_prob, top_k, device)
        for sentence in sentences
    ]


@autopsy_sentence
def _insert_contextual_words_in_sentence(
    words: list[Word],
    model: Any,
    tokenizer: Any,
    insertion_prob: float,
    top_k: int,
    device: str | torch.device,
) -> list[Word]:
    """Randomly inserts mask tokens in the list of words and fills them with language model predictions."""
    mask_token = tokenizer.mask_token
    words_with_masking = []
    for word in words:
        if random.random() < insertion_prob:
            words_with_masking.append(mask_token)
        words_with_masking.append(word)
    sentence_with_masking = join_words_into_sentence(words_with_masking)
    plausible_words = _predict_masks(sentence_with_masking, model, tokenizer, top_k, device)
    plausible_word_iter = iter(plausible_words)
    augmented_words = [next(plausible_word_iter, "") if word == mask_token else word for word in words_with_masking]
    augmented_words = remove_empty_strings(augmented_words)
    return augmented_words
