from __future__ import annotations

from typing import Any

import numpy as np
import torch
from deep_translator.exceptions import NotValidLength, RequestError, TooManyRequests, TranslationNotFound

from textmentations.augmentations.utils import (
    EMPTY_STRING,
    _flatten,
    _generate_boolean_mask,
    autopsy_sentence,
    autopsy_text,
    check_rng,
    get_translator,
    join_words_into_sentence,
    pass_empty_text,
    remove_empty_strings,
    with_cleared_double_hash_tokens,
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
@with_cleared_double_hash_tokens
def insert_contextual_words(
    text: Text,
    model: Any,
    tokenizer: Any,
    insertion_prob: float,
    top_k: int,
    device: str | torch.device,
    *,
    seed: int | np.random.Generator | None = None,
) -> Text:
    """Randomly inserts mask tokens in the text and fills them with language model predictions.

    Args:
        text: The input text.
        model: The masked language model used for making predictions.
        tokenizer: The tokenizer that will be used to encode text for the model and decode the model's output.
        insertion_prob: The probability of inserting a mask token.
        top_k: The number of candidate words to replace the masked word at each iteration
        device: The device to use for computation (e.g., "cpu", "cuda:1", torch.device("cuda")).
        seed: The seed for a random number generator. Can be None, an int, or an instance of np.random.Generator.

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
    rng = check_rng(seed)
    return _insert_contextual_words(text, model, tokenizer, insertion_prob, top_k, device, rng)


@autopsy_text
def _insert_contextual_words(
    sentences: list[Sentence],
    model: Any,
    tokenizer: Any,
    insertion_prob: float,
    top_k: int,
    device: str | torch.device,
    rng: np.random.Generator,
) -> list[Sentence]:
    """Randomly inserts mask tokens in each sentence and fills them with language model predictions."""
    return [
        _insert_contextual_words_in_sentence(sentence, model, tokenizer, insertion_prob, top_k, device, rng)
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
    rng: np.random.Generator,
) -> list[Word]:
    """Randomly inserts mask tokens in the list of words and fills them with language model predictions."""
    mask_token = tokenizer.mask_token
    insertion_mask = _generate_boolean_mask(len(words), insertion_prob, rng).tolist()
    words_with_masking = _flatten(
        [[mask_token, word] if should_insert else [word] for word, should_insert in zip(words, insertion_mask)]
    )  # Avoid inserting the mask token at the end of words because the model often predicts punctuation
    sentence_with_masking = join_words_into_sentence(words_with_masking)
    plausible_words = _predict_masks(sentence_with_masking, model, tokenizer, top_k, device, rng)
    plausible_word_iter = iter(plausible_words)
    augmented_words = [next(plausible_word_iter, EMPTY_STRING) if w == mask_token else w for w in words_with_masking]
    augmented_words = remove_empty_strings(augmented_words)
    return augmented_words


def _predict_masks(
    sentence: Sentence,
    model: Any,
    tokenizer: Any,
    top_k: int,
    device: str | torch.device,
    rng: np.random.Generator,
) -> list[Word]:
    """Predicts plausible words to replace mask tokens in the sentence using the masked language model."""
    mask_token_id = tokenizer.mask_token_id
    input_ids = tokenizer.encode(sentence, truncation=True, return_tensors="pt")
    if mask_token_id not in input_ids:
        return []
    model.to(device)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        mlm_output = model(input_ids)
    mask_tokens_index = torch.where(input_ids.eq(mask_token_id))
    topk = mlm_output.logits[mask_tokens_index].cpu().topk(top_k)
    topk_values = topk.values.numpy()
    token_ids_list = topk.indices.tolist()
    scores_list = topk_values.__truediv__(topk_values.sum(axis=-1, keepdims=True)).tolist()
    selected_indices = rng.multinomial(n=1, pvals=scores_list).argmax(axis=-1).tolist()
    token_ids_to_decode = [token_ids[index] for token_ids, index in zip(token_ids_list, selected_indices)]
    plausible_words = [tokenizer.decode(token_id) for token_id in token_ids_to_decode]
    return plausible_words


@pass_empty_text
@with_cleared_double_hash_tokens
def iterative_mask_fill(
    text: Text,
    model: Any,
    tokenizer: Any,
    top_k: int,
    device: str | torch.device,
    *,
    seed: int | np.random.Generator | None = None,
) -> Text:
    """Iteratively masks words in a randomly selected sentence and replaces them with language model predictions.

    Args:
        text: The input text.
        model: The masked language model used for making predictions.
        tokenizer: The tokenizer that will be used to encode text for the model and decode the model's output.
        top_k: The number of candidate words to replace the masked word at each iteration
        device: The device to use for computation (e.g., "cpu", "cuda:1", torch.device("cuda")).
        seed: The seed for a random number generator. Can be None, an int, or an instance of np.random.Generator.

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
    rng = check_rng(seed)
    return _iterative_mask_fill(text, model, tokenizer, top_k, device, rng)


@autopsy_text
def _iterative_mask_fill(
    sentences: list[Sentence],
    model: Any,
    tokenizer: Any,
    top_k: int,
    device: str | torch.device,
    rng: np.random.Generator,
) -> list[Sentence]:
    """Iteratively masks words in a randomly selected sentence and replaces them with language model predictions."""
    index = rng.integers(0, len(sentences))
    sentences[index] = _iterative_mask_fill_in_sentence(sentences[index], model, tokenizer, top_k, device, rng)
    return sentences


@autopsy_sentence
def _iterative_mask_fill_in_sentence(
    words: list[Word],
    model: Any,
    tokenizer: Any,
    top_k: int,
    device: str | torch.device,
    rng: np.random.Generator,
) -> list[Word]:
    """Iteratively masks each word in the list of words and replaces it with language model predictions."""
    for masking_index, word in enumerate(words):
        words[masking_index] = tokenizer.mask_token
        sentence_with_masking = join_words_into_sentence(words)
        plausible_words = _predict_masks(sentence_with_masking, model, tokenizer, top_k, device, rng)
        plausible_word_iter = iter(plausible_words)
        plausible_word = next(plausible_word_iter, word)
        words[masking_index] = plausible_word
    return words


@pass_empty_text
@with_cleared_double_hash_tokens
def replace_contextual_words(
    text: Text,
    model: Any,
    tokenizer: Any,
    masking_prob: float,
    top_k: int,
    device: str | torch.device,
    *,
    seed: int | np.random.Generator | None = None,
) -> Text:
    """Randomly replaces words in the text with mask tokens and fills them with language model predictions.

    Args:
        text: The input text.
        model: The masked language model used for making predictions.
        tokenizer: The tokenizer that will be used to encode text for the model and decode the model's output.
        masking_prob: The probability of masking a word.
        top_k: The number of candidate words to replace the masked word at each iteration
        device: The device to use for computation (e.g., "cpu", "cuda:1", torch.device("cuda")).
        seed: The seed for a random number generator. Can be None, an int, or an instance of np.random.Generator.

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
    rng = check_rng(seed)
    return _replace_contextual_words(text, model, tokenizer, masking_prob, top_k, device, rng)


@autopsy_text
def _replace_contextual_words(
    sentences: list[Sentence],
    model: Any,
    tokenizer: Any,
    masking_prob: float,
    top_k: int,
    device: str | torch.device,
    rng: np.random.Generator,
) -> list[Sentence]:
    """Randomly replaces words in each sentence with mask tokens and fills them with language model predictions."""
    return [
        _replace_contextual_words_in_sentence(sentence, model, tokenizer, masking_prob, top_k, device, rng)
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
    rng: np.random.Generator,
) -> list[Word]:
    """Randomly replaces words in the list of words with mask tokens and fills them with language model predictions."""
    mask_token = tokenizer.mask_token
    replacement_mask = _generate_boolean_mask(len(words), masking_prob, rng).tolist()
    words_with_masking = [mask_token if should_mask else word for word, should_mask in zip(words, replacement_mask)]
    sentence_with_masking = join_words_into_sentence(words_with_masking)
    plausible_words = _predict_masks(sentence_with_masking, model, tokenizer, top_k, device, rng)
    plausible_word_iter = iter(plausible_words)
    augmented_words = [
        next(plausible_word_iter, words[index]) if word == mask_token else word
        for index, word in enumerate(words_with_masking)
    ]
    return augmented_words
