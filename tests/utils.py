import random

MASK_TOKEN = "[MASK]"


def contains_mask_token(text):
    return MASK_TOKEN in text


def set_seed(seed):
    random.seed(seed)
