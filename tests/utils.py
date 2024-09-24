import random

import numpy as np

MASK_TOKEN = "[MASK]"


def contains_mask_token(text):
    return MASK_TOKEN in text


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
