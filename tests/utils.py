import random


def get_units(augmentation_cls):
    units = list(augmentation_cls().units.keys())
    return units


def set_seed(seed):
    random.seed(seed)
