import pytest

import textmentations as T

AUGMENTATIONS = [
    T.RandomDeletion,
    T.RandomDeletionSentence,
    T.RandomInsertion,
    T.RandomSwap,
    T.RandomSwapSentence,
    T.SynonymReplacement,
]

AUGMENTATIONS_WITH_N_TIMES = [
    T.RandomInsertion,
    T.RandomSwap,
    T.RandomSwapSentence,
]

AUGMENTATIONS_WITH_PROBABILITY = [
    T.RandomDeletion,
    T.RandomDeletionSentence,
    T.RandomInsertion,
    T.SynonymReplacement,
]


@pytest.fixture(params=AUGMENTATIONS)
def augmentation(request):
    return request.param


@pytest.fixture(params=AUGMENTATIONS_WITH_N_TIMES)
def augmentation_with_n_times(request):
    return request.param


@pytest.fixture(params=AUGMENTATIONS_WITH_PROBABILITY)
def augmentation_with_probability(request):
    return request.param


@pytest.fixture(params=["text_with_synonyms", "text_without_synonyms"])
def text(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def text_with_synonyms():
    return "어제 식당에 갔다. 목이 너무 말랐다. 먼저 물 한잔을 마셨다. 그리고 탕수육을 맛있게 먹었다."


@pytest.fixture
def text_without_synonyms():
    return "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."
