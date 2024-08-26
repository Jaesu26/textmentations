import pytest

import textmentations as T

# TODO: TextTransform의 subclass로 대체
AUGMENTATIONS = [
    T.AEDA,
    T.BackTranslation,
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


@pytest.fixture
def long_text():
    return (
        "동해물과 백두산이 마르고 닳도록. 하느님이 보우하사 우리나라 만세. "
        "무궁화 삼천리 화려 강산. 대한 사람 대한으로 길이 보전하세. "
        "남산 위에 저 소나무 철갑을 두른 듯. 바람 서리 불변함은 우리 기상일세. "
        "무궁화 삼천리 화려 강산. 대한 사람 대한으로 길이 보전하세. "
        "가을 하늘 공활한데 높고 구름 없이. 밝은 달은 우리 가슴 일편단심일세. "
        "무궁화 삼천리 화려 강산. 대한 사람 대한으로 길이 보전하세. "
        "이 기상과 이 맘으로 충성을 다하여. 괴로우나 즐거우나 나라 사랑하세. "
        "무궁화 삼천리 화려 강산. 대한 사람 대한으로 길이 보전하세."
    )


@pytest.fixture(params=["text_with_synonyms", "text_without_synonyms"])
def text(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def text_with_synonyms():
    return "어제 식당에 갔다. 목이 너무 말랐다. 먼저 물 한잔을 마셨다. 그리고 탕수육을 맛있게 먹었다."


@pytest.fixture
def text_without_synonyms():
    return "짜장면을 맛있게 먹었다. 짬뽕도 맛있게 먹었다. 짬짜면도 먹고 싶었다."
