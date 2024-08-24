# Textmentations

Textmentations is a Python library for augmenting Korean text.
Inspired by [albumentations](https://github.com/albumentations-team/albumentations).
Textmentations uses the albumentations as a dependency.

## Installation

```
pip install textmentations
```

## A simple example

Textmentations provides text augmentation techniques implemented using the [TextTransform](https://github.com/Jaesu26/textmentations/blob/v1.2.1/textmentations/core/transforms_interface.py#L14),
which inherits from the albumentations [BasicTransform](https://github.com/albumentations-team/albumentations/blob/1.2.1/albumentations/core/transforms_interface.py#L54).

This allows textmentations to reuse the existing functionalities of albumentations.

```python
import textmentations as T

text = "어제 식당에 갔다. 목이 너무 말랐다. 먼저 물 한잔을 마셨다. 그리고 탕수육을 맛있게 먹었다."
rd = T.RandomDeletion(deletion_prob=0.3, min_words_per_sentence=1)
ri = T.RandomInsertion(insertion_prob=0.3, n_times=1)
rs = T.RandomSwap(n_times=3)
sr = T.SynonymReplacement(replacement_prob=0.3)
eda = T.Compose([rd, ri, rs, sr])

print(rd(text=text)["text"])
# 식당에 갔다. 목이 너무 말랐다. 먼저 물. 그리고 탕수육을 맛있게.

print(ri(text=text)["text"])
# 어제 최근 식당에 갔다. 목이 너무 말랐다. 먼저 물 한잔을 마셨다 음료수. 그리고 탕수육을 맛있게 먹었다.

print(rs(text=text)["text"])
# 어제 갔다 식당에. 목이 너무 말랐다. 물 먼저 한잔을 마셨다. 그리고 먹었다 맛있게 탕수육을.

print(sr(text=text)["text"])
# 과거 식당에 갔다. 목이 너무 말랐다. 먼저 소주 한잔을 마셨다. 그리고 탕수육을 맛있게 먹었다.

print(eda(text=text)["text"])
# 식당에 어제 과거. 너무 말랐다. 상수 한잔을 마셨다 맹물. 먹었다 그리고 맛있게.
```

## List of augmentations

- [AEDA](https://github.com/Jaesu26/textmentations/blob/v1.2.1/textmentations/augmentations/transforms.py#L14)
- [BackTranslation](https://github.com/Jaesu26/textmentations/blob/v1.2.1/textmentations/augmentations/transforms.py#L101)
- [RandomDeletion](https://github.com/Jaesu26/textmentations/blob/v1.2.1/textmentations/augmentations/transforms.py#L136)
- [RandomDeletionSentence](https://github.com/Jaesu26/textmentations/blob/v1.2.1/textmentations/augmentations/transforms.py#L200)
- [RandomInsertion](https://github.com/Jaesu26/textmentations/blob/v1.2.1/textmentations/augmentations/transforms.py#L273)
- [RandomSwap](https://github.com/Jaesu26/textmentations/blob/v1.2.1/textmentations/augmentations/transforms.py#L315)
- [RandomSwapSentence](https://github.com/Jaesu26/textmentations/blob/v1.2.1/textmentations/augmentations/transforms.py#L350)
- [SynonymReplacement](https://github.com/Jaesu26/textmentations/blob/v1.2.1/textmentations/augmentations/transforms.py#L382)

## References

- [albumentations](https://github.com/albumentations-team/albumentations)
- [AEDA: An Easier Data Augmentation Technique for Text Classification](https://arxiv.org/pdf/2108.13230.pdf)
- [EDA: Easy Data Augmentation Techniques for Boosting Performance on
Text Classification Tasks](https://arxiv.org/pdf/1901.11196.pdf)
- [Korean Stopwords](https://www.ranks.nl/stopwords/korean)
- [Korean WordNet](http://wordnet.kaist.ac.kr/)
