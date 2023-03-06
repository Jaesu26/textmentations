# Textmentations

Textmentations is a Python library for text augmentation that is specialized for the Korean language. 
Inspired by [albumentations](https://github.com/albumentations-team/albumentations). 
Textmentations uses the albumentations as a dependency.

## Installation

```
pip install git+https://github.com/Jaesu26/textmentations.git
```

## A simple example

Textmentations provides various text augmentation techniques implemented using the [TextTransform](https://github.com/Jaesu26/textmentations/blob/main/textmentations/core/transforms_interface.py#L17), 
which inherits from the albumentations [BasicTransform](https://github.com/albumentations-team/albumentations/blob/1.2.1/albumentations/core/transforms_interface.py#L54). 

This allows textmentations to reuse the existing functionalities of albumentations.

```python
from albumentations import Compose
from textmentations import RandomDeletionWords, RandomDeletionSentences, RandomSwapWords, RandomSwapSentences

text = "아침에는 짜장면을 맛있게 먹었다. 점심에는 짬뽕을 맛있게 먹었다. 저녁에는 짬짜면을 맛있게 먹었다."
dw = RandomDeletionWords(deletion_prob=0.5, min_words_each_sentence=1)
ds = RandomDeletionSentences(deletion_prob=0.5, min_sentences=2)
sw = RandomSwapWords()
ss = RandomSwapSentences()
mixed_transforms = Compose([sw, ss, dw, ds])

print(dw(text=text)["text"])
# 먹었다. 점심에는 맛있게 먹었다. 저녁에는 짬짜면을 맛있게 먹었다.

print(ds(text=text)["text"])
# 아침에는 짜장면을 맛있게 먹었다. 저녁에는 짬짜면을 맛있게 먹었다.

print(sw(text=text)["text"])
# 짜장면을 아침에는 맛있게 먹었다. 점심에는 짬뽕을 맛있게 먹었다. 저녁에는 짬짜면을 맛있게 먹었다.

print(ss(text=text)["text"])
# 아침에는 짜장면을 맛있게 먹었다. 저녁에는 짬짜면을 맛있게 먹었다. 점심에는 짬뽕을 맛있게 먹었다.

print(mixed_transforms(text=text)["text"])
# 저녁에는 먹었다 짬짜면을. 점심에는 짬뽕을.
```

## List of augmentations

- `RandomDeletionWords`
- `RandomDeletionSentences`
- `RandomSwapWords`
- `RandomSwapSentences`
- `SynonymsReplacement`

## References

- [albumentations](https://github.com/albumentations-team/albumentations)

- [Korean WordNet (KWN)](http://wordnet.kaist.ac.kr/)
