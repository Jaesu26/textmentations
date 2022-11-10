# Text Augmentation

albumentations 패키지를 이용한 text augmentation

## How to install

```
pip install git+https://github.com/Jaesu26/text-augmentation.git
```

## Simple Example

```python
import albumentations as A
from text_augmentation.transforms import RandomSwapWords, RandomSwapSentences, RandomDeletionWords, RandomDeletionSentences

text = "아침에는 짜장면을 맛있게 먹었다. 점심에는 짬뽕을 맛있게 먹었다. 저녁에는 짬짜면을 맛있게 먹었다."
sw = RandomSwapWords()
ss = RandomSwapSentences()
dw = RandomDeletionWords(min_words_each_sentence=1)  # default: 5
ds = RandomDeletionSentences(min_sentences=2)  # default: 3
eda = A.Compose([sw, ss, dw, ds])

print(sw(text=text)["text"])
# "짜장면을 아침에는 맛있게 먹었다. 점심에는 짬뽕을 맛있게 먹었다. 저녁에는 짬짜면을 맛있게 먹었다."

print(ss(text=text)["text"])
# "아침에는 짜장면을 맛있게 먹었다. 저녁에는 짬짜면을 맛있게 먹었다. 점심에는 짬뽕을 맛있게 먹었다."

print(dw(text=text)["text"])
# "먹었다. 저녁에는 짬짜면을 맛있게 먹었다. 점심에는 맛있게 먹었다."

print(ds(text=text)["text"])
# "아침에는 짜장면을 맛있게 먹었다. 저녁에는 짬짜면을 맛있게 먹었다."

print(eda(text=text)["text"])
# 저녁에는 먹었다 짬짜면을. 점심에는 짬뽕을."
```

## Reference

- [albumentations](https://github.com/albumentations-team/albumentations)
