from typing import List, Sequence

__all__ = [
    "strip",
    "split_sentence",
    "split_text",
    "combine_words",
    "combine_sentences",
    "Word",
    "Sentence",
    "Text",
]

Word = str
Sentence = str
Text = str


def strip(str_sequence: Sequence[str]) -> List[str]:
    """Remove leading and trailing whitespaces each string in sequence"""
    return list(map(lambda string: string.strip(), str_sequence))

  
def split_sentence(sentence: Sentence) -> List[Word]:
    """Split the sentence to get words"""
    words = strip(sentence.split())
    return words

      
def split_text(text: Text) -> List[Sentence]:
    """Split the text to get sentences"""
    sentences = strip(text.split("."))
    if text.endswith("."):
        return sentences[:-1]
    return sentences 

  
def combine_words(words: List[Word]) -> Sentence:
    """Combine words to get a sentence"""
    sentence = " ".join(words)
    return sentence

  
def combine_sentences(sentences: List[Sentence]) -> Text:
    """Combine sentences to get a text"""
    text = ". ".join(sentences)
    if text:
        text = ".".join([text, ""])
    return text
