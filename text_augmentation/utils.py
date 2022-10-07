from typing import List, Sequence

__all__ = [
    "strip",
    "get_words_from_sentence",
    "get_sentences_from_text",
    "get_sentence_from_words",
    "get_text_from_sentences",
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

  
def get_words_from_sentence(sentence: Sentence) -> List[Word]:
        """Split the sentence to get words"""
        words = strip(sentence.split())
        return words

      
def get_sentences_from_text(text: Text) -> List[Sentence]:
    """Split the text to get sentences"""
    sentences = strip(text.split("."))
    if text.endswith("."):
        return sentences[:-1]
    return sentences 

  
def get_sentence_from_words(words: List[Word]) -> Sentence:
    """Combine words to get a sentence"""
    sentence = " ".join(words)
    return sentence

  
def get_text_from_sentences(sentences: List[Sentence]) -> Text:
    """Combine sentences to get a text"""
    text = ". ".join(sentences)
    if text:
        text = ".".join([text, ""])
    return text
