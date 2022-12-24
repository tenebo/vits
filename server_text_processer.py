import sys
import re
from text.korean import normalize
import g2pk2 as g2pk

from text.symbols import symbols
g2p = g2pk.G2p()


def remove_duplicated_punctuations(text):
    text = re.sub(r"[.?!]+\?", "?", text)
    text = re.sub(r"[.?!]+!", "!", text)
    text = re.sub(r"[.?!]+\.", ".", text)
    return text


def split_text(text):
    text += '\n'
    text = remove_duplicated_punctuations(text)

    texts = []
    for subtext in re.findall(r'[^.!?\n]*[.!?\n]', text):
        texts.append(subtext.strip())

    return texts


def normalize_multiline_text(long_text):
    texts = split_text(long_text)
    normalized_texts = [normalize(text).strip() for text in texts]
    return [text for text in normalized_texts if len(text) > 0]
