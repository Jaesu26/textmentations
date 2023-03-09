import json
import os

WORDNET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wordnet.json")

with open(WORDNET_PATH, encoding="utf-8") as f:
    WORDNET = json.load(f)
