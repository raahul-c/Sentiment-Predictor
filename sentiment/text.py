"""Text cleaning, tokenisation and the vocabulary (slide 7: text preprocessing)."""
import re
from collections import Counter

PAD, UNK = "<pad>", "<unk>"

# Keep negation contractions as one token ("wouldn't" -> "wouldnt") so the
# GRU sees the negation instead of a dangling "t".
_CONTRACTION = re.compile(r"(\w)'(\w)")
_URL = re.compile(r"https?://\S+|www\.\S+")
_TOKEN = re.compile(r"[a-z0-9]+|[!?]")


def clean(text):
    text = str(text).lower()
    text = _URL.sub(" url ", text)
    text = text.replace("\\n", " ").replace("’", "'")
    text = _CONTRACTION.sub(r"\1\2", text)
    return text


def tokenize(text):
    return _TOKEN.findall(clean(text))


class Vocab:
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, token_lists, max_size=30000, min_freq=2):
        counts = Counter(t for toks in token_lists for t in toks)
        words = [w for w, c in counts.most_common(max_size) if c >= min_freq]
        return cls([PAD, UNK] + words)

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens, max_len):
        unk = self.stoi[UNK]
        ids = [self.stoi.get(t, unk) for t in tokens[:max_len]]
        return ids or [unk]
