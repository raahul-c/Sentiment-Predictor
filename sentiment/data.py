"""Loading any CSV of comments, label generation and structured features (slides 5-7)."""
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CLASSES = ["Negative", "Neutral", "Positive"]

_LABEL_ALIASES = {
    "negative": 0, "neg": 0, "0": 0, "-1": 0,
    "neutral": 1, "neu": 1, "1": 1,
    "positive": 2, "pos": 2, "2": 2,
}


def read_csv(path, **kw):
    return pd.read_csv(path, encoding_errors="replace", on_bad_lines="skip", **kw)


def labels_from_column(series):
    """Text labels such as positive/neutral/negative (or 0/1/2) -> class ids."""
    mapped = series.astype(str).str.strip().str.lower().map(_LABEL_ALIASES)
    bad = series[mapped.isna()].unique()[:5]
    if len(bad):
        print(f"  skipping rows with unrecognised labels, e.g. {list(bad)}")
    return mapped


def labels_from_rating(series, neg_max=2.5, pos_min=4.0):
    """Slide 6 rule: rating >= pos_min -> Positive, <= neg_max -> Negative, else Neutral."""
    r = pd.to_numeric(series, errors="coerce")
    out = pd.Series(1.0, index=r.index)
    out[r <= neg_max] = 0
    out[r >= pos_min] = 2
    out[r.isna()] = np.nan
    return out


def parse_number(value):
    """'30-40 min' -> 35, '₹600' -> 600, '4.2' -> 4.2."""
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(value))]
    return float(np.mean(nums)) if nums else np.nan


class StructEncoder:
    """One-hot categoricals + clipped, standardised numerics (the structured branch)."""

    def __init__(self, cat_cols=(), num_cols=()):
        self.cat_cols, self.num_cols = list(cat_cols), list(num_cols)
        self.categories, self.stats = {}, {}

    @property
    def size(self):
        return sum(len(v) for v in self.categories.values()) + len(self.num_cols)

    def fit(self, df):
        for c in self.cat_cols:
            self.categories[c] = sorted(df[c].astype(str).str.strip().str.title().unique())
        for c in self.num_cols:
            v = df[c].map(parse_number)
            lo, hi = v.quantile(0.01), v.quantile(0.99)
            v = v.clip(lo, hi)
            self.stats[c] = (lo, hi, v.mean(), v.std() or 1.0, v.median())
        return self

    def transform(self, df):
        parts = []
        for c in self.cat_cols:
            vals = df[c].astype(str).str.strip().str.title()
            parts.append(np.stack([(vals == k).to_numpy(float) for k in self.categories[c]], 1))
        for c in self.num_cols:
            lo, hi, mean, std, median = self.stats[c]
            v = df[c].map(parse_number).fillna(median).clip(lo, hi)
            parts.append(((v - mean) / std).to_numpy(float)[:, None])
        if not parts:
            return np.zeros((len(df), 0), dtype=np.float32)
        return np.concatenate(parts, 1).astype(np.float32)

    def state(self):
        return {"cat_cols": self.cat_cols, "num_cols": self.num_cols,
                "categories": self.categories, "stats": self.stats}

    @classmethod
    def from_state(cls, s):
        enc = cls(s["cat_cols"], s["num_cols"])
        enc.categories, enc.stats = s["categories"], s["stats"]
        return enc


class CommentDataset(Dataset):
    def __init__(self, ids, struct, labels=None):
        self.ids, self.struct, self.labels = ids, struct, labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        y = -1 if self.labels is None else self.labels[i]
        return self.ids[i], self.struct[i], y


class BucketSampler(torch.utils.data.Sampler):
    """Batches of similar-length comments, so little time is wasted on padding."""

    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths, self.batch_size, self.shuffle = np.asarray(lengths), batch_size, shuffle

    def __iter__(self):
        idx = np.random.permutation(len(self.lengths)) if self.shuffle else np.arange(len(self.lengths))
        chunk = self.batch_size * 50
        batches = []
        for s in range(0, len(idx), chunk):
            part = idx[s:s + chunk]
            part = part[np.argsort(self.lengths[part], kind="stable")]
            batches += [part[i:i + self.batch_size].tolist() for i in range(0, len(part), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def collate(batch):
    ids, struct, y = zip(*batch)
    lengths = torch.tensor([len(x) for x in ids])
    padded = torch.zeros(len(ids), int(lengths.max()), dtype=torch.long)
    for i, x in enumerate(ids):
        padded[i, :len(x)] = torch.tensor(x)
    return padded, lengths, torch.tensor(np.stack(struct)), torch.tensor(y)
