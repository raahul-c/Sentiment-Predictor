
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sentiment.data import CLASSES, CommentDataset, StructEncoder, collate, labels_from_rating, read_csv
from sentiment.device import is_directml, pick_device
from sentiment.model import SentimentGRU
from sentiment.text import Vocab, tokenize

DEFAULT_GROUPS = ["City", "Area", "Food Type", "Food Item"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/sentiment_gru.pt")
    p.add_argument("--input", help="CSV file with comments to rate")
    p.add_argument("--text-col", default="Review")
    p.add_argument("--text", nargs="+", help="comment(s) to rate directly")
    p.add_argument("--out", help="output CSV (default: results/<input>_predictions.csv)")
    p.add_argument("--group-by", nargs="*", help=f"columns to summarise by (default: any of {DEFAULT_GROUPS})")
    p.add_argument("--rating-col", default="Avg Rating",
                   help="rating column to cross-check against the text sentiment, if present")
    p.add_argument("--neg-max", type=float, default=2.5)
    p.add_argument("--pos-min", type=float, default=4.0)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="auto", help="auto (CUDA, then DirectML, then CPU), cuda, dml, cpu")
    return p.parse_args()


class Predictor:
    def __init__(self, path, device="auto"):
        self.device, self.device_name = pick_device(device)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.cfg = ckpt["config"]
        self.vocab = Vocab(ckpt["vocab"])
        self.enc = StructEncoder.from_state(ckpt["struct"])
        self.model = SentimentGRU(len(self.vocab), len(CLASSES), self.cfg["embed_dim"], self.cfg["hidden"],
                                  self.cfg["dropout"], self.cfg["n_struct"])
        self.model.load_state_dict(ckpt["model"])
        self.model.use_packing = not is_directml(self.device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, texts, df=None, batch_size=512):
        """Probabilities [Negative, Neutral, Positive] for each text."""
        if self.enc.size and df is None:
            raise ValueError(f"This hybrid model also needs columns {self.enc.cat_cols + self.enc.num_cols}")
        ids = [self.vocab.encode(tokenize(t), self.cfg["max_len"]) for t in texts]
        struct = self.enc.transform(df) if self.enc.size else np.zeros((len(ids), 0), np.float32)
        loader = DataLoader(CommentDataset(ids, struct), batch_size=batch_size, collate_fn=collate)
        out = []
        for x, lengths, s, _ in loader:
            logits = self.model(x.to(self.device), lengths.to(self.device), s.to(self.device))
            out.append(torch.softmax(logits, 1).cpu())
        return torch.cat(out).numpy()


def add_predictions(df, probs):
    df = df.copy()
    df["Predicted Sentiment"] = [CLASSES[i] for i in probs.argmax(1)]
    df["Confidence"] = probs.max(1).round(4)
    for i, c in enumerate(CLASSES):
        df[f"P({c})"] = probs[:, i].round(4)
    # -1 (certainly negative) .. +1 (certainly positive), and the same on a 1-5 star scale
    df["Sentiment Score"] = (probs[:, 2] - probs[:, 0]).round(4)
    df["Predicted Stars"] = (3 + 2 * df["Sentiment Score"]).round(2)
    return df


def summarise(df, groups, a):
    lines = ["OVERALL", "-------"]
    dist = df["Predicted Sentiment"].value_counts().reindex(CLASSES, fill_value=0)
    for c, n in dist.items():
        lines.append(f"{c:<9} {n:>7}  {100 * n / len(df):5.1f}%")
    lines.append(f"Mean sentiment score {df['Sentiment Score'].mean():+.3f}   "
                 f"mean predicted stars {df['Predicted Stars'].mean():.2f}")

    lines += ["", "PER UNIQUE COMMENT (most positive first)", "----------------------------------------"]
    per_text = (df.groupby(a.text_col)
                .agg(rows=("Sentiment Score", "size"), sentiment=("Predicted Sentiment", "first"),
                     confidence=("Confidence", "mean"), score=("Sentiment Score", "mean"))
                .sort_values("score", ascending=False))
    if len(per_text) <= 200:
        lines.append(per_text.round(3).to_string())
    else:
        lines.append(f"({len(per_text)} unique comments; see the predictions CSV)")

    for g in groups:
        t = (df.groupby(g)["Predicted Sentiment"].value_counts(normalize=True)
             .unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0) * 100).round(1)
        t.columns = [f"% {c}" for c in CLASSES]
        t["Mean Score"] = df.groupby(g)["Sentiment Score"].mean().round(3)
        t["Rows"] = df.groupby(g).size()
        lines += ["", f"BY {g.upper()}", "-" * (3 + len(g)), t.sort_values("Mean Score", ascending=False).to_string()]

    if a.rating_col in df:
        # Slide 6: cross-check rating-derived sentiment against the text sentiment
        rated = labels_from_rating(df[a.rating_col], a.neg_max, a.pos_min)
        df["Rating Sentiment"] = rated.map(lambda v: CLASSES[int(v)] if v == v else "")
        df["Disagrees With Rating"] = (df["Rating Sentiment"] != df["Predicted Sentiment"]) & rated.notna()
        n = int(df["Disagrees With Rating"].sum())
        lines += ["", f"RATING vs TEXT ('{a.rating_col}': <= {a.neg_max} Negative, >= {a.pos_min} Positive)",
                  "-------------------------------------------",
                  f"{n} of {len(df)} rows ({100 * n / len(df):.1f}%) disagree; flagged in 'Disagrees With Rating'.",
                  pd.crosstab(df["Rating Sentiment"], df["Predicted Sentiment"],
                              rownames=["rating says"], colnames=["text says"]).to_string()]
    return df, "\n".join(lines)


def main():
    a = parse_args()
    if not Path(a.model).exists():
        sys.exit(f"Model not found: {a.model}. Train one first with train.py.")
    pred = Predictor(a.model, a.device)
    print(f"Device: {pred.device_name}", file=sys.stderr)

    if a.input:
        df = read_csv(a.input)
        if a.text_col not in df:
            sys.exit(f"Column '{a.text_col}' not in {a.input}. Available: {list(df.columns)}")
        df[a.text_col] = df[a.text_col].fillna("").astype(str)
        df = add_predictions(df, pred.predict_proba(df[a.text_col].tolist(), df, a.batch_size))
        groups = a.group_by if a.group_by is not None else [g for g in DEFAULT_GROUPS if g in df]
        df, report = summarise(df, groups, a)
        out = Path(a.out or f"results/{Path(a.input).stem}_predictions.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        out.with_name(out.stem.replace("_predictions", "") + "_summary.txt").write_text(report + "\n", encoding="utf-8")
        print(report)
        print(f"\nPer-row predictions -> {out}")
        print(f"Summary            -> {out.with_name(out.stem.replace('_predictions', '') + '_summary.txt')}")
        return

    texts = a.text or None
    if texts is None:
        print("Type a comment and press Enter (Ctrl+D to quit; on Windows Ctrl+Z then Enter).")
        texts = (line.strip() for line in sys.stdin)
    for t in texts:
        if not t:
            continue
        p = pred.predict_proba([t])[0]
        print(f"{CLASSES[p.argmax()]:<9} {p.max():6.1%}   "
              + "  ".join(f"{c[:3]} {v:.2f}" for c, v in zip(CLASSES, p)) + f"   | {t}", flush=True)


if __name__ == "__main__":
    main()
