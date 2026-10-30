
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from sentiment.data import (CLASSES, BucketSampler, CommentDataset, StructEncoder, collate,
                            labels_from_column, labels_from_rating, read_csv)
from sentiment.device import is_directml, pick_device
from sentiment.model import SentimentGRU
from sentiment.optim import PlainAdamW
from sentiment.text import Vocab, tokenize


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", required=True, help="training CSV file(s)")
    p.add_argument("--test-data", help="separate labelled CSV for the final evaluation (default: hold out --test-size)")
    p.add_argument("--text-col", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--label-col", help="column with positive/neutral/negative labels")
    g.add_argument("--rating-col", help="numeric rating column to derive labels from")
    p.add_argument("--neg-max", type=float, default=2.5, help="rating <= this is Negative")
    p.add_argument("--pos-min", type=float, default=4.0, help="rating >= this is Positive")
    p.add_argument("--cat-cols", nargs="*", default=[], help="categorical columns for the hybrid branch")
    p.add_argument("--num-cols", nargs="*", default=[], help="numeric columns for the hybrid branch")
    p.add_argument("--max-rows", type=int, help="randomly sample at most this many training rows")
    p.add_argument("--test-size", type=float, default=0.1)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=200, help="tokens kept per comment")
    p.add_argument("--crop-prob", type=float, default=0.25,
                   help="share of long training reviews replaced by a short random snippet (0 = off)")
    p.add_argument("--vocab-size", type=int, default=30000)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--glove", help="optional GloVe .txt file; its dimension overrides --embed-dim")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--patience", type=int, default=2, help="stop after this many epochs without val F1 gain")
    p.add_argument("--device", default="auto",
                   help="auto (CUDA, then DirectML, then CPU), cuda, dml, cpu. AMD ROCm on Linux counts as cuda")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default="models/sentiment_gru.pt")
    p.add_argument("--resume", help="saved model to continue training from (new data or more epochs)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load(paths, a):
    df = pd.concat([read_csv(p) for p in paths], ignore_index=True)
    missing = [c for c in [a.text_col, a.label_col or a.rating_col, *a.cat_cols, *a.num_cols] if c not in df]
    if missing:
        raise SystemExit(f"Columns not found: {missing}. Available: {list(df.columns)}")
    df["_y"] = (labels_from_column(df[a.label_col]) if a.label_col
                else labels_from_rating(df[a.rating_col], a.neg_max, a.pos_min))
    df = df.dropna(subset=["_y", a.text_col])
    df = df[df[a.text_col].astype(str).str.strip() != ""]
    df["_y"] = df["_y"].astype(int)
    return df.reset_index(drop=True)


def load_glove(path, vocab):
    vectors, dim = {}, None
    with open(path, encoding="utf8") as f:
        for line in f:
            word, *vals = line.rstrip().split(" ")
            if word in vocab.stoi:
                vectors[word] = np.asarray(vals, dtype=np.float32)
                dim = len(vals)
    weights = np.random.normal(0, 0.1, (len(vocab), dim)).astype(np.float32)
    weights[0] = 0
    for w, v in vectors.items():
        weights[vocab.stoi[w]] = v
    print(f"  GloVe: {len(vectors)}/{len(vocab)} vocabulary words found (dim {dim})")
    return torch.from_numpy(weights)


def random_crop(ids, prob, rng):
    """Replace some long reviews with a short random window of themselves, so the model
    also learns to judge one-line comments (like most of swiggy.csv), not only long reviews."""
    out = []
    for x in ids:
        if len(x) > 30 and rng.random() < prob:
            n = int(rng.integers(4, 21))
            s = int(rng.integers(0, len(x) - n))
            x = x[s:s + n]
        out.append(x)
    return out


def make_loader(df, a, vocab, enc, shuffle, crop_prob=0.0):
    ids = [vocab.encode(tokenize(t), a.max_len) for t in df[a.text_col]]
    if crop_prob:
        ids = random_crop(ids, crop_prob, np.random.default_rng(a.seed))
    ds = CommentDataset(ids, enc.transform(df), df["_y"].to_numpy())
    sampler = BucketSampler([len(x) for x in ids], a.batch_size, shuffle)
    return DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=a.workers)


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    probs, ys = [], []
    for ids, lengths, struct, y in loader:
        logits = model(ids.to(device), lengths.to(device), struct.to(device))
        probs.append(torch.softmax(logits, 1).cpu())
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def save_confusion_png(cm, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  confusion matrix image -> {path}")


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device, device_name = pick_device(a.device)
    print(f"Device: {device_name}")

    print("Loading data ...")
    df = load(a.data, a)
    if a.max_rows and len(df) > a.max_rows:
        df = df.sample(a.max_rows, random_state=a.seed).reset_index(drop=True)
    if a.test_data:
        test = load([a.test_data], a)
        train = df
    else:
        train, test = train_test_split(df, test_size=a.test_size, stratify=df["_y"], random_state=a.seed)
    train, val = train_test_split(train, test_size=a.val_size, stratify=train["_y"], random_state=a.seed)
    counts = np.bincount(train["_y"], minlength=3)
    print(f"  train {len(train)}  val {len(val)}  test {len(test)}  |  class counts "
          + ", ".join(f"{c}={n}" for c, n in zip(CLASSES, counts)))

    resume = torch.load(a.resume, map_location="cpu", weights_only=False) if a.resume else None
    if resume:
        # Continue from a saved model: keep its vocabulary, feature encoding and sizes
        vocab, enc = Vocab(resume["vocab"]), StructEncoder.from_state(resume["struct"])
        a.embed_dim, a.hidden, a.max_len = (resume["config"][k] for k in ["embed_dim", "hidden", "max_len"])
        a.glove = None
        print(f"  resuming from {a.resume}")
    else:
        vocab = Vocab.build((tokenize(t) for t in train[a.text_col]), max_size=a.vocab_size)
        enc = StructEncoder(a.cat_cols, a.num_cols).fit(train)
    print(f"  vocabulary {len(vocab)} words, {enc.size} structured features")
    pretrained = load_glove(a.glove, vocab) if a.glove else None
    embed_dim = pretrained.shape[1] if pretrained is not None else a.embed_dim

    train_dl = make_loader(train, a, vocab, enc, True, a.crop_prob)
    val_dl = make_loader(val, a, vocab, enc, False)
    test_dl = make_loader(test, a, vocab, enc, False)

    model = SentimentGRU(len(vocab), 3, embed_dim, a.hidden, a.dropout, enc.size, pretrained)
    if resume:
        model.load_state_dict(resume["model"])
    model.use_packing = not is_directml(device)  # DirectML lacks packed-sequence support
    model.to(device)
    # Class weights so the smaller Neutral class is not ignored (imbalanced data)
    weights = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    if is_directml(device):
        opt = PlainAdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(train_dl))

    config = {k: getattr(a, k) for k in ["text_col", "max_len", "hidden", "dropout"]}
    config.update(embed_dim=embed_dim, n_struct=enc.size, classes=CLASSES)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    best_f1, bad_epochs = -1.0, 0
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, total, seen = time.time(), 0.0, 0
        for step, (ids, lengths, struct, y) in enumerate(train_dl, 1):
            ids, lengths, struct, y = ids.to(device), lengths.to(device), struct.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(ids, lengths, struct), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(y)
            seen += len(y)
            if step % 200 == 0:
                print(f"    epoch {epoch} step {step}/{len(train_dl)} loss {total / seen:.4f}", flush=True)
        probs, ys = predict_loader(model, val_dl, device)
        f1 = f1_score(ys, probs.argmax(1), average="macro")
        print(f"  epoch {epoch}: train loss {total / seen:.4f}  val acc {accuracy_score(ys, probs.argmax(1)):.4f}"
              f"  val macro-F1 {f1:.4f}  ({time.time() - t0:.0f}s)")
        if f1 > best_f1:
            best_f1, bad_epochs = f1, 0
            torch.save({"model": model.state_dict(), "vocab": vocab.itos,
                        "struct": enc.state(), "config": config}, out)
        else:
            bad_epochs += 1
            if bad_epochs >= a.patience:
                print("  early stopping")
                break

    # Final evaluation on the held-out test set (slide 10 metrics)
    model.load_state_dict(torch.load(out, map_location="cpu", weights_only=False)["model"])
    probs, ys = predict_loader(model, test_dl, device)
    pred = probs.argmax(1)
    cm = confusion_matrix(ys, pred, labels=[0, 1, 2])
    report = classification_report(ys, pred, labels=[0, 1, 2], target_names=CLASSES, digits=4, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(ys, pred),
        "macro_f1": f1_score(ys, pred, average="macro"),
        "roc_auc_ovr": roc_auc_score(ys, probs, multi_class="ovr") if len(set(ys)) == 3 else None,
        "confusion_matrix": cm.tolist(),
        "report": classification_report(ys, pred, labels=[0, 1, 2], target_names=CLASSES,
                                        output_dict=True, zero_division=0),
    }
    print("\nTest set results")
    print(report)
    print("Confusion matrix (rows = actual, cols = predicted:", ", ".join(CLASSES) + ")")
    print(cm)
    if "source" in test:
        for src, ok in pd.Series(pred == ys, index=test.index).groupby(test["source"]):
            print(f"  accuracy on {src}: {ok.mean():.4f} ({len(ok)} rows)")
    if metrics["roc_auc_ovr"] is not None:
        print(f"ROC AUC (one-vs-rest): {metrics['roc_auc_ovr']:.4f}")
    Path(out.with_suffix(".metrics.json")).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_png(cm, out.with_suffix(".confusion.png"))
    print(f"\nModel saved to {out}")


if __name__ == "__main__":
    main()
