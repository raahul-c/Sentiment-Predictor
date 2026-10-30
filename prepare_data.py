
import argparse
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

YELP_URL = "https://s3.amazonaws.com/fast-ai-nlp/yelp_review_full_csv.tgz"
TWEET_URL = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/{split}_{kind}.txt"
DATA = Path("data")
LABELS = {0: "negative", 1: "neutral", 2: "positive"}


def download(url, dest):
    if not dest.exists():
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def yelp(split, max_rows, seed):
    archive = download(YELP_URL, DATA / "yelp_review_full_csv.tgz")
    src = DATA / "yelp_review_full_csv"
    if not src.exists():
        print("  extracting Yelp archive ...")
        with tarfile.open(archive) as t:
            t.extractall(DATA, filter="data")
    df = pd.read_csv(src / f"{split}.csv", header=None, names=["stars", "text"])
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed)
    df["text"] = df["text"].str.replace(r"\\n", " ", regex=True).str.replace('\\"', '"', regex=False)
    df["sentiment"] = pd.cut(df["stars"], [0, 2, 3, 5], labels=["negative", "neutral", "positive"])
    return df.assign(source="yelp")[["text", "sentiment", "source"]]


def tweets(splits):
    parts = []
    for split in splits:
        text = download(TWEET_URL.format(split=split, kind="text"), DATA / f"te_{split}_text.txt")
        labels = download(TWEET_URL.format(split=split, kind="labels"), DATA / f"te_{split}_labels.txt")
        t = text.read_text(encoding="utf8").splitlines()
        y = [LABELS[int(v)] for v in labels.read_text().split()]
        parts.append(pd.DataFrame({"text": t, "sentiment": y}))
    df = pd.concat(parts)
    # @user mentions carry no sentiment
    df["text"] = df["text"].str.replace("@user", "", regex=False).str.strip()
    return df.assign(source="tweeteval")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yelp-rows", type=int, default=0,
                   help="sample this many Yelp training reviews (default 0 = all 650,000)")
    p.add_argument("--no-tweets", action="store_true", help="leave out TweetEval")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    DATA.mkdir(exist_ok=True)

    print("Preparing training data ...")
    train = [yelp("train", a.yelp_rows, a.seed)] + ([] if a.no_tweets else [tweets(["train", "val"])])
    test = [yelp("test", 0, a.seed)] + ([] if a.no_tweets else [tweets(["test"])])
    for name, parts in [("train", train), ("test", test)]:
        df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=a.seed)
        df.to_csv(DATA / f"{name}.csv", index=False)
        counts = df.groupby(["source", "sentiment"], observed=True).size().unstack(fill_value=0)
        print(f"\n  data/{name}.csv  {len(df)} rows\n{counts.to_string()}")


if __name__ == "__main__":
    main()
