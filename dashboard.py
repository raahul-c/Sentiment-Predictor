
import argparse
import html
import webbrowser
from pathlib import Path

import pandas as pd

CLASSES = ["Negative", "Neutral", "Positive"]
GROUPS = [("City", "City"), ("Area", "Area"), ("Food Type", "Food type"), ("Food Item", "Food item")]

CSS = """
:root { --bg:#f7f7f4; --panel:#ffffff; --fg:#1d2321; --muted:#5f6b66; --line:#dfe3df; --accent:#2f6f8f;
        --rating:#a3aaa6; --bad:#b4442f; --grid:#eceee9; }
@media (prefers-color-scheme: dark) { :root { --bg:#151918; --panel:#1d2321; --fg:#e8ece9; --muted:#9aa5a0;
        --line:#323a37; --accent:#6fb3d4; --rating:#6d7773; --bad:#e2775f; --grid:#262d2a; color-scheme: dark; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 15px/1.5 "Segoe UI", system-ui, sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 32px 16px 48px; display: grid; gap: 20px; }
h1 { font: 400 30px/1.2 Georgia, "Times New Roman", serif; margin: 0; }
.take { color: var(--muted); margin: 6px 0 0; max-width: 75ch; }
.tiles { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 18px; min-width: 0; }
.lbl, .sub, .note { font-size: 13px; color: var(--muted); }
.num { font: 400 32px/1.2 Georgia, serif; margin: 4px 0; font-variant-numeric: tabular-nums; }
h2 { font-size: 17px; font-weight: 600; margin: 0; }
.note { margin: 4px 0 14px; }
.two { display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); }
.legend { display: flex; gap: 16px; font-size: 13px; color: var(--muted); margin-bottom: 8px; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
svg text { fill: var(--fg); font-size: 12px; }
svg .tick { fill: var(--muted); }
.scroll { overflow-x: auto; }
.heat { border-collapse: separate; border-spacing: 4px; width: 100%; font-variant-numeric: tabular-nums; }
.heat th { font-size: 13px; font-weight: 400; color: var(--muted); padding: 4px; }
.heat th.r { text-align: right; white-space: nowrap; }
.heat td { text-align: center; border-radius: 6px; padding: 10px 6px; min-width: 76px; }
.heat td b { display: block; font-size: 18px; font-weight: 600; }
.heat td small { font-size: 12px; }
.heat td.diag { outline: 2px solid var(--fg); }
.tabs { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px; }
.tabs button { font: inherit; font-size: 13px; border: 1px solid var(--line); background: none; color: var(--muted);
               padding: 4px 10px; border-radius: 6px; cursor: pointer; }
.tabs button[aria-pressed="true"] { background: var(--fg); color: var(--bg); border-color: var(--fg); }
table.g { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 14px; }
table.g th { font-size: 12px; font-weight: 400; color: var(--muted); text-align: right; padding: 4px 8px;
             border-bottom: 1px solid var(--line); white-space: nowrap; }
table.g td { text-align: right; padding: 4px 8px; border-bottom: 1px solid var(--line); white-space: nowrap; }
table.g th:first-child, table.g td:first-child { text-align: left; }
.neg { color: var(--bad); }
.boxed { max-height: 420px; overflow: auto; }
.method { font-size: 13px; color: var(--muted); max-width: 80ch; }
"""

JS = """
document.querySelectorAll('.tabs button').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('.tabs button').forEach(x => x.setAttribute('aria-pressed', x === b));
  document.querySelectorAll('[data-group]').forEach(t => t.hidden = t.dataset.group !== b.dataset.group);
}));
"""


def esc(v):
    return html.escape(str(v))


def review_chart(per):
    """Dumbbell chart: one row per review, Avg Rating vs text rating on a 1-5 star axis."""
    W, row_h, top, left, right = 1000, 42, 26, 8, 16
    H = top + len(per) * row_h + 4
    x = lambda v: left + (v - 1) / 4 * (W - left - right)
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Avg Rating vs text rating per review">']
    for t in range(1, 6):
        anchor = "start" if t == 1 else "end" if t == 5 else "middle"
        out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 4}" y2="{H - 4}" stroke="var(--grid)"/>'
                   f'<text class="tick" x="{x(t):.1f}" y="13" text-anchor="{anchor}">{t}★</text>')
    for i, r in enumerate(per.itertuples()):
        y = top + i * row_h
        cy = y + 28
        a, b = r.avg_rating, r.stars
        out.append(f'<g><title>{esc(r.review)}: Avg Rating {a:.2f}★, text rating {b:.2f}★ ({esc(r.sentiment)})</title>'
                   f'<text x="{left}" y="{y + 13}">{esc(r.review)} · {r.rows:,} orders · {esc(r.sentiment)}</text>'
                   f'<line x1="{x(min(a, b)):.1f}" x2="{x(max(a, b)):.1f}" y1="{cy}" y2="{cy}" '
                   f'stroke="var(--muted)" stroke-width="2" stroke-linecap="round"/>'
                   f'<circle cx="{x(a):.1f}" cy="{cy}" r="6" fill="var(--rating)"/>'
                   f'<circle cx="{x(b):.1f}" cy="{cy}" r="6.5" fill="var(--accent)"/></g>')
    out.append("</svg>")
    return "".join(out)


def heatmap(df):
    ct = pd.crosstab(df["Rating Sentiment"], df["Predicted Sentiment"]).reindex(
        index=CLASSES[::-1], columns=CLASSES, fill_value=0)
    peak = max(int(ct.values.max()), 1)
    rows = ['<tr><th class="r">Rating says ↓ / text says →</th>' + "".join(f"<th>{c}</th>" for c in CLASSES) + "</tr>"]
    for a in ct.index:
        total = int(ct.loc[a].sum())
        cells = []
        for b in CLASSES:
            n = int(ct.loc[a, b])
            mix = round(6 + 54 * n / peak)
            share = f"{n / total:.0%} of row" if total else "–"
            cls = ' class="diag"' if a == b else ""
            cells.append(f'<td{cls} style="background:color-mix(in srgb, var(--accent) {mix}%, transparent)">'
                         f"<b>{n:,}</b><small>{share}</small></td>")
        rows.append(f'<tr><th class="r">{a}</th>{"".join(cells)}</tr>')
    return f'<table class="heat">{"".join(rows)}</table>'


def group_tables(df):
    tabs, tables = [], []
    for i, (col, name) in enumerate(GROUPS):
        if col not in df:
            continue
        g = df.groupby(col).agg(rows=("Avg Rating", "size"), avg=("Avg Rating", "mean"), stars=("Predicted Stars", "mean"),
                                pos=("Predicted Sentiment", lambda s: (s == "Positive").mean()))
        g["gap"] = g["stars"] - g["avg"]
        g = g.sort_values("gap")
        body = "".join(
            f"<tr><td>{esc(k)}</td><td>{int(r.rows):,}</td><td>{r.avg:.2f}</td><td>{r.stars:.2f}</td>"
            f'<td class="{"neg" if r.gap < 0 else ""}">{r.gap:+.2f}</td><td>{r.pos:.1%}</td></tr>'
            for k, r in g.iterrows())
        first = not tabs
        tabs.append(f'<button type="button" data-group="{i}" aria-pressed="{str(first).lower()}">{name}</button>')
        tables.append(
            f'<div class="boxed" data-group="{i}"{"" if first else " hidden"}><table class="g"><thead><tr>'
            f"<th>{name}</th><th>Orders</th><th>Avg Rating</th><th>Text rating</th><th>Gap</th><th>Text positive</th>"
            f"</tr></thead><tbody>{body}</tbody></table></div>")
    return f'<div class="tabs">{"".join(tabs)}</div>{"".join(tables)}'


def build(df, source):
    df = df.copy()
    df["Avg Rating"] = pd.to_numeric(df["Avg Rating"], errors="coerce")
    df = df.dropna(subset=["Avg Rating"])
    n = len(df)
    agree = (df["Rating Sentiment"] == df["Predicted Sentiment"]).mean()
    rating_pos = (df["Rating Sentiment"] == "Positive").mean()
    text = df["Predicted Sentiment"].value_counts(normalize=True).reindex(CLASSES, fill_value=0)
    contra = int(((df["Avg Rating"] >= 4) & (df["Predicted Sentiment"] == "Negative")).sum())
    per = (df.groupby("Review").agg(rows=("Avg Rating", "size"), avg_rating=("Avg Rating", "mean"),
                                    stars=("Predicted Stars", "mean"), sentiment=("Predicted Sentiment", "first"))
           .reset_index().rename(columns={"Review": "review"}).sort_values("stars", ascending=False))
    per_note = ""
    if len(per) > 60:
        per_note = f" Showing the 30 most positive and 30 most negative of {len(per):,} distinct reviews."
        per = pd.concat([per.head(30), per.tail(30)])

    take = (f"Avg Rating calls {rating_pos:.0%} of orders positive; the review text says {text['Positive']:.0%} positive "
            f"and {text['Negative']:.0%} negative. They agree on only {agree:.0%} of orders.")
    tiles = [
        ("Avg Rating in the data", f"{df['Avg Rating'].mean():.2f} ★", "Mean over all orders"),
        ("Rating from review text", f"{df['Predicted Stars'].mean():.2f} ★", "Mean model stars, same orders"),
        ("Rating and text agree", f"{agree:.1%}", "Same Positive / Neutral / Negative"),
        ("Rated 4★+ but review is negative", f"{contra:,}", f"{contra / n:.1%} of all orders"),
    ]
    tiles_html = "".join(f'<div class="card"><div class="lbl">{a}</div><div class="num">{b}</div>'
                         f'<div class="sub">{c}</div></div>' for a, b, c in tiles)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Swiggy rating vs review sentiment</title><style>{CSS}</style></head>
<body><main>
<header><h1>Swiggy rating vs review sentiment</h1><p class="take">{take}</p></header>
<section class="tiles">{tiles_html}</section>
<section class="card"><h2>Each review: star rating vs what the words say</h2>
<p class="note">One row per distinct review text, sorted by the model's rating. Hover a row for its numbers.{per_note}</p>
<div class="legend"><span><i style="background:var(--rating)"></i>Avg Rating</span>
<span><i style="background:var(--accent)"></i>Text rating (model)</span></div>
<div class="scroll">{review_chart(per)}</div></section>
<div class="two">
<section class="card"><h2>Where rating and text disagree</h2>
<p class="note">Orders by the sentiment their Avg Rating implies (rows) and the sentiment of their review text (columns). Outlined cells are agreement.</p>
<div class="scroll">{heatmap(df)}</div></section>
<section class="card"><h2>By group</h2>
<p class="note">Gap = text rating minus Avg Rating; negative means the stars flatter the reviews.</p>
{group_tables(df)}</section>
</div>
<section class="method">
<p><b>Text rating</b> = 3 + 2 × (P(positive) − P(negative)) from the BiGRU model: 1★ certainly negative, 3★ neutral, 5★ certainly positive.
<b>Rating sentiment</b> uses Avg Rating: 4.0 or more Positive, 2.5 or less Negative, otherwise Neutral (as set in predict.py).</p>
<p>Built from {esc(source)} ({n:,} orders).</p>
</section>
</main><script>{JS}</script></body></html>"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="results/swiggy_predictions.csv", help="predict.py output CSV")
    p.add_argument("--out", default="results/swiggy_dashboard.html")
    p.add_argument("--no-open", action="store_true", help="don't open the page in the browser")
    a = p.parse_args()
    if not Path(a.input).exists():
        raise SystemExit(f"{a.input} not found. Run: python predict.py --input data/swiggy.csv --text-col Review")
    df = pd.read_csv(a.input)
    need = ["Review", "Avg Rating", "Predicted Sentiment", "Predicted Stars", "Rating Sentiment"]
    missing = [c for c in need if c not in df]
    if missing:
        raise SystemExit(f"{a.input} is missing columns {missing}. Re-run predict.py on a CSV with "
                         f"'Review' and 'Avg Rating' columns.")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(df, Path(a.input).name), encoding="utf-8")
    print(f"Dashboard -> {out.resolve()}")
    if not a.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
