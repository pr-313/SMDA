#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║     Instagram D2C Sentiment Analysis Toolkit                 ║
║     SMDA Term Paper — Shaurya Baveja & Pratyush Bharati      ║
╚══════════════════════════════════════════════════════════════╝

Usage:
  python sentiment_toolkit.py --help
  python sentiment_toolkit.py --input data.csv --mode A
  python sentiment_toolkit.py --input data.csv --mode B --field caption --top 10
  python sentiment_toolkit.py --input data.csv --mode C
  python sentiment_toolkit.py --input data.csv --mode D --freq W
  python sentiment_toolkit.py --input data.csv --mode E --min-posts 5
  python sentiment_toolkit.py --input data.csv --mode all
  python sentiment_toolkit.py --input data.csv --mode E --exclude mokobara
  python sentiment_toolkit.py --input data.csv --mode all --exclude mokobara thesouledstore

Modes:
  A  — Regression Analysis    (sentiment → engagement)
  B  — Descriptive Comparison (sentiment groups vs. likes/comments)
  C  — Alignment Analysis     (caption vs. comment sentiment gap)
  D  — Time-Series Analysis   (tone trends over time)
  E  — Brand-Level Aggregation(per-brand sentiment vs. performance)
  all— Run all five modes sequentially
"""

import argparse
import sys
import os
import re
import math
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from tabulate import tabulate
from colorama import init, Fore, Back, Style

warnings.filterwarnings("ignore")
init(autoreset=True)

# ══════════════════════════════════════════════════════════════
# PALETTE & DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════

C = {
    "header":  Fore.CYAN + Style.BRIGHT,
    "sub":     Fore.YELLOW + Style.BRIGHT,
    "pos":     Fore.GREEN + Style.BRIGHT,
    "neg":     Fore.RED + Style.BRIGHT,
    "neu":     Fore.WHITE,
    "dim":     Style.DIM,
    "bold":    Style.BRIGHT,
    "reset":   Style.RESET_ALL,
    "sep":     Fore.CYAN + Style.DIM,
    "accent":  Fore.MAGENTA + Style.BRIGHT,
    "warn":    Fore.YELLOW,
    "good":    Fore.GREEN,
    "info":    Fore.BLUE + Style.BRIGHT,
}

WIDTH = 70

def banner(title: str, subtitle: str = ""):
    print()
    print(C["sep"] + "═" * WIDTH)
    print(C["header"] + f"  {title}")
    if subtitle:
        print(C["dim"] + f"  {subtitle}")
    print(C["sep"] + "═" * WIDTH)

def section(title: str):
    print()
    print(C["sub"] + f"  ▶  {title}")
    print(C["sep"] + "  " + "─" * (WIDTH - 4))

def row_print(label, value, color=None):
    col = color or C["neu"]
    label_str = f"  {label:<32}"
    print(C["dim"] + label_str + col + str(value))

def bar_chart(label, value, max_val, width=30, color=None):
    col = color or Fore.CYAN
    filled = int((value / max_val) * width) if max_val > 0 else 0
    bar = "█" * filled + "░" * (width - filled)
    print(f"  {label:<20} {col}{bar}{C['reset']}  {value:.4f}")

def sentiment_color(label):
    if label == "positive": return C["pos"]
    if label == "negative": return C["neg"]
    return C["neu"]

def compound_bar(score, width=20):
    """Render a [-1, 1] score as a centered bar."""
    mid = width // 2
    if score >= 0:
        filled = int(score * mid)
        bar = " " * mid + Fore.GREEN + "█" * filled + "░" * (mid - filled)
    else:
        filled = int(abs(score) * mid)
        bar = " " * (mid - filled) + Fore.RED + "█" * filled + " " * mid
    return bar + C["reset"]

def print_table(df_or_rows, headers, title=None, colalign=None):
    if title:
        print(C["dim"] + f"\n  {title}")
    if isinstance(df_or_rows, pd.DataFrame):
        rows = df_or_rows.values.tolist()
    else:
        rows = df_or_rows
    kwargs = {"tablefmt": "rounded_outline", "headers": headers}
    if colalign:
        kwargs["colalign"] = colalign
    print(tabulate(rows, **kwargs))

def footer(note: str):
    print()
    print(C["sep"] + "  " + "─" * (WIDTH - 4))
    print(C["dim"] + f"  💡 {note}")
    print(C["sep"] + "═" * WIDTH)
    print()


# ══════════════════════════════════════════════════════════════
# LEXICON & SENTIMENT ENGINE
# ══════════════════════════════════════════════════════════════

LEXICON = {
    "love": 3.0, "amazing": 3.2, "excellent": 3.2, "fantastic": 3.5,
    "wonderful": 3.0, "brilliant": 3.0, "outstanding": 3.5, "superb": 3.5,
    "exceptional": 3.3, "perfect": 3.4, "beautiful": 3.0, "stunning": 3.2,
    "gorgeous": 3.2, "incredible": 3.2, "awesome": 3.2, "best": 2.8,
    "great": 2.8, "fabulous": 3.0, "spectacular": 3.2, "magnificent": 3.2,
    "delightful": 3.0, "charming": 2.8, "flawless": 3.2, "luxurious": 2.8,
    "premium": 2.5, "exquisite": 3.0, "masterpiece": 3.5, "revolutionary": 2.8,
    "innovative": 2.5, "celebrate": 2.5, "celebration": 2.5, "joy": 2.8,
    "happy": 2.8, "happiness": 2.8, "excited": 2.5, "exciting": 2.5,
    "thrilled": 2.8, "proud": 2.5, "confidence": 2.2, "confident": 2.2,
    "empowered": 2.5, "empowering": 2.5, "elevate": 2.2, "elevated": 2.2,
    "crafted": 2.0, "handcrafted": 2.2, "curated": 2.0, "inspired": 2.2,
    "inspiring": 2.5, "timeless": 2.2, "iconic": 2.5, "elegant": 2.5,
    "sleek": 2.2, "chic": 2.2, "stylish": 2.2, "trendy": 2.0,
    "fresh": 2.0, "vibrant": 2.2, "radiant": 2.2, "glowing": 2.2,
    "smooth": 1.8, "soft": 1.8, "comfortable": 2.0, "cozy": 2.0,
    "trustworthy": 2.5, "reliable": 2.2, "authentic": 2.2, "genuine": 2.2,
    "sustainable": 2.2, "conscious": 1.8, "ethical": 2.0, "clean": 1.8,
    "pure": 1.8, "natural": 1.8, "organic": 1.8, "wholesome": 2.0,
    "delight": 2.8, "cherish": 2.5, "treasure": 2.5, "special": 2.2,
    "unique": 2.0, "exclusive": 2.2, "limited": 1.5, "rare": 2.0,
    "welcome": 1.8, "warm": 2.0, "heartfelt": 2.5, "gratitude": 2.5,
    "thankful": 2.5, "grateful": 2.5, "blessed": 2.5, "fortunate": 2.2,
    "recommend": 2.2, "recommended": 2.2, "worth": 1.8, "value": 1.8,
    "quality": 2.0, "durable": 1.8, "satisfy": 2.2, "satisfied": 2.2,
    "satisfying": 2.2, "pleased": 2.0, "pleasant": 2.0, "helpful": 2.0,
    "supportive": 2.0, "caring": 2.2, "gentle": 1.8, "nourishing": 2.0,
    "refreshing": 2.2, "energizing": 2.2, "good": 1.8, "nice": 1.8,
    "fine": 1.2, "decent": 1.0, "solid": 1.5, "cool": 1.5, "fun": 2.0,
    "enjoy": 2.0, "enjoying": 2.0, "like": 1.5, "liked": 1.5,
    "appreciate": 2.0, "appreciated": 2.0, "glad": 2.0, "easy": 1.5,
    "better": 1.5, "improved": 1.5, "upgrade": 1.5, "upgraded": 1.5,
    "useful": 1.8, "practical": 1.5, "efficient": 1.8, "affordable": 1.5,
    "accessible": 1.5, "convenient": 1.5, "quick": 1.2, "fast": 1.2,
    "effortless": 2.0, "support": 1.5, "community": 1.5, "together": 1.5,
    "honor": 2.0, "launch": 1.0, "new": 1.0, "collection": 0.2,
    "design": 0.3, "style": 0.3, "available": 0.2, "shop": 0.2,
    "average": -1.0, "mediocre": -1.5, "ordinary": -0.5, "slow": -1.0,
    "delayed": -1.2, "issue": -1.5, "concern": -1.2, "problem": -1.8,
    "difficult": -1.2, "hard": -0.8, "complicated": -1.0, "confusing": -1.2,
    "missed": -1.2, "disappoint": -2.0, "disappointed": -2.2,
    "disappointing": -2.2, "disappointment": -2.2, "frustrate": -2.0,
    "frustrated": -2.0, "frustrating": -2.0, "frustration": -2.0,
    "overpriced": -2.0, "expensive": -1.0, "costly": -1.0, "waste": -2.0,
    "terrible": -3.2, "horrible": -3.2, "awful": -3.2, "dreadful": -3.2,
    "pathetic": -3.0, "worst": -3.2, "bad": -2.5, "poor": -2.2,
    "defective": -2.8, "broken": -2.5, "damaged": -2.5, "faulty": -2.8,
    "fraud": -3.5, "scam": -3.5, "fake": -2.8, "cheated": -3.0,
    "rude": -2.8, "unprofessional": -2.8, "unacceptable": -2.8,
    "useless": -2.5, "worthless": -2.8, "garbage": -3.0, "trash": -3.0,
    "hate": -3.2, "disgusting": -3.2, "ugly": -2.5, "dirty": -2.0,
    "toxic": -2.5, "harmful": -2.5, "dangerous": -2.2, "angry": -2.5,
    "furious": -2.8, "outraged": -2.8, "regret": -2.2, "sorry": -1.5,
    "never": -1.5, "refund": -1.5, "complaint": -2.0,
}

NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "without", "lack",
                  "lacking", "dont", "doesnt", "didnt", "wont", "wouldnt",
                  "cant", "cannot", "couldnt", "hardly", "barely", "scarcely"}

BOOSTERS = {"very": 0.3, "extremely": 0.5, "incredibly": 0.5, "absolutely": 0.4,
            "totally": 0.3, "really": 0.3, "so": 0.2, "quite": 0.1,
            "super": 0.3, "highly": 0.3, "utterly": 0.4, "truly": 0.3,
            "deeply": 0.3, "exceptionally": 0.5, "remarkably": 0.4}


def clean_text(text: str) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def score_text(text: str) -> dict:
    if not text:
        return {"compound": 0.0, "raw_score": 0.0, "pos_score": 0.0,
                "neg_score": 0.0, "neu_score": 1.0, "label": "neutral",
                "word_count": 0, "matched_words": 0}
    tokens = text.split()
    scores, pos_scores, neg_scores = [], [], []
    for i, token in enumerate(tokens):
        boost = sum(BOOSTERS.get(tokens[j], 0) for j in range(max(0, i-2), i))
        negate = any(tokens[j] in NEGATION_WORDS for j in range(max(0, i-3), i))
        if token in LEXICON:
            score = LEXICON[token] * (1 + boost)
            if negate:
                score = -score * 0.75
            scores.append(score)
            (pos_scores if score > 0 else neg_scores if score < 0 else []).append(abs(score))
    raw = sum(scores)
    alpha = 15
    compound = raw / (raw + alpha) if raw >= 0 else raw / (abs(raw) + alpha)
    compound = round(max(-1.0, min(1.0, compound)), 4)
    n = len(tokens)
    pos = round(sum(pos_scores) / (n + 1e-6), 4)
    neg = round(sum(neg_scores) / (n + 1e-6), 4)
    label = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
    return {"compound": compound, "raw_score": round(raw, 4),
            "pos_score": pos, "neg_score": neg,
            "neu_score": round(1 - min(1.0, pos + neg), 4),
            "label": label, "word_count": n, "matched_words": len(scores)}


def load_and_score(input_path: str) -> pd.DataFrame:
    """Load CSV, clean text, run sentiment — returns enriched DataFrame."""
    print(C["dim"] + f"\n  Loading: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    print(C["dim"] + f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    print(C["dim"] + "  Cleaning text fields...")
    df["caption_clean"] = df["caption"].apply(clean_text)
    df["firstComment_clean"] = df["firstComment"].apply(clean_text)

    print(C["dim"] + "  Scoring sentiment...")
    cap = df["caption_clean"].apply(score_text).apply(pd.Series).add_prefix("caption_")
    com = df["firstComment_clean"].apply(score_text).apply(pd.Series).add_prefix("comment_")
    df = pd.concat([df, cap, com], axis=1)

    # Parse timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    print(C["good"] + "  ✓ Dataset ready\n")
    return df


# ══════════════════════════════════════════════════════════════
# MODE A — REGRESSION ANALYSIS
# ══════════════════════════════════════════════════════════════

def mode_a(df: pd.DataFrame, args):
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from scipy import stats

    banner("MODE A — Regression Analysis",
           "Predicting engagement from sentiment + controls")

    target = args.target
    field  = args.field

    if target not in df.columns:
        print(C["neg"] + f"  ERROR: Column '{target}' not found.")
        return

    compound_col = f"{field}_compound"
    wc_col       = f"{field}_word_count"

    # Build feature matrix
    features = {}
    features[compound_col]  = df[compound_col]
    features[f"{field}_pos_score"] = df[f"{field}_pos_score"]
    features[f"{field}_neg_score"] = df[f"{field}_neg_score"]
    features[wc_col]               = df[wc_col]

    if "hashtag_count" in df.columns:
        features["hashtag_count"] = df["hashtag_count"]
    if "caption_length" in df.columns:
        features["caption_length"] = df["caption_length"]

    feat_df = pd.DataFrame(features)

    # Log-transform target (add 1 to handle zeros)
    y_raw = df[target].fillna(0)
    y = np.log1p(y_raw)

    mask = feat_df.notna().all(axis=1) & y.notna()
    X = feat_df[mask].values
    y = y[mask].values
    y_raw_masked = y_raw[mask].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reg = LinearRegression().fit(X_scaled, y)
    y_pred = reg.predict(X_scaled)
    residuals = y - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    n, p = X_scaled.shape
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    # t-stats
    mse = ss_res / (n - p - 1)
    var_b = mse * np.linalg.inv(X_scaled.T @ X_scaled).diagonal()
    se = np.sqrt(var_b)
    t_stats = reg.coef_ / se
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - p - 1))

    section(f"Model: log(1 + {target}) ~ {field} sentiment + controls")
    row_print("Observations", f"{n:,}")
    row_print("R²", f"{r2:.4f}")
    row_print("Adjusted R²", f"{adj_r2:.4f}")
    row_print("Target (raw)", f"mean={y_raw_masked.mean():.1f}, median={np.median(y_raw_masked):.1f}")

    section("Coefficients (Standardized)")
    coef_rows = []
    feat_names = list(features.keys())
    for fname, coef, t, p in zip(feat_names, reg.coef_, t_stats, p_vals):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        direction = C["pos"] + "▲" if coef > 0 else C["neg"] + "▼"
        coef_rows.append([
            fname, f"{coef:+.4f}", f"{t:+.3f}", f"{p:.4f}", sig
        ])

    print_table(coef_rows,
                headers=["Feature", "Coef (std)", "t-stat", "p-value", "Sig"],
                colalign=("left", "right", "right", "right", "center"))

    section("Visual: Coefficient Impact")
    max_coef = max(abs(c) for c in reg.coef_) or 1
    for fname, coef in zip(feat_names, reg.coef_):
        col = Fore.GREEN if coef > 0 else Fore.RED
        filled = int(abs(coef) / max_coef * 30)
        bar = "█" * filled + "░" * (30 - filled)
        sign = "+" if coef > 0 else "-"
        print(f"  {fname:<28} {col}{sign}{bar}{C['reset']}  {coef:+.4f}")

    section("Interaction Hint")
    print(C["dim"] + "  To test interaction: compound × followerCount (run --mode A --interaction)")
    print(C["dim"] + "  Significant sentiment coef suggests tone predicts engagement.")

    footer("Significance: *** p<0.001  ** p<0.01  * p<0.05 | Coefficients are standardized (comparable magnitudes)")


# ══════════════════════════════════════════════════════════════
# MODE B — DESCRIPTIVE COMPARISON
# ══════════════════════════════════════════════════════════════

def mode_b(df: pd.DataFrame, args):
    from scipy import stats

    banner("MODE B — Descriptive Comparison",
           "Engagement metrics across sentiment groups")

    field = args.field
    label_col = f"{field}_label"
    top_n = args.top

    section("Overall Sentiment Distribution")
    dist = df[label_col].value_counts()
    total = len(df)
    for label, count in dist.items():
        pct = count / total * 100
        filled = int(pct / 2)
        bar = "█" * filled + "░" * (50 - filled)
        col = sentiment_color(label)
        print(f"  {label:<12} {col}{bar}{C['reset']}  {count:>5} posts ({pct:.1f}%)")

    section(f"Engagement by {field.title()} Sentiment")
    metrics = ["likesCount", "commentsCount"]
    metrics = [m for m in metrics if m in df.columns]
    rows = []
    for label in ["positive", "neutral", "negative"]:
        grp = df[df[label_col] == label]
        row = [label, len(grp)]
        for m in metrics:
            vals = grp[m].dropna()
            row += [f"{vals.mean():.1f}", f"{np.median(vals):.1f}"]
        rows.append(row)

    headers = ["Sentiment", "Count"]
    for m in metrics:
        headers += [f"{m} Mean", f"{m} Median"]
    print_table(rows, headers=headers, colalign=["left"] + ["right"] * (len(headers) - 1))

    section("ANOVA — Does Sentiment Group Predict Likes?")
    if "likesCount" in df.columns:
        groups = [df[df[label_col] == lbl]["likesCount"].dropna().values
                  for lbl in ["positive", "neutral", "negative"]]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) >= 2:
            f_stat, p_val = stats.f_oneway(*groups)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
            col = C["pos"] if p_val < 0.05 else C["warn"]
            row_print("F-statistic", f"{f_stat:.4f}")
            row_print("p-value", col + f"{p_val:.6f}  {sig}")
            if p_val < 0.05:
                print(C["pos"] + "  ✓ Sentiment group significantly predicts likes.")
            else:
                print(C["warn"] + "  ✗ No significant difference in likes across sentiment groups.")

    section(f"Top {top_n} Most Engaging Posts (by likes)")
    if "likesCount" in df.columns:
        top = df.nlargest(top_n, "likesCount")[[
            "ownerUsername", label_col, f"{field}_compound", "likesCount", "commentsCount"
        ]].copy()
        top[f"{field}_compound"] = top[f"{field}_compound"].apply(lambda x: f"{x:+.3f}")
        top["likesCount"] = top["likesCount"].apply(lambda x: f"{int(x):,}")
        top["commentsCount"] = top["commentsCount"].apply(lambda x: f"{int(x):,}")
        print_table(top, headers=["Brand", "Sentiment", "Compound", "Likes", "Comments"])

    section(f"Average Compound Score by Brand (top {top_n} by post count)")
    brand_agg = df.groupby("ownerUsername").agg(
        posts=("caption_compound", "count"),
        avg_sentiment=(f"{field}_compound", "mean"),
        avg_likes=("likesCount", "mean"),
    ).sort_values("posts", ascending=False).head(top_n)
    brand_agg["avg_sentiment"] = brand_agg["avg_sentiment"].round(4)
    brand_agg["avg_likes"] = brand_agg["avg_likes"].round(1)
    print_table(brand_agg.reset_index(),
                headers=["Brand", "Posts", "Avg Sentiment", "Avg Likes"],
                colalign=["left", "right", "right", "right"])

    footer(f"Field analyzed: '{field}'. Change with --field caption|comment")


# ══════════════════════════════════════════════════════════════
# MODE C — ALIGNMENT ANALYSIS
# ══════════════════════════════════════════════════════════════

def mode_c(df: pd.DataFrame, args):
    from scipy import stats

    banner("MODE C — Alignment Analysis",
           "Caption vs. Comment sentiment gap")

    section("Sentiment Label Distribution Comparison")
    for field in ["caption", "comment"]:
        dist = df[f"{field}_label"].value_counts(normalize=True) * 100
        print(C["sub"] + f"\n  {field.upper()}:")
        for label, pct in dist.items():
            col = sentiment_color(label)
            filled = int(pct / 2)
            bar = "█" * filled + "░" * (50 - filled)
            print(f"    {label:<12} {col}{bar}{C['reset']}  {pct:.1f}%")

    section("Caption ↔ Comment Compound Correlation")
    mask = df["caption_compound"].notna() & df["comment_compound"].notna()
    cap_scores = df.loc[mask, "caption_compound"]
    com_scores = df.loc[mask, "comment_compound"]
    r, p = stats.pearsonr(cap_scores, com_scores)
    col = C["pos"] if abs(r) > 0.3 else C["warn"]
    row_print("Pearson r", col + f"{r:.4f}")
    row_print("p-value", f"{p:.6f}")
    row_print("Interpretation",
              "Strong alignment" if abs(r) > 0.5 else
              "Moderate alignment" if abs(r) > 0.3 else
              "Weak / no alignment")

    section("Alignment Classification")
    df["_aligned"] = df["caption_label"] == df["comment_label"]
    df["_gap"] = df["caption_compound"] - df["comment_compound"]
    df["_gap_type"] = pd.cut(df["_gap"],
                             bins=[-2.1, -0.1, 0.1, 2.1],
                             labels=["Caption < Comment", "Aligned", "Caption > Comment"])

    dist = df["_gap_type"].value_counts()
    total = dist.sum()
    for gtype, cnt in dist.items():
        pct = cnt / total * 100
        col = C["pos"] if gtype == "Aligned" else C["warn"]
        print(f"  {str(gtype):<25} {col}{cnt:>5} posts  ({pct:.1f}%){C['reset']}")

    section("Overpromise Detection")
    print(C["dim"] + "  Posts where caption=positive BUT comment=negative")
    overpromise = df[(df["caption_label"] == "positive") &
                     (df["comment_label"] == "negative")].copy()
    row_print("Count", f"{len(overpromise)} posts ({len(overpromise)/len(df)*100:.1f}%)")

    if len(overpromise) > 0:
        avg_likes = overpromise["likesCount"].mean() if "likesCount" in overpromise else 0
        all_avg   = df["likesCount"].mean() if "likesCount" in df else 0
        row_print("Avg Likes (overpromise posts)", f"{avg_likes:.1f}")
        row_print("Avg Likes (all posts)", f"{all_avg:.1f}")
        col = C["neg"] if avg_likes < all_avg else C["pos"]
        row_print("Engagement vs. overall", col + ("↓ Lower" if avg_likes < all_avg else "↑ Higher"))

        print(C["dim"] + "\n  Sample overpromise posts:")
        sample = overpromise[["ownerUsername", "caption_compound", "comment_compound",
                               "likesCount"]].head(5)
        sample["caption_compound"] = sample["caption_compound"].apply(lambda x: f"{x:+.3f}")
        sample["comment_compound"] = sample["comment_compound"].apply(lambda x: f"{x:+.3f}")
        print_table(sample, headers=["Brand", "Caption ↑", "Comment ↓", "Likes"])

    section("Underrated Posts")
    print(C["dim"] + "  Posts where caption=neutral/negative BUT comment=positive")
    underrated = df[(df["caption_label"].isin(["neutral", "negative"])) &
                    (df["comment_label"] == "positive")].copy()
    row_print("Count", f"{len(underrated)} posts ({len(underrated)/len(df)*100:.1f}%)")

    section("Gap Distribution (Caption − Comment Compound)")
    gap_vals = df["_gap"].dropna()
    row_print("Mean gap", f"{gap_vals.mean():+.4f}  (positive = captions more positive than comments)")
    row_print("Std deviation", f"{gap_vals.std():.4f}")
    row_print("Max gap", f"{gap_vals.max():+.4f}")
    row_print("Min gap", f"{gap_vals.min():+.4f}")

    # Histogram in terminal
    section("Gap Histogram (terminal)")
    bins = [-1.0, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 1.0]
    labels_bin = ["<-0.6", "-0.6:-0.3", "-0.3:-0.1", "-0.1:0.1",
                  "0.1:0.3", "0.3:0.6", ">0.6"]
    counts, _ = np.histogram(gap_vals.clip(-1, 1), bins=bins)
    max_c = max(counts) or 1
    for lbl, cnt in zip(labels_bin, counts):
        filled = int(cnt / max_c * 35)
        col = Fore.RED if lbl.startswith("<") or lbl.startswith("-") else Fore.GREEN if float(lbl.split(":")[0].replace("<","").replace(">","") or 0) > 0 else Fore.CYAN
        bar = "█" * filled + "░" * (35 - filled)
        print(f"  {lbl:>12}  {col}{bar}{C['reset']}  {cnt:>4}")

    footer("Gap = caption_compound − comment_compound. Positive = brand is more positive than audience.")


# ══════════════════════════════════════════════════════════════
# MODE D — TIME-SERIES ANALYSIS
# ══════════════════════════════════════════════════════════════

def mode_d(df: pd.DataFrame, args):
    banner("MODE D — Time-Series Analysis",
           "Sentiment tone trends over time")

    if "timestamp" not in df.columns or df["timestamp"].isna().all():
        print(C["neg"] + "  ERROR: No valid timestamp column found.")
        return

    freq  = args.freq
    field = args.field
    compound_col = f"{field}_compound"

    freq_label = {"D": "Day", "W": "Week", "ME": "Month", "QE": "Quarter"}.get(freq, freq)

    df_time = df[["timestamp", compound_col, "likesCount", "commentsCount",
                  f"{field}_label"]].copy()
    df_time = df_time.dropna(subset=["timestamp"])
    df_time = df_time.set_index("timestamp").sort_index()

    section(f"Aggregated by {freq_label}")
    agg = df_time.resample(freq).agg(
        posts=(compound_col, "count"),
        avg_compound=(compound_col, "mean"),
        avg_likes=("likesCount", "mean"),
        avg_comments=("commentsCount", "mean"),
        pct_positive=(f"{field}_label", lambda x: (x == "positive").mean() * 100),
        pct_negative=(f"{field}_label", lambda x: (x == "negative").mean() * 100),
    ).dropna(subset=["avg_compound"])

    # Terminal sparkline
    section(f"Sentiment Trend — {field.title()} Compound (by {freq_label})")
    vals = agg["avg_compound"].values
    min_v, max_v = vals.min(), vals.max()
    range_v = max_v - min_v or 1
    bar_chars = "▁▂▃▄▅▆▇█"

    print(f"  {'Period':<14} {'Avg Sent':>9}  {'Chart':^35}  {'Posts':>6}  {'Avg Likes':>9}")
    print(C["sep"] + "  " + "─" * 78)
    for period, row in agg.iterrows():
        v = row["avg_compound"]
        norm = (v - min_v) / range_v
        char_idx = min(int(norm * 8), 7)
        spark = bar_chars[char_idx]
        filled = int(norm * 33)
        col = Fore.GREEN if v > 0.05 else Fore.RED if v < -0.05 else Fore.WHITE
        bar = col + "█" * filled + Style.DIM + "░" * (33 - filled) + C["reset"]
        period_str = str(period.date()) if hasattr(period, 'date') else str(period)[:10]
        print(f"  {period_str:<14} {v:>+8.4f}  {bar}  {int(row['posts']):>6}  {row['avg_likes']:>9.1f}")

    section("Peak Positive & Negative Periods")
    top_pos = agg["avg_compound"].idxmax()
    top_neg = agg["avg_compound"].idxmin()
    top_pos_str = str(top_pos.date()) if hasattr(top_pos, 'date') else str(top_pos)[:10]
    top_neg_str = str(top_neg.date()) if hasattr(top_neg, 'date') else str(top_neg)[:10]
    row_print("Most Positive Period", C["pos"] + f"{top_pos_str}  (compound: {agg.loc[top_pos, 'avg_compound']:+.4f})")
    row_print("Most Negative Period", C["neg"] + f"{top_neg_str}  (compound: {agg.loc[top_neg, 'avg_compound']:+.4f})")

    section("Sentiment vs. Engagement Correlation (over time)")
    from scipy import stats
    if len(agg) >= 3:
        r_likes, p_likes = stats.pearsonr(agg["avg_compound"], agg["avg_likes"].fillna(0))
        r_comments, p_comments = stats.pearsonr(agg["avg_compound"], agg["avg_comments"].fillna(0))
        col_l = C["pos"] if abs(r_likes) > 0.3 else C["warn"]
        col_c = C["pos"] if abs(r_comments) > 0.3 else C["warn"]
        row_print("Sentiment ↔ Avg Likes (r)", col_l + f"{r_likes:+.4f}  p={p_likes:.4f}")
        row_print("Sentiment ↔ Avg Comments (r)", col_c + f"{r_comments:+.4f}  p={p_comments:.4f}")

    section("Monthly Heatmap — % Positive Posts")
    if freq in ["W", "ME", "D"]:
        monthly = df_time.resample("ME").agg(
            pct_pos=(f"{field}_label", lambda x: (x == "positive").mean() * 100)
        ).dropna()
        for period, row in monthly.iterrows():
            pct = row["pct_pos"]
            filled = int(pct / 2)
            col = Fore.GREEN if pct > 60 else Fore.YELLOW if pct > 40 else Fore.RED
            bar = "█" * filled + "░" * (50 - filled)
            period_str = str(period.date())[:7] if hasattr(period, 'date') else str(period)[:7]
            print(f"  {period_str}  {col}{bar}{C['reset']}  {pct:.1f}%")

    footer(f"Frequency: {freq_label}. Change with --freq D|W|ME|QE")


# ══════════════════════════════════════════════════════════════
# MODE E — BRAND-LEVEL AGGREGATION
# ══════════════════════════════════════════════════════════════

def mode_e(df: pd.DataFrame, args):
    from scipy import stats

    banner("MODE E — Brand-Level Aggregation",
           "Per-brand sentiment strategy vs. performance")

    field    = args.field
    min_posts = args.min_posts
    compound_col = f"{field}_compound"
    label_col    = f"{field}_label"

    brand_agg = df.groupby("ownerUsername").agg(
        posts=(compound_col, "count"),
        avg_sentiment=(compound_col, "mean"),
        std_sentiment=(compound_col, "std"),
        pct_positive=(label_col, lambda x: (x == "positive").mean() * 100),
        pct_negative=(label_col, lambda x: (x == "negative").mean() * 100),
        avg_likes=("likesCount", "mean"),
        total_likes=("likesCount", "sum"),
        avg_comments=("commentsCount", "mean"),
    ).reset_index()

    brand_agg = brand_agg[brand_agg["posts"] >= min_posts].copy()
    brand_agg = brand_agg.sort_values("avg_sentiment", ascending=False)

    section(f"Brand Sentiment Leaderboard (min {min_posts} posts)")
    print(f"  {'Brand':<22} {'Posts':>5}  {'Avg Sent':>9}  {'Chart':^30}  {'%Pos':>5}  {'%Neg':>5}  {'Avg Likes':>9}")
    print(C["sep"] + "  " + "─" * 84)

    max_sent = brand_agg["avg_sentiment"].max() or 1
    min_sent = brand_agg["avg_sentiment"].min()

    for _, row in brand_agg.iterrows():
        v = row["avg_sentiment"]
        norm = (v - min_sent) / (max_sent - min_sent + 1e-6)
        col = Fore.GREEN if v > 0.1 else Fore.RED if v < -0.05 else Fore.WHITE
        filled = int(norm * 28)
        bar = col + "█" * filled + Style.DIM + "░" * (28 - filled) + C["reset"]
        print(f"  {row['ownerUsername']:<22} {int(row['posts']):>5}  "
              f"{v:>+8.4f}  {bar}  "
              f"{row['pct_positive']:>4.0f}%  "
              f"{row['pct_negative']:>4.0f}%  "
              f"{row['avg_likes']:>9.1f}")

    section("Sentiment Consistency (Std Dev)")
    print(C["dim"] + "  Low std = consistent tone  |  High std = erratic tone")
    consistency = brand_agg.sort_values("std_sentiment").fillna(0)
    for _, row in consistency.iterrows():
        std = row["std_sentiment"]
        filled = int(min(std * 40, 40))
        col = Fore.GREEN if std < 0.15 else Fore.YELLOW if std < 0.25 else Fore.RED
        bar = col + "█" * filled + Style.DIM + "░" * (40 - filled) + C["reset"]
        print(f"  {row['ownerUsername']:<22} {bar}  σ={std:.3f}")

    section("Sentiment vs. Avg Likes Correlation (brand level)")
    if len(brand_agg) >= 3:
        r, p = stats.pearsonr(brand_agg["avg_sentiment"], brand_agg["avg_likes"].fillna(0))
        col = C["pos"] if abs(r) > 0.3 else C["warn"]
        row_print("Pearson r (sentiment ↔ avg likes)", col + f"{r:+.4f}")
        row_print("p-value", f"{p:.4f}")
        interpretation = (
            "Strong positive relationship" if r > 0.5 else
            "Moderate positive relationship" if r > 0.3 else
            "Weak relationship" if abs(r) < 0.3 else
            "Negative relationship"
        )
        row_print("Interpretation", interpretation)

    section("Brand Quadrant Analysis")
    print(C["dim"] + "  High Sentiment + High Likes  = IDEAL STRATEGY")
    print(C["dim"] + "  High Sentiment + Low Likes   = POSITIVE BUT UNDERPERFORMING")
    print(C["dim"] + "  Low Sentiment  + High Likes  = EDGY / VIRAL NEGATIVE CONTENT")
    print(C["dim"] + "  Low Sentiment  + Low Likes   = REVIEW CONTENT STRATEGY\n")

    med_sent  = brand_agg["avg_sentiment"].median()
    med_likes = brand_agg["avg_likes"].median()

    for _, row in brand_agg.iterrows():
        high_sent  = row["avg_sentiment"] >= med_sent
        high_likes = row["avg_likes"] >= med_likes
        if high_sent and high_likes:
            quad = C["pos"] + "★ IDEAL"
        elif high_sent and not high_likes:
            quad = C["warn"] + "↗ POSITIVE / LOW REACH"
        elif not high_sent and high_likes:
            quad = C["accent"] + "⚡ EDGY / VIRAL"
        else:
            quad = C["neg"] + "↙ REVIEW STRATEGY"
        print(f"  {row['ownerUsername']:<22} {quad}{C['reset']}")

    footer(f"Min posts filter: {min_posts}. Change with --min-posts N")


# ══════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════

MODES = {
    "A": ("Regression Analysis",      mode_a),
    "B": ("Descriptive Comparison",   mode_b),
    "C": ("Alignment Analysis",       mode_c),
    "D": ("Time-Series Analysis",     mode_d),
    "E": ("Brand-Level Aggregation",  mode_e),
}

def print_help_menu():
    print(C["header"] + """
╔══════════════════════════════════════════════════════════════╗
║     Instagram D2C Sentiment Analysis Toolkit v2.0           ║
║     SMDA Term Paper — Baveja & Bharati                      ║
╚══════════════════════════════════════════════════════════════╝""")
    print(C["sub"] + "\n  AVAILABLE MODES:\n")
    for key, (name, _) in MODES.items():
        print(C["bold"] + f"    --mode {key}  " + C["reset"] + f"→  {name}")
    print(C["sub"] + "\n  COMMON OPTIONS:\n")
    opts = [
        ("--input FILE",       "Input CSV path (default: instagram_combined_final.csv)"),
        ("--mode MODE",        "A | B | C | D | E | all"),
        ("--field FIELD",      "caption | comment  (default: caption)"),
        ("--target COL",       "Mode A: engagement column (default: likesCount)"),
        ("--freq FREQ",        "Mode D: D | W | ME | QE  (default: ME)"),
        ("--top N",            "Mode B: top N posts/brands (default: 10)"),
        ("--min-posts N",      "Mode E: min posts per brand (default: 5)"),
        ("--exclude BRAND ...", "Exclude one or more brands by ownerUsername"),
        ("--no-save",          "Skip saving enriched CSV output"),
    ]
    for flag, desc in opts:
        print(C["dim"] + f"    {flag:<20} {desc}")
    print()

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input",     default="instagram_combined_final.csv")
    parser.add_argument("--output",    default="sentiment_enriched.csv")
    parser.add_argument("--mode",      default=None)
    parser.add_argument("--field",     default="caption", choices=["caption", "comment"])
    parser.add_argument("--target",    default="likesCount")
    parser.add_argument("--freq",      default="ME", choices=["D", "W", "ME", "QE"])
    parser.add_argument("--top",       default=10, type=int)
    parser.add_argument("--min-posts", default=5,  type=int, dest="min_posts")
    parser.add_argument("--exclude",   default=[], nargs="+", dest="exclude",
                        metavar="BRAND",
                        help="Exclude one or more brands by ownerUsername")
    parser.add_argument("--no-save",   action="store_true")
    parser.add_argument("--help", "-h", action="store_true")
    args = parser.parse_args()

    if args.help or args.mode is None:
        print_help_menu()
        sys.exit(0)

    if not os.path.exists(args.input):
        print(C["neg"] + f"\n  ERROR: Input file not found: {args.input}\n")
        sys.exit(1)

    # Header
    print(C["header"] + """
╔══════════════════════════════════════════════════════════════╗
║     Instagram D2C Sentiment Analysis Toolkit                 ║
║     SMDA Term Paper — Baveja & Bharati                      ║
╚══════════════════════════════════════════════════════════════╝""")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row_print("Run time",   ts)
    row_print("Input file", args.input)
    row_print("Mode",       args.mode.upper())
    row_print("Field",      args.field)

    df = load_and_score(args.input)

    # ── Brand exclusion ──────────────────────────────────────────
    if args.exclude:
        all_brands     = set(df["ownerUsername"].dropna().unique())
        matched        = [b for b in args.exclude if b in all_brands]
        unmatched      = [b for b in args.exclude if b not in all_brands]

        if matched:
            before = len(df)
            df = df[~df["ownerUsername"].isin(matched)].copy()
            after  = len(df)
            print(C["warn"] + f"\n  ⚠  Brand exclusion applied:")
            for b in matched:
                dropped = before - after if b == matched[-1] else None
                print(C["dim"] + f"     ✕  {b}")
            print(C["dim"] + f"     Rows removed : {before - after:,}  →  {after:,} rows remaining")

        if unmatched:
            print(C["neg"] + f"\n  ✗  Brand(s) not found in dataset (check spelling):")
            for b in unmatched:
                # suggest close matches
                close = [x for x in all_brands if b.lower() in x.lower()]
                hint  = f"  Did you mean: {', '.join(close[:3])}?" if close else ""
                print(C["dim"] + f"     ✕  {b}{('  →  ' + hint) if hint else ''}")
        print()

    if not args.no_save:
        df.to_csv(args.output, index=False)
        print(C["good"] + f"  ✓ Enriched CSV saved → {args.output}\n")

    mode_str = args.mode.upper()

    if mode_str == "ALL":
        for key, (name, fn) in MODES.items():
            fn(df, args)
    elif mode_str in MODES:
        MODES[mode_str][1](df, args)
    else:
        print(C["neg"] + f"  Unknown mode: {args.mode}")
        print(C["dim"] + "  Valid modes: A B C D E all")
        sys.exit(1)


if __name__ == "__main__":
    main()
