#!/usr/bin/env python3
"""
kmeans_cluster.py — Generic K-Means clustering tool for CSV inputs.

Usage:
    python kmeans_cluster.py data.csv
    python kmeans_cluster.py data.csv --k 4
    python kmeans_cluster.py data.csv --k 4 --cols col1 col2 col3
    python kmeans_cluster.py data.csv --k 4 --scale --max-iter 500 --runs 20
    python kmeans_cluster.py data.csv --auto-k --k-max 8

String columns are automatically label-encoded for clustering.
In output, string columns show mode (top category) and unique count
instead of mean/std, plus a per-cluster category breakdown table.

Outputs (all in terminal):
    - Cluster averages table (numeric cols) / mode table (string cols)
    - Std dev table (numeric) / unique-count + top categories (string)
    - Cluster category profiles (string columns only, if any)
    - Cluster sizes
    - Performance metrics: Inertia, Silhouette Score, Davies-Bouldin Index
"""

import argparse
import sys
import math
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, davies_bouldin_score


# ─── ANSI colour helpers ───────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

BLACK  = "\033[30m"
WHITE  = "\033[97m"

RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
MAGENTA= "\033[95m"

BG_DARK   = "\033[48;5;235m"
BG_HEADER = "\033[48;5;17m"

CLUSTER_COLORS = [
    "\033[92m",  # green
    "\033[93m",  # yellow
    "\033[96m",  # cyan
    "\033[95m",  # magenta
    "\033[91m",  # red
    "\033[94m",  # blue
    "\033[38;5;208m",  # orange
    "\033[38;5;141m",  # purple
]


def ccluster(k: int) -> str:
    """Return ANSI color for cluster k."""
    return CLUSTER_COLORS[k % len(CLUSTER_COLORS)]


def hr(char: str = "─", width: int = 80) -> str:
    return DIM + char * width + RESET


def section_header(title: str, width: int = 80) -> str:
    pad = width - len(title) - 4
    left = pad // 2
    right = pad - left
    return (
        f"\n{BG_HEADER}{WHITE}{BOLD}"
        f"  {'─' * left} {title} {'─' * right}  "
        f"{RESET}"
    )


def fmt_float(val: float, decimals: int = 4) -> str:
    if math.isnan(val):
        return DIM + "  n/a   " + RESET
    return f"{val:>{8}.{decimals}f}"


# ─── Table renderer ───────────────────────────────────────────────────────────

def render_table(
    title: str,
    columns: list[str],
    rows: list[list],
    col_widths: Optional[list[int]] = None,
    row_colors: Optional[list[str]] = None,
    note: str = "",
) -> None:
    """Pretty-print a table to stdout."""
    if col_widths is None:
        col_widths = []
        for i, col in enumerate(columns):
            col_w = max(len(str(col)), 10)
            for row in rows:
                col_w = max(col_w, len(str(row[i])) + 2)
            col_widths.append(col_w)

    total_w = sum(col_widths) + len(col_widths) * 3 + 1

    print(f"\n  {BOLD}{CYAN}{title}{RESET}")
    print(f"  {DIM}{'─' * total_w}{RESET}")

    # Header
    header_parts = []
    for col, w in zip(columns, col_widths):
        header_parts.append(f"{BOLD}{WHITE} {str(col).center(w)} {RESET}")
    print("  │" + "│".join(header_parts) + "│")
    print(f"  {DIM}{'─' * total_w}{RESET}")

    # Rows
    for i, row in enumerate(rows):
        color = row_colors[i] if row_colors else ""
        parts = []
        for val, w in zip(row, col_widths):
            cell = str(val)
            parts.append(f"{color} {cell.center(w)} {RESET}")
        print("  │" + "│".join(parts) + "│")

    print(f"  {DIM}{'─' * total_w}{RESET}")
    if note:
        print(f"  {DIM}{note}{RESET}")


# ─── Core clustering ──────────────────────────────────────────────────────────

def load_and_prepare(
    filepath: str,
    cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str], dict[str, LabelEncoder]]:
    """Load CSV, encode string columns, return everything needed for clustering.

    Returns:
        df_raw          — original full dataframe
        df_encoded      — dataframe ready for clustering (numeric + label-encoded strings)
        df_original     — dataframe with original values (for display / mode calculation)
        numeric_cols    — column names that were originally numeric
        string_cols     — column names that were originally strings (now label-encoded)
        encoders        — {col: LabelEncoder} for reverse-mapping if needed
    """
    try:
        df_raw = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"{RED}✗ File not found: {filepath}{RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"{RED}✗ Could not read CSV: {e}{RESET}")
        sys.exit(1)

    # ── Select columns ──
    if cols:
        missing = [c for c in cols if c not in df_raw.columns]
        if missing:
            print(f"{RED}✗ Columns not found: {missing}{RESET}")
            print(f"  Available: {list(df_raw.columns)}")
            sys.exit(1)
        df = df_raw[cols].copy()
    else:
        # Auto-select: numeric + object/category columns (exclude high-cardinality IDs)
        num_df  = df_raw.select_dtypes(include=[np.number])
        str_df  = df_raw.select_dtypes(include=["object", "category", "string"])

        # Heuristic: skip string cols where unique values > 50% of rows (likely IDs / free text)
        n_rows = len(df_raw)
        usable_str_cols = [
            c for c in str_df.columns
            if str_df[c].nunique() <= max(2, n_rows * 0.5)
        ]
        skipped = [c for c in str_df.columns if c not in usable_str_cols]
        if skipped:
            print(f"{YELLOW}⚠  Skipping high-cardinality string columns (likely IDs/free text): {skipped}{RESET}")

        df = pd.concat([num_df, str_df[usable_str_cols]], axis=1)

    if df.empty or df.shape[1] == 0:
        print(f"{RED}✗ No usable columns found in the CSV.{RESET}")
        sys.exit(1)

    # ── Identify column types ──
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    string_cols  = list(df.select_dtypes(include=["object", "category", "string"]).columns)

    if not numeric_cols and not string_cols:
        print(f"{RED}✗ No usable numeric or string columns found.{RESET}")
        sys.exit(1)

    # ── Drop rows with all-NaN ──
    n_before = len(df)
    df.dropna(how="all", inplace=True)
    n_after = len(df)
    if n_before != n_after:
        print(f"{YELLOW}⚠  Dropped {n_before - n_after} fully-null rows.{RESET}")

    # Keep a copy with original string values for display
    df_original = df.copy()

    # ── Encode string columns ──
    encoders: dict[str, LabelEncoder] = {}
    df_encoded = df.copy()

    for col in string_cols:
        # Fill NaN with most frequent value before encoding
        mode_val = df_encoded[col].mode()
        fill_val = mode_val.iloc[0] if not mode_val.empty else "unknown"
        df_encoded[col] = df_encoded[col].fillna(fill_val)
        df_original[col] = df_original[col].fillna(fill_val)

        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le

    # ── Fill remaining numeric NaNs with column mean ──
    for col in numeric_cols:
        df_encoded[col] = df_encoded[col].fillna(df_encoded[col].mean())
        df_original[col] = df_original[col].fillna(df_original[col].mean())

    if string_cols:
        print(
            f"{CYAN}ℹ  String columns label-encoded for clustering: "
            f"{', '.join(string_cols)}{RESET}"
        )

    return df_raw, df_encoded, df_original, numeric_cols, string_cols, encoders


def run_kmeans(
    X: np.ndarray,
    k: int,
    max_iter: int = 300,
    n_init: int = 10,
    random_state: int = 42,
) -> tuple[KMeans, np.ndarray]:
    km = KMeans(
        n_clusters=k,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state,
    )
    labels = km.fit_predict(X)
    return km, labels


def find_optimal_k(
    X: np.ndarray,
    k_max: int = 8,
    max_iter: int = 300,
    n_init: int = 10,
) -> int:
    """Use silhouette score to find best k in [2, k_max]."""
    print(f"\n  {BOLD}Auto-selecting k (k=2 … {k_max}) via silhouette score{RESET}")
    print(f"  {DIM}{'─' * 52}{RESET}")

    best_k, best_score = 2, -1.0
    scores = []

    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, max_iter=max_iter, n_init=n_init, random_state=42)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        scores.append((k, sil))
        marker = ""
        if sil > best_score:
            best_score = sil
            best_k = k

    for k, sil in scores:
        bar_len = int(max(0, sil) * 30)
        bar = "█" * bar_len
        star = f"  {GREEN}◄ best{RESET}" if k == best_k else ""
        color = GREEN if k == best_k else DIM
        print(f"  k={k}  {color}{bar:<30}{RESET}  sil={sil:+.4f}{star}")

    print(f"\n  {GREEN}{BOLD}Optimal k = {best_k}  (silhouette = {best_score:.4f}){RESET}")
    return best_k


def compute_cluster_stats(
    df_encoded: pd.DataFrame,
    df_original: pd.DataFrame,
    labels: np.ndarray,
    numeric_cols: list[str],
    string_cols: list[str],
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return per-cluster stats split by column type.

    Returns:
        num_means   — mean of each numeric col per cluster
        num_stds    — std dev of each numeric col per cluster
        str_modes   — mode (top category) of each string col per cluster
        str_uniques — unique value count of each string col per cluster
        sizes       — row count per cluster
    """
    df_enc = df_encoded.copy()
    df_enc["_cluster"] = labels
    df_orig = df_original.copy()
    df_orig["_cluster"] = labels

    sizes = df_enc.groupby("_cluster").size().rename("Count")

    # Numeric stats
    if numeric_cols:
        num_means = df_enc.groupby("_cluster")[numeric_cols].mean()
        num_stds  = df_enc.groupby("_cluster")[numeric_cols].std(ddof=1)
    else:
        num_means = pd.DataFrame()
        num_stds  = pd.DataFrame()

    # String stats — compute on original (unencoded) values
    if string_cols:
        str_modes   = df_orig.groupby("_cluster")[string_cols].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else "n/a"
        )
        str_uniques = df_orig.groupby("_cluster")[string_cols].nunique()
    else:
        str_modes   = pd.DataFrame()
        str_uniques = pd.DataFrame()

    return num_means, num_stds, str_modes, str_uniques, sizes


# ─── Terminal output ──────────────────────────────────────────────────────────

def print_banner(
    filepath: str,
    k: int,
    n_rows: int,
    n_numeric: int,
    n_string: int,
    scaled: bool,
) -> None:
    print("\n" + hr("═"))
    print(f"  {BOLD}{CYAN}K-MEANS CLUSTERING{RESET}  {DIM}─{RESET}  {filepath}")
    print(hr("═"))
    print(
        f"  Rows: {BOLD}{n_rows}{RESET}   "
        f"Numeric features: {BOLD}{n_numeric}{RESET}   "
        f"String features: {BOLD}{n_string}{RESET}   "
        f"k: {BOLD}{k}{RESET}   "
        f"Scaling: {BOLD}{'StandardScaler' if scaled else 'none'}{RESET}"
    )
    print(hr())


def print_cluster_sizes(sizes: pd.Series, k: int, n_total: int) -> None:
    print(section_header("CLUSTER SIZES"))
    print()
    for cluster_id, count in sizes.items():
        pct = count / n_total * 100
        bar_len = int(pct / 2)  # max 50 chars
        bar = "█" * bar_len
        color = ccluster(cluster_id)
        print(
            f"  {color}{BOLD}Cluster {cluster_id}{RESET}  "
            f"{color}{bar:<50}{RESET}  "
            f"{count:>5} rows  ({pct:5.1f}%)"
        )
    print()


def print_means_table(
    num_means: pd.DataFrame,
    str_modes: pd.DataFrame,
    numeric_cols: list[str],
    string_cols: list[str],
    k: int,
) -> None:
    """Print cluster means (numeric) and cluster modes (string) in one unified table."""
    all_cols = numeric_cols + string_cols
    if not all_cols:
        return

    print(section_header("CLUSTER MEANS / MODE  (numeric: mean  |  string: top category)"))

    col_w = max(max(len(c) for c in all_cols), 14) + 2
    num_w = 14

    # Header
    header = [f"{'Variable':{col_w}}  {'Type':<8}"]
    for c in range(k):
        header.append(f"{ccluster(c)}{BOLD}{'Cluster '+str(c):^{num_w}}{RESET}")
    print("\n  " + "  ".join(header))
    print(f"  {DIM}{'─' * (col_w + 10 + (num_w + 2) * k)}{RESET}")

    # Numeric rows
    for col in numeric_cols:
        if num_means.empty or col not in num_means.columns:
            continue
        row_parts = [f"{BOLD}{col:{col_w}}{RESET}  {DIM}{'numeric':<8}{RESET}"]
        vals = [num_means.loc[c, col] for c in range(k) if c in num_means.index]
        max_v = max(abs(v) for v in vals) if vals else 1
        for c in range(k):
            if c not in num_means.index:
                row_parts.append(f"{'n/a':^{num_w}}")
                continue
            v = num_means.loc[c, col]
            intensity = abs(v) / max_v if max_v != 0 else 0
            color = ccluster(c) if intensity > 0.85 else ""
            row_parts.append(f"{color}{v:>{num_w}.4f}{RESET}")
        print("  " + "  ".join(row_parts))

    # String rows — show mode value, truncated to fit
    for col in string_cols:
        if str_modes.empty or col not in str_modes.columns:
            continue
        row_parts = [f"{BOLD}{col:{col_w}}{RESET}  {MAGENTA}{'string':<8}{RESET}"]
        for c in range(k):
            if c not in str_modes.index:
                row_parts.append(f"{'n/a':^{num_w}}")
                continue
            val = str(str_modes.loc[c, col])
            # Truncate long values
            display = val if len(val) <= num_w - 1 else val[: num_w - 2] + "…"
            row_parts.append(f"{MAGENTA}{display:^{num_w}}{RESET}")
        print("  " + "  ".join(row_parts))

    print()


def print_stds_table(
    num_stds: pd.DataFrame,
    str_uniques: pd.DataFrame,
    numeric_cols: list[str],
    string_cols: list[str],
    k: int,
) -> None:
    """Print within-cluster std dev (numeric) and unique value count (string)."""
    all_cols = numeric_cols + string_cols
    if not all_cols:
        return

    print(section_header("SPREAD  (numeric: std dev  |  string: unique value count per cluster)"))

    col_w = max(max(len(c) for c in all_cols), 14) + 2
    num_w = 14

    header = [f"{'Variable':{col_w}}  {'Type':<8}"]
    for c in range(k):
        header.append(f"{ccluster(c)}{BOLD}{'Cluster '+str(c):^{num_w}}{RESET}")
    print("\n  " + "  ".join(header))
    print(f"  {DIM}{'─' * (col_w + 10 + (num_w + 2) * k)}{RESET}")

    # Numeric: std dev
    for col in numeric_cols:
        if num_stds.empty or col not in num_stds.columns:
            continue
        row_parts = [f"{BOLD}{col:{col_w}}{RESET}  {DIM}{'numeric':<8}{RESET}"]
        for c in range(k):
            v = num_stds.loc[c, col] if c in num_stds.index else float("nan")
            if math.isnan(v):
                row_parts.append(f"{DIM}{'n/a':^{num_w}}{RESET}")
            else:
                row_parts.append(f"{DIM}{v:>{num_w}.4f}{RESET}")
        print("  " + "  ".join(row_parts))

    # String: unique count + hint about diversity
    for col in string_cols:
        if str_uniques.empty or col not in str_uniques.columns:
            continue
        row_parts = [f"{BOLD}{col:{col_w}}{RESET}  {MAGENTA}{'string':<8}{RESET}"]
        for c in range(k):
            n_uniq = str_uniques.loc[c, col] if c in str_uniques.index else 0
            label = f"{n_uniq} unique"
            row_parts.append(f"{MAGENTA}{label:^{num_w}}{RESET}")
        print("  " + "  ".join(row_parts))

    print()


def print_string_profiles(
    df_original: pd.DataFrame,
    labels: np.ndarray,
    string_cols: list[str],
    k: int,
    top_n: int = 3,
) -> None:
    """Print per-cluster category breakdown for each string column."""
    if not string_cols:
        return

    print(section_header(f"STRING COLUMN PROFILES  (top {top_n} categories per cluster)"))

    df = df_original.copy()
    df["_cluster"] = labels

    for col in string_cols:
        col_w = max(len(col), 16)
        print(f"\n  {BOLD}{MAGENTA}{col}{RESET}")
        print(f"  {DIM}{'─' * 70}{RESET}")

        # Header row
        header = f"  {'Category':<20}  {'':>6}"
        for c in range(k):
            header += f"  {ccluster(c)}{BOLD}{'Cluster '+str(c):^14}{RESET}"
        print(header)
        print(f"  {DIM}{'─' * 70}{RESET}")

        # All unique categories across full column, sorted by overall frequency
        all_cats = df[col].value_counts().index.tolist()

        # Per-cluster counts and percentages
        cluster_counts: dict[int, pd.Series] = {}
        cluster_sizes:  dict[int, int] = {}
        for c in range(k):
            sub = df[df["_cluster"] == c][col]
            cluster_sizes[c] = len(sub)
            cluster_counts[c] = sub.value_counts()

        # Show top_n categories by overall frequency
        for cat in all_cats[:top_n]:
            row = f"  {str(cat):<20}  {'':<6}"
            for c in range(k):
                count = cluster_counts[c].get(cat, 0)
                pct   = count / cluster_sizes[c] * 100 if cluster_sizes[c] > 0 else 0
                bar   = "█" * int(pct / 10)  # 0–10 chars
                row  += f"  {ccluster(c)}{bar:<10}{RESET} {pct:4.0f}%"
            print(row)

        # Remainder row if more categories exist
        remaining = len(all_cats) - top_n
        if remaining > 0:
            print(f"  {DIM}  … {remaining} more categor{'y' if remaining == 1 else 'ies'} not shown{RESET}")

    print()


def print_performance(
    km: KMeans,
    X: np.ndarray,
    labels: np.ndarray,
    k: int,
    scaled: bool,
) -> None:
    print(section_header("CLUSTERING PERFORMANCE"))

    inertia = km.inertia_

    if k > 1 and len(set(labels)) > 1:
        sil   = silhouette_score(X, labels)
        db    = davies_bouldin_score(X, labels)
    else:
        sil = db = float("nan")

    # Inertia
    print(f"\n  {BOLD}Inertia (WCSS){RESET}")
    print(f"  {DIM}Sum of squared distances of samples to their cluster centres.{RESET}")
    print(f"  {YELLOW}{BOLD}{inertia:,.4f}{RESET}")
    print(f"  {DIM}Lower is better. Compare across different k values.{RESET}")

    # Silhouette
    print(f"\n  {BOLD}Silhouette Score{RESET}")
    print(f"  {DIM}Range [-1, 1]. Measures how similar a point is to its own cluster{RESET}")
    print(f"  {DIM}vs other clusters. Higher = better separation.{RESET}")
    if not math.isnan(sil):
        sil_bar_len = int((sil + 1) / 2 * 40)
        sil_color = GREEN if sil > 0.5 else (YELLOW if sil > 0.25 else RED)
        print(f"  {sil_color}{BOLD}{'█' * sil_bar_len:<40}{RESET}  {sil:+.4f}")
        if sil >= 0.7:   rating = f"{GREEN}Excellent{RESET}"
        elif sil >= 0.5: rating = f"{GREEN}Good{RESET}"
        elif sil >= 0.25:rating = f"{YELLOW}Fair{RESET}"
        else:            rating = f"{RED}Weak{RESET}"
        print(f"  Rating: {rating}")
    else:
        print(f"  {DIM}Not computable (k=1 or single cluster){RESET}")

    # Davies-Bouldin
    print(f"\n  {BOLD}Davies-Bouldin Index{RESET}")
    print(f"  {DIM}Avg ratio of within-cluster scatter to between-cluster separation.{RESET}")
    print(f"  {DIM}Lower is better. 0 = perfect separation.{RESET}")
    if not math.isnan(db):
        db_color = GREEN if db < 0.5 else (YELLOW if db < 1.0 else RED)
        print(f"  {db_color}{BOLD}{db:.4f}{RESET}")
        if db < 0.5:    rating = f"{GREEN}Excellent{RESET}"
        elif db < 1.0:  rating = f"{GREEN}Good{RESET}"
        elif db < 1.5:  rating = f"{YELLOW}Fair{RESET}"
        else:           rating = f"{RED}Weak{RESET}"
        print(f"  Rating: {rating}")
    else:
        print(f"  {DIM}Not computable (k=1 or single cluster){RESET}")

    # Convergence
    print(f"\n  {BOLD}Convergence{RESET}")
    print(f"  Iterations to converge: {BOLD}{km.n_iter_}{RESET}")
    if scaled:
        print(f"  {DIM}⚠  Inertia is on the scaled space. Raw-space inertia not shown.{RESET}")

    print(f"\n{hr()}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="kmeans_cluster.py",
        description="Generic K-Means clustering for CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python kmeans_cluster.py iris.csv
  python kmeans_cluster.py iris.csv --k 3
  python kmeans_cluster.py data.csv --k 5 --cols age income score --scale
  python kmeans_cluster.py data.csv --auto-k --k-max 8
        """,
    )
    parser.add_argument("filepath", help="Path to the CSV file")
    parser.add_argument("--k", type=int, default=3, help="Number of clusters (default: 3)")
    parser.add_argument("--cols", nargs="+", help="Specific column names to use (default: all numeric)")
    parser.add_argument("--scale", action="store_true", help="Standardise features before clustering (recommended for mixed-scale data)")
    parser.add_argument("--max-iter", type=int, default=300, help="Max iterations per run (default: 300)")
    parser.add_argument("--runs", type=int, default=10, help="Number of random initialisations (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--auto-k", action="store_true", help="Automatically select best k via silhouette score")
    parser.add_argument("--k-max", type=int, default=8, help="Upper bound for auto-k search (default: 8)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load data (handles string encoding internally) ──
    df_raw, df_encoded, df_original, numeric_cols, string_cols, encoders = load_and_prepare(
        args.filepath, args.cols
    )

    all_feature_cols = numeric_cols + string_cols
    X_raw = df_encoded[all_feature_cols].values.astype(float)

    # ── Scale (only makes sense on the encoded matrix) ──
    if args.scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
    else:
        X = X_raw

    # ── Auto-select k ──
    if args.auto_k:
        k = find_optimal_k(X, k_max=args.k_max, max_iter=args.max_iter, n_init=args.runs)
    else:
        k = args.k
        if k < 2:
            print(f"{RED}✗ k must be ≥ 2.{RESET}")
            sys.exit(1)
        if k > len(df_encoded):
            print(f"{RED}✗ k ({k}) cannot exceed number of rows ({len(df_encoded)}).{RESET}")
            sys.exit(1)

    # ── Fit ──
    km, labels = run_kmeans(X, k, max_iter=args.max_iter, n_init=args.runs, random_state=args.seed)

    # ── Stats ──
    num_means, num_stds, str_modes, str_uniques, sizes = compute_cluster_stats(
        df_encoded, df_original, labels, numeric_cols, string_cols, k
    )

    # ── Print ──
    print_banner(
        args.filepath, k,
        n_rows=len(df_encoded),
        n_numeric=len(numeric_cols),
        n_string=len(string_cols),
        scaled=args.scale,
    )

    feature_summary = []
    if numeric_cols:
        feature_summary.append(f"{BOLD}Numeric:{RESET} {', '.join(numeric_cols)}")
    if string_cols:
        feature_summary.append(f"{BOLD}{MAGENTA}String:{RESET} {', '.join(string_cols)}")
    for line in feature_summary:
        print(f"  {line}")
    if args.scale and string_cols:
        print(f"  {DIM}⚠  Scaling applied to label-encoded string columns too.{RESET}")
    print(f"  {DIM}(Numeric stats in original units; string stats show original category labels){RESET}")

    print_cluster_sizes(sizes, k, len(df_encoded))
    print_means_table(num_means, str_modes, numeric_cols, string_cols, k)
    print_stds_table(num_stds, str_uniques, numeric_cols, string_cols, k)
    print_string_profiles(df_original, labels, string_cols, k)
    print_performance(km, X, labels, k, args.scale)


if __name__ == "__main__":
    main()
