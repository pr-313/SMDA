#!/usr/bin/env python3
"""
kmeans_cluster.py — Generic K-Means clustering tool for CSV inputs.

Usage:
    python kmeans_cluster.py data.csv
    python kmeans_cluster.py data.csv --k 4
    python kmeans_cluster.py data.csv --k 4 --cluster-cols col1 col2 col3
    python kmeans_cluster.py data.csv --k 4 --cluster-cols col1 col2 --display-cols col3 col4
    python kmeans_cluster.py data.csv --k 4 --scale --max-iter 500 --runs 20
    python kmeans_cluster.py data.csv --auto-k --k-max 8
    python kmeans_cluster.py data.csv --k 3 --export report.html

--cluster-cols  Columns used to fit K-Means (default: all usable columns).
--display-cols  Extra columns profiled per cluster but NOT used for clustering.
                Useful to inspect variables without letting them influence assignments.

String columns are automatically label-encoded for clustering.
In output, string columns show mode (top category) and unique count
instead of mean/std, plus a per-cluster category breakdown table.

Outputs (all in terminal):
    - Cluster averages table (numeric cols) / mode table (string cols)
    - CV % table (numeric) / unique-count + top categories (string)
    - Cluster category profiles (string columns only, if any)
    - Display-only column profiles (if --display-cols provided)
    - Cluster sizes
    - Performance metrics: Inertia, Silhouette Score, Davies-Bouldin Index

Optional:
    --export report.html   Write a self-contained HTML report alongside terminal output
"""

import argparse
import sys
import math
import html as html_lib
from datetime import datetime
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


def load_display_cols(
    df_raw: pd.DataFrame,
    display_col_names: list[str],
    cluster_col_names: list[str],
    index: "pd.Index",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Prepare display-only columns from an already-loaded raw dataframe.

    Display cols are profiled per cluster but never used to fit K-Means.
    They are aligned to the same row index as the clustering dataframe.

    Args:
        df_raw            — the full raw dataframe from load_and_prepare
        display_col_names — column names requested via --display-cols
        cluster_col_names — columns already used for clustering (excluded from display)
        index             — row index of the post-NaN-drop clustering df, to align rows

    Returns:
        df_disp_enc  — display df with strings label-encoded (for stats only)
        df_disp_orig — display df with original string values (for mode/profile display)
        disp_num     — numeric display column names
        disp_str     — string display column names
    """
    # Validate
    unknown = [c for c in display_col_names if c not in df_raw.columns]
    if unknown:
        print(f"{RED}✗ --display-cols not found in CSV: {unknown}{RESET}")
        print(f"  Available: {list(df_raw.columns)}")
        sys.exit(1)

    overlap = [c for c in display_col_names if c in cluster_col_names]
    if overlap:
        print(f"{YELLOW}⚠  Columns in both --cluster-cols and --display-cols (will only show in cluster section): {overlap}{RESET}")
        display_col_names = [c for c in display_col_names if c not in overlap]

    if not display_col_names:
        return pd.DataFrame(), pd.DataFrame(), [], []

    # Align to the same rows that survived NaN-dropping in load_and_prepare
    df = df_raw.loc[index, display_col_names].copy()

    disp_num = list(df.select_dtypes(include=[np.number]).columns)
    disp_str = list(df.select_dtypes(include=["object", "category", "string"]).columns)

    # Heuristic: warn about high-cardinality display string cols (don't skip — just warn)
    n_rows = len(df)
    for col in disp_str:
        if df[col].nunique() > max(2, n_rows * 0.5):
            print(f"{YELLOW}⚠  Display column '{col}' has high cardinality ({df[col].nunique()} unique values) — profile may be noisy{RESET}")

    df_orig = df.copy()
    df_enc  = df.copy()

    # Fill NaNs
    for col in disp_num:
        fill = df_enc[col].mean()
        df_enc[col]  = df_enc[col].fillna(fill)
        df_orig[col] = df_orig[col].fillna(fill)

    for col in disp_str:
        mode_val = df_enc[col].mode()
        fill_val = mode_val.iloc[0] if not mode_val.empty else "unknown"
        df_enc[col]  = df_enc[col].fillna(fill_val)
        df_orig[col] = df_orig[col].fillna(fill_val)
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))

    return df_enc, df_orig, disp_num, disp_str


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
    """Return per-cluster stats for all columns (cluster + display combined).

    numeric_cols and string_cols should already include both cluster and display
    columns — the caller merges them before calling this.

    Returns:
        num_means, num_cvs, str_modes, str_uniques, sizes
    """
    df_enc  = df_encoded.copy()
    df_enc["_cluster"] = labels
    df_orig = df_original.copy()
    df_orig["_cluster"] = labels

    sizes = df_enc.groupby("_cluster").size().rename("Count")

    # Only compute on cols that actually exist in the respective df
    valid_num = [c for c in numeric_cols if c in df_enc.columns]
    valid_str = [c for c in string_cols  if c in df_orig.columns]

    if valid_num:
        grp        = df_enc.groupby("_cluster")[valid_num]
        _means     = grp.mean()
        _stds      = grp.std(ddof=1)
        num_means  = _means
        safe_means = _means.abs().replace(0, float("nan"))
        num_cvs    = (_stds / safe_means) * 100
    else:
        num_means = pd.DataFrame()
        num_cvs   = pd.DataFrame()

    if valid_str:
        str_modes   = df_orig.groupby("_cluster")[valid_str].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else "n/a"
        )
        str_uniques = df_orig.groupby("_cluster")[valid_str].nunique()
    else:
        str_modes   = pd.DataFrame()
        str_uniques = pd.DataFrame()

    return num_means, num_cvs, str_modes, str_uniques, sizes


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
    display_cols: Optional[set[str]] = None,
) -> None:
    """Print cluster means (numeric) and cluster modes (string) in one unified table.

    display_cols: set of column names that are display-only (not used for clustering).
    These rows are marked with ◇ instead of ◆ in the Type column.
    """
    all_cols = numeric_cols + string_cols
    if not all_cols:
        return

    display_cols = display_cols or set()

    print(section_header("CLUSTER MEANS / MODE  (numeric: mean  |  string: top category)"))

    col_w = max(max(len(c) for c in all_cols), 14) + 2
    num_w = 14

    header = [f"{'Variable':{col_w}}  {'Type':<12}"]
    for c in range(k):
        header.append(f"{ccluster(c)}{BOLD}{'Cluster '+str(c):^{num_w}}{RESET}")
    print("\n  " + "  ".join(header))
    print(f"  {DIM}{'─' * (col_w + 14 + (num_w + 2) * k)}{RESET}")

    for col in numeric_cols:
        if num_means.empty or col not in num_means.columns:
            continue
        marker = f"{DIM}◇ display{RESET}" if col in display_cols else f"{DIM}◆ numeric{RESET}"
        row_parts = [f"{BOLD}{col:{col_w}}{RESET}  {marker:<20}"]
        vals  = [num_means.loc[c, col] for c in range(k) if c in num_means.index]
        max_v = max(abs(v) for v in vals) if vals else 1
        for c in range(k):
            if c not in num_means.index:
                row_parts.append(f"{'n/a':^{num_w}}")
                continue
            v         = num_means.loc[c, col]
            intensity = abs(v) / max_v if max_v != 0 else 0
            color     = ccluster(c) if intensity > 0.85 else ""
            row_parts.append(f"{color}{v:>{num_w}.4f}{RESET}")
        print("  " + "  ".join(row_parts))

    for col in string_cols:
        if str_modes.empty or col not in str_modes.columns:
            continue
        marker = f"{DIM}◇ display{RESET}" if col in display_cols else f"{MAGENTA}◆ string {RESET}"
        row_parts = [f"{BOLD}{col:{col_w}}{RESET}  {marker:<20}"]
        for c in range(k):
            if c not in str_modes.index:
                row_parts.append(f"{'n/a':^{num_w}}")
                continue
            val     = str(str_modes.loc[c, col])
            display = val if len(val) <= num_w - 1 else val[:num_w - 2] + "…"
            color   = DIM if col in display_cols else MAGENTA
            row_parts.append(f"{color}{display:^{num_w}}{RESET}")
        print("  " + "  ".join(row_parts))

    if display_cols:
        print(f"\n  {DIM}◆ used for clustering   ◇ display only (not used for clustering){RESET}")
    print()


def print_cv_table(
    num_cvs: pd.DataFrame,
    str_uniques: pd.DataFrame,
    numeric_cols: list[str],
    string_cols: list[str],
    k: int,
    display_cols: Optional[set[str]] = None,
) -> None:
    """Print within-cluster CV% (numeric) and unique value count (string).

    display_cols: set of column names that are display-only.
    """
    all_cols = numeric_cols + string_cols
    if not all_cols:
        return

    display_cols = display_cols or set()

    print(section_header("SPREAD  (numeric: CV%  =  std / |mean| × 100  |  string: unique values)"))

    col_w = max(max(len(c) for c in all_cols), 14) + 2
    num_w = 14

    header = [f"{'Variable':{col_w}}  {'Type':<12}"]
    for c in range(k):
        header.append(f"{ccluster(c)}{BOLD}{'Cluster '+str(c):^{num_w}}{RESET}")
    print("\n  " + "  ".join(header))
    print(f"  {DIM}{'─' * (col_w + 14 + (num_w + 2) * k)}{RESET}")

    for col in numeric_cols:
        if num_cvs.empty or col not in num_cvs.columns:
            continue
        marker = f"{DIM}◇ display{RESET}" if col in display_cols else f"{DIM}◆ numeric{RESET}"
        row_parts = [f"{BOLD}{col:{col_w}}{RESET}  {marker:<20}"]
        for c in range(k):
            if c not in num_cvs.index:
                row_parts.append(f"{'n/a':^{num_w}}")
                continue
            v = num_cvs.loc[c, col]
            if math.isnan(v):
                row_parts.append(f"{DIM}{'undef':^{num_w}}{RESET}")
            else:
                color = GREEN if v < 15 else (YELLOW if v < 35 else RED)
                # Dim display-only rows slightly to visually de-emphasise
                if col in display_cols:
                    color = DIM
                label = f"{v:6.1f}%"
                row_parts.append(f"{color}{label:^{num_w}}{RESET}")
        print("  " + "  ".join(row_parts))

    for col in string_cols:
        if str_uniques.empty or col not in str_uniques.columns:
            continue
        marker = f"{DIM}◇ display{RESET}" if col in display_cols else f"{MAGENTA}◆ string {RESET}"
        row_parts = [f"{BOLD}{col:{col_w}}{RESET}  {marker:<20}"]
        for c in range(k):
            n_uniq = str_uniques.loc[c, col] if c in str_uniques.index else 0
            color  = DIM if col in display_cols else MAGENTA
            row_parts.append(f"{color}{f'{n_uniq} unique':^{num_w}}{RESET}")
        print("  " + "  ".join(row_parts))

    print(f"\n  {DIM}CV% guide:  {GREEN}< 15% tight{RESET}  {DIM}│  {YELLOW}15–35% moderate{RESET}  {DIM}│  {RED}> 35% loose{RESET}")
    if display_cols:
        print(f"  {DIM}◆ used for clustering   ◇ display only (not used for clustering){RESET}")
    print()


def print_string_profiles(
    df_original: pd.DataFrame,
    labels: np.ndarray,
    string_cols: list[str],
    k: int,
    top_n: int = 3,
    display_cols: Optional[set[str]] = None,
) -> None:
    """Print per-cluster category breakdown for each string column."""
    if not string_cols:
        return

    display_cols = display_cols or set()

    print(section_header(f"STRING COLUMN PROFILES  (top {top_n} categories per cluster)"))

    df = df_original.copy()
    df["_cluster"] = labels

    for col in string_cols:
        is_display = col in display_cols
        marker     = f"  {DIM}◇ display only{RESET}" if is_display else ""
        print(f"\n  {BOLD}{MAGENTA if not is_display else DIM}{col}{RESET}{marker}")
        print(f"  {DIM}{'─' * 70}{RESET}")

        header = f"  {'Category':<20}  {'':>6}"
        for c in range(k):
            header += f"  {ccluster(c)}{BOLD}{'Cluster '+str(c):^14}{RESET}"
        print(header)
        print(f"  {DIM}{'─' * 70}{RESET}")

        all_cats = df[col].value_counts().index.tolist()

        cluster_counts: dict[int, pd.Series] = {}
        cluster_sizes:  dict[int, int] = {}
        for c in range(k):
            sub = df[df["_cluster"] == c][col]
            cluster_sizes[c]  = len(sub)
            cluster_counts[c] = sub.value_counts()

        for cat in all_cats[:top_n]:
            row = f"  {str(cat):<20}  {'':<6}"
            for c in range(k):
                count = cluster_counts[c].get(cat, 0)
                pct   = count / cluster_sizes[c] * 100 if cluster_sizes[c] > 0 else 0
                bar   = "█" * int(pct / 10)
                color = DIM if is_display else ccluster(c)
                row  += f"  {color}{bar:<10}{RESET} {pct:4.0f}%"
            print(row)

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
    inertia: float,
    sil: float,
    db: float,
) -> None:
    print(section_header("CLUSTERING PERFORMANCE"))

    # (inertia, sil, db already computed in main and passed in)

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


# ─── HTML export ──────────────────────────────────────────────────────────────

# Palette used consistently across HTML report (matches terminal cluster colours)
HTML_CLUSTER_PALETTE = [
    "#16a34a",  # green
    "#ca8a04",  # yellow
    "#0891b2",  # cyan
    "#9333ea",  # purple
    "#dc2626",  # red
    "#2563eb",  # blue
    "#ea580c",  # orange
    "#db2777",  # pink
]

HTML_CV_COLORS = {
    "tight":    ("#166534", "#dcfce7"),   # text, bg
    "moderate": ("#854d0e", "#fef9c3"),
    "loose":    ("#991b1b", "#fee2e2"),
}


def _hc(k: int) -> str:
    """HTML hex colour for cluster k."""
    return HTML_CLUSTER_PALETTE[k % len(HTML_CLUSTER_PALETTE)]


def _cv_class(v: float) -> str:
    if math.isnan(v):  return "cv-undef"
    if v < 15:         return "cv-tight"
    if v < 35:         return "cv-moderate"
    return "cv-loose"


class HtmlReport:
    """Accumulates report sections as HTML strings, then renders to a full page."""

    def __init__(self, filepath: str, k: int, n_rows: int,
                 n_numeric: int, n_string: int, scaled: bool,
                 numeric_cols: list[str], string_cols: list[str],
                 display_cols: Optional[set[str]] = None) -> None:
        self.filepath    = filepath
        self.k           = k
        self.n_rows      = n_rows
        self.n_numeric   = n_numeric
        self.n_string    = n_string
        self.scaled      = scaled
        self.numeric_cols = numeric_cols
        self.string_cols  = string_cols
        self.display_cols = display_cols or set()
        self._sections: list[str] = []   # ordered HTML blocks

    def _e(self, s: str) -> str:
        """HTML-escape a value."""
        return html_lib.escape(str(s))

    # ── Cluster sizes ──────────────────────────────────────────────────────────
    def add_cluster_sizes(self, sizes: "pd.Series", n_total: int) -> None:
        rows_html = ""
        for cid, count in sizes.items():
            pct   = count / n_total * 100
            color = _hc(cid)
            rows_html += f"""
            <tr>
              <td><span class="cluster-badge" style="background:{color}20;color:{color};border:1px solid {color}60">
                Cluster {cid}</span></td>
              <td>
                <div class="bar-wrap">
                  <div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div>
                </div>
              </td>
              <td class="num">{count:,}</td>
              <td class="num">{pct:.1f}%</td>
            </tr>"""
        self._sections.append(f"""
        <section>
          <h2>Cluster Sizes</h2>
          <table class="simple-table">
            <thead><tr><th>Cluster</th><th>Distribution</th><th>Rows</th><th>Share</th></tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
        </section>""")

    # ── Means / mode table ─────────────────────────────────────────────────────
    def add_means_table(self, num_means: "pd.DataFrame", str_modes: "pd.DataFrame") -> None:
        all_cols = self.numeric_cols + self.string_cols
        if not all_cols:
            return

        has_display = bool(self.display_cols)

        header = "<tr><th>Variable</th><th>Type</th>"
        for c in range(self.k):
            color = _hc(c)
            header += f'<th style="color:{color}">Cluster {c}</th>'
        header += "</tr>"

        rows_html = ""
        for col in self.numeric_cols:
            if num_means.empty or col not in num_means.columns:
                continue
            is_disp   = col in self.display_cols
            type_html = '<span class="type-badge disp-badge">◇ display</span>' if is_disp \
                        else '<span class="type-badge num-badge">◆ numeric</span>'
            vals  = [num_means.loc[c, col] if c in num_means.index else float("nan")
                     for c in range(self.k)]
            max_v = max((abs(v) for v in vals if not math.isnan(v)), default=1) or 1
            cells = ""
            for c, v in enumerate(vals):
                intensity = abs(v) / max_v if max_v != 0 else 0
                bold  = "font-weight:600;" if (intensity > 0.85 and not is_disp) else ""
                color = f"color:{_hc(c)};" if (intensity > 0.85 and not is_disp) else ("color:#94a3b8;" if is_disp else "")
                cells += f'<td style="{bold}{color}">{v:,.4f}</td>'
            rows_html += f"<tr{'  class=\"disp-row\"' if is_disp else ''}><td class='col-name'>{self._e(col)}</td><td>{type_html}</td>{cells}</tr>"

        for col in self.string_cols:
            if str_modes.empty or col not in str_modes.columns:
                continue
            is_disp   = col in self.display_cols
            type_html = '<span class="type-badge disp-badge">◇ display</span>' if is_disp \
                        else '<span class="type-badge str-badge">◆ string</span>'
            cells = ""
            for c in range(self.k):
                val   = str_modes.loc[c, col] if c in str_modes.index else "n/a"
                style = "color:#94a3b8" if is_disp else ""
                cells += f'<td style="{style}"><span class="cat-val">{self._e(val)}</span></td>'
            rows_html += f"<tr{'  class=\"disp-row\"' if is_disp else ''}><td class='col-name'>{self._e(col)}</td><td>{type_html}</td>{cells}</tr>"

        legend = '<p class="cv-legend">◆ used for clustering &nbsp;·&nbsp; ◇ display only (not used for clustering)</p>' \
                 if has_display else ""

        self._sections.append(f"""
        <section>
          <h2>Cluster Means <span class="subtitle">numeric: mean &nbsp;·&nbsp; string: top category</span></h2>
          <table class="data-table">
            <thead>{header}</thead>
            <tbody>{rows_html}</tbody>
          </table>
          {legend}
        </section>""")

    # ── CV table ───────────────────────────────────────────────────────────────
    def add_cv_table(self, num_cvs: "pd.DataFrame", str_uniques: "pd.DataFrame") -> None:
        all_cols = self.numeric_cols + self.string_cols
        if not all_cols:
            return

        has_display = bool(self.display_cols)

        header = "<tr><th>Variable</th><th>Type</th>"
        for c in range(self.k):
            color = _hc(c)
            header += f'<th style="color:{color}">Cluster {c}</th>'
        header += "</tr>"

        rows_html = ""
        for col in self.numeric_cols:
            if num_cvs.empty or col not in num_cvs.columns:
                continue
            is_disp   = col in self.display_cols
            type_html = '<span class="type-badge disp-badge">◇ display</span>' if is_disp \
                        else '<span class="type-badge num-badge">◆ numeric</span>'
            cells = ""
            for c in range(self.k):
                v   = num_cvs.loc[c, col] if c in num_cvs.index else float("nan")
                cls = _cv_class(v)
                if is_disp:
                    cls = "cv-undef"  # render dimly for display cols
                lbl = "undef" if math.isnan(v) else f"{v:.1f}%"
                cells += f'<td><span class="cv-pill {cls}">{lbl}</span></td>'
            rows_html += f"<tr{'  class=\"disp-row\"' if is_disp else ''}><td class='col-name'>{self._e(col)}</td><td>{type_html}</td>{cells}</tr>"

        for col in self.string_cols:
            if str_uniques.empty or col not in str_uniques.columns:
                continue
            is_disp   = col in self.display_cols
            type_html = '<span class="type-badge disp-badge">◇ display</span>' if is_disp \
                        else '<span class="type-badge str-badge">◆ string</span>'
            cells = ""
            for c in range(self.k):
                n     = str_uniques.loc[c, col] if c in str_uniques.index else 0
                style = "color:#94a3b8" if is_disp else ""
                cells += f'<td style="{style}"><span class="cat-val">{n} unique</span></td>'
            rows_html += f"<tr{'  class=\"disp-row\"' if is_disp else ''}><td class='col-name'>{self._e(col)}</td><td>{type_html}</td>{cells}</tr>"

        legend = """
        <p class="cv-legend">
          <span class="cv-pill cv-tight">tight &lt;15%</span>
          <span class="cv-pill cv-moderate">moderate 15–35%</span>
          <span class="cv-pill cv-loose">loose &gt;35%</span>
          &nbsp; CV% = std / |mean| × 100
        </p>"""
        if has_display:
            legend += '<p class="cv-legend">◆ used for clustering &nbsp;·&nbsp; ◇ display only</p>'

        self._sections.append(f"""
        <section>
          <h2>Within-Cluster Spread <span class="subtitle">CV% for numeric &nbsp;·&nbsp; unique values for string</span></h2>
          <table class="data-table">
            <thead>{header}</thead>
            <tbody>{rows_html}</tbody>
          </table>
          {legend}
        </section>""")

    # ── String profiles ────────────────────────────────────────────────────────
    def add_string_profiles(self, df_original: "pd.DataFrame",
                            labels: "np.ndarray", top_n: int = 3) -> None:
        if not self.string_cols:
            return

        import numpy as np
        df = df_original.copy()
        df["_cluster"] = labels

        profile_html = ""
        for col in self.string_cols:
            is_disp  = col in self.display_cols
            marker   = ' <span class="disp-badge-inline">◇ display only</span>' if is_disp else ""
            all_cats = df[col].value_counts().index.tolist()
            cluster_counts: dict = {}
            cluster_sizes:  dict = {}
            for c in range(self.k):
                sub = df[df["_cluster"] == c][col]
                cluster_sizes[c]  = len(sub)
                cluster_counts[c] = sub.value_counts()

            th_cells = "<th>Category</th>"
            for c in range(self.k):
                color = _hc(c)
                th_cells += f'<th style="color:{color}">Cluster {c}</th>'

            body_rows = ""
            for cat in all_cats[:top_n]:
                tds = f"<td class='col-name'>{self._e(cat)}</td>"
                for c in range(self.k):
                    count = cluster_counts[c].get(cat, 0)
                    pct   = count / cluster_sizes[c] * 100 if cluster_sizes[c] > 0 else 0
                    color = "#94a3b8" if is_disp else _hc(c)
                    tds  += f"""<td>
                      <div class="bar-wrap">
                        <div class="bar-fill" style="width:{pct:.0f}%;background:{color}"></div>
                      </div>
                      <span class="pct-label">{pct:.0f}%</span>
                    </td>"""
                body_rows += f"<tr>{tds}</tr>"

            remaining = len(all_cats) - top_n
            if remaining > 0:
                body_rows += f'<tr><td colspan="{self.k + 1}" class="more-note">… {remaining} more categor{"y" if remaining == 1 else "ies"} not shown</td></tr>'

            row_class = ' class="disp-profile"' if is_disp else ""
            profile_html += f"""
            <div class="profile-block"{row_class}>
              <h3>{self._e(col)}{marker}</h3>
              <table class="data-table">
                <thead><tr>{th_cells}</tr></thead>
                <tbody>{body_rows}</tbody>
              </table>
            </div>"""

        self._sections.append(f"""
        <section>
          <h2>String Column Profiles <span class="subtitle">top {top_n} categories per cluster</span></h2>
          {profile_html}
        </section>""")

    # ── Performance ────────────────────────────────────────────────────────────
    def add_performance(self, inertia: float, sil: float, db: float,
                        n_iter: int) -> None:

        def rating_badge(label: str, good_cond: bool, fair_cond: bool) -> str:
            if good_cond:   cls, txt = "rating-good",     "Excellent" if "0.7" in label or "0.5" in label else "Good"
            elif fair_cond: cls, txt = "rating-fair",     "Fair"
            else:           cls, txt = "rating-weak",     "Weak"
            return f'<span class="rating-badge {cls}">{txt}</span>'

        # Silhouette
        if not math.isnan(sil):
            sil_pct   = int((sil + 1) / 2 * 100)
            sil_color = "#16a34a" if sil > 0.5 else ("#ca8a04" if sil > 0.25 else "#dc2626")
            if sil >= 0.7:   sil_rating = '<span class="rating-badge rating-good">Excellent</span>'
            elif sil >= 0.5: sil_rating = '<span class="rating-badge rating-good">Good</span>'
            elif sil >= 0.25:sil_rating = '<span class="rating-badge rating-fair">Fair</span>'
            else:            sil_rating = '<span class="rating-badge rating-weak">Weak</span>'
            sil_html = f"""
              <div class="metric-bar-wrap">
                <div class="metric-bar-fill" style="width:{sil_pct}%;background:{sil_color}"></div>
              </div>
              <div class="metric-value">{sil:+.4f} &nbsp;{sil_rating}</div>"""
        else:
            sil_html = '<p class="muted">Not computable (k=1 or single cluster)</p>'

        # Davies-Bouldin
        if not math.isnan(db):
            if db < 0.5:    db_rating = '<span class="rating-badge rating-good">Excellent</span>'
            elif db < 1.0:  db_rating = '<span class="rating-badge rating-good">Good</span>'
            elif db < 1.5:  db_rating = '<span class="rating-badge rating-fair">Fair</span>'
            else:           db_rating = '<span class="rating-badge rating-weak">Weak</span>'
            db_color = "#16a34a" if db < 0.5 else ("#ca8a04" if db < 1.0 else "#dc2626")
            db_html  = f'<div class="metric-value" style="color:{db_color};font-size:1.5rem;font-weight:700">{db:.4f}</div><div>{db_rating}</div>'
        else:
            db_html = '<p class="muted">Not computable</p>'

        scaled_note = '<p class="muted">⚠ Inertia is on the scaled space.</p>' if self.scaled else ""

        self._sections.append(f"""
        <section>
          <h2>Clustering Performance</h2>
          <div class="metric-grid">

            <div class="metric-card">
              <div class="metric-label">Inertia (WCSS)</div>
              <div class="metric-desc">Sum of squared distances to cluster centres. Lower is better.</div>
              <div class="metric-value" style="font-size:1.6rem;font-weight:700;color:#ca8a04">{inertia:,.4f}</div>
              {scaled_note}
            </div>

            <div class="metric-card">
              <div class="metric-label">Silhouette Score</div>
              <div class="metric-desc">Range [−1, 1]. Similarity to own cluster vs others. Higher is better.</div>
              {sil_html}
            </div>

            <div class="metric-card">
              <div class="metric-label">Davies-Bouldin Index</div>
              <div class="metric-desc">Avg scatter-to-separation ratio. Lower is better. 0 = perfect.</div>
              {db_html}
            </div>

            <div class="metric-card">
              <div class="metric-label">Convergence</div>
              <div class="metric-desc">Iterations until centroids stabilised.</div>
              <div class="metric-value" style="font-size:1.6rem;font-weight:700">{n_iter}</div>
            </div>

          </div>
        </section>""")

    # ── Render full page ───────────────────────────────────────────────────────
    def render(self) -> str:
        ts         = datetime.now().strftime("%Y-%m-%d %H:%M")
        num_label  = f"{self.n_numeric} numeric"
        str_label  = f", {self.n_string} string" if self.n_string else ""
        scale_note = "StandardScaler applied" if self.scaled else "No scaling"
        sections   = "\n".join(self._sections)

        cluster_css_vars = "\n".join(
            f"  --c{i}: {_hc(i)};"
            for i in range(self.k)
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>K-Means Report — {self._e(self.filepath)}</title>
<style>
  :root {{
    --sans: 'Segoe UI', system-ui, -apple-system, sans-serif;
    --mono: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
    --bg:      #f8fafc;
    --surface: #ffffff;
    --border:  #e2e8f0;
    --text:    #0f172a;
    --muted:   #64748b;
    --accent:  #1e40af;
    {cluster_css_vars}
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: var(--sans);
    background: var(--bg);
    color: var(--text);
    font-size: 14px;
    line-height: 1.6;
    padding: 0 0 64px;
  }}

  /* ── Header ── */
  .report-header {{
    background: var(--accent);
    color: white;
    padding: 36px 48px 32px;
  }}

  .report-header h1 {{
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
  }}

  .report-header .filepath {{
    font-family: var(--mono);
    font-size: 0.85rem;
    opacity: 0.8;
    margin-bottom: 18px;
  }}

  .meta-pills {{ display: flex; flex-wrap: wrap; gap: 8px; }}

  .meta-pill {{
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 99px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 500;
  }}

  /* ── Layout ── */
  .content {{ max-width: 1100px; margin: 0 auto; padding: 0 32px; }}

  section {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px 32px;
    margin-top: 24px;
  }}

  section h2 {{
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 18px;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--border);
    display: flex;
    align-items: baseline;
    gap: 10px;
  }}

  .subtitle {{
    font-size: 0.75rem;
    font-weight: 400;
    color: var(--muted);
  }}

  section h3 {{
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--accent);
    margin: 20px 0 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }}

  section h3:first-child {{ margin-top: 0; }}

  /* ── Tables ── */
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}

  th {{
    text-align: left;
    padding: 9px 14px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: var(--muted);
    border-bottom: 2px solid var(--border);
    white-space: nowrap;
  }}

  td {{
    padding: 9px 14px;
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
  }}

  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f8fafc; }}

  td.num {{ font-family: var(--mono); text-align: right; }}

  .col-name {{
    font-family: var(--mono);
    font-size: 0.82rem;
    color: var(--text);
    font-weight: 500;
    white-space: nowrap;
  }}

  .data-table td {{ font-family: var(--mono); font-size: 0.82rem; }}

  /* ── Badges ── */
  .cluster-badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 99px;
    font-size: 0.78rem;
    font-weight: 600;
    white-space: nowrap;
  }}

  .type-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }}

  .num-badge {{ background: #eff6ff; color: #1d4ed8; }}
  .str-badge {{ background: #fdf4ff; color: #7e22ce; }}

  .cat-val {{
    display: inline-block;
    background: #f1f5f9;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.8rem;
    color: #334155;
  }}

  /* ── CV pills ── */
  .cv-pill {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 99px;
    font-size: 0.78rem;
    font-weight: 600;
  }}
  .cv-tight    {{ background: #dcfce7; color: #166534; }}
  .cv-moderate {{ background: #fef9c3; color: #854d0e; }}
  .cv-loose    {{ background: #fee2e2; color: #991b1b; }}
  .cv-undef    {{ background: #f1f5f9; color: #64748b; }}

  .cv-legend {{
    margin-top: 14px;
    font-size: 0.78rem;
    color: var(--muted);
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }}

  /* ── Rating badges ── */
  .rating-badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.03em;
  }}
  .rating-good {{ background: #dcfce7; color: #166534; }}
  .rating-fair {{ background: #fef9c3; color: #854d0e; }}
  .rating-weak {{ background: #fee2e2; color: #991b1b; }}

  /* ── Bar charts ── */
  .bar-wrap {{
    background: #f1f5f9;
    border-radius: 99px;
    height: 8px;
    width: 160px;
    overflow: hidden;
    display: inline-block;
    vertical-align: middle;
  }}

  .bar-fill {{
    height: 100%;
    border-radius: 99px;
    transition: width 0.3s ease;
  }}

  .simple-table .bar-wrap {{ width: 200px; }}

  .pct-label {{
    font-size: 0.78rem;
    color: var(--muted);
    margin-left: 6px;
    font-family: var(--mono);
  }}

  /* ── Metric cards ── */
  .metric-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
  }}

  .metric-card {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
  }}

  .metric-label {{
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
  }}

  .metric-desc {{
    font-size: 0.78rem;
    color: var(--muted);
    margin-bottom: 12px;
    line-height: 1.5;
  }}

  .metric-value {{ margin-top: 6px; font-family: var(--mono); }}

  .metric-bar-wrap {{
    background: #e2e8f0;
    border-radius: 99px;
    height: 10px;
    overflow: hidden;
    margin: 10px 0 8px;
  }}

  .metric-bar-fill {{ height: 100%; border-radius: 99px; }}

  /* ── Misc ── */
  .muted {{ color: var(--muted); font-size: 0.8rem; font-style: italic; margin-top: 6px; }}
  .more-note {{ color: var(--muted); font-style: italic; font-size: 0.8rem; padding: 8px 14px; }}
  .profile-block {{ margin-bottom: 24px; }}
  .profile-block:last-child {{ margin-bottom: 0; }}
  .disp-badge {{ background: #f1f5f9; color: #64748b; }}
  .disp-row td {{ color: #94a3b8; }}
  .disp-row td.col-name {{ color: #64748b; }}
  .disp-badge-inline {{
    font-size: 0.7rem; font-weight: 600; color: #94a3b8;
    background: #f1f5f9; border-radius: 4px;
    padding: 1px 6px; margin-left: 6px; vertical-align: middle;
  }}
  .disp-profile h3 {{ color: #94a3b8; }}

  /* ── Print ── */
  @media print {{
    body {{ background: white; }}
    .report-header {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    .bar-fill, .metric-bar-fill, .cv-pill, .rating-badge, .cluster-badge,
    .type-badge, .num-badge, .str-badge {{
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }}
    section {{ break-inside: avoid; }}
  }}
</style>
</head>
<body>

<div class="report-header">
  <h1>K-Means Clustering Report</h1>
  <div class="filepath">{self._e(self.filepath)}</div>
  <div class="meta-pills">
    <span class="meta-pill">k = {self.k} clusters</span>
    <span class="meta-pill">{self.n_rows:,} rows</span>
    <span class="meta-pill">{num_label}{str_label} features</span>
    <span class="meta-pill">{scale_note}</span>
    <span class="meta-pill">Generated {ts}</span>
  </div>
</div>

<div class="content">
{sections}
</div>

</body>
</html>"""


def export_html(report: HtmlReport, path: str) -> None:
    """Write the HTML report to disk."""
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(report.render())
        print(f"\n{GREEN}✓ HTML report saved → {path}{RESET}")
    except OSError as e:
        print(f"{RED}✗ Could not write HTML report: {e}{RESET}")


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
  python kmeans_cluster.py data.csv --k 5 --cluster-cols age income score --scale
  python kmeans_cluster.py data.csv --k 3 --cluster-cols age income --display-cols score region
  python kmeans_cluster.py data.csv --auto-k --k-max 8
  python kmeans_cluster.py data.csv --k 3 --export report.html
        """,
    )
    parser.add_argument("filepath", help="Path to the CSV file")
    parser.add_argument("--k", type=int, default=3, help="Number of clusters (default: 3)")
    parser.add_argument("--cluster-cols", nargs="+", dest="cluster_cols",
                        help="Columns used to fit K-Means (default: all usable columns)")
    parser.add_argument("--display-cols", nargs="+", dest="display_cols",
                        help="Extra columns profiled per cluster but NOT used for clustering")
    # Backward-compat alias — silently maps to --cluster-cols
    parser.add_argument("--cols", nargs="+", dest="cols_legacy",
                        help=argparse.SUPPRESS)
    parser.add_argument("--scale", action="store_true", help="Standardise features before clustering (recommended for mixed-scale data)")
    parser.add_argument("--max-iter", type=int, default=300, help="Max iterations per run (default: 300)")
    parser.add_argument("--runs", type=int, default=10, help="Number of random initialisations (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--auto-k", action="store_true", help="Automatically select best k via silhouette score")
    parser.add_argument("--k-max", type=int, default=8, help="Upper bound for auto-k search (default: 8)")
    parser.add_argument("--export", metavar="FILE.html", default=None, help="Export a self-contained HTML report to this path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Backward-compat: --cols maps to --cluster-cols ──
    cluster_col_input = args.cluster_cols or args.cols_legacy or None
    display_col_input = args.display_cols or None

    if args.cols_legacy:
        print(f"{YELLOW}⚠  --cols is deprecated; use --cluster-cols instead{RESET}")

    # ── Load clustering data ──
    df_raw, df_encoded, df_original, numeric_cols, string_cols, encoders = load_and_prepare(
        args.filepath, cluster_col_input
    )

    # ── Load display-only columns and merge into the stats dataframes ──
    display_cols: set[str] = set()

    if display_col_input:
        all_cluster_cols = numeric_cols + string_cols
        df_disp_enc, df_disp_orig, disp_num, disp_str = load_display_cols(
            df_raw, display_col_input, all_cluster_cols, df_encoded.index
        )

        display_cols = set(disp_num + disp_str)

        # Merge display cols into the stats dataframes (right-join on index)
        if not df_disp_enc.empty:
            df_encoded  = df_encoded.join(df_disp_enc,  how="left")
            df_original = df_original.join(df_disp_orig, how="left")

        # Unified col lists: cluster cols first, then display cols
        all_numeric = numeric_cols + disp_num
        all_string  = string_cols  + disp_str

        if disp_num or disp_str:
            parts = []
            if disp_num: parts.append(f"numeric: {', '.join(disp_num)}")
            if disp_str: parts.append(f"string: {', '.join(disp_str)}")
            print(f"{CYAN}ℹ  Display-only columns (◇): {'; '.join(parts)}{RESET}")
    else:
        all_numeric = numeric_cols
        all_string  = string_cols

    all_feature_cols = numeric_cols + string_cols  # clustering features only
    X_raw = df_encoded[all_feature_cols].values.astype(float)

    # ── Scale ──
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

    # ── Stats on ALL cols (cluster + display merged) ──
    num_means, num_cvs, str_modes, str_uniques, sizes = compute_cluster_stats(
        df_encoded, df_original, labels, all_numeric, all_string, k
    )

    # ── Performance metrics ──
    inertia = km.inertia_
    if k > 1 and len(set(labels)) > 1:
        sil = silhouette_score(X, labels)
        db  = davies_bouldin_score(X, labels)
    else:
        sil = db = float("nan")

    # ── HTML report ──
    report: Optional[HtmlReport] = None
    if args.export:
        report = HtmlReport(
            filepath=args.filepath, k=k,
            n_rows=len(df_encoded),
            n_numeric=len(all_numeric),
            n_string=len(all_string),
            scaled=args.scale,
            numeric_cols=all_numeric,
            string_cols=all_string,
            display_cols=display_cols,
        )
        report.add_cluster_sizes(sizes, len(df_encoded))
        report.add_means_table(num_means, str_modes)
        report.add_cv_table(num_cvs, str_uniques)
        report.add_string_profiles(df_original, labels)
        report.add_performance(inertia, sil, db, km.n_iter_)

    # ── Terminal output ──
    print_banner(
        args.filepath, k,
        n_rows=len(df_encoded),
        n_numeric=len(all_numeric),
        n_string=len(all_string),
        scaled=args.scale,
    )

    feature_summary = []
    if numeric_cols:
        feature_summary.append(f"{BOLD}◆ Cluster numeric:{RESET} {', '.join(numeric_cols)}")
    if string_cols:
        feature_summary.append(f"{BOLD}{MAGENTA}◆ Cluster string:{RESET} {', '.join(string_cols)}")
    if display_cols:
        disp_num_list = [c for c in all_numeric if c in display_cols]
        disp_str_list = [c for c in all_string  if c in display_cols]
        if disp_num_list:
            feature_summary.append(f"{DIM}◇ Display numeric:{RESET} {', '.join(disp_num_list)}")
        if disp_str_list:
            feature_summary.append(f"{DIM}◇ Display string:{RESET} {', '.join(disp_str_list)}")
    for line in feature_summary:
        print(f"  {line}")
    if args.scale and string_cols:
        print(f"  {DIM}⚠  Scaling applied to label-encoded string columns too.{RESET}")
    print(f"  {DIM}(◆ used for clustering  ◇ display only — not used for clustering){RESET}")

    print_cluster_sizes(sizes, k, len(df_encoded))
    print_means_table(num_means, str_modes, all_numeric, all_string, k, display_cols)
    print_cv_table(num_cvs, str_uniques, all_numeric, all_string, k, display_cols)
    print_string_profiles(df_original, labels, all_string, k, display_cols=display_cols)
    print_performance(km, X, labels, k, args.scale, inertia, sil, db)

    # ── Write HTML if requested ──
    if report is not None:
        export_html(report, args.export)


if __name__ == "__main__":
    main()
