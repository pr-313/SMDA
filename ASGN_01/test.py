"""
Factor Analysis Template (CSV -> cleanup -> FA -> pretty print -> plots -> checkpoints)

Dependencies (well-known):
- pandas, numpy, matplotlib
- scikit-learn (FactorAnalysis, StandardScaler)
- joblib (recommended for checkpointing; comes with sklearn often)

Notes:
- Handles "gaps" / messy CSVs via on_bad_lines='skip' and cleanup steps.
- Saves checkpoints after each major step so you can resume progress.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis, PCA

try:
    import joblib
except ImportError:
    joblib = None
    import pickle


# -----------------------------
# Config
# -----------------------------

@dataclass
class Config:
    csv_path: str = "data.csv"
    index_col: Optional[str] = None  # or name of id column
    encoding: Optional[str] = None   # set if you hit encoding issues, e.g. "utf-8", "latin-1"
    delimiter: Optional[str] = None  # leave None for auto; set to "," or "\t" if needed

    # Cleaning
    drop_unnamed_cols: bool = True
    strip_colnames: bool = True
    dedupe_rows: bool = True
    drop_all_null_rows: bool = True
    drop_all_null_cols: bool = True

    # Missing values handling
    max_missing_col_frac: float = 0.35   # drop columns missing > 35%
    max_missing_row_frac: float = 0.35   # drop rows missing > 35%
    impute_strategy: str = "median"      # "median" or "mean"

    # Type handling
    coerce_numeric: bool = True          # try converting object cols to numeric if they look numeric
    non_numeric_policy: str = "drop"     # "drop" or "onehot" (onehot can explode dims)

    # Analysis
    max_factors_to_test: int = 12
    n_factors: Optional[int] = None      # if None, choose based on plots/heuristic
    rotation: str = "varimax"            # "varimax" or "none"
    top_loadings_per_factor: int = 10
    random_state: int = 42

    # Files
    out_dir: str = "fa_output"
    checkpoint_path: str = "fa_output/checkpoint.joblib"


# -----------------------------
# Checkpointing
# -----------------------------

def save_checkpoint(state: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if joblib is not None and path.endswith(".joblib"):
        joblib.dump(state, path)
    else:
        with open(path.replace(".joblib", ".pkl"), "wb") as f:
            pickle.dump(state, f)


def load_checkpoint(path: str) -> Optional[Dict]:
    if os.path.exists(path) and joblib is not None and path.endswith(".joblib"):
        return joblib.load(path)
    alt = path.replace(".joblib", ".pkl")
    if os.path.exists(alt):
        with open(alt, "rb") as f:
            return pickle.load(f)
    return None


# -----------------------------
# CSV Loading + Cleanup
# -----------------------------

def read_messy_csv(cfg: Config) -> pd.DataFrame:
    """
    Robust-ish CSV read:
    - skips blank lines
    - skips malformed lines instead of dying
    """
    read_kwargs = dict(
        engine="python",            # more forgiving
        skip_blank_lines=True,
        on_bad_lines="skip",
    )
    if cfg.encoding:
        read_kwargs["encoding"] = cfg.encoding
    if cfg.delimiter:
        read_kwargs["sep"] = cfg.delimiter

    df = pd.read_csv(cfg.csv_path, **read_kwargs)

    # Optional: set index column
    if cfg.index_col and cfg.index_col in df.columns:
        df = df.set_index(cfg.index_col)

    return df


def normalize_column_names(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    cols = df.columns.astype(str)

    if cfg.strip_colnames:
        cols = cols.str.strip()

    # Normalize whitespace and punctuation a bit (optional)
    cols = cols.str.replace(r"\s+", " ", regex=True)
    df.columns = cols
    return df


def drop_junk_columns(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if cfg.drop_unnamed_cols:
        df = df.loc[:, ~df.columns.str.match(r"^Unnamed:\s*\d+$", na=False)]
    return df


def coerce_numeric_columns(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Attempts to convert object columns to numeric if they look numeric.
    Also strips commas, currency symbols, and spaces.
    """
    if not cfg.coerce_numeric:
        return df

    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            continue

        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            s = out[col].astype(str).str.strip()

            # Treat empty / "nan" / "None" as missing
            s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NA": np.nan, "N/A": np.nan})

            # Remove common formatting: commas, currency, percent signs
            s2 = s.str.replace(",", "", regex=False)
            s2 = s2.str.replace(r"[$₹€£%]", "", regex=True)
            s2 = s2.str.replace(r"\s+", "", regex=True)

            # If most values look numeric, convert
            numeric = pd.to_numeric(s2, errors="coerce")
            non_null = s2.notna().sum()
            if non_null == 0:
                continue

            conversion_rate = numeric.notna().sum() / non_null
            if conversion_rate >= 0.85:
                out[col] = numeric

    return out


def handle_missingness(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()

    if cfg.drop_all_null_cols:
        out = out.dropna(axis=1, how="all")

    if cfg.drop_all_null_rows:
        out = out.dropna(axis=0, how="all")

    # Drop columns with too much missing
    col_missing = out.isna().mean()
    out = out.loc[:, col_missing <= cfg.max_missing_col_frac]

    # Drop rows with too much missing
    row_missing = out.isna().mean(axis=1)
    out = out.loc[row_missing <= cfg.max_missing_row_frac]

    return out


def handle_non_numeric(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    FactorAnalysis needs numeric inputs.
    Policy:
      - "drop": keep only numeric columns
      - "onehot": one-hot encode non-numeric columns (can blow up dimensions)
    """
    if cfg.non_numeric_policy == "drop":
        return df.select_dtypes(include=[np.number]).copy()

    if cfg.non_numeric_policy == "onehot":
        # onehot encode object/categorical columns
        return pd.get_dummies(df, drop_first=False, dummy_na=True)

    raise ValueError(f"Unknown non_numeric_policy: {cfg.non_numeric_policy}")


def impute_missing(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    if cfg.impute_strategy == "median":
        fill = out.median(numeric_only=True)
    elif cfg.impute_strategy == "mean":
        fill = out.mean(numeric_only=True)
    else:
        raise ValueError("impute_strategy must be 'median' or 'mean'")
    out = out.fillna(fill)
    return out


def dedupe(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if cfg.dedupe_rows:
        # If index is meaningful, keep it; otherwise dedupe by all columns
        if df.index.is_unique:
            return df
        return df[~df.index.duplicated(keep="first")]
    return df


# -----------------------------
# Rotation (Varimax)
# -----------------------------

def varimax(Phi: np.ndarray, gamma: float = 1.0, q: int = 20, tol: float = 1e-6) -> np.ndarray:
    """
    Classic varimax rotation.
    Input: loadings matrix (n_features, n_factors)
    Output: rotated loadings matrix
    """
    p, k = Phi.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(q):
        d_old = d
        Lambda = Phi @ R
        u, s, vh = np.linalg.svd(
            Phi.T @ (Lambda**3 - (gamma / p) * Lambda @ np.diag(np.diag(Lambda.T @ Lambda)))
        )
        R = u @ vh
        d = s.sum()
        if d_old != 0 and (d - d_old) < tol:
            break
    return Phi @ R


# -----------------------------
# Pretty printing results
# -----------------------------

def loadings_to_df(loadings: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    cols = [f"Factor_{i+1}" for i in range(loadings.shape[1])]
    return pd.DataFrame(loadings, index=feature_names, columns=cols)


def pretty_print_top_loadings(loadings_df: pd.DataFrame, top_n: int = 10) -> None:
    """
    Prints top +/- loadings per factor in a readable way.
    """
    for factor in loadings_df.columns:
        s = loadings_df[factor].sort_values(key=lambda x: np.abs(x), ascending=False)
        print(f"\n=== {factor}: Top {top_n} loadings (by absolute value) ===")
        print(s.head(top_n).to_string())


# -----------------------------
# Model + Plots
# -----------------------------

def plot_scree_pca(X: np.ndarray, out_path: str) -> None:
    """
    PCA eigenvalue scree plot as a quick heuristic for factor count.
    """
    pca = PCA(n_components=min(X.shape[1], 50), random_state=0)
    pca.fit(X)
    eigenvalues = pca.explained_variance_

    plt.figure()
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker="o")
    plt.xlabel("Component #")
    plt.ylabel("Eigenvalue (PCA)")
    plt.title("Scree Plot (PCA eigenvalues) - heuristic for # factors")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_reconstruction_curve(X: np.ndarray, max_factors: int, out_path: str, random_state: int = 0) -> None:
    """
    Rough heuristic: FA reconstruction error vs number of factors.
    (Not perfect, but useful for sanity checking.)
    """
    errs = []
    ks = list(range(1, max_factors + 1))

    for k in ks:
        fa = FactorAnalysis(n_components=k, random_state=random_state)
        fa.fit(X)
        # Reconstruct: X_hat = Z W^T + mean
        Z = fa.transform(X)
        X_hat = Z @ fa.components_ + fa.mean_
        err = np.mean((X - X_hat) ** 2)
        errs.append(err)

    plt.figure()
    plt.plot(ks, errs, marker="o")
    plt.xlabel("# Factors")
    plt.ylabel("Mean Squared Reconstruction Error")
    plt.title("FA Reconstruction Error vs # Factors (heuristic)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_loadings_heatmap(loadings_df: pd.DataFrame, out_path: str) -> None:
    """
    Basic heatmap using matplotlib only (no seaborn).
    """
    data = loadings_df.values
    plt.figure(figsize=(max(6, 1 + 0.6 * data.shape[1]), max(6, 0.18 * data.shape[0])))
    plt.imshow(data, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Loading")
    plt.yticks(range(len(loadings_df.index)), loadings_df.index, fontsize=7)
    plt.xticks(range(len(loadings_df.columns)), loadings_df.columns)
    plt.title("Factor Loadings Heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# -----------------------------
# Pipeline
# -----------------------------

def run_factor_analysis(cfg: Config) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Try resume
    state = load_checkpoint(cfg.checkpoint_path) or {"cfg": asdict(cfg), "step": "start"}
    print(f"Checkpoint step: {state.get('step')}")

    # 1) Load
    if state.get("step") == "start":
        df_raw = read_messy_csv(cfg)
        df_raw = normalize_column_names(df_raw, cfg)
        df_raw = drop_junk_columns(df_raw, cfg)
        df_raw = dedupe(df_raw, cfg)

        state.update({
            "step": "loaded",
            "df_raw_shape": df_raw.shape,
            "df_raw_head": df_raw.head(3),
        })
        save_checkpoint(state, cfg.checkpoint_path)
        print("Loaded CSV:", df_raw.shape)

    # Re-load df_raw if needed (don’t store entire df in checkpoint by default)
    df_raw = read_messy_csv(cfg)
    df_raw = normalize_column_names(df_raw, cfg)
    df_raw = drop_junk_columns(df_raw, cfg)
    df_raw = dedupe(df_raw, cfg)

    # 2) Coerce numeric + missingness + numeric selection + impute
    if state.get("step") in ["loaded", "start"]:
        df = coerce_numeric_columns(df_raw, cfg)
        df = handle_missingness(df, cfg)
        df = handle_non_numeric(df, cfg)
        df = impute_missing(df, cfg)

        # sanity checks
        if df.shape[1] < 2:
            raise ValueError("Not enough numeric columns after cleanup to run factor analysis.")
        if df.shape[0] < 10:
            raise ValueError("Not enough rows after cleanup to run factor analysis.")

        clean_path = os.path.join(cfg.out_dir, "clean_numeric_imputed.csv")
        df.to_csv(clean_path, index=True)

        state.update({
            "step": "cleaned",
            "df_clean_shape": df.shape,
            "clean_path": clean_path,
            "feature_names": df.columns.tolist(),
        })
        save_checkpoint(state, cfg.checkpoint_path)
        print("Cleaned data:", df.shape)

    # Load cleaned df
    df = pd.read_csv(state["clean_path"], engine="python")
    # If your CSV saved an index column, it may appear as "Unnamed: 0"
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed:\s*\d+$", na=False)]

    feature_names = df.columns.tolist()
    X = df.values.astype(float)

    # 3) Scale
    if state.get("step") == "cleaned":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        state.update({
            "step": "scaled",
            "scaler": scaler,
            "X_shape": Xs.shape,
        })
        save_checkpoint(state, cfg.checkpoint_path)
        print("Scaled:", Xs.shape)

    scaler = state["scaler"]
    Xs = scaler.transform(X)

    # 4) Heuristic plots for choosing # factors
    if state.get("step") == "scaled":
        scree_path = os.path.join(cfg.out_dir, "scree_pca.png")
        recon_path = os.path.join(cfg.out_dir, "fa_reconstruction_curve.png")

        plot_scree_pca(Xs, scree_path)
        plot_reconstruction_curve(Xs, cfg.max_factors_to_test, recon_path, random_state=cfg.random_state)

        state.update({
            "step": "plotted_heuristics",
            "scree_path": scree_path,
            "recon_path": recon_path,
        })
        save_checkpoint(state, cfg.checkpoint_path)
        print("Wrote plots:", scree_path, recon_path)

    # 5) Fit FactorAnalysis
    if state.get("step") == "plotted_heuristics":
        k = cfg.n_factors
        if k is None:
            # Simple default heuristic:
            # pick min(6, max_factors_to_test) unless you set cfg.n_factors manually.
            # (You SHOULD override this after looking at the scree/recon plots.)
            k = min(6, cfg.max_factors_to_test)

        fa = FactorAnalysis(n_components=k, random_state=cfg.random_state)
        fa.fit(Xs)

        # sklearn: components_ is (n_components, n_features)
        # loadings usually shown as (n_features, n_components)
        loadings = fa.components_.T

        state.update({
            "step": "fit",
            "n_factors": k,
            "fa_model": fa,
            "loadings": loadings,
        })
        save_checkpoint(state, cfg.checkpoint_path)
        print(f"Fit FactorAnalysis with n_factors={k}")

    fa = state["fa_model"]
    loadings = state["loadings"]
    k = state["n_factors"]

    # 6) Rotate (optional)
    if state.get("step") == "fit":
        if cfg.rotation.lower() == "varimax":
            rot = varimax(loadings)
        else:
            rot = loadings

        state.update({
            "step": "rotated",
            "rot_loadings": rot,
        })
        save_checkpoint(state, cfg.checkpoint_path)
        print("Rotation:", cfg.rotation)

    rot = state["rot_loadings"]

    # 7) Outputs: pretty print + files + plots
    if state.get("step") == "rotated":
        load_df = loadings_to_df(rot, feature_names)

        # Save tables
        loadings_csv = os.path.join(cfg.out_dir, "rotated_loadings.csv")
        load_df.to_csv(loadings_csv)

        # Pretty print
        pretty_print_top_loadings(load_df, top_n=cfg.top_loadings_per_factor)

        # Heatmap plot
        heatmap_path = os.path.join(cfg.out_dir, "rotated_loadings_heatmap.png")
        plot_loadings_heatmap(load_df, heatmap_path)

        state.update({
            "step": "done",
            "rotated_loadings_csv": loadings_csv,
            "heatmap_path": heatmap_path,
        })
        save_checkpoint(state, cfg.checkpoint_path)
        print("\nDone.")
        print("Saved:", loadings_csv)
        print("Saved:", heatmap_path)
        print("Checkpoint:", cfg.checkpoint_path)


# -----------------------------
# Entry point
# -----------------------------

if __name__ == "__main__":
    cfg = Config(
        csv_path="DBs/database.csv",
        # index_col="CustomerID",
        # delimiter=",",
        # encoding="utf-8",
        out_dir="fa_output",
        checkpoint_path="fa_output/checkpoint.joblib",
        n_factors=None,          # set after you look at plots
        rotation="varimax",      # or "none"
        non_numeric_policy="drop"  # or "onehot"
    )
    run_factor_analysis(cfg)

