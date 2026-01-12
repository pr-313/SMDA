"""
Simple Factor Analysis template:
- Reads messy-ish CSV
- Keeps numeric columns
- Drops rows/cols with too many missing values
- Imputes remaining missing values
- Standardizes
- Fits FactorAnalysis
- Optional varimax rotation
- Pretty-prints top loadings
- Saves a checkpoint after each stage (so you can resume)

Deps: pandas, numpy, matplotlib, scikit-learn, joblib (optional but common)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis

try:
    import joblib
except ImportError:
    joblib = None
    import pickle


# -------------------------
# Minimal checkpoint helpers
# -------------------------

CKPT_PATH = "fa_ckpt.joblib"  # change if you want

def save_ckpt(obj, path=CKPT_PATH):
    if joblib and path.endswith(".joblib"):
        joblib.dump(obj, path)
    else:
        with open(path.replace(".joblib", ".pkl"), "wb") as f:
            pickle.dump(obj, f)

def load_ckpt(path=CKPT_PATH):
    if joblib and os.path.exists(path) and path.endswith(".joblib"):
        return joblib.load(path)
    alt = path.replace(".joblib", ".pkl")
    if os.path.exists(alt):
        with open(alt, "rb") as f:
            return pickle.load(f)
    return None


# -------------------------
# Simple varimax rotation
# -------------------------

def varimax(loadings, gamma=1.0, q=20, tol=1e-6):
    """loadings: (n_features, n_factors) -> rotated loadings"""
    p, k = loadings.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(q):
        d_old = d
        L = loadings @ R
        u, s, vh = np.linalg.svd(
            loadings.T @ (L**3 - (gamma / p) * L @ np.diag(np.diag(L.T @ L)))
        )
        R = u @ vh
        d = s.sum()
        if d_old and (d - d_old) < tol:
            break
    return loadings @ R


# -------------------------
# Main
# -------------------------

def run_fa(
    csv_path,
    n_factors=5,
    rotation="varimax",   # "varimax" or "none"
    max_missing_col=0.35, # drop columns missing > 35%
    max_missing_row=0.35, # drop rows missing > 35%
    out_dir="fa_out",
):
    os.makedirs(out_dir, exist_ok=True)

    # If resume exists, use it
    state = load_ckpt() or {"step": "start"}

    # 1) Read CSV (forgiving)
    if state["step"] == "start":
        df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip", skip_blank_lines=True)
        # Drop useless "Unnamed" cols often created by Excel exports
        df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed:\s*\d+$", na=False)]
        state = {"step": "loaded", "df": df}
        save_ckpt(state)
        print("Loaded:", df.shape)

    df = state["df"]

    # 2) Keep numeric only + missing cleanup + impute
    if state["step"] == "loaded":
        # Try converting obvious numeric strings to numbers
        df2 = df.copy()
        for c in df2.columns:
            if df2[c].dtype == "object":
                # strip commas/currency/percent; convert if possible
                s = df2[c].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
                s = s.str.replace(",", "", regex=False)
                s = s.str.replace(r"[$₹€£%]", "", regex=True)
                df2[c] = pd.to_numeric(s, errors="ignore")

        num = df2.select_dtypes(include=[np.number]).copy()

        # drop all-null rows/cols
        num = num.dropna(axis=0, how="all").dropna(axis=1, how="all")

        # drop too-missing cols/rows
        num = num.loc[:, num.isna().mean() <= max_missing_col]
        num = num.loc[num.isna().mean(axis=1) <= max_missing_row]

        if num.shape[1] < 2:
            raise ValueError("Not enough numeric columns after cleanup.")
        if num.shape[0] < 10:
            raise ValueError("Not enough rows after cleanup.")

        # simple impute with median
        num = num.fillna(num.median(numeric_only=True))

        state = {"step": "cleaned", "num": num}
        save_ckpt(state)
        print("Cleaned numeric:", num.shape)

        num.to_csv(os.path.join(out_dir, "clean_numeric.csv"), index=False)

    num = state["num"]

    # 3) Standardize
    if state["step"] == "cleaned":
        scaler = StandardScaler()
        X = scaler.fit_transform(num.values)
        state = {"step": "scaled", "num_cols": num.columns.tolist(), "scaler": scaler, "X": X}
        save_ckpt(state)
        print("Scaled X:", X.shape)

    cols = state["num_cols"]
    scaler = state["scaler"]
    X = state["X"]

    # 4) Fit FactorAnalysis
    if state["step"] == "scaled":
        fa = FactorAnalysis(n_components=n_factors, random_state=0)
        fa.fit(X)

        # sklearn components_: (n_factors, n_features) -> transpose to (n_features, n_factors)
        loadings = fa.components_.T

        state = {"step": "fit", "num_cols": cols, "scaler": scaler, "X": X, "fa": fa, "loadings": loadings}
        save_ckpt(state)
        print("Fit FA with factors =", n_factors)

    fa = state["fa"]
    loadings = state["loadings"]

    # 5) Rotate + pretty output
    if state["step"] == "fit":
        rot = varimax(loadings) if rotation.lower() == "varimax" else loadings

        load_df = pd.DataFrame(rot, index=cols, columns=[f"Factor_{i+1}" for i in range(rot.shape[1])])
        load_df.to_csv(os.path.join(out_dir, "loadings_rotated.csv"))

        # Pretty print: top 8 per factor
        for f in load_df.columns:
            s = load_df[f].sort_values(key=lambda x: np.abs(x), ascending=False).head(8)
            print(f"\n{f} top loadings:\n{s.to_string()}")

        # Quick heatmap (matplotlib only)
        plt.figure(figsize=(8, max(4, 0.2 * len(cols))))
        plt.imshow(load_df.values, aspect="auto", interpolation="nearest")
        plt.colorbar(label="Loading")
        plt.xticks(range(load_df.shape[1]), load_df.columns)
        plt.yticks(range(load_df.shape[0]), load_df.index, fontsize=7)
        plt.title("Rotated Factor Loadings")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "loadings_heatmap.png"), dpi=160)
        plt.close()

        state = {"step": "done"}
        save_ckpt(state)
        print("\nDone. Outputs in:", out_dir)
        print("Checkpoint:", CKPT_PATH)


if __name__ == "__main__":
    run_fa(
        csv_path="DBs/database.csv",
        n_factors=5,         # change this
        rotation="varimax",  # or "none"
        out_dir="fa_out"
    )

