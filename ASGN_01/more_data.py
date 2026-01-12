"""
Factor Analysis template (CSV -> cleanup -> FA -> outputs to a dedicated folder)
- Saves: cleaned data, unrotated loadings + communalities + variance tables,
         rotated (Varimax) loadings + communalities + variance tables,
         variance plots (PNG), plus a short run summary (TXT).

Well-known libs only:
  pip install pandas numpy scikit-learn matplotlib
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis


# -----------------------------
# Pretty print (console only)
# -----------------------------
def set_pretty_print() -> None:
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")


# -----------------------------
# Data cleanup
# -----------------------------
def load_and_clean_csv(
    path: str | Path,
    *,
    drop_non_numeric: bool = True,
    exclude_cols: list[str] | None = None,
    na_tokens: list[str] | None = None,
    min_nonnull_frac_col: float = 0.70,   # drop columns with >30% missing
    min_nonnull_frac_row: float = 0.70,   # drop rows with >30% missing
) -> pd.DataFrame:
    exclude_cols = exclude_cols or []
    na_tokens = na_tokens or ["", " ", "NA", "N/A", "na", "n/a", "null", "None", "-", "--", "?", "nan"]

    df = pd.read_csv(path, na_values=na_tokens, keep_default_na=True)

    # Drop explicitly excluded columns (IDs, names, etc.)
    keep_cols = [c for c in df.columns if c not in set(exclude_cols)]
    df = df[keep_cols].copy()

    # Coerce to numeric where possible (turns messy strings into NaN)
    if drop_non_numeric:
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Keep only numeric columns
        df = df.select_dtypes(include=[np.number]).copy()

    # Drop columns with too many missing
    col_nonnull_frac = df.notna().mean(axis=0)
    df = df.loc[:, col_nonnull_frac >= min_nonnull_frac_col].copy()

    # Drop rows with too many missing
    row_nonnull_frac = df.notna().mean(axis=1)
    df = df.loc[row_nonnull_frac >= min_nonnull_frac_row].copy()

    if df.shape[1] < 2:
        raise ValueError(
            "After cleanup, fewer than 2 usable numeric columns remained. "
            "Relax thresholds, adjust exclude_cols, or check your CSV."
        )

    return df


# -----------------------------
# Rotation (Varimax)
# -----------------------------
def varimax(Phi: np.ndarray, gamma: float = 1.0, q: int = 50, tol: float = 1e-6) -> np.ndarray:
    """
    Orthogonal Varimax rotation.
    Phi: (p x k) loading matrix
    Returns rotated loadings (p x k).
    """
    p, k = Phi.shape
    R = np.eye(k)
    d_old = 0.0

    for _ in range(q):
        Lambda = Phi @ R
        u, s, vh = np.linalg.svd(
            Phi.T @ (Lambda**3 - (gamma / p) * Lambda @ np.diag(np.diag(Lambda.T @ Lambda))),
            full_matrices=False
        )
        R = u @ vh
        d = s.sum()
        if d_old != 0 and (d - d_old) < tol:
            break
        d_old = d

    return Phi @ R


# -----------------------------
# Factor analysis core
# -----------------------------
def fit_fa_and_report(
    df: pd.DataFrame,
    n_factors: int,
    *,
    rotation: str | None,
    impute_strategy: str = "median",
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Fits FA on standardized data and returns:
      - loadings (p x k)
      - communalities (p,)
      - variance_table (k x 3): SS_Loadings, Proportion_Variance, Cumulative_Variance

    Variance explained uses standard FA reporting convention:
      SS_loadings_j = sum_i loading_ij^2
      PropVar_j     = SS_loadings_j / p
    """
    X = df.to_numpy(dtype=float)
    feature_names = df.columns.to_list()
    p = len(feature_names)

    # Impute missing values
    imputer = SimpleImputer(strategy=impute_strategy)
    X_imp = imputer.fit_transform(X)

    # Standardize (recommended unless already same scale)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_imp)

    fa = FactorAnalysis(n_components=n_factors, random_state=random_state)
    fa.fit(X_std)

    # sklearn.components_: (k x p) -> transpose to (p x k)
    loadings = fa.components_.T

    if rotation is not None:
        rot = rotation.lower()
        if rot == "varimax":
            loadings = varimax(loadings)
        else:
            raise ValueError("Unsupported rotation. Use None or 'varimax'.")

    communalities = (loadings**2).sum(axis=1)

    ss_loadings = (loadings**2).sum(axis=0)
    prop_var = ss_loadings / p
    cum_var = np.cumsum(prop_var)

    loading_cols = [f"Factor_{i+1}" for i in range(n_factors)]

    loadings_df = pd.DataFrame(loadings, index=feature_names, columns=loading_cols)
    comm_df = pd.DataFrame({"Communality": communalities}, index=feature_names)

    variance_df = pd.DataFrame(
        {
            "SS_Loadings": ss_loadings,
            "Proportion_Variance": prop_var,
            "Cumulative_Variance": cum_var,
        },
        index=loading_cols,
    )

    return {
        "loadings": loadings_df,
        "communalities": comm_df,
        "variance_table": variance_df,
    }


# -----------------------------
# Plotting (saved to files)
# -----------------------------
def save_variance_plots(variance_df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    x = np.arange(len(variance_df))
    prop = variance_df["Proportion_Variance"].to_numpy()
    cum = variance_df["Cumulative_Variance"].to_numpy()

    # Bar: proportion variance
    plt.figure()
    plt.bar(x, prop)
    plt.xticks(x, variance_df.index, rotation=0)
    plt.ylabel("Proportion of Variance")
    plt.title(f"Variance Explained ({prefix})")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_variance_explained.png", dpi=200)
    plt.close()

    # Line: cumulative variance
    plt.figure()
    plt.plot(x, cum, marker="o")
    plt.xticks(x, variance_df.index, rotation=0)
    plt.ylim(0, 1.05)
    plt.ylabel("Cumulative Variance")
    plt.title(f"Cumulative Variance Explained ({prefix})")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_cumulative_variance.png", dpi=200)
    plt.close()


# -----------------------------
# I/O helpers
# -----------------------------
def write_outputs(
    results: dict[str, pd.DataFrame],
    out_dir: Path,
    *,
    prefix: str,
    decimals: int = 4,
) -> None:
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    loadings_df = results["loadings"].round(decimals)
    comm_df = results["communalities"].round(decimals)
    variance_df = results["variance_table"].round(decimals)

    loadings_df.to_csv(out_dir / "tables" / f"{prefix}_loadings.csv")
    comm_df.to_csv(out_dir / "tables" / f"{prefix}_communalities.csv")
    variance_df.to_csv(out_dir / "tables" / f"{prefix}_variance_table.csv")

    save_variance_plots(variance_df, out_dir / "plots", prefix=prefix)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    set_pretty_print()

    # ---- USER SETTINGS ----
    CSV_PATH = "DBs/database.csv"  # <-- change
    OUTPUT_DIR = Path("fa_output_1")  # dedicated output folder
    N_FACTORS = 8  # <-- choose
    EXCLUDE_COLS = ["id", "customer_id", "name"]  # <-- adjust
    MIN_NONNULL_FRAC_COL = 0.70
    MIN_NONNULL_FRAC_ROW = 0.70
    IMPUTE_STRATEGY = "median"
    RANDOM_STATE = 42
    # -----------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load + clean
    df = load_and_clean_csv(
        CSV_PATH,
        exclude_cols=EXCLUDE_COLS,
        min_nonnull_frac_col=MIN_NONNULL_FRAC_COL,
        min_nonnull_frac_row=MIN_NONNULL_FRAC_ROW,
    )

    # Save cleaned data
    df.to_csv(OUTPUT_DIR / "cleaned_data.csv", index=False)

    # Summarize missingness (post-clean, pre-impute)
    missing_by_col = df.isna().mean().sort_values(ascending=False)
    missing_by_row = df.isna().mean(axis=1).describe()

    # Run unrotated
    res_unrot = fit_fa_and_report(
        df,
        n_factors=N_FACTORS,
        rotation=None,
        impute_strategy=IMPUTE_STRATEGY,
        random_state=RANDOM_STATE,
    )
    write_outputs(res_unrot, OUTPUT_DIR, prefix="unrotated")

    # Run rotated (Varimax)
    res_varimax = fit_fa_and_report(
        df,
        n_factors=N_FACTORS,
        rotation="varimax",
        impute_strategy=IMPUTE_STRATEGY,
        random_state=RANDOM_STATE,
    )
    write_outputs(res_varimax, OUTPUT_DIR, prefix="varimax")

    # Console prints (quick sanity)
    print("\nCleaned data shape:", df.shape)
    print("\n=== UNROTATED: Variance Table ===")
    print(res_unrot["variance_table"].round(4))
    print("\n=== VARIMAX: Variance Table ===")
    print(res_varimax["variance_table"].round(4))

    # Write a run summary
    summary_path = OUTPUT_DIR / "run_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("FA Run Summary\n")
        f.write("==============\n")
        f.write(f"CSV_PATH: {CSV_PATH}\n")
        f.write(f"OUTPUT_DIR: {OUTPUT_DIR.resolve()}\n")
        f.write(f"N_FACTORS: {N_FACTORS}\n")
        f.write(f"EXCLUDE_COLS: {EXCLUDE_COLS}\n")
        f.write(f"MIN_NONNULL_FRAC_COL: {MIN_NONNULL_FRAC_COL}\n")
        f.write(f"MIN_NONNULL_FRAC_ROW: {MIN_NONNULL_FRAC_ROW}\n")
        f.write(f"IMPUTE_STRATEGY: {IMPUTE_STRATEGY}\n")
        f.write(f"RANDOM_STATE: {RANDOM_STATE}\n")
        f.write("\nPost-clean missingness (fraction) by column:\n")
        f.write(missing_by_col.to_string())
        f.write("\n\nPost-clean missingness summary by row:\n")
        f.write(missing_by_row.to_string())
        f.write("\n")

    print(f"\nSaved outputs to: {OUTPUT_DIR.resolve()}")
    print("Files:")
    print(" - cleaned_data.csv")
    print(" - tables/unrotated_loadings.csv, unrotated_communalities.csv, unrotated_variance_table.csv")
    print(" - tables/varimax_loadings.csv, varimax_communalities.csv, varimax_variance_table.csv")
    print(" - plots/*.png")
    print(" - run_summary.txt")

