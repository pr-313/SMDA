#!/usr/bin/env python3
"""
factor_analysis.py — Factor Analysis tool for CSV inputs.

Usage:
    python factor_analysis.py data.csv
    python factor_analysis.py data.csv --factors 4
    python factor_analysis.py data.csv --cols col1 col2 col3
    python factor_analysis.py data.csv --factors 3 --rotation varimax
    python factor_analysis.py data.csv --auto-factors --min-eigen 1.0
    python factor_analysis.py data.csv --factors 3 --export report.html

Only numeric columns are used. String columns are ignored with a warning.

Outputs (all in terminal):
    - Eigenvalues table with scree bar chart
    - Cumulative variance explained table
    - Rotated factor loading matrix (Varimax or Oblimin)
    - Communalities table (how much variance each variable shares with factors)
    - Inter-variable correlation matrix

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
from scipy import linalg


# ─── ANSI colour helpers ───────────────────────────────────────────────────────

RESET    = "\033[0m"
BOLD     = "\033[1m"
DIM      = "\033[2m"
WHITE    = "\033[97m"
RED      = "\033[91m"
GREEN    = "\033[92m"
YELLOW   = "\033[93m"
CYAN     = "\033[96m"
MAGENTA  = "\033[95m"
BG_HEADER = "\033[48;5;17m"

FACTOR_COLORS = [
    "\033[92m",       # green
    "\033[93m",       # yellow
    "\033[96m",       # cyan
    "\033[95m",       # magenta
    "\033[91m",       # red
    "\033[94m",       # blue
    "\033[38;5;208m", # orange
    "\033[38;5;141m", # purple
]

HTML_FACTOR_PALETTE = [
    "#16a34a", "#ca8a04", "#0891b2", "#9333ea",
    "#dc2626", "#2563eb", "#ea580c", "#db2777",
]


def cfactor(f: int) -> str:
    return FACTOR_COLORS[f % len(FACTOR_COLORS)]


def _hf(f: int) -> str:
    return HTML_FACTOR_PALETTE[f % len(HTML_FACTOR_PALETTE)]


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


# ─── Core factor analysis ─────────────────────────────────────────────────────

def load_and_prepare(
    filepath: str,
    cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load CSV, keep only numeric columns, return df + col list."""
    try:
        df_raw = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"{RED}✗ File not found: {filepath}{RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"{RED}✗ Could not read CSV: {e}{RESET}")
        sys.exit(1)

    if cols:
        missing = [c for c in cols if c not in df_raw.columns]
        if missing:
            print(f"{RED}✗ Columns not found: {missing}{RESET}")
            print(f"  Available: {list(df_raw.columns)}")
            sys.exit(1)
        df = df_raw[cols].copy()
    else:
        df = df_raw.select_dtypes(include=[np.number]).copy()

    # Warn about dropped string columns
    string_cols = list(df_raw.select_dtypes(exclude=[np.number]).columns)
    if string_cols and not cols:
        print(f"{YELLOW}⚠  Ignoring non-numeric columns: {string_cols}{RESET}")

    if df.empty or df.shape[1] < 2:
        print(f"{RED}✗ Need at least 2 numeric columns for factor analysis.{RESET}")
        sys.exit(1)

    # Drop rows with all-NaN, fill remaining NaNs with column mean
    n_before = len(df)
    df.dropna(how="all", inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    n_after = len(df)
    if n_before != n_after:
        print(f"{YELLOW}⚠  Dropped {n_before - n_after} fully-null rows.{RESET}")

    if len(df) < df.shape[1]:
        print(f"{RED}✗ More variables than observations — factor analysis not reliable.{RESET}")
        sys.exit(1)

    return df, list(df.columns)


def compute_correlation_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return Pearson correlation matrix as numpy array."""
    return df.corr(method="pearson").values


def extract_factors_pca(
    corr: np.ndarray,
    n_factors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract initial factor loadings via PCA on the correlation matrix.

    Returns:
        eigenvalues  — shape (p,)   all eigenvalues, descending
        loadings     — shape (p, n_factors)  unrotated loadings
    """
    eigenvalues, eigenvectors = linalg.eigh(corr)

    # eigh returns ascending order — reverse to descending
    idx         = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors= eigenvectors[:, idx]

    # Clamp tiny negatives from floating-point noise
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Unrotated loadings = eigenvectors * sqrt(eigenvalues)
    loadings = eigenvectors[:, :n_factors] * np.sqrt(eigenvalues[:n_factors])
    return eigenvalues, loadings


def varimax_rotation(loadings: np.ndarray, max_iter: int = 1000,
                     tol: float = 1e-6) -> np.ndarray:
    """Apply Varimax (orthogonal) rotation to factor loadings.

    Maximises variance of squared loadings within each factor,
    encouraging each variable to load strongly on one factor only.

    Args:
        loadings: (p, k) unrotated loading matrix
        max_iter: maximum rotation iterations
        tol:      convergence tolerance

    Returns:
        rotated loading matrix of shape (p, k)
    """
    p, k = loadings.shape
    rotation = np.eye(k)

    for _ in range(max_iter):
        old_rotation = rotation.copy()
        for i in range(k):
            for j in range(i + 1, k):
                # Compute rotation angle for pair (i, j)
                x = loadings @ rotation
                u = x[:, i] ** 2 - x[:, j] ** 2
                v = 2 * x[:, i] * x[:, j]
                A = np.sum(u)
                B = np.sum(v)
                C = np.sum(u ** 2 - v ** 2)
                D = 2 * np.sum(u * v)
                num = D - 2 * A * B / p
                den = C - (A ** 2 - B ** 2) / p
                theta = 0.25 * np.arctan2(num, den)
                # Apply Givens rotation
                rot = np.eye(k)
                rot[i, i] =  np.cos(theta)
                rot[j, j] =  np.cos(theta)
                rot[i, j] = -np.sin(theta)
                rot[j, i] =  np.sin(theta)
                rotation = rotation @ rot

        if np.max(np.abs(rotation - old_rotation)) < tol:
            break

    return loadings @ rotation


def oblimin_rotation(loadings: np.ndarray, gamma: float = 0.0,
                     max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
    """Apply Oblimin (oblique) rotation to factor loadings.

    Allows factors to be correlated. gamma=0 is Direct Oblimin.

    Args:
        loadings: (p, k) unrotated loading matrix
        gamma:    obliqueness parameter (0 = oblimin, 1 = quartimin)
        max_iter: maximum iterations
        tol:      convergence tolerance

    Returns:
        rotated pattern matrix of shape (p, k)
    """
    p, k = loadings.shape
    L = loadings.copy()
    T = np.eye(k)   # transformation matrix

    for _ in range(max_iter):
        L_old = L.copy()
        L2    = L ** 2
        # Gradient of the oblimin criterion
        grad  = L * (L2 @ np.ones((k, k)) - gamma * np.ones((p, p)) @ L2) * 2
        # Update via gradient step + re-orthogonalisation
        T_new = T - 0.001 * loadings.T @ grad
        # QR decomposition to keep T well-conditioned
        Q, R  = np.linalg.qr(T_new)
        T     = Q
        L     = loadings @ T

        if np.max(np.abs(L - L_old)) < tol:
            break

    return L


def compute_communalities(loadings: np.ndarray) -> np.ndarray:
    """Communality = sum of squared loadings across factors for each variable."""
    return np.sum(loadings ** 2, axis=1)


def compute_variance_explained(
    eigenvalues: np.ndarray,
    loadings: np.ndarray,
    n_factors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-factor variance stats.

    Returns:
        var_each   — variance explained by each retained factor (via loading SS)
        pct_each   — % of total variance per factor
        pct_cum    — cumulative % of total variance
    """
    p = loadings.shape[0]
    # Variance explained = sum of squared loadings per factor (post-rotation)
    var_each = np.sum(loadings ** 2, axis=0)
    pct_each = var_each / p * 100
    pct_cum  = np.cumsum(pct_each)
    return var_each, pct_each, pct_cum


def suggest_n_factors(eigenvalues: np.ndarray, min_eigen: float = 1.0) -> int:
    """Return number of factors with eigenvalue >= min_eigen (Kaiser criterion)."""
    n = int(np.sum(eigenvalues >= min_eigen))
    return max(2, min(n, len(eigenvalues) - 1))


# ─── Terminal output ──────────────────────────────────────────────────────────

def print_banner(
    filepath: str,
    n_vars: int,
    n_obs: int,
    n_factors: int,
    rotation: str,
) -> None:
    print("\n" + hr("═"))
    print(f"  {BOLD}{CYAN}FACTOR ANALYSIS{RESET}  {DIM}─{RESET}  {filepath}")
    print(hr("═"))
    print(
        f"  Variables: {BOLD}{n_vars}{RESET}   "
        f"Observations: {BOLD}{n_obs}{RESET}   "
        f"Factors: {BOLD}{n_factors}{RESET}   "
        f"Rotation: {BOLD}{rotation.capitalize()}{RESET}"
    )
    print(hr())


def print_eigenvalues(
    eigenvalues: np.ndarray,
    n_factors: int,
) -> None:
    print(section_header("EIGENVALUES  (Kaiser criterion: retain factors ≥ 1.0)"))

    max_e   = max(eigenvalues) if len(eigenvalues) > 0 else 1.0
    bar_max = 40

    print(f"\n  {'Factor':<10} {'Eigenvalue':>12}  {'Scree':42}  {'Retain?'}")
    print(f"  {DIM}{'─' * 74}{RESET}")

    for i, ev in enumerate(eigenvalues):
        bar_len = int(ev / max_e * bar_max) if max_e > 0 else 0
        bar     = "█" * bar_len
        retained = i < n_factors
        color   = cfactor(i) if retained else DIM
        flag    = f"{GREEN}✓ retained{RESET}" if retained else f"{DIM}✗ dropped{RESET}"
        marker  = f"{DIM}─── 1.0{RESET}" if abs(ev - 1.0) < 0.15 else ""
        print(
            f"  {color}{BOLD}Factor {i+1:<3}{RESET} "
            f"{color}{ev:>12.4f}{RESET}  "
            f"{color}{bar:<{bar_max}}{RESET}  "
            f"{flag}"
        )

    print()


def print_variance_table(
    var_each: np.ndarray,
    pct_each: np.ndarray,
    pct_cum: np.ndarray,
    n_factors: int,
) -> None:
    print(section_header("VARIANCE EXPLAINED"))

    col_w = 12
    print(f"\n  {'Factor':<12} {'SS Loadings':>12}  {'% Variance':>12}  {'Cumulative %':>14}  {'':30}")
    print(f"  {DIM}{'─' * 76}{RESET}")

    bar_max = 30
    for i in range(n_factors):
        bar_len  = int(pct_each[i] / 100 * bar_max)
        cum_bar  = int(pct_cum[i] / 100 * bar_max)
        color    = cfactor(i)
        cum_color= GREEN if pct_cum[i] >= 60 else (YELLOW if pct_cum[i] >= 40 else DIM)
        bar      = "█" * bar_len + "░" * (bar_max - bar_len)
        print(
            f"  {color}{BOLD}Factor {i+1:<5}{RESET} "
            f"{color}{var_each[i]:>12.4f}{RESET}  "
            f"{color}{pct_each[i]:>11.2f}%{RESET}  "
            f"{cum_color}{pct_cum[i]:>13.2f}%{RESET}  "
            f"{color}{bar}{RESET}"
        )

    total_var = sum(pct_each)
    print(f"  {DIM}{'─' * 76}{RESET}")
    print(f"  {'Total':<12} {sum(var_each):>12.4f}  {total_var:>11.2f}%")
    print()


def print_loading_matrix(
    loadings: np.ndarray,
    variables: list[str],
    n_factors: int,
    rotation: str,
    threshold: float = 0.3,
) -> None:
    title = f"ROTATED FACTOR MATRIX  ({rotation.upper()})  |loadings| ≥ {threshold} highlighted"
    print(section_header(title))

    col_w   = max(max(len(v) for v in variables), 14) + 2
    num_w   = 10

    # Header
    header  = [f"{'Variable':{col_w}}"]
    for f in range(n_factors):
        header.append(f"{cfactor(f)}{BOLD}{'Factor '+str(f+1):^{num_w}}{RESET}")
    header.append(f"  {'Communality':>12}")
    print("\n  " + "  ".join(header))
    print(f"  {DIM}{'─' * (col_w + (num_w + 2) * n_factors + 16)}{RESET}")

    communalities = compute_communalities(loadings)

    for i, var in enumerate(variables):
        row_parts = [f"{BOLD}{var:{col_w}}{RESET}"]
        for f in range(n_factors):
            v = loadings[i, f]
            abs_v = abs(v)
            if abs_v >= threshold:
                color  = cfactor(f)
                weight = BOLD if abs_v >= 0.6 else ""
                row_parts.append(f"{color}{weight}{v:>{num_w}.4f}{RESET}")
            else:
                row_parts.append(f"{DIM}{v:>{num_w}.4f}{RESET}")
        comm = communalities[i]
        comm_color = GREEN if comm >= 0.6 else (YELLOW if comm >= 0.4 else RED)
        row_parts.append(f"  {comm_color}{comm:>12.4f}{RESET}")
        print("  " + "  ".join(row_parts))

    print(f"\n  {DIM}Loading guide: {RESET}"
          f"{BOLD}bold ≥ 0.6{RESET}  "
          f"{DIM}dim < {threshold} (suppressed){RESET}  "
          f"  Communality: {GREEN}≥ 0.6 good{RESET}  {YELLOW}0.4–0.6 moderate{RESET}  {RED}< 0.4 weak{RESET}")
    print()


def print_communalities(
    communalities: np.ndarray,
    variables: list[str],
) -> None:
    print(section_header("COMMUNALITIES  (proportion of variance explained by the factors)"))

    col_w   = max(max(len(v) for v in variables), 14) + 2
    bar_max = 40

    print(f"\n  {'Variable':{col_w}}  {'h²':>8}  {'':42}  Rating")
    print(f"  {DIM}{'─' * (col_w + 60)}{RESET}")

    for var, h2 in zip(variables, communalities):
        bar_len = int(h2 * bar_max)
        bar     = "█" * bar_len + "░" * (bar_max - bar_len)
        if h2 >= 0.6:
            color, rating = GREEN,  "good"
        elif h2 >= 0.4:
            color, rating = YELLOW, "moderate"
        else:
            color, rating = RED,    "weak"
        print(
            f"  {BOLD}{var:{col_w}}{RESET}  "
            f"{color}{h2:>8.4f}{RESET}  "
            f"{color}{bar}{RESET}  "
            f"{color}{rating}{RESET}"
        )

    mean_h2 = np.mean(communalities)
    print(f"  {DIM}{'─' * (col_w + 60)}{RESET}")
    print(f"  {'Mean h²':{col_w}}  {mean_h2:>8.4f}")
    print()


def print_correlation_matrix(
    corr: np.ndarray,
    variables: list[str],
) -> None:
    print(section_header("CORRELATION MATRIX"))

    n     = len(variables)
    # Dynamic column width based on variable name length
    cw    = max(max(len(v) for v in variables), 6) + 1
    nw    = max(cw, 7)

    # Header row
    print()
    header = f"  {'':{cw}}"
    for v in variables:
        header += f"  {v[:nw]:^{nw}}"
    print(header)
    print(f"  {DIM}{'─' * (cw + (nw + 2) * n + 2)}{RESET}")

    for i, row_var in enumerate(variables):
        row = f"  {BOLD}{row_var:{cw}}{RESET}"
        for j in range(n):
            v = corr[i, j]
            if i == j:
                row += f"  {DIM}{'1.000':^{nw}}{RESET}"
            else:
                abs_v = abs(v)
                if abs_v >= 0.7:
                    color = GREEN if v > 0 else RED
                    weight = BOLD
                elif abs_v >= 0.4:
                    color = YELLOW if v > 0 else MAGENTA
                    weight = ""
                else:
                    color, weight = DIM, ""
                row += f"  {color}{weight}{v:^{nw}.3f}{RESET}"
        print(row)

    print(f"\n  {DIM}Correlation guide: "
          f"{GREEN}{BOLD}≥ 0.7 strong positive{RESET}  "
          f"{DIM}{YELLOW}0.4–0.7 moderate positive{RESET}  "
          f"{DIM}{RED}{BOLD}≤ -0.7 strong negative{RESET}  "
          f"{DIM}{MAGENTA}−0.4–−0.7 moderate negative{RESET}")
    print()


# ─── HTML export ──────────────────────────────────────────────────────────────

class HtmlReport:
    """Accumulates report sections as HTML strings, then renders to a full page."""

    def __init__(self, filepath: str, n_vars: int, n_obs: int,
                 n_factors: int, rotation: str, variables: list[str]) -> None:
        self.filepath  = filepath
        self.n_vars    = n_vars
        self.n_obs     = n_obs
        self.n_factors = n_factors
        self.rotation  = rotation
        self.variables = variables
        self._sections: list[str] = []

    def _e(self, s) -> str:
        return html_lib.escape(str(s))

    # ── Eigenvalues ────────────────────────────────────────────────────────────
    def add_eigenvalues(self, eigenvalues: np.ndarray) -> None:
        max_e = max(eigenvalues) if len(eigenvalues) else 1.0
        rows  = ""
        for i, ev in enumerate(eigenvalues):
            retained = i < self.n_factors
            color    = _hf(i) if retained else "#94a3b8"
            pct      = ev / max_e * 100
            badge    = f'<span class="rating-badge rating-good">retained</span>' if retained \
                       else f'<span class="rating-badge rating-weak">dropped</span>'
            rows += f"""
            <tr>
              <td><span class="factor-badge" style="background:{color}20;color:{color};border:1px solid {color}60">
                Factor {i+1}</span></td>
              <td class="mono">{ev:.4f}</td>
              <td>
                <div class="bar-wrap">
                  <div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div>
                </div>
              </td>
              <td>{badge}</td>
            </tr>"""

        self._sections.append(f"""
        <section>
          <h2>Eigenvalues <span class="subtitle">Kaiser criterion: retain factors ≥ 1.0</span></h2>
          <table class="simple-table">
            <thead><tr><th>Factor</th><th>Eigenvalue</th><th>Relative Size</th><th>Status</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </section>""")

    # ── Variance explained ─────────────────────────────────────────────────────
    def add_variance_table(self, var_each: np.ndarray,
                           pct_each: np.ndarray, pct_cum: np.ndarray) -> None:
        rows = ""
        for i in range(self.n_factors):
            color     = _hf(i)
            cum_color = "#16a34a" if pct_cum[i] >= 60 else ("#ca8a04" if pct_cum[i] >= 40 else "#64748b")
            rows += f"""
            <tr>
              <td><span class="factor-badge" style="background:{color}20;color:{color};border:1px solid {color}60">
                Factor {i+1}</span></td>
              <td class="mono">{var_each[i]:.4f}</td>
              <td class="mono">{pct_each[i]:.2f}%
                <div class="bar-wrap" style="width:120px;margin-top:4px">
                  <div class="bar-fill" style="width:{pct_each[i]:.1f}%;background:{color}"></div>
                </div>
              </td>
              <td class="mono" style="color:{cum_color};font-weight:600">{pct_cum[i]:.2f}%
                <div class="bar-wrap" style="width:120px;margin-top:4px">
                  <div class="bar-fill" style="width:{pct_cum[i]:.1f}%;background:{cum_color}"></div>
                </div>
              </td>
            </tr>"""

        total_var = sum(pct_each)
        rows += f"""
            <tr style="font-weight:700;border-top:2px solid #e2e8f0">
              <td>Total</td>
              <td class="mono">{sum(var_each):.4f}</td>
              <td class="mono">{total_var:.2f}%</td>
              <td></td>
            </tr>"""

        self._sections.append(f"""
        <section>
          <h2>Variance Explained</h2>
          <table class="simple-table">
            <thead><tr><th>Factor</th><th>SS Loadings</th><th>% Variance</th><th>Cumulative %</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </section>""")

    # ── Loading matrix ─────────────────────────────────────────────────────────
    def add_loading_matrix(self, loadings: np.ndarray,
                           threshold: float = 0.3) -> None:
        communalities = compute_communalities(loadings)

        header = "<tr><th>Variable</th>"
        for f in range(self.n_factors):
            color = _hf(f)
            header += f'<th style="color:{color}">Factor {f+1}</th>'
        header += "<th>Communality h²</th></tr>"

        rows = ""
        for i, var in enumerate(self.variables):
            cells = f"<td class='col-name'>{self._e(var)}</td>"
            for f in range(self.n_factors):
                v     = loadings[i, f]
                abs_v = abs(v)
                color = _hf(f)
                if abs_v >= 0.6:
                    style = f"color:{color};font-weight:700"
                elif abs_v >= threshold:
                    style = f"color:{color}"
                else:
                    style = "color:#94a3b8"
                cells += f'<td class="mono" style="{style}">{v:.4f}</td>'

            h2    = communalities[i]
            h2_color = "#16a34a" if h2 >= 0.6 else ("#ca8a04" if h2 >= 0.4 else "#dc2626")
            cells += f'<td class="mono" style="color:{h2_color};font-weight:600">{h2:.4f}</td>'
            rows  += f"<tr>{cells}</tr>"

        legend = f"""
        <p class="cv-legend">
          <span style="font-weight:700">Bold</span> loading ≥ 0.6 &nbsp;·&nbsp;
          <span style="color:#94a3b8">Dim</span> loading &lt; {threshold} (suppressed) &nbsp;·&nbsp;
          Communality: <span style="color:#16a34a;font-weight:600">≥ 0.6 good</span>
          <span style="color:#ca8a04;font-weight:600">0.4–0.6 moderate</span>
          <span style="color:#dc2626;font-weight:600">&lt; 0.4 weak</span>
        </p>"""

        self._sections.append(f"""
        <section>
          <h2>Rotated Factor Matrix <span class="subtitle">{self._e(self.rotation.capitalize())} rotation</span></h2>
          <table class="data-table">
            <thead>{header}</thead>
            <tbody>{rows}</tbody>
          </table>
          {legend}
        </section>""")

    # ── Communalities ──────────────────────────────────────────────────────────
    def add_communalities(self, communalities: np.ndarray) -> None:
        rows = ""
        for var, h2 in zip(self.variables, communalities):
            pct   = h2 * 100
            color = "#16a34a" if h2 >= 0.6 else ("#ca8a04" if h2 >= 0.4 else "#dc2626")
            badge_cls = "rating-good" if h2 >= 0.6 else ("rating-fair" if h2 >= 0.4 else "rating-weak")
            badge_txt = "Good" if h2 >= 0.6 else ("Moderate" if h2 >= 0.4 else "Weak")
            rows += f"""
            <tr>
              <td class="col-name">{self._e(var)}</td>
              <td class="mono" style="color:{color};font-weight:600">{h2:.4f}</td>
              <td>
                <div class="bar-wrap" style="width:200px">
                  <div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div>
                </div>
              </td>
              <td><span class="rating-badge {badge_cls}">{badge_txt}</span></td>
            </tr>"""

        mean_h2 = float(np.mean(communalities))
        rows += f"""
            <tr style="font-weight:700;border-top:2px solid #e2e8f0">
              <td>Mean h²</td>
              <td class="mono">{mean_h2:.4f}</td>
              <td></td><td></td>
            </tr>"""

        self._sections.append(f"""
        <section>
          <h2>Communalities <span class="subtitle">proportion of variance each variable shares with the factors</span></h2>
          <table class="simple-table">
            <thead><tr><th>Variable</th><th>h²</th><th>Visual</th><th>Rating</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </section>""")

    # ── Correlation matrix ─────────────────────────────────────────────────────
    def add_correlation_matrix(self, corr: np.ndarray) -> None:
        n = len(self.variables)

        header = "<tr><th></th>"
        for v in self.variables:
            header += f"<th>{self._e(v)}</th>"
        header += "</tr>"

        rows = ""
        for i, row_var in enumerate(self.variables):
            cells = f"<td class='col-name'>{self._e(row_var)}</td>"
            for j in range(n):
                v = corr[i, j]
                if i == j:
                    cells += '<td class="mono corr-diag">1.000</td>'
                else:
                    abs_v = abs(v)
                    if abs_v >= 0.7:
                        bg = "#dcfce7" if v > 0 else "#fee2e2"
                        fw = "font-weight:700;"
                        tc = "#166534" if v > 0 else "#991b1b"
                    elif abs_v >= 0.4:
                        bg = "#fef9c3" if v > 0 else "#fdf4ff"
                        fw = ""
                        tc = "#854d0e" if v > 0 else "#7e22ce"
                    else:
                        bg, fw, tc = "#f8fafc", "", "#94a3b8"
                    cells += f'<td class="mono" style="background:{bg};{fw}color:{tc}">{v:.3f}</td>'
            rows += f"<tr>{cells}</tr>"

        legend = """
        <p class="cv-legend">
          <span style="background:#dcfce7;color:#166534;font-weight:700;padding:2px 8px;border-radius:4px">≥ 0.7 strong +</span>
          <span style="background:#fef9c3;color:#854d0e;padding:2px 8px;border-radius:4px">0.4–0.7 moderate +</span>
          <span style="background:#fee2e2;color:#991b1b;font-weight:700;padding:2px 8px;border-radius:4px">≤ −0.7 strong −</span>
          <span style="background:#fdf4ff;color:#7e22ce;padding:2px 8px;border-radius:4px">−0.4–−0.7 moderate −</span>
        </p>"""

        self._sections.append(f"""
        <section>
          <h2>Correlation Matrix <span class="subtitle">Pearson — colour-coded by strength</span></h2>
          <div style="overflow-x:auto">
            <table class="data-table corr-table">
              <thead>{header}</thead>
              <tbody>{rows}</tbody>
            </table>
          </div>
          {legend}
        </section>""")

    # ── Render full page ───────────────────────────────────────────────────────
    def render(self) -> str:
        ts       = datetime.now().strftime("%Y-%m-%d %H:%M")
        sections = "\n".join(self._sections)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Factor Analysis Report — {self._e(self.filepath)}</title>
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
  .report-header {{
    background: var(--accent);
    color: white;
    padding: 36px 48px 32px;
  }}
  .report-header h1 {{ font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 6px; }}
  .report-header .filepath {{ font-family: var(--mono); font-size: 0.85rem; opacity: 0.8; margin-bottom: 18px; }}
  .meta-pills {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .meta-pill {{
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 99px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 500;
  }}
  .content {{ max-width: 1100px; margin: 0 auto; padding: 0 32px; }}
  section {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px 32px;
    margin-top: 24px;
  }}
  section h2 {{
    font-size: 1rem; font-weight: 700; color: var(--text);
    margin-bottom: 18px; padding-bottom: 10px;
    border-bottom: 2px solid var(--border);
    display: flex; align-items: baseline; gap: 10px;
  }}
  .subtitle {{ font-size: 0.75rem; font-weight: 400; color: var(--muted); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{
    text-align: left; padding: 9px 14px;
    font-size: 0.75rem; font-weight: 600;
    letter-spacing: 0.04em; color: var(--muted);
    border-bottom: 2px solid var(--border); white-space: nowrap;
  }}
  td {{ padding: 9px 14px; border-bottom: 1px solid var(--border); vertical-align: middle; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f8fafc; }}
  .mono {{ font-family: var(--mono); font-size: 0.82rem; }}
  .col-name {{ font-family: var(--mono); font-size: 0.82rem; color: var(--text); font-weight: 500; white-space: nowrap; }}
  .data-table td {{ font-family: var(--mono); font-size: 0.82rem; }}
  .factor-badge {{
    display: inline-block; padding: 3px 10px;
    border-radius: 99px; font-size: 0.78rem; font-weight: 600; white-space: nowrap;
  }}
  .rating-badge {{
    display: inline-block; padding: 2px 10px;
    border-radius: 99px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.03em;
  }}
  .rating-good {{ background: #dcfce7; color: #166534; }}
  .rating-fair {{ background: #fef9c3; color: #854d0e; }}
  .rating-weak {{ background: #fee2e2; color: #991b1b; }}
  .bar-wrap {{
    background: #f1f5f9; border-radius: 99px;
    height: 8px; width: 160px; overflow: hidden;
    display: inline-block; vertical-align: middle;
  }}
  .bar-fill {{ height: 100%; border-radius: 99px; }}
  .simple-table .bar-wrap {{ width: 200px; }}
  .cv-legend {{
    margin-top: 14px; font-size: 0.78rem; color: var(--muted);
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  }}
  .corr-table th, .corr-table td {{ padding: 7px 10px; text-align: center; white-space: nowrap; }}
  .corr-table th:first-child, .corr-table td:first-child {{ text-align: left; }}
  .corr-diag {{ color: #94a3b8; }}
  @media print {{
    body {{ background: white; }}
    .report-header {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    .bar-fill, .rating-badge, .factor-badge {{
      -webkit-print-color-adjust: exact; print-color-adjust: exact;
    }}
    section {{ break-inside: avoid; }}
  }}
</style>
</head>
<body>
<div class="report-header">
  <h1>Factor Analysis Report</h1>
  <div class="filepath">{self._e(self.filepath)}</div>
  <div class="meta-pills">
    <span class="meta-pill">{self.n_factors} factors retained</span>
    <span class="meta-pill">{self.n_vars} variables</span>
    <span class="meta-pill">{self.n_obs:,} observations</span>
    <span class="meta-pill">{self._e(self.rotation.capitalize())} rotation</span>
    <span class="meta-pill">Generated {ts}</span>
  </div>
</div>
<div class="content">
{sections}
</div>
</body>
</html>"""


def export_html(report: HtmlReport, path: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(report.render())
        print(f"\n{GREEN}✓ HTML report saved → {path}{RESET}")
    except OSError as e:
        print(f"{RED}✗ Could not write HTML report: {e}{RESET}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="factor_analysis.py",
        description="Factor Analysis for CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python factor_analysis.py data.csv
  python factor_analysis.py data.csv --factors 4
  python factor_analysis.py data.csv --cols age income score visits
  python factor_analysis.py data.csv --rotation oblimin
  python factor_analysis.py data.csv --auto-factors --min-eigen 1.0
  python factor_analysis.py data.csv --factors 3 --export report.html
        """,
    )
    parser.add_argument("filepath",        help="Path to the CSV file")
    parser.add_argument("--factors",       type=int, default=None,
                        help="Number of factors to retain (default: auto via Kaiser criterion)")
    parser.add_argument("--cols",          nargs="+",
                        help="Specific column names to use (default: all numeric)")
    parser.add_argument("--rotation",      choices=["varimax", "oblimin"], default="varimax",
                        help="Rotation method: varimax (orthogonal) or oblimin (oblique). Default: varimax")
    parser.add_argument("--auto-factors",  action="store_true",
                        help="Automatically select number of factors via Kaiser criterion (eigenvalue ≥ min-eigen)")
    parser.add_argument("--min-eigen",     type=float, default=1.0,
                        help="Minimum eigenvalue threshold for auto-factor selection (default: 1.0)")
    parser.add_argument("--loading-threshold", type=float, default=0.3,
                        help="Abs loading value below which cells are dimmed (default: 0.3)")
    parser.add_argument("--export",        metavar="FILE.html", default=None,
                        help="Export a self-contained HTML report to this path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load ──
    df, variables = load_and_prepare(args.filepath, args.cols)
    n_vars = len(variables)
    n_obs  = len(df)

    # ── Correlation matrix ──
    corr = compute_correlation_matrix(df)

    # ── Eigenvalues via PCA on correlation matrix ──
    # Initial extraction with max possible factors to get all eigenvalues
    eigenvalues, _ = extract_factors_pca(corr, n_vars)

    # ── Determine number of factors ──
    if args.factors:
        n_factors = args.factors
        if n_factors < 1 or n_factors >= n_vars:
            print(f"{RED}✗ --factors must be between 1 and {n_vars - 1}.{RESET}")
            sys.exit(1)
    else:
        n_factors = suggest_n_factors(eigenvalues, args.min_eigen)
        print(f"{CYAN}ℹ  Kaiser criterion (eigenvalue ≥ {args.min_eigen}): retaining {n_factors} factors{RESET}")

    # ── Extract and rotate loadings ──
    _, loadings_unrotated = extract_factors_pca(corr, n_factors)

    if args.rotation == "varimax":
        loadings = varimax_rotation(loadings_unrotated)
    else:
        loadings = oblimin_rotation(loadings_unrotated)

    # ── Derived stats ──
    communalities          = compute_communalities(loadings)
    var_each, pct_each, pct_cum = compute_variance_explained(eigenvalues, loadings, n_factors)

    # ── Terminal output ──
    print_banner(args.filepath, n_vars, n_obs, n_factors, args.rotation)
    print(f"\n  {BOLD}Variables:{RESET} {', '.join(variables)}")
    print(f"  {DIM}(All factors extracted from standardised Pearson correlation matrix){RESET}")

    print_eigenvalues(eigenvalues[:min(n_vars, 12)], n_factors)
    print_variance_table(var_each, pct_each, pct_cum, n_factors)
    print_loading_matrix(loadings, variables, n_factors, args.rotation, args.loading_threshold)
    print_communalities(communalities, variables)
    print_correlation_matrix(corr, variables)
    print(hr("═") + "\n")

    # ── HTML export ──
    if args.export:
        report = HtmlReport(
            filepath=args.filepath, n_vars=n_vars, n_obs=n_obs,
            n_factors=n_factors, rotation=args.rotation, variables=variables,
        )
        report.add_eigenvalues(eigenvalues[:min(n_vars, 12)])
        report.add_variance_table(var_each, pct_each, pct_cum)
        report.add_loading_matrix(loadings, args.loading_threshold)
        report.add_communalities(communalities)
        report.add_correlation_matrix(corr)
        export_html(report, args.export)


if __name__ == "__main__":
    main()
