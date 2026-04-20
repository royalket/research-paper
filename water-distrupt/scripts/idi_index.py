"""
idi_index.py
─────────────────────────────────────────────────────────────────────────────
Responsibility: Build the Infrastructure Dependency Index (IDI).

Pipeline inside this file:
  1. Score 3 dimensions (0-3 each) per household
  2. Fit PCA on dimensions → first PC = empirical weights
  3. Monte Carlo: perturb input scores 1,000x → IDI distribution per HH
  4. Attach mean IDI + 95% CI back to household DataFrame
  5. Validate: Cronbach's alpha, AUC, discriminant validity

Key design decisions:
  - PCA is fit ONCE on the observed scores. MC does NOT refit PCA.
    Only the INPUT scores vary. This means weights are stable.
  - Noise is Gaussian, clipped to ±1 so scores stay in [0,3] range.
  - All MC runs also refit the core logistic regression so we get
    a distribution of the piped-water OR — that's our robustness proof.
─────────────────────────────────────────────────────────────────────────────
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
from typing import Dict, Tuple

from config import Config

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# DIMENSION SCORING  (each returns a float Series, values 0.0 – 3.0)
# ─────────────────────────────────────────────────────────────────────────────

def score_dim1_source_diversity(df: pd.DataFrame) -> pd.Series:
    """
    Dimension 1 — Source Diversification (lack of fallback).

    Concept: Does the household have ANY backup if its primary source fails?
    A household with only piped water and no alternative is maximally locked in.

    Score │ Situation
    ──────┼────────────────────────────────────────────────────────────────
      3   │ Piped primary + no alternative   OR  piped primary + piped alt
          │   → completely dependent on one centralized system
      2   │ Piped primary + non-piped alternative
          │   → system-dependent but has an escape route
      1   │ Non-piped primary + same-type alternative
          │   → some diversity but not much
      0   │ Non-piped primary + different-type alternative
          │   → genuinely diversified, lowest lock-in
    """
    scores = pd.Series(0.0, index=df.index)

    primary_piped = df["piped_flag"] == 1
    no_alt        = df["alt_source"] == "No Other Source"
    alt_piped     = df["alt_source"] == "Piped Water"
    alt_exists    = ~no_alt

    # Score 3: piped only, or piped + piped (no real alternative)
    scores[primary_piped & (no_alt | alt_piped)] = 3.0

    # Score 2: piped + a real non-piped backup
    scores[primary_piped & alt_exists & ~alt_piped] = 2.0

    # Score 1: non-piped but alternative is same category (weak diversity)
    same_cat = (
        ~primary_piped & alt_exists &
        (df["water_source"] == df["alt_source"])
    )
    scores[same_cat] = 1.0

    # Score 0: non-piped + genuinely different alternative → already 0.0

    return scores


def score_dim2_access_complexity(df: pd.DataFrame) -> pd.Series:
    """
    Dimension 2 — Access Complexity (preparedness for fetching).

    Key insight: This is INVERTED from traditional vulnerability.
    Households that already walk to fetch water are EXPERIENCED at coping.
    Households with water in-dwelling are UNPREPARED when the tap runs dry.

    Score │ Situation
    ──────┼────────────────────────────────────────────────────────────────
      3   │ Water in dwelling (on-premises, zero fetching habit)
      2   │ Water in yard/plot (minimal fetching experience)
      1   │ Water elsewhere, < 15 minutes (some experience)
      0   │ Water elsewhere, ≥ 15 minutes (fetching-experienced, adaptable)
    """
    scores = pd.Series(1.0, index=df.index)   # sensible default

    loc  = df["water_location"]
    time = df["time_to_water_min"].fillna(np.nan)

    scores[loc == "In Dwelling"]                           = 3.0
    scores[loc == "In Yard/Plot"]                          = 2.0
    scores[(loc == "Elsewhere") & (time <  15)]            = 1.0
    scores[(loc == "Elsewhere") & (time >= 15)]            = 0.0

    return scores


def score_dim3_system_dependency(df: pd.DataFrame) -> pd.Series:
    """
    Dimension 3 — Market / System Dependency.

    How much does alternative-sourcing require money or formal system access?
    If coping means buying from a tanker, that costs money AND is unreliable.

    Score │ Situation
    ──────┼────────────────────────────────────────────────────────────────
      3   │ Tanker / bottled water (pure market — expensive, unreliable)
      2   │ Piped water (system-dependent — single point of failure)
      1   │ Community well / protected spring (semi-managed)
      0   │ Own well / rainwater / surface water (self-sufficient)
    """
    scores = pd.Series(0.0, index=df.index)

    src = df["water_source"]

    scores[src.isin(["Tanker/Cart", "Bottled Water"])]               = 3.0
    scores[src == "Piped Water"]                                      = 2.0
    scores[src.isin(["Community RO Plant",
                     "Protected Well/Spring",
                     "Protected Spring"])]                            = 1.0
    # Own well, rainwater, surface → already 0.0

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# PCA WEIGHTING
# ─────────────────────────────────────────────────────────────────────────────

def fit_pca(dim_df: pd.DataFrame) -> Tuple[PCA, StandardScaler, np.ndarray]:
    """
    Fit PCA on the three dimension scores.

    Returns
    -------
    pca       : fitted sklearn PCA object (1 component)
    scaler    : fitted StandardScaler (z-score normalisation before PCA)
    loadings  : ndarray shape (3,) — first PC loadings (our weights)
    """
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(dim_df.values)

    pca = PCA(n_components=1, random_state=0)
    pca.fit(X_scaled)

    loadings         = pca.components_[0]          # shape (3,)
    explained_var    = pca.explained_variance_ratio_[0]

    print(f"\n  PCA Results:")
    print(f"    Dimension weights (loadings on PC1):")
    for name, w in zip(dim_df.columns, loadings):
        print(f"      {name:30s}: {w:+.4f}")
    print(f"    Variance explained by PC1: {explained_var*100:.1f}%")

    return pca, scaler, loadings


def apply_pca(dim_df: pd.DataFrame,
              pca: PCA,
              scaler: StandardScaler) -> np.ndarray:
    """
    Project dimension scores through fitted PCA → raw IDI scores.
    Scores are NOT yet normalised to 0-100 here; that happens after MC.
    """
    X_scaled = scaler.transform(dim_df.values)
    return pca.transform(X_scaled).flatten()   # shape (n_households,)


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO UNCERTAINTY
# ─────────────────────────────────────────────────────────────────────────────

def run_monte_carlo(
    df: pd.DataFrame,
    dim_cols: list,
    pca: PCA,
    scaler: StandardScaler,
    cfg: Config,
) -> Dict[str, np.ndarray]:
    """
    Monte Carlo uncertainty quantification.

    For each of N=1000 runs:
      1. Add Gaussian noise to dimension scores (noise ~ N(0, cfg.MONTE_CARLO_NOISE))
         Clip so scores stay within [0, 3].
      2. Re-project through FIXED PCA (weights do not change between runs).
      3. Refit logistic regression: disruption ~ piped_flag + idi + piped×idi + controls
      4. Store: per-household IDI score, OR for piped_flag, OR for interaction.

    After all runs:
      - Per-household IDI: mean, 2.5th, 97.5th percentile → confidence interval
      - OR(piped): distribution → mean ± 95% CI
      - Robustness: % of runs where OR(piped) > 1.0

    Parameters
    ----------
    df       : clean household DataFrame (must contain dim_cols + regression vars)
    dim_cols : list of 3 dimension column names in df
    pca      : fitted PCA (from fit_pca)
    scaler   : fitted StandardScaler (from fit_pca)
    cfg      : Config object

    Returns
    -------
    Dictionary with keys:
      'idi_all_runs'     : ndarray (n_runs, n_hh)  — IDI score each run
      'or_piped'         : ndarray (n_runs,)        — OR(piped) each run
      'or_interaction'   : ndarray (n_runs,)        — OR(piped×idi) each run
      'pct_runs_or_gt1'  : float                    — robustness metric
    """
    print(f"\n  Running Monte Carlo ({cfg.MONTE_CARLO_RUNS} iterations)...")

    rng     = np.random.default_rng(cfg.MONTE_CARLO_SEED)
    n_hh    = len(df)
    n_runs  = cfg.MONTE_CARLO_RUNS

    dim_array = df[dim_cols].values.astype(float)   # shape (n_hh, 3)

    idi_all_runs   = np.empty((n_runs, n_hh))
    or_piped       = np.empty(n_runs)
    or_interaction = np.empty(n_runs)

    # Prepare regression data once (control variables don't change)
    reg_df = df[[
        cfg.VAR_DISRUPTED,
        "piped_flag",
        "wealth_q_num",
        "urban",
        "hh_size",
        "children_u5",
        "region",
        cfg.VAR_PSU,
        "weight",
    ]].dropna().copy()

    for run in range(n_runs):
        # ── Step 1: perturb dimension scores ─────────────────────────────
        noise    = rng.normal(0, cfg.MONTE_CARLO_NOISE, size=dim_array.shape)
        perturbed = np.clip(dim_array + noise, 0, 3)

        # ── Step 2: project through FIXED PCA ────────────────────────────
        X_scaled = scaler.transform(perturbed)
        idi_run  = pca.transform(X_scaled).flatten()
        idi_all_runs[run] = idi_run

        # ── Step 3: attach IDI to regression df and fit model ─────────────
        reg_df_run = reg_df.copy()
        reg_df_run["idi"] = idi_run[reg_df.index.map(
            lambda i: np.where(df.index == i)[0][0]
            if i in df.index else np.nan
        )]

        # Safer index alignment
        idx_map = {orig: pos for pos, orig in enumerate(df.index)}
        reg_df_run["idi"] = reg_df.index.map(
            lambda i: idi_run[idx_map[i]] if i in idx_map else np.nan
        )
        reg_df_run.dropna(subset=["idi"], inplace=True)

        if len(reg_df_run) < 100:
            or_piped[run]       = np.nan
            or_interaction[run] = np.nan
            continue

        try:
            formula = (
                f"{cfg.VAR_DISRUPTED} ~ piped_flag + idi + piped_flag:idi "
                f"+ wealth_q_num + urban + hh_size + children_u5 + C(region)"
            )
            model = smf.logit(
                formula=formula,
                data=reg_df_run,
                freq_weights=reg_df_run["weight"],
            ).fit(
                disp=False, maxiter=300,
                cov_type="cluster",
                cov_kwds={"groups": reg_df_run[cfg.VAR_PSU]},
            )
            or_piped[run]       = np.exp(model.params.get("piped_flag", np.nan))
            or_interaction[run] = np.exp(model.params.get("piped_flag:idi", np.nan))
        except Exception:
            or_piped[run]       = np.nan
            or_interaction[run] = np.nan

        if (run + 1) % 100 == 0:
            print(f"    Run {run+1}/{n_runs} complete")

    # ── Robustness metric ─────────────────────────────────────────────────
    valid_or    = or_piped[~np.isnan(or_piped)]
    pct_gt1     = (valid_or > 1.0).mean() * 100

    print(f"\n  Monte Carlo complete.")
    print(f"    Piped water OR > 1 in {pct_gt1:.1f}% of runs "
          f"({'ROBUST' if pct_gt1 >= 95 else 'MODERATE'} robustness)")

    return {
        "idi_all_runs":    idi_all_runs,
        "or_piped":        or_piped,
        "or_interaction":  or_interaction,
        "pct_runs_or_gt1": pct_gt1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_idi(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Three validation checks reported as a clean DataFrame.

    Check 1 — Predictive validity
        Correlation between IDI and disruption.
        AUC: can IDI alone discriminate disrupted from not-disrupted?
        Target: AUC > 0.60

    Check 2 — Discriminant validity
        Correlation between IDI and wealth score.
        We want IDI to capture something DIFFERENT from wealth.
        Target: |r| < 0.50  (if r > 0.80, IDI is just a wealth proxy)

    Check 3 — Known-groups validity
        IDI should be substantially higher for piped water users.
        This is a sanity check that our scoring makes intuitive sense.
    """
    rows = []

    # Check 1a: Pearson r (IDI vs disruption)
    valid = df[["idi_mean", cfg.VAR_DISRUPTED, "wealth_score"]].dropna()
    r_dis, p_dis = pearsonr(valid["idi_mean"], valid[cfg.VAR_DISRUPTED])
    rows.append({
        "Check": "Predictive validity",
        "Metric": "Pearson r (IDI vs Disruption)",
        "Value": round(r_dis, 3),
        "p-value": "<0.001" if p_dis < 0.001 else f"{p_dis:.3f}",
        "Target": "r > 0",
        "Pass": "✓" if r_dis > 0 and p_dis < 0.05 else "✗",
    })

    # Check 1b: AUC
    try:
        auc = roc_auc_score(
            valid[cfg.VAR_DISRUPTED],
            valid["idi_mean"],
            sample_weight=df.loc[valid.index, "weight"],
        )
        rows.append({
            "Check": "Predictive validity",
            "Metric": "ROC-AUC (IDI predicting disruption)",
            "Value": round(auc, 3),
            "p-value": "—",
            "Target": "> 0.60",
            "Pass": "✓" if auc > 0.60 else "✗",
        })
    except Exception:
        rows.append({
            "Check": "Predictive validity",
            "Metric": "ROC-AUC",
            "Value": np.nan,
            "p-value": "Error",
            "Target": "> 0.60",
            "Pass": "✗",
        })

    # Check 2: Discriminant validity (IDI vs wealth)
    r_wlth, p_wlth = pearsonr(valid["idi_mean"], valid["wealth_score"])
    rows.append({
        "Check": "Discriminant validity",
        "Metric": "Pearson r (IDI vs Wealth Score)",
        "Value": round(r_wlth, 3),
        "p-value": "<0.001" if p_wlth < 0.001 else f"{p_wlth:.3f}",
        "Target": "|r| < 0.50",
        "Pass": "✓" if abs(r_wlth) < 0.50 else "✗",
    })

    # Check 3: Known-groups validity
    piped_idi = df[df["piped_flag"] == 1]["idi_mean"].mean()
    well_idi  = df[df["tube_well_flag"] == 1]["idi_mean"].mean()
    diff      = piped_idi - well_idi
    rows.append({
        "Check": "Known-groups validity",
        "Metric": "Mean IDI: Piped vs Tube Well (difference)",
        "Value": round(diff, 3),
        "p-value": "—",
        "Target": "Piped > Tube Well",
        "Pass": "✓" if diff > 0 else "✗",
    })

    val_df = pd.DataFrame(rows)
    print("\n  IDI Validation:")
    print(val_df.to_string(index=False))
    return val_df


def cronbach_alpha(df: pd.DataFrame, cols: list) -> float:
    """Compute Cronbach's alpha for internal consistency of IDI dimensions."""
    k = len(cols)
    item_vars  = df[cols].var(axis=0, ddof=1).sum()
    total_var  = df[cols].sum(axis=1).var(ddof=1)
    alpha      = (k / (k - 1)) * (1 - item_vars / total_var)
    return round(alpha, 3)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BUILDER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class IDIBuilder:
    """
    Orchestrates IDI construction.
    Call .build() → returns df with IDI columns attached.

    New columns added to df:
      idi_dim1        : Dimension 1 score (0-3)
      idi_dim2        : Dimension 2 score (0-3)
      idi_dim3        : Dimension 3 score (0-3)
      idi_raw_pca     : Raw PCA projection (unbounded)
      idi_mean        : Mean IDI across 1000 MC runs, normalised 0-100
      idi_ci_lower    : 2.5th percentile across MC runs
      idi_ci_upper    : 97.5th percentile across MC runs
      idi_ci_width    : CI width (higher = more uncertain)
    """

    DIM_COLS = ["idi_dim1", "idi_dim2", "idi_dim3"]

    def __init__(self, df: pd.DataFrame, cfg: Config):
        self.df  = df.copy()
        self.cfg = cfg
        self.pca     = None
        self.scaler  = None
        self.loadings = None
        self.mc_results = None
        self.validation_df = None

    def build(self) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("STEP 3 — Building IDI (PCA + Monte Carlo)")
        print("=" * 60)

        # 1. Score dimensions
        self.df["idi_dim1"] = score_dim1_source_diversity(self.df)
        self.df["idi_dim2"] = score_dim2_access_complexity(self.df)
        self.df["idi_dim3"] = score_dim3_system_dependency(self.df)

        # Internal consistency check
        alpha = cronbach_alpha(self.df, self.DIM_COLS)
        print(f"\n  Cronbach's Alpha (dimension consistency): {alpha}")
        print(f"  {'✓ Good internal consistency' if alpha > 0.6 else '⚠ Low consistency — review dimension definitions'}")

        # 2. Fit PCA (on non-missing rows)
        dim_df = self.df[self.DIM_COLS].dropna()
        self.pca, self.scaler, self.loadings = fit_pca(dim_df)

        # 3. Raw PCA score on all rows
        self.df["idi_raw_pca"] = np.nan
        valid_idx = self.df[self.DIM_COLS].dropna().index
        self.df.loc[valid_idx, "idi_raw_pca"] = apply_pca(
            self.df.loc[valid_idx, self.DIM_COLS],
            self.pca,
            self.scaler,
        )

        # 4. Monte Carlo
        self.mc_results = run_monte_carlo(
            self.df, self.DIM_COLS, self.pca, self.scaler, self.cfg
        )

        # 5. Summarise MC runs per household
        idi_runs = self.mc_results["idi_all_runs"]   # shape (n_runs, n_hh)

        # Normalise each run to 0-100 before computing percentiles
        run_min = idi_runs.min(axis=1, keepdims=True)
        run_max = idi_runs.max(axis=1, keepdims=True)
        denom   = np.where(run_max - run_min == 0, 1, run_max - run_min)
        idi_runs_normed = (idi_runs - run_min) / denom * 100

        self.df["idi_mean"]     = np.nanmean(idi_runs_normed, axis=0)
        self.df["idi_ci_lower"] = np.nanpercentile(idi_runs_normed, 2.5,  axis=0)
        self.df["idi_ci_upper"] = np.nanpercentile(idi_runs_normed, 97.5, axis=0)
        self.df["idi_ci_width"] = self.df["idi_ci_upper"] - self.df["idi_ci_lower"]

        # 6. Validate
        self.validation_df = validate_idi(self.df, self.cfg)

        # 7. Save MC robustness summary
        self._save_mc_summary()

        print("\n  ✓  IDI construction complete")
        print(f"     Mean IDI: {self.df['idi_mean'].mean():.1f}  "
              f"(SD {self.df['idi_mean'].std():.1f})")
        print(f"     Mean CI width: {self.df['idi_ci_width'].mean():.1f} pp")
        print("=" * 60)

        return self.df

    def _save_mc_summary(self):
        """Save Monte Carlo OR distribution summary to CSV."""
        or_piped = self.mc_results["or_piped"]
        or_inter  = self.mc_results["or_interaction"]

        summary = pd.DataFrame({
            "Metric": [
                "OR (Piped Water) — Mean",
                "OR (Piped Water) — 2.5th pctile",
                "OR (Piped Water) — 97.5th pctile",
                "OR (Piped × IDI interaction) — Mean",
                "OR (Piped × IDI interaction) — 2.5th pctile",
                "OR (Piped × IDI interaction) — 97.5th pctile",
                "% MC runs where OR(Piped) > 1.0  [ROBUSTNESS]",
            ],
            "Value": [
                np.nanmean(or_piped),
                np.nanpercentile(or_piped, 2.5),
                np.nanpercentile(or_piped, 97.5),
                np.nanmean(or_inter),
                np.nanpercentile(or_inter, 2.5),
                np.nanpercentile(or_inter, 97.5),
                self.mc_results["pct_runs_or_gt1"],
            ],
        }).round(3)

        out_path = self.cfg.OUTPUT_DIR / "results" / "idi_monte_carlo_summary.csv"
        summary.to_csv(out_path, index=False)
        print(f"\n  MC summary saved → {out_path}")

        # Also save validation
        val_path = self.cfg.OUTPUT_DIR / "results" / "idi_validation.csv"
        self.validation_df.to_csv(val_path, index=False)
        print(f"  IDI validation saved → {val_path}")

    @property
    def pca_loadings(self) -> pd.DataFrame:
        """Return PCA loadings as a readable DataFrame."""
        return pd.DataFrame({
            "Dimension": self.DIM_COLS,
            "Loading (PC1)": self.loadings.round(4),
            "Contribution (%)": (self.loadings**2 / (self.loadings**2).sum() * 100).round(1),
        })
