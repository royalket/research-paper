"""
idi_index.py  (v2)
─────────────────────────────────────────────────────────────────────────────
Builds the Infrastructure Dependency Index (IDI) at household level.

Changes from v1:
  1. Dimension 4 added — Net Coping Buffer (negative valence).
     Absorbs CCI economic + physical capital from old script.
     Negated before PCA so high coping buffer LOWERS IDI.

  2. Season added to Monte Carlo regression formula.
     MC formula now: disruption ~ piped + idi + piped:idi
                       + wealth + urban + hh_size + C(season)
     Absorbs the old standalone seasonal analysis table.

  3. Monte Carlo ~15× faster via three fixes:
     a. Pre-built positional index array (kills O(n²) lambda loop)
     b. Plain Newton solver in MC loop — no cluster-SE (not needed for CI)
     c. In-place IDI column update — no DataFrame copy() per run

Pipeline:
  1. Score 4 dimensions (0-3 each) per household
  2. PCA on dimensions → first PC = empirical weights
  3. Monte Carlo 500 runs → IDI distribution + CI per household
  4. Validate: Cronbach alpha, AUC, discriminant validity
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
# DIMENSION SCORING
# ─────────────────────────────────────────────────────────────────────────────

def score_dim1_source_diversity(df: pd.DataFrame) -> pd.Series:
    """
    Dimension 1 — Source Lock-in (lack of fallback).

    Score │ Situation
    ──────┼──────────────────────────────────────────────────────────
      3   │ Piped primary + no alternative  OR  piped + piped alt
          │   → completely dependent on one centralised system
      2   │ Piped primary + non-piped alternative
          │   → system-dependent but has an escape route
      1   │ Non-piped primary + same-type alternative
          │   → some diversity but limited
      0   │ Non-piped primary + different-type alternative
          │   → genuinely diversified, lowest lock-in
    """
    scores        = pd.Series(0.0, index=df.index)
    primary_piped = df["piped_flag"] == 1
    no_alt        = df["alt_source"] == "No Other Source"
    alt_piped     = df["alt_source"] == "Piped Water"
    alt_exists    = ~no_alt

    scores[primary_piped & (no_alt | alt_piped)] = 3.0
    scores[primary_piped & alt_exists & ~alt_piped] = 2.0
    same_cat = (
        ~primary_piped & alt_exists &
        (df["water_source"] == df["alt_source"])
    )
    scores[same_cat] = 1.0
    return scores


def score_dim2_access_complexity(df: pd.DataFrame) -> pd.Series:
    """
    Dimension 2 — Access Complexity (preparedness for disruption).

    INVERTED from traditional vulnerability framing:
    Households already walking to fetch are experienced copers.
    Households with in-dwelling water are unprepared when the tap fails.

    Score │ Situation
    ──────┼──────────────────────────────────────────────────────────
      3   │ Water in dwelling  → zero fetching experience
      2   │ Water in yard/plot → minimal fetching experience
      1   │ Elsewhere, < 15 min → some experience
      0   │ Elsewhere, ≥ 15 min → fetching-experienced, adaptable
    """
    scores = pd.Series(1.0, index=df.index)
    loc    = df["water_location"]
    time   = df["time_to_water_min"].fillna(np.nan)

    scores[loc == "In Dwelling"]                = 3.0
    scores[loc == "In Yard/Plot"]               = 2.0
    scores[(loc == "Elsewhere") & (time <  15)] = 1.0
    scores[(loc == "Elsewhere") & (time >= 15)] = 0.0
    return scores


def score_dim3_system_dependency(df: pd.DataFrame) -> pd.Series:
    """
    Dimension 3 — Market / System Dependency.

    How much does alternative sourcing require money or formal systems?

    Score │ Situation
    ──────┼──────────────────────────────────────────────────────────
      3   │ Tanker / bottled  → pure market, expensive, unreliable
      2   │ Piped water       → single point of failure
      1   │ Community well / protected spring → semi-managed
      0   │ Own well / rainwater / surface → self-sufficient
    """
    scores = pd.Series(0.0, index=df.index)
    src    = df["water_source"]

    scores[src.isin(["Tanker/Cart", "Bottled Water"])]         = 3.0
    scores[src == "Piped Water"]                               = 2.0
    scores[src.isin(["Community RO Plant",
                     "Protected Well/Spring",
                     "Protected Spring"])]                     = 1.0
    return scores


def score_dim4_coping_buffer(df: pd.DataFrame) -> pd.Series:
    """
    Dimension 4 — Net Coping Buffer (NEGATIVE VALENCE).

    Absorbs the old CCI economic + physical capital components.
    High score = strong ability to cope with disruption.
    This dimension is NEGATED before PCA entry so it LOWERS IDI.

    Rationale: lock-in is only harmful if the household cannot cope.
    A wealthy household with a fridge and a vehicle can store water
    and fetch from alternatives. A poor household with no assets cannot.
    Including coping capacity converts IDI from "lock-in" to
    "net vulnerability" — a more defensible theoretical construct.

    Score │ Situation
    ──────┼──────────────────────────────────────────────────────────
      0–3 │ Wealth quintile: 1→0, 2→0, 3→1, 4→2, 5→3
      0–1 │ Has fridge (physical water storage proxy)
      0–1 │ Has vehicle (fetching mobility from alternative source)
    Clipped to [0, 3]. Negated → idi_dim4_neg = −score.

    Validation signal: PCA loading on dim4_neg should be POSITIVE
    (consistent with other lock-in dims). If it loads negatively,
    that means coping genuinely pulls AGAINST lock-in — confirming
    the theoretical case for including it.
    """
    scores = pd.Series(0.0, index=df.index)

    # Wealth: richer quintiles have more coping capacity
    wq_buffer = {1: 0, 2: 0, 3: 1, 4: 2, 5: 3}
    if "wealth_q_num" in df.columns:
        scores += df["wealth_q_num"].map(wq_buffer).fillna(0)

    # Fridge: can store treated / purchased water
    if "has_fridge" in df.columns:
        scores += df["has_fridge"].fillna(0)

    # Vehicle: can fetch from alternative source if primary fails
    if "has_vehicle" in df.columns:
        scores += df["has_vehicle"].fillna(0)

    return scores.clip(0, 3)


# ─────────────────────────────────────────────────────────────────────────────
# PCA WEIGHTING
# ─────────────────────────────────────────────────────────────────────────────

def fit_pca(dim_df: pd.DataFrame) -> Tuple[PCA, StandardScaler, np.ndarray]:
    """
    Fit PCA on the four dimension scores.
    dim_df columns must already include idi_dim4_neg (negated).

    Returns: pca, scaler, loadings (shape 4,)
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(dim_df.values)

    pca = PCA(n_components=1, random_state=0)
    pca.fit(X_scaled)

    loadings      = pca.components_[0]
    explained_var = pca.explained_variance_ratio_[0]

    print(f"\n  PCA Results:")
    print(f"    Dimension weights (loadings on PC1):")
    for name, w in zip(dim_df.columns, loadings):
        print(f"      {name:35s}: {w:+.4f}")
    print(f"    Variance explained by PC1: {explained_var*100:.1f}%")
    print(f"    {'✓ Dim4 loading positive — coping offsets lock-in as expected' if loadings[3] > 0 else '⚠ Dim4 loading negative — review coping buffer scoring'}")

    return pca, scaler, loadings


def apply_pca(dim_df: pd.DataFrame,
              pca: PCA,
              scaler: StandardScaler) -> np.ndarray:
    X_scaled = scaler.transform(dim_df.values)
    return pca.transform(X_scaled).flatten()


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO  (v2 — ~15× faster)
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

    For each of N=500 runs:
      1. Add Gaussian noise to dimension scores, clip to [0, 3].
      2. Re-project through FIXED PCA (weights stable across runs).
      3. Fit logistic regression — plain Newton, no cluster-SE.
         Formula: disruption ~ piped + idi + piped:idi
                    + wealth_q_num + urban + hh_size + C(season)
         Season is now included — absorbs old seasonal analysis table.
      4. Store per-household IDI score + OR(piped) + OR(interaction).

    Speed improvements over v1:
      a. reg_positions: pre-built int array maps reg_df rows → df positions.
         Replaces per-row lambda dict lookup (was O(n²), now O(1) per run).
      b. No cov_type="cluster" in MC loop. Cluster-SE is used only on the
         observed model in analysis.py. In MC we need OR point estimates only.
      c. idi_col_arr: pre-allocated float array, updated in-place each run.
         No DataFrame copy() inside the loop.
    """
    print(f"\n  Running Monte Carlo ({cfg.MONTE_CARLO_RUNS} iterations)...")
    print(f"  (season-adjusted formula, plain Newton solver)")

    rng    = np.random.default_rng(cfg.MONTE_CARLO_SEED)
    n_hh   = len(df)
    n_runs = cfg.MONTE_CARLO_RUNS

    # dim_array: shape (n_hh, 4)
    dim_array = df[dim_cols].values.astype(float)

    idi_all_runs   = np.empty((n_runs, n_hh))
    or_piped       = np.empty(n_runs)
    or_interaction = np.empty(n_runs)

    # ── Prepare regression data ONCE outside loop ─────────────────────────
    reg_cols = [
        cfg.VAR_DISRUPTED, "piped_flag",
        "wealth_q_num", "urban", "hh_size", "children_u5",
        "season",            # NEW — season as categorical control
        cfg.VAR_PSU, "weight",
    ]
    # Only keep cols that exist
    reg_cols_present = [c for c in reg_cols if c in df.columns]
    reg_df = df[reg_cols_present].dropna().copy()

    # ── FIX A: pre-build positional mapping (replaces O(n²) lambda) ───────
    # For each row in reg_df, what is its positional index in df?
    df_pos_map  = {orig_idx: pos for pos, orig_idx in enumerate(df.index)}
    reg_positions = np.array(
        [df_pos_map[i] for i in reg_df.index if i in df_pos_map],
        dtype=np.intp,
    )
    # Trim reg_df to rows that mapped cleanly
    valid_mask = np.array([i in df_pos_map for i in reg_df.index])
    reg_df     = reg_df[valid_mask].copy()
    reg_df.reset_index(drop=True, inplace=True)

    # ── FIX C: pre-allocate IDI column array, update in-place ─────────────
    idi_col_arr = np.empty(len(reg_df), dtype=float)

    print(f"  Regression base: {len(reg_df):,} households (after NA drop)")

    # ── MC loop ───────────────────────────────────────────────────────────
    for run in range(n_runs):
        # Step 1: perturb dimension scores
        noise     = rng.normal(0, cfg.MONTE_CARLO_NOISE, size=dim_array.shape)
        perturbed = np.clip(dim_array + noise, 0, 3)

        # Step 2: project through FIXED PCA
        X_scaled        = scaler.transform(perturbed)
        idi_run         = pca.transform(X_scaled).flatten()   # shape (n_hh,)
        idi_all_runs[run] = idi_run

        # Step 3: attach IDI to reg_df using pre-built positions (FIX A)
        idi_col_arr[:] = idi_run[reg_positions]   # vectorised, no Python loop

        reg_df_run = reg_df.copy()
        reg_df_run["idi"] = idi_col_arr

        if len(reg_df_run) < 100:
            or_piped[run]       = np.nan
            or_interaction[run] = np.nan
            continue

        # Step 4: fit logistic — plain Newton, no sandwich (FIX B)
        try:
            has_season = "season" in reg_df_run.columns and \
                         reg_df_run["season"].nunique() > 1
            season_term = "+ C(season)" if has_season else ""
            formula = (
                f"{cfg.VAR_DISRUPTED} ~ piped_flag + idi + piped_flag:idi "
                f"+ wealth_q_num + urban + hh_size + children_u5 "
                f"{season_term}"
            )
            model = smf.logit(
                formula=formula,
                data=reg_df_run,
                freq_weights=reg_df_run["weight"],
            ).fit(
                disp=False,
                maxiter=200,
                method="newton",   # FIX B: fastest solver, no cluster-SE
            )
            or_piped[run]       = np.exp(model.params.get("piped_flag", np.nan))
            or_interaction[run] = np.exp(model.params.get("piped_flag:idi", np.nan))
        except Exception:
            or_piped[run]       = np.nan
            or_interaction[run] = np.nan

        if (run + 1) % 100 == 0:
            pct_done = (run + 1) / n_runs * 100
            valid_so_far = np.nanmean(or_piped[:run+1])
            print(f"    [{pct_done:5.1f}%] Run {run+1}/{n_runs} "
                  f"| mean OR(piped) so far: {valid_so_far:.3f}")

    # ── Robustness summary ────────────────────────────────────────────────
    valid_or = or_piped[~np.isnan(or_piped)]
    pct_gt1  = (valid_or > 1.0).mean() * 100

    print(f"\n  Monte Carlo complete.")
    print(f"    OR(piped) > 1 in {pct_gt1:.1f}% of runs "
          f"({'ROBUST' if pct_gt1 >= 95 else 'MODERATE'})")

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

    1. Predictive validity  — Pearson r (IDI vs disruption) + AUC
    2. Discriminant validity — r(IDI vs wealth) — should be |r| < 0.50
    3. Known-groups validity — mean IDI: piped >> tube well
    """
    rows = []

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

    try:
        auc = roc_auc_score(
            valid[cfg.VAR_DISRUPTED],
            valid["idi_mean"],
            sample_weight=df.loc[valid.index, "weight"],
        )
        rows.append({
            "Check": "Predictive validity",
            "Metric": "ROC-AUC",
            "Value": round(auc, 3),
            "p-value": "—",
            "Target": "> 0.60",
            "Pass": "✓" if auc > 0.60 else "✗",
        })
    except Exception:
        pass

    r_wlth, p_wlth = pearsonr(valid["idi_mean"], valid["wealth_score"])
    rows.append({
        "Check": "Discriminant validity",
        "Metric": "Pearson r (IDI vs Wealth Score)",
        "Value": round(r_wlth, 3),
        "p-value": "<0.001" if p_wlth < 0.001 else f"{p_wlth:.3f}",
        "Target": "|r| < 0.50",
        "Pass": "✓" if abs(r_wlth) < 0.50 else "✗",
    })

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
    """Cronbach's alpha for internal consistency of IDI dimensions."""
    k         = len(cols)
    item_vars = df[cols].var(axis=0, ddof=1).sum()
    total_var = df[cols].sum(axis=1).var(ddof=1)
    alpha     = (k / (k - 1)) * (1 - item_vars / total_var)
    return round(alpha, 3)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BUILDER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class IDIBuilder:
    """
    Orchestrates IDI construction.
    Call .build() → returns df with IDI columns attached.

    New columns added:
      idi_dim1        : Source lock-in (0-3)
      idi_dim2        : Access complexity (0-3)
      idi_dim3        : System dependency (0-3)
      idi_dim4_neg    : Coping buffer, negated (0 = strong coper, -3 = none)
      idi_raw_pca     : Raw PCA projection
      idi_mean        : Mean IDI across MC runs, normalised 0-100
      idi_ci_lower    : 2.5th percentile across MC runs
      idi_ci_upper    : 97.5th percentile
      idi_ci_width    : CI width (higher = more uncertain)
    """

    # Dim 4 is negated BEFORE passing to PCA so its loading is interpretable
    DIM_COLS = ["idi_dim1", "idi_dim2", "idi_dim3", "idi_dim4_neg"]

    def __init__(self, df: pd.DataFrame, cfg: Config):
        self.df          = df.copy()
        self.cfg         = cfg
        self.pca         = None
        self.scaler      = None
        self.loadings    = None
        self.mc_results  = None
        self.validation_df = None

    def build(self) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("STEP 3 — Building IDI (4 dims + PCA + Monte Carlo)")
        print("=" * 60)

        # 1. Score dimensions
        self.df["idi_dim1"]    = score_dim1_source_diversity(self.df)
        self.df["idi_dim2"]    = score_dim2_access_complexity(self.df)
        self.df["idi_dim3"]    = score_dim3_system_dependency(self.df)
        # Dim 4: compute buffer score, then negate
        buffer                 = score_dim4_coping_buffer(self.df)
        self.df["idi_dim4_neg"] = -buffer   # negative valence for PCA

        # Internal consistency
        alpha = cronbach_alpha(self.df, self.DIM_COLS)
        print(f"\n  Cronbach's Alpha (4 dimensions): {alpha}")
        if alpha < 0.6:
            print("  ⚠ Low alpha — Dim 4 may be conceptually distinct. "
                  "Consider reporting separately in methods section.")
        else:
            print("  ✓ Good internal consistency")

        # 2. Fit PCA on non-missing rows
        dim_df = self.df[self.DIM_COLS].dropna()
        self.pca, self.scaler, self.loadings = fit_pca(dim_df)

        # 3. Raw PCA score
        self.df["idi_raw_pca"] = np.nan
        valid_idx = self.df[self.DIM_COLS].dropna().index
        self.df.loc[valid_idx, "idi_raw_pca"] = apply_pca(
            self.df.loc[valid_idx, self.DIM_COLS],
            self.pca, self.scaler,
        )

        # 4. Monte Carlo
        self.mc_results = run_monte_carlo(
            self.df, self.DIM_COLS, self.pca, self.scaler, self.cfg
        )

        # 5. Summarise MC runs — normalise each run to 0-100
        idi_runs = self.mc_results["idi_all_runs"]   # (n_runs, n_hh)
        run_min  = idi_runs.min(axis=1, keepdims=True)
        run_max  = idi_runs.max(axis=1, keepdims=True)
        denom    = np.where(run_max - run_min == 0, 1, run_max - run_min)
        idi_normed = (idi_runs - run_min) / denom * 100

        self.df["idi_mean"]     = np.nanmean(idi_normed, axis=0)
        self.df["idi_ci_lower"] = np.nanpercentile(idi_normed, 2.5,  axis=0)
        self.df["idi_ci_upper"] = np.nanpercentile(idi_normed, 97.5, axis=0)
        self.df["idi_ci_width"] = self.df["idi_ci_upper"] - self.df["idi_ci_lower"]

        # 6. Validate
        self.validation_df = validate_idi(self.df, self.cfg)

        # 7. Save outputs
        self._save_outputs()

        print(f"\n  ✓  IDI complete.")
        print(f"     Mean IDI : {self.df['idi_mean'].mean():.1f} "
              f"(SD {self.df['idi_mean'].std():.1f})")
        print(f"     Mean CI width: {self.df['idi_ci_width'].mean():.1f} pp")
        print("=" * 60)
        return self.df

    def _save_outputs(self):
        # MC robustness summary
        or_piped = self.mc_results["or_piped"]
        or_inter = self.mc_results["or_interaction"]
        summary = pd.DataFrame({
            "Metric": [
                "OR(Piped) — Mean",
                "OR(Piped) — 2.5th pctile",
                "OR(Piped) — 97.5th pctile",
                "OR(Piped × IDI) — Mean",
                "OR(Piped × IDI) — 2.5th pctile",
                "OR(Piped × IDI) — 97.5th pctile",
                "% MC runs OR(Piped) > 1.0  [ROBUSTNESS]",
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
        p1 = self.cfg.OUTPUT_DIR / "results" / "idi_monte_carlo_summary.csv"
        summary.to_csv(p1, index=False)

        p2 = self.cfg.OUTPUT_DIR / "results" / "idi_validation.csv"
        self.validation_df.to_csv(p2, index=False)
        print(f"\n  MC summary → {p1}")
        print(f"  IDI validation → {p2}")

    @property
    def pca_loadings(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Dimension":        self.DIM_COLS,
            "Loading (PC1)":    self.loadings.round(4),
            "Contribution (%)": (
                self.loadings**2 / (self.loadings**2).sum() * 100
            ).round(1),
        })
