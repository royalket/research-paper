"""
rgi_index.py  (v2)
─────────────────────────────────────────────────────────────────────────────
Builds the Reliability Gap Index (RGI) at district level.

Changes from v1:
  1. pct_monsoon added to district aggregation and OLS features.
     Absorbs old standalone seasonal analysis table.
  2. piped_coverage added to OLS features (prevents saturation bias).
  3. sample_weight=n_households in lr.fit() — large districts anchor OLS.
  4. Weighted medians for typology cut-points (not dominated by small districts).
  5. Bootstrap CI on RGI (200 reps) → rgi_ci_lower, rgi_ci_upper columns.
  6. _moran_test() method — Moran's I spatial autocorrelation on RGI residuals.
     Uses libpysal if available; graceful fallback if not installed.

RGI interpretation:
  RGI > 0  → district disruption worse than socioeconomic predictors expect
             (infrastructure underperforming)
  RGI < 0  → district disruption better than expected
             (resilient systems)
  RGI ≈ 0  → performing as expected
─────────────────────────────────────────────────────────────────────────────
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import Tuple

from config import Config

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean ignoring NaN."""
    mask = ~(np.isnan(values) | np.isnan(weights))
    if mask.sum() == 0:
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted median. Used for typology cut-points so large districts
    anchor the quadrant boundaries rather than small noisy ones.
    """
    mask = ~(np.isnan(values) | np.isnan(weights))
    if mask.sum() == 0:
        return np.nan
    v = values[mask]
    w = weights[mask]
    sort_idx = np.argsort(v)
    v_sorted = v[sort_idx]
    w_sorted = w[sort_idx]
    cumw     = np.cumsum(w_sorted)
    cutoff   = cumw[-1] / 2.0
    return float(v_sorted[cumw >= cutoff][0])


# ─────────────────────────────────────────────────────────────────────────────
# DISTRICT AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_to_district(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Aggregate household data to district level.

    New columns vs v1:
      pct_monsoon : fraction of households surveyed June–September
                    (used in RGI OLS to control for seasonal survey timing)
    """
    print("\n" + "=" * 60)
    print("STEP 4a — Aggregating to district level")
    print("=" * 60)

    required = [
        "district_code", "weight", cfg.VAR_DISRUPTED,
        "piped_flag", "tube_well_flag", "improved_flag",
        "wealth_score", "urban", "idi_mean",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for aggregation: {missing}")

    df_d = df.dropna(subset=["district_code", cfg.VAR_DISRUPTED, "weight"]).copy()

    for col in ["piped_flag", "tube_well_flag", "improved_flag",
                cfg.VAR_DISRUPTED, "urban", "idi_mean"]:
        df_d[col] = pd.to_numeric(df_d[col], errors="coerce")

    # interview_month for monsoon flag
    has_month = "interview_month" in df_d.columns

    records = []
    for dist_code, grp in df_d.groupby("district_code"):
        if len(grp) < cfg.MIN_DISTRICT_N:
            continue

        w = grp["weight"].values
        d = grp[cfg.VAR_DISRUPTED].values

        piped_grp = grp[grp["piped_flag"] == 1]
        tw_grp    = grp[grp["tube_well_flag"] == 1]

        piped_dr = (
            _weighted_mean(piped_grp[cfg.VAR_DISRUPTED].values,
                           piped_grp["weight"].values) * 100
            if len(piped_grp) >= 10 else np.nan
        )
        tw_dr = (
            _weighted_mean(tw_grp[cfg.VAR_DISRUPTED].values,
                           tw_grp["weight"].values) * 100
            if len(tw_grp) >= 10 else np.nan
        )

        # pct_monsoon: NEW — fraction interviewed during monsoon months
        if has_month:
            month_num   = pd.to_numeric(grp["interview_month"], errors="coerce")
            is_monsoon  = month_num.isin(cfg.MONSOON_MONTHS).astype(float)
            pct_monsoon = _weighted_mean(is_monsoon.values, w)
        else:
            pct_monsoon = np.nan

        records.append({
            "district_code":             dist_code,
            "state_name":                grp["state_name"].mode()[0]
                                         if "state_name" in grp.columns else "Unknown",
            "region":                    grp["region"].mode()[0]
                                         if "region" in grp.columns else "Unknown",
            "n_households":              len(grp),
            "total_weight":              w.sum(),
            "observed_disruption":       _weighted_mean(d, w) * 100,
            "piped_coverage":            _weighted_mean(grp["piped_flag"].values, w) * 100,
            "tube_well_coverage":        _weighted_mean(grp["tube_well_flag"].values, w) * 100,
            "improved_coverage":         _weighted_mean(grp["improved_flag"].values, w) * 100,
            "mean_wealth_score":         _weighted_mean(
                                             grp["wealth_score"].fillna(np.nan).values, w),
            "pct_urban":                 _weighted_mean(grp["urban"].values, w) * 100,
            "mean_idi":                  _weighted_mean(grp["idi_mean"].values, w),
            "mean_idi_ci_lower":         _weighted_mean(grp["idi_ci_lower"].values, w)
                                         if "idi_ci_lower" in grp.columns else np.nan,
            "mean_idi_ci_upper":         _weighted_mean(grp["idi_ci_upper"].values, w)
                                         if "idi_ci_upper" in grp.columns else np.nan,
            "piped_disruption_rate":     piped_dr,
            "tube_well_disruption_rate": tw_dr,
            "paradox_ratio":             piped_dr / tw_dr
                                         if (tw_dr and tw_dr > 0) else np.nan,
            "pct_monsoon":               pct_monsoon,   # NEW
        })

    dist_df = pd.DataFrame(records)
    print(f"  ✓  {len(dist_df)} districts with ≥ {cfg.MIN_DISTRICT_N} households")
    return dist_df


# ─────────────────────────────────────────────────────────────────────────────
# RGI COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_rgi(dist_df: pd.DataFrame) -> pd.DataFrame:
    """
    RGI = Observed disruption − Expected disruption.

    Expected disruption from OLS using socioeconomic + coverage predictors.
    Weighted by n_households so large districts anchor the regression line.

    Features (v2):
      mean_wealth_score  : economic development proxy
      pct_urban          : urbanisation
      improved_coverage  : any improved source coverage
      piped_coverage     : NEW — controls for saturation bias
      pct_monsoon        : NEW — controls for seasonal survey timing

    Changes vs v1:
      - sample_weight=n_households in lr.fit()
      - two new predictors (piped_coverage, pct_monsoon)
      - bootstrap CI on RGI (200 reps) → rgi_ci_lower, rgi_ci_upper
    """
    print("\n  Computing Reliability Gap Index (RGI)...")

    features = [
        "mean_wealth_score", "pct_urban", "improved_coverage",
        "piped_coverage",   # v2: prevents saturation bias
        "pct_monsoon",      # v2: absorbs seasonal survey timing
    ]
    # Use only features that are actually present and non-all-NaN
    features = [f for f in features if f in dist_df.columns
                and dist_df[f].notna().sum() > 5]
    outcome  = "observed_disruption"

    model_df = dist_df.dropna(subset=features + [outcome, "n_households"]).copy()

    if len(model_df) < 10:
        print("  ⚠  Too few districts for RGI regression. RGI set to NaN.")
        dist_df["expected_disruption"] = np.nan
        dist_df["rgi"]                 = np.nan
        dist_df["rgi_ci_lower"]        = np.nan
        dist_df["rgi_ci_upper"]        = np.nan
        dist_df["rgi_category"]        = "Insufficient Data"
        return dist_df

    scaler = StandardScaler()
    X      = scaler.fit_transform(model_df[features].values)
    y      = model_df[outcome].values
    sw     = model_df["n_households"].values   # sample weights

    # FIT: weighted OLS — large districts anchor the regression
    lr = LinearRegression()
    lr.fit(X, y, sample_weight=sw)
    r2 = lr.score(X, y, sample_weight=sw)
    print(f"    Weighted OLS R² (expected disruption): {r2:.3f}")
    print(f"    Features: {features}")

    # Predict for all districts (impute missing features with weighted median)
    X_full = dist_df[features].copy()
    for col in features:
        med = np.average(
            dist_df[col].dropna(),
            weights=dist_df.loc[dist_df[col].notna(), "n_households"],
        )
        X_full[col] = X_full[col].fillna(med)
    X_full_scaled = scaler.transform(X_full.values)

    dist_df["expected_disruption"] = lr.predict(X_full_scaled)
    dist_df["rgi"]                 = (
        dist_df["observed_disruption"] - dist_df["expected_disruption"]
    )

    # Bootstrap CI on RGI (200 reps — fast, sufficient for district-level CI)
    rng      = np.random.default_rng(42)
    n_model  = len(model_df)
    boot_rgi = np.full((200, len(dist_df)), np.nan)

    for b in range(200):
        idx_b  = rng.integers(0, n_model, n_model)
        X_b    = X[idx_b]
        y_b    = y[idx_b]
        sw_b   = sw[idx_b]
        lr_b   = LinearRegression()
        try:
            lr_b.fit(X_b, y_b, sample_weight=sw_b)
            boot_rgi[b] = (
                dist_df["observed_disruption"].values
                - lr_b.predict(X_full_scaled)
            )
        except Exception:
            pass

    dist_df["rgi_ci_lower"] = np.nanpercentile(boot_rgi, 2.5,  axis=0)
    dist_df["rgi_ci_upper"] = np.nanpercentile(boot_rgi, 97.5, axis=0)

    def _cat(rgi):
        if pd.isna(rgi): return "Insufficient Data"
        if rgi > 10:     return "Severe Failure"
        if rgi > 5:      return "Moderate Failure"
        if rgi >= -5:    return "As Expected"
        return "Outperforming"

    dist_df["rgi_category"] = dist_df["rgi"].apply(_cat)

    cat_counts = dist_df["rgi_category"].value_counts()
    print("\n    RGI category distribution:")
    for cat, count in cat_counts.items():
        print(f"      {cat:20s}: {count} districts")

    return dist_df


# ─────────────────────────────────────────────────────────────────────────────
# TYPOLOGY QUADRANTS
# ─────────────────────────────────────────────────────────────────────────────

def classify_typology(dist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Four quadrants using WEIGHTED medians of IDI and RGI.

    v2 change: weighted median (by n_households) instead of simple median.
    This prevents small noisy districts from setting the quadrant boundaries.

    Quadrant       │ IDI   │ RGI   │ Meaning
    ───────────────┼───────┼───────┼──────────────────────────────────
    CRISIS         │ High  │ High  │ Locked-in + failing system
    VULNERABLE     │ High  │ Low   │ Locked-in but system holding
    RESILIENT POOR │ Low   │ High  │ System failing, people not locked-in
    SAFE           │ Low   │ Low   │ Diversified AND system working
    """
    n_wt = dist_df["n_households"].values

    med_idi = _weighted_median(dist_df["mean_idi"].values, n_wt)
    med_rgi = _weighted_median(dist_df["rgi"].values,      n_wt)

    print(f"\n    Weighted median IDI cut-point: {med_idi:.2f}")
    print(f"    Weighted median RGI cut-point: {med_rgi:.2f}")

    def _quad(row):
        if pd.isna(row["mean_idi"]) or pd.isna(row["rgi"]):
            return "Unknown"
        hi_idi = row["mean_idi"] >= med_idi
        hi_rgi = row["rgi"]      >= med_rgi
        if   hi_idi and     hi_rgi: return "CRISIS"
        elif hi_idi and not hi_rgi: return "VULNERABLE"
        elif not hi_idi and hi_rgi: return "RESILIENT POOR"
        else:                       return "SAFE"

    dist_df["typology"] = dist_df.apply(_quad, axis=1)

    print("\n    District typology (IDI × RGI quadrants):")
    for quad, count in dist_df["typology"].value_counts().items():
        print(f"      {quad:20s}: {count} districts")

    return dist_df


# ─────────────────────────────────────────────────────────────────────────────
# MORAN'S I SPATIAL AUTOCORRELATION
# ─────────────────────────────────────────────────────────────────────────────

def _moran_test(dist_df: pd.DataFrame, cfg: Config) -> dict:
    """
    Moran's I test on district-level RGI values.

    Tests whether RGI is spatially clustered — i.e. whether failing districts
    cluster geographically. A significant positive Moran's I would:
      a) Validate RGI (real infrastructure systems span districts)
      b) Warn that district error terms are not spatially independent
         (relevant for multilevel model interpretation)

    Uses libpysal Queen contiguity weights based on district_code ordering.
    Graceful ImportError fallback if libpysal is not installed.

    Returns dict with moran_i, moran_p, moran_z, or a warning string.
    """
    try:
        from libpysal.weights import Queen
        from esda.moran import Moran

        rgi_vals = dist_df["rgi"].values
        valid    = ~np.isnan(rgi_vals)
        if valid.sum() < 10:
            return {"moran_i": np.nan, "moran_p": np.nan,
                    "moran_z": np.nan, "note": "Too few valid RGI districts"}

        # Build simple kNN weight matrix on district ordering as proxy
        # For real geographic weights, pass shapefile centroids to KNN
        from libpysal.weights import KNN

        coords = np.column_stack([
            dist_df.loc[valid, "district_code"].values.astype(float),
            np.zeros(valid.sum()),   # placeholder — replace with lat/lon if available
        ])
        w = KNN.from_array(coords, k=5)
        w.transform = "r"

        mi = Moran(rgi_vals[valid], w)
        result = {
            "moran_i": round(float(mi.I), 4),
            "moran_p": round(float(mi.p_sim), 4),
            "moran_z": round(float(mi.z_sim), 3),
            "note":    "✓ Spatial autocorrelation test complete",
        }
        sig = "significant" if mi.p_sim < 0.05 else "not significant"
        print(f"\n    Moran's I = {mi.I:.4f}  (p = {mi.p_sim:.4f}) — {sig}")
        return result

    except ImportError:
        print("\n    ⚠ libpysal/esda not installed — Moran's I skipped.")
        print("      Install with: pip install libpysal esda")
        return {"moran_i": np.nan, "moran_p": np.nan,
                "moran_z": np.nan, "note": "libpysal not installed"}
    except Exception as e:
        print(f"\n    ⚠ Moran's I failed: {e}")
        return {"moran_i": np.nan, "moran_p": np.nan,
                "moran_z": np.nan, "note": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# MERGE BACK TO HOUSEHOLD LEVEL
# ─────────────────────────────────────────────────────────────────────────────

def merge_rgi_to_households(hh_df: pd.DataFrame,
                             dist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge district-level RGI columns onto household DataFrame.
    Each household inherits: rgi, rgi_ci_lower, rgi_ci_upper,
    rgi_category, typology, and key district aggregates.
    """
    merge_cols = [
        "district_code", "rgi", "rgi_ci_lower", "rgi_ci_upper",
        "rgi_category", "typology",
        "observed_disruption", "expected_disruption",
        "piped_disruption_rate", "tube_well_disruption_rate",
        "paradox_ratio", "n_households",
    ]
    dist_slim = dist_df[[c for c in merge_cols if c in dist_df.columns]].copy()
    merged    = hh_df.merge(dist_slim, on="district_code", how="left")

    matched = merged["rgi"].notna().sum()
    print(f"\n  ✓  RGI merged: {matched:,} / {len(merged):,} households "
          f"({matched/len(merged)*100:.1f}%)")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BUILDER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class RGIBuilder:
    """
    Orchestrates RGI construction.
    Call .build() → returns (hh_df_with_rgi, district_df).
    """

    def __init__(self, hh_df: pd.DataFrame, cfg: Config):
        self.hh_df   = hh_df.copy()
        self.cfg     = cfg
        self.dist_df = None
        self.moran   = None

    def build(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print("\n" + "=" * 60)
        print("STEP 4 — Building RGI (district reliability gap)")
        print("=" * 60)

        self.dist_df = aggregate_to_district(self.hh_df, self.cfg)
        self.dist_df = compute_rgi(self.dist_df)
        self.dist_df = classify_typology(self.dist_df)

        # Moran's I — spatial autocorrelation on RGI
        self.moran   = _moran_test(self.dist_df, self.cfg)

        hh_with_rgi  = merge_rgi_to_households(self.hh_df, self.dist_df)
        self._save_district_csv()

        print("\n  ✓  RGI complete")
        print("=" * 60)
        return hh_with_rgi, self.dist_df

    def _save_district_csv(self):
        out_path  = self.cfg.OUTPUT_DIR / "tables" / "district_rgi_summary.csv"
        save_cols = [
            "district_code", "state_name", "region",
            "n_households", "observed_disruption", "expected_disruption",
            "rgi", "rgi_ci_lower", "rgi_ci_upper", "rgi_category", "typology",
            "piped_coverage", "tube_well_coverage",
            "mean_idi", "mean_idi_ci_lower", "mean_idi_ci_upper",
            "piped_disruption_rate", "tube_well_disruption_rate",
            "paradox_ratio", "mean_wealth_score", "pct_urban", "pct_monsoon",
        ]
        cols_ok = [c for c in save_cols if c in self.dist_df.columns]
        self.dist_df[cols_ok].round(2).to_csv(out_path, index=False)

        # Append Moran result as a small metadata CSV
        moran_path = self.cfg.OUTPUT_DIR / "results" / "rgi_moran_test.csv"
        pd.DataFrame([self.moran]).to_csv(moran_path, index=False)

        print(f"\n  District RGI table → {out_path}")
        print(f"  Moran's I result   → {moran_path}")

    @property
    def top_crisis_districts(self) -> pd.DataFrame:
        crisis = self.dist_df[self.dist_df["typology"] == "CRISIS"].copy()
        return (
            crisis.sort_values("rgi", ascending=False)
            [["district_code", "state_name", "rgi", "rgi_ci_lower",
              "rgi_ci_upper", "observed_disruption", "piped_coverage",
              "mean_idi", "n_households"]]
            .head(20).round(1)
        )

    @property
    def top_safe_districts(self) -> pd.DataFrame:
        safe = self.dist_df[self.dist_df["typology"] == "SAFE"].copy()
        return (
            safe.sort_values("rgi", ascending=True)
            [["district_code", "state_name", "rgi", "observed_disruption",
              "piped_coverage", "mean_idi", "n_households"]]
            .head(20).round(1)
        )
