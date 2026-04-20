"""
rgi_index.py
─────────────────────────────────────────────────────────────────────────────
Responsibility: Build the Reliability Gap Index (RGI) at district level,
then merge it back to household level so the multilevel model can use both
IDI (household) and RGI (district) simultaneously.

Pipeline inside this file:
  1. Aggregate household data to district level
  2. Predict expected disruption rate from socioeconomic predictors alone
     (wealth, urban %, improved source coverage)
  3. RGI = Observed disruption rate − Expected disruption rate
  4. Classify each district into one of four typology quadrants
     using median IDI and median RGI as cut-points
  5. Merge district-level RGI back onto household DataFrame

How RGI connects to IDI:
  - IDI is household-level: "how locked in is this household?"
  - RGI is district-level: "how badly is this district's system failing?"
  - Together in the multilevel model:
      disruption ~ IDI + RGI + IDI×RGI + controls
    The interaction term IDI×RGI is the key finding:
    being locked in AND living in a failing district multiplies your risk.
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
# DISTRICT AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_to_district(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Aggregate household-level data to district level.

    Every row in the output = one district.
    All rates are weighted by survey weight.

    Columns produced
    ────────────────
    district_code          : district identifier
    state_name             : majority state in district
    n_households           : unweighted household count
    total_weight           : sum of survey weights
    observed_disruption    : weighted disruption rate (0-100)
    piped_coverage         : % households with piped water (weighted)
    tube_well_coverage     : % households with tube well (weighted)
    improved_coverage      : % households with improved source (weighted)
    mean_wealth_score      : weighted mean wealth score
    pct_urban              : % urban households (weighted)
    mean_idi               : weighted mean IDI score
    mean_idi_ci_lower      : weighted mean of household IDI CI lower bounds
    mean_idi_ci_upper      : weighted mean of household IDI CI upper bounds
    piped_disruption_rate  : disruption rate among piped users only
    tube_well_disruption_rate : disruption rate among tube well users only
    paradox_ratio          : piped_disruption / tube_well_disruption
    """
    print("\n" + "=" * 60)
    print("STEP 4a — Aggregating to district level")
    print("=" * 60)

    required = ["district_code", "weight", cfg.VAR_DISRUPTED,
                "piped_flag", "tube_well_flag", "improved_flag",
                "wealth_score", "urban", "idi_mean"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for district aggregation: {missing}")

    # Drop rows with no district code
    df_d = df.dropna(subset=["district_code", cfg.VAR_DISRUPTED, "weight"]).copy()

    # Ensure numeric
    for col in ["piped_flag", "tube_well_flag", "improved_flag",
                cfg.VAR_DISRUPTED, "urban", "idi_mean"]:
        df_d[col] = pd.to_numeric(df_d[col], errors="coerce")

    def wm(values, weights):
        """Weighted mean, ignoring NaN."""
        mask = ~(np.isnan(values) | np.isnan(weights))
        if mask.sum() == 0:
            return np.nan
        return np.average(values[mask], weights=weights[mask])

    records = []
    for dist_code, grp in df_d.groupby("district_code"):
        if len(grp) < cfg.MIN_DISTRICT_N:
            continue   # skip districts with too few households

        w = grp["weight"].values
        d = grp[cfg.VAR_DISRUPTED].values

        # Piped-user subgroup
        piped_grp    = grp[grp["piped_flag"] == 1]
        tw_grp       = grp[grp["tube_well_flag"] == 1]

        piped_dr = (
            wm(piped_grp[cfg.VAR_DISRUPTED].values,
               piped_grp["weight"].values) * 100
            if len(piped_grp) >= 10 else np.nan
        )
        tw_dr = (
            wm(tw_grp[cfg.VAR_DISRUPTED].values,
               tw_grp["weight"].values) * 100
            if len(tw_grp) >= 10 else np.nan
        )

        records.append({
            "district_code":           dist_code,
            "state_name":              grp["state_name"].mode()[0]
                                       if "state_name" in grp.columns else "Unknown",
            "region":                  grp["region"].mode()[0]
                                       if "region" in grp.columns else "Unknown",
            "n_households":            len(grp),
            "total_weight":            w.sum(),
            "observed_disruption":     wm(d, w) * 100,
            "piped_coverage":          wm(grp["piped_flag"].values, w) * 100,
            "tube_well_coverage":      wm(grp["tube_well_flag"].values, w) * 100,
            "improved_coverage":       wm(grp["improved_flag"].values, w) * 100,
            "mean_wealth_score":       wm(grp["wealth_score"].fillna(np.nan).values, w),
            "pct_urban":               wm(grp["urban"].values, w) * 100,
            "mean_idi":                wm(grp["idi_mean"].values, w),
            "mean_idi_ci_lower":       wm(grp["idi_ci_lower"].values, w)
                                       if "idi_ci_lower" in grp.columns else np.nan,
            "mean_idi_ci_upper":       wm(grp["idi_ci_upper"].values, w)
                                       if "idi_ci_upper" in grp.columns else np.nan,
            "piped_disruption_rate":   piped_dr,
            "tube_well_disruption_rate": tw_dr,
            "paradox_ratio":           piped_dr / tw_dr
                                       if (tw_dr and tw_dr > 0) else np.nan,
        })

    dist_df = pd.DataFrame(records)
    print(f"  ✓  {len(dist_df)} districts with ≥ {cfg.MIN_DISTRICT_N} households")
    return dist_df


# ─────────────────────────────────────────────────────────────────────────────
# RELIABILITY GAP CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_rgi(dist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Reliability Gap Index (RGI) for each district.

    RGI = Observed disruption rate − Expected disruption rate

    Expected disruption is predicted by a simple OLS regression using
    ONLY socioeconomic/development predictors (not infrastructure type).
    This isolates how much of a district's disruption is explained by
    its poverty/urbanisation alone vs. how much is 'extra' — the gap.

    Predictors for expected disruption
    ────────────────────────────────────
      mean_wealth_score   : economic development proxy
      pct_urban           : urbanisation level
      improved_coverage   : access to any improved source
      (deliberately excludes piped_coverage — that's what we're testing)

    RGI interpretation
    ───────────────────
      RGI > 0  → district disruption WORSE than its development predicts
                 (infrastructure underperforming)
      RGI < 0  → district disruption BETTER than predicted
                 (infrastructure overperforming / resilient systems)
      RGI ≈ 0  → performing as expected

    Severity categories
    ────────────────────
      Severe failure   : RGI > +10 percentage points
      Moderate failure : RGI +5 to +10
      As expected      : RGI −5 to +5
      Outperforming    : RGI < −5
    """
    print("\n  Computing Reliability Gap Index (RGI)...")

    features = ["mean_wealth_score", "pct_urban", "improved_coverage"]
    outcome  = "observed_disruption"

    model_df = dist_df.dropna(subset=features + [outcome]).copy()

    if len(model_df) < 10:
        print("  ⚠  Too few districts for RGI regression. RGI set to NaN.")
        dist_df["expected_disruption"] = np.nan
        dist_df["rgi"]                 = np.nan
        dist_df["rgi_category"]        = "Insufficient Data"
        return dist_df

    scaler = StandardScaler()
    X      = scaler.fit_transform(model_df[features].values)
    y      = model_df[outcome].values

    lr = LinearRegression()
    lr.fit(X, y)

    r2 = lr.score(X, y)
    print(f"    OLS R² (expected disruption model): {r2:.3f}")
    print(f"    Predictors: {features}")

    # Predict for all districts (impute missing features with median)
    X_full = dist_df[features].copy()
    for col in features:
        X_full[col] = X_full[col].fillna(dist_df[col].median())
    X_full_scaled = scaler.transform(X_full.values)

    dist_df["expected_disruption"] = lr.predict(X_full_scaled)
    dist_df["rgi"]                 = (
        dist_df["observed_disruption"] - dist_df["expected_disruption"]
    )

    # Categorise
    def _cat(rgi):
        if pd.isna(rgi):
            return "Insufficient Data"
        if rgi > 10:
            return "Severe Failure"
        if rgi > 5:
            return "Moderate Failure"
        if rgi >= -5:
            return "As Expected"
        return "Outperforming"

    dist_df["rgi_category"] = dist_df["rgi"].apply(_cat)

    # Summary
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
    Classify each district into one of four quadrants using
    median IDI and median RGI as cut-points.

    Quadrant      │ IDI   │ RGI   │ Meaning
    ──────────────┼───────┼───────┼────────────────────────────────────
    CRISIS        │ High  │ High  │ Locked-in households in failing system
                  │       │       │ → Highest risk, priority intervention
    VULNERABLE    │ High  │ Low   │ Locked-in but system holding
                  │       │       │ → Monitor, maintain quality
    RESILIENT POOR│ Low   │ High  │ System failing but people not locked-in
                  │       │       │ → Preserve traditional sources
    SAFE          │ Low   │ Low   │ Diversified AND system working
                  │       │       │ → Study as success model
    """
    med_idi = dist_df["mean_idi"].median()
    med_rgi = dist_df["rgi"].median()

    def _quad(row):
        if pd.isna(row["mean_idi"]) or pd.isna(row["rgi"]):
            return "Unknown"
        high_idi = row["mean_idi"] >= med_idi
        high_rgi = row["rgi"]      >= med_rgi
        if   high_idi and     high_rgi: return "CRISIS"
        elif high_idi and not high_rgi: return "VULNERABLE"
        elif not high_idi and high_rgi: return "RESILIENT POOR"
        else:                           return "SAFE"

    dist_df["typology"] = dist_df.apply(_quad, axis=1)

    print("\n    District typology (IDI × RGI quadrants):")
    for quad, count in dist_df["typology"].value_counts().items():
        print(f"      {quad:20s}: {count} districts")

    return dist_df


# ─────────────────────────────────────────────────────────────────────────────
# MERGE BACK TO HOUSEHOLD LEVEL
# ─────────────────────────────────────────────────────────────────────────────

def merge_rgi_to_households(
    hh_df: pd.DataFrame,
    dist_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge district-level RGI columns back onto household DataFrame.

    Each household inherits:
      rgi          : district reliability gap
      rgi_category : district severity category
      typology     : district quadrant (CRISIS / VULNERABLE / etc.)

    This is what allows the multilevel model to run at household level
    with both IDI (household) and RGI (district) as predictors.
    """
    merge_cols = ["district_code", "rgi", "rgi_category", "typology",
                  "observed_disruption", "expected_disruption",
                  "piped_disruption_rate", "tube_well_disruption_rate",
                  "paradox_ratio"]

    dist_slim = dist_df[merge_cols].copy()
    merged    = hh_df.merge(dist_slim, on="district_code", how="left")

    matched = merged["rgi"].notna().sum()
    print(f"\n  ✓  RGI merged: {matched:,} / {len(merged):,} households matched "
          f"({matched/len(merged)*100:.1f}%)")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BUILDER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class RGIBuilder:
    """
    Orchestrates RGI construction.
    Call .build() → returns (hh_df_with_rgi, district_df).

    district_df is also saved to CSV for map-ready export.
    """

    def __init__(self, hh_df: pd.DataFrame, cfg: Config):
        self.hh_df  = hh_df.copy()
        self.cfg    = cfg
        self.dist_df = None

    def build(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print("\n" + "=" * 60)
        print("STEP 4 — Building RGI (district-level reliability gap)")
        print("=" * 60)

        # 1. Aggregate to district
        self.dist_df = aggregate_to_district(self.hh_df, self.cfg)

        # 2. Compute RGI
        self.dist_df = compute_rgi(self.dist_df)

        # 3. Classify into typology quadrants
        self.dist_df = classify_typology(self.dist_df)

        # 4. Merge back to household level
        hh_with_rgi = merge_rgi_to_households(self.hh_df, self.dist_df)

        # 5. Save district summary
        self._save_district_csv()

        print("\n  ✓  RGI complete")
        print("=" * 60)

        return hh_with_rgi, self.dist_df

    def _save_district_csv(self):
        """Save district-level summary for GIS mapping."""
        out_path = self.cfg.OUTPUT_DIR / "tables" / "district_rgi_summary.csv"
        save_cols = [
            "district_code", "state_name", "region",
            "n_households", "observed_disruption", "expected_disruption", "rgi",
            "rgi_category", "typology",
            "piped_coverage", "tube_well_coverage",
            "mean_idi", "mean_idi_ci_lower", "mean_idi_ci_upper",
            "piped_disruption_rate", "tube_well_disruption_rate", "paradox_ratio",
            "mean_wealth_score", "pct_urban",
        ]
        save_cols_present = [c for c in save_cols if c in self.dist_df.columns]
        self.dist_df[save_cols_present].round(2).to_csv(out_path, index=False)
        print(f"\n  District RGI table saved → {out_path}")
        print(f"  (Use this CSV in QGIS / GeoPandas for the choropleth map)")

    @property
    def top_crisis_districts(self) -> pd.DataFrame:
        """Top 20 CRISIS districts ranked by RGI (worst first)."""
        crisis = self.dist_df[self.dist_df["typology"] == "CRISIS"].copy()
        return (
            crisis.sort_values("rgi", ascending=False)
            [["district_code", "state_name", "rgi", "observed_disruption",
              "piped_coverage", "mean_idi", "n_households"]]
            .head(20)
            .round(1)
        )

    @property
    def top_safe_districts(self) -> pd.DataFrame:
        """Top 20 SAFE districts ranked by RGI (most outperforming first)."""
        safe = self.dist_df[self.dist_df["typology"] == "SAFE"].copy()
        return (
            safe.sort_values("rgi", ascending=True)
            [["district_code", "state_name", "rgi", "observed_disruption",
              "piped_coverage", "mean_idi", "n_households"]]
            .head(20)
            .round(1)
        )
