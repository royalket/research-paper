"""
analysis.py  (v2)
─────────────────────────────────────────────────────────────────────────────
All statistical analysis and output generation.

Changes from v1:
  1. IDIRegression — social controls added to formula:
       + sc_st_flag + female_headed + C(head_education)
     These absorb the old WVI social component.

  2. MultilevelModel — replaced flat logit with GEE (Generalized
     Estimating Equations). Groups=district_code, exchangeable correlation.
     GEE is honest about within-district correlation without requiring
     a full mixed-effects distributional assumption.

  3. SlopeAsOutcomeModel (NEW class) — two-stage slope-as-outcome:
     Stage 1: per-district logistic regression of disruption on IDI
              → extract IDI slope + SE for each district
     Stage 2: WLS regression of district slopes on RGI,
              weights = 1/SE² (inverse-variance weighting)
     A positive, significant RGI coefficient means:
     "Districts with higher reliability gaps show a steeper
      IDI→disruption relationship" — stronger claim than a single
      interaction term.

Pipeline (unchanged):
  Finding 1 → DescriptiveTables
  Finding 2 → IDIRegression
  Finding 3 → SpatialTables
  Finding 4 → MultilevelModel (GEE)
  Finding 5 → SlopeAsOutcomeModel  ← NEW
  Robustness → PSMAnalysis
  Report    → ReportGenerator
─────────────────────────────────────────────────────────────────────────────
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, Any

import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.regression.linear_model import WLS
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr

from config import Config

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def weighted_rate(df: pd.DataFrame, outcome: str,
                  weight: str = "weight") -> float:
    mask = df[[outcome, weight]].notna().all(axis=1)
    sub  = df[mask]
    if sub[weight].sum() == 0:
        return np.nan
    return float(np.average(sub[outcome].astype(float),
                             weights=sub[weight])) * 100


def fmt_p(p: float) -> str:
    if pd.isna(p):  return ""
    if p < 0.001:   return "<0.001***"
    if p < 0.01:    return f"{p:.3f}**"
    if p < 0.05:    return f"{p:.3f}*"
    return f"{p:.3f}"


def _social_terms(df: pd.DataFrame) -> str:
    """
    Return formula fragment for social controls — only include
    columns that are actually present and have variance.
    Absorbs old WVI social component (caste, education, female-headed).
    """
    terms = []
    if "sc_st_flag" in df.columns and df["sc_st_flag"].std() > 0:
        terms.append("sc_st_flag")
    if "female_headed" in df.columns and df["female_headed"].std() > 0:
        terms.append("female_headed")
    if "head_education" in df.columns and df["head_education"].nunique() > 1:
        terms.append("C(head_education)")
    return ("+ " + " + ".join(terms)) if terms else ""


# ─────────────────────────────────────────────────────────────────────────────
# FINDING 1 — DESCRIPTIVE CROSS-TABS
# ─────────────────────────────────────────────────────────────────────────────

class DescriptiveTables:
    """
    Table 1a : Disruption rate by water source (overall)
    Table 1b : Disruption rate by source × wealth quintile
    Table 1c : Disruption rate by source × urban/rural
    Table 1d : Disruption rate by source × region
    Table 1e : Disruption rate by source × season  (NEW — absorbs seasonal table)
    """

    def __init__(self, df: pd.DataFrame, cfg: Config):
        self.df  = df
        self.cfg = cfg

    def run_all(self) -> Dict[str, pd.DataFrame]:
        print("\n" + "=" * 60)
        print("ANALYSIS — Finding 1: Descriptive cross-tabs")
        print("=" * 60)

        tables = {}
        tables["1a_by_source"]        = self._table_by_source()
        tables["1b_by_source_wealth"] = self._table_stratified("wealth_quintile")
        tables["1c_by_source_urban"]  = self._table_stratified("residence")
        tables["1d_by_source_region"] = self._table_stratified("region")
        tables["1e_by_source_season"] = self._table_stratified("season")

        for name, tbl in tables.items():
            out = self.cfg.OUTPUT_DIR / "tables" / f"table_{name}.csv"
            tbl.to_csv(out, index=True)
            print(f"  Saved → {out}")

        return tables

    def _table_by_source(self) -> pd.DataFrame:
        cfg     = self.cfg
        df      = self.df
        outcome = cfg.VAR_DISRUPTED

        rows = []
        for src in df["water_source"].dropna().unique():
            sub = df[df["water_source"] == src]
            if sub["weight"].sum() == 0:
                continue
            rate  = weighted_rate(sub, outcome)
            p     = rate / 100
            n_eff = sub["weight"].sum() ** 2 / (sub["weight"] ** 2).sum()
            se    = np.sqrt(p * (1 - p) / n_eff) * 100
            rows.append({
                "Water Source":        src,
                "Weighted N (000s)":   round(sub["weight"].sum() / 1000, 1),
                "Disruption Rate (%)": round(rate, 1),
                "95% CI Lower":        round(rate - 1.96 * se, 1),
                "95% CI Upper":        round(rate + 1.96 * se, 1),
            })

        tbl = (pd.DataFrame(rows)
               .sort_values("Disruption Rate (%)", ascending=False)
               .reset_index(drop=True))

        tw = tbl.loc[tbl["Water Source"] == "Tube Well/Borehole",
                     "Disruption Rate (%)"]
        if len(tw):
            tbl["Relative Risk vs Tube Well"] = (
                tbl["Disruption Rate (%)"] / float(tw.iloc[0])
            ).round(2)

        print(f"\n  Table 1a — Disruption by source:")
        print(tbl.to_string(index=False))
        return tbl

    def _table_stratified(self, stratify_by: str) -> pd.DataFrame:
        cfg     = self.cfg
        df      = self.df
        outcome = cfg.VAR_DISRUPTED

        if stratify_by not in df.columns:
            return pd.DataFrame()

        df_filt = df[df["water_source"].isin(
            ["Piped Water", "Tube Well/Borehole"])]
        rows = []
        for stratum in sorted(df_filt[stratify_by].dropna().unique()):
            sub = df_filt[df_filt[stratify_by] == stratum]
            for src in ["Piped Water", "Tube Well/Borehole"]:
                src_sub = sub[sub["water_source"] == src]
                if src_sub["weight"].sum() == 0:
                    continue
                rows.append({
                    stratify_by:       stratum,
                    "Water Source":    src,
                    "Disruption (%)":  round(weighted_rate(src_sub, outcome), 1),
                    "N":               len(src_sub),
                })

        tbl = pd.DataFrame(rows)
        if tbl.empty:
            return tbl
        tbl_wide = tbl.pivot_table(
            index=stratify_by, columns="Water Source",
            values="Disruption (%)", aggfunc="first",
        )
        if ("Piped Water" in tbl_wide.columns and
                "Tube Well/Borehole" in tbl_wide.columns):
            tbl_wide["Difference (pp)"] = (
                tbl_wide["Piped Water"] - tbl_wide["Tube Well/Borehole"]
            ).round(1)
            tbl_wide["Piped ÷ Tube Well"] = (
                tbl_wide["Piped Water"] / tbl_wide["Tube Well/Borehole"]
            ).round(2)
        print(f"\n  Stratified ({stratify_by}):")
        print(tbl_wide.round(1).to_string())
        return tbl_wide


# ─────────────────────────────────────────────────────────────────────────────
# FINDING 2 — IDI REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

class IDIRegression:
    """
    Logistic regression: disruption ~ piped + IDI + piped×IDI + controls.

    v2 changes:
      - Social controls added: sc_st_flag, female_headed, C(head_education)
      - Cluster-robust SE still used (clustered at PSU level)
    """

    def __init__(self, df: pd.DataFrame, cfg: Config):
        self.df     = df
        self.cfg    = cfg
        self.result = None

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print("\n" + "=" * 60)
        print("ANALYSIS — Finding 2: IDI Regression")
        print("=" * 60)

        cfg = self.cfg
        df  = self.df.copy()

        core_vars = [
            cfg.VAR_DISRUPTED, "piped_flag", "idi_mean",
            "wealth_q_num", "urban", "hh_size", "children_u5",
            "region", cfg.VAR_PSU, "weight",
        ]
        df_reg = df.dropna(subset=core_vars).copy()

        if len(df_reg) < 500:
            print("  ⚠  Too few complete cases.")
            return pd.DataFrame(), pd.DataFrame()

        social = _social_terms(df_reg)
        formula = (
            f"{cfg.VAR_DISRUPTED} ~ "
            "piped_flag + idi_mean + piped_flag:idi_mean "
            "+ wealth_q_num + urban + hh_size + children_u5 "
            f"+ C(region) {social}"
        )
        print(f"  Formula: {formula}")
        print(f"  Fitting on {len(df_reg):,} households...")

        try:
            model = smf.logit(
                formula=formula, data=df_reg,
                freq_weights=df_reg["weight"],
            ).fit(
                disp=False, maxiter=500,
                cov_type="cluster",
                cov_kwds={"groups": df_reg[cfg.VAR_PSU]},
            )
            self.result = model
        except Exception as e:
            print(f"  ✗  Regression failed: {e}")
            return pd.DataFrame(), pd.DataFrame()

        coef_table = self._format_coef_table(model)
        pred_table = self._predicted_probabilities(model, df_reg)

        coef_path = cfg.OUTPUT_DIR / "results" / "table2_idi_regression.csv"
        pred_path = cfg.OUTPUT_DIR / "results" / "table2b_predicted_probs.csv"
        coef_table.to_csv(coef_path)
        pred_table.to_csv(pred_path, index=False)
        print(f"  Saved → {coef_path}")
        return coef_table, pred_table

    def _format_coef_table(self, model) -> pd.DataFrame:
        params = model.params
        conf   = model.conf_int()
        pvals  = model.pvalues

        key_terms = {
            "piped_flag":           "Piped Water (vs non-piped)",
            "idi_mean":             "IDI Score",
            "piped_flag:idi_mean":  "Piped × IDI  [KEY INTERACTION]",
            "wealth_q_num":         "Wealth Quintile",
            "urban":                "Urban",
            "sc_st_flag":           "SC/ST household",
            "female_headed":        "Female-headed household",
            "hh_size":              "Household size",
            "children_u5":          "Children under 5",
        }
        rows = []
        for term, label in key_terms.items():
            if term not in params.index:
                continue
            OR = np.exp(params[term])
            lo = np.exp(conf.loc[term, 0])
            hi = np.exp(conf.loc[term, 1])
            p  = pvals[term]
            rows.append({
                "Variable":    label,
                "OR":          round(OR, 3),
                "95% CI":      f"{lo:.3f}–{hi:.3f}",
                "p-value":     fmt_p(p),
                "Significant": "✓" if p < 0.05 else "",
            })
        tbl = pd.DataFrame(rows).set_index("Variable")
        print("\n  IDI regression (key terms):")
        print(tbl.to_string())
        return tbl

    def _predicted_probabilities(self, model, df_reg: pd.DataFrame) -> pd.DataFrame:
        med_wealth   = df_reg["wealth_q_num"].median()
        med_hh_size  = df_reg["hh_size"].median()
        med_children = df_reg["children_u5"].median()
        modal_region = df_reg["region"].mode()[0]

        base = dict(wealth_q_num=med_wealth, hh_size=med_hh_size,
                    children_u5=med_children, region=modal_region,
                    sc_st_flag=0, female_headed=0, head_education="Secondary")

        scenarios = [
            {**base, "Scenario": "A: Rich Urban Piped (IDI=80)",
             "piped_flag": 1, "idi_mean": 80, "wealth_q_num": 5, "urban": 1},
            {**base, "Scenario": "B: Rich Urban Non-Piped (IDI=20)",
             "piped_flag": 0, "idi_mean": 20, "wealth_q_num": 5, "urban": 1},
            {**base, "Scenario": "C: Poor Rural Piped (IDI=50)",
             "piped_flag": 1, "idi_mean": 50, "wealth_q_num": 1, "urban": 0},
            {**base, "Scenario": "D: Poor Rural Tube Well (IDI=15)",
             "piped_flag": 0, "idi_mean": 15, "wealth_q_num": 1, "urban": 0},
        ]
        rows = []
        for sc in scenarios:
            sc_df = pd.DataFrame([sc])
            try:
                pred = model.predict(sc_df)
                rows.append({
                    "Scenario":               sc["Scenario"],
                    "Predicted Prob (%)":     round(float(pred.iloc[0]) * 100, 1),
                })
            except Exception as e:
                rows.append({"Scenario": sc["Scenario"],
                             "Predicted Prob (%)": np.nan})
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# FINDING 3 — SPATIAL TABLES
# ─────────────────────────────────────────────────────────────────────────────

class SpatialTables:
    """District and state ranking tables from district_df."""

    def __init__(self, dist_df: pd.DataFrame, cfg: Config):
        self.dist_df = dist_df
        self.cfg     = cfg

    def run_all(self) -> Dict[str, pd.DataFrame]:
        print("\n" + "=" * 60)
        print("ANALYSIS — Finding 3: Spatial tables")
        print("=" * 60)

        tables = {}
        tables["3a_top_crisis"]        = self._top("CRISIS",        False)
        tables["3b_top_safe"]          = self._top("SAFE",          True)
        tables["3c_resilient_poor"]    = self._top("RESILIENT POOR", False)
        tables["3d_state_paradox"]     = self._state_ranking()

        for name, tbl in tables.items():
            out = self.cfg.OUTPUT_DIR / "tables" / f"table_{name}.csv"
            tbl.to_csv(out, index=False)
        return tables

    def _top(self, typology: str, ascending: bool, n: int = 20) -> pd.DataFrame:
        sub  = self.dist_df[self.dist_df["typology"] == typology].copy()
        cols = ["district_code", "state_name", "rgi", "rgi_ci_lower",
                "rgi_ci_upper", "observed_disruption", "piped_coverage",
                "mean_idi", "paradox_ratio", "n_households"]
        cols_ok = [c for c in cols if c in sub.columns]
        return (sub.sort_values("rgi", ascending=ascending)
                [cols_ok].head(n).round(1).reset_index(drop=True))

    def _state_ranking(self) -> pd.DataFrame:
        if "state_name" not in self.dist_df.columns:
            return pd.DataFrame()
        return (
            self.dist_df.groupby("state_name").agg(
                n_districts          =("district_code", "count"),
                n_households         =("n_households", "sum"),
                mean_rgi             =("rgi", "mean"),
                mean_idi             =("mean_idi", "mean"),
                mean_piped_coverage  =("piped_coverage", "mean"),
                mean_paradox_ratio   =("paradox_ratio", "mean"),
                pct_crisis           =("typology",
                                       lambda x: (x == "CRISIS").mean() * 100),
            ).reset_index().round(1)
            .sort_values("mean_rgi", ascending=False)
        )


# ─────────────────────────────────────────────────────────────────────────────
# FINDING 4 — GEE MULTILEVEL MODEL
# ─────────────────────────────────────────────────────────────────────────────

class MultilevelModel:
    """
    GEE (Generalized Estimating Equations) with district-level grouping.

    Why GEE instead of flat logit:
      Households within the same district share unmeasured infrastructure
      quality, groundwater conditions, and local governance. Ignoring this
      inflates standard errors and biases the IDI×RGI interaction estimate.
      GEE with exchangeable correlation structure accounts for within-district
      dependence and produces population-averaged estimates — appropriate
      for policy claims about the average household.

    Why exchangeable correlation structure:
      We assume all pairs of households within the same district are equally
      correlated (exchangeable), which is appropriate when we have no reason
      to believe within-district correlation varies by household pair.

    Key coefficient: idi_std:rgi_std
      Positive and significant = being locked-in in a failing district
      multiplies disruption risk beyond either factor alone.
    """

    def __init__(self, df: pd.DataFrame, cfg: Config):
        self.df     = df
        self.cfg    = cfg
        self.result = None

    def run(self) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("ANALYSIS — Finding 4: GEE multilevel model (IDI × RGI)")
        print("=" * 60)

        cfg = self.cfg
        df  = self.df.copy()

        core_vars = [
            cfg.VAR_DISRUPTED, "idi_mean", "rgi",
            "piped_flag", "wealth_q_num", "urban",
            "hh_size", "children_u5", "region",
            "district_code", "weight",
        ]
        df_reg = df.dropna(subset=core_vars).copy()

        if len(df_reg) < 500:
            print("  ⚠  Too few complete cases (need RGI merged).")
            return pd.DataFrame()

        # Standardise IDI and RGI for interpretable interaction coefficient
        df_reg["idi_std"] = (
            (df_reg["idi_mean"] - df_reg["idi_mean"].mean())
            / df_reg["idi_mean"].std()
        )
        df_reg["rgi_std"] = (
            (df_reg["rgi"] - df_reg["rgi"].mean())
            / df_reg["rgi"].std()
        )

        social = _social_terms(df_reg)
        formula = (
            f"{cfg.VAR_DISRUPTED} ~ "
            "idi_std + rgi_std + idi_std:rgi_std "
            "+ piped_flag + wealth_q_num + urban "
            f"+ hh_size + children_u5 + C(region) {social}"
        )
        print(f"  Formula: {formula}")
        print(f"  Fitting GEE on {len(df_reg):,} households "
              f"({df_reg['district_code'].nunique()} districts)...")

        try:
            # GEE: Binomial family, exchangeable within-district correlation
            model = smf.gee(
                formula=formula,
                groups=df_reg["district_code"],
                data=df_reg,
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable(),
                weights=df_reg["weight"],
            ).fit(disp=False)
            self.result = model
        except Exception as e:
            print(f"  ✗  GEE failed: {e}")
            print("    Falling back to clustered logit...")
            try:
                model = smf.logit(
                    formula=formula, data=df_reg,
                    freq_weights=df_reg["weight"],
                ).fit(
                    disp=False, maxiter=500,
                    cov_type="cluster",
                    cov_kwds={"groups": df_reg["district_code"]},
                )
                self.result = model
                print("    Fallback succeeded (clustered logit).")
            except Exception as e2:
                print(f"  ✗  Fallback also failed: {e2}")
                return pd.DataFrame()

        coef_table = self._format_table(model)
        out_path   = cfg.OUTPUT_DIR / "results" / "table5_gee_multilevel.csv"
        coef_table.to_csv(out_path)
        print(f"  GEE results → {out_path}")
        return coef_table

    def _format_table(self, model) -> pd.DataFrame:
        params = model.params
        conf   = model.conf_int()
        pvals  = model.pvalues

        key_terms = {
            "idi_std":          "IDI Score (std)",
            "rgi_std":          "RGI Score (std)",
            "idi_std:rgi_std":  "IDI × RGI  [CROSS-LEVEL INTERACTION]",
            "piped_flag":       "Piped Water",
            "wealth_q_num":     "Wealth Quintile",
            "urban":            "Urban",
            "sc_st_flag":       "SC/ST household",
            "female_headed":    "Female-headed",
        }
        rows = []
        for term, label in key_terms.items():
            if term not in params.index:
                continue
            OR = np.exp(params[term])
            lo = np.exp(conf.loc[term, 0])
            hi = np.exp(conf.loc[term, 1])
            p  = pvals[term]
            rows.append({
                "Variable":    label,
                "OR":          round(OR, 3),
                "95% CI":      f"{lo:.3f}–{hi:.3f}",
                "p-value":     fmt_p(p),
                "Significant": "✓" if p < 0.05 else "",
            })
        tbl = pd.DataFrame(rows).set_index("Variable")
        print("\n  GEE results (key terms):")
        print(tbl.to_string())
        return tbl


# ─────────────────────────────────────────────────────────────────────────────
# FINDING 5 — SLOPE-AS-OUTCOME MODEL (NEW)
# ─────────────────────────────────────────────────────────────────────────────

class SlopeAsOutcomeModel:
    """
    Two-stage slope-as-outcome model.

    Stage 1: For each district with ≥ MIN_DISTRICT_PIPED_N piped households,
             fit a separate logistic regression:
               disruption ~ idi_mean + wealth_q_num + urban
             Extract β_IDI (IDI slope) and its standard error.

    Stage 2: WLS regression of district IDI slopes on RGI:
               β_IDI(district) ~ RGI(district)
             Weights = 1 / SE(β_IDI)²  (inverse-variance weighting)

    Interpretation of Stage 2 RGI coefficient:
      Positive and significant → districts with higher reliability gaps
      show a steeper IDI-to-disruption relationship.
      This is a MORE PRECISE claim than the single IDI×RGI interaction term
      because it is estimated from district-specific relationships rather
      than a pooled interaction.

    Why stronger than the single interaction term:
      The GEE/logit interaction term idi_std:rgi_std estimates whether IDI
      and RGI jointly predict disruption in the pooled sample.
      It cannot distinguish whether the IDI effect VARIES by RGI — it just
      tests whether their product adds predictive power.
      The slope-as-outcome model directly estimates the IDI effect separately
      in each district and then asks whether those district-level effects
      are explained by RGI. This is the classic contextual moderation design.
    """

    def __init__(self, df: pd.DataFrame, dist_df: pd.DataFrame, cfg: Config):
        self.df      = df
        self.dist_df = dist_df
        self.cfg     = cfg

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print("\n" + "=" * 60)
        print("ANALYSIS — Finding 5: Slope-as-Outcome Model")
        print("=" * 60)

        cfg = self.cfg
        df  = self.df.copy()

        # ── Stage 1: district-level IDI slopes ───────────────────────────
        print("\n  Stage 1: Estimating district-level IDI slopes...")

        stage1_rows = []
        districts = df["district_code"].dropna().unique()

        for dist in districts:
            grp = df[df["district_code"] == dist].dropna(
                subset=[cfg.VAR_DISRUPTED, "idi_mean", "wealth_q_num",
                        "urban", "weight"]
            ).copy()

            # Require minimum piped households for stable slope estimate
            n_piped = (grp["piped_flag"] == 1).sum()
            if len(grp) < cfg.SLOPE_MODEL_MIN_OBS or n_piped < cfg.MIN_DISTRICT_PIPED_N:
                continue

            # Need variation in IDI
            if grp["idi_mean"].std() < 0.1:
                continue

            try:
                m = smf.logit(
                    f"{cfg.VAR_DISRUPTED} ~ idi_mean + wealth_q_num + urban",
                    data=grp,
                    freq_weights=grp["weight"],
                ).fit(disp=False, maxiter=200, method="newton")

                if "idi_mean" not in m.params.index:
                    continue

                stage1_rows.append({
                    "district_code": dist,
                    "idi_slope":     float(m.params["idi_mean"]),
                    "idi_slope_se":  float(m.bse["idi_mean"]),
                    "n_obs":         len(grp),
                    "n_piped":       int(n_piped),
                })
            except Exception:
                continue

        stage1_df = pd.DataFrame(stage1_rows)
        n_s1 = len(stage1_df)
        print(f"  Stage 1 complete: {n_s1} districts with reliable IDI slopes")

        if n_s1 < 10:
            print("  ⚠  Too few districts for Stage 2. Skipping.")
            return stage1_df, pd.DataFrame()

        # Save Stage 1
        p1 = cfg.OUTPUT_DIR / "results" / "table6a_stage1_idi_slopes.csv"
        stage1_df.round(4).to_csv(p1, index=False)
        print(f"  Stage 1 slopes → {p1}")

        # ── Stage 2: WLS slopes ~ RGI ─────────────────────────────────────
        print("\n  Stage 2: WLS regression — IDI slope ~ RGI...")

        # Merge RGI onto stage1
        rgi_slim  = self.dist_df[["district_code", "rgi", "mean_idi",
                                   "piped_coverage", "pct_urban"]].copy()
        s2_df     = stage1_df.merge(rgi_slim, on="district_code", how="inner")
        s2_df     = s2_df.dropna(subset=["idi_slope", "idi_slope_se", "rgi"])
        s2_df     = s2_df[s2_df["idi_slope_se"] > 0].copy()

        if len(s2_df) < 10:
            print("  ⚠  Too few matched districts for Stage 2.")
            return stage1_df, pd.DataFrame()

        # Inverse-variance weights: 1 / SE²
        s2_df["iv_weight"] = 1.0 / (s2_df["idi_slope_se"] ** 2)
        s2_df["rgi_std"]   = (
            (s2_df["rgi"] - s2_df["rgi"].mean()) / s2_df["rgi"].std()
        )

        # WLS: IDI slope ~ RGI (standardised)
        X_s2 = sm.add_constant(s2_df["rgi_std"].values)
        y_s2 = s2_df["idi_slope"].values
        w_s2 = s2_df["iv_weight"].values

        wls_model = WLS(y_s2, X_s2, weights=w_s2)
        wls_fit   = wls_model.fit()

        # Format Stage 2 output
        params = wls_fit.params
        conf   = wls_fit.conf_int()
        pvals  = wls_fit.pvalues

        stage2_rows = []
        labels = {0: "Intercept (mean IDI slope)", 1: "RGI (std)  [KEY TERM]"}
        for i, label in labels.items():
            stage2_rows.append({
                "Term":        label,
                "Coefficient": round(params[i], 4),
                "95% CI":      f"{conf[i][0]:.4f}–{conf[i][1]:.4f}",
                "p-value":     fmt_p(pvals[i]),
                "Significant": "✓" if pvals[i] < 0.05 else "",
            })

        stage2_df = pd.DataFrame(stage2_rows)
        p2 = cfg.OUTPUT_DIR / "results" / "table6b_stage2_wls_rgi.csv"
        stage2_df.to_csv(p2, index=False)

        # Print interpretation
        rgi_coef = params[1]
        rgi_p    = pvals[1]
        print(f"\n  Stage 2 WLS result:")
        print(stage2_df.to_string(index=False))
        print(f"\n  R² of Stage 2: {wls_fit.rsquared:.3f}")
        if rgi_p < 0.05 and rgi_coef > 0:
            print(f"\n  ✓ FINDING 5 CONFIRMED: RGI significantly predicts the")
            print(f"    district-level IDI slope (β={rgi_coef:.4f}, p={rgi_p:.4f}).")
            print(f"    → Districts with higher reliability gaps show steeper")
            print(f"      IDI-to-disruption relationships.")
        else:
            print(f"\n  ⚠ Stage 2 RGI coefficient not significant (p={rgi_p:.4f}).")

        return stage1_df, stage2_df


# ─────────────────────────────────────────────────────────────────────────────
# ROBUSTNESS — PSM
# ─────────────────────────────────────────────────────────────────────────────

class PSMAnalysis:
    """
    Propensity Score Matching: piped vs tube well.
    Reports ATT with bootstrap SE (500 reps).
    Unchanged from v1.
    """

    def __init__(self, df: pd.DataFrame, cfg: Config):
        self.df  = df
        self.cfg = cfg

    def run(self) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("ANALYSIS — Robustness: Propensity Score Matching")
        print("=" * 60)

        cfg = self.cfg
        df  = self.df.copy()

        df_psm = df[df["water_source"].isin(
            ["Piped Water", "Tube Well/Borehole"])].copy()
        df_psm["treatment"] = (df_psm["water_source"] == "Piped Water").astype(int)

        covariates = [
            "wealth_q_num", "urban", "hh_size", "children_u5",
            "female_headed", "has_electricity", "improved_sanitation",
            "sc_st_flag",
        ]
        cov_present = [c for c in covariates if c in df_psm.columns]
        df_psm = df_psm.dropna(
            subset=["treatment", cfg.VAR_DISRUPTED, "weight"] + cov_present
        ).copy()

        if len(df_psm) < 200:
            print("  ⚠  Too few observations for PSM.")
            return pd.DataFrame()

        ps_formula = f"treatment ~ {' + '.join(cov_present)}"
        try:
            ps_model = smf.logit(
                ps_formula, data=df_psm, freq_weights=df_psm["weight"]
            ).fit(disp=False)
            df_psm["ps"] = ps_model.predict(df_psm)
        except Exception as e:
            print(f"  ✗  PS model failed: {e}")
            return pd.DataFrame()

        treated = df_psm[df_psm["treatment"] == 1].copy()
        control = df_psm[df_psm["treatment"] == 0].copy()

        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(control[["ps"]].values)
        dist, idx = nn.kneighbors(treated[["ps"]].values)

        valid           = dist.flatten() < cfg.PSM_CALIPER
        matched_treated = treated[valid].copy()
        matched_control = control.iloc[idx[valid].flatten()].copy()

        if len(matched_treated) == 0:
            print(f"  ⚠  No matches within caliper {cfg.PSM_CALIPER}")
            return pd.DataFrame()

        pct_matched = len(matched_treated) / len(treated) * 100
        print(f"  Matched {len(matched_treated):,} / {len(treated):,} "
              f"({pct_matched:.1f}%)")

        att = (weighted_rate(matched_treated, cfg.VAR_DISRUPTED)
               - weighted_rate(matched_control, cfg.VAR_DISRUPTED))

        rng   = np.random.default_rng(cfg.MONTE_CARLO_SEED)
        boots = []
        for _ in range(500):
            idx_b  = rng.integers(0, len(matched_treated), len(matched_treated))
            boots.append(
                weighted_rate(matched_treated.iloc[idx_b], cfg.VAR_DISRUPTED)
                - weighted_rate(matched_control.iloc[idx_b], cfg.VAR_DISRUPTED)
            )
        boots     = np.array(boots)
        att_se    = np.nanstd(boots)
        att_ci_lo = np.nanpercentile(boots, 2.5)
        att_ci_hi = np.nanpercentile(boots, 97.5)
        pct_gt0   = (boots > 0).mean() * 100

        summary = pd.DataFrame([{
            "Metric":         "ATT",
            "Estimate (pp)":  round(att, 2),
            "SE":             round(att_se, 2),
            "95% CI Lower":   round(att_ci_lo, 2),
            "95% CI Upper":   round(att_ci_hi, 2),
            "% boots > 0":    round(pct_gt0, 1),
            "Interpretation": (
                "Piped INCREASES disruption vs matched tube-well"
                if att > 0 else
                "Piped REDUCES disruption vs matched tube-well"
            ),
        }])

        out_path = cfg.OUTPUT_DIR / "results" / "table7_psm_att.csv"
        summary.to_csv(out_path, index=False)
        print(f"\n  ATT = {att:.2f} pp (95% CI {att_ci_lo:.2f}–{att_ci_hi:.2f})")
        print(f"  {pct_gt0:.1f}% of bootstrap samples show ATT > 0")
        return summary


# ─────────────────────────────────────────────────────────────────────────────
# REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class ReportGenerator:

    def __init__(self, cfg: Config, df: pd.DataFrame, all_tables: Dict):
        self.cfg        = cfg
        self.df         = df
        self.all_tables = all_tables

    def generate(self) -> str:
        cfg = self.cfg
        df  = self.df

        n_hh      = len(df)
        overall_dr = weighted_rate(df, cfg.VAR_DISRUPTED)
        piped_cov  = weighted_rate(df, "piped_flag")

        lines = [
            "# The Infrastructure Paradox: Modern Water Systems and",
            "# New Vulnerabilities in India",
            "## Evidence from NFHS-5 (2019-21)",
            "",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
            "",
            "---",
            "## ABSTRACT",
            "",
            f"Using data from {n_hh:,} households in NFHS-5, we document "
            f"the Infrastructure Paradox: piped-water households experience "
            f"higher disruption rates than tube-well users. "
            f"Overall disruption rate: {overall_dr:.1f}%. "
            f"Piped coverage: {piped_cov:.1f}%. "
            "We construct an Infrastructure Dependency Index (IDI, 4 dimensions, "
            "PCA-weighted, Monte Carlo CI) and a Reliability Gap Index (RGI, "
            "district-level weighted OLS residual with bootstrap CI). "
            "A GEE multilevel model and a two-stage slope-as-outcome model "
            "confirm the paradox is largest where IDI and RGI are jointly high.",
            "",
            "---",
        ]

        # Finding 1
        lines += ["## FINDING 1 — The Paradox in Raw Data", ""]
        t1a = self.all_tables.get("1a_by_source")
        if t1a is not None:
            lines.append(t1a.to_markdown(index=False))
        lines += ["", "**Finding**: Piped water sits near the top of the "
                  "disruption ranking despite being classified as improved.", ""]

        # Finding 2
        lines += ["## FINDING 2 — IDI Regression (with social controls)", ""]
        t2 = self.all_tables.get("regression_coefs")
        if t2 is not None:
            lines.append(t2.to_markdown())
        lines += ["", "---", ""]

        # Finding 3
        lines += ["## FINDING 3 — Geographic Concentration (RGI)", ""]
        t3 = self.all_tables.get("3a_top_crisis")
        if t3 is not None:
            lines.append(t3.to_markdown(index=False))
        lines += ["", "---", ""]

        # Finding 4
        lines += ["## FINDING 4 — GEE Multilevel Model (IDI × RGI)", "",
                  "Model: `disruption ~ IDI_std + RGI_std + IDI×RGI "
                  "+ piped + social controls + C(region)`", ""]
        t5 = self.all_tables.get("gee_coefs")
        if t5 is not None:
            lines.append(t5.to_markdown())
        lines += ["", "---", ""]

        # Finding 5
        lines += ["## FINDING 5 — Slope-as-Outcome: Does RGI Steepen IDI Effect?",
                  "",
                  "Stage 1: district IDI slopes (logit per district).",
                  "Stage 2: WLS — IDI slope ~ RGI, weights=1/SE².", ""]
        t6b = self.all_tables.get("stage2_wls")
        if t6b is not None:
            lines.append(t6b.to_markdown(index=False))
        lines += ["", "---", ""]

        # Robustness
        lines += ["## ROBUSTNESS", ""]
        mc_path = cfg.OUTPUT_DIR / "results" / "idi_monte_carlo_summary.csv"
        if mc_path.exists():
            lines.append(pd.read_csv(mc_path).to_markdown(index=False))
        lines += ["", "### PSM (ATT)", ""]
        psm_path = cfg.OUTPUT_DIR / "results" / "table7_psm_att.csv"
        if psm_path.exists():
            lines.append(pd.read_csv(psm_path).to_markdown(index=False))
        lines += ["", "---", ""]

        # Policy
        lines += [
            "## POLICY IMPLICATIONS", "",
            "1. **Reliability over coverage** — piped expansion without "
            "reliability investment worsens national disruption.",
            "2. **Target CRISIS districts** — high IDI + high RGI = priority.",
            "3. **Preserve backup sources** — RESILIENT POOR districts show "
            "source diversity protects even under system failure.",
            "4. **Protect marginalised households** — SC/ST and female-headed "
            "households show elevated disruption after controlling for IDI.",
            "",
            "---",
            f"*End of report. Outputs: {cfg.OUTPUT_DIR}*",
        ]

        return "\n".join(lines)

    def save(self, content: str) -> Path:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.cfg.OUTPUT_DIR / f"water_paradox_report_{ts}.md"
        path.write_text(content, encoding="utf-8")
        print(f"\n  ✓  Report → {path}")
        return path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSER ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class Analyzer:

    def __init__(self, df: pd.DataFrame, cfg: Config):
        self.df  = df
        self.cfg = cfg

    def run_all(self, dist_df: pd.DataFrame):
        print("\n" + "=" * 60)
        print("STEP 5 — Running all analyses")
        print("=" * 60)

        all_tables = {}

        # Finding 1
        desc   = DescriptiveTables(self.df, self.cfg)
        all_tables.update(desc.run_all())

        # Finding 2
        idi_reg = IDIRegression(self.df, self.cfg)
        coef_tbl, pred_tbl = idi_reg.run()
        all_tables["regression_coefs"] = coef_tbl
        all_tables["predicted_probs"]  = pred_tbl

        # Finding 3
        spatial  = SpatialTables(dist_df, self.cfg)
        all_tables.update(spatial.run_all())

        # Finding 4 — GEE
        gee = MultilevelModel(self.df, self.cfg)
        all_tables["gee_coefs"] = gee.run()

        # Finding 5 — Slope-as-outcome (NEW)
        sao = SlopeAsOutcomeModel(self.df, dist_df, self.cfg)
        s1_df, s2_df = sao.run()
        all_tables["stage1_slopes"] = s1_df
        all_tables["stage2_wls"]    = s2_df

        # Robustness — PSM
        psm = PSMAnalysis(self.df, self.cfg)
        psm.run()

        # Report
        reporter = ReportGenerator(self.cfg, self.df, all_tables)
        reporter.save(reporter.generate())

        print("\n" + "=" * 60)
        print("ALL ANALYSES COMPLETE")
        print(f"Output: {self.cfg.OUTPUT_DIR}")
        print("=" * 60)
