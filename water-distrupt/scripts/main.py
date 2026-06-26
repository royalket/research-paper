"""
main.py
─────────────────────────────────────────────────────────────────────────────
Entry point. All output is flushed immediately so the terminal never looks
frozen even during the long Monte Carlo and GEE phases.

Pipeline:
  1. config.py       → Config()
  2. data_pipeline   → DataLoader → DataProcessor  (adds season, sc_st_flag)
  3. idi_index       → IDIBuilder                  (4 dims, MC speedup)
  4. rgi_index       → RGIBuilder                  (weighted OLS, Moran's I)
  5. analysis        → Analyzer.run_all()
       Finding 1 : Descriptive tables (+ seasonal)
       Finding 1b: IDI dimension profiles
       Finding 2 : IDI logit (+ social controls)
       Finding 3 : Spatial tables
       Finding 4 : GEE multilevel model
       Finding 5 : Slope-as-outcome
       Robustness: PSM
       Report    : Markdown
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import time

from config        import Config
from data_pipeline import DataLoader, DataProcessor
from idi_index     import IDIBuilder
from rgi_index     import RGIBuilder
from analysis      import Analyzer


# ── Progress helpers ──────────────────────────────────────────────────────────

def p(msg: str = ""):
    """Print + flush immediately — terminal never looks frozen."""
    print(msg, flush=True)


_step_total  = 5
_step_start  = None
_run_start   = None


def step(n: int, label: str):
    global _step_start
    elapsed = time.time() - _run_start
    _step_start = time.time()
    p(f"\n{'─'*60}")
    p(f"  STEP {n}/{_step_total}  {label}")
    p(f"  [pipeline elapsed: {elapsed:.0f}s]")
    p(f"{'─'*60}")


def done(label: str):
    elapsed = time.time() - _step_start
    p(f"  ✓  {label}  ({elapsed:.1f}s)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _run_start
    _run_start = time.time()

    p("\n" + "=" * 60)
    p("  NFHS-5 Water Infrastructure Paradox Analysis  (v2)")
    p("=" * 60)

    cfg = Config()
    p(f"\n  Data file : {cfg.DATA_FILE_PATH}")
    p(f"  Output dir: {cfg.OUTPUT_DIR}")
    p(f"  MC runs   : {cfg.MONTE_CARLO_RUNS}")
    p(f"\n  Pipeline has {_step_total} steps. Long steps show live progress.")
    p(f"  Step 3 (Monte Carlo, {cfg.MONTE_CARLO_RUNS} runs) takes ~5–10 min.")
    p(f"  Step 5 (GEE model) takes ~1–3 min.")

    # ── 1. Load ───────────────────────────────────────────────────────────
    step(1, "Loading raw data")
    df_raw = DataLoader(cfg).load()
    if df_raw.empty:
        p("\n✗ Data loading failed. Check DATA_FILE_PATH in config.py")
        sys.exit(1)
    done(f"Loaded {len(df_raw):,} rows")

    # ── 2. Clean ──────────────────────────────────────────────────────────
    step(2, "Processing & cleaning")
    df_clean = DataProcessor(df_raw, cfg).process()
    if df_clean.empty:
        p("\n✗ Data processing produced empty DataFrame.")
        sys.exit(1)

    n     = len(df_clean)
    dr    = (df_clean[cfg.VAR_DISRUPTED] * df_clean["weight"]).sum() \
            / df_clean["weight"].sum() * 100
    piped = (df_clean["piped_flag"] * df_clean["weight"]).sum() \
            / df_clean["weight"].sum() * 100

    p(f"\n  Sanity check:")
    p(f"    Households : {n:,}")
    p(f"    Disruption : {dr:.1f}%")
    p(f"    Piped      : {piped:.1f}%")
    seasons = df_clean["season"].value_counts().to_dict()
    for s, c in sorted(seasons.items()):
        p(f"    {s:<14}: {c:,}")
    done(f"{n:,} clean households ready")

    # ── 3. IDI ────────────────────────────────────────────────────────────
    step(3, "Building IDI  (4 dims → PCA → Monte Carlo CI)")
    p("  Sub-steps:")
    p("    3a  Score 4 dimensions per household")
    p("    3b  Fit PCA → empirical weights")
    p(f"   3c  Monte Carlo  {cfg.MONTE_CARLO_RUNS} runs  [progress every 100 runs]")
    p("    3d  Validate: Cronbach α, AUC, discriminant validity")
    p("    3e  Dimension profiles by wealth / urban / source")

    idi_builder = IDIBuilder(df_clean, cfg)
    df_with_idi = idi_builder.build()

    p("\n  PCA loadings (4 dimensions):")
    p(idi_builder.pca_loadings.to_string(index=False))
    done("IDI built and validated")

    # ── 4. RGI ────────────────────────────────────────────────────────────
    step(4, "Building RGI  (district reliability gap → typology)")
    p("  Sub-steps:")
    p("    4a  Aggregate households to district level")
    p("    4b  Weighted OLS: expected disruption from development predictors")
    p("    4c  RGI = observed − expected  (residual) with bootstrap CI")
    p("    4d  Assign district typology: CRISIS / SAFE / VULNERABLE / RESILIENT POOR")
    p("    4e  Moran's I spatial autocorrelation (if libpysal installed)")

    rgi_builder           = RGIBuilder(df_with_idi, cfg)
    df_final, district_df = rgi_builder.build()

    p("\n  Top 10 CRISIS districts:")
    p(rgi_builder.top_crisis_districts.head(10).to_string(index=False))
    done(f"RGI built  |  {district_df['typology'].value_counts().to_dict()}")

    # ── 5. Analysis ───────────────────────────────────────────────────────
    step(5, "Running all analyses")
    p("  Sub-steps:")
    p("    5a  Finding 1  — Descriptive cross-tabs (source × wealth/urban/region/season)")
    p("    5b  Finding 1b — IDI dimension profiles")
    p("    5c  Finding 2  — IDI logistic regression + social controls")
    p("    5d  Finding 3  — Spatial tables (CRISIS / SAFE / state ranking)")
    p("    5e  Finding 4  — GEE multilevel model  IDI × RGI  [~1–3 min]")
    p("    5f  Finding 5  — Two-stage slope-as-outcome")
    p("    5g  Robustness — PSM  ATT with bootstrap SE")
    p("    5h  Report     — Markdown summary")

    analyzer = Analyzer(df_final, cfg)
    analyzer.run_all(district_df, idi_dim_profiles=idi_builder.dim_profiles)

    total = time.time() - _run_start
    p("\n" + "=" * 60)
    p("  PIPELINE COMPLETE")
    p(f"  Total time : {total/60:.1f} min")
    p(f"  Outputs    : {cfg.OUTPUT_DIR}")
    p("=" * 60 + "\n")


if __name__ == "__main__":
    main()
