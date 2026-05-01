"""
main.py
─────────────────────────────────────────────────────────────────────────────
Entry point. Unchanged pipeline order — all new logic is inside modules.

Pipeline:
  1. config.py       → Config()
  2. data_pipeline   → DataLoader → DataProcessor  (adds season, sc_st_flag)
  3. idi_index       → IDIBuilder                  (4 dims, MC speedup)
  4. rgi_index       → RGIBuilder                  (weighted OLS, Moran's I)
  5. analysis        → Analyzer.run_all()
       Finding 1: Descriptive tables (+ seasonal)
       Finding 2: IDI logit (+ social controls)
       Finding 3: Spatial tables
       Finding 4: GEE multilevel model
       Finding 5: Slope-as-outcome (NEW)
       Robustness: PSM
       Report: Markdown
─────────────────────────────────────────────────────────────────────────────
"""

import sys
from config        import Config
from data_pipeline import DataLoader, DataProcessor
from idi_index     import IDIBuilder
from rgi_index     import RGIBuilder
from analysis      import Analyzer


def main():
    print("\n" + "=" * 60)
    print("  NFHS-5 Water Infrastructure Paradox Analysis  (v2)")
    print("=" * 60)

    cfg = Config()
    print(f"\n  Data file : {cfg.DATA_FILE_PATH}")
    print(f"  Output dir: {cfg.OUTPUT_DIR}")
    print(f"  MC runs   : {cfg.MONTE_CARLO_RUNS}")

    # ── 1. Load ───────────────────────────────────────────────────────────
    df_raw = DataLoader(cfg).load()
    if df_raw.empty:
        print("\n✗ Data loading failed. Check DATA_FILE_PATH in config.py")
        sys.exit(1)

    # ── 2. Clean ──────────────────────────────────────────────────────────
    df_clean = DataProcessor(df_raw, cfg).process()
    if df_clean.empty:
        print("\n✗ Data processing produced empty DataFrame.")
        sys.exit(1)

    n     = len(df_clean)
    dr    = (df_clean[cfg.VAR_DISRUPTED] * df_clean["weight"]).sum() \
            / df_clean["weight"].sum() * 100
    piped = (df_clean["piped_flag"] * df_clean["weight"]).sum() \
            / df_clean["weight"].sum() * 100
    print(f"\n  Sanity check:")
    print(f"    Households : {n:,}")
    print(f"    Disruption : {dr:.1f}%")
    print(f"    Piped      : {piped:.1f}%")
    print(f"    Seasons    : {df_clean['season'].value_counts().to_dict()}")

    # ── 3. IDI ────────────────────────────────────────────────────────────
    idi_builder = IDIBuilder(df_clean, cfg)
    df_with_idi = idi_builder.build()

    print("\n  PCA loadings (4 dimensions):")
    print(idi_builder.pca_loadings.to_string(index=False))

    # ── 4. RGI ────────────────────────────────────────────────────────────
    rgi_builder           = RGIBuilder(df_with_idi, cfg)
    df_final, district_df = rgi_builder.build()

    print("\n  Top 10 CRISIS districts:")
    print(rgi_builder.top_crisis_districts.head(10).to_string(index=False))

    # ── 5. Analysis ───────────────────────────────────────────────────────
    analyzer = Analyzer(df_final, cfg)
    analyzer.run_all(district_df)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Outputs: {cfg.OUTPUT_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
