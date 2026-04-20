"""
main.py
─────────────────────────────────────────────────────────────────────────────
Entry point. Calls the five modules in order.

To run:
    python main.py

Pipeline order:
    1. config.py      → Config()           constants and paths
    2. data_pipeline  → DataLoader         load raw .dta
                     → DataProcessor      clean + derive base variables
    3. idi_index      → IDIBuilder         PCA weights + Monte Carlo CI
    4. rgi_index      → RGIBuilder         district reliability gap
    5. analysis       → Analyzer           all tables, models, report

Each step prints its own progress. If a step fails, it fails loudly
with an informative error so you know exactly where to look.
─────────────────────────────────────────────────────────────────────────────
"""

import sys
from config       import Config
from data_pipeline import DataLoader, DataProcessor
from idi_index    import IDIBuilder
from rgi_index    import RGIBuilder
from analysis     import Analyzer


def main():
    print("\n" + "=" * 60)
    print("  NFHS-5 Water Infrastructure Paradox Analysis")
    print("=" * 60)

    # ── 1. Configuration ──────────────────────────────────────────────────
    cfg = Config()
    print(f"\n  Data file : {cfg.DATA_FILE_PATH}")
    print(f"  Output dir: {cfg.OUTPUT_DIR}")

    # ── 2. Load and clean data ────────────────────────────────────────────
    df_raw   = DataLoader(cfg).load()
    if df_raw.empty:
        print("\n✗ Data loading failed. Check DATA_FILE_PATH in config.py")
        sys.exit(1)

    df_clean = DataProcessor(df_raw, cfg).process()
    if df_clean.empty:
        print("\n✗ Data processing produced empty DataFrame. Check logs above.")
        sys.exit(1)

    # Quick sanity check
    n      = len(df_clean)
    dr     = (df_clean[cfg.VAR_DISRUPTED] * df_clean["weight"]).sum() \
             / df_clean["weight"].sum() * 100
    piped  = (df_clean["piped_flag"] * df_clean["weight"]).sum() \
             / df_clean["weight"].sum() * 100
    print(f"\n  Sanity check:")
    print(f"    Households      : {n:,}")
    print(f"    Disruption rate : {dr:.1f}%")
    print(f"    Piped coverage  : {piped:.1f}%")

    # ── 3. Build IDI (PCA + Monte Carlo) ─────────────────────────────────
    idi_builder  = IDIBuilder(df_clean, cfg)
    df_with_idi  = idi_builder.build()

    # Print PCA loadings for the paper's methods section
    print("\n  PCA loadings (for Table X in paper):")
    print(idi_builder.pca_loadings.to_string(index=False))

    # ── 4. Build RGI (district reliability gap) ───────────────────────────
    rgi_builder              = RGIBuilder(df_with_idi, cfg)
    df_final, district_df    = rgi_builder.build()

    # Print top crisis districts
    print("\n  Top 10 CRISIS districts:")
    print(rgi_builder.top_crisis_districts.head(10).to_string(index=False))

    # ── 5. Run all analyses and generate report ────────────────────────────
    analyzer = Analyzer(df_final, cfg)
    analyzer.run_all(district_df)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  All outputs in: {cfg.OUTPUT_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
