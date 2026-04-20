"""
data_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Responsibility: Load raw NFHS-5 .dta → return clean, analysis-ready DataFrame.

Nothing here does statistics or builds indices.
It only cleans and derives base variables that every downstream file needs.

Output columns guaranteed to exist after DataProcessor.process():
  Identifiers  : state_code, state_name, region, district_code, district_name
                 urban, residence, season, weight
  Outcome      : water_disrupted  (0/1)
  Water        : water_source, alt_source, piped_flag, tube_well_flag,
                 improved_flag, time_to_water_min, water_location,
                 water_on_premises, women_fetch, children_fetch
  Socioeconomic: wealth_quintile, wealth_score, hh_size, children_u5,
                 head_sex, female_headed, head_education,
                 religion, caste
  Assets       : has_electricity, has_tv, has_fridge, has_vehicle,
                 improved_sanitation, house_type
─────────────────────────────────────────────────────────────────────────────
"""

import warnings
import numpy as np
import pandas as pd
import pyreadstat

from config import Config

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# LOADER
# ─────────────────────────────────────────────────────────────────────────────

class DataLoader:
    """Reads the NFHS-5 .dta file, loading only the columns we need."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def load(self) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("STEP 1 — Loading raw data")
        print("=" * 60)
        print(f"  File: {self.cfg.DATA_FILE_PATH}")

        try:
            # First pass: metadata only (fast) to check what columns exist
            _, meta = pyreadstat.read_dta(
                self.cfg.DATA_FILE_PATH, metadataonly=True
            )
            available = set(meta.column_names)
            wanted    = set(self.cfg.COLS_TO_LOAD)
            to_load   = list(wanted & available)
            missing   = wanted - available

            if missing:
                print(f"  ⚠  {len(missing)} requested columns not in file "
                      f"(will be skipped): {sorted(missing)}")
            if not to_load:
                raise ValueError("No required columns found in .dta file.")

            # Second pass: actual data load
            df, _ = pyreadstat.read_dta(
                self.cfg.DATA_FILE_PATH, usecols=to_load
            )
            print(f"  ✓  Loaded {len(df):,} rows × {len(df.columns)} columns")
            return df

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Data file not found: {self.cfg.DATA_FILE_PATH}\n"
                "Update Config.DATA_FILE_PATH to the correct path."
            )


# ─────────────────────────────────────────────────────────────────────────────
# PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

class DataProcessor:
    """
    Cleans raw data and creates all derived base variables.
    Call .process() → returns clean DataFrame.
    """

    def __init__(self, df_raw: pd.DataFrame, cfg: Config):
        self.df  = df_raw.copy()
        self.cfg = cfg

    # ── Public entry point ────────────────────────────────────────────────

    def process(self) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("STEP 2 — Processing & cleaning data")
        print("=" * 60)
        print(f"  Starting rows: {len(self.df):,}")

        self._replace_missing_codes()
        self._apply_weights()
        self._create_outcome()
        self._create_geography()
        self._create_temporal()
        self._create_water_vars()
        self._create_socioeconomic()
        self._create_assets()
        self._drop_raw_cols()

        print(f"  ✓  Final rows: {len(self.df):,}")
        print("=" * 60)
        return self.df

    # ── Private helpers ───────────────────────────────────────────────────

    def _replace_missing_codes(self):
        """Replace NFHS missing codes (8,9,99,etc.) with NaN."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].replace(
            self.cfg.MISSING_CODES, np.nan
        )
        # Handle time-to-water = 996 (on premises) separately BEFORE replacing
        ttw = self.cfg.VAR_TIME_TO_WATER
        if ttw in self.df.columns:
            self.df["water_on_premises"] = (self.df[ttw] == 996).astype(int)
            self.df[ttw] = self.df[ttw].replace(996, 0)
        else:
            self.df["water_on_premises"] = 0

    def _apply_weights(self):
        """Create normalised survey weight. Drop rows with missing weight."""
        w = self.cfg.VAR_WEIGHT
        if w in self.df.columns:
            self.df["weight"] = pd.to_numeric(
                self.df[w], errors="coerce"
            ) / 1_000_000
            n_before = len(self.df)
            self.df.dropna(subset=["weight"], inplace=True)
            dropped = n_before - len(self.df)
            if dropped:
                print(f"  ⚠  Dropped {dropped:,} rows with missing weight")
        else:
            self.df["weight"] = 1.0
            print("  ⚠  Weight column not found — analysis will be UNWEIGHTED")

    def _create_outcome(self):
        """
        Create binary water_disrupted from raw sh37b.
        Drop rows where response is invalid (not 0 or 1 after cleaning).
        """
        raw = self.cfg.VAR_DISRUPTED_RAW
        out = self.cfg.VAR_DISRUPTED

        if raw not in self.df.columns:
            raise ValueError(
                f"Outcome column '{raw}' not found. "
                "Cannot proceed without water disruption variable."
            )

        self.df[raw] = pd.to_numeric(self.df[raw], errors="coerce")
        self.df[out] = self.df[raw].apply(
            lambda x: 1 if x == 1 else (0 if x == 0 else np.nan)
        )
        n_before = len(self.df)
        self.df.dropna(subset=[out], inplace=True)
        self.df[out] = self.df[out].astype(int)
        dropped = n_before - len(self.df)
        if dropped:
            print(f"  ⚠  Dropped {dropped:,} rows with missing disruption status")

        rate = self.df[out].mean() * 100
        print(f"  ✓  Outcome created. Overall disruption rate: {rate:.1f}%")

    def _create_geography(self):
        """State, region, residence (urban/rural), district."""
        cfg = self.cfg

        # State
        if cfg.VAR_STATE in self.df.columns:
            self.df["state_code"] = pd.to_numeric(
                self.df[cfg.VAR_STATE], errors="coerce"
            )
            self.df["state_name"] = self.df["state_code"].map(
                cfg.STATE_NAMES
            ).fillna("Unknown State")
        else:
            self.df["state_code"] = np.nan
            self.df["state_name"] = "Unknown State"

        # Region
        def _region(sc):
            if pd.isna(sc):
                return "Unknown"
            for name, codes in cfg.REGIONS.items():
                if int(sc) in codes:
                    return name
            return "Other"

        self.df["region"] = self.df["state_code"].apply(_region)

        # Urban / rural
        if cfg.VAR_URBAN in self.df.columns:
            self.df["urban"] = (
                pd.to_numeric(self.df[cfg.VAR_URBAN], errors="coerce") == 1
            ).astype(int)
            self.df["residence"] = self.df["urban"].map(
                {1: "Urban", 0: "Rural"}
            )
        else:
            self.df["urban"]     = 0
            self.df["residence"] = "Unknown"

        # District
        if cfg.VAR_DISTRICT in self.df.columns:
            self.df["district_code"] = pd.to_numeric(
                self.df[cfg.VAR_DISTRICT], errors="coerce"
            )
        else:
            self.df["district_code"] = np.nan

    def _create_temporal(self):
        """Season from interview month."""
        cfg = self.cfg
        if cfg.VAR_MONTH in self.df.columns:
            self.df["month"] = pd.to_numeric(
                self.df[cfg.VAR_MONTH], errors="coerce"
            )

            def _season(m):
                if pd.isna(m):
                    return "Unknown"
                for name, months in cfg.SEASONS.items():
                    if int(m) in months:
                        return name
                return "Unknown"

            self.df["season"] = self.df["month"].apply(_season)
        else:
            self.df["season"] = "Unknown"

    def _create_water_vars(self):
        """Water source, alternative source, time, location, fetcher."""
        cfg = self.cfg
        src_map = cfg.WATER_SOURCE_MAP

        # Primary source
        if cfg.VAR_SOURCE_PRIMARY in self.df.columns:
            self.df[cfg.VAR_SOURCE_PRIMARY] = pd.to_numeric(
                self.df[cfg.VAR_SOURCE_PRIMARY], errors="coerce"
            )
            self.df["water_source"] = (
                self.df[cfg.VAR_SOURCE_PRIMARY].map(src_map).fillna("Other Source")
            )
        else:
            self.df["water_source"] = "Unknown"

        # Alternative source
        if cfg.VAR_SOURCE_ALT in self.df.columns:
            self.df[cfg.VAR_SOURCE_ALT] = pd.to_numeric(
                self.df[cfg.VAR_SOURCE_ALT], errors="coerce"
            )
            self.df["alt_source"] = (
                self.df[cfg.VAR_SOURCE_ALT].map(src_map).fillna("No Other Source")
            )
        else:
            self.df["alt_source"] = "No Other Source"

        # Convenience flags
        self.df["piped_flag"]    = (self.df["water_source"] == "Piped Water").astype(int)
        self.df["tube_well_flag"] = (self.df["water_source"] == "Tube Well/Borehole").astype(int)
        self.df["improved_flag"] = self.df["water_source"].isin([
            "Piped Water", "Tube Well/Borehole",
            "Protected Well/Spring", "Protected Spring",
            "Bottled Water", "Community RO Plant",
        ]).astype(int)

        # Time to water
        if cfg.VAR_TIME_TO_WATER in self.df.columns:
            self.df["time_to_water_min"] = pd.to_numeric(
                self.df[cfg.VAR_TIME_TO_WATER], errors="coerce"
            )
        else:
            self.df["time_to_water_min"] = np.nan

        # Water location
        loc_map = {1: "In Dwelling", 2: "In Yard/Plot", 3: "Elsewhere"}
        if cfg.VAR_WATER_LOCATION in self.df.columns:
            self.df["water_location"] = (
                pd.to_numeric(self.df[cfg.VAR_WATER_LOCATION], errors="coerce")
                .map(loc_map).fillna("Unknown")
            )
        else:
            self.df["water_location"] = "Unknown"

        # Who fetches water
        if cfg.VAR_FETCHER_MAIN in self.df.columns:
            fetcher = pd.to_numeric(
                self.df[cfg.VAR_FETCHER_MAIN], errors="coerce"
            )
            self.df["women_fetch"]    = (fetcher == 1).astype(int)
            self.df["children_fetch"] = (fetcher == 3).astype(int)
        else:
            self.df["women_fetch"]    = 0
            self.df["children_fetch"] = 0

    def _create_socioeconomic(self):
        """Wealth, household size, head characteristics, religion, caste, education."""
        cfg = self.cfg

        # Wealth quintile
        wq_map = {1: "Poorest", 2: "Poorer", 3: "Middle", 4: "Richer", 5: "Richest"}
        if cfg.VAR_WEALTH_QUINTILE in self.df.columns:
            self.df["wealth_quintile"] = (
                pd.to_numeric(self.df[cfg.VAR_WEALTH_QUINTILE], errors="coerce")
                .map(wq_map)
            )
            # Keep numeric version for models
            self.df["wealth_q_num"] = pd.to_numeric(
                self.df[cfg.VAR_WEALTH_QUINTILE], errors="coerce"
            )
        else:
            self.df["wealth_quintile"] = "Unknown"
            self.df["wealth_q_num"]    = np.nan

        # Wealth score (continuous)
        if cfg.VAR_WEALTH_SCORE in self.df.columns:
            self.df["wealth_score"] = pd.to_numeric(
                self.df[cfg.VAR_WEALTH_SCORE], errors="coerce"
            )
        else:
            self.df["wealth_score"] = np.nan

        # Household size
        if cfg.VAR_HH_SIZE in self.df.columns:
            self.df["hh_size"] = pd.to_numeric(
                self.df[cfg.VAR_HH_SIZE], errors="coerce"
            )
        else:
            self.df["hh_size"] = np.nan

        # Children under 5
        if cfg.VAR_CHILDREN_U5 in self.df.columns:
            self.df["children_u5"] = pd.to_numeric(
                self.df[cfg.VAR_CHILDREN_U5], errors="coerce"
            ).fillna(0)
        else:
            self.df["children_u5"] = 0

        # Head sex
        if cfg.VAR_HEAD_SEX in self.df.columns:
            head_sex = pd.to_numeric(self.df[cfg.VAR_HEAD_SEX], errors="coerce")
            self.df["female_headed"] = (head_sex == 2).astype(int)
            self.df["head_sex"]      = head_sex.map({1: "Male", 2: "Female"}).fillna("Unknown")
        else:
            self.df["female_headed"] = 0
            self.df["head_sex"]      = "Unknown"

        # HH head education (derived from hv101_XX / hv106_XX member records)
        edu_map = {0: "No Education", 1: "Primary", 2: "Secondary", 3: "Higher"}
        self.df["head_education"] = "Unknown"
        found_any = False
        for i in range(1, 16):
            rel_col = f"hv101_{i:02d}"
            edu_col = f"hv106_{i:02d}"
            if rel_col in self.df.columns and edu_col in self.df.columns:
                found_any = True
                rel = pd.to_numeric(self.df[rel_col], errors="coerce")
                edu = pd.to_numeric(self.df[edu_col], errors="coerce")
                is_head      = rel == 1
                valid_edu    = edu.isin(edu_map.keys())
                still_unknwn = self.df["head_education"] == "Unknown"
                mask = is_head & valid_edu & still_unknwn
                self.df.loc[mask, "head_education"] = edu[mask].map(edu_map)
        if not found_any:
            print("  ⚠  hv101_XX / hv106_XX columns not found — head_education = 'Unknown'")

        # Religion
        rel_map = {
            1: "Hindu", 2: "Muslim", 3: "Christian", 4: "Sikh",
            5: "Buddhist", 6: "Jain", 9: "No Religion", 96: "Other",
        }
        if cfg.VAR_RELIGION in self.df.columns:
            self.df["religion"] = (
                pd.to_numeric(self.df[cfg.VAR_RELIGION], errors="coerce")
                .map(rel_map).fillna("Unknown")
            )
        else:
            self.df["religion"] = "Unknown"

        # Caste
        caste_map = {1: "SC", 2: "ST", 3: "OBC", 4: "General"}
        if cfg.VAR_CASTE in self.df.columns:
            self.df["caste"] = (
                pd.to_numeric(self.df[cfg.VAR_CASTE], errors="coerce")
                .map(caste_map).fillna("Unknown")
            )
        else:
            self.df["caste"] = "Unknown"

    def _create_assets(self):
        """Binary asset flags and house/sanitation type."""
        cfg = self.cfg

        asset_cols = {
            cfg.VAR_ELECTRICITY: "has_electricity",
            cfg.VAR_TV:          "has_tv",
            cfg.VAR_FRIDGE:      "has_fridge",
            cfg.VAR_MOTORCYCLE:  "has_motorcycle",
            cfg.VAR_CAR:         "has_car",
            cfg.VAR_MOBILE:      "has_mobile",
        }
        for raw_col, new_col in asset_cols.items():
            if raw_col in self.df.columns:
                val = pd.to_numeric(self.df[raw_col], errors="coerce")
                self.df[new_col] = (val == 1).astype(int)
            else:
                self.df[new_col] = 0

        self.df["has_vehicle"] = (
            (self.df["has_motorcycle"] == 1) | (self.df["has_car"] == 1)
        ).astype(int)

        # House type
        house_map = {1: "Pucca", 2: "Semi-pucca", 3: "Katcha"}
        if cfg.VAR_HOUSE_TYPE in self.df.columns:
            self.df["house_type"] = (
                pd.to_numeric(self.df[cfg.VAR_HOUSE_TYPE], errors="coerce")
                .map(house_map).fillna("Unknown")
            )
        else:
            self.df["house_type"] = "Unknown"

        # Improved sanitation
        if cfg.VAR_TOILET in self.df.columns:
            toilet = pd.to_numeric(self.df[cfg.VAR_TOILET], errors="coerce")
            # Flush toilet (11-15) or improved pit latrine (21-22) = improved
            self.df["improved_sanitation"] = toilet.apply(
                lambda x: 1 if (not pd.isna(x) and int(x) in
                                list(range(11, 16)) + [21, 22]) else 0
            )
        else:
            self.df["improved_sanitation"] = 0

    def _drop_raw_cols(self):
        """
        Drop original NFHS column names — downstream files use our clean names.
        Keep weight, PSU, stratum for survey-design-aware models.
        """
        always_keep = {
            "weight", "water_on_premises",
            self.cfg.VAR_PSU,
            self.cfg.VAR_STRATUM,
            self.cfg.VAR_CLUSTER,
            self.cfg.VAR_WEALTH_SCORE,   # needed by RGI
            self.cfg.VAR_WEALTH_QUINTILE, # numeric original needed by models
        }
        raw_to_drop = [
            c for c in self.df.columns
            if c in set(self.cfg.COLS_TO_LOAD) and c not in always_keep
        ]
        self.df.drop(columns=raw_to_drop, inplace=True, errors="ignore")
