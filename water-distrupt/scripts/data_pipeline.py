"""
data_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Responsibility: Load raw NFHS-5 .dta → return clean, analysis-ready DataFrame.

Changes from v1:
  - _create_temporal() now also exposes interview_month (int) for RGI
  - _create_assets() confirms has_fridge and has_vehicle (unchanged, verified)
  - _create_socioeconomic() adds sc_st_flag for social control in regressions
  - No other logic changes — pipeline order unchanged

Output columns guaranteed after DataProcessor.process():
  Identifiers  : state_code, state_name, region, district_code,
                 urban, residence, season, interview_month, weight
  Outcome      : water_disrupted  (0/1)
  Water        : water_source, alt_source, piped_flag, tube_well_flag,
                 improved_flag, time_to_water_min, water_location,
                 water_on_premises, women_fetch, children_fetch
  Socioeconomic: wealth_quintile, wealth_q_num, wealth_score,
                 hh_size, children_u5, female_headed, head_education,
                 religion, caste, sc_st_flag
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

    def process(self) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("STEP 2 — Processing & cleaning data")
        print("=" * 60)
        print(f"  Starting rows: {len(self.df):,}")

        self._replace_missing_codes()
        self._apply_weights()
        self._create_outcome()
        self._create_geography()
        self._create_temporal()      # adds season + interview_month
        self._create_water_vars()
        self._create_socioeconomic() # adds sc_st_flag
        self._create_assets()        # has_fridge, has_vehicle confirmed
        self._drop_raw_cols()

        print(f"  ✓  Final rows: {len(self.df):,}")
        print("=" * 60)
        return self.df

    # ── Private helpers ───────────────────────────────────────────────────

    def _replace_missing_codes(self):
        """Replace NFHS missing codes with NaN. Handle 996 (on-premises) first."""
        ttw = self.cfg.VAR_TIME_TO_WATER
        if ttw in self.df.columns:
            self.df["water_on_premises"] = (
                pd.to_numeric(self.df[ttw], errors="coerce") == 996
            ).astype(int)
            self.df[ttw] = self.df[ttw].replace(996, 0)
        else:
            self.df["water_on_premises"] = 0

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].replace(
            self.cfg.MISSING_CODES, np.nan
        )

    def _apply_weights(self):
        """Create normalised survey weight (raw / 1e6). Drop missing."""
        w = self.cfg.VAR_WEIGHT
        if w in self.df.columns:
            self.df["weight"] = pd.to_numeric(self.df[w], errors="coerce") / 1_000_000
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
        Binary water_disrupted from raw sh37b.
        1 = disrupted, 0 = not disrupted. Drop invalid responses.
        """
        raw = self.cfg.VAR_DISRUPTED_RAW
        out = self.cfg.VAR_DISRUPTED

        if raw not in self.df.columns:
            raise ValueError(
                f"Outcome column '{raw}' not found. Cannot proceed."
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
        """State, region, urban/rural, district."""
        cfg = self.cfg

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

        def _region(sc):
            if pd.isna(sc):
                return "Unknown"
            for name, codes in cfg.REGIONS.items():
                if int(sc) in codes:
                    return name
            return "Other"

        self.df["region"] = self.df["state_code"].apply(_region)

        if cfg.VAR_URBAN in self.df.columns:
            self.df["urban"] = (
                pd.to_numeric(self.df[cfg.VAR_URBAN], errors="coerce") == 1
            ).astype(int)
            self.df["residence"] = self.df["urban"].map(
                {1: "Urban", 0: "Rural"}
            ).fillna("Unknown")
        else:
            self.df["urban"]     = 0
            self.df["residence"] = "Unknown"

        if cfg.VAR_DISTRICT in self.df.columns:
            self.df["district_code"] = pd.to_numeric(
                self.df[cfg.VAR_DISTRICT], errors="coerce"
            )
        else:
            self.df["district_code"] = np.nan

    def _create_temporal(self):
        """
        Season from interview month (hv006).
        Also expose interview_month as integer for RGI pct_monsoon calculation.
        """
        cfg = self.cfg
        if cfg.VAR_MONTH in self.df.columns:
            month = pd.to_numeric(self.df[cfg.VAR_MONTH], errors="coerce")
            self.df["interview_month"] = month  # integer 1–12, used by RGI

            def _season(m):
                if pd.isna(m):
                    return "Unknown"
                for name, months in cfg.SEASONS.items():
                    if int(m) in months:
                        return name
                return "Unknown"

            self.df["season"] = month.apply(_season)
        else:
            self.df["interview_month"] = np.nan
            self.df["season"]          = "Unknown"
            print(f"  ⚠  '{cfg.VAR_MONTH}' not found — season = Unknown")

    def _create_water_vars(self):
        """Water source, flags, time, location, fetcher."""
        cfg     = self.cfg
        src_map = cfg.WATER_SOURCE_MAP

        if cfg.VAR_SOURCE_PRIMARY in self.df.columns:
            self.df[cfg.VAR_SOURCE_PRIMARY] = pd.to_numeric(
                self.df[cfg.VAR_SOURCE_PRIMARY], errors="coerce"
            )
            self.df["water_source"] = (
                self.df[cfg.VAR_SOURCE_PRIMARY].map(src_map).fillna("Other Source")
            )
        else:
            self.df["water_source"] = "Unknown"

        if cfg.VAR_SOURCE_ALT in self.df.columns:
            self.df[cfg.VAR_SOURCE_ALT] = pd.to_numeric(
                self.df[cfg.VAR_SOURCE_ALT], errors="coerce"
            )
            self.df["alt_source"] = (
                self.df[cfg.VAR_SOURCE_ALT].map(src_map).fillna("No Other Source")
            )
        else:
            self.df["alt_source"] = "No Other Source"

        self.df["piped_flag"]    = (self.df["water_source"] == "Piped Water").astype(int)
        self.df["tube_well_flag"] = (self.df["water_source"] == "Tube Well/Borehole").astype(int)
        self.df["improved_flag"] = self.df["water_source"].isin([
            "Piped Water", "Tube Well/Borehole",
            "Protected Well/Spring", "Protected Spring",
            "Bottled Water", "Community RO Plant",
        ]).astype(int)

        if cfg.VAR_TIME_TO_WATER in self.df.columns:
            self.df["time_to_water_min"] = pd.to_numeric(
                self.df[cfg.VAR_TIME_TO_WATER], errors="coerce"
            )
        else:
            self.df["time_to_water_min"] = np.nan

        loc_map = {1: "In Dwelling", 2: "In Yard/Plot", 3: "Elsewhere"}
        if cfg.VAR_WATER_LOCATION in self.df.columns:
            self.df["water_location"] = (
                pd.to_numeric(self.df[cfg.VAR_WATER_LOCATION], errors="coerce")
                .map(loc_map).fillna("Unknown")
            )
        else:
            self.df["water_location"] = "Unknown"

        if cfg.VAR_FETCHER_MAIN in self.df.columns:
            fetcher = pd.to_numeric(self.df[cfg.VAR_FETCHER_MAIN], errors="coerce")
            self.df["women_fetch"]    = (fetcher == 1).astype(int)
            self.df["children_fetch"] = (fetcher == 3).astype(int)
        else:
            self.df["women_fetch"]    = 0
            self.df["children_fetch"] = 0

    def _create_socioeconomic(self):
        """
        Wealth, household characteristics, education, religion, caste.
        NEW: sc_st_flag — 1 if household head is SC or ST, 0 otherwise.
             Used as social control in regression formulas.
        """
        cfg = self.cfg

        wq_map = {1: "Poorest", 2: "Poorer", 3: "Middle", 4: "Richer", 5: "Richest"}
        if cfg.VAR_WEALTH_QUINTILE in self.df.columns:
            wq_raw = pd.to_numeric(self.df[cfg.VAR_WEALTH_QUINTILE], errors="coerce")
            self.df["wealth_quintile"] = wq_raw.map(wq_map)
            self.df["wealth_q_num"]    = wq_raw
        else:
            self.df["wealth_quintile"] = "Unknown"
            self.df["wealth_q_num"]    = np.nan

        if cfg.VAR_WEALTH_SCORE in self.df.columns:
            self.df["wealth_score"] = pd.to_numeric(
                self.df[cfg.VAR_WEALTH_SCORE], errors="coerce"
            )
        else:
            self.df["wealth_score"] = np.nan

        if cfg.VAR_HH_SIZE in self.df.columns:
            self.df["hh_size"] = pd.to_numeric(
                self.df[cfg.VAR_HH_SIZE], errors="coerce"
            )
        else:
            self.df["hh_size"] = np.nan

        if cfg.VAR_CHILDREN_U5 in self.df.columns:
            self.df["children_u5"] = pd.to_numeric(
                self.df[cfg.VAR_CHILDREN_U5], errors="coerce"
            ).fillna(0)
        else:
            self.df["children_u5"] = 0

        if cfg.VAR_HEAD_SEX in self.df.columns:
            head_sex = pd.to_numeric(self.df[cfg.VAR_HEAD_SEX], errors="coerce")
            self.df["female_headed"] = (head_sex == 2).astype(int)
        else:
            self.df["female_headed"] = 0

        # HH head education from member records
        edu_map = {0: "No Education", 1: "Primary", 2: "Secondary", 3: "Higher"}
        self.df["head_education"] = "Unknown"
        for i in range(1, 16):
            rel_col = f"hv101_{i:02d}"
            edu_col = f"hv106_{i:02d}"
            if rel_col in self.df.columns and edu_col in self.df.columns:
                rel = pd.to_numeric(self.df[rel_col], errors="coerce")
                edu = pd.to_numeric(self.df[edu_col], errors="coerce")
                mask = (rel == 1) & edu.isin(edu_map) & (self.df["head_education"] == "Unknown")
                self.df.loc[mask, "head_education"] = edu[mask].map(edu_map)

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

        caste_map = {1: "SC", 2: "ST", 3: "OBC", 4: "General"}
        if cfg.VAR_CASTE in self.df.columns:
            caste_raw = pd.to_numeric(self.df[cfg.VAR_CASTE], errors="coerce")
            self.df["caste"]      = caste_raw.map(caste_map).fillna("Unknown")
            # NEW: binary flag for marginalised caste (SC=1, ST=2)
            self.df["sc_st_flag"] = caste_raw.isin([1, 2]).astype(int)
        else:
            self.df["caste"]      = "Unknown"
            self.df["sc_st_flag"] = 0

    def _create_assets(self):
        """
        Binary asset flags, house type, sanitation.
        has_fridge: physical water storage proxy (used in IDI Dim 4)
        has_vehicle: fetching mobility proxy (used in IDI Dim 4)
        """
        cfg = self.cfg

        asset_cols = {
            cfg.VAR_ELECTRICITY: "has_electricity",
            cfg.VAR_TV:          "has_tv",
            cfg.VAR_FRIDGE:      "has_fridge",    # IDI Dim 4
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

        # has_vehicle: motorcycle OR car — mobility proxy for fetching backup
        self.df["has_vehicle"] = (
            (self.df["has_motorcycle"] == 1) | (self.df["has_car"] == 1)
        ).astype(int)

        house_map = {1: "Pucca", 2: "Semi-pucca", 3: "Katcha"}
        if cfg.VAR_HOUSE_TYPE in self.df.columns:
            self.df["house_type"] = (
                pd.to_numeric(self.df[cfg.VAR_HOUSE_TYPE], errors="coerce")
                .map(house_map).fillna("Unknown")
            )
        else:
            self.df["house_type"] = "Unknown"

        if cfg.VAR_TOILET in self.df.columns:
            toilet = pd.to_numeric(self.df[cfg.VAR_TOILET], errors="coerce")
            self.df["improved_sanitation"] = toilet.apply(
                lambda x: 1 if (not pd.isna(x) and int(x) in
                                list(range(11, 16)) + [21, 22]) else 0
            )
        else:
            self.df["improved_sanitation"] = 0

    def _drop_raw_cols(self):
        """Drop original NFHS column names — downstream uses clean names only."""
        always_keep = {
            "weight", "water_on_premises",
            self.cfg.VAR_PSU,
            self.cfg.VAR_STRATUM,
            self.cfg.VAR_CLUSTER,
            self.cfg.VAR_WEALTH_SCORE,
            self.cfg.VAR_WEALTH_QUINTILE,
        }
        raw_to_drop = [
            c for c in self.df.columns
            if c in set(self.cfg.COLS_TO_LOAD) and c not in always_keep
        ]
        self.df.drop(columns=raw_to_drop, inplace=True, errors="ignore")
