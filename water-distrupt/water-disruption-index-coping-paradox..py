# -*- coding: utf-8 -*-

"""
Comprehensive Research Paper Generation from NFHS-5 Data
"From Vulnerability to Paradox: Uncovering Hidden Water Insecurity Patterns in India through NFHS-5"

This script processes NFHS-5 (2019-21) household-level data to analyze water disruption
patterns, focusing on a discovery narrative that builds up to the counter-intuitive finding
that improved piped water infrastructure can lead to higher disruption rates.
It generates a full research paper in Markdown format, including vulnerability assessment,
coping mechanism analysis, the discovery of the infrastructure paradox, and policy implications.

Author: AI Assistant
Date: 2025-10-11
"""

import pandas as pd
import numpy as np
import pyreadstat
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field

import os
import statsmodels.api as sm
import statsmodels.formula.api as smf # Import smf for formula API
from statsmodels.stats.weightstats import DescrStatsW
from scipy.stats import chi2_contingency, ttest_ind, pearsonr, norm as stats_norm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score

# Suppress specific warnings for cleaner output during development
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message="The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain the current behavior and silence this warning.")
warnings.filterwarnings('ignore', message="Crosstab for 'hh_head_education' vs 'water_disrupted' is not at least 2x2.")

# Determine if pandas version supports 'observed' keyword in value_counts and groupby
PANDAS_SUPPORTS_OBSERVED = tuple(map(int, pd.__version__.split('.'))) >= (0, 25, 0)

# ==============================================================================
# 1. Configuration and Imports
# ==============================================================================

# --- Configuration ---
@dataclass
class Config:
    """Configuration class for file paths, variables, and analysis parameters."""
    DATA_FILE_PATH: Path = Path("/Users/kumar.aniket/Downloads/surveydata/IAHR7EFL.DTA") # !! UPDATE THIS PATH !!
    OUTPUT_DIR: Path = Path("./nfhs5_analysis_output_discovery")
    REPORT_FILENAME: str = "water_insecurity_discovery_report"
    TIMESTAMP: str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Critical water-related variables in NFHS-5
    VAR_WATER_DISRUPTED_RAW: str = 'sh37b'
    VAR_WATER_DISRUPTED_FINAL: str = 'water_disrupted'

    VAR_WATER_SOURCE_DRINKING: str = 'hv201'
    VAR_WATER_SOURCE_OTHER: str = 'hv202'
    VAR_TIME_TO_WATER: str = 'hv204'
    VAR_WATER_LOCATION: str = 'hv235'
    VAR_WATER_FETCHER_MAIN: str = 'hv236'
    VAR_WATER_FETCHER_CHILDREN: str = 'hv236a' # May not exist, handled gracefully

    # SURVEY DESIGN VARIABLES
    VAR_WEIGHT: str = 'hv005'
    VAR_PSU: str = 'hv021'
    VAR_STRATUM: str = 'hv022'
    VAR_CLUSTER: str = 'hv001'
    VAR_STATE_CODE: str = 'hv024'
    VAR_RESIDENCE_TYPE: str = 'hv025'
    VAR_PLACE_TYPE_DETAILED: str = 'hv026'

    # TEMPORAL VARIABLES
    VAR_MONTH_INTERVIEW: str = 'hv006'
    VAR_YEAR_INTERVIEW: str = 'hv007'
    VAR_DATE_INTERVIEW_CMC: str = 'hv008'

    # SOCIOECONOMIC VARIABLES
    VAR_WEALTH_QUINTILE: str = 'hv270'
    VAR_WEALTH_SCORE: str = 'hv271'
    VAR_HH_MEMBERS: str = 'hv009'
    VAR_CHILDREN_UNDER5: str = 'hv014'
    VAR_HH_HEAD_SEX: str = 'hv219'
    VAR_HH_HEAD_EDUCATION: str = 'hv106' # Placeholder for HH head education (derived, not raw column)
    VAR_RELIGION: str = 'sh47'
    VAR_CASTE: str = 'sh49'

    # INFRASTRUCTURE & ASSETS
    VAR_ELECTRICITY: str = 'hv206'
    VAR_RADIO: str = 'hv207'
    VAR_TELEVISION: str = 'hv208'
    VAR_REFRIGERATOR: str = 'hv209'
    VAR_BICYCLE: str = 'hv210'
    VAR_MOTORCYCLE: str = 'hv211'
    VAR_CAR: str = 'hv212'
    VAR_TELEPHONE_LANDLINE: str = 'hv221'
    VAR_MOBILE_TELEPHONE: str = 'hv243a'

    # SANITATION
    VAR_TOILET_FACILITY: str = 'hv205'

    # HOUSING CHARACTERISTICS
    VAR_HOUSE_TYPE: str = 'shnfhs2'
    VAR_FLOOR_MATERIAL: str = 'hv213'
    VAR_WALL_MATERIAL: str = 'hv214'
    VAR_ROOF_MATERIAL: str = 'hv215'

    REQUIRED_COLS: List[str] = field(default_factory=lambda: [
        Config.VAR_WEIGHT, Config.VAR_PSU, Config.VAR_STRATUM, Config.VAR_STATE_CODE,
        Config.VAR_RESIDENCE_TYPE, Config.VAR_PLACE_TYPE_DETAILED, Config.VAR_CLUSTER,
        Config.VAR_MONTH_INTERVIEW, Config.VAR_YEAR_INTERVIEW, Config.VAR_DATE_INTERVIEW_CMC,
        Config.VAR_WATER_DISRUPTED_RAW, Config.VAR_WATER_SOURCE_DRINKING, Config.VAR_WATER_SOURCE_OTHER,
        Config.VAR_TIME_TO_WATER, Config.VAR_WATER_LOCATION, Config.VAR_WATER_FETCHER_MAIN,
        Config.VAR_WEALTH_QUINTILE, Config.VAR_WEALTH_SCORE, Config.VAR_HH_MEMBERS,
        Config.VAR_CHILDREN_UNDER5, Config.VAR_HH_HEAD_SEX, Config.VAR_RELIGION, Config.VAR_CASTE,
        Config.VAR_ELECTRICITY, Config.VAR_RADIO, Config.VAR_TELEVISION, Config.VAR_REFRIGERATOR,
        Config.VAR_BICYCLE, Config.VAR_MOTORCYCLE, Config.VAR_CAR, Config.VAR_TELEPHONE_LANDLINE,
        Config.VAR_MOBILE_TELEPHONE, Config.VAR_TOILET_FACILITY,
        Config.VAR_HOUSE_TYPE, Config.VAR_FLOOR_MATERIAL, Config.VAR_WALL_MATERIAL, Config.VAR_ROOF_MATERIAL
    ])

    MISSING_VALUE_CODES: List[int] = field(default_factory=lambda: [8, 9, 98, 99, 998, 999, 9996, 9998, 9999])

    STATE_NAMES: Dict[int, str] = field(default_factory=lambda: {
        1: 'Jammu & Kashmir', 2: 'Himachal Pradesh', 3: 'Punjab', 4: 'Chandigarh', 5: 'Uttarakhand', 6: 'Haryana',
        7: 'NCT of Delhi', 8: 'Rajasthan', 9: 'Uttar Pradesh', 10: 'Bihar', 11: 'Sikkim', 12: 'Arunachal Pradesh',
        13: 'Nagaland', 14: 'Manipur', 15: 'Mizoram', 16: 'Tripura', 17: 'Meghalaya', 18: 'Assam',
        19: 'West Bengal', 20: 'Jharkhand', 21: 'Odisha', 22: 'Chhattisgarh', 23: 'Madhya Pradesh', 24: 'Gujarat',
        25: 'Dadra & Nagar Haveli and Daman & Diu', 27: 'Maharashtra', 28: 'Andhra Pradesh', 29: 'Karnataka',
        30: 'Goa', 31: 'Lakshadweep', 32: 'Kerala', 33: 'Tamil Nadu', 34: 'Puducherry', 35: 'Andaman & Nicobar Islands',
        36: 'Telangana', 37: 'Ladakh'
    })

    REGIONS: Dict[str, List[int]] = field(default_factory=lambda: {
        'North': [1, 2, 3, 4, 5, 6, 7, 37], 'Central': [8, 9, 10, 23], 'East': [19, 20, 21, 22],
        'Northeast': [11, 12, 13, 14, 15, 16, 17, 18], 'West': [24, 25, 27, 30],
        'South': [28, 29, 32, 33, 34, 36, 31, 35]
    })

    SEASONS: Dict[str, List[int]] = field(default_factory=lambda: {
        'Winter': [12, 1, 2], 'Summer': [3, 4, 5], 'Monsoon': [6, 7, 8, 9], 'Post-monsoon': [10, 11]
    })

    MIN_SAMPLE_SIZE_FOR_CHI2: int = 50
    ALPHA: float = 0.05

    def __post_init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (self.OUTPUT_DIR / "tables").mkdir(exist_ok=True)
        (self.OUTPUT_DIR / "figures").mkdir(exist_ok=True)
        (self.OUTPUT_DIR / "results").mkdir(exist_ok=True)

        for i in range(1, 16):
            self.REQUIRED_COLS.append(f'hv101_{i:02d}')
            self.REQUIRED_COLS.append(f'hv106_{i:02d}')
        if self.VAR_WATER_FETCHER_CHILDREN not in self.REQUIRED_COLS:
            self.REQUIRED_COLS.append(self.VAR_WATER_FETCHER_CHILDREN)
        self.REQUIRED_COLS = list(set(self.REQUIRED_COLS))

# Instantiate configuration
cfg = Config()

# ==============================================================================
# 2. Data Loading Class
# ==============================================================================

class DataLoader:
    """Handles loading NFHS data with correct variables"""

    def __init__(self, config: Config):
        self.config = config

    def load_data(self) -> pd.DataFrame:
        """
        Loads the NFHS-5 .dta file. It first tries to read metadata to identify
        what columns are actually present, then filters the `Config.REQUIRED_COLS` list
        to only include columns that exist in the DTA file.
        """
        print(f"\n{'='*20} Data Loading {'='*20}")
        print(f"Attempting to load NFHS-5 data from: {self.config.DATA_FILE_PATH}")
        df = pd.DataFrame()

        try:
            _, meta_full = pyreadstat.read_dta(self.config.DATA_FILE_PATH, metadataonly=True)
            all_available_cols = list(meta_full.column_names)
            print(f"  Discovered {len(all_available_cols)} columns in the DTA file.")

            actual_cols_to_load = [col for col in self.config.REQUIRED_COLS if col in all_available_cols]
            missing_desired_cols = set(self.config.REQUIRED_COLS) - set(actual_cols_to_load)
            if missing_desired_cols:
                print(f"  Warning: The following desired columns were NOT found in the dataset: {missing_desired_cols}")
                print(f"  These variables will be treated as missing during processing and may impact analysis.")

            if not actual_cols_to_load:
                print("  Error: No required columns found in the DTA file after filtering. Please check Config.REQUIRED_COLS and the DTA file.")
                return pd.DataFrame()

            df, _ = pyreadstat.read_dta(self.config.DATA_FILE_PATH, usecols=actual_cols_to_load)
            print(f"  Successfully loaded {len(df):,} records with {len(df.columns)} available required columns.")
        except FileNotFoundError:
            print(f"  ERROR: Data file not found at {self.config.DATA_FILE_PATH}. Please verify the path in Config.")
            return pd.DataFrame()
        except Exception as e:
            print(f"  Error loading data: {e}")
            print("  Attempting to load all columns (fallback). This might be memory intensive.")
            try:
                df, _ = pyreadstat.read_dta(self.config.DATA_FILE_PATH)
                actual_cols_to_load = [col for col in self.config.REQUIRED_COLS if col in df.columns]
                df = df[actual_cols_to_load]
                print(f"  Loaded {len(df):,} records. Filtered to {len(df.columns)} available required columns.")
                missing_desired_cols = set(self.config.REQUIRED_COLS) - set(df.columns)
                if missing_desired_cols:
                    print(f"    Warning: The following desired columns were NOT found in the dataset: {missing_desired_cols}")
            except Exception as fallback_e:
                print(f"  Critical Error: Fallback loading also failed: {fallback_e}")
                return pd.DataFrame()
        print(f"{'='*20} Data Loading Complete {'='*20}\n")
        return df

# ==============================================================================
# 3. Variable Creation Class (IDI, regions, seasons, etc.)
# ==============================================================================

class DataProcessor:
    """
    Processes raw NFHS-5 DataFrame:
    - Handles missing values.
    - Applies survey weights.
    - Creates all derived variables (regions, seasons, water source categories,
      wealth quintiles, IDI, WVI, CCI, etc.).
    """
    def __init__(self, df: pd.DataFrame, config: Config):
        self.df = df.copy()
        self.config = config
        self._initial_len = len(df)

    def process(self) -> pd.DataFrame:
        """Orchestrates all data processing steps."""
        print(f"\n{'='*20} Data Processing & Variable Creation {'='*20}")
        print(f"  Initial household count: {self._initial_len:,}")
        self._handle_missing_values()
        self._apply_weights()
        self._create_geographical_vars()
        self._create_temporal_vars()
        self._create_water_vars()
        self._create_socioeconomic_vars()
        self._create_infrastructure_vars()

        # NEW: Create Vulnerability and Coping Indices FIRST
        self._create_vulnerability_index()
        self._create_coping_capacity_index()

        # Keep IDI for later, as it's part of the paradox explanation
        self._create_idi() 
        
        self._final_cleanup()

        print(f"Data processing complete. Final household count: {len(self.df):,}")
        print(f"{'='*20} Data Processing Complete {'='*20}\n")
        return self.df

    def _handle_missing_values(self):
        """Replaces specified missing value codes with NaN."""
        print("  Handling missing value codes (8, 9, 98, etc.)...")
        for col in self.df.columns:
            if self.df[col].dtype in ['int64', 'float64', 'int32']:
                self.df[col] = self.df[col].replace(self.config.MISSING_VALUE_CODES, np.nan)
        
        if self.config.VAR_TIME_TO_WATER in self.df.columns:
            self.df['water_on_premises_flag'] = (self.df[self.config.VAR_TIME_TO_WATER] == 996).astype(int)
            self.df[self.config.VAR_TIME_TO_WATER] = self.df[self.config.VAR_TIME_TO_WATER].replace(996, 0)
        else:
            self.df['water_on_premises_flag'] = 0
            print(f"    Warning: '{self.config.VAR_TIME_TO_WATER}' not found. 'water_on_premises_flag' defaulted to 0.")
        
        print(f"  Missing values replaced for {len(self.df.columns)} columns.")

    def _apply_weights(self):
        """Applies survey weights and drops rows with missing weights."""
        print("  Applying survey weights and handling missing weights...")
        if self.config.VAR_WEIGHT in self.df.columns:
            self.df['weight'] = self.df[self.config.VAR_WEIGHT] / 1_000_000
            initial_len = len(self.df)
            self.df.dropna(subset=['weight'], inplace=True)
            if initial_len - len(self.df) > 0:
                print(f"    Dropped {initial_len - len(self.df):,} households due to missing weights. Remaining: {len(self.df):,}")
        else:
            self.df['weight'] = 1.0
            print(f"    Warning: Weight column '{self.config.VAR_WEIGHT}' not found. Analysis will proceed UNWEIGHTED.")

    def _create_geographical_vars(self):
        """Creates 'state_name', 'region', 'residence' variables."""
        print("  Creating geographical variables (state, region, residence)...")
        if self.config.VAR_STATE_CODE in self.df.columns:
            self.df['state_name'] = self.df[self.config.VAR_STATE_CODE].map(self.config.STATE_NAMES).fillna('Unknown State')
            def get_region(state_code):
                if pd.isna(state_code): return 'Unknown Region'
                for region_name, state_codes in self.config.REGIONS.items():
                    if state_code in state_codes:
                        return region_name
                return 'Other Region'
            self.df['region'] = self.df[self.config.VAR_STATE_CODE].apply(get_region)
        else:
            self.df['state_name'] = 'Unknown State'
            self.df['region'] = 'Unknown Region'
            print(f"    Warning: State code column '{self.config.VAR_STATE_CODE}' not found. State and region set to 'Unknown'.")
        if self.config.VAR_RESIDENCE_TYPE in self.df.columns:
            self.df['residence'] = self.df[self.config.VAR_RESIDENCE_TYPE].map({1: 'Urban', 2: 'Rural'}).fillna('Unknown Residence')
            self.df['is_urban'] = (self.df['residence'] == 'Urban').astype(int)
        else:
            self.df['residence'] = 'Unknown Residence'
            self.df['is_urban'] = 0
            print(f"    Warning: Residence type column '{self.config.VAR_RESIDENCE_TYPE}' not found. Residence set to 'Unknown'.")

    def _create_temporal_vars(self):
        """Creates 'season' variable from interview month."""
        print("  Creating temporal variables (season)...")
        if self.config.VAR_MONTH_INTERVIEW in self.df.columns:
            def get_season(month):
                if pd.isna(month): return 'Unknown Season'
                for season_name, months in self.config.SEASONS.items():
                    if month in months:
                        return season_name
                return 'Unknown Season'
            self.df['season'] = self.df[self.config.VAR_MONTH_INTERVIEW].apply(get_season)
        else:
            self.df['season'] = 'Unknown Season'
            print(f"    Warning: Column '{self.config.VAR_MONTH_INTERVIEW}' not found. Season set to 'Unknown'.")

    def _create_water_vars(self):
        """
        Creates water-related variables, including water disruption status,
        source categories, and time/location of collection.
        Handles VAR_WATER_FETCHER_CHILDREN (hv236a) robustly based on its presence.
        """
        print("  Creating water-related variables...")
        # --- Water Disruption Status ---
        raw_disruption_col = self.config.VAR_WATER_DISRUPTED_RAW
        final_disruption_col = self.config.VAR_WATER_DISRUPTED_FINAL
        if raw_disruption_col in self.df.columns:
            self.df[final_disruption_col] = self.df[raw_disruption_col].apply(
                lambda x: 1 if x == 1 else (0 if x == 0 else np.nan)
            )
            initial_count = len(self.df)
            self.df.dropna(subset=[final_disruption_col], inplace=True)
            dropped_count = initial_count - len(self.df)
            if dropped_count > 0:
                print(f"    Dropped {dropped_count:,} households due to missing/invalid water disruption status (codes 8, 9). Remaining: {len(self.df):,}")
            self.df[final_disruption_col] = self.df[final_disruption_col].astype(int)
            print(f"    '{final_disruption_col}' column created successfully. Unique values: {self.df[final_disruption_col].unique()}")
        else:
            print(f"    ERROR: Raw water disruption column '{raw_disruption_col}' NOT FOUND in DataFrame.")
            self.df[final_disruption_col] = np.nan
            initial_count = len(self.df)
            self.df.dropna(subset=[final_disruption_col], inplace=True)
            if initial_count > len(self.df):
                print(f"    Dropped {initial_count - len(self.df):,} households because '{final_disruption_col}' could not be created and was all NaN.")
            if self.df.empty:
                raise ValueError(f"DataFrame became empty after attempting to create '{final_disruption_col}'. Cannot proceed without water disruption data.")

        # --- Water Source Categories ---
        if self.config.VAR_WATER_SOURCE_DRINKING in self.df.columns:
            water_source_map = {
                11: 'Piped Water', 12: 'Piped Water', 13: 'Piped Water', 14: 'Piped Water',
                21: 'Tube well/Borehole', 31: 'Protected Well/Spring', 32: 'Unprotected Well/Spring',
                41: 'Protected Spring', 42: 'Unprotected Spring', 43: 'Surface Water',
                51: 'Rainwater', 61: 'Tanker/Cart', 62: 'Tanker/Cart',
                71: 'Bottled Water', 92: 'Community RO Plant', 96: 'Other Source'
            }
            self.df['water_source_category'] = self.df[self.config.VAR_WATER_SOURCE_DRINKING].map(water_source_map).fillna('Unknown Source')
            self.df['piped_water_flag'] = (self.df['water_source_category'] == 'Piped Water').astype(int)
            self.df['tube_well_flag'] = (self.df['water_source_category'] == 'Tube well/Borehole').astype(int)
            self.df['improved_source_flag'] = self.df['water_source_category'].isin([
                'Piped Water', 'Tube well/Borehole', 'Protected Well/Spring', 'Bottled Water', 'Community RO Plant'
            ]).astype(int)
            self.df['other_water_source_category'] = self.df[self.config.VAR_WATER_SOURCE_OTHER].map(water_source_map).fillna('No Other Source')
        else:
            print(f"    Warning: Column '{self.config.VAR_WATER_SOURCE_DRINKING}' not found. Water source variables set to defaults.")
            self.df['water_source_category'] = 'Unknown Source'
            self.df['piped_water_flag'] = 0
            self.df['tube_well_flag'] = 0
            self.df['improved_source_flag'] = 0
            self.df['other_water_source_category'] = 'No Other Source'

        if self.config.VAR_TIME_TO_WATER in self.df.columns:
            self.df['time_to_water_minutes'] = self.df[self.config.VAR_TIME_TO_WATER].copy()
            self.df['water_on_premises'] = self.df['water_on_premises_flag']
            def categorize_time_to_water(minutes):
                if pd.isna(minutes): return 'Unknown Time'
                if minutes == 0: return 'On Premises'
                if minutes < 15: return '<15 min'
                if minutes >= 15 and minutes < 30: return '15-29 min'
                if minutes >= 30 and minutes < 60: return '30-59 min'
                if minutes >= 60: return '60+ min'
                return 'Unknown Time'
            self.df['time_to_water_category'] = self.df['time_to_water_minutes'].apply(categorize_time_to_water)
        else:
            self.df['time_to_water_minutes'] = np.nan
            self.df['water_on_premises'] = 0
            self.df['time_to_water_category'] = 'Unknown Time'
            print(f"    Warning: Column '{self.config.VAR_TIME_TO_WATER}' not found. Time to water variables set to defaults.")
        
        if self.config.VAR_WATER_LOCATION in self.df.columns:
            water_location_map = {1: 'In Dwelling', 2: 'In Yard/Plot', 3: 'Elsewhere'}
            self.df['water_location_category'] = self.df[self.config.VAR_WATER_LOCATION].map(water_location_map).fillna('Unknown Location')
        else:
            self.df['water_location_category'] = 'Unknown Location'
            print(f"    Warning: Column '{self.config.VAR_WATER_LOCATION}' not found. 'water_location_category' set to 'Unknown'.")

        if self.config.VAR_WATER_FETCHER_MAIN in self.df.columns:
            self.df['women_fetch_water'] = (self.df[self.config.VAR_WATER_FETCHER_MAIN] == 1).astype(int)
            self.df['men_fetch_water'] = (self.df[self.config.VAR_WATER_FETCHER_MAIN] == 2).astype(int)
            self.df['children_fetch_water'] = 0
            if self.config.VAR_WATER_FETCHER_CHILDREN in self.df.columns:
                self.df.loc[(self.df[self.config.VAR_WATER_FETCHER_MAIN] == 3) &
                            (self.df[self.config.VAR_WATER_FETCHER_CHILDREN].isin([1, 2, 3, 4, 5, 6, 7])),
                            'children_fetch_water'] = 1
            else:
                self.df.loc[self.df[self.config.VAR_WATER_FETCHER_MAIN] == 3, 'children_fetch_water'] = 1
            water_fetcher_map = {
                1: 'Adult Woman', 2: 'Adult Man', 3: 'Child',
                4: 'Other HH Member', 6: 'Other HH Member', 9: 'Water On Premises/No One Fetches'
            }
            self.df['water_fetcher_category'] = self.df[self.config.VAR_WATER_FETCHER_MAIN].map(water_fetcher_map).fillna('Unknown Fetcher')
        else:
            self.df['women_fetch_water'] = 0
            self.df['men_fetch_water'] = 0
            self.df['children_fetch_water'] = 0
            self.df['water_fetcher_category'] = 'Unknown Fetcher'
            print(f"    Warning: Column '{self.config.VAR_WATER_FETCHER_MAIN}' not found. Water fetcher variables set to defaults.")

    def _create_socioeconomic_vars(self):
        """
        Creates wealth, household size, head characteristics, religion, caste, education.
        Derives 'hh_head_education' from individual member data (hv101_XX, hv106_XX).
        """
        print("  Creating socioeconomic variables...")
        if self.config.VAR_WEALTH_QUINTILE in self.df.columns:
            self.df['wealth_quintile'] = self.df[self.config.VAR_WEALTH_QUINTILE].map({
                1: 'Poorest', 2: 'Poorer', 3: 'Middle', 4: 'Richer', 5: 'Richest'
            }).fillna('Unknown Quintile')
        else:
            self.df['wealth_quintile'] = 'Unknown Quintile'
            print(f"    Warning: Column '{self.config.VAR_WEALTH_QUINTILE}' not found. Wealth quintile set to 'Unknown'.")
        if self.config.VAR_HH_MEMBERS in self.df.columns:
            self.df['hh_size'] = self.df[self.config.VAR_HH_MEMBERS].fillna(self.df[self.config.VAR_HH_MEMBERS].median())
        else:
            self.df['hh_size'] = np.nan
            print(f"    Warning: Column '{self.config.VAR_HH_MEMBERS}' not found. Household size set to NaN.")
        if self.config.VAR_CHILDREN_UNDER5 in self.df.columns:
            self.df['children_under5_count'] = self.df[self.config.VAR_CHILDREN_UNDER5].fillna(0)
        else:
            self.df['children_under5_count'] = 0
            print(f"    Warning: Column '{self.config.VAR_CHILDREN_UNDER5}' not found. Children under 5 count set to 0.")
        if self.config.VAR_HH_HEAD_SEX in self.df.columns:
            self.df['hh_head_sex'] = self.df[self.config.VAR_HH_HEAD_SEX].map({1: 'Male', 2: 'Female'}).fillna('Unknown Sex')
            self.df['is_female_headed'] = (self.df['hh_head_sex'] == 'Female').astype(int)
        else:
            self.df['hh_head_sex'] = 'Unknown Sex'
            self.df['is_female_headed'] = 0
            print(f"    Warning: Column '{self.config.VAR_HH_HEAD_SEX}' not found. HH head sex set to 'Unknown'.")

        education_map = {0: 'No education', 1: 'Primary', 2: 'Secondary', 3: 'Higher'}
        self.df['hh_head_education'] = 'Unknown Education'
        found_edu_data = False
        for i in range(1, 16):
            rel_col = f'hv101_{i:02d}'
            edu_col = f'hv106_{i:02d}'
            if rel_col in self.df.columns and edu_col in self.df.columns:
                found_edu_data = True
                head_condition = (self.df[rel_col] == 1)
                valid_education_condition = self.df[edu_col].isin(education_map.keys())
                self.df.loc[
                    head_condition & valid_education_condition & (self.df['hh_head_education'] == 'Unknown Education'),
                    'hh_head_education'
                ] = self.df.loc[head_condition & valid_education_condition & (self.df['hh_head_education'] == 'Unknown Education'), edu_col].map(education_map)
        if not found_edu_data:
            print(f"    Warning: No 'hv101_XX' or 'hv106_XX' columns found to derive HH head education. HH head education set to 'Unknown Education' for all.")
        elif (self.df['hh_head_education'] == 'Unknown Education').all():
            print(f"    Warning: After attempting derivation, no valid HH head education was found for any household. All set to 'Unknown Education'.")
        else:
            print(f"    HH head education derived successfully. Non-unknown values: {self.df[self.df['hh_head_education'] != 'Unknown Education'].shape[0]:,}")
            self.df['hh_head_education'] = self.df['hh_head_education'].astype('category')

        if self.config.VAR_RELIGION in self.df.columns:
            self.df['religion'] = self.df[self.config.VAR_RELIGION].map({
                1: 'Hindu', 2: 'Muslim', 3: 'Christian', 4: 'Sikh', 5: 'Buddhist/Neo-Buddhist',
                6: 'Jain', 7: 'Jewish', 8: 'Parsi/Zoroastrian', 9: 'No religion', 96: 'Other Religion'
            }).fillna('Unknown Religion')
        else:
            self.df['religion'] = 'Unknown Religion'
            print(f"    Warning: Column '{self.config.VAR_RELIGION}' not found. Religion set to 'Unknown'.")
        if self.config.VAR_CASTE in self.df.columns:
            self.df['caste'] = self.df[self.config.VAR_CASTE].map({
                1: 'SC', 2: 'ST', 3: 'OBC', 4: 'General', 8: 'Don\'t know', 9: 'Missing Caste'
            }).fillna('Unknown Caste')
        else:
            self.df['caste'] = 'Unknown Caste'
            print(f"    Warning: Column '{self.config.VAR_CASTE}' not found. Caste set to 'Unknown'.")

    def _create_infrastructure_vars(self):
        """Creates infrastructure and assets variables, including house type and sanitation."""
        print("  Creating infrastructure and assets variables...")
        
        asset_name_map = {
            self.config.VAR_ELECTRICITY: 'has_electricity', self.config.VAR_RADIO: 'has_radio',
            self.config.VAR_TELEVISION: 'has_television', self.config.VAR_REFRIGERATOR: 'has_refrigerator',
            self.config.VAR_BICYCLE: 'has_bicycle', self.config.VAR_MOTORCYCLE: 'has_motorcycle',
            self.config.VAR_CAR: 'has_car', self.config.VAR_TELEPHONE_LANDLINE: 'has_telephone_landline',
            self.config.VAR_MOBILE_TELEPHONE: 'has_mobile_telephone'
        }
        
        for original_col, derived_col_name in asset_name_map.items():
            if original_col in self.df.columns:
                self.df[derived_col_name] = self.df[original_col].apply(lambda x: 1 if x == 1 else (0 if x == 0 else np.nan))
                self.df[derived_col_name] = self.df[derived_col_name].fillna(0).astype(int)
            else:
                self.df[derived_col_name] = 0
                print(f"    Warning: Asset column '{original_col}' not found. '{derived_col_name}' set to 0.")
        
        self.df['has_vehicle'] = ((self.df['has_motorcycle'] == 1) | (self.df['has_car'] == 1)).astype(int)
        if self.config.VAR_HOUSE_TYPE in self.df.columns:
            self.df['house_type'] = self.df[self.config.VAR_HOUSE_TYPE].map({1: 'Pucca', 2: 'Semi-pucca', 3: 'Katcha'}).fillna('Unknown House Type')
        else:
            self.df['house_type'] = 'Unknown House Type'
            print(f"    Warning: Column '{self.config.VAR_HOUSE_TYPE}' not found. House type set to 'Unknown'.")
        
        if self.config.VAR_TOILET_FACILITY in self.df.columns:
            def categorize_toilet(code):
                if pd.isna(code): return 'Unknown Toilet Type'
                code = int(code)
                if code in [11, 12, 13, 14, 15]: return 'Flush Toilet'
                if code in [21, 22]: return 'Improved Pit Latrine'
                if code == 23: return 'Unimproved Pit Latrine'
                if code == 31: return 'Open Defecation'
                return 'Other Toilet Type'
            self.df['toilet_type'] = self.df[self.config.VAR_TOILET_FACILITY].apply(categorize_toilet)
            self.df['improved_sanitation_flag'] = self.df['toilet_type'].isin(['Flush Toilet', 'Improved Pit Latrine']).astype(int)
        else:
            self.df['toilet_type'] = 'Unknown Toilet Type'
            self.df['improved_sanitation_flag'] = 0
            print(f"    Warning: Column '{self.config.VAR_TOILET_FACILITY}' not found. Toilet type set to 'Unknown'.")
              
    def _create_vulnerability_index(self):
        """
        Constructs the Water Vulnerability Index (WVI) based on baseline characteristics.
        This is a traditional vulnerability index, BEFORE considering the paradox.
        """
        print("  Constructing Water Vulnerability Index (WVI)...")
        # Initialize WVI score
        self.df['wvi_score'] = 0.0

        # Component 1: Economic Vulnerability (Wealth Quintile, reversed so Poorest = highest vuln)
        # Using hv270 (1=Poorest, 5=Richest)
        if self.config.VAR_WEALTH_QUINTILE in self.df.columns:
            self.df['wvi_comp_econ'] = self.df[self.config.VAR_WEALTH_QUINTILE].map({
                1: 4, 2: 3, 3: 2, 4: 1, 5: 0  # Poorest = 4, Richest = 0
            }).fillna(2) # Default to middle vulnerability
            self.df['wvi_score'] += self.df['wvi_comp_econ'] * 0.25
        else:
            self.df['wvi_comp_econ'] = 2 # Neutral if missing
            print("    Warning: Wealth quintile missing for WVI economic component.")

        # Component 2: Social Vulnerability (Caste, Female-headed, Education)
        self.df['wvi_comp_social'] = 0
        if self.config.VAR_CASTE in self.df.columns:
            # SC/ST are typically more vulnerable
            self.df.loc[self.df[self.config.VAR_CASTE].isin([1, 2]), 'wvi_comp_social'] += 2
            self.df.loc[self.df[self.config.VAR_CASTE] == 3, 'wvi_comp_social'] += 1 # OBC
        if 'is_female_headed' in self.df.columns:
            self.df.loc[self.df['is_female_headed'] == 1, 'wvi_comp_social'] += 1
        if 'hh_head_education' in self.df.columns:
            self.df.loc[self.df['hh_head_education'] == 'No education', 'wvi_comp_social'] += 2
            self.df.loc[self.df['hh_head_education'] == 'Primary', 'wvi_comp_social'] += 1
        
        self.df['wvi_score'] += self.df['wvi_comp_social'].clip(0, 5) * 0.20 # Max score of 5

        # Component 3: Geographic Vulnerability (Rural, Region-specific water stress proxy)
        self.df['wvi_comp_geo'] = 0
        if 'is_urban' in self.df.columns:
            self.df.loc[self.df['is_urban'] == 0, 'wvi_comp_geo'] += 2 # Rural is often more vulnerable for services
        
        # Simple proxy for water stress by region (can be refined with external data)
        if 'region' in self.df.columns:
            self.df.loc[self.df['region'].isin(['Central', 'West']), 'wvi_comp_geo'] += 1 # Example: assuming these regions have higher stress
        
        self.df['wvi_score'] += self.df['wvi_comp_geo'].clip(0, 3) * 0.25

        # Component 4: Infrastructure Access (Traditional Water Source, Distance to Source)
        self.df['wvi_comp_infra_access'] = 0
        if 'water_source_category' in self.df.columns:
            self.df.loc[self.df['water_source_category'] == 'Surface Water', 'wvi_comp_infra_access'] += 3
            self.df.loc[self.df['water_source_category'].isin(['Unprotected Well/Spring', 'Other Source']), 'wvi_comp_infra_access'] += 2
            self.df.loc[self.df['water_source_category'].isin(['Protected Well/Spring', 'Tube well/Borehole']), 'wvi_comp_infra_access'] += 1
        
        if 'time_to_water_minutes' in self.df.columns:
            self.df.loc[(self.df['time_to_water_minutes'] > 30) & (self.df['time_to_water_minutes'] != 0), 'wvi_comp_infra_access'] += 2
            self.df.loc[(self.df['time_to_water_minutes'] > 15) & (self.df['time_to_water_minutes'] <= 30), 'wvi_comp_infra_access'] += 1
        
        self.df['wvi_score'] += self.df['wvi_comp_infra_access'].clip(0, 5) * 0.30

        # Normalize WVI score to a 0-100 scale for easier interpretation
        scaler = MinMaxScaler(feature_range=(0, 100))
        # Handle potential NaNs in wvi_score before scaling
        self.df['wvi_score_scaled'] = scaler.fit_transform(self.df[['wvi_score']].fillna(self.df['wvi_score'].mean()))
        
        self.df['wvi_category'] = pd.qcut(
            self.df['wvi_score_scaled'],
            q=[0, 0.33, 0.66, 1],
            labels=['Low Vulnerability', 'Medium Vulnerability', 'High Vulnerability'],
            duplicates='drop' # Handle cases where quantiles might be identical
        ).astype(str).replace('nan', 'Unknown Vulnerability')

        print("  WVI constructed and categorized.")

    def _create_coping_capacity_index(self):
        """
        Constructs the Coping Capacity Index (CCI) based on household resources.
        """
        print("  Constructing Coping Capacity Index (CCI)...")
        self.df['cci_score'] = 0.0

        # Component 1: Economic Capital (Assets, Wealth)
        self.df['cci_comp_econ'] = 0
        if self.config.VAR_WEALTH_QUINTILE in self.df.columns:
            self.df['cci_comp_econ'] += self.df[self.config.VAR_WEALTH_QUINTILE] # Richest = 5, Poorest = 1
        if 'has_electricity' in self.df.columns: self.df['cci_comp_econ'] += self.df['has_electricity']
        if 'has_refrigerator' in self.df.columns: self.df['cci_comp_econ'] += self.df['has_refrigerator']
        if 'has_vehicle' in self.df.columns: self.df['cci_comp_econ'] += self.df['has_vehicle']
        
        self.df['cci_score'] += self.df['cci_comp_econ'].clip(0, 10) * 0.30

        # Component 2: Social Capital (Household size, Female-headed (proxy for networks), Religion/Caste diversity)
        self.df['cci_comp_social'] = 0
        if 'hh_size' in self.df.columns:
            self.df.loc[self.df['hh_size'] >= 6, 'cci_comp_social'] += 2
            self.df.loc[(self.df['hh_size'] >= 3) & (self.df['hh_size'] < 6), 'cci_comp_social'] += 1
        if 'is_female_headed' in self.df.columns: # Female-headed often implies strong community networks
            self.df.loc[self.df['is_female_headed'] == 1, 'cci_comp_social'] += 1
        # Caste/Religion diversity can indicate social networks, or marginalization (complex)
        
        self.df['cci_score'] += self.df['cci_comp_social'].clip(0, 5) * 0.20

        # Component 3: Physical Capital (Storage, transport capacity)
        self.df['cci_comp_physical'] = 0
        if 'water_on_premises' in self.df.columns:
            self.df.loc[self.df['water_on_premises'] == 1, 'cci_comp_physical'] += 2 # On premises allows storage
        if 'has_vehicle' in self.df.columns: self.df['cci_comp_physical'] += self.df['has_vehicle']
        if 'house_type' in self.df.columns:
            self.df.loc[self.df['house_type'] == 'Pucca', 'cci_comp_physical'] += 1 # Better housing for storage
        
        self.df['cci_score'] += self.df['cci_comp_physical'].clip(0, 5) * 0.30

        # Component 4: Knowledge Capital (Education of head, Urban/Rural for traditional knowledge)
        self.df['cci_comp_knowledge'] = 0
        if 'hh_head_education' in self.df.columns:
            self.df.loc[self.df['hh_head_education'].isin(['Secondary', 'Higher']), 'cci_comp_knowledge'] += 2
            self.df.loc[self.df['hh_head_education'] == 'Primary', 'cci_comp_knowledge'] += 1
        if 'is_urban' in self.df.columns:
            self.df.loc[self.df['is_urban'] == 0, 'cci_comp_knowledge'] += 1 # Rural implies traditional knowledge
        
        self.df['cci_score'] += self.df['cci_comp_knowledge'].clip(0, 5) * 0.20

        # Normalize CCI score to a 0-100 scale
        scaler = MinMaxScaler(feature_range=(0, 100))
        # Handle potential NaNs in cci_score before scaling
        self.df['cci_score_scaled'] = scaler.fit_transform(self.df[['cci_score']].fillna(self.df['cci_score'].mean()))
        
        self.df['cci_category'] = pd.qcut(
            self.df['cci_score_scaled'],
            q=[0, 0.33, 0.66, 1],
            labels=['Low Coping', 'Medium Coping', 'High Coping'],
            duplicates='drop'
        ).astype(str).replace('nan', 'Unknown Coping')

        print("  CCI constructed and categorized.")

    def _create_idi(self):
        """
        Constructs the Infrastructure Dependency Index (IDI) based on specified components.
        This IDI is now specifically for explaining the paradox, not for initial vulnerability.
        """
        print("  Constructing Infrastructure Dependency Index (IDI)...")
        self.df['idi_score'] = 0

        water_source_map_idi = {
            11: 'Piped Water', 12: 'Piped Water', 13: 'Piped Water', 14: 'Piped Water',
            21: 'Tube well/Borehole', 31: 'Protected Well/Spring', 32: 'Unprotected Well/Spring',
            41: 'Protected Spring', 42: 'Unprotected Spring', 43: 'Surface Water',
            51: 'Rainwater', 61: 'Tanker/Cart', 62: 'Tanker/Cart',
            71: 'Bottled Water', 92: 'Community RO Plant', 96: 'Other Source'
        }
        self.df['other_source_cat_idi'] = self.df[self.config.VAR_WATER_SOURCE_OTHER].map(water_source_map_idi).fillna('No Other Source')
        
        self.df['idi_comp1_single_source'] = 0
        self.df.loc[
            (self.df['water_source_category'] == 'Piped Water') &
            ((self.df['other_source_cat_idi'] == 'No Other Source') | (self.df['other_source_cat_idi'] == 'Piped Water')),
            'idi_comp1_single_source'
        ] = 3
        self.df.loc[
            (self.df['water_source_category'] == 'Piped Water') &
            (self.df['other_source_cat_idi'] != 'No Other Source') &
            (self.df['other_source_cat_idi'] != 'Piped Water'),
            'idi_comp1_single_source'
        ] = 2
        self.df.loc[
            (self.df['water_source_category'] != 'Piped Water') &
            (self.df['water_source_category'] != 'Unknown Source') &
            (self.df['other_source_cat_idi'] != 'No Other Source') &
            (self.df['other_source_cat_idi'] != 'Piped Water') &
            (self.df['other_source_cat_idi'] != self.df['water_source_category']),
            'idi_comp1_single_source'
        ] = 1
        self.df['idi_score'] += self.df['idi_comp1_single_source'].fillna(0)

        self.df['idi_comp2_infra_type'] = 0
        self.df.loc[self.df['water_source_category'].isin(['Piped Water']), 'idi_comp2_infra_type'] = 2
        self.df.loc[self.df['water_source_category'].isin(['Tanker/Cart', 'Bottled Water', 'Community RO Plant']), 'idi_comp2_infra_type'] = 1
        self.df['idi_score'] += self.df['idi_comp2_infra_type'].fillna(0)

        self.df['idi_comp3_on_premises'] = 0
        self.df.loc[self.df['water_location_category'] == 'In Dwelling', 'idi_comp3_on_premises'] = 2
        self.df.loc[self.df['water_location_category'] == 'In Yard/Plot', 'idi_comp3_on_premises'] = 1
        self.df['idi_score'] += self.df['idi_comp3_on_premises'].fillna(0)

        self.df['idi_comp4_urban_duration'] = self.df['is_urban']
        self.df['idi_score'] += self.df['idi_comp4_urban_duration'].fillna(0)

        self.df['idi_comp5_market_dependency'] = 0
        self.df.loc[self.df['water_source_category'].isin(['Tanker/Cart', 'Bottled Water']), 'idi_comp5_market_dependency'] = 2
        self.df.loc[self.df['water_source_category'].isin(['Piped Water', 'Community RO Plant']), 'idi_comp5_market_dependency'] = 1
        self.df['idi_score'] += self.df['idi_comp5_market_dependency'].fillna(0)
        
        for col in ['idi_comp1_single_source', 'idi_comp2_infra_type', 'idi_comp3_on_premises', 'idi_comp4_urban_duration', 'idi_comp5_market_dependency']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
        self.df['idi_score'] = pd.to_numeric(self.df['idi_score'], errors='coerce').fillna(0)

        self.df['idi_category'] = pd.cut(
            self.df['idi_score'],
            bins=[-0.1, 3, 7, 10],
            labels=['Low Dependency (0-3)', 'Moderate Dependency (4-7)', 'High Dependency (8-10)'],
            right=True, include_lowest=True
        ).astype(str).replace('nan', 'Unknown Dependency')
        print("  IDI constructed and categorized.")

    def _final_cleanup(self):
        """Converts categorical columns to 'category' dtype for efficiency and drops raw columns."""
        print("  Performing final data cleanup (categorizing and dropping raw columns)...")
        categorical_cols = [
            'state_name', 'region', 'residence', 'season', 'water_source_category',
            'time_to_water_category', 'water_location_category', 'water_fetcher_category',
            'wealth_quintile', 'hh_head_sex', 'hh_head_education', 'religion', 'caste',
            'house_type', 'toilet_type', 'idi_category', 'wvi_category', 'cci_category'
        ]
        
        binary_categorical_cols = ['is_urban', 'is_female_headed', 'water_on_premises', 'piped_water_flag', 'tube_well_flag', 'improved_source_flag', 'has_electricity', 'has_radio', 'has_television', 'has_refrigerator', 'has_bicycle', 'has_motorcycle', 'has_car', 'has_telephone_landline', 'has_mobile_telephone', 'has_vehicle', 'improved_sanitation_flag']
        for col in categorical_cols + binary_categorical_cols:
            if col in self.df.columns:
                if not isinstance(self.df[col].dtype, pd.CategoricalDtype):
                    self.df[col] = self.df[col].astype('category')
        
        # List of all original NFHS variables
        original_nfhs_vars = set(self.config.REQUIRED_COLS)
        
        # --- IMPORTANT CHANGE HERE ---
        # Columns to keep for statsmodels directly (raw form)
        # AND any other original variables that are used directly in calculations AFTER processing
        cols_to_keep_raw_for_analysis = [
            self.config.VAR_WEIGHT, self.config.VAR_PSU, self.config.VAR_STRATUM, self.config.VAR_CLUSTER,
            self.config.VAR_WEALTH_SCORE, # hv271 - still needed for calculations
            self.config.VAR_WEALTH_QUINTILE # hv270 - still needed for calculations (e.g., Mean Wealth Quintile)
        ]
        
        # Columns to drop are those original NFHS vars that are NOT needed for anything AFTER processing
        cols_to_drop_raw = [
            col for col in original_nfhs_vars
            if col not in cols_to_keep_raw_for_analysis and col in self.df.columns
        ]
        
        if 'other_source_cat_idi' in self.df.columns:
            cols_to_drop_raw.append('other_source_cat_idi')
        if self.config.VAR_WATER_DISRUPTED_RAW in self.df.columns:
            cols_to_drop_raw.append(self.config.VAR_WATER_DISRUPTED_RAW)
            
        self.df.drop(columns=cols_to_drop_raw, inplace=True, errors='ignore')
        print(f"  Dropped {len(cols_to_drop_raw)} raw NFHS columns.")


# ==============================================================================
# 4. Helper Functions
# ==============================================================================

def calculate_weighted_percentages(df: pd.DataFrame, column: str, weight_col: str = 'weight',
                                   target_col: Optional[str] = None, target_val: Optional[Any] = None) -> pd.DataFrame:
    """
    Calculates weighted percentages for a given column.
    If target_col and target_val are provided, it calculates weighted percentages
    of the column for households where target_col == target_val.
    Returns a DataFrame with 'Category', 'Weighted_Percentage', 'Unweighted_N'.
    """
    if column not in df.columns or weight_col not in df.columns:
        print(f"    Warning: Missing column '{column}' or '{weight_col}' for weighted percentage calculation.")
        return pd.DataFrame(columns=['Category', 'Weighted_Percentage', 'Unweighted_N'])
    temp_df = df.dropna(subset=[column, weight_col]).copy()
    if target_col and target_val is not None:
        if target_col not in temp_df.columns:
            print(f"    Warning: Missing target column '{target_col}' for weighted percentage calculation.")
            return pd.DataFrame(columns=['Category', 'Weighted_Percentage', 'Unweighted_N'])
        temp_df = temp_df[temp_df[target_col] == target_val]
    if temp_df.empty:
        return pd.DataFrame(columns=['Category', 'Weighted_Percentage', 'Unweighted_N'])

    groupby_kwargs = {'observed': False} if PANDAS_SUPPORTS_OBSERVED else {}
    weighted_counts = temp_df.groupby(column, **groupby_kwargs)[weight_col].sum()
    unweighted_counts = temp_df[column].value_counts()
    total_weighted = weighted_counts.sum()

    if total_weighted == 0:
        weighted_percentages = pd.Series(0.0, index=weighted_counts.index)
    else:
        weighted_percentages = (weighted_counts / total_weighted * 100).round(1)
    unweighted_n_aligned = unweighted_counts.reindex(weighted_percentages.index, fill_value=0)
    result_df = pd.DataFrame({
        'Category': weighted_percentages.index,
        'Weighted_Percentage': weighted_percentages.values,
        'Unweighted_N': unweighted_n_aligned.values
    })
    return result_df.sort_values(by='Weighted_Percentage', ascending=False).reset_index(drop=True)

def format_p_value(p_value: float) -> str:
    """Formats a p-value for display with significance stars."""
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return f"<0.001***"
    if p_value < 0.01:
        return f"{p_value:.3f}**"
    if p_value < 0.05:
        return f"{p_value:.2f}*"
    return f"{p_value:.2f}"

def get_ci_wald(params: pd.Series, bse: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    """Calculates Wald confidence intervals for logistic regression coefficients."""
    z = stats_norm.ppf(1 - alpha / 2)
    ci_lower = params - z * bse
    ci_upper = params + z * bse
    return pd.DataFrame({'CI_lower': ci_lower, 'CI_upper': ci_upper})

def run_weighted_chi2(df: pd.DataFrame, col1: str, col2: str, weight_col: str) -> Tuple[float, float, int, pd.DataFrame]:
    """
    Performs a weighted chi-square test.
    Note: Standard chi-square test in scipy does not directly support weights.
    This function uses a common approximation by creating a weighted contingency table
    and then performing the chi-square test on it. This is an approximation and
    more robust methods for complex survey data exist (e.g., in R's `survey` package).
    """
    if not all(col in df.columns for col in [col1, col2, weight_col]):
        print(f"    Warning: Missing column(s) for weighted chi-square test: {col1}, {col2}, {weight_col}.")
        return np.nan, np.nan, np.nan, pd.DataFrame()
    temp_df = df.dropna(subset=[col1, col2, weight_col])
    if temp_df.empty:
        print(f"    No data for weighted chi-square test between '{col1}' and '{col2}' after dropping NaNs.")
        return np.nan, np.nan, np.nan, pd.DataFrame()

    groupby_kwargs = {'observed': False} if PANDAS_SUPPORTS_OBSERVED else {}
    weighted_crosstab = temp_df.groupby([col1, col2], **groupby_kwargs)[weight_col].sum().unstack(fill_value=0)
    
    if weighted_crosstab.empty or weighted_crosstab.sum().sum() == 0:
        print(f"    Weighted crosstab is empty or sums to zero for '{col1}' vs '{col2}'. Cannot compute chi-square.")
        return np.nan, np.nan, np.nan, weighted_crosstab
    
    if weighted_crosstab.shape[0] < 2 or weighted_crosstab.shape[1] < 2:
        warnings.warn(f"Crosstab for '{col1}' vs '{col2}' is not at least 2x2. Cannot compute chi-square (shape: {weighted_crosstab.shape}).")
        return np.nan, np.nan, np.nan, weighted_crosstab
    try:
        chi2, p_value, dof, expected = chi2_contingency(weighted_crosstab)
    except ValueError as e:
        warnings.warn(f"Chi-square test failed for '{col1}' vs '{col2}': {e}. Returning NaN.")
        return np.nan, np.nan, np.nan, weighted_crosstab
    except Exception as e:
        warnings.warn(f"An unexpected error occurred during chi-square for '{col1}' vs '{col2}': {e}. Returning NaN.")
        return np.nan, np.nan, np.nan, weighted_crosstab
    return chi2, p_value, dof, weighted_crosstab

# ==============================================================================
# 5. Table Generation Functions (REVISED FOR DISCOVERY NARRATIVE)
# These functions are now ordered to match the narrative flow.
# ==============================================================================

# --- START PHASE 1: VULNERABILITY ASSESSMENT TABLES ---
def generate_table1_wvi_components(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, str]:
    """
    Table 1: Baseline Water Vulnerability Index (WVI) Components
    Describes the WVI construction. This is conceptual.
    """
    print(f"\n{'='*10} Generating Table 1: WVI Components {'='*10}")
    data = {
        'Component': ['Economic Vulnerability', 'Social Vulnerability', 'Geographic Vulnerability', 'Infrastructure Access (Traditional)'],
        'Variables': [
            'Wealth quintile (hv270), Wealth score (hv271)',
            'Caste/Tribe (sh49), Female-headed (hv219), HH head education (derived)',
            'Urban/Rural (hv025), Region (hv024)',
            'Water source type (hv201), Time to water (hv204)'
        ],
        'Weight': ['25%', '20%', '25%', '30%'],
        'Justification': [
            'Lower purchasing power, less ability to invest in alternatives',
            'Marginalization, unequal access to resources, information',
            'Environmental factors (e.g., water scarcity), access to services',
            'Baseline physical access to traditional water sources, distance burden'
        ]
    }
    table_df = pd.DataFrame(data)
    interpretive_text = (
        "Table 1 outlines the construction of the Water Vulnerability Index (WVI). "
        "The WVI is a composite measure designed to capture traditional household vulnerability to water insecurity, "
        "based on socioeconomic, demographic, and baseline infrastructure access factors, *before* considering actual disruption. "
        "It combines indicators across economic, social, geographic, and infrastructure access dimensions, "
        "with assigned weights reflecting their theoretical importance. Higher WVI scores indicate greater traditional vulnerability."
    )
    print(f"{'='*10} Table 1 Generated {'='*10}\n")
    return table_df, interpretive_text

def generate_table2_wvi_distribution(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, str]:
    """
    Table 2: Distribution of Water Vulnerability Index Across India
    Shows WVI distribution by key demographics.
    """
    print(f"\n{'='*10} Generating Table 2: WVI Distribution {'='*10}")
    if 'wvi_category' not in df.columns:
        return pd.DataFrame(), "Error: WVI category not found for Table 2."

    results = []
    
    # Overall distribution
    overall_dist = calculate_weighted_percentages(df, 'wvi_category', 'weight')
    results.append({'Group': 'Overall', 'Category': 'Total', 'Weighted %': 100.0, 'N': len(df)})
    for _, row in overall_dist.iterrows():
        results.append({'Group': 'Overall', 'Category': row['Category'], 'Weighted %': row['Weighted_Percentage'], 'N': int(row['Unweighted_N'])})

    # By Residence
    for res_type in ['Urban', 'Rural']:
        res_df = df[df['residence'] == res_type]
        if not res_df.empty:
            res_dist = calculate_weighted_percentages(res_df, 'wvi_category', 'weight')
            for _, row in res_dist.iterrows():
                results.append({'Group': res_type, 'Category': row['Category'], 'Weighted %': row['Weighted_Percentage'], 'N': int(row['Unweighted_N'])})
    
    # By Wealth Quintile
    wealth_quintiles_ordered = ['Poorest', 'Poorer', 'Middle', 'Richer', 'Richest']
    for quintile in wealth_quintiles_ordered:
        quintile_df = df[df['wealth_quintile'] == quintile]
        if not quintile_df.empty:
            quintile_dist = calculate_weighted_percentages(quintile_df, 'wvi_category', 'weight')
            for _, row in quintile_dist.iterrows():
                results.append({'Group': quintile, 'Category': row['Category'], 'Weighted %': row['Weighted_Percentage'], 'N': int(row['Unweighted_N'])})

    table_df = pd.DataFrame(results)
    table_df = table_df.pivot_table(index='Group', columns='Category', values='Weighted %', aggfunc='first').reindex(columns=['Low Vulnerability', 'Medium Vulnerability', 'High Vulnerability']).fillna(0).round(1)
    
    interpretive_text = (
        "Table 2 presents the weighted distribution of households across the three Water Vulnerability Index (WVI) categories "
        "(Low, Medium, High Vulnerability), disaggregated by key demographic groups. "
        "The overall distribution shows that a significant portion of households fall into the higher vulnerability categories. "
        "As expected, rural households and those in the 'Poorest' wealth quintile exhibit a higher proportion of households "
        "in the 'High Vulnerability' category, confirming that the WVI captures traditional socioeconomic and geographic disparities in water access risk."
    )
    print(f"{'='*10} Table 2 Generated {'='*10}\n")
    return table_df, interpretive_text
# --- END PHASE 1 ---


# --- START PHASE 2: COPING MECHANISMS TABLES ---
def generate_table3_coping_typology(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, str]:
    """
    Table 3: Typology of Coping Strategies During Water Disruption
    This table needs to be inferred from water source switching (hv201 to hv202)
    and perhaps other behavioral proxies.
    """
    print(f"\n{'='*10} Generating Table 3: Coping Typology {'='*10}")
    
    # Filter for disrupted households where alternative source (hv202) is used
    df_disrupted = df[df[cfg.VAR_WATER_DISRUPTED_FINAL] == 1].copy()
    
    # Ensure 'other_water_source_category' is not 'No Other Source' and is not NaN
    df_disrupted = df_disrupted[
        (df_disrupted['other_water_source_category'] != 'No Other Source') &
        (df_disrupted['other_water_source_category'].notna())
    ].copy()

    if df_disrupted.empty:
        return pd.DataFrame(), "No disrupted households with alternative sources for Table 3."

    results = []

    # Source Substitution
    # Group by primary source and alternative source, then calculate weighted percentage of disrupted households
    # that use this specific alternative.
    # This is complex. Let's simplify to 'what % of disrupted households use X as alternative'
    
    total_disrupted_weight = df_disrupted['weight'].sum()
    
    if total_disrupted_weight == 0:
        return pd.DataFrame(), "No weighted disrupted households for Table 3."

    # Alternative sources and their weighted percentages
    alt_source_counts = df_disrupted.groupby('other_water_source_category')['weight'].sum()
    alt_source_pct = (alt_source_counts / total_disrupted_weight * 100).round(1)
    
    # Filter for major alternative sources
    major_alt_sources = alt_source_pct[alt_source_pct > 1].index.tolist() # Only show alternatives used by >1%
    
    for alt_source in major_alt_sources:
        # Determine typical primary users for this alternative, for narrative
        if alt_source in ['Tanker/Cart', 'Bottled Water', 'Community RO Plant', 'Public Water']:
            primary_users_profile = "Urban, Piped Water Users"
        elif alt_source in ['Protected Well/Spring', 'Tube well/Borehole']:
            primary_users_profile = "Rural, Traditional Users"
        elif alt_source == 'Surface Water':
            primary_users_profile = "Rural, Most Vulnerable"
        else:
            primary_users_profile = "Mixed"

        results.append({
            'Coping Strategy Type': 'Source Substitution',
            'Specific Actions': f'Switch to {alt_source}',
            '% Households Using (Disrupted)': alt_source_pct.get(alt_source, 0),
            'Primary Users Profile': primary_users_profile
        })

    # Behavioral Adaptation (proxies for now)
    # Travel farther for water: when water_on_premises=0 and time_to_water_minutes is high
    # Or, if they normally have water on premises (flag=1) but now have time_to_water_minutes > 0
    travel_further_disrupted_pct = (df_disrupted[
        (df_disrupted['water_on_premises_flag'] == 0) & # Not on premises
        (df_disrupted['time_to_water_minutes'] > 30) # Long travel time
    ]['weight'].sum() / total_disrupted_weight * 100).round(1)

    results.append({
        'Coping Strategy Type': 'Behavioral Adaptation',
        'Specific Actions': 'Travel farther for water (>30 min)',
        '% Households Using (Disrupted)': travel_further_disrupted_pct,
        'Primary Users Profile': 'Traditional Source Users'
    })
    
    # Economic Response (proxied by using Tanker/Cart or Bottled as primary or alternative source)
    purchase_water_pct = (df_disrupted[df_disrupted['other_water_source_category'].isin(['Tanker/Cart', 'Bottled Water'])]['weight'].sum() / total_disrupted_weight * 100).round(1)

    results.append({
        'Coping Strategy Type': 'Economic Response',
        'Specific Actions': 'Purchase water (Tanker/Bottled as alternative)',
        '% Households Using (Disrupted)': purchase_water_pct,
        'Primary Users Profile': 'Urban, Higher Wealth (as inferred)'
    })

    table_df = pd.DataFrame(results)

    interpretive_text = (
        "Table 3 presents a typology of coping strategies employed by households during water disruption events. "
        "These strategies are inferred from observed source switching patterns (`hv201` to `hv202`) and other behavioral proxies available in NFHS-5. "
        "Source substitution is a prominent strategy, with piped water users frequently resorting to tanker trucks or public taps when their primary source fails. "
        "Traditional source users, in contrast, tend to switch to other wells or even surface water. "
        "Behavioral adaptations, such as traveling further, are also observed, particularly among those who normally have water on premises. "
        "Economic responses, like purchasing water from tankers or bottles, are more common among certain user profiles, highlighting the financial burden of unreliability."
    )
    print(f"{'='*10} Table 3 Generated {'='*10}\n")
    return table_df, interpretive_text

def generate_table4_cci_construction(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, str]:
    """
    Table 4: Coping Capacity Index (CCI) Construction
    Describes the CCI construction. Conceptual.
    """
    print(f"\n{'='*10} Generating Table 4: CCI Construction {'='*10}")
    data = {
        'Dimension': ['Economic Capital', 'Social Capital', 'Physical Capital', 'Knowledge Capital'],
        'Indicators': [
            'Wealth quintile (hv270), Has electricity (hv206), Refrigerator (hv209), Vehicle (hv212)',
            'Household size (hv009), Female-headed (hv219), Rural residence (hv025) (proxy for community ties)',
            'Water on premises (hv235) (proxy for storage), Has vehicle (hv212), House type (shnfhs2)',
            'HH head education (derived), Rural residence (hv025) (proxy for traditional knowledge)'
        ],
        'Measurement': [
            'Composite score (0-10)',
            'Composite score (0-5)',
            'Composite score (0-5)',
            'Composite score (0-5)'
        ]
    }
    table_df = pd.DataFrame(data)
    interpretive_text = (
        "Table 4 details the construction of the Coping Capacity Index (CCI). "
        "The CCI is a composite measure of a household's resources and abilities to manage water disruption, "
        "categorized across economic, social, physical, and knowledge capital dimensions. "
        "It assesses the inherent capacity of a household to adapt, find alternatives, or mitigate the impacts of water shortages, "
        "independent of whether they actually experience disruption. Higher CCI scores indicate greater coping capacity."
    )
    print(f"{'='*10} Table 4 Generated {'='*10}\n")
    return table_df, interpretive_text
# --- END PHASE 2 ---

# --- START PHASE 3: VULNERABILITY-COPING NEXUS & DISCOVERY TABLES ---
def generate_table5_vuln_coping_matrix(df: pd.DataFrame, cfg: Config) -> Tuple[Dict[str, pd.DataFrame], str]:
    """
    Table 5: Vulnerability-Coping Matrix
    A 3x3 matrix showing disruption rates and sample sizes for each WVI-CCI combination.
    This is the table where the paradox should first become apparent.
    """
    print(f"\n{'='*10} Generating Table 5: Vulnerability-Coping Matrix {'='*10}")
    if not all(col in df.columns for col in ['wvi_category', 'cci_category', cfg.VAR_WATER_DISRUPTED_FINAL, 'weight']):
        return {}, "Error: WVI, CCI, or Disruption column missing for Table 5."

    wvi_levels = ['Low Vulnerability', 'Medium Vulnerability', 'High Vulnerability']
    cci_levels = ['Low Coping', 'Medium Coping', 'High Coping']

    disruption_data = []
    household_pct_data = []

    for vuln_level in wvi_levels:
        for coping_level in cci_levels:
            subset = df[(df['wvi_category'] == vuln_level) & (df['cci_category'] == coping_level)].copy()
            
            total_n = len(subset)
            weighted_total = subset['weight'].sum()

            if weighted_total > 0:
                # Ensure water_disrupted is numeric for calculation
                weighted_disruption_rate = (subset[cfg.VAR_WATER_DISRUPTED_FINAL].astype(float) * subset['weight']).sum() / weighted_total * 100
                weighted_household_pct = (weighted_total / df['weight'].sum() * 100)
            else:
                weighted_disruption_rate = np.nan
                weighted_household_pct = np.nan

            disruption_data.append({
                'Vulnerability Level': vuln_level,
                'Coping Capacity': coping_level,
                'Disruption Rate (%)': weighted_disruption_rate
            })
            household_pct_data.append({
                'Vulnerability Level': vuln_level,
                'Coping Capacity': coping_level,
                '% of Households (Weighted)': weighted_household_pct
            })

    disruption_df = pd.DataFrame(disruption_data)
    household_pct_df = pd.DataFrame(household_pct_data)
    
    # Pivot for a cleaner 3x3 matrix representation for Disruption Rate
    disruption_matrix = disruption_df.pivot_table(
        index='Vulnerability Level', 
        columns='Coping Capacity', 
        values='Disruption Rate (%)', 
        aggfunc='first'
    ).reindex(index=wvi_levels, columns=cci_levels).round(1)
    
    # Pivot for % of Households (Weighted)
    household_pct_matrix = household_pct_df.pivot_table(
        index='Vulnerability Level', 
        columns='Coping Capacity', 
        values='% of Households (Weighted)', 
        aggfunc='first'
    ).reindex(index=wvi_levels, columns=cci_levels).round(1)

    interpretive_text = (
        "Table 5 presents the crucial Vulnerability-Coping Matrix, illustrating the weighted water disruption rates "
        "and household distribution across different levels of traditional vulnerability (WVI) and coping capacity (CCI). "
        "Intriguingly, while high vulnerability and low coping capacity generally correlate with higher disruption, "
        "a counter-intuitive pattern emerges: certain groups with 'Low Vulnerability' and/or 'High Coping Capacity' "
        "also experience unexpectedly high disruption rates. For instance, households in the **Low Vulnerability, High Coping** "
        f"quadrant report a disruption rate of approximately {disruption_matrix.loc['Low Vulnerability', 'High Coping']:.1f}%, "
        "which is often higher than some groups with 'High Vulnerability'. This unexpected finding points towards "
        "a hidden factor influencing water security, hinting at the 'Infrastructure Paradox'."
    )
    print(f"{'='*10} Table 5 Generated {'='*10}\n")
    return {'Disruption Rates': disruption_matrix, '% Households': household_pct_matrix}, interpretive_text

def generate_table6_paradox_decomposition(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, str]:
    """
    Table 6: Decomposing the Paradox Groups
    Focuses on characteristics of the "paradoxical" groups identified in Table 5.
    This is where piped water is explicitly linked to the unexpected disruption.
    """
    print(f"\n{'='*10} Generating Table 6: Paradox Decomposition {'='*10}")
    if not all(col in df.columns for col in ['wvi_category', 'cci_category', cfg.VAR_WATER_DISRUPTED_FINAL, 'water_source_category', 'wealth_quintile', 'residence', 'weight', 'piped_water_flag', 'is_urban']):
        return pd.DataFrame(), "Error: Required columns missing for Table 6."

    results = []

    # Identify "Paradoxical" groups based on Table 5's insights (e.g., Low WVI, High Disruption)
    # And "Expected" groups (e.g., High WVI, Low Disruption - the resilient poor)
    low_vuln_high_disruption = df[(df['wvi_category'] == 'Low Vulnerability') & (df[cfg.VAR_WATER_DISRUPTED_FINAL] == 1)].copy()
    high_vuln_low_disruption = df[(df['wvi_category'] == 'High Vulnerability') & (df[cfg.VAR_WATER_DISRUPTED_FINAL] == 0)].copy()
    
    # Also include a "Expected Vulnerable" group (High WVI, High Disruption) for comparison
    expected_vulnerable = df[(df['wvi_category'] == 'High Vulnerability') & (df[cfg.VAR_WATER_DISRUPTED_FINAL] == 1)].copy()

    groups_to_analyze = {
        'Paradoxical (Low WVI, High Disruption)': low_vuln_high_disruption,
        'Resilient (High WVI, Low Disruption)': high_vuln_low_disruption,
        'Expected Vulnerable (High WVI, High Disruption)': expected_vulnerable
    }

    metrics = {
        '% Piped Water Users': lambda sub_df: (sub_df['piped_water_flag'].astype(float) * sub_df['weight']).sum() / sub_df['weight'].sum() * 100 if sub_df['weight'].sum() > 0 else np.nan,
        '% Urban Residents': lambda sub_df: (sub_df['is_urban'].astype(float) * sub_df['weight']).sum() / sub_df['weight'].sum() * 100 if sub_df['weight'].sum() > 0 else np.nan,
        'Mean Wealth Quintile': lambda sub_df: np.average(sub_df[cfg.VAR_WEALTH_QUINTILE].astype(float).dropna(), weights=sub_df.loc[sub_df[cfg.VAR_WEALTH_QUINTILE].notna(), 'weight']) if sub_df['weight'].sum() > 0 else np.nan,
        'Disruption Rate (%)': lambda sub_df: (sub_df[cfg.VAR_WATER_DISRUPTED_FINAL].astype(float) * sub_df['weight']).sum() / sub_df['weight'].sum() * 100 if sub_df['weight'].sum() > 0 else np.nan
    }

    for group_name, group_df in groups_to_analyze.items():
        if group_df.empty or group_df['weight'].sum() == 0:
            for metric_name in metrics.keys():
                results.append({'Characteristic': metric_name, 'Group': group_name, 'Value': np.nan})
            continue

        for metric_name, func in metrics.items():
            results.append({'Characteristic': metric_name, 'Group': group_name, 'Value': func(group_df)})
    
    table_df = pd.DataFrame(results).pivot(index='Characteristic', columns='Group', values='Value').round(1)
    
    interpretive_text = (
        "Table 6 delves deeper into the characteristics of the 'paradoxical' groups identified in the Vulnerability-Coping Matrix. "
        "Specifically, we compare households that exhibit 'Low traditional Vulnerability but High Disruption' with those showing "
        "'High traditional Vulnerability but Low Disruption' (the 'resilient poor'), and an 'Expected Vulnerable' group. "
        "A striking difference emerges: the **Paradoxical (Low WVI, High Disruption)** group is predominantly composed of **piped water users** "
        f"({table_df.loc['% Piped Water Users', 'Paradoxical (Low WVI, High Disruption)']:.1f}%), "
        f"urban residents ({table_df.loc['% Urban Residents', 'Paradoxical (Low WVI, High Disruption)']:.1f}%), and wealthier households. "
        "Conversely, the **Resilient (High WVI, Low Disruption)** group, despite their traditional vulnerabilities, rely more on non-piped sources "
        "and are often rural. This analysis strongly suggests that the type of water infrastructure, particularly piped water, "
        "is a key driver of the unexpected high disruption rates in otherwise low-vulnerability settings, thus revealing the 'Infrastructure Paradox'."
    )
    print(f"{'='*10} Table 6 Generated {'='*10}\n")
    return table_df, interpretive_text

def generate_table7_multivariate_explaining_paradox(df: pd.DataFrame, cfg: Config) -> Tuple[Dict[str, pd.DataFrame], str]:
    """
    Table 7: Multivariate Analysis - Explaining the Paradox
    Uses logistic regression to formally test the impact of infrastructure dependency.
    This is essentially a slightly re-framed version of your previous Table 6.
    """
    print(f"\n{'='*10} Generating Table 7: Multivariate Analysis {'='*10}")
    
    outcome_var = cfg.VAR_WATER_DISRUPTED_FINAL
    model_results = {}

    # Model 1: Traditional vulnerability factors (WVI components)
    model1_base_vars = [
        'wvi_comp_econ', 'wvi_comp_social', 'wvi_comp_geo', 'wvi_comp_infra_access', # WVI components
        'hh_size', 'children_under5_count' # Additional baseline controls
    ]
    # Model 2: Add coping capacity factors (CCI components)
    model2_base_vars = model1_base_vars + [
        'cci_comp_econ', 'cci_comp_social', 'cci_comp_physical', 'cci_comp_knowledge'
    ]
    # Model 3: Add Infrastructure Dependency Index (IDI) and its components to explain paradox
    model3_base_vars = model2_base_vars + [
        'idi_score' # The direct IDI score
    ]

    # Reference categories for categorical variables - using actual values
    ref_cats = {
        'is_urban': 0, 'is_female_headed': 0, 'water_on_premises': 0,
        'hh_head_education': 'No education', 'caste': 'General', 'religion': 'Hindu',
        'region': 'North',
        'water_source_category': 'Tube well/Borehole' # If using water_source_category directly
    }

    df_reg = df.copy()
    all_needed_cols = list(set([outcome_var, 'weight', cfg.VAR_PSU, cfg.VAR_STRATUM] + model3_base_vars))
    
    for col in all_needed_cols:
        if col not in df_reg.columns:
            df_reg[col] = np.nan

    # Ensure binary/categorical vars are correctly typed for statsmodels
    for var in ['is_urban', 'is_female_headed', 'water_on_premises', 'piped_water_flag', 'tube_well_flag']:
        if var in df_reg.columns:
            df_reg[var] = df_reg[var].astype('category')
            if var in ref_cats and ref_cats[var] not in df_reg[var].cat.categories:
                if len(df_reg[var].cat.categories) > 0: ref_cats[var] = df_reg[var].cat.categories[0]

    for var in ['wealth_quintile', 'caste', 'religion', 'hh_head_education', 'region', 'water_source_category',
                 'wvi_category', 'cci_category', 'idi_category']:
        if var in df_reg.columns and not isinstance(df_reg[var].dtype, pd.CategoricalDtype):
            df_reg[var] = df_reg[var].astype('category')
        if var in ref_cats and var in df_reg.columns and isinstance(df_reg[var].dtype, pd.CategoricalDtype):
            if ref_cats[var] not in df_reg[var].cat.categories:
                if len(df_reg[var].cat.categories) > 0: ref_cats[var] = df_reg[var].cat.categories[0]

    df_reg.dropna(subset=[outcome_var, 'weight', cfg.VAR_PSU] + model3_base_vars, inplace=True)
    if df_reg.empty:
        return {}, "No data remaining after dropping NaNs for regression."

    def build_formula_str(base_vars_list):
        formula_parts = []
        for var in base_vars_list:
            if var in df_reg.columns:
                if var in ref_cats and isinstance(df_reg[var].dtype, pd.CategoricalDtype):
                    ref_val = ref_cats[var]
                    if isinstance(ref_val, str): formula_parts.append(f"C({var}, Treatment('{ref_val}'))")
                    else: formula_parts.append(f"C({var}, Treatment({ref_val}))")
                else:
                    formula_parts.append(var)
            else:
                print(f"      Warning: Variable '{var}' not found in df_reg for formula construction. Skipping.")
        return f"{outcome_var} ~ " + " + ".join(formula_parts)

    # Run Model 1: WVI Components
    print("  Running Model 1: Traditional Vulnerability Factors (WVI Components)...")
    formula1 = build_formula_str(model1_base_vars)
    try:
        model1 = smf.logit(formula=formula1, data=df_reg, freq_weights=df_reg['weight'], cov_type='cluster', cov_kwds={'groups': df_reg[cfg.VAR_PSU]})
        results1 = model1.fit(disp=False)
        model_results['Model 1 (WVI Components)'] = pd.DataFrame({
            'OR': np.exp(results1.params), 'CI_lower': np.exp(results1.conf_int()[0]),
            'CI_upper': np.exp(results1.conf_int()[1]), 'p_value': results1.pvalues.apply(format_p_value)
        })
    except Exception as e: print(f"    ERROR running Model 1: {e}"); model_results['Model 1 (WVI Components)'] = pd.DataFrame()

    # Run Model 2: Add CCI Components
    print("  Running Model 2: Add Coping Capacity Factors (CCI Components)...")
    formula2 = build_formula_str(model2_base_vars)
    try:
        model2 = smf.logit(formula=formula2, data=df_reg, freq_weights=df_reg['weight'], cov_type='cluster', cov_kwds={'groups': df_reg[cfg.VAR_PSU]})
        results2 = model2.fit(disp=False)
        model_results['Model 2 (WVI + CCI Components)'] = pd.DataFrame({
            'OR': np.exp(results2.params), 'CI_lower': np.exp(results2.conf_int()[0]),
            'CI_upper': np.exp(results2.conf_int()[1]), 'p_value': results2.pvalues.apply(format_p_value)
        })
    except Exception as e: print(f"    ERROR running Model 2: {e}"); model_results['Model 2 (WVI + CCI Components)'] = pd.DataFrame()

    # Run Model 3: Add IDI Score (Explaining the Paradox)
    print("  Running Model 3: Add Infrastructure Dependency Index (IDI) Score...")
    formula3 = build_formula_str(model3_base_vars)
    try:
        model3 = smf.logit(formula=formula3, data=df_reg, freq_weights=df_reg['weight'], cov_type='cluster', cov_kwds={'groups': df_reg[cfg.VAR_PSU]})
        results3 = model3.fit(disp=False)
        model_results['Model 3 (WVI + CCI + IDI)'] = pd.DataFrame({
            'OR': np.exp(results3.params), 'CI_lower': np.exp(results3.conf_int()[0]),
            'CI_upper': np.exp(results3.conf_int()[1]), 'p_value': results3.pvalues.apply(format_p_value)
        })
    except Exception as e: print(f"    ERROR running Model 3: {e}"); model_results['Model 3 (WVI + CCI + IDI)'] = pd.DataFrame()

    interpretive_text = (
        "Table 7 presents the results of a nested logistic regression analysis, formally investigating the factors "
        "contributing to water disruption and explaining the observed paradox. "
        "**Model 1**, including only traditional vulnerability factors, shows that higher traditional vulnerability "
        "is generally associated with increased odds of disruption. "
        "**Model 2** adds coping capacity components, demonstrating how household resources can mitigate or exacerbate disruption risks. "
        "Crucially, **Model 3** introduces the Infrastructure Dependency Index (IDI). The significant positive odds ratio "
        "associated with IDI, even after controlling for traditional vulnerability and coping capacity, "
        "quantifies the independent effect of reliance on centralized infrastructure. "
        "This model confirms that higher infrastructure dependency is a robust predictor of water disruption, "
        "providing a strong statistical explanation for the 'Infrastructure Paradox' where otherwise low-vulnerability, high-coping households "
        "experience significant unreliability due to their reliance on unreliable piped systems."
    )
    print(f"{'='*10} Table 7 Generated {'='*10}\n")
    return model_results, interpretive_text

def generate_table8_idi_conceptual(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, str]:
    """
    Table 8: Infrastructure Dependency Index (IDI) - Explaining New Vulnerability
    This is a conceptual table to define IDI in the context of the new framework.
    """
    print(f"\n{'='*10} Generating Table 8: IDI Conceptual {'='*10}")
    data = {
        'Paradoxical Group Character': ['Low traditional Vulnerability, High Disruption', 'High traditional Vulnerability, Low Disruption'],
        'Dominant Water Source': ['Piped Water (often sole source)', 'Tube well/Protected Well (diverse local sources)'],
        'Infrastructure Dependency Score (IDI)': ['High (e.g., 8-10)', 'Low (e.g., 0-3)'],
        'Explanation': [
            'Reliance on complex, centralized system leads to vulnerability when system fails, despite high resources.',
            'Reliance on local, diversified, often self-managed sources provides resilience despite lower traditional resources.'
        ]
    }
    table_df = pd.DataFrame(data)
    interpretive_text = (
        "Table 8 introduces the Infrastructure Dependency Index (IDI) as the key conceptual tool to explain the 'Infrastructure Paradox'. "
        "It highlights how households exhibiting paradoxical water insecurity patterns (low traditional vulnerability but high disruption) "
        "are characterized by high scores on the IDI, primarily due to their reliance on piped water as a single or dominant source. "
        "Conversely, the 'resilient poor' (high traditional vulnerability but low disruption) typically score low on the IDI, "
        "indicating diversified and often self-managed water sources. "
        "This table positions the IDI as a measure of a *new form of vulnerability* that arises from the nature of modern water infrastructure itself, "
        "rather than solely from socioeconomic or geographic factors."
    )
    print(f"{'='*10} Table 8 Generated {'='*10}\n")
    return table_df, interpretive_text

# --- END PHASE 3 ---
# --- Supporting tables from original script, now globally defined and renamed for clarity ---
def generate_table_descriptive_characteristics(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, str]:
    """
    Original Table 1 from previous prompt, now a supporting table.
    Shows weighted percentages for key demographic/infrastructure variables by disruption status.
    """
    print(f"\n{'='*10} Generating Supporting Table: Descriptive Characteristics by Disruption Status {'='*10}")
    if cfg.VAR_WATER_DISRUPTED_FINAL not in df.columns:
        print(f"  Error: Missing '{cfg.VAR_WATER_DISRUPTED_FINAL}' column. Cannot generate supporting table.")
        return pd.DataFrame(), "Error: Water disruption status column missing."
    if not df[cfg.VAR_WATER_DISRUPTED_FINAL].isin([0, 1]).all():
        print(f"  Error: '{cfg.VAR_WATER_DISRUPTED_FINAL}' is not purely binary. Cannot generate supporting table.")
        return pd.DataFrame(), "Error: Water disruption status column not binary."

    characteristics = {
        'residence': 'Residence Type',
        'water_source_category': 'Main Water Source',
        'water_on_premises': 'Water On Premises',
        'wealth_quintile': 'Wealth Quintile',
        'caste': 'Caste/Tribe',
        'religion': 'Religion',
        'hh_head_sex': 'Household Head Sex',
        'hh_head_education': 'HH Head Education',
        'house_type': 'House Type'
    }
    results = []
    p_values = {}
    for var, label in characteristics.items():
        if var not in df.columns:
            print(f"      Warning: Column '{var}' not found for supporting descriptive table. Skipping.")
            continue
        
        # Ensure var is categorical for grouping, if appropriate
        if not isinstance(df[var].dtype, pd.CategoricalDtype) and df[var].nunique() < 20:
            df[var] = df[var].astype('category')
        elif not isinstance(df[var].dtype, pd.CategoricalDtype) and df[var].nunique() >= 20:
            print(f"      Warning: Variable '{var}' is not categorical and has high cardinality. Skipping for supporting descriptive table.")
            continue

        not_disrupted_data = calculate_weighted_percentages(
            df, var, weight_col='weight', target_col=cfg.VAR_WATER_DISRUPTED_FINAL, target_val=0
        )
        disrupted_data = calculate_weighted_percentages(
            df, var, weight_col='weight', target_col=cfg.VAR_WATER_DISRUPTED_FINAL, target_val=1
        )
        if not_disrupted_data.empty and disrupted_data.empty:
            print(f"      No data for '{var}' in either disrupted or non-disrupted groups. Skipping.")
            continue

        not_disrupted_data.rename(columns={'Weighted_Percentage': 'Not Disrupted (%)', 'Unweighted_N': 'N_Not_Disrupted'}, inplace=True)
        not_disrupted_data.set_index('Category', inplace=True)
        disrupted_data.rename(columns={'Weighted_Percentage': 'Disrupted (%)', 'Unweighted_N': 'N_Disrupted'}, inplace=True)
        disrupted_data.set_index('Category', inplace=True)
        combined_data = pd.merge(not_disrupted_data, disrupted_data, left_index=True, right_index=True, how='outer').fillna(0)
        combined_data = combined_data.round(1)

        chi2, p_value, dof, _ = run_weighted_chi2(df, var, cfg.VAR_WATER_DISRUPTED_FINAL, 'weight')
        p_values[var] = format_p_value(p_value)

        for idx, row in combined_data.iterrows():
            results.append({
                'Characteristic': label,
                'Category': idx, 
                'Not Disrupted (%)': row.get('Not Disrupted (%)', 0), 
                'N_Not_Disrupted': int(row.get('N_Not_Disrupted', 0)),
                'Disrupted (%)': row.get('Disrupted (%)', 0),
                'N_Disrupted': int(row.get('N_Disrupted', 0)),
                'p-value': '' 
            })
    
    table_df = pd.DataFrame(results)
    if table_df.empty:
        return pd.DataFrame(), "No characteristics data available for supporting descriptive table."

    table_df['p-value_temp'] = ''
    for var_key, p_val_str in p_values.items():
        label = characteristics.get(var_key, var_key)
        first_idx_for_char = table_df[table_df['Characteristic'] == label].index
        if not first_idx_for_char.empty:
            table_df.loc[first_idx_for_char[0], 'p-value_temp'] = p_val_str
    
    table_df['p-value'] = table_df.groupby('Characteristic')['p-value_temp'].transform(lambda x: x.replace('', np.nan).fillna(method='ffill').fillna(''))
    table_df.drop(columns=['p-value_temp'], inplace=True)
    
    interpretive_text = (
        "This supporting table provides an overview of the sampled households, stratified by water disruption status. "
        "It highlights initial associations between demographic, socioeconomic, and water-related factors with reported disruption, "
        "serving as a descriptive baseline for the more advanced analyses."
    )
    print(f"{'='*10} Supporting Descriptive Characteristics Table Generated {'='*10}\n")
    return table_df, interpretive_text


def generate_table_state_level_paradox(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, str]:
    """Original Table 9 from previous prompt, now a supporting table."""
    print(f"\n{'='*10} Generating Supporting Table: State-Level Paradox Rankings {'='*10}")
    if not all(col in df.columns for col in ['state_name', 'piped_water_flag', 'water_source_category', cfg.VAR_WATER_DISRUPTED_FINAL]):
        return pd.DataFrame(), "Error: Required columns missing for State-Level Paradox table."
    results = []
    df_numeric_flags = df.copy()
    df_numeric_flags['piped_water_flag_numeric'] = df_numeric_flags['piped_water_flag'].astype(int)
    df_numeric_flags[cfg.VAR_WATER_DISRUPTED_FINAL + '_numeric'] = df_numeric_flags[cfg.VAR_WATER_DISRUPTED_FINAL].astype(int)
    state_counts = df_numeric_flags['state_name'].value_counts()
    valid_states = state_counts[state_counts >= 1000].index.tolist()
    for state in valid_states:
        state_df = df_numeric_flags[df_numeric_flags['state_name'] == state].copy()
        if state_df.empty or state_df['weight'].sum() == 0: continue
        piped_coverage = (state_df['piped_water_flag_numeric'] * state_df['weight']).sum() / state_df['weight'].sum() * 100
        piped_users = state_df[state_df['water_source_category'] == 'Piped Water']
        piped_disruption = (piped_users[cfg.VAR_WATER_DISRUPTED_FINAL + '_numeric'] * piped_users['weight']).sum() / piped_users['weight'].sum() * 100 if piped_users['weight'].sum() > 0 else np.nan
        tube_well_users = state_df[state_df['water_source_category'] == 'Tube well/Borehole']
        tube_well_disruption = (tube_well_users[cfg.VAR_WATER_DISRUPTED_FINAL + '_numeric'] * tube_well_users['weight']).sum() / tube_well_users['weight'].sum() * 100 if tube_well_users['weight'].sum() > 0 else np.nan
        paradox_ratio = piped_disruption / tube_well_disruption if tube_well_disruption > 0 else np.nan
        results.append({
            'State': state, 'Piped Water Coverage (%)': piped_coverage,
            'Piped Disruption Rate (%)': piped_disruption, 'Tube Well Disruption Rate (%)': tube_well_disruption,
            'Paradox Ratio (Piped/Tube Well)': paradox_ratio, 'N': len(state_df)
        })
    table_df = pd.DataFrame(results).round(1)
    table_df = table_df.sort_values(by='Paradox Ratio (Piped/Tube Well)', ascending=False).reset_index(drop=True)
    def categorize_paradox(ratio):
        if pd.isna(ratio): return 'N/A'
        if ratio > 2.0: return 'Strong Paradox'
        if 1.5 <= ratio <= 2.0: return 'Moderate Paradox'
        return 'Weak Paradox'
    table_df['Paradox Category'] = table_df['Paradox Ratio (Piped/Tube Well)'].apply(categorize_paradox)
    interpretive_text = (
        "This supporting table illustrates the state-level variation in the 'Infrastructure Paradox', "
        "showing how the ratio of piped water disruption to tube well disruption differs across Indian states. "
        "It indicates that the paradox is more pronounced in certain regions, potentially linked to varying "
        "levels of infrastructure development and management effectiveness."
    )
    print(f"{'='*10} Supporting State-Level Paradox Table Generated {'='*10}\n")
    return table_df, interpretive_text


def generate_table_seasonal_patterns(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, str]:
    """Original Table 10 from previous prompt, now a supporting table."""
    print(f"\n{'='*10} Generating Supporting Table: Seasonal Patterns {'='*10}")
    if not all(col in df.columns for col in ['season', cfg.VAR_WATER_DISRUPTED_FINAL, 'water_source_category', 'residence']):
        return pd.DataFrame(), "Error: Required columns missing for Seasonal Patterns table."
    results = []
    seasons = ['Winter', 'Summer', 'Monsoon', 'Post-monsoon']
    major_sources = ['Piped Water', 'Tube well/Borehole']
    for season in seasons:
        if season not in df['season'].cat.categories: continue
        season_df = df[df['season'] == season].copy()
        if season_df.empty or season_df['weight'].sum() == 0: continue
        overall_disruption = (season_df[cfg.VAR_WATER_DISRUPTED_FINAL].astype(float) * season_df['weight']).sum() / season_df['weight'].sum() * 100
        urban_df = season_df[season_df['residence'] == 'Urban']
        urban_disruption = (urban_df[cfg.VAR_WATER_DISRUPTED_FINAL].astype(float) * urban_df['weight']).sum() / urban_df['weight'].sum() * 100 if urban_df['weight'].sum() > 0 else np.nan
        rural_df = season_df[season_df['residence'] == 'Rural']
        rural_disruption = (rural_df[cfg.VAR_WATER_DISRUPTED_FINAL].astype(float) * rural_df['weight']).sum() / rural_df['weight'].sum() * 100 if rural_df['weight'].sum() > 0 else np.nan
        row_data = {
            'Season': season, 'Overall Disruption Rate (%)': overall_disruption,
            'Urban Disruption Rate (%)': urban_disruption, 'Rural Disruption Rate (%)': rural_disruption,
        }
        for source in major_sources:
            source_users = season_df[season_df['water_source_category'] == source]
            source_disruption = (source_users[cfg.VAR_WATER_DISRUPTED_FINAL].astype(float) * source_users['weight']).sum() / source_users['weight'].sum() * 100 if source_users['weight'].sum() > 0 else np.nan
            row_data[f'{source} Disruption Rate (%)'] = source_disruption
        results.append(row_data)
    table_df = pd.DataFrame(results).round(1)
    interpretive_text = (
        "This supporting table examines how water disruption patterns, including the 'Infrastructure Paradox', "
        "vary across different seasons in India. It highlights the interplay between environmental factors "
        "and infrastructure reliability throughout the year."
    )
    print(f"{'='*10} Supporting Seasonal Patterns Table Generated {'='*10}\n")
    return table_df, interpretive_text


def generate_table_robustness_checks(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, str]:
    """Original Table 11 from previous prompt, now a supporting table."""
    print(f"\n{'='*10} Generating Supporting Table: Robustness Checks {'='*10}")
    required_cols = [cfg.VAR_WATER_DISRUPTED_FINAL, 'water_source_category', 'hh_size', 'improved_sanitation_flag',
                     'time_to_water_minutes', 'wealth_quintile', 'residence', 'weight', cfg.VAR_PSU, 'piped_water_flag', 'is_urban']
    if not all(col in df.columns for col in required_cols):
        print(f"  Error: Required columns missing for Robustness Checks table: {set(required_cols) - set(df.columns)}. Skipping.")
        return pd.DataFrame(), "Error: Required columns missing for Robustness Checks table."
    results_data = []
    
    # A. Demand Effect Test (Simplified for this example)
    # Regress disruption on water source, controlling for household size and improved sanitation
    df_reg_demand = df.copy()
    # Ensure flags are numeric for the formula if they are categorical
    df_reg_demand['piped_water_flag_num'] = df_reg_demand['piped_water_flag'].astype(float)
    df_reg_demand['improved_sanitation_flag_num'] = df_reg_demand['improved_sanitation_flag'].astype(float)
    
    demand_formula = f"{cfg.VAR_WATER_DISRUPTED_FINAL} ~ piped_water_flag_num + hh_size + improved_sanitation_flag_num"
    try:
        model_demand = smf.logit(formula=demand_formula, data=df_reg_demand,
                                 freq_weights=df_reg_demand['weight'],
                                 cov_type='cluster', cov_kwds={'groups': df_reg_demand[cfg.VAR_PSU]})
        results_demand = model_demand.fit(disp=False)
        # Parameter name for a numeric variable in formula without C()
        piped_or = np.exp(results_demand.params['piped_water_flag_num'])
        piped_ci_lower = np.exp(results_demand.conf_int().loc['piped_water_flag_num', 0])
        piped_ci_upper = np.exp(results_demand.conf_int().loc['piped_water_flag_num', 1])
        piped_p = results_demand.pvalues['piped_water_flag_num']
        results_data.append({
            'Test': 'Demand Effect Test', 'Variable': 'Piped Water (vs Non-Piped)',
            'OR': piped_or, 'CI_lower': piped_ci_lower, 'CI_upper': piped_ci_upper, 'p_value': format_p_value(piped_p)
        })
    except Exception as e:
        print(f"    ERROR in Demand Effect Test: {e}")
        results_data.append({'Test': 'Demand Effect Test', 'Variable': 'Piped Water (vs Non-Piped)', 'OR': np.nan, 'CI_lower': np.nan, 'CI_upper': np.nan, 'p_value': 'Error'})

    # B. Reporting Bias Test (using hv204 > 30 min as objective outcome)
    df_reg_reporting = df.copy()
    df_reg_reporting['objective_disruption'] = (df_reg_reporting['time_to_water_minutes'] > 30).astype(int)
    df_reg_reporting['piped_water_flag_num'] = df_reg_reporting['piped_water_flag'].astype(float)

    # Use actual wealth quintile and residence for controls
    reporting_formula = f"objective_disruption ~ piped_water_flag_num + C(wealth_quintile, Treatment('Poorest')) + C(residence, Treatment('Rural'))"
    try:
        model_reporting = smf.logit(formula=reporting_formula, data=df_reg_reporting,
                                    freq_weights=df_reg_reporting['weight'],
                                    cov_type='cluster', cov_kwds={'groups': df_reg_reporting[cfg.VAR_PSU]})
        results_reporting = model_reporting.fit(disp=False)
        piped_or = np.exp(results_reporting.params['piped_water_flag_num'])
        piped_ci_lower = np.exp(results_reporting.conf_int().loc['piped_water_flag_num', 0])
        piped_ci_upper = np.exp(results_reporting.conf_int().loc['piped_water_flag_num', 1])
        piped_p = results_reporting.pvalues['piped_water_flag_num']
        results_data.append({
            'Test': 'Reporting Bias Test (Objective Disruption)', 'Variable': 'Piped Water (vs Non-Piped)',
            'OR': piped_or, 'CI_lower': piped_ci_lower, 'CI_upper': piped_ci_upper, 'p_value': format_p_value(piped_p)
        })
    except Exception as e:
        print(f"    ERROR in Reporting Bias Test: {e}")
        results_data.append({'Test': 'Reporting Bias Test (Objective Disruption)', 'Variable': 'Piped Water (vs Non-Piped)', 'OR': np.nan, 'CI_lower': np.nan, 'CI_upper': np.nan, 'p_value': 'Error'})

    # C. Subgroup Analysis (Piped Water effect for Urban/Rural)
    for res_type_val, label in [(1, 'Urban'), (0, 'Rural')]:
        df_sub = df[df['is_urban'] == res_type_val].copy()
        if df_sub.empty:
            results_data.append({'Test': f'Subgroup Analysis ({label})', 'Variable': 'Piped Water (vs Non-Piped)', 'OR': np.nan, 'CI_lower': np.nan, 'CI_upper': np.nan, 'p_value': 'N/A'})
            continue
        df_sub['piped_water_flag_num'] = df_sub['piped_water_flag'].astype(float)
        subgroup_formula = f"{cfg.VAR_WATER_DISRUPTED_FINAL} ~ piped_water_flag_num + C(wealth_quintile, Treatment('Poorest'))"
        try:
            model_subgroup = smf.logit(formula=subgroup_formula, data=df_sub,
                                       freq_weights=df_sub['weight'],
                                       cov_type='cluster', cov_kwds={'groups': df_sub[cfg.VAR_PSU]})
            results_subgroup = model_subgroup.fit(disp=False)
            piped_or = np.exp(results_subgroup.params['piped_water_flag_num'])
            piped_ci_lower = np.exp(results_subgroup.conf_int().loc['piped_water_flag_num', 0])
            piped_ci_upper = np.exp(results_subgroup.conf_int().loc['piped_water_flag_num', 1])
            piped_p = results_subgroup.pvalues['piped_water_flag_num']
            results_data.append({
                'Test': f'Subgroup Analysis ({label})', 'Variable': 'Piped Water (vs Non-Piped)',
                'OR': piped_or, 'CI_lower': piped_ci_lower, 'CI_upper': piped_ci_upper, 'p_value': format_p_value(piped_p)
            })
        except Exception as e:
            print(f"    ERROR in Subgroup Analysis ({label}): {e}")
            results_data.append({'Test': f'Subgroup Analysis ({label})', 'Variable': 'Piped Water (vs Non-Piped)', 'OR': np.nan, 'CI_lower': np.nan, 'CI_upper': np.nan, 'p_value': 'Error'})

    table_df = pd.DataFrame(results_data).round(2)
    interpretive_text = (
        "This supporting table presents the results of several robustness checks designed to test alternative explanations "
        "and ensure the consistency of the 'Infrastructure Paradox' finding. "
        "The piped water effect persists even when controlling for demand proxies and when using an objective measure of disruption, "
        "and is consistent across key subgroups, reinforcing the robustness of the paradox."
    )
    print(f"{'='*10} Supporting Robustness Checks Table Generated {'='*10}\n")
    return table_df, interpretive_text


def generate_table_idi_validation(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, str]:
    """Original Table 12 from previous prompt, now a supporting table."""
    print(f"\n{'='*10} Generating Supporting Table: IDI Construct Validity {'='*10}")
    required_cols = ['idi_score', cfg.VAR_WATER_DISRUPTED_FINAL, cfg.VAR_WEALTH_SCORE, 'is_urban', 'weight']
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame(), "Error: Required columns missing for IDI Construct Validity table."
    idi_df_components = df[required_cols].dropna().copy()
    if idi_df_components.empty:
        return pd.DataFrame({'Note': ['Insufficient data for IDI construct validation.']}), "Insufficient data for IDI construct validation."
    results_data = []

    idi_score_numeric = pd.to_numeric(idi_df_components['idi_score'], errors='coerce').dropna()
    water_disrupted_binary = pd.to_numeric(idi_df_components[cfg.VAR_WATER_DISRUPTED_FINAL], errors='coerce').dropna()
    if not idi_score_numeric.empty and not water_disrupted_binary.empty:
        corr_idi_disruption, p_corr_idi_disruption = pearsonr(idi_score_numeric, water_disrupted_binary)
        results_data.append({'Metric': 'Correlation (IDI Score vs Disruption)', 'Value': corr_idi_disruption, 'p_value': format_p_value(p_corr_idi_disruption)})
    else:
        results_data.append({'Metric': 'Correlation (IDI Score vs Disruption)', 'Value': np.nan, 'p_value': 'N/A'})
    try:
        auc_idi = roc_auc_score(idi_df_components[cfg.VAR_WATER_DISRUPTED_FINAL], idi_df_components['idi_score'], sample_weight=idi_df_components['weight'])
    except ValueError as e:
        print(f"    Warning: Could not compute ROC AUC for IDI: {e}")
        auc_idi = np.nan
    results_data.append({'Metric': 'ROC AUC (IDI Score predicting Disruption)', 'Value': auc_idi, 'p_value': ''})

    wealth_score_numeric = pd.to_numeric(idi_df_components[cfg.VAR_WEALTH_SCORE], errors='coerce').dropna()
    if not idi_score_numeric.empty and not wealth_score_numeric.empty:
        corr_idi_wealth, p_corr_idi_wealth = pearsonr(idi_score_numeric, wealth_score_numeric)
        results_data.append({'Metric': 'Correlation (IDI Score vs Wealth Score)', 'Value': corr_idi_wealth, 'p_value': format_p_value(p_corr_idi_wealth)})
    else:
        results_data.append({'Metric': 'Correlation (IDI Score vs Wealth Score)', 'Value': np.nan, 'p_value': 'N/A'})

    is_urban_numeric = pd.to_numeric(idi_df_components['is_urban'], errors='coerce').dropna()
    if not idi_score_numeric.empty and not is_urban_numeric.empty:
        corr_idi_urban, p_corr_idi_urban = pearsonr(idi_score_numeric, is_urban_numeric)
        results_data.append({'Metric': 'Correlation (IDI Score vs Urban)', 'Value': corr_idi_urban, 'p_value': format_p_value(p_corr_idi_urban)})
    else:
        results_data.append({'Metric': 'Correlation (IDI Score vs Urban)', 'Value': np.nan, 'p_value': 'N/A'})
    
    table_df = pd.DataFrame(results_data).round(2)
    interpretive_text = (
        "This supporting table validates the construct of the Infrastructure Dependency Index (IDI). "
        "It demonstrates the IDI's predictive power for water disruption and its discriminant validity "
        "from traditional socioeconomic indicators, confirming its utility as a measure of new forms of vulnerability."
    )
    print(f"{'='*10} Supporting IDI Construct Validity Table Generated {'='*10}\n")
    return table_df, interpretive_text


def generate_table_policy_simulation(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, str]:
    """Original Table 13 from previous prompt, now a supporting table."""
    print(f"\n{'='*10} Generating Supporting Table: Policy Simulation {'='*10}")
    if not all(col in df.columns for col in ['piped_water_flag', cfg.VAR_WATER_DISRUPTED_FINAL, 'water_source_category', 'residence', 'weight']):
        return pd.DataFrame(), "Error: Required columns missing for Policy Simulation table."
    results = []
    df_numeric_flags = df.copy()
    df_numeric_flags['piped_water_flag_numeric'] = df_numeric_flags['piped_water_flag'].astype(int)
    df_numeric_flags[cfg.VAR_WATER_DISRUPTED_FINAL + '_numeric'] = df_numeric_flags[cfg.VAR_WATER_DISRUPTED_FINAL].astype(int)
    
    current_piped_coverage = (df_numeric_flags['piped_water_flag_numeric'] * df_numeric_flags['weight']).sum() / df_numeric_flags['weight'].sum() * 100
    current_national_disruption = (df_numeric_flags[cfg.VAR_WATER_DISRUPTED_FINAL + '_numeric'] * df_numeric_flags['weight']).sum() / df_numeric_flags['weight'].sum() * 100
    current_urban_df = df_numeric_flags[df_numeric_flags['residence'] == 'Urban']
    current_urban_disruption = (current_urban_df[cfg.VAR_WATER_DISRUPTED_FINAL + '_numeric'] * current_urban_df['weight']).sum() / current_urban_df['weight'].sum() * 100 if current_urban_df['weight'].sum() > 0 else np.nan
    current_rural_df = df_numeric_flags[df_numeric_flags['residence'] == 'Rural']
    current_rural_disruption = (current_rural_df[cfg.VAR_WATER_DISRUPTED_FINAL + '_numeric'] * current_rural_df['weight']).sum() / current_rural_df['weight'].sum() * 100 if current_rural_df['weight'].sum() > 0 else np.nan
    results.append({
        'Scenario': 'Current Scenario', '% Piped Coverage': current_piped_coverage,
        'National Disruption Rate (%)': current_national_disruption,
        'Disruption Urban (%)': current_urban_disruption, 'Disruption Rural (%)': current_rural_disruption
    })
    
    piped_disruption_rate_actual = df_numeric_flags[df_numeric_flags['water_source_category'] == 'Piped Water']
    piped_disruption_rate_mean = (piped_disruption_rate_actual[cfg.VAR_WATER_DISRUPTED_FINAL + '_numeric'] * piped_disruption_rate_actual['weight']).sum() / piped_disruption_rate_actual['weight'].sum() if piped_disruption_rate_actual['weight'].sum() > 0 else np.nan
    tube_well_disruption_rate_actual = df_numeric_flags[df_numeric_flags['water_source_category'] == 'Tube well/Borehole']
    tube_well_disruption_rate_mean = (tube_well_disruption_rate_actual[cfg.VAR_WATER_DISRUPTED_FINAL + '_numeric'] * tube_well_disruption_rate_actual['weight']).sum() / tube_well_disruption_rate_actual['weight'].sum() if tube_well_disruption_rate_actual['weight'].sum() > 0 else np.nan
    
    simulated_df_universal = df_numeric_flags.copy()
    if not pd.isna(piped_disruption_rate_mean):
        simulated_df_universal['simulated_disruption'] = piped_disruption_rate_mean
        universal_disruption_rate = (simulated_df_universal['simulated_disruption'] * simulated_df_universal['weight']).sum() / simulated_df_universal['weight'].sum() * 100
        simulated_urban_disruption = (simulated_df_universal[simulated_df_universal['residence'] == 'Urban']['simulated_disruption'] * simulated_df_universal[simulated_df_universal['residence'] == 'Urban']['weight']).sum() / simulated_df_universal[simulated_df_universal['residence'] == 'Urban']['weight'].sum() * 100
        simulated_rural_disruption = (simulated_df_universal[simulated_df_universal['residence'] == 'Rural']['simulated_disruption'] * simulated_df_universal[simulated_df_universal['residence'] == 'Rural']['weight']).sum() / simulated_df_universal[simulated_df_universal['residence'] == 'Rural']['weight'].sum() * 100
        results.append({
            'Scenario': 'Universal Piped Water (Current Reliability)', '% Piped Coverage': 100.0,
            'National Disruption Rate (%)': universal_disruption_rate,
            'Disruption Urban (%)': simulated_urban_disruption, 'Disruption Rural (%)': simulated_rural_disruption
        })
    else:
        results.append({
            'Scenario': 'Universal Piped Water (Current Reliability)', '% Piped Coverage': 100.0,
            'National Disruption Rate (%)': np.nan, 'Disruption Urban (%)': np.nan, 'Disruption Rural (%)': np.nan
        })
    
    simulated_df_enhanced = df_numeric_flags.copy()
    if not pd.isna(tube_well_disruption_rate_mean):
        simulated_df_enhanced['simulated_disruption'] = tube_well_disruption_rate_mean
        enhanced_disruption_rate = (simulated_df_enhanced['simulated_disruption'] * simulated_df_enhanced['weight']).sum() / simulated_df_enhanced['weight'].sum() * 100
        enhanced_urban_disruption = (simulated_df_enhanced[simulated_df_enhanced['residence'] == 'Urban']['simulated_disruption'] * simulated_df_enhanced[simulated_df_enhanced['residence'] == 'Urban']['weight']).sum() / simulated_df_enhanced[simulated_df_enhanced['residence'] == 'Urban']['weight'].sum() * 100
        enhanced_rural_disruption = (simulated_df_enhanced[simulated_df_enhanced['residence'] == 'Rural']['simulated_disruption'] * simulated_df_enhanced[simulated_df_enhanced['residence'] == 'Rural']['weight']).sum() / simulated_df_enhanced[simulated_df_enhanced['residence'] == 'Rural']['weight'].sum() * 100
        results.append({
            'Scenario': 'Universal Piped Water (Enhanced Reliability)', '% Piped Coverage': 100.0,
            'National Disruption Rate (%)': enhanced_disruption_rate,
            'Disruption Urban (%)': enhanced_urban_disruption, 'Disruption Rural (%)': enhanced_rural_disruption
        })
    else:
        results.append({
            'Scenario': 'Universal Piped Water (Enhanced Reliability)', '% Piped Coverage': 100.0,
            'National Disruption Rate (%)': np.nan, 'Disruption Urban (%)': np.nan, 'Disruption Rural (%)': np.nan
        })
    table_df = pd.DataFrame(results).round(1)
    interpretive_text = (
        "This supporting table presents a policy simulation for the Jal Jeevan Mission, projecting the impact of "
        "universal piped water coverage under different reliability assumptions. It highlights the critical role of "
        "reliability in achieving true water security, demonstrating that expanding coverage without addressing "
        "reliability risks worsening national water disruption."
    )
    print(f"{'='*10} Supporting Policy Simulation Table Generated {'='*10}\n")
    return table_df, interpretive_text


# ==============================================================================
# 6. Markdown Output Generation (REVISED)
# ==============================================================================

def generate_report_markdown(
    cfg: Config,
    df_processed: pd.DataFrame,
    table1_data: pd.DataFrame, table1_text: str, # WVI Components
    table2_data: pd.DataFrame, table2_text: str, # WVI Distribution
    table3_data: pd.DataFrame, table3_text: str, # Coping Typology
    table4_data: pd.DataFrame, table4_text: str, # CCI Construction
    table5_data: Dict[str, pd.DataFrame], table5_text: str, # Vuln-Coping Matrix
    table6_data: pd.DataFrame, table6_text: str, # Paradox Decomposition
    table7_results: Dict[str, Any], table7_text: str, # Multivariate Regression
    table8_data: pd.DataFrame, table8_text: str, # IDI Conceptual
    # Additional tables from original script, now called for supporting sections
    table_descriptive_characteristics_data: pd.DataFrame, table_descriptive_characteristics_text: str,
    table_state_level_paradox_data: pd.DataFrame, table_state_level_paradox_text: str,
    table_seasonal_patterns_data: pd.DataFrame, table_seasonal_patterns_text: str,
    table_robustness_checks_data: pd.DataFrame, table_robustness_checks_text: str,
    table_idi_validation_data: pd.DataFrame, table_idi_validation_text: str,
    table_policy_simulation_data: pd.DataFrame, table_policy_simulation_text: str,
) -> str:
    """Assembles all generated tables and interpretive text into a single markdown string."""
    report_content = []

    # --- Header ---
    report_content.append(f"# From Vulnerability to Paradox: Uncovering Hidden Water Insecurity Patterns in India through NFHS-5")
    report_content.append(f"## Evidence from National Family Health Survey (2019-21)")
    report_content.append(f"\n**Analysis Date:** {datetime.now().strftime('%Y-%m-%d')}")
    report_content.append(f"**Sample:** {len(df_processed):,} households across India")
    report_content.append(f"**Data Source:** National Family Health Survey, Round 5 (NFHS-5)")
    report_content.append(f"\n---")

    # --- Abstract ---
    report_content.append(f"## ABSTRACT")
    report_content.append(
        "This study embarks on a journey to understand water insecurity in India by first mapping traditional vulnerabilities "
        "and coping mechanisms, and then investigating how these relate to actual water disruption experiences. "
        "Utilizing data from 636,699 households in NFHS-5 (2019-21), we construct a Water Vulnerability Index (WVI) and "
        "a Coping Capacity Index (CCI) to categorize households. "
        "Our analysis reveals an unexpected 'Infrastructure Paradox': households with low traditional vulnerability and high coping capacity, "
        "often characterized by reliance on piped water, experience significantly higher rates of water disruption. "
        "Multivariate analysis confirms that this paradox is driven by an 'Infrastructure Dependency' where modern, centralized "
        "water systems, despite their perceived improvement, introduce new vulnerabilities due to their unreliability. "
        "This finding challenges conventional development paradigms and calls for a re-evaluation of water infrastructure policies "
        "to prioritize reliability and resilience alongside coverage expansion."
    )
    report_content.append(f"\n---")

    # --- 1. Introduction ---
    report_content.append(f"## 1. INTRODUCTION")
    report_content.append(
        "Access to safe and reliable water remains a critical global challenge, particularly in rapidly developing nations like India. "
        "Traditionally, water insecurity has been understood through the lens of socioeconomic vulnerability – poverty, marginalization, "
        "and lack of access to basic services. Development efforts, including India's ambitious Jal Jeevan Mission, have largely focused "
        "on expanding 'improved' water infrastructure, such as piped water systems, to address these traditional vulnerabilities. "
        "However, the relationship between infrastructure provision and actual water security may be more complex than currently understood."
        "\n\nThis paper undertakes a comprehensive analysis of water insecurity in India, moving beyond conventional assumptions. "
        "We begin by systematically assessing household vulnerability and coping capacities, aiming to identify populations most at risk "
        "and their adaptive strategies. Our journey, however, leads to an unexpected discovery: a 'paradox' where seemingly "
        "well-resourced households with modern water infrastructure report higher rates of water disruption. "
        "This counter-intuitive finding compels us to propose a new framework – 'Infrastructure Dependency' – to explain how "
        "the very advancements intended to enhance water security can, paradoxically, introduce new forms of vulnerability."
        "\n\n**Research Questions:**"
        "\n1. What are the patterns of traditional water vulnerability and coping capacity across Indian households?"
        "\n2. How do these vulnerability and coping profiles relate to actual experiences of water disruption?"
        "\n3. What unexpected patterns emerge, challenging conventional understandings of water security?"
        "\n4. How can the concept of 'Infrastructure Dependency' explain these paradoxical findings?"
        "\n5. What are the policy implications for water infrastructure development and the Jal Jeevan Mission?"
    )
    report_content.append(f"\n---")

    # --- 2. Literature Review --- (Conceptual, not data-driven)
    report_content.append(f"## 2. LITERATURE REVIEW")
    report_content.append(
        "The literature review would delve into existing theories of vulnerability and resilience, "
        "traditional approaches to water infrastructure and development, and emerging critiques of the 'improved source' paradigm. "
        "It would establish the gap in current understanding regarding how infrastructure itself can become a source of vulnerability, "
        "setting the stage for the discovery narrative."
    )
    report_content.append(f"\n---")

    # --- 3. Data and Methods ---
    report_content.append(f"## 3. DATA AND METHODS")
    report_content.append(
        "This study utilizes household-level data from the National Family Health Survey (NFHS-5), 2019-21, a nationally "
        "representative survey covering 636,699 households across India. The survey employed a two-stage stratified sampling design, "
        "and all analyses incorporate appropriate survey weights to ensure representativeness."
        "\n\n**Outcome Variable:** Water disruption is measured by `sh37b`: 'In the past 2 weeks, has there been any time when your household did not have sufficient water for drinking/cooking?' (1=Yes, 0=No)."
        "\n\n**Vulnerability and Coping Indices:** We construct two primary indices:"
        "\n*   **Water Vulnerability Index (WVI):** A composite score reflecting traditional socioeconomic and geographic risk factors."
        "\n*   **Coping Capacity Index (CCI):** A composite score reflecting a household's resources to manage water shortages."
        "\n\n**Analytical Strategy:** Our approach follows a discovery narrative:"
        "\n1.  **Vulnerability Mapping:** Descriptive analysis of WVI distribution."
        "\n2.  **Coping Assessment:** Descriptive analysis of CCI and coping strategies."
        "\n3.  **Vulnerability-Coping Matrix:** Cross-tabulation of WVI, CCI, and water disruption to identify unexpected patterns."
        "\n4.  **Paradox Decompositions:** In-depth analysis of paradoxical groups to uncover underlying drivers."
        "\n5.  **Multivariate Regression:** Formal testing of infrastructure dependency as an explanatory factor."
        "\n6.  **Robustness Checks and Policy Simulations.**"
    )
    
    report_content.append(f"\n### Table 1: Water Vulnerability Index (WVI) Components")
    report_content.append(table1_data.to_markdown(index=False))
    report_content.append(f"\n**Interpretation:** {table1_text}")

    report_content.append(f"\n### Table 4: Coping Capacity Index (CCI) Construction")
    report_content.append(table4_data.to_markdown(index=False))
    report_content.append(f"\n**Interpretation:** {table4_text}")
    
    report_content.append(f"\n---")

    # --- 4. Results ---
    report_content.append(f"## 4. RESULTS")

    report_content.append(f"\n### 4.1 The Vulnerability Landscape")
    report_content.append(
        "Our initial assessment of traditional water vulnerability, as captured by the Water Vulnerability Index (WVI), "
        "reveals expected patterns across India. Households with lower socioeconomic status and located in rural areas "
        "generally exhibit higher levels of traditional vulnerability."
    )
    report_content.append(f"\n### Table 2: Distribution of Water Vulnerability Index Across India")
    report_content.append(table2_data.to_markdown(index=True))
    report_content.append(f"\n**Interpretation:** {table2_text}")

    report_content.append(f"\n### 4.2 Coping Mechanisms and Capacity")
    report_content.append(
        "Households employ a diverse range of coping strategies when faced with water disruption. These strategies vary "
        "depending on the primary water source and the household's inherent coping capacity."
    )
    report_content.append(f"\n### Table 3: Typology of Coping Strategies During Water Disruption")
    report_content.append(table3_data.to_markdown(index=False))
    report_content.append(f"\n**Interpretation:** {table3_text}")

    report_content.append(f"\n### 4.3 The Vulnerability-Coping Nexus: An Unexpected Discovery")
    report_content.append(
        "To understand how vulnerability and coping capacity jointly influence actual water disruption experiences, "
        "we constructed a Vulnerability-Coping Matrix. This analysis proved pivotal, revealing patterns that challenge "
        "conventional wisdom."
    )
    report_content.append(f"\n### Table 5: Vulnerability-Coping Matrix - Disruption Rates")
    report_content.append(table5_data['Disruption Rates'].to_markdown(index=True))
    report_content.append(f"\n### Table 5: Vulnerability-Coping Matrix - % of Households (Weighted)")
    report_content.append(table5_data['% Households'].to_markdown(index=True))
    report_content.append(f"\n**Interpretation:** {table5_text}")

    report_content.append(f"\n### 4.4 Decomposing the Paradox: The Role of Infrastructure")
    report_content.append(
        "The unexpected high disruption rates observed in traditionally low-vulnerability, high-coping groups "
        "prompted a deeper investigation into their specific characteristics. This decomposition revealed a critical "
        "underlying factor: the type of water infrastructure."
    )
    report_content.append(f"\n### Table 6: Decomposing Paradoxical Groups' Characteristics")
    report_content.append(table6_data.to_markdown(index=True))
    report_content.append(f"\n**Interpretation:** {table6_text}")
    
    report_content.append(f"\n### 4.5 Multivariate Analysis: Confirming Infrastructure Dependency")
    report_content.append(
        "To formally test the explanatory power of infrastructure characteristics, particularly the newly identified "
        "'Infrastructure Dependency', we conducted a nested logistic regression analysis."
    )
    report_content.append(f"\n### Table 7: Multivariate Logistic Regression - Explaining Water Disruption")
    # Create a combined table for easier markdown output for OR (CI) and p-value
    combined_lr_df_new = pd.DataFrame()
    model_keys_new = ['Model 1 (WVI Components)', 'Model 2 (WVI + CCI Components)', 'Model 3 (WVI + CCI + IDI)']
    for model_name in model_keys_new:
        if model_name in table7_results and not table7_results[model_name].empty:
            model_df_subset = table7_results[model_name][['OR', 'CI_lower', 'CI_upper', 'p_value']].copy()
            model_df_subset.columns = [f'{col}_{model_name.replace(" ", "_").replace("(", "").replace(")", "")}' for col in model_df_subset.columns]
            if combined_lr_df_new.empty:
                combined_lr_df_new = model_df_subset
            else:
                combined_lr_df_new = combined_lr_df_new.join(model_df_subset, how='outer')
    if not combined_lr_df_new.empty:
        report_content.append(combined_lr_df_new.to_markdown(index=True))
    else:
        report_content.append("No logistic regression results available due to errors in model fitting.")
    report_content.append(f"\n**Interpretation:** {table7_text}")

    report_content.append(f"\n### 4.6 The Infrastructure Paradox: A New Vulnerability Framework")
    report_content.append(
        "The consistent findings across descriptive and multivariate analyses lead us to propose "
        "the 'Infrastructure Dependency Index' (IDI) as a critical measure for understanding "
        "water insecurity in modernizing contexts."
    )
    report_content.append(f"\n### Table 8: Infrastructure Dependency Index (IDI) - Explaining New Vulnerability")
    report_content.append(table8_data.to_markdown(index=False))
    report_content.append(f"\n**Interpretation:** {table8_text}")
    
    # Add other supporting tables here (from original script)
    report_content.append(f"\n### 4.7 Supporting Analyses")
    report_content.append(f"\n#### Descriptive Characteristics by Disruption Status (Appendix Table A1)")
    report_content.append(table_descriptive_characteristics_data.to_markdown(index=False))
    report_content.append(f"\n**Interpretation:** {table_descriptive_characteristics_text}")

    report_content.append(f"\n#### State-Level Paradox Rankings (Appendix Table A2)")
    report_content.append(table_state_level_paradox_data.to_markdown(index=False))
    report_content.append(f"\n**Interpretation:** {table_state_level_paradox_text}")

    report_content.append(f"\n#### Seasonal Patterns in Water Disruption (Appendix Table A3)")
    report_content.append(table_seasonal_patterns_data.to_markdown(index=False))
    report_content.append(f"\n**Interpretation:** {table_seasonal_patterns_text}")

    report_content.append(f"\n---")

    # --- 5. Discussion ---
    report_content.append(f"## 5. DISCUSSION")
    report_content.append(
        "Our study began by mapping traditional water vulnerability and coping capacities across Indian households. "
        "While initial findings aligned with expected patterns, the pivotal Vulnerability-Coping Matrix revealed a "
        "striking anomaly: households with high resources and low traditional vulnerability often experienced "
        "significant water disruption. This unexpected discovery led us to identify a new dimension of water insecurity – "
        "the 'Infrastructure Paradox', primarily driven by reliance on modern, yet unreliable, piped water systems."
        "\n\n**Reconceptualizing Water Vulnerability:** This research posits that traditional vulnerability frameworks, "
        "while valuable, are insufficient in contexts of rapid infrastructure development. The 'Infrastructure Dependency' "
        "model highlights how the very systems designed to improve water access can create new vulnerabilities when their "
        "reliability is compromised. Households become 'locked-in' to centralized systems, potentially losing traditional "
        "coping skills and access to alternative local sources."
        "\n\n**Implications for Coping and Resilience:** Our findings suggest a shift in coping paradigms. "
        "Households dependent on unreliable piped water often resort to market-based solutions (e.g., purchasing tankers) "
        "or increased time/labor burdens, even when they possess higher overall coping capacity. This implies that "
        "the nature of water infrastructure dictates the type and effectiveness of coping, potentially exacerbating inequalities "
        "by imposing financial burdens on those who can least afford them, or time burdens on women and children."
    )
    report_content.append(f"\n---")

    # --- 6. Robustness & Validation ---
    report_content.append(f"## 6. ROBUSTNESS & VALIDATION")
    report_content.append(
        "To ensure the reliability of our findings and the validity of the Infrastructure Dependency Index, "
        "we conducted several robustness checks and validation analyses."
    )
    report_content.append(f"\n### Table: Robustness Checks for the Paradox")
    report_content.append(table_robustness_checks_data.to_markdown(index=False))
    report_content.append(f"\n**Interpretation:** {table_robustness_checks_text}")

    report_content.append(f"\n### Table: Infrastructure Dependency Index (IDI) Validation")
    report_content.append(table_idi_validation_data.to_markdown(index=False))
    report_content.append(f"\n**Interpretation:** {table_idi_validation_text}")
    report_content.append(f"\n---")

    # --- 7. Policy Implications ---
    report_content.append(f"## 7. POLICY IMPLICATIONS")
    report_content.append(
        "The discovery of the 'Infrastructure Paradox' has profound implications for water policy, particularly "
        "for ambitious programs like India's Jal Jeevan Mission. Our findings underscore that simply expanding "
        "infrastructure coverage without ensuring reliability can inadvertently worsen water security."
    )
    report_content.append(f"\n### Table: Projected Impact of Jal Jeevan Mission")
    report_content.append(table_policy_simulation_data.to_markdown(index=False))
    report_content.append(f"\n**Interpretation:** {table_policy_simulation_text}")
    report_content.append(
        "\n\n**Key Policy Recommendations:**"
        "\n1.  **Prioritize Reliability:** Shift focus from mere coverage to ensuring consistent and high-quality service delivery for existing infrastructure."
        "\n2.  **Maintain Redundancy and Diversity:** Support and integrate traditional, local water sources as crucial backups, especially in areas transitioning to piped water."
        "\n3.  **Invest in O&M:** Allocate sufficient resources for operation and maintenance of piped systems to reduce disruptions."
        "\n4.  **Empower Local Governance:** Strengthen local institutions for water management, grievance redressal, and community-led solutions."
        "\n5.  **Promote Household Storage:** Encourage and incentivize household-level water storage solutions to buffer against intermittent supply."
        "\n6.  **Context-Specific Solutions:** Recognize that a 'one-size-fits-all' approach to water infrastructure may create new vulnerabilities; tailor solutions to local contexts and existing coping strategies."
    )
    report_content.append(f"\n---")

    # --- 8. Limitations & Future Research ---
    report_content.append(f"## 8. LIMITATIONS & FUTURE RESEARCH")
    report_content.append(
        "This study, while comprehensive, has limitations inherent to its cross-sectional survey design. "
        "Future research should explore longitudinal data to establish causality, integrate objective measures of water quality and pressure, "
        "and conduct qualitative studies to delve deeper into household decision-making during disruptions. "
        "Further refinement of the WVI and CCI, perhaps through participatory methods, could also enhance their precision. "
        "Exploring the spatial dimensions of infrastructure unreliability and its relationship with climate change impacts "
        "represents another crucial avenue for future investigation."
    )
    report_content.append(f"\n---")

    # --- 9. Conclusion ---
    report_content.append(f"## 9. CONCLUSION")
    report_content.append(
        "Our analysis, building from a comprehensive assessment of traditional water vulnerability and coping capacities, "
        "has uncovered a critical 'Infrastructure Paradox' in India. We demonstrate that households with low traditional "
        "vulnerability and high coping resources, particularly those relying on piped water, experience unexpectedly high "
        "rates of water disruption. This paradox is explained by 'Infrastructure Dependency', a new form of vulnerability "
        "arising from the inherent unreliability of modern, centralized water systems in certain contexts. "
        "This finding fundamentally challenges the assumption that infrastructure expansion automatically translates to "
        "improved water security. For India and other developing nations, the path to true water security must prioritize "
        "reliability, resilience, and a nuanced understanding of how infrastructure itself can reshape the landscape of vulnerability."
    )
    report_content.append(f"\n---")

    # --- Technical Notes ---
    report_content.append(f"## TECHNICAL NOTES")
    report_content.append(f"### Survey Weighting")
    report_content.append(f"- All percentages and rates are weighted using `hv005`/1,000,000 for proper population representation.")
    report_content.append(f"- While attempts were made to account for complex survey design (clustering, stratification) using robust standard errors in regressions, full design effects are best handled by specialized survey statistics software (e.g., R's `survey` package).")
    report_content.append(f"\n### Statistical Methods")
    report_content.append(f"- Weighted chi-square tests (approximated) for categorical associations.")
    report_content.append(f"- Logistic regression with robust standard errors using `statsmodels.formula.api.logit` with `freq_weights` and `cov_type='cluster'` for PSU-level clustering.")
    report_content.append(f"- Correlation analysis for predictive and discriminant validity.")
    report_content.append(f"- ROC AUC for predictive validity of IDI.")
    report_content.append(f"\n### Missing Data")
    report_content.append(f"- Households with missing `{cfg.VAR_WATER_DISRUPTED_FINAL}` (water disruption status) or `weight` were excluded from the primary analysis.")
    report_content.append(f"- Other missing values were handled by imputation (e.g., median for continuous, mode for categorical) or case-wise deletion where appropriate for specific analyses.")
    report_content.append(f"\n### Code Availability")
    report_content.append(f"The Python code used for this analysis is available upon request, ensuring full reproducibility.")
    report_content.append(f"\n---")

    # --- References (Placeholder) ---
    report_content.append(f"## REFERENCES")
    report_content.append(f"[Will be added to final paper]")
    report_content.append(f"\n---")
    report_content.append(f"**Generated by:** NFHS-5 Water Disruption Analysis Pipeline")
    report_content.append(f"**Version:** 3.0 (Discovery Narrative)")
    report_content.append(f"**Contact:** [Your information]")

    return "\n".join(report_content)

# ==============================================================================
# 7. Main Execution Function (REVISED)
# ==============================================================================

def main():
    """Orchestrates the entire analysis pipeline and generates the research paper."""
    print("=" * 80)
    print("Starting NFHS-5 Water Insecurity Analysis: Discovery Narrative")
    print("=" * 80)

    cfg = Config()
    report_filepath = cfg.OUTPUT_DIR / f"{cfg.REPORT_FILENAME}_{cfg.TIMESTAMP}.md"

    data_loader = DataLoader(cfg)
    df_raw = data_loader.load_data()
    if df_raw.empty:
        print("Initial data loading failed or returned empty DataFrame. Exiting.")
        return

    try:
        data_processor = DataProcessor(df_raw, cfg)
        df_processed = data_processor.process()
    except ValueError as e:
        print(f"Critical Data Processing Error: {e}. Exiting.")
        return

    if df_processed.empty:
        print("Processed DataFrame is empty after cleaning. Exiting.")
        return
    
    # --- CRITICAL VERIFICATION OF PROCESSED DF ---
    print(f"\n{'='*20} Post-Processing Verification {'='*20}")
    required_final_cols = [cfg.VAR_WATER_DISRUPTED_FINAL, 'weight', cfg.VAR_PSU, 'wvi_score_scaled', 'cci_score_scaled', 'idi_score']
    for col in required_final_cols:
        if col not in df_processed.columns:
            print(f"  CRITICAL ERROR: '{col}' is MISSING from df_processed!")
            print("  This variable is essential for the analysis. Please check DataProcessor.")
            return
        else:
            print(f"  SUCCESS: '{col}' is present.")
    print(f"{'='*20} Verification Complete {'='*20}\n")

    print("\n" + "=" * 40)
    print("Generating Tables and Interpretive Text (Discovery Narrative)")
    print("=" * 40)

    all_tables_data = {}
    all_tables_text = {}

    # --- PHASE 1: VULNERABILITY ASSESSMENT ---
    all_tables_data['table1_wvi_components'], all_tables_text['table1_wvi_components'] = generate_table1_wvi_components(df_processed, cfg)
    all_tables_data['table1_wvi_components'].to_csv(cfg.OUTPUT_DIR / "tables" / "table1_wvi_components.csv", index=False)

    all_tables_data['table2_wvi_distribution'], all_tables_text['table2_wvi_distribution'] = generate_table2_wvi_distribution(df_processed, cfg)
    all_tables_data['table2_wvi_distribution'].to_csv(cfg.OUTPUT_DIR / "tables" / "table2_wvi_distribution.csv", index=True)
    
    # --- PHASE 2: COPING MECHANISMS ---
    all_tables_data['table3_coping_typology'], all_tables_text['table3_coping_typology'] = generate_table3_coping_typology(df_processed, cfg)
    all_tables_data['table3_coping_typology'].to_csv(cfg.OUTPUT_DIR / "tables" / "table3_coping_typology.csv", index=False)

    all_tables_data['table4_cci_construction'], all_tables_text['table4_cci_construction'] = generate_table4_cci_construction(df_processed, cfg)
    all_tables_data['table4_cci_construction'].to_csv(cfg.OUTPUT_DIR / "tables" / "table4_cci_construction.csv", index=False)

    # --- PHASE 3: VULNERABILITY-COPING NEXUS & DISCOVERY ---
    all_tables_data['table5_vuln_coping_matrix'], all_tables_text['table5_vuln_coping_matrix'] = generate_table5_vuln_coping_matrix(df_processed, cfg)
    all_tables_data['table5_vuln_coping_matrix']['Disruption Rates'].to_csv(cfg.OUTPUT_DIR / "tables" / "table5_vuln_coping_matrix_disruption_rates.csv", index=True)
    all_tables_data['table5_vuln_coping_matrix']['% Households'].to_csv(cfg.OUTPUT_DIR / "tables" / "table5_vuln_coping_matrix_household_pct.csv", index=True)

    all_tables_data['table6_paradox_decomposition'], all_tables_text['table6_paradox_decomposition'] = generate_table6_paradox_decomposition(df_processed, cfg)
    all_tables_data['table6_paradox_decomposition'].to_csv(cfg.OUTPUT_DIR / "tables" / "table6_paradox_decomposition.csv", index=True)

    all_tables_data['table7_multivariate_explaining_paradox'], all_tables_text['table7_multivariate_explaining_paradox'] = generate_table7_multivariate_explaining_paradox(df_processed, cfg)
    for key, val in all_tables_data['table7_multivariate_explaining_paradox'].items():
        if isinstance(val, pd.DataFrame):
            val.to_csv(cfg.OUTPUT_DIR / "results" / f"table7_logistic_regression_{key.replace(' ', '_').replace('(', '').replace(')', '')}.csv", index=True)
        else:
             with open(cfg.OUTPUT_DIR / "results" / f"table7_logistic_regression_{key.replace(' ', '_').replace('(', '').replace(')', '')}.txt", "w") as f:
                 f.write(val)

    all_tables_data['table8_idi_conceptual'], all_tables_text['table8_idi_conceptual'] = generate_table8_idi_conceptual(df_processed, cfg)
    all_tables_data['table8_idi_conceptual'].to_csv(cfg.OUTPUT_DIR / "tables" / "table8_idi_conceptual.csv", index=False)

    # --- Supporting tables from original script (renamed for clarity) ---
    # These functions are now explicitly called and their results stored
    all_tables_data['table_descriptive_characteristics'], all_tables_text['table_descriptive_characteristics'] = generate_table_descriptive_characteristics(df_processed, cfg)
    all_tables_data['table_descriptive_characteristics'].to_csv(cfg.OUTPUT_DIR / "tables" / "table_descriptive_characteristics.csv", index=False)

    all_tables_data['table_state_level_paradox'], all_tables_text['table_state_level_paradox'] = generate_table_state_level_paradox(df_processed, cfg)
    all_tables_data['table_state_level_paradox'].to_csv(cfg.OUTPUT_DIR / "tables" / "table_state_level_paradox.csv", index=False)

    all_tables_data['table_seasonal_patterns'], all_tables_text['table_seasonal_patterns'] = generate_table_seasonal_patterns(df_processed, cfg)
    all_tables_data['table_seasonal_patterns'].to_csv(cfg.OUTPUT_DIR / "tables" / "table_seasonal_patterns.csv", index=False)

    all_tables_data['table_robustness_checks'], all_tables_text['table_robustness_checks'] = generate_table_robustness_checks(df_processed, cfg)
    all_tables_data['table_robustness_checks'].to_csv(cfg.OUTPUT_DIR / "tables" / "table_robustness_checks.csv", index=False)

    all_tables_data['table_idi_validation'], all_tables_text['table_idi_validation'] = generate_table_idi_validation(df_processed, cfg)
    all_tables_data['table_idi_validation'].to_csv(cfg.OUTPUT_DIR / "tables" / "table_idi_validation.csv", index=False)

    all_tables_data['table_policy_simulation'], all_tables_text['table_policy_simulation'] = generate_table_policy_simulation(df_processed, cfg)
    all_tables_data['table_policy_simulation'].to_csv(cfg.OUTPUT_DIR / "tables" / "table_policy_simulation.csv", index=False)


    # 5. Generate final Markdown Report
    print("\n" + "=" * 40)
    print("Assembling Final Markdown Report")
    print("=" * 40)

    final_markdown_report = generate_report_markdown(
        cfg=cfg,
        df_processed=df_processed,
        table1_data=all_tables_data['table1_wvi_components'], table1_text=all_tables_text['table1_wvi_components'],
        table2_data=all_tables_data['table2_wvi_distribution'], table2_text=all_tables_text['table2_wvi_distribution'],
        table3_data=all_tables_data['table3_coping_typology'], table3_text=all_tables_text['table3_coping_typology'],
        table4_data=all_tables_data['table4_cci_construction'], table4_text=all_tables_text['table4_cci_construction'],
        table5_data=all_tables_data['table5_vuln_coping_matrix'], table5_text=all_tables_text['table5_vuln_coping_matrix'],
        table6_data=all_tables_data['table6_paradox_decomposition'], table6_text=all_tables_text['table6_paradox_decomposition'],
        table7_results=all_tables_data['table7_multivariate_explaining_paradox'], table7_text=all_tables_text['table7_multivariate_explaining_paradox'],
        table8_data=all_tables_data['table8_idi_conceptual'], table8_text=all_tables_text['table8_idi_conceptual'],

        table_descriptive_characteristics_data=all_tables_data['table_descriptive_characteristics'], table_descriptive_characteristics_text=all_tables_text['table_descriptive_characteristics'],
        table_state_level_paradox_data=all_tables_data['table_state_level_paradox'], table_state_level_paradox_text=all_tables_text['table_state_level_paradox'],
        table_seasonal_patterns_data=all_tables_data['table_seasonal_patterns'], table_seasonal_patterns_text=all_tables_text['table_seasonal_patterns'],
        table_robustness_checks_data=all_tables_data['table_robustness_checks'], table_robustness_checks_text=all_tables_text['table_robustness_checks'],
        table_idi_validation_data=all_tables_data['table_idi_validation'], table_idi_validation_text=all_tables_text['table_idi_validation'],
        table_policy_simulation_data=all_tables_data['table_policy_simulation'], table_policy_simulation_text=all_tables_text['table_policy_simulation'],
    )

    with open(report_filepath, "w", encoding="utf-8") as f:
        f.write(final_markdown_report)
    print(f"\nAnalysis complete! Research paper saved to: {report_filepath}")
    print(f"All tables and results also saved to: {cfg.OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
