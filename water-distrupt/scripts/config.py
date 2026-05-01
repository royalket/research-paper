"""
config.py
─────────────────────────────────────────────────────────────────────────────
Single source of truth for ALL constants, paths, variable names, mappings.

Rules:
  - NO logic here. Only data.
  - When NFHS-6 arrives, only this file changes.
  - When you move servers, only OUTPUT_DIR changes.

Changes from v1:
  - MONTE_CARLO_RUNS reduced 1000 → 500  (sufficient for 95% CI convergence)
  - Added MIN_DISTRICT_PIPED_N  (stage-1 slope filter)
  - Added MONSOON_MONTHS         (shared by RGI aggregation + MC formula)
  - Added SLOPE_MODEL_MIN_OBS   (minimum observations per district-regression)
  - Added GEE_CORRELATION        (drives MultilevelModel cov structure)
─────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Config:

    # ─── Paths ────────────────────────────────────────────────────────────
    DATA_FILE_PATH: Path = Path("/Users/kumar.aniket/Downloads/surveydata/IAHR7EFL.DTA")
    OUTPUT_DIR:     Path = Path("./nfhs5_output")

    # ─── NFHS-5 Core Variable Names ───────────────────────────────────────

    VAR_DISRUPTED_RAW:   str = "sh37b"
    VAR_DISRUPTED:       str = "water_disrupted"

    VAR_WEIGHT:   str = "hv005"
    VAR_PSU:      str = "hv021"
    VAR_STRATUM:  str = "hv022"
    VAR_CLUSTER:  str = "hv001"

    VAR_STATE:    str = "hv024"
    VAR_DISTRICT: str = "shdist"
    VAR_URBAN:    str = "hv025"
    VAR_PLACE:    str = "hv026"

    VAR_SOURCE_PRIMARY:   str = "hv201"
    VAR_SOURCE_ALT:       str = "hv202"
    VAR_TIME_TO_WATER:    str = "hv204"
    VAR_WATER_LOCATION:   str = "hv235"
    VAR_FETCHER_MAIN:     str = "hv236"
    VAR_FETCHER_CHILD:    str = "hv236a"

    VAR_WEALTH_QUINTILE:  str = "hv270"
    VAR_WEALTH_SCORE:     str = "hv271"
    VAR_HH_SIZE:          str = "hv009"
    VAR_CHILDREN_U5:      str = "hv014"
    VAR_HEAD_SEX:         str = "hv219"
    VAR_RELIGION:         str = "sh47"
    VAR_CASTE:            str = "sh49"

    VAR_ELECTRICITY:  str = "hv206"
    VAR_RADIO:        str = "hv207"
    VAR_TV:           str = "hv208"
    VAR_FRIDGE:       str = "hv209"
    VAR_BICYCLE:      str = "hv210"
    VAR_MOTORCYCLE:   str = "hv211"
    VAR_CAR:          str = "hv212"
    VAR_LANDLINE:     str = "hv221"
    VAR_MOBILE:       str = "hv243a"
    VAR_TOILET:       str = "hv205"
    VAR_HOUSE_TYPE:   str = "shnfhs2"

    VAR_MONTH:  str = "hv006"
    VAR_YEAR:   str = "hv007"
    VAR_CMC:    str = "hv008"

    # ─── Missing codes ────────────────────────────────────────────────────
    MISSING_CODES: List[int] = field(default_factory=lambda: [
        8, 9, 98, 99, 998, 999, 9996, 9998, 9999
    ])

    # ─── Water source mapping ─────────────────────────────────────────────
    WATER_SOURCE_MAP: Dict[int, str] = field(default_factory=lambda: {
        11: "Piped Water", 12: "Piped Water",
        13: "Piped Water", 14: "Piped Water",
        21: "Tube Well/Borehole",
        31: "Protected Well/Spring",
        32: "Unprotected Well/Spring",
        41: "Protected Spring",
        42: "Unprotected Spring",
        43: "Surface Water",
        51: "Rainwater",
        61: "Tanker/Cart",  62: "Tanker/Cart",
        71: "Bottled Water",
        92: "Community RO Plant",
        96: "Other Source",
    })

    SOURCE_ORDER: List[str] = field(default_factory=lambda: [
        "Surface Water", "Unprotected Well/Spring", "Unprotected Spring",
        "Tanker/Cart", "Piped Water", "Rainwater", "Community RO Plant",
        "Protected Well/Spring", "Protected Spring",
        "Tube Well/Borehole", "Bottled Water", "Other Source",
    ])

    # ─── Geography ────────────────────────────────────────────────────────
    STATE_NAMES: Dict[int, str] = field(default_factory=lambda: {
        1:  "Jammu & Kashmir",    2:  "Himachal Pradesh",
        3:  "Punjab",             4:  "Chandigarh",
        5:  "Uttarakhand",        6:  "Haryana",
        7:  "NCT of Delhi",       8:  "Rajasthan",
        9:  "Uttar Pradesh",      10: "Bihar",
        11: "Sikkim",             12: "Arunachal Pradesh",
        13: "Nagaland",           14: "Manipur",
        15: "Mizoram",            16: "Tripura",
        17: "Meghalaya",          18: "Assam",
        19: "West Bengal",        20: "Jharkhand",
        21: "Odisha",             22: "Chhattisgarh",
        23: "Madhya Pradesh",     24: "Gujarat",
        25: "Dadra & NH and DD",  27: "Maharashtra",
        28: "Andhra Pradesh",     29: "Karnataka",
        30: "Goa",                31: "Lakshadweep",
        32: "Kerala",             33: "Tamil Nadu",
        34: "Puducherry",         35: "Andaman & Nicobar",
        36: "Telangana",          37: "Ladakh",
    })

    REGIONS: Dict[str, List[int]] = field(default_factory=lambda: {
        "North":     [1, 2, 3, 4, 5, 6, 7, 37],
        "Central":   [8, 9, 10, 23],
        "East":      [19, 20, 21, 22],
        "Northeast": [11, 12, 13, 14, 15, 16, 17, 18],
        "West":      [24, 25, 27, 30],
        "South":     [28, 29, 32, 33, 34, 36, 31, 35],
    })

    SEASONS: Dict[str, List[int]] = field(default_factory=lambda: {
        "Winter":       [12, 1, 2],
        "Summer":       [3, 4, 5],
        "Monsoon":      [6, 7, 8, 9],
        "Post-monsoon": [10, 11],
    })

    # ─── NEW: Monsoon months (shared by RGI aggregation + MC formula) ─────
    MONSOON_MONTHS: List[int] = field(default_factory=lambda: [6, 7, 8, 9])

    # ─── Analysis parameters ──────────────────────────────────────────────
    ALPHA:               float = 0.05
    MIN_DISTRICT_N:      int   = 100   # min HH for district to appear in RGI
    MIN_DISTRICT_PIPED_N: int  = 50    # NEW: min piped-HH for slope-as-outcome
    SLOPE_MODEL_MIN_OBS: int   = 30    # NEW: min obs for per-district logit
    MONTE_CARLO_RUNS:    int   = 500   # REDUCED from 1000 — CI stable at 500
    MONTE_CARLO_NOISE:   float = 0.3
    MONTE_CARLO_SEED:    int   = 42
    PSM_CALIPER:         float = 0.05
    PSM_N_NEIGHBORS:     int   = 1
    GEE_CORRELATION:     str   = "exchangeable"  # NEW: for MultilevelModel

    # ─── Columns to load from .dta ────────────────────────────────────────
    COLS_TO_LOAD: List[str] = field(default_factory=list)

    def __post_init__(self):
        for sub in ["tables", "figures", "results"]:
            (self.OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)

        base = [
            self.VAR_WEIGHT, self.VAR_PSU, self.VAR_STRATUM, self.VAR_CLUSTER,
            self.VAR_STATE, self.VAR_DISTRICT, self.VAR_URBAN, self.VAR_PLACE,
            self.VAR_DISRUPTED_RAW,
            self.VAR_SOURCE_PRIMARY, self.VAR_SOURCE_ALT,
            self.VAR_TIME_TO_WATER, self.VAR_WATER_LOCATION,
            self.VAR_FETCHER_MAIN, self.VAR_FETCHER_CHILD,
            self.VAR_WEALTH_QUINTILE, self.VAR_WEALTH_SCORE,
            self.VAR_HH_SIZE, self.VAR_CHILDREN_U5,
            self.VAR_HEAD_SEX, self.VAR_RELIGION, self.VAR_CASTE,
            self.VAR_ELECTRICITY, self.VAR_RADIO, self.VAR_TV,
            self.VAR_FRIDGE, self.VAR_BICYCLE, self.VAR_MOTORCYCLE,
            self.VAR_CAR, self.VAR_LANDLINE, self.VAR_MOBILE,
            self.VAR_TOILET, self.VAR_HOUSE_TYPE,
            self.VAR_MONTH, self.VAR_YEAR, self.VAR_CMC,
        ]
        for i in range(1, 16):
            base.append(f"hv101_{i:02d}")
            base.append(f"hv106_{i:02d}")

        self.COLS_TO_LOAD = list(set(base))
