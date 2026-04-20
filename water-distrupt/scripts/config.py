"""
config.py
─────────────────────────────────────────────────────────────────────────────
Single source of truth for ALL constants, paths, variable names, and mappings.

Rules:
  - NO logic here. Only data.
  - When NFHS-6 arrives, only this file changes.
  - When you move servers, only OUTPUT_DIR changes.
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

    # Outcome
    VAR_DISRUPTED_RAW:   str = "sh37b"       # Raw disruption question (1=yes, 0=no, 8/9=missing)
    VAR_DISRUPTED:       str = "water_disrupted"  # Cleaned binary outcome we create

    # Survey design
    VAR_WEIGHT:   str = "hv005"
    VAR_PSU:      str = "hv021"
    VAR_STRATUM:  str = "hv022"
    VAR_CLUSTER:  str = "hv001"

    # Geography
    VAR_STATE:    str = "hv024"
    VAR_DISTRICT: str = "shdist"
    VAR_URBAN:    str = "hv025"   # 1=Urban, 2=Rural
    VAR_PLACE:    str = "hv026"

    # Water
    VAR_SOURCE_PRIMARY:   str = "hv201"   # Primary drinking water source
    VAR_SOURCE_ALT:       str = "hv202"   # Alternative / other source
    VAR_TIME_TO_WATER:    str = "hv204"   # Minutes (996 = on premises)
    VAR_WATER_LOCATION:   str = "hv235"   # 1=In dwelling, 2=In yard, 3=Elsewhere
    VAR_FETCHER_MAIN:     str = "hv236"   # Who fetches water (1=adult woman, 2=man, 3=child)
    VAR_FETCHER_CHILD:    str = "hv236a"  # Child fetcher detail (may not exist)

    # Socioeconomic
    VAR_WEALTH_QUINTILE:  str = "hv270"   # 1=Poorest … 5=Richest
    VAR_WEALTH_SCORE:     str = "hv271"   # Continuous wealth score
    VAR_HH_SIZE:          str = "hv009"
    VAR_CHILDREN_U5:      str = "hv014"
    VAR_HEAD_SEX:         str = "hv219"   # 1=Male, 2=Female
    VAR_RELIGION:         str = "sh47"
    VAR_CASTE:            str = "sh49"

    # Assets / infrastructure
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
    VAR_HOUSE_TYPE:   str = "shnfhs2"    # 1=Pucca, 2=Semi-pucca, 3=Katcha

    # Interview timing
    VAR_MONTH:  str = "hv006"
    VAR_YEAR:   str = "hv007"
    VAR_CMC:    str = "hv008"

    # ─── Codes treated as missing ─────────────────────────────────────────
    MISSING_CODES: List[int] = field(default_factory=lambda: [
        8, 9, 98, 99, 998, 999, 9996, 9998, 9999
    ])

    # ─── Water source grouping ────────────────────────────────────────────
    # Maps raw hv201/hv202 codes → human-readable category
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

    # Groups used for the Finding 1 dot-plot ordering
    # (ordered worst → best historically, so piped standing out is visible)
    SOURCE_ORDER: List[str] = field(default_factory=lambda: [
        "Surface Water",
        "Unprotected Well/Spring",
        "Unprotected Spring",
        "Tanker/Cart",
        "Piped Water",           # ← paradox: should be low, often is high
        "Rainwater",
        "Community RO Plant",
        "Protected Well/Spring",
        "Protected Spring",
        "Tube Well/Borehole",
        "Bottled Water",
        "Other Source",
    ])

    # ─── Geography mappings ───────────────────────────────────────────────
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

    # ─── Analysis parameters ──────────────────────────────────────────────
    ALPHA:               float = 0.05
    MIN_DISTRICT_N:      int   = 100    # min households for district to appear in RGI
    MONTE_CARLO_RUNS:    int   = 1000
    MONTE_CARLO_NOISE:   float = 0.3    # std of Gaussian noise on dimension scores
    MONTE_CARLO_SEED:    int   = 42
    PSM_CALIPER:         float = 0.05
    PSM_N_NEIGHBORS:     int   = 1

    # ─── Columns to load from .dta (extended in __post_init__) ────────────
    COLS_TO_LOAD: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Build output sub-dirs
        for sub in ["tables", "figures", "results"]:
            (self.OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)

        # Base columns always needed
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
        # Add hv101_XX and hv106_XX for HH head education derivation
        for i in range(1, 16):
            base.append(f"hv101_{i:02d}")
            base.append(f"hv106_{i:02d}")

        self.COLS_TO_LOAD = list(set(base))
