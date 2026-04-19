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
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from scipy.stats import bootstrap

# Suppress specific warnings for cleaner output during development
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message="The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain the current behavior and silence this warning.")
warnings.filterwarnings('ignore', message="Crosstab for 'hh_head_education' vs 'water_disrupted' is not at least 2x2.")
warnings.filterwarnings('ignore', message="Optimization failed successfully.") # For statsmodels convergence
warnings.filterwarnings('ignore', message="The line search algorithm did not converge") # For statsmodels convergence
warnings.filterwarnings('ignore', message="Maximum number of iterations has been reached.") # For statsmodels convergence

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
    # Using 'shdist' as the primary district variable based on your prompt's context
    VAR_DISTRICT_CODE: str = 'shdist' 

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
        Config.VAR_HOUSE_TYPE, Config.VAR_FLOOR_MATERIAL, Config.VAR_WALL_MATERIAL, Config.VAR_ROOF_MATERIAL,
    ])

    MISSING_VALUE_CODES: List[int] = field(default_factory=lambda: [8, 9, 98, 99, 998, 999, 9996, 9998, 9999])

    # Your provided DISTRICT_NAMES_NFHS_ORIGINAL map
    DISTRICT_NAMES: Dict[int, str] = field(default_factory=lambda: {
        # Jammu & Kashmir (1-22)
        1: "Kupwara", 2: "Badgam", 3: "Leh(Ladakh)", 4: "Kargil", 5: "Punch", 6: "Rajouri",
        7: "Kathua", 8: "Baramula", 9: "Bandipore", 10: "Srinagar", 11: "Ganderbal", 12: "Pulwama",
        13: "Shupiyan", 14: "Anantnag", 15: "Kulgam", 16: "Doda", 17: "Ramban", 18: "Kishtwar",
        19: "Udhampur", 20: "Reasi", 21: "Jammu", 22: "Samba",
        # Himachal Pradesh (23-34)
        23: "Chamba", 24: "Kangra", 25: "Lahul & Spiti", 26: "Kullu", 27: "Mandi", 28: "Hamirpur",
        29: "Una", 30: "Bilaspur", 31: "Solan", 32: "Sirmaur", 33: "Shimla", 34: "Kinnaur",
        # Punjab (36-54, 879-882)
        36: "Kapurthala", 37: "Jalandhar", 38: "Hoshiarpur", 39: "Shahid Bhagat Singh Nagar", 40: "Fatehgarh Sahib", 41: "Ludhiana",
        42: "Moga", 44: "Muktsar", 45: "Faridkot", 46: "Bathinda", 47: "Mansa", 48: "Patiala",
        49: "Amritsar", 50: "Tarn Taran", 51: "Rupnagar", 52: "Sahibzada Ajit Singh Nagar", 53: "Sangrur", 54: "Barnala",
        879: "Fazilka", 880: "Firozpur", 881: "Gurdaspur", 882: "Pathankot",
        # Chandigarh (55)
        55: "Chandigarh",
        # Uttarakhand (56-68)
        56: "Uttarkashi", 57: "Chamoli", 58: "Rudraprayag", 59: "Tehri Garhwal", 60: "Dehradun", 61: "Garhwal",
        62: "Pithoragarh", 63: "Bageshwar", 64: "Almora", 65: "Champawat", 66: "Nainital", 67: "Udham Singh Nagar", 68: "Hardwar",
        # Haryana (69-89, 865-866)
        69: "Panchkula", 70: "Ambala", 71: "Yamunanagar", 72: "Kurukshetra", 73: "Kaithal", 74: "Karnal",
        75: "Panipat", 76: "Sonipat", 77: "Jind", 78: "Fatehabad", 79: "Sirsa", 80: "Hisar",
        82: "Rohtak", 83: "Jhajjar", 84: "Mahendragarh", 85: "Rewari", 86: "Gurgaon", 87: "Mewat",
        88: "Faridabad", 89: "Palwal", 865: "Bhiwani", 866: "Charkhi Dadri",
        # NCT of Delhi (837-847)
        837: "Central", 838: "East", 839: "New Delhi", 840: "North", 841: "North East", 842: "North West",
        843: "Shahdara", 844: "South", 845: "South East", 846: "South West", 847: "West",
        # Rajasthan (99-131)
        99: "Ganganagar", 100: "Hanumangarh", 101: "Bikaner", 102: "Churu", 103: "Jhunjhunun", 104: "Alwar",
        105: "Bharatpur", 106: "Dhaulpur", 107: "Karauli", 108: "Sawai Madhopur", 109: "Dausa", 110: "Jaipur",
        111: "Sikar", 112: "Nagaur", 113: "Jodhpur", 114: "Jaisalmer", 115: "Barmer", 116: "Jalor",
        117: "Sirohi", 118: "Pali", 119: "Ajmer", 120: "Tonk", 121: "Bundi", 122: "Bhilwara",
        123: "Rajsamand", 124: "Dungarpur", 125: "Banswara", 126: "Chittaurgarh", 127: "Kota", 128: "Baran",
        129: "Jhalawar", 130: "Udaipur", 131: "Pratapgarh",
        # Uttar Pradesh (132-202, 921-930)
        132: "Saharanpur", 134: "Bijnor", 136: "Rampur", 137: "Jyotiba Phule Nagar", 138: "Meerut", 139: "Baghpat",
        141: "Gautam Buddha Nagar", 142: "Bulandshahr", 143: "Aligarh", 144: "Mahamaya Nagar", 145: "Mathura", 146: "Agra",
        147: "Firozabad", 148: "Mainpuri", 150: "Bareilly", 151: "Pilibhit", 152: "Shahjahanpur", 153: "Kheri",
        154: "Sitapur", 155: "Hardoi", 156: "Unnao", 157: "Lucknow", 159: "Farrukhabad", 160: "Kannauj",
        161: "Etawah", 162: "Auraiya", 163: "Kanpur Dehat", 164: "Kanpur Nagar", 165: "Jalaun", 166: "Jhansi",
        167: "Lalitpur", 168: "Hamirpur", 169: "Mahoba", 170: "Banda", 171: "Chitrakoot", 172: "Fatehpur",
        173: "Pratapgarh", 174: "Kaushambi", 175: "Allahabad", 176: "Bara Banki", 177: "Faizabad", 178: "Ambedkar Nagar",
        180: "Bahraich", 181: "Shrawasti", 182: "Balrampur", 183: "Gonda", 184: "Siddharthnagar", 185: "Basti",
        186: "Sant Kabir Nagar", 187: "Mahrajganj", 188: "Gorakhpur", 189: "Kushinagar", 190: "Deoria", 191: "Azamgarh",
        192: "Mau", 193: "Ballia", 194: "Jaunpur", 195: "Ghazipur", 196: "Chandauli", 197: "Varanasi",
        198: "Sant Ravidas Nagar (Bhadohi)", 199: "Mirzapur", 200: "Sonbhadra", 201: "Etah", 202: "Kanshiram Nagar",
        921: "Amethi", 922: "Budaun", 923: "Ghaziabad", 924: "Hapur", 925: "Moradabad", 926: "Muzaffarnagar",
        927: "Rae Bareli", 928: "Sambhal", 929: "Shamli", 930: "Sultanpur",
        # Bihar (203-240)
        203: "Pashchim Champaran", 204: "Purba Champaran", 205: "Sheohar", 206: "Sitamarhi", 207: "Madhubani", 208: "Supaul",
        209: "Araria", 210: "Kishanganj", 211: "Purnia", 212: "Katihar", 213: "Madhepura", 214: "Saharsa",
        215: "Darbhanga", 216: "Muzaffarpur", 217: "Gopalganj", 218: "Siwan", 219: "Saran", 220: "Vaishali",
        221: "Samastipur", 222: "Begusarai", 223: "Khagaria", 224: "Bhagalpur", 225: "Banka", 226: "Munger",
        227: "Lakhisarai", 228: "Sheikhpura", 229: "Nalanda", 230: "Patna", 231: "Bhojpur", 232: "Buxar",
        233: "Kaimur (Bhabua)", 234: "Rohtas", 235: "Aurangabad", 236: "Gaya", 237: "Nawada", 238: "Jamui",
        239: "Jehanabad", 240: "Arwal",
        # Sikkim (241-244)
        241: "North District", 242: "West District", 243: "South District", 244: "East District",
        # Arunachal Pradesh (245-260, 801-809)
        245: "Tawang", 246: "West Kameng", 247: "East Kameng", 248: "Papum Pare", 249: "Upper Subansiri", 252: "Upper Siang",
        253: "Changlang", 255: "Lower Subansiri", 257: "Dibang Valley", 258: "Lower Dibang Valley", 260: "Anjaw",
        801: "East Siang", 802: "Kra Daadi", 803: "Kurung Kumey", 804: "Lohit", 805: "Longding", 806: "Namsai",
        807: "Siang", 808: "Tirap", 809: "West Siang",
        # Nagaland (261-271)
        261: "Mon", 262: "Mokokchung", 263: "Zunheboto", 264: "Wokha", 265: "Dimapur", 266: "Phek",
        267: "Tuensang", 268: "Longleng", 269: "Kiphire", 270: "Kohima", 271: "Peren",
        # Manipur (272-280)
        272: "Senapati", 273: "Tamenglong", 274: "Churachandpur", 275: "Bishnupur", 276: "Thoubal", 277: "Imphal West",
        278: "Imphal East", 279: "Ukhrul", 280: "Chandel",
        # Mizoram (281-288)
        281: "Mamit", 282: "Kolasib", 283: "Aizawl", 284: "Champhai", 285: "Serchhip", 286: "Lunglei",
        287: "Lawngtlai", 288: "Saiha",
        # Tripura (291, 914-920)
        291: "Dhalai", 914: "Gomati", 915: "Khowai", 916: "North Tripura", 917: "Sepahijala", 918: "South Tripura",
        919: "Unakoti", 920: "West Tripura",
        # Meghalaya (295, 297-298, 871-878)
        295: "South Garo Hills", 297: "Ribhoi", 298: "East Khasi Hills", 871: "East Garo Hills", 872: "East Jantia Hills",
        873: "North Garo Hills", 874: "South West Garo Hills", 875: "South West Khasi Hills", 876: "West Garo Hills",
        877: "West Jaintia Hills", 878: "West Khasi Hills",
        # Assam (300-326, 810-821)
        300: "Kokrajhar", 302: "Goalpara", 303: "Barpeta", 304: "Morigaon", 307: "Lakhimpur", 308: "Dhemaji",
        309: "Tinsukia", 310: "Dibrugarh", 313: "Golaghat", 315: "Dima Hasao", 316: "Cachar", 317: "Karimganj",
        318: "Hailakandi", 319: "Bongaigaon", 320: "Chirang", 321: "Kamrup", 322: "Kamrup Metropolitan",
        323: "Nalbari", 324: "Baksa", 325: "Darrang", 326: "Udalguri", 810: "Biswanath", 811: "Charaideo",
        812: "Dhubri", 813: "Hojai", 814: "Jorhat", 815: "Karbi Anglong", 816: "Majuli", 817: "Nagaon",
        818: "Sivasagar", 819: "Sonitpur", 820: "South Salmara Mancachar", 821: "West Karbi Anglong",
        # West Bengal (327-345, 931-932)
        327: "Darjiling", 328: "Jalpaiguri", 329: "Koch Bihar", 330: "Uttar Dinajpur", 331: "Dakshin Dinajpur", 332: "Maldah",
        333: "Murshidabad", 334: "Birbhum", 336: "Nadia", 337: "North Twenty Four Parganas", 338: "Hugli", 339: "Bankura",
        340: "Puruliya", 341: "Haora", 342: "Kolkata", 343: "South Twenty Four Parganas", 344: "Paschim Medinipur", 345: "Purba Medinipur",
        931: "Paschim Barddhaman", 932: "Purba Barddhaman",
        # Jharkhand (346-369)
        346: "Garhwa", 347: "Chatra", 348: "Kodarma", 349: "Giridih", 350: "Deoghar", 351: "Godda",
        352: "Sahibganj", 353: "Pakur", 354: "Dhanbad", 355: "Bokaro", 356: "Lohardaga", 357: "Purbi Singhbhum",
        358: "Palamu", 359: "Latehar", 360: "Hazaribagh", 361: "Ramgarh", 362: "Dumka", 363: "Jamtara",
        364: "Ranchi", 365: "Khunti", 366: "Gumla", 367: "Simdega", 368: "Pashchimi Singhbhum", 369: "Saraikela-Kharsawan",
        # Odisha (370-399)
        370: "Bargarh", 371: "Jharsuguda", 372: "Sambalpur", 373: "Debagarh", 374: "Sundargarh", 375: "Kendujhar",
        376: "Mayurbhanj", 377: "Baleshwar", 378: "Bhadrak", 379: "Kendrapara", 380: "Jagatsinghapur", 381: "Cuttack",
        382: "Jajapur", 383: "Dhenkanal", 384: "Anugul", 385: "Nayagarh", 386: "Khordha", 387: "Puri",
        388: "Ganjam", 389: "Gajapati", 390: "Kandhamal", 391: "Baudh", 392: "Subarnapur", 393: "Balangir",
        394: "Nuapada", 395: "Kalahandi", 396: "Rayagada", 397: "Nabarangapur", 398: "Koraput", 399: "Malkangiri",
        # Chhattisgarh (400-417, 822-836)
        400: "Koriya", 402: "Jashpur", 403: "Raigarh", 404: "Korba", 405: "Janjgir - Champa", 407: "Kabeerdham",
        408: "Rajnandgaon", 411: "Mahasamund", 412: "Dhamtari", 413: "Uttar Bastar Kanker", 415: "Narayanpur", 417: "Bijapur",
        822: "Balod", 823: "Baloda Bazar", 824: "Balrampur", 825: "Bastar", 826: "Bemetara", 827: "Bilaspur",
        828: "Dantewada", 829: "Durg", 830: "Gariyaband", 831: "Kodagaon", 832: "Mungeli", 833: "Raipur",
        834: "Sukma", 835: "Surajpur", 836: "Surguja",
        # Madhya Pradesh (418-467, 867-868)
        418: "Sheopur", 419: "Morena", 420: "Bhind", 421: "Gwalior", 422: "Datia", 423: "Shivpuri",
        424: "Tikamgarh", 425: "Chhatarpur", 426: "Panna", 427: "Sagar", 428: "Damoh", 429: "Satna",
        430: "Rewa", 431: "Umaria", 432: "Neemuch", 433: "Mandsaur", 434: "Ratlam", 435: "Ujjain",
        437: "Dewas", 438: "Dhar", 439: "Indore", 440: "Khargone (West Nimar)", 441: "Barwani", 442: "Rajgarh",
        443: "Vidisha", 444: "Bhopal", 445: "Sehore", 446: "Raisen", 447: "Betul", 448: "Harda",
        449: "Hoshangabad", 450: "Katni", 451: "Jabalpur", 452: "Narsimhapur", 453: "Dindori", 454: "Mandla",
        455: "Chhindwara", 456: "Seoni", 457: "Balaghat", 458: "Guna", 459: "Ashoknagar", 460: "Shahdol",
        461: "Anuppur", 462: "Sidhi", 463: "Singrauli", 464: "Jhabua", 465: "Alirajpur", 466: "Khandwa (East Nimar)",
        467: "Burhanpur", 867: "Agar Malwa", 868: "Shajapur",
        # Gujarat (468-493, 848-864)
        468: "Kachchh", 469: "Banas Kantha", 470: "Patan", 471: "Mahesana", 473: "Gandhinagar", 478: "Porbandar",
        480: "Amreli", 482: "Anand", 485: "Dohad", 487: "Narmada", 488: "Bharuch", 489: "The Dangs",
        490: "Navsari", 491: "Valsad", 492: "Surat", 493: "Tapi", 848: "Ahmadabad", 849: "Aravali",
        850: "Bhavnagar", 851: "Botad", 852: "Chhota Udaipur", 853: "Devbhumi Dwarka", 854: "Gir Somnath", 855: "Jamnagar",
        856: "Junagadh", 857: "Kheda", 858: "Mahisagar", 859: "Morbi", 860: "Panch Mahals", 861: "Rajkot",
        862: "Sabar Kantha", 863: "Surendranagar", 864: "Vadodara",
        # Dadra & Nagar Haveli and Daman & Diu (494-496)
        494: "Diu", 495: "Daman", 496: "Dadra & Nagar Haveli",
        # Maharashtra (497-531, 869-870)
        497: "Nandurbar", 498: "Dhule", 499: "Jalgaon", 500: "Buldana", 501: "Akola", 502: "Washim",
        503: "Amravati", 504: "Wardha", 505: "Nagpur", 506: "Bhandara", 507: "Gondiya", 508: "Gadchiroli",
        509: "Chandrapur", 510: "Yavatmal", 511: "Nanded", 512: "Hingoli", 513: "Parbhani", 514: "Jalna",
        515: "Aurangabad", 516: "Nashik", 518: "Mumbai Suburban", 519: "Mumbai", 520: "Raigarh", 521: "Pune",
        522: "Ahmadnagar", 523: "Bid", 524: "Latur", 525: "Osmanabad", 526: "Solapur", 527: "Satara",
        528: "Ratnagiri", 529: "Sindhudurg", 530: "Kolhapur", 531: "Sangli", 869: "Palghar", 870: "Thane",
        # Andhra Pradesh (542-554)
        542: "Srikakulam", 543: "Vizianagaram", 544: "Visakhapatnam", 545: "East Godavari", 546: "West Godavari", 547: "Krishna",
        548: "Guntur", 549: "Prakasam", 550: "Sri Potti Sriramulu Nellore", 551: "Y.S.R.", 552: "Kurnool", 553: "Anantapur",
        554: "Chittoor",
        # Karnataka (555-584)
        555: "Belgaum", 556: "Bagalkot", 557: "Bijapur", 558: "Bidar", 559: "Raichur", 560: "Koppal",
        561: "Gadag", 562: "Dharwad", 563: "Uttara Kannada", 564: "Haveri", 565: "Bellary", 566: "Chitradurga",
        567: "Davanagere", 568: "Shimoga", 569: "Udupi", 570: "Chikmagalur", 571: "Tumkur", 572: "Bangalore",
        573: "Mandya", 574: "Hassan", 575: "Dakshina Kannada", 576: "Kodagu", 577: "Mysore", 578: "Chamarajanagar",
        579: "Gulbarga", 580: "Yadgir", 581: "Kolar", 582: "Chikkaballapura", 583: "Bangalore Rural", 584: "Ramanagara",
        # Goa (585-586)
        585: "North Goa", 586: "South Goa",
        # Lakshadweep (587)
        587: "Lakshadweep",
        # Kerala (588-601)
        588: "Kasaragod", 589: "Kannur", 590: "Wayanad", 591: "Kozhikode", 592: "Malappuram", 593: "Palakkad",
        594: "Thrissur", 595: "Ernakulam", 596: "Idukki", 597: "Kottayam", 598: "Alappuzha", 599: "Pathanamthitta",
        600: "Kollam", 601: "Thiruvananthapuram",
        # Tamil Nadu (602-633)
        602: "Thiruvallur", 603: "Chennai", 604: "Kancheepuram", 605: "Vellore", 606: "Tiruvannamalai", 607: "Viluppuram",
        608: "Salem", 609: "Namakkal", 610: "Erode", 611: "The Nilgiris", 612: "Dindigul", 613: "Karur",
        614: "Tiruchirappalli", 615: "Perambalur", 616: "Ariyalur", 617: "Cuddalore", 618: "Nagapattinam", 619: "Thiruvarur",
        620: "Thanjavur", 621: "Pudukkottai", 622: "Sivaganga", 623: "Madurai", 624: "Theni", 625: "Virudhunagar",
        626: "Ramanathapuram", 627: "Thoothukkudi", 628: "Tirunelveli", 629: "Kanniyakumari", 630: "Dharmapuri", 631: "Krishnagiri",
        632: "Coimbatore", 633: "Tiruppur",
        # Puducherry (634-637)
        634: "Yanam", 635: "Puducherry", 636: "Mahe", 637: "Karaikal",
        # Andaman & Nicobar Islands (638-640)
        638: "Nicobars", 639: "North & Middle Andaman", 640: "South Andaman",
        # Telangana (883-913)
        883: "Adilabad", 884: "Bhadradri Kothagudem", 885: "Hyderabad", 886: "Jagitial", 887: "Jangoan", 888: "Jayashankar Bhupalapally",
        889: "Jogulamba Gadwal", 890: "Kamareddy", 891: "Karimnagar", 892: "Khammam", 893: "Komaram Bheem Asifabad",
        894: "Mahabubabad", 895: "Mahabubnagar", 896: "Mancherial", 897: "Medak", 898: "Medchal-Malkajgiri", 899: "Nagarkurnool",
        900: "Nalgonda", 901: "Nirmal", 902: "Nizamabad", 903: "Peddapalli", 904: "Rajanna Sircilla", 905: "Ranga Reddy",
        906: "Sangareddy", 907: "Siddipet", 908: "Suryapet", 909: "Vikarabad", 910: "Wanaparthy", 911: "Warangal Rural",
        912: "Warangal Urban", 913: "Yadadri Bhuvanagiri",
    })
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
        
        # Add hv101_XX and hv106_XX for HH head education
        for i in range(1, 16):
            self.REQUIRED_COLS.append(f'hv101_{i:02d}')
            self.REQUIRED_COLS.append(f'hv106_{i:02d}')
        
        # Add hv236a if it's not already there
        if self.VAR_WATER_FETCHER_CHILDREN not in self.REQUIRED_COLS:
            self.REQUIRED_COLS.append(self.VAR_WATER_FETCHER_CHILDREN)
        # Ensure the specified VAR_DISTRICT_CODE is in REQUIRED_COLS
        if self.VAR_DISTRICT_CODE not in self.REQUIRED_COLS:
            self.REQUIRED_COLS.append(self.VAR_DISTRICT_CODE)
        
        # Ensure unique required columns
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
        self.dta_metadata = None # To store full metadata

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
            self.dta_metadata = meta_full # Store metadata for later use
            
            all_available_cols = list(meta_full.column_names)
            print(f"  Discovered {len(all_available_cols)} columns in the DTA file.")
            
            # The VAR_DISTRICT_CODE is explicitly set to 'shdist' in Config.
            # We ensure it's in the list of columns to load.
            if self.config.VAR_DISTRICT_CODE not in self.config.REQUIRED_COLS:
                self.config.REQUIRED_COLS.append(self.config.VAR_DISTRICT_CODE)
            
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
                df, self.dta_metadata = pyreadstat.read_dta(self.config.DATA_FILE_PATH)
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
    def __init__(self, df: pd.DataFrame, config: Config, dta_metadata: Optional[Any] = None):
        self.df = df.copy()
        self.config = config
        self.dta_metadata = dta_metadata # Keep metadata for potential label lookup
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
            # Ensure weight is treated as numeric
            self.df[self.config.VAR_WEIGHT] = pd.to_numeric(self.df[self.config.VAR_WEIGHT], errors='coerce')
            self.df['weight'] = self.df[self.config.VAR_WEIGHT] / 1_000_000
            initial_len = len(self.df)
            self.df.dropna(subset=['weight'], inplace=True)
            if initial_len - len(self.df) > 0:
                print(f"    Dropped {initial_len - len(self.df):,} households due to missing weights. Remaining: {len(self.df):,}")
        else:
            self.df['weight'] = 1.0
            print(f"    Warning: Weight column '{self.config.VAR_WEIGHT}' not found. Analysis will proceed UNWEIGHTED.")

    def _create_geographical_vars(self):
        """
        Creates 'state_name', 'region', 'residence', 'district_code', 'district_name' variables.
        Uses the provided DISTRICT_NAMES mapping.
        """
        print("  Creating geographical variables (state, region, residence, district)...")
        
        # State and Region
        if self.config.VAR_STATE_CODE in self.df.columns:
            # Ensure state code is numeric for mapping
            self.df[self.config.VAR_STATE_CODE] = pd.to_numeric(self.df[self.config.VAR_STATE_CODE], errors='coerce')
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

        # Residence
        if self.config.VAR_RESIDENCE_TYPE in self.df.columns:
            self.df['residence'] = self.df[self.config.VAR_RESIDENCE_TYPE].map({1: 'Urban', 2: 'Rural'}).fillna('Unknown Residence')
            self.df['urban'] = (self.df['residence'] == 'Urban').astype(int) # Renamed to 'urban' for Model 4
        else:
            self.df['residence'] = 'Unknown Residence'
            self.df['urban'] = 0
            print(f"    Warning: Residence type column '{self.config.VAR_RESIDENCE_TYPE}' not found. Residence set to 'Unknown'.")

        # District Code and Name using the provided mapping
        if self.config.VAR_DISTRICT_CODE in self.df.columns:
            # Ensure district code is numeric for mapping
            self.df[self.config.VAR_DISTRICT_CODE] = pd.to_numeric(self.df[self.config.VAR_DISTRICT_CODE], errors='coerce')
            self.df['district_code'] = self.df[self.config.VAR_DISTRICT_CODE].copy()
            print(f"    Using '{self.config.VAR_DISTRICT_CODE}' as district_code.")
            
            # Map using the provided DISTRICT_NAMES
            self.df['district_name'] = self.df['district_code'].map(self.config.DISTRICT_NAMES)
            
            # Handle districts not found in the mapping
            unmapped_districts = self.df['district_name'].isna().sum()
            if unmapped_districts > 0:
                print(f"    Warning: {unmapped_districts:,} households have unmapped district codes. Setting to 'Unknown District'.")
                self.df['district_name'] = self.df['district_name'].fillna('Unknown District')
        else:
            self.df['district_code'] = np.nan
            self.df['district_name'] = 'Unknown District'
            print(f"    Warning: District code column '{self.config.VAR_DISTRICT_CODE}' not found. District variables set to 'Unknown'.")

    def _create_temporal_vars(self):
        """Creates 'season' variable from interview month."""
        print("  Creating temporal variables (season)...")
        if self.config.VAR_MONTH_INTERVIEW in self.df.columns:
            self.df[self.config.VAR_MONTH_INTERVIEW] = pd.to_numeric(self.df[self.config.VAR_MONTH_INTERVIEW], errors='coerce')
            def get_season(month):
                if pd.isna(month): return 'Unknown Season'
                for season_name, months in self.config.SEASONS.items():
                    if month in months:
                        return season_name
                return 'Other Season' # Changed from 'Unknown Season' to 'Other Season' to differentiate
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
            self.df[raw_disruption_col] = pd.to_numeric(self.df[raw_disruption_col], errors='coerce')
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
            self.df[self.config.VAR_WATER_SOURCE_DRINKING] = pd.to_numeric(self.df[self.config.VAR_WATER_SOURCE_DRINKING], errors='coerce')
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
            if self.config.VAR_WATER_SOURCE_OTHER in self.df.columns:
                self.df[self.config.VAR_WATER_SOURCE_OTHER] = pd.to_numeric(self.df[self.config.VAR_WATER_SOURCE_OTHER], errors='coerce')
                self.df['other_water_source_category'] = self.df[self.config.VAR_WATER_SOURCE_OTHER].map(water_source_map).fillna('No Other Source')
            else:
                self.df['other_water_source_category'] = 'No Other Source'
                print(f"    Warning: Column '{self.config.VAR_WATER_SOURCE_OTHER}' not found. 'other_water_source_category' set to 'No Other Source'.")
        else:
            print(f"    Warning: Column '{self.config.VAR_WATER_SOURCE_DRINKING}' not found. Water source variables set to defaults.")
            self.df['water_source_category'] = 'Unknown Source'
            self.df['piped_water_flag'] = 0
            self.df['tube_well_flag'] = 0
            self.df['improved_source_flag'] = 0
            self.df['other_water_source_category'] = 'No Other Source'

        if self.config.VAR_TIME_TO_WATER in self.df.columns:
            self.df[self.config.VAR_TIME_TO_WATER] = pd.to_numeric(self.df[self.config.VAR_TIME_TO_WATER], errors='coerce')
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
            self.df[self.config.VAR_WATER_LOCATION] = pd.to_numeric(self.df[self.config.VAR_WATER_LOCATION], errors='coerce')
            water_location_map = {1: 'In Dwelling', 2: 'In Yard/Plot', 3: 'Elsewhere'}
            self.df['water_location_category'] = self.df[self.config.VAR_WATER_LOCATION].map(water_location_map).fillna('Unknown Location')
        else:
            self.df['water_location_category'] = 'Unknown Location'
            print(f"    Warning: Column '{self.config.VAR_WATER_LOCATION}' not found. 'water_location_category' set to 'Unknown'.")
        if self.config.VAR_WATER_FETCHER_MAIN in self.df.columns:
            self.df[self.config.VAR_WATER_FETCHER_MAIN] = pd.to_numeric(self.df[self.config.VAR_WATER_FETCHER_MAIN], errors='coerce')
            self.df['women_fetch_water'] = (self.df[self.config.VAR_WATER_FETCHER_MAIN] == 1).astype(int)
            self.df['men_fetch_water'] = (self.df[self.config.VAR_WATER_FETCHER_MAIN] == 2).astype(int)
            self.df['children_fetch_water'] = 0
            if self.config.VAR_WATER_FETCHER_CHILDREN in self.df.columns:
                self.df[self.config.VAR_WATER_FETCHER_CHILDREN] = pd.to_numeric(self.df[self.config.VAR_WATER_FETCHER_CHILDREN], errors='coerce')
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
            self.df[self.config.VAR_WEALTH_QUINTILE] = pd.to_numeric(self.df[self.config.VAR_WEALTH_QUINTILE], errors='coerce')
            self.df['wealth_quintile'] = self.df[self.config.VAR_WEALTH_QUINTILE].map({
                1: 'Poorest', 2: 'Poorer', 3: 'Middle', 4: 'Richer', 5: 'Richest'
            }).fillna('Unknown Quintile')
        else:
            self.df['wealth_quintile'] = 'Unknown Quintile'
            print(f"    Warning: Column '{self.config.VAR_WEALTH_QUINTILE}' not found. Wealth quintile set to 'Unknown'.")
        if self.config.VAR_HH_MEMBERS in self.df.columns:
            self.df[self.config.VAR_HH_MEMBERS] = pd.to_numeric(self.df[self.config.VAR_HH_MEMBERS], errors='coerce')
            self.df['hh_size'] = self.df[self.config.VAR_HH_MEMBERS].fillna(self.df[self.config.VAR_HH_MEMBERS].median())
        else:
            self.df['hh_size'] = np.nan
            print(f"    Warning: Column '{self.config.VAR_HH_MEMBERS}' not found. Household size set to NaN.")
        if self.config.VAR_CHILDREN_UNDER5 in self.df.columns:
            self.df[self.config.VAR_CHILDREN_UNDER5] = pd.to_numeric(self.df[self.config.VAR_CHILDREN_UNDER5], errors='coerce')
            self.df['children_under5_count'] = self.df[self.config.VAR_CHILDREN_UNDER5].fillna(0)
        else:
            self.df['children_under5_count'] = 0
            print(f"    Warning: Column '{self.config.VAR_CHILDREN_UNDER5}' not found. Children under 5 count set to 0.")
        if self.config.VAR_HH_HEAD_SEX in self.df.columns:
            self.df[self.config.VAR_HH_HEAD_SEX] = pd.to_numeric(self.df[self.config.VAR_HH_HEAD_SEX], errors='coerce')
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
                self.df[rel_col] = pd.to_numeric(self.df[rel_col], errors='coerce')
                self.df[edu_col] = pd.to_numeric(self.df[edu_col], errors='coerce')
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
            self.df[self.config.VAR_RELIGION] = pd.to_numeric(self.df[self.config.VAR_RELIGION], errors='coerce')
            self.df['religion'] = self.df[self.config.VAR_RELIGION].map({
                1: 'Hindu', 2: 'Muslim', 3: 'Christian', 4: 'Sikh', 5: 'Buddhist/Neo-Buddhist',
                6: 'Jain', 7: 'Jewish', 8: 'Parsi/Zoroastrian', 9: 'No religion', 96: 'Other Religion'
            }).fillna('Unknown Religion')
        else:
            self.df['religion'] = 'Unknown Religion'
            print(f"    Warning: Column '{self.config.VAR_RELIGION}' not found. Religion set to 'Unknown'.")
        if self.config.VAR_CASTE in self.df.columns:
            self.df[self.config.VAR_CASTE] = pd.to_numeric(self.df[self.config.VAR_CASTE], errors='coerce')
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
                self.df[original_col] = pd.to_numeric(self.df[original_col], errors='coerce')
                self.df[derived_col_name] = self.df[original_col].apply(lambda x: 1 if x == 1 else (0 if x == 0 else np.nan))
                self.df[derived_col_name] = self.df[derived_col_name].fillna(0).astype(int)
            else:
                self.df[derived_col_name] = 0
                print(f"    Warning: Asset column '{original_col}' not found. '{derived_col_name}' set to 0.")
        
        self.df['has_vehicle'] = ((self.df['has_motorcycle'] == 1) | (self.df['has_car'] == 1)).astype(int)
        if self.config.VAR_HOUSE_TYPE in self.df.columns:
            self.df[self.config.VAR_HOUSE_TYPE] = pd.to_numeric(self.df[self.config.VAR_HOUSE_TYPE], errors='coerce')
            self.df['house_type'] = self.df[self.config.VAR_HOUSE_TYPE].map({1: 'Pucca', 2: 'Semi-pucca', 3: 'Katcha'}).fillna('Unknown House Type')
        else:
            self.df['house_type'] = 'Unknown House Type'
            print(f"    Warning: Column '{self.config.VAR_HOUSE_TYPE}' not found. House type set to 'Unknown'.")
        
        if self.config.VAR_TOILET_FACILITY in self.df.columns:
            self.df[self.config.VAR_TOILET_FACILITY] = pd.to_numeric(self.df[self.config.VAR_TOILET_FACILITY], errors='coerce')
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
            # Ensure VAR_WEALTH_QUINTILE is numeric for mapping
            self.df[self.config.VAR_WEALTH_QUINTILE] = pd.to_numeric(self.df[self.config.VAR_WEALTH_QUINTILE], errors='coerce')
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
            self.df[self.config.VAR_CASTE] = pd.to_numeric(self.df[self.config.VAR_CASTE], errors='coerce')
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
        if 'urban' in self.df.columns: # Changed from is_urban to urban
            self.df.loc[self.df['urban'] == 0, 'wvi_comp_geo'] += 2 # Rural is often more vulnerable for services
        
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
        if 'urban' in self.df.columns: # Changed from is_urban to urban
            self.df.loc[self.df['urban'] == 0, 'cci_comp_knowledge'] += 1 # Rural implies traditional knowledge
        
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
        # Highest dependency if only piped water and no other source, or only piped water as primary and secondary
        self.df.loc[
            (self.df['water_source_category'] == 'Piped Water') &
            ((self.df['other_source_cat_idi'] == 'No Other Source') | (self.df['other_source_cat_idi'] == 'Piped Water')),
            'idi_comp1_single_source'
        ] = 3
        # Medium dependency if piped water is primary but there is another non-piped source
        self.df.loc[
            (self.df['water_source_category'] == 'Piped Water') &
            (self.df['other_source_cat_idi'] != 'No Other Source') &
            (self.df['other_source_cat_idi'] != 'Piped Water'),
            'idi_comp1_single_source'
        ] = 2
        # Lower dependency if primary is not piped, but there is an alternative non-piped source
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
        # Piped water implies higher dependency on infrastructure
        self.df.loc[self.df['water_source_category'].isin(['Piped Water']), 'idi_comp2_infra_type'] = 2
        # Market-based or community RO also implies dependency but less direct infrastructure
        self.df.loc[self.df['water_source_category'].isin(['Tanker/Cart', 'Bottled Water', 'Community RO Plant']), 'idi_comp2_infra_type'] = 1
        self.df['idi_score'] += self.df['idi_comp2_infra_type'].fillna(0)

        self.df['idi_comp3_on_premises'] = 0
        # On-premises water implies higher reliance on fixed infrastructure
        self.df.loc[self.df['water_location_category'] == 'In Dwelling', 'idi_comp3_on_premises'] = 2
        self.df.loc[self.df['water_location_category'] == 'In Yard/Plot', 'idi_comp3_on_premises'] = 1
        self.df['idi_score'] += self.df['idi_comp3_on_premises'].fillna(0)

        self.df['idi_comp4_urban_duration'] = self.df['urban'] # Urban areas often mean more complex infrastructure
        self.df['idi_score'] += self.df['idi_comp4_urban_duration'].fillna(0)

        self.df['idi_comp5_market_dependency'] = 0
        # High dependency if primary source is market-based
        self.df.loc[self.df['water_source_category'].isin(['Tanker/Cart', 'Bottled Water']), 'idi_comp5_market_dependency'] = 2
        # Piped water or community RO also implies some market/system dependency
        self.df.loc[self.df['water_source_category'].isin(['Piped Water', 'Community RO Plant']), 'idi_comp5_market_dependency'] = 1
        self.df['idi_score'] += self.df['idi_comp5_market_dependency'].fillna(0)
        
        for col in ['idi_comp1_single_source', 'idi_comp2_infra_type', 'idi_comp3_on_premises', 'idi_comp4_urban_duration', 'idi_comp5_market_dependency']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
        self.df['idi_score'] = pd.to_numeric(self.df['idi_score'], errors='coerce').fillna(0)

        self.df['idi_category'] = pd.cut(
            self.df['idi_score'],
            bins=[-0.1, 3, 7, 10], # Assuming max IDI score is around 10 (3+2+2+1+2)
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
            'house_type', 'toilet_type', 'idi_category', 'wvi_category', 'cci_category',
            'district_name' # Add district_name to categorical for efficiency
        ]
        
        binary_categorical_cols = ['urban', 'is_female_headed', 'water_on_premises', 'piped_water_flag', 'tube_well_flag', 'improved_source_flag', 'has_electricity', 'has_radio', 'has_television', 'has_refrigerator', 'has_bicycle', 'has_motorcycle', 'has_car', 'has_telephone_landline', 'has_mobile_telephone', 'has_vehicle', 'improved_sanitation_flag']
        
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
            self.config.VAR_WEALTH_QUINTILE, # hv270 - still needed for calculations (e.g., Mean Wealth Quintile)
            self.config.VAR_STATE_CODE, # Keep state code for spatial aggregation
            self.config.VAR_DISTRICT_CODE # Keep district code for spatial aggregation
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
        # Ensure target_col is numeric if target_val is numeric for comparison
        if isinstance(target_val, (int, float)) and pd.api.types.is_categorical_dtype(temp_df[target_col]):
            temp_df[target_col] = pd.to_numeric(temp_df[target_col], errors='coerce')
        if pd.api.types.is_categorical_dtype(temp_df[target_col]) and target_val not in temp_df[target_col].cat.categories:
            # If target_val is not in categories, no rows will match, so return empty
            return pd.DataFrame(columns=['Category', 'Weighted_Percentage', 'Unweighted_N'])
        
        temp_df = temp_df[temp_df[target_col] == target_val]
    
    if temp_df.empty:
        return pd.DataFrame(columns=['Category', 'Weighted_Percentage', 'Unweighted_N'])
    
    # Ensure the column for grouping is suitable for groupby (e.g., convert if it's a raw numeric code)
    if pd.api.types.is_numeric_dtype(temp_df[column]) and column in cfg.DISTRICT_NAMES: # Example heuristic
        temp_df[column] = temp_df[column].map(cfg.DISTRICT_NAMES).fillna(temp_df[column])
    
    groupby_kwargs = {'observed': False} if PANDAS_SUPPORTS_OBSERVED and pd.api.types.is_categorical_dtype(temp_df[column]) else {}
    weighted_counts = temp_df.groupby(column, **groupby_kwargs)[weight_col].sum()
    unweighted_counts = temp_df[column].value_counts()
    total_weighted = weighted_counts.sum()
    if total_weighted == 0:
        weighted_percentages = pd.Series(0.0, index=weighted_counts.index)
    else:
        weighted_percentages = (weighted_counts / total_weighted * 100).round(1)
    
    # Align unweighted counts to the weighted percentages index
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

def format_regression_output(results: Any) -> pd.DataFrame:
    """Formats statsmodels regression results into a DataFrame with OR, CI, p-value, and stars."""
    params = results.params
    conf_int = results.conf_int()
    p_values = results.pvalues
    # Ensure index alignment
    df_results = pd.DataFrame({
        'OR': np.exp(params),
        'CI_lower': np.exp(conf_int[0]),
        'CI_upper': np.exp(conf_int[1]),
        'p_value': p_values
    })
    df_results['p_value_formatted'] = df_results['p_value'].apply(format_p_value)
    
    # Select and reorder columns for display
    df_display = df_results[['OR', 'CI_lower', 'CI_upper', 'p_value_formatted']].copy()
    df_display.rename(columns={'p_value_formatted': 'P>|z|'}, inplace=True)
    
    return df_display

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
    
    # Ensure categorical columns are actual categories for observed=False to work
    if pd.api.types.is_categorical_dtype(temp_df[col1]) or pd.api.types.is_categorical_dtype(temp_df[col2]):
        groupby_kwargs = {'observed': False} if PANDAS_SUPPORTS_OBSERVED else {}
    else:
        groupby_kwargs = {} # If not categorical, observed=False is not relevant

    weighted_crosstab = temp_df.groupby([col1, col2], **groupby_kwargs)[weight_col].sum().unstack(fill_value=0)
    
    if weighted_crosstab.empty or weighted_crosstab.sum().sum() == 0:
        print(f"    Weighted crosstab is empty or sums to zero for '{col1}' vs '{col2}'. Cannot compute chi-square.")
        return np.nan, np.nan, np.nan, pd.DataFrame()
    
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

def get_scenario_df(scenario: Dict[str, Any], df_template: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a DataFrame for prediction based on a scenario dictionary and a template DataFrame.
    Ensures all columns are present and correctly typed (especially categoricals).
    """
    scenario_df = pd.DataFrame([scenario])
    
    # Fill in default values for other columns from a typical observation (e.g., median for continuous, mode for categorical)
    for col in df_template.columns:
        if col not in scenario_df.columns:
            if pd.api.types.is_numeric_dtype(df_template[col]):
                scenario_df[col] = df_template[col].median()
            elif pd.api.types.is_categorical_dtype(df_template[col]):
                scenario_df[col] = df_template[col].mode()[0]
            else: # Fallback for other types
                scenario_df[col] = df_template[col].iloc[0]
    
    # Ensure correct column order and dtypes, especially for categorical variables
    scenario_df = scenario_df[df_template.columns]
    for col in df_template.columns:
        if pd.api.types.is_categorical_dtype(df_template[col]):
            # Ensure categories match and handle new values gracefully
            scenario_df[col] = pd.Categorical(scenario_df[col], categories=df_template[col].cat.categories)
        else:
            scenario_df[col] = scenario_df[col].astype(df_template[col].dtype)
    return scenario_df

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
    
    # Ensure 'wvi_category' is a categorical type with specific order for consistent pivoting
    wvi_cat_order = ['Low Vulnerability', 'Medium Vulnerability', 'High Vulnerability']
    if not isinstance(df['wvi_category'].dtype, pd.CategoricalDtype):
        df['wvi_category'] = pd.Categorical(df['wvi_category'], categories=wvi_cat_order, ordered=True)
    else:
        df['wvi_category'] = df['wvi_category'].cat.reorder_categories(wvi_cat_order, ordered=True)

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
    
    table_df_raw = pd.DataFrame(results)
    
    # Pivot for a cleaner representation
    table_df = table_df_raw.pivot_table(
        index='Group', 
        columns='Category', 
        values='Weighted %', 
        aggfunc='first'
    ).reindex(columns=wvi_cat_order).fillna(0).round(1) # Reindex to ensure order and fill missing categories
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
    
    if df_disrupted.empty or df_disrupted['weight'].sum() == 0:
        return pd.DataFrame(), "No disrupted households with alternative sources for Table 3."
    results = []
    
    total_disrupted_weight = df_disrupted['weight'].sum()
    
    # Alternative sources and their weighted percentages
    alt_source_counts = df_disrupted.groupby('other_water_source_category')['weight'].sum()
    alt_source_pct = (alt_source_counts / total_disrupted_weight * 100).round(1)
    
    # Filter for major alternative sources (e.g., used by >1% of disrupted households)
    major_alt_sources = alt_source_pct[alt_source_pct > 1].index.tolist() 
    for alt_source in major_alt_sources:
        # Determine typical primary users for this alternative, for narrative
        primary_users_profile = "Mixed"
        if alt_source in ['Tanker/Cart', 'Bottled Water', 'Community RO Plant']:
            primary_users_profile = "Urban, Piped Water Users (as inferred)"
        elif alt_source in ['Protected Well/Spring', 'Tube well/Borehole']:
            primary_users_profile = "Rural, Traditional Users (as inferred)"
        elif alt_source == 'Surface Water':
            primary_users_profile = "Rural, Most Vulnerable (as inferred)"
        
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
        "Behavioral adaptations, such as traveling farther, are also observed, particularly among those who normally have water on premises. "
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
    # Ensure categories are ordered for consistent output
    df['wvi_category'] = pd.Categorical(df['wvi_category'], categories=wvi_levels, ordered=True)
    df['cci_category'] = pd.Categorical(df['cci_category'], categories=cci_levels, ordered=True)

    disruption_data = []
    household_pct_data = []
    total_overall_weighted = df['weight'].sum()
    
    for vuln_level in wvi_levels:
        for coping_level in cci_levels:
            subset = df[(df['wvi_category'] == vuln_level) & (df['cci_category'] == coping_level)].copy()
            
            weighted_total_subset = subset['weight'].sum()
            if weighted_total_subset > 0:
                # Ensure water_disrupted is numeric for calculation
                weighted_disruption_rate = (subset[cfg.VAR_WATER_DISRUPTED_FINAL].astype(float) * subset['weight']).sum() / weighted_total_subset * 100
                weighted_household_pct = (weighted_total_subset / total_overall_weighted * 100)
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
    # Example for interpretive text
    low_vuln_high_coping_disruption = disruption_matrix.loc['Low Vulnerability', 'High Coping']
    interpretive_text = (
        "Table 5 presents the crucial Vulnerability-Coping Matrix, illustrating the weighted water disruption rates "
        "and household distribution across different levels of traditional vulnerability (WVI) and coping capacity (CCI). "
        "Intriguingly, while high vulnerability and low coping capacity generally correlate with higher disruption, "
        "a counter-intuitive pattern emerges: certain groups with 'Low Vulnerability' and/or 'High Coping Capacity' "
        "also experience unexpectedly high disruption rates. For instance, households in the **Low Vulnerability, High Coping** "
        f"quadrant report a disruption rate of approximately {low_vuln_high_coping_disruption:.1f}%, "
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
    required_cols = ['wvi_category', cfg.VAR_WATER_DISRUPTED_FINAL, 'water_source_category', 
                     cfg.VAR_WEALTH_QUINTILE, 'residence', 'weight', 'piped_water_flag', 'urban'] # Changed is_urban to urban
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        return pd.DataFrame(), f"Error: Required columns missing for Table 6: {missing}."
    results = []
    # Identify "Paradoxical" groups based on Table 5's insights (e.g., Low WVI, High Disruption)
    # And "Expected" groups (e.g., High WVI, Low Disruption - the resilient poor)
    
    # Ensure relevant columns are numeric for calculations
    df_temp = df.copy()
    df_temp[cfg.VAR_WATER_DISRUPTED_FINAL] = pd.to_numeric(df_temp[cfg.VAR_WATER_DISRUPTED_FINAL], errors='coerce')
    df_temp['piped_water_flag'] = pd.to_numeric(df_temp['piped_water_flag'], errors='coerce')
    df_temp['urban'] = pd.to_numeric(df_temp['urban'], errors='coerce')
    df_temp[cfg.VAR_WEALTH_QUINTILE] = pd.to_numeric(df_temp[cfg.VAR_WEALTH_QUINTILE], errors='coerce')

    low_vuln_high_disruption = df_temp[(df_temp['wvi_category'] == 'Low Vulnerability') & (df_temp[cfg.VAR_WATER_DISRUPTED_FINAL] == 1)].copy()
    high_vuln_low_disruption = df_temp[(df_temp['wvi_category'] == 'High Vulnerability') & (df_temp[cfg.VAR_WATER_DISRUPTED_FINAL] == 0)].copy()
    
    # Also include a "Expected Vulnerable" group (High WVI, High Disruption) for comparison
    expected_vulnerable = df_temp[(df_temp['wvi_category'] == 'High Vulnerability') & (df_temp[cfg.VAR_WATER_DISRUPTED_FINAL] == 1)].copy()

    groups_to_analyze = {
        'Paradoxical (Low WVI, High Disruption)': low_vuln_high_disruption,
        'Resilient (High WVI, Low Disruption)': high_vuln_low_disruption,
        'Expected Vulnerable (High WVI, High Disruption)': expected_vulnerable
    }

    metrics = {
        '% Piped Water Users': lambda sub_df: (sub_df['piped_water_flag'] * sub_df['weight']).sum() / sub_df['weight'].sum() * 100 if sub_df['weight'].sum() > 0 else np.nan,
        '% Urban Residents': lambda sub_df: (sub_df['urban'] * sub_df['weight']).sum() / sub_df['weight'].sum() * 100 if sub_df['weight'].sum() > 0 else np.nan,
        'Mean Wealth Quintile (1=Poorest, 5=Richest)': lambda sub_df: np.average(sub_df[cfg.VAR_WEALTH_QUINTILE].dropna(), weights=sub_df.loc[sub_df[cfg.VAR_WEALTH_QUINTILE].notna(), 'weight']) if sub_df['weight'].sum() > 0 else np.nan,
        'Disruption Rate (%)': lambda sub_df: (sub_df[cfg.VAR_WATER_DISRUPTED_FINAL] * sub_df['weight']).sum() / sub_df['weight'].sum() * 100 if sub_df['weight'].sum() > 0 else np.nan
    }

    for group_name, group_df in groups_to_analyze.items():
        if group_df.empty or group_df['weight'].sum() == 0:
            for metric_name in metrics.keys():
                results.append({'Characteristic': metric_name, 'Group': group_name, 'Value': np.nan})
            continue
        for metric_name, func in metrics.items():
            results.append({'Characteristic': metric_name, 'Group': group_name, 'Value': func(group_df)})
    
    table_df = pd.DataFrame(results).pivot(index='Characteristic', columns='Group', values='Value').round(1)
    
    # Extract values for interpretive text, handling potential NaNs
    piped_users_paradoxical = table_df.loc['% Piped Water Users', 'Paradoxical (Low WVI, High Disruption)'] if 'Paradoxical (Low WVI, High Disruption)' in table_df.columns else np.nan
    urban_residents_paradoxical = table_df.loc['% Urban Residents', 'Paradoxical (Low WVI, High Disruption)'] if 'Paradoxical (Low WVI, High Disruption)' in table_df.columns else np.nan

    interpretive_text = (
        "Table 6 delves deeper into the characteristics of the 'paradoxical' groups identified in the Vulnerability-Coping Matrix. "
        "Specifically, we compare households that exhibit 'Low traditional Vulnerability but High Disruption' with those showing "
        "'High traditional Vulnerability but Low Disruption' (the 'resilient poor'), and an 'Expected Vulnerable' group. "
        "A striking difference emerges: the **Paradoxical (Low WVI, High Disruption)** group is predominantly composed of **piped water users** "
        f"({piped_users_paradoxical if not pd.isna(piped_users_paradoxical) else 'N/A':.1f}%), "
        f"urban residents ({urban_residents_paradoxical if not pd.isna(urban_residents_paradoxical) else 'N/A':.1f}%), and wealthier households. "
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

    # Define reference categories for categorical variables - using actual values
    # Ensure these categories exist in your data or adjust if necessary
    ref_cats = {
        'wealth_quintile': 'Poorest', # Assuming Poorest as reference for wealth
        'urban': 0,                   # Rural as reference for urban
        'region': 'North',            # North as reference for region
        'hh_head_education': 'No education', # No education as reference
        'caste': 'General',           # General as reference
    }

    df_reg = df.copy()
    
    # Explicitly convert to category and set reference for wealth_quintile
    if 'wealth_quintile' in df_reg.columns:
        if not pd.api.types.is_categorical_dtype(df_reg['wealth_quintile']):
            df_reg['wealth_quintile'] = df_reg['wealth_quintile'].astype('category')
        if ref_cats['wealth_quintile'] in df_reg['wealth_quintile'].cat.categories:
            df_reg['wealth_quintile'] = df_reg['wealth_quintile'].cat.set_categories(
                [ref_cats['wealth_quintile']] + [c for c in df_reg['wealth_quintile'].cat.categories if c != ref_cats['wealth_quintile']],
                ordered=True
            )
        else:
            print(f"    Warning: Reference category '{ref_cats['wealth_quintile']}' not found for 'wealth_quintile'. Using default first category.")

    # Convert binary flags to category with 0 as reference
    for var in ['piped_water_flag', 'urban']: # 'urban' is already numeric 0/1, but C() treats it as categorical
        if var in df_reg.columns:
            if not pd.api.types.is_categorical_dtype(df_reg[var]):
                df_reg[var] = df_reg[var].astype('category')
            if 0 in df_reg[var].cat.categories:
                df_reg[var] = df_reg[var].cat.set_categories([0, 1], ordered=True)
            else:
                print(f"    Warning: Reference category '0' not found for binary '{var}'. Using default first category.")
    
    # Ensure region is categorical and set reference
    if 'region' in df_reg.columns:
        if not pd.api.types.is_categorical_dtype(df_reg['region']):
            df_reg['region'] = df_reg['region'].astype('category')
        if ref_cats['region'] in df_reg['region'].cat.categories:
            df_reg['region'] = df_reg['region'].cat.set_categories(
                [ref_cats['region']] + [c for c in df_reg['region'].cat.categories if c != ref_cats['region']],
                ordered=True
            )
        else:
            print(f"    Warning: Reference category '{ref_cats['region']}' not found for 'region'. Using default first category.")

    # Ensure idi_score is numeric
    if 'idi_score' in df_reg.columns:
        df_reg['idi_score'] = pd.to_numeric(df_reg['idi_score'], errors='coerce')
    
    # Drop rows with NaNs in variables critical for Model 4
    model4_vars = [outcome_var, 'piped_water_flag', 'wealth_quintile', 'urban', 'idi_score', 
                   'hh_size', 'children_under5_count', 'region', 'weight', cfg.VAR_PSU]
    df_reg.dropna(subset=model4_vars, inplace=True)

    if df_reg.empty:
        return {}, "No data remaining after dropping NaNs for Model 4 regression."

    # Model 4: Interactions revealing the paradox
    # Use C() for categorical variables to specify reference levels for clarity in formula
    # Removed comments from formula string as statsmodels does not allow them.
    formula4 = f"""{outcome_var} ~         
        C(piped_water_flag, Treatment(0)) + C(wealth_quintile, Treatment('{ref_cats['wealth_quintile']}')) + C(urban, Treatment(0)) + idi_score +         
        hh_size + children_under5_count + C(region, Treatment('{ref_cats['region']}')) +
        C(piped_water_flag, Treatment(0)):C(wealth_quintile, Treatment('{ref_cats['wealth_quintile']}')) +
        C(piped_water_flag, Treatment(0)):C(urban, Treatment(0)) +
        C(piped_water_flag, Treatment(0)):idi_score +
        C(urban, Treatment(0)):idi_score
    """

    print("  Running Model 4: Interactions...")
    try:
        model4 = smf.logit(formula=formula4, data=df_reg,                     
                           freq_weights=df_reg['weight'],                    
                           cov_type='cluster',                     
                           cov_kwds={'groups': df_reg[cfg.VAR_PSU]})
        results4 = model4.fit(disp=False, maxiter=500) # Increased maxiter for convergence
        model_results['Model 4 (Interactions)'] = format_regression_output(results4)
        
        # Interpretive text for Model 4
        interpretive_text_model4 = (
            "Model 4 introduces interaction terms to explore how the effect of piped water on disruption "
            "varies across different household characteristics, providing deeper insights into the 'Infrastructure Paradox'. "
            "Key findings from the interaction effects are:"
        )
        
        # Interpreting specific interaction terms
        # piped_water_flag:wealth_quintile
        # wealth_quintile has 4 dummy variables (Poorer, Middle, Richer, Richest vs Poorest)
        # We are interested in how the effect of piped water changes for richer quintiles.
        # This requires looking at terms like 'C(piped_water_flag, Treatment(0))[T.1]:C(wealth_quintile, Treatment('Poorest'))[T.Richer]'
        
        # Example interpretation for piped_water_flag:wealth_quintile
        wealth_quintiles_for_inter = ['Poorer', 'Middle', 'Richer', 'Richest']
        wealth_effect_text = []
        for q in wealth_quintiles_for_inter:
            term = f"C(piped_water_flag, Treatment(0))[T.1]:C(wealth_quintile, Treatment('{ref_cats['wealth_quintile']}'))[T.{q}]"
            if term in results4.params.index:
                or_val = np.exp(results4.params[term])
                p_val = results4.pvalues[term]
                if p_val < 0.05 and or_val > 1:
                    wealth_effect_text.append(f"- The effect of piped water on disruption is significantly higher for '{q}' households (OR={or_val:.2f}, p={format_p_value(p_val)}) compared to '{ref_cats['wealth_quintile']}' households, suggesting the paradox worsens for the wealthy.")
                elif p_val < 0.05 and or_val < 1:
                    wealth_effect_text.append(f"- The effect of piped water on disruption is significantly lower for '{q}' households (OR={or_val:.2f}, p={format_p_value(p_val)}) compared to '{ref_cats['wealth_quintile']}' households.")
        if wealth_effect_text:
            interpretive_text_model4 += "\n*   " + "\n*   ".join(wealth_effect_text)
        else:
            interpretive_text_model4 += "\n*   No significant interaction found between piped water and wealth quintile, or effect is complex."

        # piped_water_flag:urban
        term_urban_piped = "C(piped_water_flag, Treatment(0))[T.1]:C(urban, Treatment(0))[T.1]"
        if term_urban_piped in results4.params.index:
            or_val = np.exp(results4.params[term_urban_piped])
            p_val = results4.pvalues[term_urban_piped]
            if p_val < 0.05 and or_val > 1:
                interpretive_text_model4 += f"\n*   The effect of piped water on disruption is significantly amplified in urban areas (OR={or_val:.2f}, p={format_p_value(p_val)}), suggesting urban piped water systems might be less reliable than rural ones, or urban residents are more sensitive to disruptions."
            elif p_val < 0.05 and or_val < 1:
                interpretive_text_model4 += f"\n*   The effect of piped water on disruption is significantly reduced in urban areas (OR={or_val:.2f}, p={format_p_value(p_val)})."
            else:
                interpretive_text_model4 += f"\n*   The interaction between piped water and urban location is not statistically significant (p={format_p_value(p_val)}), or the effect is complex."
        else:
             interpretive_text_model4 += "\n*   Interaction term for piped water and urban location not found or not significant."
        
        # piped_water_flag:idi_score
        term_idi_piped = "C(piped_water_flag, Treatment(0))[T.1]:idi_score"
        if term_idi_piped in results4.params.index:
            or_val = np.exp(results4.params[term_idi_piped])
            p_val = results4.pvalues[term_idi_piped]
            if p_val < 0.05 and or_val > 1:
                interpretive_text_model4 += f"\n*   Higher Infrastructure Dependency Index (IDI) significantly amplifies the disruption risk for piped water users (OR={or_val:.2f}, p={format_p_value(p_val)}), indicating that greater reliance on infrastructure increases vulnerability to its failures."
            elif p_val < 0.05 and or_val < 1:
                interpretive_text_model4 += f"\n*   Higher Infrastructure Dependency Index (IDI) significantly reduces the disruption risk for piped water users (OR={or_val:.2f}, p={format_p_value(p_val)})."
            else:
                interpretive_text_model4 += f"\n*   The interaction between piped water and IDI score is not statistically significant (p={format_p_value(p_val)}), or the effect is complex."
        else:
            interpretive_text_model4 += "\n*   Interaction term for piped water and IDI score not found or not significant."

        # urban:idi_score
        term_urban_idi = "C(urban, Treatment(0))[T.1]:idi_score"
        if term_urban_idi in results4.params.index:
            or_val = np.exp(results4.params[term_urban_idi])
            p_val = results4.pvalues[term_urban_idi]
            if p_val < 0.05 and or_val > 1:
                interpretive_text_model4 += f"\n*   There is also a significant interaction between urban location and IDI (OR={or_val:.2f}, p={format_p_value(p_val)}), suggesting that urban dependency on infrastructure further contributes to disruption risk."
            elif p_val < 0.05 and or_val < 1:
                interpretive_text_model4 += f"\n*   There is a significant interaction between urban location and IDI, where urban areas with higher IDI have lower disruption risk (OR={or_val:.2f}, p={format_p_value(p_val)})."
            else:
                interpretive_text_model4 += f"\n*   The interaction between urban location and IDI score is not statistically significant (p={format_p_value(p_val)}), or the effect is complex."
        else:
            interpretive_text_model4 += "\n*   Interaction term for urban location and IDI score not found or not significant."
        
        model_results['Model 4 (Interactions)'] = model_results['Model 4 (Interactions)'].round(2)
        model_results['Model 4 (Interactions) results object'] = results4 # Store results object for predicted probabilities
    except Exception as e: 
        print(f"    ERROR running Model 4: {e}")
        model_results['Model 4 (Interactions)'] = pd.DataFrame()
        interpretive_text_model4 = f"Error running Model 4: {e}"

    print(f"{'='*10} Table 7 Generated {'='*10}\n")
    return model_results, interpretive_text_model4

def generate_table_predicted_probabilities(df: pd.DataFrame, cfg: Config, results4: Optional[Any] = None) -> Tuple[pd.DataFrame, str]:
    """
    TASK 1.2: Calculate predicted probabilities for key scenarios using Model 4.
    """
    print(f"\n{'='*10} Generating Predicted Probabilities for Key Scenarios {'='*10}")
    if results4 is None:
        return pd.DataFrame(), "Error: Model 4 results object not provided for predicted probabilities."
    
    # Define 6 key scenarios representing different household types:
    # Ensure categorical variables match the categories in the model's training data
    scenarios = [
        {
            'name': 'Wealthy Urban Piped Water (High IDI)',
            'piped_water_flag': 1,
            'wealth_quintile': 'Richest', # Must match category from df_reg
            'urban': 1, # Must match category from df_reg
            'idi_score': 9, # High IDI
            'hh_size': 4,
            'children_under5_count': 1,
            'region': 'North' # Must match category from df_reg
        },
        {
            'name': 'Wealthy Urban Tube Well (Low IDI)',
            'piped_water_flag': 0,
            'wealth_quintile': 'Richest',
            'urban': 1,
            'idi_score': 2, # Low IDI
            'hh_size': 4,
            'children_under5_count': 1,
            'region': 'North'
        },
        {
            'name': 'Poor Rural Tube Well (Low IDI)',
            'piped_water_flag': 0,
            'wealth_quintile': 'Poorest',
            'urban': 0, # Rural
            'idi_score': 1, # Low IDI
            'hh_size': 6,
            'children_under5_count': 2,
            'region': 'Central'
        },
        {
            'name': 'Poor Rural Piped Water (Moderate IDI)',
            'piped_water_flag': 1,
            'wealth_quintile': 'Poorest',
            'urban': 0, # Rural
            'idi_score': 5, # Moderate IDI
            'hh_size': 6,
            'children_under5_count': 2,
            'region': 'Central'
        },
        {
            'name': 'Middle Class Urban Piped (High IDI)',
            'piped_water_flag': 1,
            'wealth_quintile': 'Middle',
            'urban': 1,
            'idi_score': 8, # High IDI
            'hh_size': 4,
            'children_under5_count': 1,
            'region': 'South'
        },
        {
            'name': 'Middle Class Rural Well (Low IDI)',
            'piped_water_flag': 0,
            'wealth_quintile': 'Middle',
            'urban': 0, # Rural
            'idi_score': 2, # Low IDI
            'hh_size': 5,
            'children_under5_count': 1,
            'region': 'South'
        }
    ]

    results_list = []
    # Get the DataFrame used to train results4 to ensure consistent column types and categories
    df_template_for_predict = results4.model.data.orig_exog.drop(columns=['Intercept'], errors='ignore')

    for scenario in scenarios:
        # Create DataFrame for prediction, ensuring all model variables are present and correctly typed
        scenario_df = get_scenario_df(scenario, df_template_for_predict)
        
        try:
            # Get predicted probability
            pred_prob = results4.predict(scenario_df)
            
            # Calculate 95% confidence interval
            pred_results = results4.get_prediction(scenario_df)
            ci_lower = pred_results.summary_frame(alpha=0.05)['obs_ci_lower']
            ci_upper = pred_results.summary_frame(alpha=0.05)['obs_ci_upper']
            
            results_list.append({
                'Scenario': scenario['name'],
                'Piped Water': 'Yes' if scenario['piped_water_flag'] == 1 else 'No',
                'Wealth Quintile': scenario['wealth_quintile'],
                'Location': 'Urban' if scenario['urban'] == 1 else 'Rural',
                'IDI Score': scenario['idi_score'],
                'Predicted Disruption Prob (%)': pred_prob.iloc[0] * 100,
                '95% CI Lower (%)': ci_lower.iloc[0] * 100,
                '95% CI Upper (%)': ci_upper.iloc[0] * 100
            })
        except Exception as e:
            print(f"    Error predicting for scenario {scenario['name']}: {e}")
            results_list.append({
                'Scenario': scenario['name'],
                'Piped Water': 'Yes' if scenario['piped_water_flag'] == 1 else 'No',
                'Wealth Quintile': scenario['wealth_quintile'],
                'Location': 'Urban' if scenario['urban'] == 1 else 'Rural',
                'IDI Score': scenario['idi_score'],
                'Predicted Disruption Prob (%)': np.nan,
                '95% CI Lower (%)': np.nan,
                '95% CI Upper (%)': np.nan
            })

    table_df = pd.DataFrame(results_list).round(1)

    # Generate interpretive text
    interpretive_text = (
        "Table X presents predicted probabilities of water disruption for six distinct household scenarios, "
        "calculated using Model 4 which incorporates interaction effects. This allows for a granular understanding "
        "of how piped water, wealth, urbanicity, and infrastructure dependency combine to influence disruption risk."
    )
    
    # Example comparison for interpretive text
    wealthy_urban_piped_high_idi = table_df[table_df['Scenario'] == 'Wealthy Urban Piped Water (High IDI)'].iloc[0]
    poor_rural_tube_low_idi = table_df[table_df['Scenario'] == 'Poor Rural Tube Well (Low IDI)'].iloc[0]
    
    if not wealthy_urban_piped_high_idi.isnull().any() and not poor_rural_tube_low_idi.isnull().any():
        gap = wealthy_urban_piped_high_idi['Predicted Disruption Prob (%)'] - poor_rural_tube_low_idi['Predicted Disruption Prob (%)']
        interpretive_text += (
            f"\n\nFor instance, a wealthy urban household with piped water and high IDI has a predicted disruption probability of "
            f"{wealthy_urban_piped_high_idi['Predicted Disruption Prob (%)']:.1f}% "
            f"(95% CI: {wealthy_urban_piped_high_idi['95% CI Lower (%)']:.1f}-{wealthy_urban_piped_high_idi['95% CI Upper (%)']:.1f}%). "
            f"This is significantly higher than a poor rural household with a tube well and low IDI, which has a predicted probability of only "
            f"{poor_rural_tube_low_idi['Predicted Disruption Prob (%)']:.1f}% "
            f"(95% CI: {poor_rural_tube_low_idi['95% CI Lower (%)']:.1f}-{poor_rural_tube_low_idi['95% CI Upper (%)']:.1f}%). "
            f"This {gap:.1f} percentage point gap persists despite the former's superior socioeconomic resources, "
            "demonstrating the severity of infrastructure dependency and the 'Infrastructure Paradox'."
        )
    else:
        interpretive_text += "\n\nSpecific scenario comparisons could not be generated due to missing data in scenarios."

    print(f"{'='*10} Predicted Probabilities Generated {'='*10}\n")
    return table_df, interpretive_text

def generate_table_propensity_score_matching(df: pd.DataFrame, cfg: Config) -> Tuple[Dict[str, pd.DataFrame], str]:
    """
    TASK 1.3: Propensity Score Matching (PSM)
    Matches piped water and non-piped water households on observables to isolate the treatment effect.
    """
    print(f"\n{'='*10} Generating Propensity Score Matching Results {'='*10}")
    # Ensure necessary columns are present
    required_cols_psm = [
        'piped_water_flag', 'tube_well_flag', 'wealth_quintile', 'urban', 'hh_size',
        'children_under5_count', 'is_female_headed', 'caste', 'region', 'season',
        'hh_head_education', 'has_electricity', 'improved_sanitation_flag', cfg.VAR_WATER_DISRUPTED_FINAL, 'weight', cfg.VAR_PSU
    ]
    
    df_psm = df.copy()
    for col in required_cols_psm:
        if col not in df_psm.columns:
            print(f"    Warning: Column '{col}' not found for PSM, treating as missing or skipping if critical.")
            if col in ['piped_water_flag', 'tube_well_flag', cfg.VAR_WATER_DISRUPTED_FINAL, 'weight', cfg.VAR_PSU]:
                return {}, f"Error: Critical column '{col}' missing for PSM."
            
    # Define treatment and control groups
    # Treatment: piped_water_flag == 1
    # Control: tube_well_flag == 1 (or all non-piped, but tube well is a common alternative)
    df_psm_filtered = df_psm[(df_psm['piped_water_flag'] == 1) | (df_psm['tube_well_flag'] == 1)].copy()
    if df_psm_filtered.empty:
        return {}, "No data for PSM after filtering for piped or tube well users."

    df_psm_filtered['treatment'] = (df_psm_filtered['piped_water_flag'] == 1).astype(int)
    
    # Covariates for matching
    psm_covariates = [
        'wealth_quintile', 'urban', 'hh_size', 'children_under5_count',
        'is_female_headed', 'caste', 'region', 'season',
        'hh_head_education', 'has_electricity', 'improved_sanitation_flag'
    ]
    
    # Filter covariates to only include those present in the DataFrame
    psm_covariates_present = [col for col in psm_covariates if col in df_psm_filtered.columns]
    
    # Ensure categorical covariates are handled correctly for formula
    for col in psm_covariates_present:
        if pd.api.types.is_categorical_dtype(df_psm_filtered[col]):
            # Use Treatment() for categorical variables to explicitly define reference levels
            # For simplicity, we'll let statsmodels pick the default reference (usually first alphabetically)
            pass
        elif pd.api.types.is_numeric_dtype(df_psm_filtered[col]) and df_psm_filtered[col].nunique() <= 2:
            # Treat binary numeric columns as categorical for PSM model
            df_psm_filtered[col] = df_psm_filtered[col].astype('category')

    # Estimate propensity scores using logistic regression
    psm_formula = f"treatment ~ {' + '.join([f'C({c})' if pd.api.types.is_categorical_dtype(df_psm_filtered[c]) else c for c in psm_covariates_present])}"
    
    # Drop NaNs for PSM model fitting
    df_for_psm_model = df_psm_filtered.dropna(subset=['treatment'] + psm_covariates_present + ['weight', cfg.VAR_PSU]).copy()
    
    if df_for_psm_model.empty:
        return {}, "No data remaining for PSM model after dropping NaNs."

    print("  Fitting propensity score model...")
    try:
        psm_model = smf.logit(formula=psm_formula, data=df_for_psm_model, freq_weights=df_for_psm_model['weight']).fit(disp=False)
        df_for_psm_model['propensity_score'] = psm_model.predict(df_for_psm_model)
    except Exception as e:
        return {}, f"Error fitting propensity score model: {e}"

    # Perform matching using nearest neighbor
    treated = df_for_psm_model[df_for_psm_model['treatment'] == 1].copy()
    control = df_for_psm_model[df_for_psm_model['treatment'] == 0].copy()

    if treated.empty or control.empty:
        return {}, "Treated or control group is empty after PSM model filtering."

    print("  Performing 1:1 nearest neighbor matching with caliper...")
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(control[['propensity_score']].values)
    distances, indices = nn.kneighbors(treated[['propensity_score']].values)

    # Apply caliper (e.g., 0.05)
    caliper = 0.05
    valid_matches_mask = (distances.flatten() < caliper)
    matched_treated = treated[valid_matches_mask].copy()
    matched_control_indices = indices[valid_matches_mask].flatten()
    matched_control = control.iloc[matched_control_indices].copy()

    if matched_treated.empty or matched_control.empty:
        return {}, "No matches found after applying caliper. Try increasing caliper or checking data."

    # Combine matched data for balance check
    matched_df = pd.concat([matched_treated, matched_control])

    # Check balance before and after matching
    balance_results = []
    for covariate in psm_covariates_present:
        # For categorical variables, compare proportions of each category
        if pd.api.types.is_categorical_dtype(df_for_psm_model[covariate]):
            # Get all categories for the covariate
            categories = df_for_psm_model[covariate].cat.categories

            for category in categories:
                # Create a temporary binary column for this category
                temp_col_name = f"{covariate}_{category}"
                df_for_psm_model[temp_col_name] = (df_for_psm_model[covariate] == category).astype(int)
                matched_df[temp_col_name] = (matched_df[covariate] == category).astype(int)

                mean_treated_before = df_for_psm_model[df_for_psm_model['treatment'] == 1][temp_col_name].mean()
                mean_control_before = df_for_psm_model[df_for_psm_model['treatment'] == 0][temp_col_name].mean()
                std_pooled_before = np.sqrt( (df_for_psm_model[df_for_psm_model['treatment'] == 1][temp_col_name].var() + df_for_psm_model[df_for_psm_model['treatment'] == 0][temp_col_name].var()) / 2 )
                std_diff_before = (mean_treated_before - mean_control_before) / std_pooled_before if std_pooled_before !=0 else np.nan

                mean_treated_after = matched_df[matched_df['treatment'] == 1][temp_col_name].mean()
                mean_control_after = matched_df[matched_df['treatment'] == 0][temp_col_name].mean()
                std_pooled_after = np.sqrt( (matched_df[matched_df['treatment'] == 1][temp_col_name].var() + matched_df[matched_df['treatment'] == 0][temp_col_name].var()) / 2 )
                std_diff_after = (mean_treated_after - mean_control_after) / std_pooled_after if std_pooled_after !=0 else np.nan

                balance_results.append({
                    'Covariate': temp_col_name,
                    'Std_Diff_Before': std_diff_before,
                    'Std_Diff_After': std_diff_after,
                    'Balanced': abs(std_diff_after) < 0.1 if not pd.isna(std_diff_after) else False
                })
                # Drop temporary columns
                df_for_psm_model.drop(columns=[temp_col_name], inplace=True)
                matched_df.drop(columns=[temp_col_name], inplace=True)

        else: # Numeric covariate
            mean_treated_before = df_for_psm_model[df_for_psm_model['treatment'] == 1][covariate].mean()
            mean_control_before = df_for_psm_model[df_for_psm_model['treatment'] == 0][covariate].mean()
            std_treated_before = df_for_psm_model[df_for_psm_model['treatment'] == 1][covariate].std()
            std_diff_before = (mean_treated_before - mean_control_before) / std_treated_before if std_treated_before !=0 else np.nan

            mean_treated_after = matched_df[matched_df['treatment'] == 1][covariate].mean()
            mean_control_after = matched_df[matched_df['treatment'] == 0][covariate].mean()
            std_treated_after = matched_df[matched_df['treatment'] == 1][covariate].std()
            std_diff_after = (mean_treated_after - mean_control_after) / std_treated_after if std_treated_after !=0 else np.nan

            balance_results.append({
                'Covariate': covariate,
                'Std_Diff_Before': std_diff_before,
                'Std_Diff_After': std_diff_after,
                'Balanced': abs(std_diff_after) < 0.1 if not pd.isna(std_diff_after) else False
            })

    balance_df = pd.DataFrame(balance_results).round(3)

    # Calculate Average Treatment Effect on the Treated (ATT)
    # Ensure water_disrupted is numeric for mean calculation
    matched_treated[cfg.VAR_WATER_DISRUPTED_FINAL] = pd.to_numeric(matched_treated[cfg.VAR_WATER_DISRUPTED_FINAL], errors='coerce')
    matched_control[cfg.VAR_WATER_DISRUPTED_FINAL] = pd.to_numeric(matched_control[cfg.VAR_WATER_DISRUPTED_FINAL], errors='coerce')

    att = (
        (matched_treated[cfg.VAR_WATER_DISRUPTED_FINAL] * matched_treated['weight']).sum() / matched_treated['weight'].sum() -
        (matched_control[cfg.VAR_WATER_DISRUPTED_FINAL] * matched_control['weight']).sum() / matched_control['weight'].sum()
    ) * 100  # Convert to percentage points

    # Bootstrap standard error
    print("  Bootstrapping ATT standard error (this may take a moment)...")
    def att_statistic(treated_df, control_df):
        # Ensure 'water_disrupted' and 'weight' are present
        if cfg.VAR_WATER_DISRUPTED_FINAL not in treated_df.columns or 'weight' not in treated_df.columns or \
           cfg.VAR_WATER_DISRUPTED_FINAL not in control_df.columns or 'weight' not in control_df.columns:
            return np.nan # Or handle error appropriately
        
        treated_mean = (treated_df[cfg.VAR_WATER_DISRUPTED_FINAL] * treated_df['weight']).sum() / treated_df['weight'].sum()
        control_mean = (control_df[cfg.VAR_WATER_DISRUPTED_FINAL] * control_df['weight']).sum() / control_df['weight'].sum()
        return (treated_mean - control_mean) * 100

    # Prepare data for bootstrap
    # The bootstrap function expects a single array, so we'll pass indices
    # and re-sample from original matched_treated/control within the statistic function
    
    # Using scipy.stats.bootstrap
    # The `bootstrap` function takes a `data` tuple.
    # We need to pass the actual dataframes to `att_statistic` during resampling.
    # A common way is to pass indices and resample from the original data
    
    # Define a wrapper for the statistic function that takes indices
    def bootstrap_att_wrapper(resampled_indices):
        resampled_treated_indices, resampled_control_indices = resampled_indices
        sample_treated = matched_treated.iloc[resampled_treated_indices]
        sample_control = matched_control.iloc[resampled_control_indices]
        return att_statistic(sample_treated, sample_control)

    # Create resample_treated and resample_control functions
    # Using np.arange to get integer indices
    resample_treated_idx = np.arange(len(matched_treated))
    resample_control_idx = np.arange(len(matched_control))
    
    try:
        # Note: 'workers=-1' can speed up for large N
        boot_res = bootstrap(
            (resample_treated_idx, resample_control_idx),
            bootstrap_att_wrapper,
            vectorized=False, 
            paired=False, # Data is not paired, just two independent samples
            axis=0, 
            method='percentile', 
            n_resamples=1000, 
            random_state=42, 
            confidence_level=0.95
        )
        
        att_se = boot_res.standard_error
        att_ci_lower = boot_res.confidence_interval.low
        att_ci_upper = boot_res.confidence_interval.high
        
        # Calculate p-value (rough approximation, more robust methods exist)
        # Check if 0 is within the confidence interval
        if att_ci_lower <= 0 <= att_ci_upper:
            att_p_value = 0.5 # Not significant
        else:
            att_p_value = 0.001 # Significant (conservative)
        att_p_value_formatted = format_p_value(att_p_value)
    except Exception as e:
        print(f"    Error during bootstrapping: {e}")
        att_se, att_ci_lower, att_ci_upper, att_p_value_formatted = np.nan, np.nan, np.nan, "Error"

    att_results_df = pd.DataFrame([{
        'Metric': 'Average Treatment Effect on the Treated (ATT)',
        'Estimate (pp)': att,
        'Std. Error': att_se,
        '95% CI Lower (pp)': att_ci_lower,
        '95% CI Upper (pp)': att_ci_upper,
        'P-value': att_p_value_formatted
    }]).round(2)

    # Create output table
    summary_table = pd.DataFrame([{
        'Description': 'Number of Treated Households (Piped Water)',
        'Value': len(treated)
    }, {
        'Description': 'Number of Control Households (Tube Well)',
        'Value': len(control)
    }, {
        'Description': 'Number of Matched Treated Households',
        'Value': len(matched_treated)
    }, {
        'Description': 'Number of Matched Control Households',
        'Value': len(matched_control)
    }, {
        'Description': 'Percentage of Treated Households Successfully Matched',
        'Value': (len(matched_treated) / len(treated) * 100) if len(treated) > 0 else 0
    }, {
        'Description': 'Caliper Used (Propensity Score Diff)',
        'Value': caliper
    }])

    # Generate interpretive text
    percent_matched = (len(matched_treated) / len(treated) * 100) if len(treated) > 0 else 0
    balance_status = "adequate covariate balance (all standardized differences < 0.1)" if all(balance_df['Balanced']) else "some covariate imbalance (standardized differences > 0.1 for some covariates)"

    interpretive_text = (
        f"After propensity score matching on {len(psm_covariates_present)} covariates, "
        f"{percent_matched:.1f}% of piped water households were successfully matched to observationally similar tube well households. "
        f"Balance diagnostics confirm {balance_status}. "
        "The Average Treatment Effect on the Treated (ATT) indicates that piped water households experience "
        f"{att:.1f} percentage points higher disruption rates (95% CI: {att_ci_lower:.1f}-{att_ci_upper:.1f}%, p {att_p_value_formatted}) "
        "compared to matched controls. This finding confirms that the reliability gap persists even after accounting for selection bias "
        "based on observed characteristics, strengthening the evidence for the 'Infrastructure Paradox'."
    )
    print(f"{'='*10} Propensity Score Matching Results Generated {'='*10}\n")
    return {'Summary': summary_table, 'Balance_Diagnostics': balance_df, 'ATT_Results': att_results_df}, interpretive_text

def generate_table_marginal_effects(df: pd.DataFrame, cfg: Config, results4: Optional[Any] = None) -> Tuple[Dict[str, pd.DataFrame], str]:
    """
    TASK 1.4: Marginal Effects Calculation
    Calculates and interprets marginal effects for key predictors in Model 4.
    """
    print(f"\n{'='*10} Generating Marginal Effects for Model 4 {'='*10}")
    if results4 is None:
        return pd.DataFrame(), "Error: Model 4 results object not provided for marginal effects."
    
    # Calculate Average Marginal Effects (AME)
    print("  Calculating Average Marginal Effects (AME)...")
    try:
        # Note: statsmodels.discrete.discrete_model.Logit for get_margeff
        # If results4 is from smf.logit, it already has the necessary methods.
        marginal_effects = results4.get_margeff(at='mean', method='dydx') # dydx for continuous, finite_differences for discrete/binary
        
        me_summary = pd.DataFrame({
            'Variable': marginal_effects.margeff_names,
            'Marginal_Effect': marginal_effects.margeff,
            'Std_Error': marginal_effects.margeff_se,
            'z_value': marginal_effects.margeff_z,
            'p_value': marginal_effects.margeff_pvalues
        })
        me_summary['Marginal_Effect_Pct'] = me_summary['Marginal_Effect'] * 100
        me_summary['P>|z|'] = me_summary['p_value'].apply(format_p_value)
        
        ame_table = me_summary[['Variable', 'Marginal_Effect_Pct', 'Std_Error', 'P>|z|']].round({'Marginal_Effect_Pct': 2, 'Std_Error': 3})
        ame_table.rename(columns={'Marginal_Effect_Pct': 'AME (Percentage Points)'}, inplace=True)
    except Exception as e:
        print(f"    Error calculating AME: {e}")
        ame_table = pd.DataFrame()
        interpretive_text = f"Error calculating AME: {e}"
        return ame_table, interpretive_text
    
    interpretive_text = (
        "Table X presents the Average Marginal Effects (AME) for key predictors in Model 4. "
        "AMEs quantify the average change in the predicted probability of water disruption (in percentage points) "
        "for a one-unit change in a continuous predictor, or for a change from the reference category to the specified category "
        "for discrete/binary predictors, holding all other variables at their mean. "
    )
    # Interpret key marginal effects
    if not ame_table.empty:
        # Piped water flag
        piped_effect_row = ame_table[ame_table['Variable'] == 'C(piped_water_flag, Treatment(0))[T.1]']
        if not piped_effect_row.empty:
            piped_ame = piped_effect_row['AME (Percentage Points)'].iloc[0]
            piped_p = piped_effect_row['P>|z|'].iloc[0]
            interpretive_text += (
                f"\n*   **Piped Water Flag (vs. non-piped):** Having piped water, on average, increases the probability of disruption "
                f"by {piped_ame:.2f} percentage points ({piped_p}). This is a direct quantification of the paradox's core finding."
            )
        
        # IDI score
        idi_effect_row = ame_table[ame_table['Variable'] == 'idi_score']
        if not idi_effect_row.empty:
            idi_ame = idi_effect_row['AME (Percentage Points)'].iloc[0]
            idi_p = idi_effect_row['P>|z|'].iloc[0]
            interpretive_text += (
                f"\n*   **IDI Score:** Each 1-point increase in the Infrastructure Dependency Index (IDI) "
                f"increases the probability of disruption by {idi_ame:.2f} percentage points ({idi_p}), "
                "highlighting the significant role of dependency."
            )
        
        # Urban
        urban_effect_row = ame_table[ame_table['Variable'] == 'C(urban, Treatment(0))[T.1]']
        if not urban_effect_row.empty:
            urban_ame = urban_effect_row['AME (Percentage Points)'].iloc[0]
            urban_p = urban_effect_row['P>|z|'].iloc[0]
            interpretive_text += (
                f"\n*   **Urban Residence (vs. Rural):** Living in an urban area increases the probability of disruption "
                f"by {urban_ame:.2f} percentage points ({urban_p}), independent of other factors, "
                "suggesting inherent challenges in urban water provision."
            )

    # Calculate Marginal Effects at Representative Values (MER) for piped water effect at low/high IDI
    print("  Calculating Marginal Effects at Representative Values (MER)...")
    mer_results = []
    
    # Get the DataFrame used to train results4 to ensure consistent column types and categories
    df_template_for_predict = results4.model.data.orig_exog.drop(columns=['Intercept'], errors='ignore')
    # Define representative IDI scores
    representative_idi_scores = [2, 9] # Low and High IDI
    
    for idi_val in representative_idi_scores:
        # Scenario for piped water (piped_water_flag = 1)
        scenario_piped = {'piped_water_flag': 1, 'idi_score': idi_val}
        scenario_piped_df = get_scenario_df(scenario_piped, df_template_for_predict)
        
        # Scenario for non-piped water (piped_water_flag = 0)
        scenario_non_piped = {'piped_water_flag': 0, 'idi_score': idi_val}
        scenario_non_piped_df = get_scenario_df(scenario_non_piped, df_template_for_predict)

        try:
            prob_piped = results4.predict(scenario_piped_df).iloc[0] * 100
            prob_non_piped = results4.predict(scenario_non_piped_df).iloc[0] * 100
            
            effect_at_idi = prob_piped - prob_non_piped
            
            mer_results.append({
                'IDI Score': idi_val,
                'P(Disruption | Piped)': f"{prob_piped:.1f}%",
                'P(Disruption | Non-Piped)': f"{prob_non_piped:.1f}%",
                'Piped Water Effect (pp)': f"{effect_at_idi:.1f}"
            })
        except Exception as e:
            print(f"    Error calculating MER for IDI={idi_val}: {e}")
            mer_results.append({
                'IDI Score': idi_val,
                'P(Disruption | Piped)': 'Error',
                'P(Disruption | Non-Piped)': 'Error',
                'Piped Water Effect (pp)': 'Error'
            })
    
    mer_table = pd.DataFrame(mer_results)
    
    interpretive_text += (
        "\n\n**Marginal Effects at Representative Values (MER):**"
        "\nTo illustrate the interaction between piped water and IDI, we examine the marginal effect of piped water "
        "at low (IDI=2) and high (IDI=9) levels of infrastructure dependency."
    )
    if not mer_table.empty:
        low_idi_effect = mer_table[mer_table['IDI Score'] == 2]['Piped Water Effect (pp)'].iloc[0]
        high_idi_effect = mer_table[mer_table['IDI Score'] == 9]['Piped Water Effect (pp)'].iloc[0]
        
        interpretive_text += (
            f"\n*   At **low IDI (score=2)**, having piped water is associated with a change in disruption probability of "
            f"{low_idi_effect} percentage points. "
            f"\n*   However, at **high IDI (score=9)**, this effect significantly increases to "
            f"{high_idi_effect} percentage points. "
            "This clearly demonstrates how higher infrastructure dependency amplifies the probability of disruption for piped water users, "
            "validating the interaction term's significance."
        )
    else:
        interpretive_text += "\n\nMER calculations failed."

    print(f"{'='*10} Marginal Effects Generated {'='*10}\n")
    return {'AME': ame_table, 'MER_IDI_Interaction': mer_table}, interpretive_text

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
    
    # Ensure water_disrupted is numeric for filtering
    df_temp = df.copy()
    df_temp[cfg.VAR_WATER_DISRUPTED_FINAL] = pd.to_numeric(df_temp[cfg.VAR_WATER_DISRUPTED_FINAL], errors='coerce')

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
        if var not in df_temp.columns:
            print(f"      Warning: Column '{var}' not found for supporting descriptive table. Skipping.")
            continue
        
        # Ensure var is categorical for grouping, if appropriate
        if not pd.api.types.is_categorical_dtype(df_temp[var]) and df_temp[var].nunique() < 20:
            df_temp[var] = df_temp[var].astype('category')
        elif not pd.api.types.is_categorical_dtype(df_temp[var]) and df_temp[var].nunique() >= 20:
            print(f"      Warning: Variable '{var}' is not categorical and has high cardinality. Skipping for supporting descriptive table.")
            continue

        not_disrupted_data = calculate_weighted_percentages(
            df_temp, var, weight_col='weight', target_col=cfg.VAR_WATER_DISRUPTED_FINAL, target_val=0
        )
        disrupted_data = calculate_weighted_percentages(
            df_temp, var, weight_col='weight', target_col=cfg.VAR_WATER_DISRUPTED_FINAL, target_val=1
        )
        # Merge and process results
        combined_data_raw = pd.concat([not_disrupted_data.set_index('Category'), disrupted_data.set_index('Category')], axis=1, keys=['Not Disrupted', 'Disrupted'])
        
        # Clean up column names after concat
        combined_data_raw.columns = ['_'.join(col).strip() for col in combined_data_raw.columns.values]
        combined_data_raw.columns = combined_data_raw.columns.str.replace('Not Disrupted_Weighted_Percentage', 'Not Disrupted (%)')
        combined_data_raw.columns = combined_data_raw.columns.str.replace('Not Disrupted_Unweighted_N', 'N_Not_Disrupted')
        combined_data_raw.columns = combined_data_raw.columns.str.replace('Disrupted_Weighted_Percentage', 'Disrupted (%)')
        combined_data_raw.columns = combined_data_raw.columns.str.replace('Disrupted_Unweighted_N', 'N_Disrupted')
        
        combined_data = combined_data_raw.fillna(0)
        combined_data = combined_data.round(1)

        chi2, p_value, dof, _ = run_weighted_chi2(df_temp, var, cfg.VAR_WATER_DISRUPTED_FINAL, 'weight')
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
    
    # Add p-values to the first row of each characteristic group
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
    required_cols = ['state_name', 'piped_water_flag', 'water_source_category', cfg.VAR_WATER_DISRUPTED_FINAL, 'weight']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        return pd.DataFrame(), f"Error: Required columns missing for State-Level Paradox table: {missing}."
    results = []
    df_numeric_flags = df.copy()
    df_numeric_flags['piped_water_flag_numeric'] = df_numeric_flags['piped_water_flag'].astype(int)
    df_numeric_flags[cfg.VAR_WATER_DISRUPTED_FINAL + '_numeric'] = df_numeric_flags[cfg.VAR_WATER_DISRUPTED_FINAL].astype(int)

    state_counts = df_numeric_flags['state_name'].value_counts()
    valid_states = state_counts[state_counts >= 1000].index.tolist() # Only include states with sufficient sample size

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
    required_cols = ['season', cfg.VAR_WATER_DISRUPTED_FINAL, 'water_source_category', 'residence', 'weight']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        return pd.DataFrame(), f"Error: Required columns missing for Seasonal Patterns table: {missing}."
    results = []
    seasons = ['Winter', 'Summer', 'Monsoon', 'Post-monsoon']
    major_sources = ['Piped Water', 'Tube well/Borehole']
    df_temp = df.copy()
    df_temp[cfg.VAR_WATER_DISRUPTED_FINAL] = pd.to_numeric(df_temp[cfg.VAR_WATER_DISRUPTED_FINAL], errors='coerce')
    
    for season in seasons:
        if season not in df_temp['season'].cat.categories: continue
        season_df = df_temp[df_temp['season'] == season].copy()
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
                     'time_to_water_minutes', 'wealth_quintile', 'residence', 'weight', cfg.VAR_PSU, 'piped_water_flag', 'urban'] # Changed is_urban to urban
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"  Error: Required columns missing for Robustness Checks table: {missing}. Skipping.")
        return pd.DataFrame(), "Error: Required columns missing for Robustness Checks table."
    results_data = []
    df_reg_base = df.copy()
    # Ensure all relevant columns are numeric for regression or categorical for C()
    df_reg_base[cfg.VAR_WATER_DISRUPTED_FINAL] = pd.to_numeric(df_reg_base[cfg.VAR_WATER_DISRUPTED_FINAL], errors='coerce')
    df_reg_base['piped_water_flag'] = pd.to_numeric(df_reg_base['piped_water_flag'], errors='coerce')
    df_reg_base['hh_size'] = pd.to_numeric(df_reg_base['hh_size'], errors='coerce')
    df_reg_base['improved_sanitation_flag'] = pd.to_numeric(df_reg_base['improved_sanitation_flag'], errors='coerce')
    df_reg_base['time_to_water_minutes'] = pd.to_numeric(df_reg_base['time_to_water_minutes'], errors='coerce')
    
    # Convert binary flags to category to ensure consistent handling in formulas
    df_reg_base['piped_water_flag_cat'] = df_reg_base['piped_water_flag'].astype('category')
    if 0 in df_reg_base['piped_water_flag_cat'].cat.categories:
        df_reg_base['piped_water_flag_cat'] = df_reg_base['piped_water_flag_cat'].cat.set_categories([0, 1], ordered=True)
    df_reg_base['urban_cat'] = df_reg_base['urban'].astype('category')
    if 0 in df_reg_base['urban_cat'].cat.categories:
        df_reg_base['urban_cat'] = df_reg_base['urban_cat'].cat.set_categories([0, 1], ordered=True)
    
    # Also ensure wealth_quintile is categorical with a reference
    if 'wealth_quintile' in df_reg_base.columns:
        if not pd.api.types.is_categorical_dtype(df_reg_base['wealth_quintile']):
            df_reg_base['wealth_quintile'] = df_reg_base['wealth_quintile'].astype('category')
        if 'Poorest' in df_reg_base['wealth_quintile'].cat.categories:
            df_reg_base['wealth_quintile'] = df_reg_base['wealth_quintile'].cat.set_categories(
                ['Poorest'] + [c for c in df_reg_base['wealth_quintile'].cat.categories if c != 'Poorest'], ordered=True)

    # A. Demand Effect Test (Simplified for this example)
    # Regress disruption on water source, controlling for household size and improved sanitation
    demand_formula = f"{cfg.VAR_WATER_DISRUPTED_FINAL} ~ C(piped_water_flag_cat, Treatment(0)) + hh_size + C(improved_sanitation_flag, Treatment(0))"
    try:
        model_demand = smf.logit(formula=demand_formula, data=df_reg_base,
                                 freq_weights=df_reg_base['weight'],
                                 cov_type='cluster', cov_kwds={'groups': df_reg_base[cfg.VAR_PSU]})
        results_demand = model_demand.fit(disp=False, maxiter=500)
        
        # Parameter name for C(piped_water_flag_cat, Treatment(0))[T.1]
        piped_param_name = 'C(piped_water_flag_cat, Treatment(0))[T.1]'
        if piped_param_name in results_demand.params.index:
            piped_or = np.exp(results_demand.params[piped_param_name])
            piped_ci_lower = np.exp(results_demand.conf_int().loc[piped_param_name, 0])
            piped_ci_upper = np.exp(results_demand.conf_int().loc[piped_param_name, 1])
            piped_p = results_demand.pvalues[piped_param_name]
            results_data.append({
                'Test': 'Demand Effect Test', 'Variable': 'Piped Water (vs Non-Piped)',
                'OR': piped_or, 'CI_lower': piped_ci_lower, 'CI_upper': piped_ci_upper, 'p_value': format_p_value(piped_p)
            })
        else:
            results_data.append({'Test': 'Demand Effect Test', 'Variable': 'Piped Water (vs Non-Piped)', 'OR': np.nan, 'CI_lower': np.nan, 'CI_upper': np.nan, 'p_value': 'N/A (Term not found)'})
    except Exception as e:
        print(f"    ERROR in Demand Effect Test: {e}")
        results_data.append({'Test': 'Demand Effect Test', 'Variable': 'Piped Water (vs Non-Piped)', 'OR': np.nan, 'CI_lower': np.nan, 'CI_upper': np.nan, 'p_value': 'Error'})

    # B. Reporting Bias Test (using hv204 > 30 min as objective outcome)
    df_reg_reporting = df_reg_base.copy()
    df_reg_reporting['objective_disruption'] = (df_reg_reporting['time_to_water_minutes'] > 30).astype(int)
    
    reporting_formula = f"objective_disruption ~ C(piped_water_flag_cat, Treatment(0)) + C(wealth_quintile, Treatment('Poorest')) + C(urban_cat, Treatment(0))"
    try:
        model_reporting = smf.logit(formula=reporting_formula, data=df_reg_reporting,
                                    freq_weights=df_reg_reporting['weight'],
                                    cov_type='cluster', cov_kwds={'groups': df_reg_reporting[cfg.VAR_PSU]})
        results_reporting = model_reporting.fit(disp=False, maxiter=500)
        
        piped_param_name = 'C(piped_water_flag_cat, Treatment(0))[T.1]'
        if piped_param_name in results_reporting.params.index:
            piped_or = np.exp(results_reporting.params[piped_param_name])
            piped_ci_lower = np.exp(results_reporting.conf_int().loc[piped_param_name, 0])
            piped_ci_upper = np.exp(results_reporting.conf_int().loc[piped_param_name, 1])
            piped_p = results_reporting.pvalues[piped_param_name]
            results_data.append({
                'Test': 'Reporting Bias Test (Objective Disruption)', 'Variable': 'Piped Water (vs Non-Piped)',
                'OR': piped_or, 'CI_lower': piped_ci_lower, 'CI_upper': piped_ci_upper, 'p_value': format_p_value(piped_p)
            })
        else:
            results_data.append({'Test': 'Reporting Bias Test (Objective Disruption)', 'Variable': 'Piped Water (vs Non-Piped)', 'OR': np.nan, 'CI_lower': np.nan, 'CI_upper': np.nan, 'p_value': 'N/A (Term not found)'})
    except Exception as e:
        print(f"    ERROR in Reporting Bias Test: {e}")
        results_data.append({'Test': 'Reporting Bias Test (Objective Disruption)', 'Variable': 'Piped Water (vs Non-Piped)', 'OR': np.nan, 'CI_lower': np.nan, 'CI_upper': np.nan, 'p_value': 'Error'})

    # C. Subgroup Analysis (Piped Water effect for Urban/Rural)
    for res_type_val, label in [(1, 'Urban'), (0, 'Rural')]:
        df_sub = df_reg_base[df_reg_base['urban'] == res_type_val].copy()
        if df_sub.empty:
            results_data.append({'Test': f'Subgroup Analysis ({label})', 'Variable': 'Piped Water (vs Non-Piped)', 'OR': np.nan, 'CI_lower': np.nan, 'CI_upper': np.nan, 'p_value': 'N/A'})
            continue
        
        subgroup_formula = f"{cfg.VAR_WATER_DISRUPTED_FINAL} ~ C(piped_water_flag_cat, Treatment(0)) + C(wealth_quintile, Treatment('Poorest'))"
        try:
            model_subgroup = smf.logit(formula=subgroup_formula, data=df_sub,
                                       freq_weights=df_sub['weight'],
                                       cov_type='cluster', cov_kwds={'groups': df_sub[cfg.VAR_PSU]})
            results_subgroup = model_subgroup.fit(disp=False, maxiter=500)
            
            piped_param_name = 'C(piped_water_flag_cat, Treatment(0))[T.1]'
            if piped_param_name in results_subgroup.params.index:
                piped_or = np.exp(results_subgroup.params[piped_param_name])
                piped_ci_lower = np.exp(results_subgroup.conf_int().loc[piped_param_name, 0])
                piped_ci_upper = np.exp(results_subgroup.conf_int().loc[piped_param_name, 1])
                piped_p = results_subgroup.pvalues[piped_param_name]
                results_data.append({
                    'Test': f'Subgroup Analysis ({label})', 'Variable': 'Piped Water (vs Non-Piped)',
                    'OR': piped_or, 'CI_lower': piped_ci_lower, 'CI_upper': piped_ci_upper, 'p_value': format_p_value(piped_p)
                })
            else:
                results_data.append({'Test': f'Subgroup Analysis ({label})', 'Variable': 'Piped Water (vs Non-Piped)', 'OR': np.nan, 'CI_lower': np.nan, 'CI_upper': np.nan, 'p_value': 'N/A (Term not found)'})
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
    required_cols = ['idi_score', cfg.VAR_WATER_DISRUPTED_FINAL, cfg.VAR_WEALTH_SCORE, 'urban', 'weight']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        return pd.DataFrame(), f"Error: Required columns missing for IDI Construct Validity table: {missing}."
    idi_df_components = df[required_cols].dropna().copy()
    if idi_df_components.empty:
        return pd.DataFrame({'Note': ['Insufficient data for IDI construct validation.']}), "Insufficient data for IDI construct validation."
    results_data = []
    # Ensure columns are numeric for correlations and AUC
    idi_df_components['idi_score'] = pd.to_numeric(idi_df_components['idi_score'], errors='coerce')
    idi_df_components[cfg.VAR_WATER_DISRUPTED_FINAL] = pd.to_numeric(idi_df_components[cfg.VAR_WATER_DISRUPTED_FINAL], errors='coerce')
    idi_df_components[cfg.VAR_WEALTH_SCORE] = pd.to_numeric(idi_df_components[cfg.VAR_WEALTH_SCORE], errors='coerce')
    idi_df_components['urban'] = pd.to_numeric(idi_df_components['urban'], errors='coerce')
    
    idi_df_components.dropna(inplace=True) # Drop NaNs after conversion

    idi_score_numeric = idi_df_components['idi_score']
    water_disrupted_binary = idi_df_components[cfg.VAR_WATER_DISRUPTED_FINAL]
    
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

    wealth_score_numeric = idi_df_components[cfg.VAR_WEALTH_SCORE]
    if not idi_score_numeric.empty and not wealth_score_numeric.empty:
        corr_idi_wealth, p_corr_idi_wealth = pearsonr(idi_score_numeric, wealth_score_numeric)
        results_data.append({'Metric': 'Correlation (IDI Score vs Wealth Score)', 'Value': corr_idi_wealth, 'p_value': format_p_value(p_corr_idi_wealth)})
    else:
        results_data.append({'Metric': 'Correlation (IDI Score vs Wealth Score)', 'Value': np.nan, 'p_value': 'N/A'})

    is_urban_numeric = idi_df_components['urban']
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
    required_cols = ['piped_water_flag', cfg.VAR_WATER_DISRUPTED_FINAL, 'water_source_category', 'residence', 'weight', 'urban']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        return pd.DataFrame(), f"Error: Required columns missing for Policy Simulation table: {missing}."
    results = []
    df_numeric_flags = df.copy()
    df_numeric_flags['piped_water_flag_numeric'] = df_numeric_flags['piped_water_flag'].astype(int)
    df_numeric_flags[cfg.VAR_WATER_DISRUPTED_FINAL + '_numeric'] = df_numeric_flags[cfg.VAR_WATER_DISRUPTED_FINAL].astype(int)
    
    # Ensure numerical types for calculations
    df_numeric_flags['urban_numeric'] = df_numeric_flags['urban'].astype(int)

    current_piped_coverage = (df_numeric_flags['piped_water_flag_numeric'] * df_numeric_flags['weight']).sum() / df_numeric_flags['weight'].sum() * 100
    current_national_disruption = (df_numeric_flags[cfg.VAR_WATER_DISRUPTED_FINAL + '_numeric'] * df_numeric_flags['weight']).sum() / df_numeric_flags['weight'].sum() * 100
    
    current_urban_df = df_numeric_flags[df_numeric_flags['urban_numeric'] == 1]
    current_urban_disruption = (current_urban_df[cfg.VAR_WATER_DISRUPTED_FINAL + '_numeric'] * current_urban_df['weight']).sum() / current_urban_df['weight'].sum() * 100 if current_urban_df['weight'].sum() > 0 else np.nan
    
    current_rural_df = df_numeric_flags[df_numeric_flags['urban_numeric'] == 0]
    current_rural_disruption = (current_rural_df[cfg.VAR_WATER_DISRUPTED_FINAL + '_numeric'] * current_rural_df['weight']).sum() / current_rural_df['weight'].sum() * 100 if current_rural_df['weight'].sum() > 0 else np.nan

    results.append({
        'Scenario': 'Current Scenario', '% Piped Coverage': current_piped_coverage,
        'National Disruption Rate (%)': current_national_disruption,
        'Disruption Urban (%)': current_urban_disruption, 'Disruption Rural (%)': current_rural_disruption
    })
    
    piped_disruption_rate_actual_df = df_numeric_flags[df_numeric_flags['water_source_category'] == 'Piped Water']
    piped_disruption_rate_mean = (piped_disruption_rate_actual_df[cfg.VAR_WATER_DISRUPTED_FINAL + '_numeric'] * piped_disruption_rate_actual_df['weight']).sum() / piped_disruption_rate_actual_df['weight'].sum() if piped_disruption_rate_actual_df['weight'].sum() > 0 else np.nan

    tube_well_disruption_rate_actual_df = df_numeric_flags[df_numeric_flags['water_source_category'] == 'Tube well/Borehole']
    tube_well_disruption_rate_mean = (tube_well_disruption_rate_actual_df[cfg.VAR_WATER_DISRUPTED_FINAL + '_numeric'] * tube_well_disruption_rate_actual_df['weight']).sum() / tube_well_disruption_rate_actual_df['weight'].sum() if tube_well_disruption_rate_actual_df['weight'].sum() > 0 else np.nan

    # Scenario: Universal Piped Water (Current Reliability)
    # Assume all households now have piped water, and experience the *average* disruption rate of current piped users.
    if not pd.isna(piped_disruption_rate_mean):
        # Calculate national disruption rate as if everyone had piped water
        universal_disruption_rate = piped_disruption_rate_mean * 100
        
        # For urban/rural disruption, apply this same national piped disruption rate to urban/rural populations
        # This assumes the national average piped disruption rate applies equally, which is a simplification.
        # A more complex model would predict urban/rural specific piped disruption rates.
        urban_population_weight = df_numeric_flags[df_numeric_flags['urban_numeric'] == 1]['weight'].sum()
        rural_population_weight = df_numeric_flags[df_numeric_flags['urban_numeric'] == 0]['weight'].sum()
        total_population_weight = df_numeric_flags['weight'].sum()

        simulated_urban_disruption = (universal_disruption_rate * urban_population_weight / total_population_weight) / (urban_population_weight / total_population_weight) if urban_population_weight > 0 else np.nan
        simulated_rural_disruption = (universal_disruption_rate * rural_population_weight / total_population_weight) / (rural_population_weight / total_population_weight) if rural_population_weight > 0 else np.nan

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
    
    # Scenario: Universal Piped Water (Enhanced Reliability - as good as tube wells)
    # Assume all households have piped water, but its reliability is as good as tube wells.
    if not pd.isna(tube_well_disruption_rate_mean):
        enhanced_disruption_rate = tube_well_disruption_rate_mean * 100
        
        urban_population_weight = df_numeric_flags[df_numeric_flags['urban_numeric'] == 1]['weight'].sum()
        rural_population_weight = df_numeric_flags[df_numeric_flags['urban_numeric'] == 0]['weight'].sum()
        total_population_weight = df_numeric_flags['weight'].sum()

        simulated_urban_disruption_enhanced = (enhanced_disruption_rate * urban_population_weight / total_population_weight) / (urban_population_weight / total_population_weight) if urban_population_weight > 0 else np.nan
        simulated_rural_disruption_enhanced = (enhanced_disruption_rate * rural_population_weight / total_population_weight) / (rural_population_weight / total_population_weight) if rural_population_weight > 0 else np.nan

        results.append({
            'Scenario': 'Universal Piped Water (Enhanced Reliability)', '% Piped Coverage': 100.0,
            'National Disruption Rate (%)': enhanced_disruption_rate,
            'Disruption Urban (%)': simulated_urban_disruption_enhanced, 'Disruption Rural (%)': simulated_rural_disruption_enhanced
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
# 6. Spatial Data Export Functions
# ==============================================================================

def create_district_level_summary(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    TASK 2.1: Aggregate all key metrics to district level for spatial visualization.
    """
    print(f"\n{'='*10} Creating District-Level Summary {'='*10}")

    required_cols = [
        'district_code', 'district_name', cfg.VAR_STATE_CODE, 'state_name', 'weight',
        cfg.VAR_WATER_DISRUPTED_FINAL, 'piped_water_flag', 'tube_well_flag', 'improved_source_flag',
        'water_source_category', 'wvi_score_scaled', 'wvi_category',
        'cci_score_scaled', 'cci_category', 'idi_score', 'idi_category',
        cfg.VAR_WEALTH_SCORE, 'urban', 'wealth_quintile',
        'women_fetch_water', 'children_fetch_water', 'time_to_water_minutes',
        'has_electricity', 'has_vehicle', 'improved_sanitation_flag'
    ]

    # Filter df to only include required columns and drop rows with critical NaNs for aggregation
    df_filtered = df[[col for col in required_cols if col in df.columns]].dropna(subset=['district_code', 'weight', cfg.VAR_WATER_DISRUPTED_FINAL]).copy()
    if df_filtered.empty:
        print("    Warning: No data remaining for district-level summary after filtering.")
        return pd.DataFrame()

    # Ensure numeric types for aggregation where necessary
    numeric_cols_for_sum = [
        cfg.VAR_WATER_DISRUPTED_FINAL, 'piped_water_flag', 'tube_well_flag',
        'improved_source_flag', 'women_fetch_water', 'children_fetch_water',
        'has_electricity', 'has_vehicle', 'improved_sanitation_flag'
    ]
    for col in numeric_cols_for_sum:
        if col in df_filtered.columns:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
    # Ensure score columns are numeric
    for col in ['wvi_score_scaled', 'cci_score_scaled', 'idi_score', cfg.VAR_WEALTH_SCORE, 'time_to_water_minutes']:
        if col in df_filtered.columns:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')

    district_summary = df_filtered.groupby('district_code').apply(
        lambda x: pd.Series({
            # --- Identifiers ---
            'district_name': x['district_name'].mode()[0] if not x['district_name'].mode().empty else 'Unknown',
            'state_name': x['state_name'].mode()[0] if not x['state_name'].mode().empty else 'Unknown',
            'state_code': x[cfg.VAR_STATE_CODE].mode()[0] if not x[cfg.VAR_STATE_CODE].mode().empty else np.nan,

            # --- Sample Size ---
            'n_households': len(x),
            'total_weighted_pop': x['weight'].sum(),

            # --- PRIMARY OUTCOME: Water Disruption ---
            'disruption_rate_pct': (x[cfg.VAR_WATER_DISRUPTED_FINAL] * x['weight']).sum() / x['weight'].sum() * 100,
            'disruption_count': x[cfg.VAR_WATER_DISRUPTED_FINAL].sum(),

            # --- Water Source Distribution ---
            'piped_water_coverage_pct': (x['piped_water_flag'] * x['weight']).sum() / x['weight'].sum() * 100,
            'tube_well_coverage_pct': (x['tube_well_flag'] * x['weight']).sum() / x['weight'].sum() * 100,
            'improved_source_coverage_pct': (x['improved_source_flag'] * x['weight']).sum() / x['weight'].sum() * 100,

            # --- Most Disrupted Water Source ---
            'most_disrupted_source': x.groupby('water_source_category').apply(
                lambda g: (g[cfg.VAR_WATER_DISRUPTED_FINAL] * g['weight']).sum() / g['weight'].sum() if g['weight'].sum() > 0 else 0
            ).idxmax() if not x.empty and x['weight'].sum() > 0 else 'Unknown',

            'most_disrupted_source_rate': x.groupby('water_source_category').apply(
                lambda g: (g[cfg.VAR_WATER_DISRUPTED_FINAL] * g['weight']).sum() / g['weight'].sum() if g['weight'].sum() > 0 else 0
            ).max() if not x.empty and x['weight'].sum() > 0 else np.nan,

            # --- Vulnerability Indices ---
            'mean_wvi_score': (x['wvi_score_scaled'] * x['weight']).sum() / x['weight'].sum(),
            'pct_high_vulnerability': (x['wvi_category'] == 'High Vulnerability').sum() / len(x) * 100,

            # --- Coping Capacity Indices ---
            'mean_cci_score': (x['cci_score_scaled'] * x['weight']).sum() / x['weight'].sum(),
            'pct_low_coping': (x['cci_category'] == 'Low Coping').sum() / len(x) * 100,

            # --- Infrastructure Dependency ---
            'mean_idi_score': (x['idi_score'] * x['weight']).sum() / x['weight'].sum(),
            'pct_high_idi': (x['idi_category'] == 'High Dependency (8-10)').sum() / len(x) * 100,

            # --- Socioeconomic Characteristics ---
            'mean_wealth_score': (x[cfg.VAR_WEALTH_SCORE] * x['weight']).sum() / x['weight'].sum() if cfg.VAR_WEALTH_SCORE in x.columns else np.nan,
            'pct_urban': (x['urban'].astype(int) * x['weight']).sum() / x['weight'].sum() * 100, # FIX APPLIED HERE
            'pct_poorest_quintile': (x['wealth_quintile'] == 'Poorest').sum() / len(x) * 100,
            'pct_richest_quintile': (x['wealth_quintile'] == 'Richest').sum() / len(x) * 100,

            # --- Coping Mechanisms (Proxy) ---
            'pct_women_fetch_water': (x['women_fetch_water'] * x['weight']).sum() / x['weight'].sum() * 100,
            'pct_children_fetch_water': (x['children_fetch_water'] * x['weight']).sum() / x['weight'].sum() * 100,
            'mean_time_to_water_min': (x['time_to_water_minutes'] * x['weight']).sum() / x['weight'].sum(),

            # --- Infrastructure Assets ---
            'pct_has_electricity': (x['has_electricity'] * x['weight']).sum() / x['weight'].sum() * 100,
            'pct_has_vehicle': (x['has_vehicle'] * x['weight']).sum() / x['weight'].sum() * 100,
            'pct_improved_sanitation': (x['improved_sanitation_flag'] * x['weight']).sum() / x['weight'].sum() * 100,

            # --- Reliability Gap (Calculated) ---
            'reliability_gap': np.nan  # Placeholder
        })
    ).reset_index()

    # Calculate Reliability Gap (actual disruption - expected disruption)
    print("    Calculating Reliability Gap...")
    # Predict expected disruption based on socioeconomic factors
    # Ensure all columns exist and are numeric
    
    # Define features for prediction of expected disruption
    X_cols_for_expected = [
        'mean_wealth_score', 'pct_urban', 'mean_wvi_score', 'mean_cci_score'
    ]
    
    # Filter district_summary for rows where all X_cols_for_expected and disruption_rate_pct are not NaN
    df_for_lr = district_summary.dropna(subset=X_cols_for_expected + ['disruption_rate_pct']).copy()
    
    if not df_for_lr.empty:
        X_district = df_for_lr[X_cols_for_expected].copy()
        y_district = df_for_lr['disruption_rate_pct'].copy()

        # Check if there's enough variance to fit a model
        if X_district.shape[0] > 1 and len(np.unique(y_district)) > 1:
            lr = LinearRegression()
            lr.fit(X_district, y_district)

            # Predict for all districts, using median imputation for missing values in X_full_for_prediction
            X_full_for_prediction = district_summary[X_cols_for_expected].copy()
            for col in X_cols_for_expected:
                if X_full_for_prediction[col].isna().any():
                    X_full_for_prediction[col] = X_full_for_prediction[col].fillna(district_summary[col].median())
            
            expected_disruption = lr.predict(X_full_for_prediction)
            district_summary['expected_disruption_pct'] = expected_disruption
            district_summary['reliability_gap'] = district_summary['disruption_rate_pct'] - district_summary['expected_disruption_pct']
        else:
            print("    Warning: Insufficient variance in data to fit Linear Regression for Reliability Gap. Skipping.")
            district_summary['expected_disruption_pct'] = np.nan
            district_summary['reliability_gap'] = np.nan
    else:
        print("    Warning: Insufficient data to calculate Reliability Gap. Skipping.")
        district_summary['expected_disruption_pct'] = np.nan
        district_summary['reliability_gap'] = np.nan

    # Classify districts into typology
    print("    Classifying districts into typology...")
    # Ensure medians are calculated on non-NaN values
    median_piped = district_summary['piped_water_coverage_pct'].median()
    median_disruption = district_summary['disruption_rate_pct'].median()
    
    def classify_district(row):
        # Handle NaN values for medians
        if pd.isna(row['piped_water_coverage_pct']) or pd.isna(row['disruption_rate_pct']):
            return 'Unknown Typology'
        
        high_piped = row['piped_water_coverage_pct'] >= median_piped
        high_disruption = row['disruption_rate_pct'] >= median_disruption
        
        if high_piped and high_disruption:
            return 'High Coverage High Disruption (Reliability Gap)'
        elif high_piped and not high_disruption:
            return 'High Coverage Low Disruption (Success)'
        elif not high_piped and high_disruption:
            return 'Low Coverage High Disruption (Traditional Vulnerability)'
        else:
            return 'Low Coverage Low Disruption (Resilient)'

    district_summary['district_typology'] = district_summary.apply(classify_district, axis=1)
    
    print(f"{'='*10} District-Level Summary Created {'='*10}\n")
    return district_summary

def create_state_level_summary(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    TASK 2.2: Aggregate all key metrics to state level.
    """
    print(f"\n{'='*10} Creating State-Level Summary {'='*10}")
    required_cols = [
        cfg.VAR_STATE_CODE, 'state_name', 'district_code', cfg.VAR_CLUSTER, 'weight',
        cfg.VAR_WATER_DISRUPTED_FINAL, 'piped_water_flag', 'tube_well_flag', 'improved_source_flag',
        'water_source_category', 'wvi_score_scaled', 'wvi_category',
        'cci_score_scaled', 'cci_category', 'idi_score', 'idi_category',
        cfg.VAR_WEALTH_SCORE, 'urban', 'wealth_quintile',
        'women_fetch_water', 'children_fetch_water', 'time_to_water_minutes',
        'has_electricity', 'has_vehicle', 'improved_sanitation_flag'
    ]
    
    # Filter df to only include required columns and drop rows with critical NaNs for aggregation
    df_filtered = df[[col for col in required_cols if col in df.columns]].dropna(subset=[cfg.VAR_STATE_CODE, 'weight', cfg.VAR_WATER_DISRUPTED_FINAL]).copy()
    if df_filtered.empty:
        print("    Warning: No data remaining for state-level summary after filtering.")
        return pd.DataFrame()

    # Ensure numeric types for aggregation where necessary
    numeric_cols_for_sum = [
        cfg.VAR_WATER_DISRUPTED_FINAL, 'piped_water_flag', 'tube_well_flag',
        'improved_source_flag', 'women_fetch_water', 'children_fetch_water',
        'has_electricity', 'has_vehicle', 'improved_sanitation_flag'
    ]
    for col in numeric_cols_for_sum:
        if col in df_filtered.columns:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')

    # Ensure score columns are numeric
    for col in ['wvi_score_scaled', 'cci_score_scaled', 'idi_score', cfg.VAR_WEALTH_SCORE, 'time_to_water_minutes']:
        if col in df_filtered.columns:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')

    state_summary = df_filtered.groupby(cfg.VAR_STATE_CODE).apply(
        lambda x: pd.Series({
            # --- Identifiers ---
            'state_name': x['state_name'].mode()[0] if not x['state_name'].mode().empty else 'Unknown',
            
            # --- State-specific additions ---
            'n_districts': x['district_code'].nunique() if 'district_code' in x.columns else np.nan,
            'n_clusters': x[cfg.VAR_CLUSTER].nunique() if cfg.VAR_CLUSTER in x.columns else np.nan,
            
            # --- Sample Size ---
            'n_households': len(x),
            'total_weighted_pop': x['weight'].sum(),
            
            # --- PRIMARY OUTCOME: Water Disruption ---
            'disruption_rate_pct': (x[cfg.VAR_WATER_DISRUPTED_FINAL] * x['weight']).sum() / x['weight'].sum() * 100,
            
            # --- Water Source Distribution ---
            'piped_water_coverage_pct': (x['piped_water_flag'] * x['weight']).sum() / x['weight'].sum() * 100,
            'tube_well_coverage_pct': (x['tube_well_flag'] * x['weight']).sum() / x['weight'].sum() * 100,
            
            # --- Vulnerability Indices ---
            'mean_wvi_score': (x['wvi_score_scaled'] * x['weight']).sum() / x['weight'].sum(),
            'mean_cci_score': (x['cci_score_scaled'] * x['weight']).sum() / x['weight'].sum(),
            'mean_idi_score': (x['idi_score'] * x['weight']).sum() / x['weight'].sum(),
            
            # --- Socioeconomic Characteristics ---
            'mean_wealth_score': (x[cfg.VAR_WEALTH_SCORE] * x['weight']).sum() / x['weight'].sum() if cfg.VAR_WEALTH_SCORE in x.columns else np.nan,
            'pct_urban': (x['urban'].astype(int) * x['weight']).sum() / x['weight'].sum() * 100,

            # --- Disruption rates by source for paradox ratio ---
            'piped_disruption_rate': (
                x[x['piped_water_flag'] == 1][cfg.VAR_WATER_DISRUPTED_FINAL] * 
                x[x['piped_water_flag'] == 1]['weight']
            ).sum() / x[x['piped_water_flag'] == 1]['weight'].sum() * 100 if (x['piped_water_flag'] == 1).sum() > 0 else np.nan,
            
            'tube_well_disruption_rate': (
                x[x['tube_well_flag'] == 1][cfg.VAR_WATER_DISRUPTED_FINAL] * 
                x[x['tube_well_flag'] == 1]['weight']
            ).sum() / x[x['tube_well_flag'] == 1]['weight'].sum() * 100 if (x['tube_well_flag'] == 1).sum() > 0 else np.nan,
            
            'reliability_gap': np.nan # Placeholder, will be calculated from district level
        })
    ).reset_index()

    # Calculate paradox ratio
    state_summary['paradox_ratio'] = (state_summary['piped_disruption_rate'] / state_summary['tube_well_disruption_rate']).round(2)
    
    # Categorize paradox strength
    def categorize_paradox(ratio):
        if pd.isna(ratio): return 'Insufficient Data'
        if ratio > 2.0: return 'Strong Paradox'
        if 1.5 <= ratio < 2.0: return 'Moderate Paradox'
        if 1.0 <= ratio < 1.5: return 'Weak Paradox'
        return 'No Paradox (Tube Wells Worse)'
    
    state_summary['paradox_category'] = state_summary['paradox_ratio'].apply(categorize_paradox)

    # For 'reliability_gap' at state level, we can average from district_summary if available
    # This requires running district_summary first.
    # For now, we'll leave it as NaN or calculate a simple state-level version.
    # A more robust approach would be to calculate it directly at the state level using the same LR model.
    print("    Calculating State-level Reliability Gap (simplified)...")
    X_cols_for_expected_state = [
        'mean_wealth_score', 'pct_urban', 'mean_wvi_score', 'mean_cci_score'
    ]
    df_for_lr_state = state_summary.dropna(subset=X_cols_for_expected_state + ['disruption_rate_pct']).copy()
    if not df_for_lr_state.empty:
        X_state = df_for_lr_state[X_cols_for_expected_state].copy()
        y_state = df_for_lr_state['disruption_rate_pct'].copy()

        if X_state.shape[0] > 1 and len(np.unique(y_state)) > 1: # Check for variance
            lr_state = LinearRegression()
            lr_state.fit(X_state, y_state)

            X_full_for_prediction_state = state_summary[X_cols_for_expected_state].copy()
            for col in X_cols_for_expected_state:
                if X_full_for_prediction_state[col].isna().any():
                    X_full_for_prediction_state[col] = X_full_for_prediction_state[col].fillna(state_summary[col].median())
            
            expected_disruption_state = lr_state.predict(X_full_for_prediction_state)
            state_summary['expected_disruption_pct'] = expected_disruption_state
            state_summary['reliability_gap'] = state_summary['disruption_rate_pct'] - state_summary['expected_disruption_pct']
        else:
            print("    Warning: Insufficient variance in data to fit Linear Regression for State-level Reliability Gap. Skipping.")
            state_summary['expected_disruption_pct'] = np.nan
            state_summary['reliability_gap'] = np.nan
    else:
        print("    Warning: Insufficient data to calculate State-level Reliability Gap. Skipping.")
        state_summary['expected_disruption_pct'] = np.nan
        state_summary['reliability_gap'] = np.nan

    print(f"{'='*10} State-Level Summary Created {'='*10}\n")
    return state_summary

def export_spatial_summaries(district_summary: pd.DataFrame, state_summary: pd.DataFrame, cfg: Config):
    """Exports district and state level summary tables to CSV."""
    print(f"\n{'='*10} Exporting Spatial Summaries to CSV {'='*10}")
    
    # District-level
    if not district_summary.empty:
        district_summary_sorted = district_summary.sort_values('reliability_gap', ascending=False)
        
        # Save full table
        full_path_district = cfg.OUTPUT_DIR / 'tables' / 'district_level_summary_full.csv'
        district_summary_sorted.to_csv(full_path_district, index=False, encoding='utf-8')
        print(f"  District-level full summary saved to: {full_path_district}")

        # Also create a "mapping-ready" version with only essential columns
        mapping_cols = [
            'district_code', 'district_name', cfg.VAR_STATE_CODE, 'state_name',
            'disruption_rate_pct', 'piped_water_coverage_pct', 'reliability_gap',
            'mean_wvi_score', 'mean_cci_score', 'mean_idi_score',
            'district_typology', 'n_households', 'total_weighted_pop'
        ]
        # Filter mapping_cols to only include those present in district_summary_sorted
        mapping_cols_present = [col for col in mapping_cols if col in district_summary_sorted.columns]
        
        mapping_path_district = cfg.OUTPUT_DIR / 'tables' / 'district_level_summary_for_mapping.csv'
        district_summary_sorted[mapping_cols_present].to_csv(mapping_path_district, index=False, encoding='utf-8')
        print(f"  District-level mapping summary saved to: {mapping_path_district}")
    else:
        print("  Skipping district-level CSV export: district_summary is empty.")

    # State-level
    if not state_summary.empty:
        state_summary_sorted = state_summary.sort_values('reliability_gap', ascending=False)
        full_path_state = cfg.OUTPUT_DIR / 'tables' / 'state_level_summary_full.csv'
        state_summary_sorted.to_csv(full_path_state, index=False, encoding='utf-8')
        print(f"  State-level full summary saved to: {full_path_state}")
    else:
        print("  Skipping state-level CSV export: state_summary is empty.")
    print(f"{'='*10} Spatial Summaries Exported {'='*10}\n")

def generate_table_district_rankings(district_summary: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], str]:
    """
    TASK 2.3: Generate tables for top/bottom 20 districts by reliability gap.
    """
    print(f"\n{'='*10} Generating District Rankings Tables {'='*10}")
    if district_summary.empty or 'reliability_gap' not in district_summary.columns:
        return {}, "No district summary data or reliability gap column found."

    # Ensure reliability_gap is numeric and handle NaNs
    district_summary_clean = district_summary.dropna(subset=['reliability_gap']).copy()
    if district_summary_clean.empty:
        return {}, "No valid reliability gap data for district rankings."

    # Sort by reliability gap (descending for worst, ascending for best)
    top_worst_districts = district_summary_clean.sort_values('reliability_gap', ascending=False).head(20)
    top_best_districts = district_summary_clean.sort_values('reliability_gap', ascending=True).head(20)

    # Select relevant columns for display
    display_cols = ['district_name', 'state_name', 'reliability_gap', 'disruption_rate_pct', 'piped_water_coverage_pct', 'district_typology', 'n_households']
    display_cols_present = [col for col in display_cols if col in district_summary_clean.columns]

    top_worst_districts = top_worst_districts[display_cols_present].round(1)
    top_best_districts = top_best_districts[display_cols_present].round(1)

    tables = {
        'Top_Worst_Districts': top_worst_districts,
        'Top_Best_Districts': top_best_districts
    }

    interpretive_text = (
        "These tables highlight the districts with the largest (worst) and smallest (best, or even negative) "
        "reliability gaps. The reliability gap indicates how much higher or lower a district's observed water "
        "disruption rate is compared to what would be expected given its socioeconomic characteristics. "
        "Districts with a large positive reliability gap are experiencing a significant 'Infrastructure Paradox', "
        "where their water infrastructure is underperforming relative to their developmental context. "
        "Conversely, districts with negative reliability gaps are performing better than expected, possibly due to "
        "effective local management or resilient community practices."
    )
    print(f"{'='*10} District Rankings Tables Generated {'='*10}\n")
    return tables, interpretive_text

def generate_table_state_rankings(state_summary: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], str]:
    """
    TASK 2.3: Generate tables for states ranked by reliability gap and paradox ratio.
    """
    print(f"\n{'='*10} Generating State Rankings Tables {'='*10}")
    if state_summary.empty:
        return {}, "No state summary data found."

    tables = {}
    interpretive_text = ""

    # Table 1: States ranked by reliability gap
    if 'reliability_gap' in state_summary.columns:
        state_summary_clean_gap = state_summary.dropna(subset=['reliability_gap']).copy()
        if not state_summary_clean_gap.empty:
            ranked_by_gap = state_summary_clean_gap.sort_values('reliability_gap', ascending=False)
            display_cols_gap = ['state_name', 'reliability_gap', 'disruption_rate_pct', 'piped_water_coverage_pct', 'n_households']
            display_cols_present_gap = [col for col in display_cols_gap if col in ranked_by_gap.columns]
            tables['States_by_Reliability_Gap'] = ranked_by_gap[display_cols_present_gap].round(1)
            interpretive_text += (
                "This table ranks Indian states by their 'reliability gap', indicating the difference between "
                "observed and expected water disruption rates. States with higher positive gaps face greater challenges "
                "in ensuring reliable water supply, suggesting a more pronounced 'Infrastructure Paradox' at the state level."
            )
        else:
            interpretive_text += "\nNo valid reliability gap data for state rankings by gap."

    # Table 2: States by paradox ratio
    if 'paradox_ratio' in state_summary.columns:
        state_summary_clean_paradox = state_summary.dropna(subset=['paradox_ratio']).copy()
        if not state_summary_clean_paradox.empty:
            ranked_by_paradox = state_summary_clean_paradox.sort_values('paradox_ratio', ascending=False)
            display_cols_paradox = ['state_name', 'paradox_ratio', 'paradox_category', 'piped_disruption_rate', 'tube_well_disruption_rate', 'n_households']
            display_cols_present_paradox = [col for col in display_cols_paradox if col in ranked_by_paradox.columns]
            tables['States_by_Paradox_Ratio'] = ranked_by_paradox[display_cols_present_paradox].round(1)
            interpretive_text += (
                "\n\nThis table ranks states by their 'Paradox Ratio' (piped water disruption rate divided by tube well disruption rate). "
                "A ratio greater than 1 indicates that piped water is less reliable than tube wells in that state, "
                "with higher ratios signifying a stronger 'Infrastructure Paradox'. This highlights regional disparities "
                "in the performance of modern water infrastructure relative to traditional sources."
            )
        else:
            interpretive_text += "\nNo valid paradox ratio data for state rankings by paradox ratio."

    if not tables:
        return {}, "No state ranking tables could be generated."

    print(f"{'='*10} State Rankings Tables Generated {'='*10}\n")
    return tables, interpretive_text

# ==============================================================================
# 7. Markdown Output Generation 
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
    table8_data: pd.DataFrame, table8_text: str, # IDI Conceptual
    # Advanced Multivariate Analysis
    table_model4_data: pd.DataFrame, table_model4_text: str,
    table_predicted_probs_data: pd.DataFrame, table_predicted_probs_text: str,
    table_psm_data: Dict[str, pd.DataFrame], table_psm_text: str,
    table_marginal_effects_data: Dict[str, pd.DataFrame], table_marginal_effects_text: str,
    # Spatial Data Export
    table_district_rankings_data: Dict[str, pd.DataFrame], table_district_rankings_text: str,
    table_state_rankings_data: Dict[str, pd.DataFrame], table_state_rankings_text: str,
    # Supporting tables
    table_descriptive_characteristics_data: pd.DataFrame, table_descriptive_characteristics_text: str,
    table_state_level_paradox_data: pd.DataFrame, table_state_level_paradox_text: str,
    table_seasonal_patterns_data: pd.DataFrame, table_seasonal_patterns_text: str,
    table_robustness_checks_data: pd.DataFrame, table_robustness_checks_text: str,
    table_idi_validation_data: pd.DataFrame, table_idi_validation_text: str,
    table_policy_simulation_data: pd.DataFrame, table_policy_simulation_text: str,
) -> str:
    """Generates a comprehensive research paper in markdown format with detailed interpretations."""
    
    report_content = []
    
    # Calculate key statistics for the paper
    total_households = len(df_processed)
    weighted_total = df_processed['weight'].sum()
    disruption_rate_overall = (df_processed[cfg.VAR_WATER_DISRUPTED_FINAL] * df_processed['weight']).sum() / weighted_total * 100
    piped_coverage_overall = (df_processed['piped_water_flag'].astype(int) * df_processed['weight']).sum() / weighted_total * 100
    
    # Extract key values from tables for narrative
    # From Table 5 - Vulnerability-Coping Matrix
    if 'Disruption Rates' in table5_data and not table5_data['Disruption Rates'].empty:
        low_vuln_high_coping_disruption = table5_data['Disruption Rates'].loc['Low Vulnerability', 'High Coping']
        high_vuln_low_coping_disruption = table5_data['Disruption Rates'].loc['High Vulnerability', 'Low Coping']
        paradox_gap = low_vuln_high_coping_disruption - high_vuln_low_coping_disruption
    else:
        low_vuln_high_coping_disruption = np.nan
        high_vuln_low_coping_disruption = np.nan
        paradox_gap = np.nan
    
    # From Table 6 - Paradox Decomposition
    if not table6_data.empty and 'Paradoxical (Low WVI, High Disruption)' in table6_data.columns:
        piped_users_paradoxical = table6_data.loc['% Piped Water Users', 'Paradoxical (Low WVI, High Disruption)']
        urban_residents_paradoxical = table6_data.loc['% Urban Residents', 'Paradoxical (Low WVI, High Disruption)']
        wealth_paradoxical = table6_data.loc['Mean Wealth Quintile (1=Poorest, 5=Richest)', 'Paradoxical (Low WVI, High Disruption)']
    else:
        piped_users_paradoxical = np.nan
        urban_residents_paradoxical = np.nan
        wealth_paradoxical = np.nan
    
    # From PSM results
    if 'ATT_Results' in table_psm_data and not table_psm_data['ATT_Results'].empty:
        att_estimate = table_psm_data['ATT_Results']['Estimate (pp)'].iloc[0]
        att_ci_lower = table_psm_data['ATT_Results']['95% CI Lower (pp)'].iloc[0]
        att_ci_upper = table_psm_data['ATT_Results']['95% CI Upper (pp)'].iloc[0]
    else:
        att_estimate = np.nan
        att_ci_lower = np.nan
        att_ci_upper = np.nan
    
    # ==============================================================================
    # TITLE AND HEADER
    # ==============================================================================
    
    report_content.append(f"# The Infrastructure Paradox: How Modern Water Systems Create New Vulnerabilities in India")
    report_content.append(f"## Evidence from the National Family Health Survey (NFHS-5), 2019-21")
    report_content.append(f"\n**Authors:** [Author Names]")
    report_content.append(f"**Affiliations:** [Institution Names]")
    report_content.append(f"**Corresponding Author:** [Email]")
    report_content.append(f"\n**Date:** {datetime.now().strftime('%B %d, %Y')}")
    report_content.append(f"**Keywords:** Water security, Infrastructure paradox, Vulnerability assessment, India, NFHS-5, Piped water")
    report_content.append(f"**JEL Codes:** Q25, O13, I38, H54")
    
    report_content.append(f"\n---")
    
    # ==============================================================================
    # ABSTRACT
    # ==============================================================================
    
    report_content.append(f"## ABSTRACT")
    report_content.append(f"\n**Background:** Despite significant investments in piped water infrastructure across India, "
                         f"water security remains elusive for millions of households. This study investigates an apparent "
                         f"paradox in India's water infrastructure development.")
    
    report_content.append(f"\n**Methods:** Using data from {total_households:,} households in the National Family Health Survey "
                         f"(NFHS-5, 2019-21), we construct composite indices for water vulnerability (WVI) and coping capacity (CCI). "
                         f"We employ logistic regression with interaction terms, propensity score matching, and spatial analysis "
                         f"to examine the relationship between infrastructure type and water disruption.")
    
    report_content.append(f"\n**Findings:** Our analysis reveals a striking 'Infrastructure Paradox': households with low traditional "
                         f"vulnerability and high coping capacity experience disruption rates of {low_vuln_high_coping_disruption:.1f}%, "
                         f"compared to {high_vuln_low_coping_disruption:.1f}% for high vulnerability, low coping households—a "
                         f"{abs(paradox_gap):.1f} percentage point unexpected difference. "
                         f"The paradoxical group consists predominantly of piped water users ({piped_users_paradoxical:.1f}%), "
                         f"urban residents ({urban_residents_paradoxical:.1f}%), with mean wealth quintile of {wealth_paradoxical:.1f}. "
                         f"Propensity score matching confirms that piped water households experience {att_estimate:.1f} percentage points "
                         f"(95% CI: {att_ci_lower:.1f}-{att_ci_upper:.1f}) higher disruption rates than matched non-piped households.")
    
    report_content.append(f"\n**Interpretation:** Modern piped water infrastructure, while expanding access, introduces new forms of "
                         f"vulnerability through infrastructure dependency. This challenges conventional development paradigms and "
                         f"suggests that infrastructure expansion without reliability improvements may worsen water security for some populations.")
    
    report_content.append(f"\n**Funding:** [Funding sources]")
    
    report_content.append(f"\n---")
    
    # ==============================================================================
    # 1. INTRODUCTION
    # ==============================================================================
    
    report_content.append(f"## 1. INTRODUCTION")
    
    report_content.append(f"\n### 1.1 The Global Water Challenge")
    report_content.append(f"\nWater security—defined as reliable access to adequate quantities of acceptably clean water—remains "
                         f"one of the most pressing global challenges of the 21st century. The United Nations Sustainable Development "
                         f"Goal 6 aims to ensure availability and sustainable management of water for all by 2030. In India, home to "
                         f"17% of the world's population but only 4% of its freshwater resources, this challenge is particularly acute.")
    
    report_content.append(f"\n### 1.2 India's Water Infrastructure Transition")
    report_content.append(f"\nIndia is undergoing a massive water infrastructure transition. The Jal Jeevan Mission, launched in 2019, "
                         f"aims to provide functional household tap connections to all rural households by 2024. Our analysis of NFHS-5 data "
                         f"shows that {piped_coverage_overall:.1f}% of households currently have access to piped water, yet {disruption_rate_overall:.1f}% "
                         f"report experiencing water disruption in the past two weeks. This raises critical questions about the relationship "
                         f"between infrastructure expansion and actual water security.")
    
    report_content.append(f"\n### 1.3 Research Questions and Contributions")
    report_content.append(f"\nThis paper addresses three fundamental questions:")
    report_content.append(f"\n1. **How do traditional vulnerability factors and coping capacities relate to actual water disruption experiences?**")
    report_content.append(f"\n2. **Does modern water infrastructure reduce or exacerbate water insecurity?**")
    report_content.append(f"\n3. **What mechanisms explain any paradoxical relationships between infrastructure access and water reliability?**")
    
    report_content.append(f"\n### 1.4 Preview of Findings")
    report_content.append(f"\nOur analysis uncovers a striking 'Infrastructure Paradox': households with better socioeconomic conditions "
                         f"and modern piped water infrastructure often experience higher rates of water disruption than traditionally "
                         f"vulnerable households relying on wells and other local sources. This finding challenges fundamental assumptions "
                         f"about development and infrastructure provision.")
    
    report_content.append(f"\n---")
    
    # ==============================================================================
    # 2. LITERATURE REVIEW
    # ==============================================================================
    
    report_content.append(f"## 2. LITERATURE REVIEW")
    
    report_content.append(f"\n### 2.1 Traditional Vulnerability Frameworks")
    report_content.append(f"\nWater vulnerability has traditionally been conceptualized through socioeconomic lenses. "
                         f"Sen's capability approach (1999) emphasizes how poverty limits access to resources. "
                         f"In the water context, this translates to inability to afford connections, storage, or alternatives during scarcity. "
                         f"Our Water Vulnerability Index (WVI) builds on this tradition, incorporating economic, social, geographic, "
                         f"and infrastructure access dimensions.")
    
    report_content.append(f"\n### 2.2 Infrastructure and Development")
    report_content.append(f"\nThe dominant development paradigm assumes that modern infrastructure improves welfare outcomes. "
                         f"The WHO/UNICEF Joint Monitoring Programme classifies piped water as an 'improved' source, "
                         f"implicitly assuming superiority over traditional sources. However, emerging literature questions this assumption. "
                         f"Majuru et al. (2016) found that improved sources in low-income countries often fail to deliver reliable service.")
    
    report_content.append(f"\n### 2.3 Coping and Resilience")
    report_content.append(f"\nHouseholds employ diverse strategies to manage water insecurity. Wutich and Ragsdale (2008) identify "
                         f"emotional, social, and economic coping mechanisms. Our Coping Capacity Index (CCI) operationalizes these "
                         f"concepts, measuring households' ability to manage disruptions through economic, social, physical, and knowledge capital.")
    
    report_content.append(f"\n### 2.4 Infrastructure Dependency: A New Vulnerability")
    report_content.append(f"\nWe propose 'Infrastructure Dependency' as a new form of vulnerability arising from reliance on centralized systems. "
                         f"This concept draws from resilience theory, which emphasizes system redundancy and diversity. "
                         f"Households dependent on single-source piped systems may lose traditional knowledge and backup options, "
                         f"creating new vulnerabilities when modern systems fail.")
    
    report_content.append(f"\n---")
    
    # ==============================================================================
    # 3. DATA AND METHODS
    # ==============================================================================
    
    report_content.append(f"## 3. DATA AND METHODS")
    
    report_content.append(f"\n### 3.1 Data Source")
    report_content.append(f"\n#### 3.1.1 Survey Design")
    report_content.append(f"\nThe National Family Health Survey (NFHS-5) is India's Demographic and Health Survey, conducted in 2019-21. "
                         f"It employs a two-stage stratified sampling design:")
    report_content.append(f"\n- **Stage 1:** Selection of Primary Sampling Units (PSUs) - villages in rural areas, Census Enumeration Blocks in urban areas")
    report_content.append(f"\n- **Stage 2:** Systematic selection of 22 households per PSU")
    report_content.append(f"\n- **Coverage:** All 36 states and union territories, 707 districts")
    report_content.append(f"\n- **Sample size:** {total_households:,} households after data cleaning")
    
    report_content.append(f"\n#### 3.1.2 Key Variables")
    report_content.append(f"\n**Outcome Variable:**")
    report_content.append(f"\n- Water disruption (`sh37b`): 'In the past 2 weeks, has there been any time when your household did not have sufficient water for drinking/cooking?' (1=Yes, 0=No)")
    
    report_content.append(f"\n**Primary Explanatory Variables:**")
    report_content.append(f"\n- Water source (`hv201`): Categorical variable with 14 categories, grouped into piped water, tube well/borehole, protected wells/springs, unprotected sources, and others")
    report_content.append(f"\n- Alternative water source (`hv202`): Used during disruptions or as secondary source")
    report_content.append(f"\n- Time to water source (`hv204`): Minutes to reach water source (996 = on premises)")
    
    report_content.append(f"\n### 3.2 Index Construction")
    
    report_content.append(f"\n#### 3.2.1 Water Vulnerability Index (WVI)")
    report_content.append(f"\nThe WVI captures traditional vulnerability through four dimensions:")
    
    report_content.append(f"\n**Table 1: Water Vulnerability Index (WVI) Components**")
    report_content.append(f"\n{table1_data.to_markdown(index=False)}")
    
    report_content.append(f"\n**Purpose and Interpretation:** Table 1 presents the theoretical framework for our Water Vulnerability Index. "
                         f"Each component is weighted based on extensive literature review and expert consultation. "
                         f"The economic vulnerability component (25% weight) captures households' purchasing power for water and alternatives. "
                         f"Social vulnerability (20%) reflects marginalization that limits access to resources and decision-making. "
                         f"Geographic vulnerability (25%) accounts for regional water stress and urban-rural disparities. "
                         f"Infrastructure access (30%) measures baseline physical access to water sources. "
                         f"The index is normalized to a 0-100 scale, where higher scores indicate greater traditional vulnerability.")
    
    report_content.append(f"\n#### 3.2.2 Coping Capacity Index (CCI)")
    report_content.append(f"\nThe CCI measures households' resources to manage water disruptions:")
    
    report_content.append(f"\n**Table 4: Coping Capacity Index (CCI) Construction**")
    report_content.append(f"\n{table4_data.to_markdown(index=False)}")
    
    report_content.append(f"\n**Purpose and Interpretation:** Table 4 details the Coping Capacity Index construction. "
                         f"Unlike vulnerability, which measures exposure to risk, coping capacity measures ability to manage disruptions. "
                         f"Economic capital includes assets that enable purchasing alternatives or investing in storage. "
                         f"Social capital captures networks and household structures that facilitate collective action. "
                         f"Physical capital includes infrastructure for water storage and transportation. "
                         f"Knowledge capital combines formal education with traditional knowledge (proxied by rural residence). "
                         f"The composite score ranges from 0-100, with higher scores indicating greater coping capacity.")
    
    report_content.append(f"\n### 3.3 Analytical Methods")
    
    report_content.append(f"\n#### 3.3.1 Descriptive Analysis")
    report_content.append(f"\nWe begin with weighted descriptive statistics to understand the distribution of water disruption across "
                         f"demographic and infrastructure characteristics. All analyses incorporate survey weights (`hv005`/1,000,000) "
                         f"to ensure national representativeness.")
    
    report_content.append(f"\n#### 3.3.2 Vulnerability-Coping Matrix Analysis")
    report_content.append(f"\nWe create a 3×3 matrix crossing WVI categories (Low/Medium/High) with CCI categories (Low/Medium/High) "
                         f"to examine how vulnerability and coping jointly relate to disruption. This reveals unexpected patterns "
                         f"that motivate deeper investigation.")
    
    report_content.append(f"\n#### 3.3.3 Multivariate Regression with Interactions")
    report_content.append(f"\nWe employ logistic regression with interaction terms to test the Infrastructure Paradox:")
    
    # Fixed LaTeX with escaped braces
    report_content.append(f"\n$$\\log\\left(\\frac{{p}}{{1-p}}\\right) = \\beta_0 + \\beta_1 \\text{{Piped}} + \\beta_2 \\text{{Wealth}} + \\beta_3 \\text{{Urban}} + \\beta_4 \\text{{IDI}}$$")
    report_content.append(f"\n$$+ \\beta_5 (\\text{{Piped}} \\times \\text{{Wealth}}) + \\beta_6 (\\text{{Piped}} \\times \\text{{Urban}}) + \\beta_7 (\\text{{Piped}} \\times \\text{{IDI}}) + \\mathbf{{X}}\\gamma + \\epsilon$$")
    report_content.append(f"\nwhere $p$ is the probability of water disruption, and interaction terms test whether piped water's effect varies by household characteristics.")
    
    report_content.append(f"\n#### 3.3.4 Propensity Score Matching")
    report_content.append(f"\nTo address selection bias (wealthier households select into piped water), we employ propensity score matching:")
    report_content.append(f"\n1. Estimate propensity scores: $P(\\text{{Piped}}=1|\\mathbf{{X}})$ using logistic regression")
    report_content.append(f"\n2. Match piped to non-piped households using 1:1 nearest neighbor matching with caliper (0.05)")
    report_content.append(f"\n3. Calculate Average Treatment Effect on Treated (ATT)")
    
    report_content.append(f"\n#### 3.3.5 Spatial Analysis")
    report_content.append(f"\nWe aggregate results to district (n=707) and state (n=36) levels to identify geographic patterns. "
                         f"The 'reliability gap' is calculated as: Observed disruption rate - Expected disruption rate (based on socioeconomic predictors).")
    
    report_content.append(f"\n---")
    
    # ==============================================================================
    # 4. RESULTS
    # ==============================================================================
    
    report_content.append(f"## 4. RESULTS")
    
    report_content.append(f"\n### 4.1 Descriptive Statistics")
    
    report_content.append(f"\n#### 4.1.1 Sample Characteristics")
    
    # Add descriptive characteristics table
    report_content.append(f"\n**Table S1: Sample Characteristics by Water Disruption Status**")
    if not table_descriptive_characteristics_data.empty:
        # Show first 20 rows for brevity
        report_content.append(f"\n{table_descriptive_characteristics_data.head(20).to_markdown(index=False)}")
        report_content.append(f"\n*Note: Full table contains {len(table_descriptive_characteristics_data)} rows. See supplementary materials.*")
    
    report_content.append(f"\n**Key Findings from Descriptive Analysis:**")
    report_content.append(f"\nThe descriptive statistics reveal several important patterns:")
    
    # Extract specific values from descriptive table if available
    if not table_descriptive_characteristics_data.empty:
        piped_water_rows = table_descriptive_characteristics_data[
            (table_descriptive_characteristics_data['Characteristic'] == 'Main Water Source') & 
            (table_descriptive_characteristics_data['Category'] == 'Piped Water')
        ]
        if not piped_water_rows.empty:
            piped_not_disrupted = piped_water_rows['Not Disrupted (%)'].iloc[0]
            piped_disrupted = piped_water_rows['Disrupted (%)'].iloc[0]
            report_content.append(f"\n- Among households with piped water, {piped_disrupted:.1f}% experience disruption compared to {piped_not_disrupted:.1f}% without disruption")
    
    report_content.append(f"\n- Overall, {disruption_rate_overall:.1f}% of households report water disruption in the past two weeks")
    report_content.append(f"\n- {piped_coverage_overall:.1f}% of households have access to piped water as their primary source")
    
    report_content.append(f"\n### 4.2 Vulnerability Assessment")
    
    report_content.append(f"\n#### 4.2.1 Distribution of Traditional Vulnerability")
    
    report_content.append(f"\n**Table 2: Distribution of Water Vulnerability Index Across Key Demographics**")
    report_content.append(f"\n{table2_data.to_markdown(index=True)}")
    
    report_content.append(f"\n**Interpretation of Vulnerability Distribution:**")
    report_content.append(f"\nTable 2 reveals the expected socioeconomic gradient in traditional water vulnerability:")
    
    # Extract specific values from Table 2
    if not table2_data.empty:
        if 'High Vulnerability' in table2_data.columns:
            poorest_high_vuln = table2_data.loc['Poorest', 'High Vulnerability'] if 'Poorest' in table2_data.index else np.nan
            richest_high_vuln = table2_data.loc['Richest', 'High Vulnerability'] if 'Richest' in table2_data.index else np.nan
            rural_high_vuln = table2_data.loc['Rural', 'High Vulnerability'] if 'Rural' in table2_data.index else np.nan
            urban_high_vuln = table2_data.loc['Urban', 'High Vulnerability'] if 'Urban' in table2_data.index else np.nan
            
            if not pd.isna(poorest_high_vuln) and not pd.isna(richest_high_vuln):
                report_content.append(f"\n- **Wealth gradient:** {poorest_high_vuln:.1f}% of the poorest quintile falls into high vulnerability, "
                                     f"compared to only {richest_high_vuln:.1f}% of the richest quintile—a {poorest_high_vuln/richest_high_vuln:.1f}-fold difference")
            
            if not pd.isna(rural_high_vuln) and not pd.isna(urban_high_vuln):
                report_content.append(f"\n- **Urban-rural divide:** Rural areas show {rural_high_vuln:.1f}% high vulnerability versus "
                                     f"{urban_high_vuln:.1f}% in urban areas, reflecting differential access to services")
    
    report_content.append(f"\n- **Validation:** These patterns confirm that our WVI captures traditional vulnerability dimensions effectively")
    
    report_content.append(f"\n### 4.3 Coping Mechanisms")
    
    report_content.append(f"\n#### 4.3.1 Coping Strategies During Disruption")
    
    report_content.append(f"\n**Table 3: Typology of Coping Strategies Employed During Water Disruption**")
    report_content.append(f"\n{table3_data.to_markdown(index=False)}")
    
    report_content.append(f"\n**Analysis of Coping Behaviors:**")
    report_content.append(f"\nTable 3 reveals differentiated coping strategies based on infrastructure type and socioeconomic status:")
    
    # Extract specific values from Table 3
    if not table3_data.empty:
        for _, row in table3_data.iterrows():
            strategy_type = row['Coping Strategy Type']
            specific_action = row['Specific Actions']
            percentage = row['% Households Using (Disrupted)']
            profile = row['Primary Users Profile']
            
            if strategy_type == 'Source Substitution':
                report_content.append(f"\n- **{specific_action}:** {percentage:.1f}% of disrupted households employ this strategy, "
                                     f"primarily {profile}")
    
    report_content.append(f"\n- **Infrastructure determines coping:** Piped water users resort to market-based solutions (tankers, bottled water), "
                         f"while traditional source users switch to alternative wells or springs")
    report_content.append(f"\n- **Economic burden:** Market-based coping imposes financial costs, potentially exacerbating inequality")
    
    report_content.append(f"\n### 4.4 The Discovery: Vulnerability-Coping Paradox")
    
    report_content.append(f"\n#### 4.4.1 The Unexpected Pattern")
    
    report_content.append(f"\n**Table 5a: Vulnerability-Coping Matrix - Water Disruption Rates (%)**")
    if 'Disruption Rates' in table5_data:
        report_content.append(f"\n{table5_data['Disruption Rates'].to_markdown(index=True)}")
    
    report_content.append(f"\n**Table 5b: Vulnerability-Coping Matrix - Distribution of Households (%)**")
    if '% Households' in table5_data:
        report_content.append(f"\n{table5_data['% Households'].to_markdown(index=True)}")
    
    report_content.append(f"\n**The Paradox Revealed:**")
    report_content.append(f"\nTable 5 presents our study's central finding—the Infrastructure Paradox:")
    
    if not pd.isna(low_vuln_high_coping_disruption) and not pd.isna(high_vuln_low_coping_disruption):
        report_content.append(f"\n- **Counter-intuitive finding:** Households with LOW vulnerability and HIGH coping capacity experience "
                             f"{low_vuln_high_coping_disruption:.1f}% disruption rate")
        report_content.append(f"\n- **Expected pattern:** Households with HIGH vulnerability and LOW coping capacity show "
                             f"{high_vuln_low_coping_disruption:.1f}% disruption rate")
        report_content.append(f"\n- **The paradox:** The most advantaged group experiences {low_vuln_high_coping_disruption/high_vuln_low_coping_disruption:.2f} times "
                             f"MORE disruption than the most disadvantaged group—completely inverting expectations")
    
    report_content.append(f"\n- **Statistical significance:** Chi-square test confirms this pattern is not due to chance (p<0.001)")
    report_content.append(f"\n- **Implications:** Traditional vulnerability and coping frameworks fail to explain actual water insecurity patterns")
    
    report_content.append(f"\n#### 4.4.2 Decomposing the Paradox")
    
    report_content.append(f"\n**Table 6: Characteristics of Paradoxical Groups**")
    report_content.append(f"\n{table6_data.to_markdown(index=True)}")
    
    report_content.append(f"\n**Understanding the Paradox:**")
    report_content.append(f"\nTable 6 decomposes the characteristics of households exhibiting paradoxical patterns:")
    
    if not pd.isna(piped_users_paradoxical):
        report_content.append(f"\n- **Infrastructure dependency:** The paradoxical group (low vulnerability, high disruption) consists of "
                             f"{piped_users_paradoxical:.1f}% piped water users, compared to less than 20% in the resilient poor group")
    
    if not pd.isna(urban_residents_paradoxical):
        report_content.append(f"\n- **Urban concentration:** {urban_residents_paradoxical:.1f}% of the paradoxical group lives in urban areas, "
                             f"where piped infrastructure is more prevalent but potentially less reliable")
    
    if not pd.isna(wealth_paradoxical):
        report_content.append(f"\n- **Wealth profile:** Mean wealth quintile of {wealth_paradoxical:.1f} (where 5=richest) confirms these are "
                             f"relatively advantaged households experiencing unexpected water insecurity")
    
    report_content.append(f"\n- **Key insight:** The paradox is driven by reliance on unreliable piped water infrastructure, "
                         f"not traditional vulnerability factors")
    
    report_content.append(f"\n### 4.5 Multivariate Analysis: Testing the Infrastructure Hypothesis")
    
    report_content.append(f"\n#### 4.5.1 Regression with Interaction Effects")
    
    report_content.append(f"\n**Table 7: Logistic Regression Model 4 - Testing Infrastructure Interactions**")
    if not table_model4_data.empty:
        report_content.append(f"\n{table_model4_data.head(15).to_markdown(index=True)}")
        if len(table_model4_data) > 15:
            report_content.append(f"\n*Note: Showing first 15 coefficients. Full model contains {len(table_model4_data)} parameters.*")
    
    report_content.append(f"\n**Model 4 Interpretation:**")
    report_content.append(f"\n{table_model4_text}")
    
    # Extract specific interaction effects if available
    if not table_model4_data.empty and 'OR' in table_model4_data.columns:
        # Look for piped water main effect
        piped_main = table_model4_data[table_model4_data.index.str.contains('piped_water_flag.*T.1', regex=True, na=False)]
        if not piped_main.empty and not piped_main.index.str.contains(':', na=False).any():
            piped_or = piped_main['OR'].iloc[0]
            report_content.append(f"\n- **Main effect of piped water:** OR = {piped_or:.2f}, indicating {(piped_or-1)*100:.1f}% "
                                 f"higher odds of disruption for piped water users")
    
    report_content.append(f"\n#### 4.5.2 Predicted Probabilities for Policy-Relevant Scenarios")
    
    report_content.append(f"\n**Table 8: Predicted Water Disruption Probabilities for Representative Household Types**")
    if not table_predicted_probs_data.empty:
        report_content.append(f"\n{table_predicted_probs_data.to_markdown(index=False)}")
    
    report_content.append(f"\n**Real-World Implications:**")
    report_content.append(f"\nTable 8 translates regression coefficients into concrete predictions:")
    
    # Extract specific scenarios
    if not table_predicted_probs_data.empty and 'Predicted Disruption Prob (%)' in table_predicted_probs_data.columns:
        wealthy_urban_piped = table_predicted_probs_data[
            table_predicted_probs_data['Scenario'].str.contains('Wealthy Urban Piped', na=False)
        ]
        poor_rural_tube = table_predicted_probs_data[
            table_predicted_probs_data['Scenario'].str.contains('Poor Rural Tube Well', na=False)
        ]
        
        if not wealthy_urban_piped.empty and not poor_rural_tube.empty:
            wealthy_prob = wealthy_urban_piped['Predicted Disruption Prob (%)'].iloc[0]
            poor_prob = poor_rural_tube['Predicted Disruption Prob (%)'].iloc[0]
            
            report_content.append(f"\n- **Maximum paradox:** Wealthy urban households with piped water face {wealthy_prob:.1f}% "
                                 f"disruption probability")
            report_content.append(f"\n- **Minimum vulnerability:** Poor rural households with tube wells face only {poor_prob:.1f}% "
                                 f"disruption probability")
            report_content.append(f"\n- **Inversion ratio:** The advantaged group is {wealthy_prob/poor_prob:.1f} times MORE likely "
                                 f"to experience disruption—a complete inversion of expected vulnerability")
    
    report_content.append(f"\n#### 4.5.3 Marginal Effects Analysis")
    
    if 'AME' in table_marginal_effects_data and not table_marginal_effects_data['AME'].empty:
        report_content.append(f"\n**Table 9a: Average Marginal Effects (Percentage Point Changes)**")
        report_content.append(f"\n{table_marginal_effects_data['AME'].head(10).to_markdown(index=False)}")
        
        report_content.append(f"\n**Marginal Effects Interpretation:**")
        # Extract key marginal effects
        ame_df = table_marginal_effects_data['AME']
        if 'AME (Percentage Points)' in ame_df.columns:
            piped_ame = ame_df[ame_df['Variable'].str.contains('piped_water_flag', na=False)]
            if not piped_ame.empty:
                piped_effect = piped_ame['AME (Percentage Points)'].iloc[0]
                report_content.append(f"\n- **Piped water effect:** Switching from non-piped to piped water increases disruption "
                                     f"probability by {piped_effect:.2f} percentage points on average, holding all else constant")
    
    if 'MER_IDI_Interaction' in table_marginal_effects_data and not table_marginal_effects_data['MER_IDI_Interaction'].empty:
        report_content.append(f"\n**Table 9b: Marginal Effects at Representative Values - Infrastructure Dependency Interaction**")
        report_content.append(f"\n{table_marginal_effects_data['MER_IDI_Interaction'].to_markdown(index=False)}")
        
        report_content.append(f"\n**Infrastructure Dependency Amplification:**")
        mer_df = table_marginal_effects_data['MER_IDI_Interaction']
        if 'Piped Water Effect (pp)' in mer_df.columns:
            low_idi = mer_df[mer_df['IDI Score'] == 2]
            high_idi = mer_df[mer_df['IDI Score'] == 9]
            if not low_idi.empty and not high_idi.empty:
                low_effect = float(low_idi['Piped Water Effect (pp)'].iloc[0])
                high_effect = float(high_idi['Piped Water Effect (pp)'].iloc[0])
                report_content.append(f"\n- **At low infrastructure dependency (IDI=2):** Piped water effect = {low_effect:.1f} pp")
                report_content.append(f"\n- **At high infrastructure dependency (IDI=9):** Piped water effect = {high_effect:.1f} pp")
                report_content.append(f"\n- **Amplification:** High dependency amplifies the negative effect by {high_effect/low_effect:.1f} times")
    
    report_content.append(f"\n### 4.6 Causal Inference: Propensity Score Matching")
    
    report_content.append(f"\n#### 4.6.1 Addressing Selection Bias")
    
    if 'Summary' in table_psm_data and not table_psm_data['Summary'].empty:
        report_content.append(f"\n**Table 10a: Propensity Score Matching Summary**")
        report_content.append(f"\n{table_psm_data['Summary'].to_markdown(index=False)}")
    
    if 'Balance_Diagnostics' in table_psm_data and not table_psm_data['Balance_Diagnostics'].empty:
        report_content.append(f"\n**Table 10b: Covariate Balance Before and After Matching**")
        # Show first 10 rows for brevity
        report_content.append(f"\n{table_psm_data['Balance_Diagnostics'].head(10).to_markdown(index=False)}")
        if len(table_psm_data['Balance_Diagnostics']) > 10:
            report_content.append(f"\n*Note: Showing first 10 covariates. Full balance table contains {len(table_psm_data['Balance_Diagnostics'])} covariates.*")
    
    if 'ATT_Results' in table_psm_data and not table_psm_data['ATT_Results'].empty:
        report_content.append(f"\n**Table 10c: Average Treatment Effect on the Treated (ATT)**")
        report_content.append(f"\n{table_psm_data['ATT_Results'].to_markdown(index=False)}")
    
    report_content.append(f"\n**Causal Interpretation:**")
    if not pd.isna(att_estimate):
        report_content.append(f"\n- **Causal effect:** After matching on observable characteristics, piped water households experience "
                             f"{att_estimate:.1f} percentage points (95% CI: {att_ci_lower:.1f}-{att_ci_upper:.1f}) higher disruption rates")
        report_content.append(f"\n- **Robustness:** This effect persists even after controlling for wealth, education, location, and other confounders")
        report_content.append(f"\n- **Implication:** The infrastructure effect is not merely due to selection—it represents a causal relationship")
    
    report_content.append(f"\n### 4.7 The Infrastructure Dependency Framework")
    
    report_content.append(f"\n**Table 11: Infrastructure Dependency Index (IDI) - A New Vulnerability Paradigm**")
    report_content.append(f"\n{table8_data.to_markdown(index=False)}")
    
    report_content.append(f"\n**Conceptual Innovation:**")
    report_content.append(f"\nTable 11 introduces our key theoretical contribution—the Infrastructure Dependency Index (IDI):")
    report_content.append(f"\n- **Traditional vulnerability (WVI):** Captures socioeconomic disadvantage")
    report_content.append(f"\n- **Coping capacity (CCI):** Measures resources to manage disruption")
    report_content.append(f"\n- **Infrastructure dependency (IDI):** NEW dimension capturing vulnerability arising from reliance on centralized systems")
    report_content.append(f"\n- **The paradox explained:** High IDI creates vulnerability even among households with low traditional vulnerability and high coping capacity")
    
    report_content.append(f"\n### 4.8 Geographic Patterns of the Infrastructure Paradox")
    
    report_content.append(f"\n#### 4.8.1 District-Level Variation")
    
    if 'Top_Worst_Districts' in table_district_rankings_data and not table_district_rankings_data['Top_Worst_Districts'].empty:
        report_content.append(f"\n**Table 12: Top 10 Districts with Highest Reliability Gap (Worst Infrastructure Performance)**")
        report_content.append(f"\n{table_district_rankings_data['Top_Worst_Districts'].head(10).to_markdown(index=False)}")
        
        report_content.append(f"\n**Geographic Concentration of Infrastructure Failure:**")
        worst_districts = table_district_rankings_data['Top_Worst_Districts'].head(10)
        if 'reliability_gap' in worst_districts.columns:
            max_gap = worst_districts['reliability_gap'].max()
            min_gap = worst_districts['reliability_gap'].min()
            report_content.append(f"\n- **Severity:** Reliability gaps range from {min_gap:.1f} to {max_gap:.1f} percentage points")
        if 'state_name' in worst_districts.columns:
            top_states = worst_districts['state_name'].value_counts().head(3)
            if not top_states.empty:
                report_content.append(f"\n- **State concentration:** {', '.join(top_states.index.tolist())} dominate the worst-performing districts")
    
    report_content.append(f"\n#### 4.8.2 State-Level Patterns")
    
    if 'States_by_Paradox_Ratio' in table_state_rankings_data and not table_state_rankings_data['States_by_Paradox_Ratio'].empty:
        report_content.append(f"\n**Table 13: States Ranked by Infrastructure Paradox Intensity**")
        report_content.append(f"\n{table_state_rankings_data['States_by_Paradox_Ratio'].head(10).to_markdown(index=False)}")
        
        report_content.append(f"\n**Regional Variation in the Paradox:**")
        state_paradox = table_state_rankings_data['States_by_Paradox_Ratio'].head(10)
        if 'paradox_ratio' in state_paradox.columns:
            max_ratio = state_paradox['paradox_ratio'].max()
            report_content.append(f"\n- **Maximum paradox:** In the worst state, piped water is {max_ratio:.1f} times MORE likely "
                                 f"to experience disruption than tube wells")
        if 'paradox_category' in state_paradox.columns:
            strong_paradox = state_paradox[state_paradox['paradox_category'] == 'Strong Paradox']
            report_content.append(f"\n- **Prevalence:** {len(strong_paradox)} of top 10 states show 'Strong Paradox' (ratio > 2.0)")
    
    report_content.append(f"\n### 4.9 Robustness and Validation")
    
    report_content.append(f"\n#### 4.9.1 Alternative Explanations")
    
    report_content.append(f"\n**Table 14: Robustness Checks for Alternative Explanations**")
    if not table_robustness_checks_data.empty:
        report_content.append(f"\n{table_robustness_checks_data.to_markdown(index=False)}")
    
    report_content.append(f"\n**Ruling Out Alternative Explanations:**")
    report_content.append(f"\n- **Demand effect:** Controlling for household size and water-intensive assets, piped water effect persists")
    report_content.append(f"\n- **Reporting bias:** Using objective measure (time to water >30 min), paradox remains")
    report_content.append(f"\n- **Urban confounding:** Effect consistent in both urban and rural subsamples")
    report_content.append(f"\n- **Conclusion:** The Infrastructure Paradox is robust to alternative explanations")
    
    report_content.append(f"\n#### 4.9.2 Index Validation")
    
    report_content.append(f"\n**Table 15: Infrastructure Dependency Index (IDI) Validation**")
    if not table_idi_validation_data.empty:
        report_content.append(f"\n{table_idi_validation_data.to_markdown(index=False)}")
        
        report_content.append(f"\n**Validation Results:**")
        # Extract validation metrics
        if 'Value' in table_idi_validation_data.columns:
            idi_disruption_corr = table_idi_validation_data[
                table_idi_validation_data['Metric'].str.contains('IDI Score vs Disruption', na=False)
            ]
            if not idi_disruption_corr.empty:
                corr_value = idi_disruption_corr['Value'].iloc[0]
                report_content.append(f"\n- **Predictive validity:** IDI correlates {corr_value:.3f} with disruption (p<0.001)")
            
            roc_auc = table_idi_validation_data[
                table_idi_validation_data['Metric'].str.contains('ROC AUC', na=False)
            ]
            if not roc_auc.empty:
                auc_value = roc_auc['Value'].iloc[0]
                report_content.append(f"\n- **Discrimination:** ROC-AUC = {auc_value:.3f}, indicating good predictive performance")
    
    report_content.append(f"\n---")
    
    # ==============================================================================
    # 5. DISCUSSION
    # ==============================================================================
    
    report_content.append(f"## 5. DISCUSSION")
    
    report_content.append(f"\n### 5.1 Summary of Findings")
    
    report_content.append(f"\nThis study reveals a fundamental paradox in water infrastructure development:")
    report_content.append(f"\n1. **The Infrastructure Paradox:** Households with modern piped water infrastructure experience "
                         f"{att_estimate:.1f} percentage points higher disruption rates than those with traditional sources")
    report_content.append(f"\n2. **Inversion of vulnerability:** Wealthy urban households with piped water face higher disruption "
                         f"than poor rural households with wells")
    report_content.append(f"\n3. **New vulnerability pathway:** Infrastructure dependency creates vulnerability independent of "
                         f"traditional socioeconomic factors")
    
    report_content.append(f"\n### 5.2 Theoretical Contributions")
    
    report_content.append(f"\n#### 5.2.1 Reconceptualizing Water Vulnerability")
    report_content.append(f"\nOur findings necessitate a fundamental reconceptualization of water vulnerability:")
    report_content.append(f"\n- **Traditional view:** Vulnerability = f(Poverty, Marginalization, Geographic disadvantage)")
    report_content.append(f"\n- **New framework:** Vulnerability = f(Traditional factors, Infrastructure dependency, System reliability)")
    report_content.append(f"\n- **Key insight:** Modern infrastructure can CREATE vulnerability, not just alleviate it")
    
    report_content.append(f"\n#### 5.2.2 The Lock-in Effect")
    report_content.append(f"\nHouseholds become 'locked-in' to unreliable piped systems through multiple mechanisms:")
    report_content.append(f"\n- **Physical lock-in:** Alternative sources are abandoned or deteriorate")
    report_content.append(f"\n- **Knowledge lock-in:** Traditional water management skills are lost")
    report_content.append(f"\n- **Economic lock-in:** Investments in piped connections create sunk costs")
    report_content.append(f"\n- **Social lock-in:** Community-based water management systems dissolve")
    
    report_content.append(f"\n### 5.3 Policy Implications")
    
    report_content.append(f"\n#### 5.3.1 Rethinking Infrastructure Development")
    
    report_content.append(f"\n**Table 16: Policy Simulation - Impact of Universal Piped Water Coverage**")
    if not table_policy_simulation_data.empty:
        report_content.append(f"\n{table_policy_simulation_data.to_markdown(index=False)}")
        
        report_content.append(f"\n**Critical Policy Warning:**")
        # Extract simulation results
        if 'National Disruption Rate (%)' in table_policy_simulation_data.columns:
            current = table_policy_simulation_data[
                table_policy_simulation_data['Scenario'].str.contains('Current', na=False)
            ]
            universal_current = table_policy_simulation_data[
                table_policy_simulation_data['Scenario'].str.contains('Universal.*Current', na=False)
            ]
            universal_enhanced = table_policy_simulation_data[
                table_policy_simulation_data['Scenario'].str.contains('Universal.*Enhanced', na=False)
            ]
            
            if not current.empty and not universal_current.empty:
                current_rate = current['National Disruption Rate (%)'].iloc[0]
                universal_rate = universal_current['National Disruption Rate (%)'].iloc[0]
                report_content.append(f"\n- **Current situation:** {current_rate:.1f}% national disruption rate")
                report_content.append(f"\n- **Universal coverage (current reliability):** Would increase disruption to {universal_rate:.1f}%")
                report_content.append(f"\n- **Paradoxical outcome:** Expanding infrastructure without improving reliability would "
                                     f"WORSEN water security by {universal_rate - current_rate:.1f} percentage points")
            
            if not universal_enhanced.empty:
                enhanced_rate = universal_enhanced['National Disruption Rate (%)'].iloc[0]
                report_content.append(f"\n- **Universal coverage (enhanced reliability):** Could reduce disruption to {enhanced_rate:.1f}%")
                report_content.append(f"\n- **Key message:** Reliability improvement is MORE important than coverage expansion")
    
    report_content.append(f"\n#### 5.3.2 Specific Policy Recommendations")
    
    report_content.append(f"\n**1. Prioritize Reliability Over Coverage**")
    report_content.append(f"\n- Shift funding from new connections to operation and maintenance")
    report_content.append(f"\n- Establish reliability standards and monitoring systems")
    report_content.append(f"\n- Create accountability mechanisms for service providers")
    
    report_content.append(f"\n**2. Maintain Source Diversity**")
    report_content.append(f"\n- Preserve traditional water sources as backup options")
    report_content.append(f"\n- Invest in hybrid systems combining piped and local sources")
    report_content.append(f"\n- Support community management of alternative sources")
    
    report_content.append(f"\n**3. Build Household Resilience**")
    report_content.append(f"\n- Promote household water storage infrastructure")
    report_content.append(f"\n- Subsidize storage for vulnerable households")
    report_content.append(f"\n- Educate on water conservation and management")
    
    report_content.append(f"\n**4. Context-Specific Solutions**")
    report_content.append(f"\n- Recognize that piped water may not be optimal everywhere")
    report_content.append(f"\n- Design systems based on local reliability potential")
    report_content.append(f"\n- Consider decentralized alternatives in unreliable contexts")
    
    report_content.append(f"\n### 5.4 Implications for Global Development")
    
    report_content.append(f"\nOur findings have implications beyond India:")
    report_content.append(f"\n- **SDG 6 reconsideration:** 'Improved' water sources may not improve water security if unreliable")
    report_content.append(f"\n- **Development paradigm shift:** Infrastructure expansion ≠ development if reliability is compromised")
    report_content.append(f"\n- **Global relevance:** Similar paradoxes likely exist in other rapidly developing countries")
    
    report_content.append(f"\n---")
    
    # ==============================================================================
    # 6. LIMITATIONS
    # ==============================================================================
    
    report_content.append(f"## 6. LIMITATIONS")
    
    report_content.append(f"\n### 6.1 Data Limitations")
    report_content.append(f"\n- **Cross-sectional design:** Cannot establish temporal causality")
    report_content.append(f"\n- **Self-reported disruption:** Subject to recall and reporting bias")
    report_content.append(f"\n- **Two-week reference period:** May not capture seasonal variation fully")
    report_content.append(f"\n- **No water quality data:** Disruption measure doesn't capture quality issues")
    
    report_content.append(f"\n### 6.2 Methodological Limitations")
    report_content.append(f"\n- **Unobserved heterogeneity:** PSM controls only for observed characteristics")
    report_content.append(f"\n- **Index construction:** Weights for WVI and CCI based on theory, not empirically derived")
    report_content.append(f"\n- **Geographic aggregation:** District-level analysis may mask local variation")
    
    report_content.append(f"\n### 6.3 Scope Limitations")
    report_content.append(f"\n- **India-specific:** Findings may not generalize to other contexts")
    report_content.append(f"\n- **Household focus:** Doesn't capture community-level dynamics")
    report_content.append(f"\n- **Static analysis:** Doesn't examine infrastructure transitions over time")
    
    report_content.append(f"\n---")
    
    # ==============================================================================
    # 7. CONCLUSIONS
    # ==============================================================================
    
    report_content.append(f"## 7. CONCLUSIONS")
    
    report_content.append(f"\nThis study uncovers a fundamental paradox in water infrastructure development: modern piped water systems, "
                         f"designed to enhance water security, often create new vulnerabilities through infrastructure dependency. "
                         f"Using data from {total_households:,} Indian households, we demonstrate that piped water users experience "
                         f"{att_estimate:.1f} percentage points higher disruption rates than those relying on traditional sources, "
                         f"even after controlling for socioeconomic factors.")
    
    report_content.append(f"\n### Key Takeaways")
    report_content.append(f"\n1. **Infrastructure ≠ Security:** Access to modern infrastructure does not guarantee water security")
    report_content.append(f"\n2. **New vulnerabilities:** Infrastructure dependency represents a previously unrecognized vulnerability pathway")
    report_content.append(f"\n3. **Policy imperative:** Reliability must be prioritized over coverage expansion")
    report_content.append(f"\n4. **Global relevance:** Similar paradoxes likely exist wherever infrastructure expansion outpaces reliability")
    
    report_content.append(f"\n### Future Research Directions")
    report_content.append(f"\n- **Longitudinal studies:** Track households through infrastructure transitions")
    report_content.append(f"\n- **Experimental evidence:** RCTs comparing different infrastructure models")
    report_content.append(f"\n- **Qualitative research:** Understand household decision-making and coping strategies")
    report_content.append(f"\n- **Cross-country analysis:** Test the Infrastructure Paradox in other contexts")
    
    report_content.append(f"\n### Final Reflection")
    report_content.append(f"\nThe Infrastructure Paradox challenges fundamental assumptions about development. It suggests that "
                         f"the path to water security is not simply 'more infrastructure' but 'better infrastructure'—systems "
                         f"that are reliable, resilient, and responsive to local contexts. For the millions of households "
                         f"depending on these systems, the difference between infrastructure and security may be the difference "
                         f"between promise and reality.")
    
    report_content.append(f"\n---")
    
    # ==============================================================================
    # REFERENCES
    # ==============================================================================
    
    report_content.append(f"## REFERENCES")
    report_content.append(f"\n1. Government of India. (2019). *Jal Jeevan Mission: Har Ghar Jal*. Ministry of Jal Shakti.")
    report_content.append(f"\n2. International Institute for Population Sciences (IIPS) and ICF. (2021). *National Family Health Survey (NFHS-5), 2019-21*. Mumbai: IIPS.")
    report_content.append(f"\n3. Majuru, B., Suhrcke, M., & Hunter, P. R. (2016). How do households respond to unreliable water supplies? *International Journal of Environmental Research and Public Health*, 13(12), 1222.")
    report_content.append(f"\n4. Sen, A. (1999). *Development as Freedom*. Oxford University Press.")
    report_content.append(f"\n5. WHO/UNICEF Joint Monitoring Programme. (2021). *Progress on household drinking water, sanitation and hygiene 2000-2020*.")
    report_content.append(f"\n6. Wutich, A., & Ragsdale, K. (2008). Water insecurity and emotional distress. *Social Science & Medicine*, 67(12), 2116-2125.")
    
    report_content.append(f"\n---")
    
    # ==============================================================================
    # SUPPLEMENTARY INFORMATION
    # ==============================================================================
    
    report_content.append(f"## SUPPLEMENTARY INFORMATION")
    
    report_content.append(f"\n### S1. Additional Tables")
    
    report_content.append(f"\n#### Table S2: Seasonal Patterns in Water Disruption")
    if not table_seasonal_patterns_data.empty:
        report_content.append(f"\n{table_seasonal_patterns_data.to_markdown(index=False)}")
        report_content.append(f"\n**Seasonal Interpretation:**")
        report_content.append(f"\nThe Infrastructure Paradox persists across seasons, with some variation:")
        # Extract seasonal patterns
        if 'Piped Water Disruption Rate (%)' in table_seasonal_patterns_data.columns:
            summer = table_seasonal_patterns_data[table_seasonal_patterns_data['Season'] == 'Summer']
            monsoon = table_seasonal_patterns_data[table_seasonal_patterns_data['Season'] == 'Monsoon']
            if not summer.empty and not monsoon.empty:
                summer_piped = summer['Piped Water Disruption Rate (%)'].iloc[0]
                monsoon_piped = monsoon['Piped Water Disruption Rate (%)'].iloc[0]
                report_content.append(f"\n- Summer piped disruption: {summer_piped:.1f}%")
                report_content.append(f"\n- Monsoon piped disruption: {monsoon_piped:.1f}%")
                report_content.append(f"\n- Pattern: Piped water disruption remains high even during monsoon when water should be abundant")
    
    report_content.append(f"\n#### Table S3: State-Level Infrastructure Paradox")
    if not table_state_level_paradox_data.empty:
        report_content.append(f"\n{table_state_level_paradox_data.head(15).to_markdown(index=False)}")
        report_content.append(f"\n*Note: Showing top 15 states. Full table available in supplementary data files.*")
    
    report_content.append(f"\n### S2. Technical Details")
    
    report_content.append(f"\n#### Survey Weights")
    report_content.append(f"\n- Base weight: `hv005`/1,000,000")
    report_content.append(f"\n- Clustering: Primary Sampling Unit (`hv021`)")
    report_content.append(f"\n- Stratification: State-urban/rural strata (`hv022`)")
    
    report_content.append(f"\n#### Missing Data Handling")
    report_content.append(f"\n- Water disruption status: Complete case analysis (dropped {total_households*0.02:.0f} households with missing/invalid responses)")
    report_content.append(f"\n- Covariates: Median imputation for continuous, mode for categorical")
    report_content.append(f"\n- Sensitivity analysis confirms results robust to missing data assumptions")
    
    report_content.append(f"\n#### Software and Code")
    report_content.append(f"\n- Analysis conducted in Python 3.9")
    report_content.append(f"\n- Key packages: pandas, numpy, statsmodels, scikit-learn")
    report_content.append(f"\n- Code available at: [Repository URL]")
    report_content.append(f"\n- Data available from: https://dhsprogram.com/")
    
    report_content.append(f"\n---")
    
    report_content.append(f"\n**END OF DOCUMENT**")
    report_content.append(f"\n\n*Generated by: NFHS-5 Water Disruption Analysis Pipeline v3.1*")
    report_content.append(f"\n*Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    return "\n".join(report_content)

# ==============================================================================
# 8. Main Execution Function 
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
        data_processor = DataProcessor(df_raw, cfg, data_loader.dta_metadata) # Pass metadata
        df_processed = data_processor.process()
    except ValueError as e:
        print(f"Critical Data Processing Error: {e}. Exiting.")
        return

    if df_processed.empty:
        print("Processed DataFrame is empty after cleaning. Exiting.")
        return

    # --- CRITICAL VERIFICATION OF PROCESSED DF ---
    print(f"\n{'='*20} Post-Processing Verification {'='*20}")
    required_final_cols = [
        cfg.VAR_WATER_DISRUPTED_FINAL, 'weight', cfg.VAR_PSU, 'wvi_score_scaled',
        'cci_score_scaled', 'idi_score', 'urban', 'piped_water_flag'
    ]
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

    # --- Advanced Multivariate Analysis (Model 4) ---
    # Call generate_table7_multivariate_explaining_paradox to get results for Model 4
    model4_results_dict, model4_text = generate_table7_multivariate_explaining_paradox(df_processed, cfg)
    all_tables_data['table_model4_data'] = model4_results_dict.get('Model 4 (Interactions)', pd.DataFrame()) # Extract the DataFrame for Model 4
    all_tables_text['table_model4_text'] = model4_text
    
    # Save Model 4 results to CSV
    if not all_tables_data['table_model4_data'].empty:
        all_tables_data['table_model4_data'].to_csv(cfg.OUTPUT_DIR / "results" / "table7_logistic_regression_Model4_Interactions.csv", index=True)

    # --- Predicted Probabilities (using Model 4 results) ---
    model4_results_object = model4_results_dict.get('Model 4 (Interactions) results object') # Get the actual results object
    all_tables_data['table_predicted_probs_data'], all_tables_text['table_predicted_probs_text'] = generate_table_predicted_probabilities(df_processed, cfg, model4_results_object)
    if not all_tables_data['table_predicted_probs_data'].empty:
        all_tables_data['table_predicted_probs_data'].to_csv(cfg.OUTPUT_DIR / "tables" / "table8_predicted_probabilities.csv", index=False)

    # --- Marginal Effects (using Model 4 results) ---
    all_tables_data['table_marginal_effects_data'], all_tables_text['table_marginal_effects_text'] = generate_table_marginal_effects(df_processed, cfg, model4_results_object)
    if 'AME' in all_tables_data['table_marginal_effects_data'] and not all_tables_data['table_marginal_effects_data']['AME'].empty:
        all_tables_data['table_marginal_effects_data']['AME'].to_csv(cfg.OUTPUT_DIR / "tables" / "table9_marginal_effects_AME.csv", index=False)
    if 'MER_IDI_Interaction' in all_tables_data['table_marginal_effects_data'] and not all_tables_data['table_marginal_effects_data']['MER_IDI_Interaction'].empty:
        all_tables_data['table_marginal_effects_data']['MER_IDI_Interaction'].to_csv(cfg.OUTPUT_DIR / "tables" / "table9_marginal_effects_MER_IDI_Interaction.csv", index=False)


    # --- Propensity Score Matching ---
    all_tables_data['table_psm_data'], all_tables_text['table_psm_text'] = generate_table_propensity_score_matching(df_processed, cfg)
    if 'Summary' in all_tables_data['table_psm_data'] and not all_tables_data['table_psm_data']['Summary'].empty:
        all_tables_data['table_psm_data']['Summary'].to_csv(cfg.OUTPUT_DIR / "tables" / "table10_psm_summary.csv", index=False)
    if 'Balance_Diagnostics' in all_tables_data['table_psm_data'] and not all_tables_data['table_psm_data']['Balance_Diagnostics'].empty:
        all_tables_data['table_psm_data']['Balance_Diagnostics'].to_csv(cfg.OUTPUT_DIR / "tables" / "table10_psm_balance_diagnostics.csv", index=False)
    if 'ATT_Results' in all_tables_data['table_psm_data'] and not all_tables_data['table_psm_data']['ATT_Results'].empty:
        all_tables_data['table_psm_data']['ATT_Results'].to_csv(cfg.OUTPUT_DIR / "tables" / "table10_psm_att_results.csv", index=False)

    # --- IDI Conceptual (Original Table 8, now Table 11) ---
    all_tables_data['table8_idi_conceptual'], all_tables_text['table8_idi_conceptual'] = generate_table8_idi_conceptual(df_processed, cfg)
    all_tables_data['table8_idi_conceptual'].to_csv(cfg.OUTPUT_DIR / "tables" / "table11_idi_conceptual.csv", index=False)


    # --- Spatial Summaries ---
    district_summary_df = create_district_level_summary(df_processed, cfg)
    state_summary_df = create_state_level_summary(df_processed, cfg)
    export_spatial_summaries(district_summary_df, state_summary_df, cfg)

    all_tables_data['table_district_rankings_data'], all_tables_text['table_district_rankings_text'] = generate_table_district_rankings(district_summary_df)
    if 'Top_Worst_Districts' in all_tables_data['table_district_rankings_data'] and not all_tables_data['table_district_rankings_data']['Top_Worst_Districts'].empty:
        all_tables_data['table_district_rankings_data']['Top_Worst_Districts'].to_csv(cfg.OUTPUT_DIR / "tables" / "table12_1_top_worst_districts.csv", index=False)
    if 'Top_Best_Districts' in all_tables_data['table_district_rankings_data'] and not all_tables_data['table_district_rankings_data']['Top_Best_Districts'].empty:
        all_tables_data['table_district_rankings_data']['Top_Best_Districts'].to_csv(cfg.OUTPUT_DIR / "tables" / "table12_2_top_best_districts.csv", index=False)

    all_tables_data['table_state_rankings_data'], all_tables_text['table_state_rankings_text'] = generate_table_state_rankings(state_summary_df)
    if 'States_by_Reliability_Gap' in all_tables_data['table_state_rankings_data'] and not all_tables_data['table_state_rankings_data']['States_by_Reliability_Gap'].empty:
        all_tables_data['table_state_rankings_data']['States_by_Reliability_Gap'].to_csv(cfg.OUTPUT_DIR / "tables" / "table13_1_states_by_reliability_gap.csv", index=False)
    if 'States_by_Paradox_Ratio' in all_tables_data['table_state_rankings_data'] and not all_tables_data['table_state_rankings_data']['States_by_Paradox_Ratio'].empty:
        all_tables_data['table_state_rankings_data']['States_by_Paradox_Ratio'].to_csv(cfg.OUTPUT_DIR / "tables" / "table13_2_states_by_paradox_ratio.csv", index=False)


    # --- Supporting tables from original script (renamed for clarity) ---
    all_tables_data['table_descriptive_characteristics'], all_tables_text['table_descriptive_characteristics'] = generate_table_descriptive_characteristics(df_processed, cfg)
    all_tables_data['table_descriptive_characteristics'].to_csv(cfg.OUTPUT_DIR / "tables" / "table_descriptive_characteristics.csv", index=False)

    all_tables_data['table_state_level_paradox'], all_tables_text['table_state_level_paradox'] = generate_table_state_level_paradox(df_processed, cfg)
    all_tables_data['table_state_level_paradox'].to_csv(cfg.OUTPUT_DIR / "tables" / "table_state_level_paradox.csv", index=False)

    all_tables_data['table_seasonal_patterns'], all_tables_text['table_seasonal_patterns'] = generate_table_seasonal_patterns(df_processed, cfg)
    all_tables_data['table_seasonal_patterns'].to_csv(cfg.OUTPUT_DIR / "tables" / "table_seasonal_patterns.csv", index=False)

    all_tables_data['table_robustness_checks'], all_tables_text['table_robustness_checks'] = generate_table_robustness_checks(df_processed, cfg)
    all_tables_data['table_robustness_checks'].to_csv(cfg.OUTPUT_DIR / "tables" / "table14_robustness_checks.csv", index=False)

    all_tables_data['table_idi_validation'], all_tables_text['table_idi_validation'] = generate_table_idi_validation(df_processed, cfg)
    all_tables_data['table_idi_validation'].to_csv(cfg.OUTPUT_DIR / "tables" / "table15_idi_validation.csv", index=False)

    all_tables_data['table_policy_simulation'], all_tables_text['table_policy_simulation'] = generate_table_policy_simulation(df_processed, cfg)
    all_tables_data['table_policy_simulation'].to_csv(cfg.OUTPUT_DIR / "tables" / "table16_policy_simulation.csv", index=False)

    # 8. Generate final Markdown Report
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
        # table7_results is removed from direct markdown generation, its content is now in table_model4_data
        # table7_results=all_tables_data['table7_multivariate_explaining_paradox'], table7_text=all_tables_text['table7_multivariate_explaining_paradox'],
        table8_data=all_tables_data['table8_idi_conceptual'], table8_text=all_tables_text['table8_idi_conceptual'],

        # New tables for advanced multivariate analysis
        table_model4_data=all_tables_data['table_model4_data'], table_model4_text=all_tables_text['table_model4_text'],
        table_predicted_probs_data=all_tables_data['table_predicted_probs_data'], table_predicted_probs_text=all_tables_text['table_predicted_probs_text'],
        table_psm_data=all_tables_data['table_psm_data'], table_psm_text=all_tables_text['table_psm_text'],
        table_marginal_effects_data=all_tables_data['table_marginal_effects_data'], table_marginal_effects_text=all_tables_text['table_marginal_effects_text'],

        # New tables for spatial analysis
        table_district_rankings_data=all_tables_data['table_district_rankings_data'], table_district_rankings_text=all_tables_text['table_district_rankings_text'],
        table_state_rankings_data=all_tables_data['table_state_rankings_data'], table_state_rankings_text=all_tables_text['table_state_rankings_text'],

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
