import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# File paths - update these with your actual paths
SHAPEFILE_PATH = "/Users/kumar.aniket/Downloads/surveydata/shapefile/district-shape/DISTRICT_BOUNDARY.shp"  # Update this
CSV_PATH = "/Users/kumar.aniket/Downloads/surveydata/water-distrupt/nfhs5_analysis_output_discovery/tables/district_level_summary_for_mapping.csv"  # UPDATE THIS PATH to your water disruption CSV

def fix_corrupted_text(text):
    """Fix common character encoding issues in district names"""
    if pd.isna(text):
        return text
    replacements = {
        '>': 'A',
        '@': 'U',
        '|': 'I',
    }
    fixed = str(text)
    for old, new in replacements.items():
        fixed = fixed.replace(old, new)
    return fixed

def create_comprehensive_district_mapping():
    """Create comprehensive mapping including all unmatched districts"""
    
    mapping = {
        # Corrupted text fixes
        'AGRA': 'AGRA',
        'ALAPPUZHA': 'ALAPPUZHA',
        'AIZAWL': 'AIZAWL',
        'AHMADABAD': 'AHMEDABAD',
        'AHAMADNAGAR': 'AHMADNAGAR',
        'ALIGARH': 'ALIGARH',
        'AMARAVATI': 'AMRAVATI',
        'AMBALA': 'AMBALA',
        'AMBEDKARNAGAR': 'AMBEDKAR NAGAR',
        'ANANTAPUR': 'ANANTHAPUR',
        'ANANTNAG': 'ANANTNAG',
        'ARARIA': 'ARARIA',
        'ARAVALLI': 'ARAVALI',
        'ARIYALUR': 'ARIYALUR',
        'ASHOKNAGAR': 'ASHOK NAGAR',
        'AURANGABAD': 'AURANGABAD',
        
        # Different naming conventions
        'BALASORE (BALESHWAR)': 'BALESHWAR',
        'BALODA BAZAR': 'BALODABAZAR',
        'BANAS KANTHA': 'BANASKANTHA',
        'BANDIPURA': 'BANDIPORE',
        'BARAGARH': 'BARGARH',
        'BAUDH (BAUDA)': 'BAUDH',
        'BELAGAVI': 'BELGAUM',
        'BALLARI': 'BELLARY',
        'BENGALURU RURAL': 'BENGALURU RURAL',
        'BENGALURU URBAN': 'BENGALURU URBAN',
        'BHADRADRI KOTHAGUDEM': 'B.KOTHAGUDEM',
        'BIJAPUR': 'BIJAPUR',
        'BILASPUR': 'BILASPUR',
        'BOLANGIR (BALANGIR)': 'BALANGIR',
        'BULANDSHAHR': 'BULANDSHAHAR',
        
        # NCT Delhi districts
        'SHAHADRA': 'SHAHDARA',
        
        # Special cases
        'DAKSHIN BASTAR DANTEWADA': 'DANTEWADA',
        'DAKSHIN DINAJPUR': 'DAKSHIN DINAJPUR',
        'DAKSHINA  KANNADA': 'DAKSHINA KANNADA',
        'DAKSHINA KANNADA': 'DAKSHINA KANNADA',
        'DEVBHUMI DWARKA': 'DEVBHUMI DWARKA',
        'DIBANG VALLEY': 'DIBANG VALLEY',
        'DIMA HASAO': 'DIMA HASAO',
        
        # East/West variations
        'EAST GARO HILLS': 'EAST GARO HILLS',
        'EAST JAINTIA HILLS': 'EAST JANTIA HILLS',
        'EAST KHASI HILLS': 'EAST KHASI HILLS',
        'EAST SINGHBHUM': 'E. SINGHBHUM',
        
        # More mappings
        'GAUTAMBUDHNAGAR': 'G.B. NAGAR',
        'GAZIPUR': 'GHAZIPUR',
        'GURUGRAM': 'GURUGRAM',
        'HUGLI': 'HOOGHLY',
        'JANJGIR - CHAMPA': 'JANJGIR CHAMPA',
        'JAYASHANKAR BHUPALAPALLY': 'J.BHUPALAPALLY',
        'JOGULAMBA GADWAL': 'JOGULAMBA',
        'KABIRDHAM': 'KAWARDHA',
        'KALABURAGI': 'GULBARGA',
        'KANPUR': 'KANPUR NAGAR',
        'KHERI': 'LAKHIMPUR KHERI',
        'KOCH BIHAR': 'COOCH BEHAR',
        'KONDAGAON': 'KODAGAON',
        'KOREA': 'KORIYA',
        'KUMURAM BHEEM': 'KB ASIFABAD',
        'LEH': 'LEH(LADAKH)',
        'MARIGAON': 'MORIGAON',
        'MEDCHAL-MALKAJGIRI': 'MEDCHAL MALKANJGIRI',
        'MEWAT': 'MEWAT',
        'MUMBAI CITY': 'MUMBAI',
        'SUB URBAN MUMBAI': 'MUMBAI SUBURBAN',
        'MUZAFFARPUR': 'MUZZAFARPUR',
        'MYSURU': 'MYSORE',
        'NICOBAR': 'NICOBARS',
        'NIMACH': 'NEEMUCH',
        'NORTH TWENTY-FOUR PARGANAS': 'NORTH 24 PARGANAS',
        'PASCHIM BARDHAMAN': 'PASCHIM BARDHAMAN',
        'PASCHIM MEDINIPUR': 'PASCHIM MEDINIPUR',
        'PASHCHIMI CHAMPARAN': 'WEST CHAMPARAN',
        'PATTANAMTHITTA': 'PATHANAMTHITA',
        'PEDDAPALLI': 'PEDAPALLE',
        'POTTI SRIRAMULU NELLORE': 'NELLORE',
        'PRAKASAM': 'PRAKASHAM',
        'PURBA BARDHAMAN': 'PURBA BARDHAMAN',
        'PURBA MEDINIPUR': 'PURBA MEDINIPUR',
        'PURBI CHAMPARAN': 'PURBA CHAMPARAN',
        'RAIBEARELI': 'RAEBARELI',
        'RAJ NANDGAON': 'RAJNANDGAON',
        'RAJ SAMAND': 'RAJSAMAND',
        'RANJANNA SIRCILLA': 'R. SIRCILLA',
        'RI-BHOI': 'RIBHOI',
        'SAHIBGANJ': 'SAHEBGANJ',
        'SANTKABIRNAGAR': 'SANT KABIR NAGAR',
        'SARAIKELA-KHARSAWAN': 'SARAIKELA',
        'SAS NAGAR (SAHIBZADA AJIT SINGH NAGAR)': 'SAS NAGAR',
        'SAWAI MADHOPUR': 'SAWAI MADHOPUR',
        'SHAHID BHAGAT SINGH NAGAR': 'NAWANSHAHR',
        'SHEIKHPURA': 'SEIKHPURA',
        'SHIVAMOGGA': 'SHIMOGA',
        'SIBSAGAR': 'SIVASAGAR',
        'SIDDHARTHNAGAR': 'SIDDHARTH NAGAR',
        'SOUTH TWENTY-FOUR PARGANAS': 'SOUTH 24 PARGANAS',
        'SRI MUKTSAR SAHIB': 'MUKTSAR',
        'THIRUVANANTHAPURAM': 'TRIVANDRUM',
        'TIRUCHIRAPPALLI': 'TRICHY',
        'TUMAKURU': 'TUMKUR',
        'UNOKOTI': 'UNAKOTI',
        'UTTAR BASTAR KANKER': 'KANKER',
        'UTTAR DINAJPUR': 'UTTAR DINAJPUR',
        'UTTARA  KANNADA': 'UTTARA KANNADA',
        'UTTARA KANNADA': 'UTTARA KANNADA',
        'VIJAYAPURA': 'BIJAPUR',
        'WARANGAL (RURAL)': 'WARANGAL',
        'WARANGAL (URBAN)': 'WARANGAL',
        'WEST SINGHBHUM': 'W SINGHBHUM',
        'Y S R KADAPA': 'YSR KADAPA',
        'BHADOHI': 'SANT RAVIDAS NAGAR',
        # New mappings for unmatched districts
        'ALIPUR DUAR': 'JALPAIGURI',  # Alipurduar is part of Jalpaiguri
        'BID': 'BEED',
        'BULDHANA': 'BULDANA',
        'CHAMARAJANAGAR': 'CHAMARAJANAGARA',
        'CHENGALPATTU': 'KANCHEEPURAM',  # Chengalpattu was carved out of Kancheepuram
        'CHIK BALLAPUR': 'CHIKBALLAPUR',
        'DAHOD': 'DOHAD',
        'DANGS': 'DANG',
        'DAVANGERE': 'DAVANAGERE',
        'DEOGARH': 'DEBAGARH',
        'EAST NIMAR': 'KHANDWA',  # East Nimar is now Khandwa
        
        # Handle disputed areas - map to one of the districts
        'DISPUTED (ALIRAJPUR & DAHOD)': 'ALIRAJPUR',
        'DISPUTED (BARAN & SHEOPUR': 'BARAN',
        'DISPUTED (MANDSAUR & JHALAWAR)': 'MANDSAUR',
        'DISPUTED (NIMACH & CHITTAURGARH )': 'NEEMUCH',
        'DISPUTED (RATLAM & BANSWARA)': 'RATLAM',
        'DISPUTED (RATLAM & MANDSAUR)': 'RATLAM',
        'DISPUTED (SABAR KANTHA & SIROHI)': 'SABARKANTHA',
        'DISPUTED (SABAR KANTHA & UDAIPUR)': 'SABARKANTHA',
        'DISPUTED (SAHIBGANJ, MALDAH & KATIHAR)': 'SAHEBGANJ',
        
        # More comprehensive mappings
        'BALASORE (BALESHWAR)': 'BALESHWAR',
        'BALODA BAZAR': 'BALODABAZAR',
        'BANAS KANTHA': 'BANASKANTHA',
        'BANDIPURA': 'BANDIPORE',
        'BARAGARH': 'BARGARH',
        'BAUDH (BAUDA)': 'BAUDH',
        'BELAGAVI': 'BELGAUM',
        'BALLARI': 'BELLARY',
        'BENGALURU RURAL': 'BENGALURU RURAL',
        'BENGALURU URBAN': 'BENGALURU URBAN',
        'BHADRADRI KOTHAGUDEM': 'B.KOTHAGUDEM',
        'BIJAPUR': 'BIJAPUR',
        'BILASPUR': 'BILASPUR',
        'BOLANGIR (BALANGIR)': 'BALANGIR',
        'BULANDSHAHR': 'BULANDSHAHAR',
        
        # Additional mappings with special characters fixed
        'FARIDABAD': 'FARIDABAD',
        'FARIDKOT': 'FARIDKOT',
        'FATEHABAD': 'FATEHABAD',
        'FAZILKA': 'FAZILKA',
        'FIROZABAD': 'FIROZABAD',
        'FIROZPUR': 'FEROZEPUR',
        'GANDHINAGAR': 'GANDHINAGAR',
        'GANGANAGAR': 'GANGANAGAR',
        'GANJAM': 'GANJAM',
        'GARIYABAND': 'GARIYABANDH',
        'GAURELA-PENDRA-MARWAHI': 'KORBA',  # New district carved from Korba
        'GAUTAMBUDHNAGAR': 'G.B. NAGAR',
        'GAZIPUR': 'GHAZIPUR',
        'GHAZIABAD': 'GHAZIABAD',
        'GONDIA': 'GONDIYA',
        'GOPALGANJ': 'GOPALGANJ',
        'GURDASPUR': 'GURDASPUR',
        'GURUGRAM': 'GURUGRAM',
        'GIR SOMNATH': 'GIR SOMNATH',
        
        # More special character fixes
        'HAORA': 'HOWRAH',
        'HAPUR': 'HAPUR',
        'HATHRAS': 'HATHRAS',
        'HAVERI': 'HAVERI',
        'HAMIRPUR': 'HAMIRPUR',
        'HANUMANGARH': 'HANUMANGARH',
        'HARIDWAR': 'HARIDWAR',
        'HISAR': 'HISAR',
        'HOSHANGABAD': 'HOSHANGABAD',
        'HOSHIARPUR': 'HOSHIARPUR',
        'HUGLI': 'HOOGHLY',
        
        # J districts
        'JAJAPUR': 'JAJAPUR',
        'JALNA': 'JALNA',
        'JALOR': 'JALORE',
        'JAMNAGAR': 'JAMNAGAR',
        'JANJGIR - CHAMPA': 'JANJGIR CHAMPA',
        'JANJGIR-CHAMPA': 'JANJGIR CHAMPA',
        'JUNAGADH': 'JUNAGADH',
        'JAGATSINGHPUR': 'JAGATSINGHAPUR',
        'JAHANABAD': 'JEHANABAD',
        'JALPAIGURI': 'JALPAIGURI',
        'JAMUI': 'JAMUI',
        'JAYASHANKAR BHUPALAPALLY': 'J.BHUPALAPALLY',
        'JHALAWAR': 'JHALAWAR',
        'JHARGRAM': 'JHARSUGUDA',
        'JHARSUGUDA': 'JHARSUGUDA',
        'JHUNJHUNUN': 'JHUNJHUNU',
        'JIND': 'JIND',
        'JOGULAMBA GADWAL': 'JOGULAMBA',
        
        # K districts
        'KALIMPONG': 'DARJILING',  # Part of Darjeeling
        'KAMJONG': 'UKHRUL',  # Part of Ukhrul
        'KANCHIPURAM': 'KANCHEEPURAM',
        'KANGPOKPI': 'SENAPATI',  # Part of Senapati
        'KANGRA': 'KANGRA',
        'KARAIKAL': 'KARAIKAL',
        'KASARAGOD': 'KASARGOD',
        'KABIRDHAM': 'KAWARDHA',
        'KAKCHING': 'THOUBAL',  # Part of Thoubal
        'KALAHANDI': 'KALAHANDI',
        'KALABURAGI': 'GULBARGA',
        'KALLAKKURICHI': 'VILLUPURAM',  # New district from Villupuram
        'KAMLE': 'UPPER SUBANSIRI',  # Part of Upper Subansiri
        'KAMRUP METRO': 'KAMRUP METROPOLITAN',
        'KAMRUP RURAL': 'KAMRUP',
        'KANDHAMAL': 'KANDHAMAL',
        'KANNUR': 'KANNUR',
        'KANPUR': 'KANPUR NAGAR',
        'KANYAKUMARI': 'KANYAKUMARI',
        'KAPURTHALA': 'KAPURTHALA',
        'KARUR': 'KARUR',
        'KARNAL': 'KARNAL',
        'KATIHAR': 'KATIHAR',
        'KENDRAPARHA': 'KENDRAPARA',
        'KEONJHAR (KENDUJHAR)': 'KENDUJHAR',
        'KHERI': 'LAKHIMPUR KHERI',
        'KISHTWAR': 'KISHTWAR',
        'KOCH BIHAR': 'COOCH BEHAR',
        'KOLHAPUR': 'KOLHAPUR',
        'KOLKATA': 'KOLKATA',
        'KONDAGAON': 'KODAGAON',
        'KORAPUT': 'KORAPUT',
        'KOREA': 'KORIYA',
        'KULGAM': 'KULGAM',
        'KUMURAM BHEEM': 'KB ASIFABAD',
        
        # L districts
        'LAHUL & SPITI': 'LAHUL & SPITI',
        'LATUR': 'LATUR',
        'LAKHISARAI': 'LAKHISARAI',
        'LEH': 'LEH(LADAKH)',
        'LEPA RADA': 'LOWER DIBANG VALLEY',
        'LOWER DIBANG VALLEY': 'LOWER DIBANG VALLEY',
        'LOWER SIANG': 'SIANG',
        'LUDHIANA': 'LUDHIANA',
        
        # M districts
        'MALDAH': 'MALDA',
        'MAMIT': 'MAMIT',
        'MANSA': 'MANSA',
        'MAHASAMUND': 'MAHASAMUND',
        'MAHESANA': 'MAHESANA',
        'MARIGAON': 'MORIGAON',
        'MAYURBHANJ': 'MAYURBHANJ',
        'MEDCHAL-MALKAJGIRI': 'MEDCHAL MALKANJGIRI',
        'MEWAT': 'MEWAT',
        'MORADABAD': 'MORADABAD',
        'MULUGU': 'WARANGAL',  # Part of Warangal
        'MUMBAI CITY': 'MUMBAI',
        'SUB URBAN MUMBAI': 'MUMBAI SUBURBAN',
        'MURSHIDABAD': 'MURSHIDABAD',
        'MUZAFFARABAD': 'MUZAFFARNAGAR',
        'MUZAFFARPUR': 'MUZZAFARPUR',
        'MYSURU': 'MYSORE',
        'MIRPUR': 'MIRZAPUR',
        
        # N districts
        'NAGAPATTINAM': 'NAGAPATTINAM',
        'NAGAUR': 'NAGAUR',
        'NAGPUR': 'NAGPUR',
        'NALANDA': 'NALANDA',
        'NAMAKKAL': 'NAMAKKAL',
        'NANDED': 'NANDED',
        'NARAINPUR': 'NARAYANPUR',
        'NASHIK': 'NASHIK',
        'NUAPARHA': 'NUAPADA',
        'NAINITAL': 'NAINITAL',
        'NANDURBAR': 'NANDURBAR',
        'NARAYANPET': 'MAHABUBNAGAR',  # Part of Mahabubnagar
        'NARMADA': 'NARMADA',
        'NARSHIMAPURA': 'NARSINGHPUR',
        'NAVSARI': 'NAVSARI',
        'NAWADA': 'NAWADA',
        'NAYAGARH': 'NAYAGARH',
        'NICOBAR': 'NICOBARS',
        'NIMACH': 'NEEMUCH',
        'NIVARI': 'TIKAMGARH',  # New district from Tikamgarh
        'NONEI': 'TAMENGLONG',  # Part of Tamenglong
        'NORTH GARO HILLS': 'NORTH GARO HILLS',
        'NORTH TWENTY-FOUR PARGANAS': 'NORTH 24 PARGANAS',
        'NILGIRIS': 'NILGIRIS',
        
        # O-P districts
        'OSMANABAD': 'OSMANABAD',
        'PALAKKAD': 'PALAKKAD',
        'PALGHAR': 'PALGHAR',
        'PALI': 'PALI',
        'PANCH MAHALS': 'PANCHMAHAL',
        'PANCHMAHAL': 'PANCHMAHAL',
        'PANIPAT': 'PANIPAT',
        'PATAN': 'PATAN',
        'PUNCH': 'PUNCH',
        'PURBI CHAMPARAN': 'PURBA CHAMPARAN',
        'PURNIA': 'PURNEA',
        'PAKKE KESSANG': 'EAST KAMENG',  # Part of East Kameng
        'PAREN': 'PEREN',
        'PASCHIM BARDHAMAN': 'PASCHIM BARDHAMAN',
        'PASCHIM MEDINIPUR': 'PASCHIM MEDINIPUR',
        'PASHCHIMI CHAMPARAN': 'WEST CHAMPARAN',
        'PATHANKOT': 'PATHANKOT',
        'PATIALA': 'PATIALA',
        'PATTANAMTHITTA': 'PATHANAMTHITA',
        'PAURI GARHWAL': 'PAURI GARHWAL',
        'PEDDAPALLI': 'PEDAPALLE',
        'PERAMBALUR': 'PERAMBALUR',
        'PHERZAWL': 'CHURACHANDPUR',  # Part of Churachandpur
        'PITHORAGARH': 'PITHORAGARH',
        'POTTI SRIRAMULU NELLORE': 'NELLORE',
        'PRAKASAM': 'PRAKASHAM',
        'PRATAPGARH': 'PRATAPGARH',
        'PURBA BARDHAMAN': 'PURBA BARDHAMAN',
        'PURBA MEDINIPUR': 'PURBA MEDINIPUR',
        'PURULIYA': 'PURULIA',
        'PILIBHIT': 'PILIBHIT',
        
        # R districts
        'RAICHUR': 'RAICHUR',
        'RAJ NANDGAON': 'RAJNANDGAON',
        'RAJ SAMAND': 'RAJSAMAND',
        'RAJAURI': 'RAJOURI',
        'RAJKOT': 'RAJKOT',
        'RAMANATHAPURAM': 'RAMANATHAPURAM',
        'RAMANAGARAM': 'RAMANAGARA',
        'RAMBAN': 'RAMBAN',
        'RAMPUR': 'RAMPUR',
        'RANIPPETTAI': 'VELLORE',  # New district from Vellore
        'RAYAGARHA': 'RAYAGADA',
        'RAYGAD': 'RAIGARH',
        'RUPNAGAR': 'RUPNAGAR',
        'RAIBEARELI': 'RAEBARELI',
        'RANJANNA SIRCILLA': 'R. SIRCILLA',
        'RATNAGIRI': 'RATNAGIRI',
        'REWARI': 'REWARI',
        'RI-BHOI': 'RIBHOI',
        'RIASI': 'REASI',
        'ROHTAS': 'ROHTAS',
        'RUDRAPRAYAG': 'RUDRAPRAYAG',
        
        # S districts
        'SABAR KANTHA': 'SABARKANTHA',
        'SAMBA': 'SAMBA',
        'SANGLI': 'SANGLI',
        'SARAN': 'SARAN',
        'SATARA': 'SATARA',
        'SURAJPUR': 'SURAJPUR',
        'SURAT': 'SURAT',
        'SAHARANPUR': 'SAHARANPUR',
        'SAHIBGANJ': 'SAHEBGANJ',
        'SAMASTIPUR': 'SAMASTIPUR',
        'SANGRUR': 'SANGRUR',
        'SANTKABIRNAGAR': 'SANT KABIR NAGAR',
        'SARAIKELA-KHARSAWAN': 'SARAIKELA',
        'SAS NAGAR (SAHIBZADA AJIT SINGH NAGAR)': 'SAS NAGAR',
        'SAWAI MADHOPUR': 'SAWAI MADHOPUR',
        'SENAPATI': 'SENAPATI',
        'SERCHHIP': 'SERCHHIP',
        'SHAHJAHANPUR': 'SHAHJAHANPUR',
        'SHAJAPUR': 'SHAJAPUR',
        'SHAMLI': 'SHAMLI',
        'SHAHADRA': 'SHAHDARA',
        'SHAHID BHAGAT SINGH NAGAR': 'NAWANSHAHR',
        'SHEIKHPURA': 'SEIKHPURA',
        'SHI YOMI': 'LONGDING',  # Part of Longding
        'SHIVAMOGGA': 'SHIMOGA',
        'SHRAWASTI': 'SHRAWASTI',
        'SHUPIYAN': 'SHUPIYAN',
        'SIBSAGAR': 'SIVASAGAR',
        'SIDDHARTHNAGAR': 'SIDDHARTH NAGAR',
        'SIWAN': 'SIWAN',
        'SOLAPUR': 'SOLAPUR',
        'SONIPAT': 'SONIPAT',
        'SOUTH GARO HILLS': 'SOUTH GARO HILLS',
        'SOUTH TWENTY-FOUR PARGANAS': 'SOUTH 24 PARGANAS',
        'SOUTH WEST GARO HILLS': 'SOUTH WEST GARO HILLS',
        'SOUTH WEST KHASI HILLS': 'SOUTH WEST KHASI HILLS',
        'SRI MUKTSAR SAHIB': 'MUKTSAR',
        'SRINAGAR': 'SRINAGAR',
        'SIKAR': 'SIKAR',
        'SITAMARHI': 'SITAMARHI',
        'SITAPUR': 'SITAPUR',
        
        # T districts
        'TAPI': 'TAPI',
        'TARN TARAN': 'TARN TARAN',
        'TEHRI GARHWAL': 'TEHRI GARHWAL',
        'TENGNOUPAL': 'CHANDEL',  # Part of Chandel
        'TENI': 'THENI',
        'TENKASI': 'TIRUNELVELI',  # New district from Tirunelveli
        'THANE': 'THANE',
        'THANJAVUR': 'THANJAVUR',
        'THIRUVARUR': 'TIRUVARUR',
        'THIRUVANANTHAPURAM': 'TRIVANDRUM',
        'THOUBAL': 'THOUBAL',
        'TIRAP': 'TIRAP',
        'TIRUCHIRAPPALLI': 'TRICHY',
        'TIRUPPUR': 'TIRUPPUR',
        'TIRUPPATTUR': 'VELLORE',  # Part of Vellore
        'TIRUVALLUR': 'THIRUVALLUR',
        'TIRUVANNAMALAI': 'THIRUVANNAMALAI',
        'TRISSUR': 'THRISSUR',
        'TUMAKURU': 'TUMKUR',
        
        # U-V districts
        'UNOKOTI': 'UNAKOTI',
        'USMANABAD': 'OSMANABAD',
        'UTTAR BASTAR KANKER': 'KANKER',
        'UTTAR DINAJPUR': 'UTTAR DINAJPUR',
        'UTTARA KANNADA': 'UTTARA KANNADA',
        'UTTARKASHI': 'UTTARKASHI',
        'VAISHALI': 'VAISHALI',
        'VALSAD': 'VALSAD',
        'VIJAYAPURA': 'BIJAPUR',
        
        # W-Y districts
        'WASHIM': 'WASHIM',
        'WARANGAL (RURAL)': 'WARANGAL',
        'WARANGAL (URBAN)': 'WARANGAL',
        'WEST GARO HILLS': 'WEST GARO HILLS',
        'WEST KHASI HILLS': 'WEST KHASI HILLS',
        'WEST NIMAR': 'KHARGONE',  # West Nimar is now Khargone
        'WEST SINGHBHUM': 'W SINGHBHUM',
        'Y S R KADAPA': 'YSR KADAPA',
        'YADGIR': 'YADGIR',
        'YANAM': 'YANAM',
        'YAMUNANAGAR': 'YAMUNANAGAR',
        'YAVATMAL': 'YAVATMAL',
        
        # Handle BHADOHI
        'BHADOHI': 'SANT RAVIDAS NAGAR',
        
        # Additional mappings for special characters
        'SHAHID BHAGAT SINGH NAGAR': 'NAWANSHAHR',
        'SAS NAGAR': 'SAS NAGAR',
        'JIRIBAM': 'IMPHAL EAST',  # Jiribam is a new district, might be counted with Imphal East in your data
        'PASCHIM BARDDHAMAN': 'PASCHIM BARDHAMAN',  # Note: BARDDHAMAN vs BARDHAMAN (extra D)
        'PURBA BARDDHAMAN': 'PURBA BARDHAMAN',  # Note: BARDDHAMAN vs BARDHAMAN (extra D)
        'TIRUCHIRAPALLI': 'TRICHY',  # Full name vs common abbreviation
        'TIRUCHIRAPPALLI': 'TRICHY',  # Alternative spelling
        
        # Also add these variations just in case:
        'PASCHIM BARDDHAM>N': 'PASCHIM BARDHAMAN',
        'PURBA BARDDHAM>N': 'PURBA BARDHAMAN',
        'TIRUCHIR>PALLI': 'TRICHY',
    }
    
    
    return mapping

def load_and_fix_shapefile(shapefile_path):
    """Load shapefile and fix encoding issues"""
    print("Loading shapefile...")
    gdf = gpd.read_file(shapefile_path)
    print(f"Shapefile columns: {gdf.columns.tolist()}")
    
    district_col = None
    for col in ['District', 'DISTRICT', 'district', 'NAME', 'dtname']:
        if col in gdf.columns:
            district_col = col
            break
    
    if district_col is None:
        district_col = gdf.columns[0]
    
    print(f"Using district column: {district_col}")
    
    gdf['District_Original'] = gdf[district_col].copy()
    gdf['District_Fixed'] = gdf[district_col].apply(fix_corrupted_text)
    
    mapping = create_comprehensive_district_mapping()
    
    def apply_mapping(name):
        if pd.isna(name):
            return name
        name_upper = str(name).upper().strip()
        if name_upper in mapping:
            return mapping[name_upper]
        fixed = fix_corrupted_text(name_upper)
        if fixed in mapping:
            return mapping[fixed]
        return name_upper
    
    gdf['District_Mapped'] = gdf['District_Fixed'].apply(apply_mapping)
    
    return gdf

def merge_with_csv(gdf, csv_path):
    """Merge shapefile with CSV data"""
    print("Loading CSV data...")
    df = pd.read_csv(csv_path)
    
    print(f"CSV columns: {df.columns.tolist()}")
    print(f"Sample districts from CSV: {df['district_name'].head().tolist()}")
    
    # Prepare district names for merging
    df['district_name_upper'] = df['district_name'].str.upper().str.strip()
    
    print("Merging data...")
    merged = gdf.merge(
        df, 
        left_on='District_Mapped', 
        right_on='district_name_upper', 
        how='left',
        suffixes=('_shp', '_csv')
    )
    
    total_shp = len(gdf)
    total_csv = len(df)
    matched = merged['district_name'].notna().sum()
    
    print(f"\nMerge Statistics:")
    print(f"Total districts in shapefile: {total_shp}")
    print(f"Total districts in CSV: {total_csv}")
    print(f"Successfully matched: {matched}")
    print(f"Match rate: {matched/total_shp*100:.1f}%")
    
    # Show unmatched districts from CSV
    unmatched_csv = df[~df['district_name_upper'].isin(gdf['District_Mapped'])]
    if len(unmatched_csv) > 0:
        print(f"\nUnmatched districts from CSV ({len(unmatched_csv)}):")
        for _, row in unmatched_csv.head(10).iterrows():
            print(f"  - {row['district_name']} ({row['state_name']})")
        if len(unmatched_csv) > 10:
            print(f"  ... and {len(unmatched_csv) - 10} more")
    
    return merged

def create_water_disruption_heatmaps(merged_gdf):
    """Create heatmaps for water disruption indicators"""
    if merged_gdf is None:
        print("Cannot create heatmaps: No data to plot")
        return
    
    # Define columns to plot with their properties
    columns_config = {
        'disruption_rate_pct': {
            'title': 'Water Disruption Rate',
            'cmap': 'Reds',  # Red for problems
            'unit': '%',
            'description': 'Higher values indicate more disruption'
        },
        'piped_water_coverage_pct': {
            'title': 'Piped Water Coverage',
            'cmap': 'Blues',  # Blue for water coverage
            'unit': '%',
            'description': 'Higher values indicate better coverage'
        },
        'reliability_gap': {
            'title': 'Reliability Gap',
            'cmap': 'RdYlGn_r',  # Red-Yellow-Green reversed (red for high gap)
            'unit': 'pp',
            'description': 'Higher values indicate worse reliability'
        },
        'mean_wvi_score': {
            'title': 'Water Vulnerability Index',
            'cmap': 'YlOrRd',  # Yellow-Orange-Red for vulnerability
            'unit': 'score',
            'description': 'Higher values indicate greater vulnerability'
        },
        'mean_cci_score': {
            'title': 'Coping Capacity Index',
            'cmap': 'Greens',  # Green for capacity
            'unit': 'score',
            'description': 'Higher values indicate better coping capacity'
        },
        'mean_idi_score': {
            'title': 'Infrastructure Dependency Index',
            'cmap': 'Purples',  # Purple for dependency
            'unit': 'score',
            'description': 'Higher values indicate greater dependency'
        }
    }
    
    # Check which columns are actually available
    available_columns = [col for col in columns_config.keys() if col in merged_gdf.columns]
    
    if len(available_columns) == 0:
        print("No data columns found for plotting!")
        return
    
    print(f"\nPlotting columns: {available_columns}")
    
    # Create figure with subplots
    n_cols = 2
    n_rows = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 18))
    axes = axes.flatten()
    
    for idx, column in enumerate(available_columns):
        ax = axes[idx]
        config = columns_config[column]
        
        # Plot base map (districts with no data in light gray)
        merged_gdf.plot(ax=ax, color='#f0f0f0', edgecolor='#666666', linewidth=0.3)
        
        # Plot data
        valid_data = merged_gdf[merged_gdf[column].notna()]
        
        if len(valid_data) > 0:
            # Get statistics
            vmin = valid_data[column].min()
            vmax = valid_data[column].max()
            mean_val = valid_data[column].mean()
            median_val = valid_data[column].median()
            
            # Plot with color
            valid_data.plot(
                column=column,
                ax=ax,
                legend=True,
                cmap=config['cmap'],
                edgecolor='#333333',
                linewidth=0.3,
                vmin=vmin,
                vmax=vmax,
                legend_kwds={
                    'label': f'{config["unit"]}',
                    'orientation': 'horizontal',
                    'shrink': 0.8,
                    'pad': 0.02,
                    'aspect': 20
                }
            )
            
            # Add title with statistics
            title = f'{config["title"]}\n'
            title += f'{config["description"]}\n'
            title += f'Mean: {mean_val:.1f} | Median: {median_val:.1f} | Range: {vmin:.1f}-{vmax:.1f}'
        else:
            title = f'{config["title"]}\n(No data available)'
        
        ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
        ax.axis('off')
    
    # Hide unused axes
    for idx in range(len(available_columns), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('District-wise Water Security Indicators Analysis\nNFHS-5 (2019-21)', 
                 fontsize=16, fontweight='bold', y=1.00)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig('water_disruption_heatmaps.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    print("\nHeatmap saved as 'water_disruption_heatmaps.png'")

def create_single_indicator_heatmap(merged_gdf, column, config, output_file=None):
    """Create a single heatmap for one indicator"""
    if column not in merged_gdf.columns:
        print(f"Column {column} not found in data")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Plot base map
    merged_gdf.plot(ax=ax, color='#f0f0f0', edgecolor='#666666', linewidth=0.3)
    
    # Plot data
    valid_data = merged_gdf[merged_gdf[column].notna()]
    
    if len(valid_data) > 0:
        vmin = valid_data[column].min()
        vmax = valid_data[column].max()
        mean_val = valid_data[column].mean()
        median_val = valid_data[column].median()
        std_val = valid_data[column].std()
        
        # Identify top districts
        top_districts = valid_data.nlargest(5, column)[['district_name', 'state_name', column]]
        
        im = valid_data.plot(
            column=column,
            ax=ax,
            legend=True,
            cmap=config['cmap'],
            edgecolor='#333333',
            linewidth=0.3,
            vmin=vmin,
            vmax=vmax,
            legend_kwds={
                'label': f'{config["unit"]}',
                'orientation': 'vertical',
                'shrink': 0.7,
                'pad': 0.02
            }
        )
        
        # Add title with comprehensive statistics
        title = f'{config["title"]}\n'
        title += f'{config["description"]}\n\n'
        title += f'Statistics: Mean={mean_val:.1f} | Median={median_val:.1f} | Std={std_val:.1f}\n'
        title += f'Range: {vmin:.1f} - {vmax:.1f} | Coverage: {len(valid_data)}/{len(merged_gdf)} districts'
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
        
        # Add text box with top districts
        # Position it in the upper right area (above Nepal/Tibet region)
        textstr = 'Top 5 Districts:\n'
        for idx, row in top_districts.iterrows():
            textstr += f"{row['district_name']}, {row['state_name']}: {row[column]:.1f}\n"
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        
        # Position the text box in the upper-right area (above Nepal)
        # Using relative coordinates: x=0.75 (75% from left), y=0.95 (95% from bottom)
        ax.text(0.55, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
    
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    plt.show()

def create_reliability_gap_focused_map(merged_gdf):
    """Create a focused map on reliability gap with annotations"""
    if 'reliability_gap' not in merged_gdf.columns:
        print("Reliability gap column not found")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Plot base map
    merged_gdf.plot(ax=ax, color='#f0f0f0', edgecolor='#666666', linewidth=0.3)
    
    # Plot reliability gap with diverging colormap
    valid_data = merged_gdf[merged_gdf['reliability_gap'].notna()]
    
    if len(valid_data) > 0:
        # Use diverging colormap centered at 0
        vmax = max(abs(valid_data['reliability_gap'].min()), 
                   abs(valid_data['reliability_gap'].max()))
        vmin = -vmax
        
        valid_data.plot(
            column='reliability_gap',
            ax=ax,
            legend=True,
            cmap='RdBu_r',  # Red for positive (bad), Blue for negative (good)
            edgecolor='#333333',
            linewidth=0.3,
            vmin=vmin,
            vmax=vmax,
            legend_kwds={
                'label': 'Reliability Gap (percentage points)',
                'orientation': 'horizontal',
                'shrink': 0.8,
                'pad': 0.02,
                'aspect': 30
            }
        )
        
        # Identify extreme districts
        worst_districts = valid_data.nlargest(10, 'reliability_gap')[['district_name', 'state_name', 'reliability_gap']]
        best_districts = valid_data.nsmallest(10, 'reliability_gap')[['district_name', 'state_name', 'reliability_gap']]
        
        # Title
        title = 'Water Infrastructure Reliability Gap\n'
        title += 'Difference between Actual and Expected Disruption Rates\n'
        title += '(Red = Worse than expected | Blue = Better than expected)'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add text boxes for worst and best districts
        # Position them in areas that don't overlap with India's map
        worst_text = 'Worst Performing Districts:\n'
        for idx, row in worst_districts.head(5).iterrows():
            worst_text += f"{row['district_name']}: +{row['reliability_gap']:.1f}pp\n"
        
        best_text = 'Best Performing Districts:\n'
        for idx, row in best_districts.head(5).iterrows():
            best_text += f"{row['district_name']}: {row['reliability_gap']:.1f}pp\n"
        
        props_worst = dict(boxstyle='round', facecolor='#ffcccc', alpha=0.9)
        props_best = dict(boxstyle='round', facecolor='#ccccff', alpha=0.9)
        
        # Position worst districts box in upper-right (above Nepal/Tibet)
        ax.text(0.80, 0.95, worst_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props_worst)
        
        # Position best districts box in upper-center-right (also above Nepal/Tibet but more centered)
        ax.text(0.52, 0.95, best_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props_best)
    
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('reliability_gap_focused.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    print("Reliability gap focused map saved as 'reliability_gap_focused.png'")

def create_single_indicator_heatmap_with_inset(merged_gdf, column, config, output_file=None):
    """Create a single heatmap with an inset for statistics"""
    if column not in merged_gdf.columns:
        print(f"Column {column} not found in data")
        return
    
    # Create figure with specific layout
    fig = plt.figure(figsize=(16, 12))
    
    # Main map axis
    ax_main = plt.axes([0.05, 0.05, 0.75, 0.85])  # [left, bottom, width, height]
    
    # Inset axis for statistics (positioned to the right of the map)
    ax_inset = plt.axes([0.62, 0.65, 0.15, 0.30])  # Upper right area
    
    # Plot base map on main axis
    merged_gdf.plot(ax=ax_main, color='#f0f0f0', edgecolor='#666666', linewidth=0.3)
    
    # Plot data
    valid_data = merged_gdf[merged_gdf[column].notna()]
    
    if len(valid_data) > 0:
        vmin = valid_data[column].min()
        vmax = valid_data[column].max()
        mean_val = valid_data[column].mean()
        median_val = valid_data[column].median()
        std_val = valid_data[column].std()
        
        # Identify top and bottom districts
        top_districts = valid_data.nlargest(5, column)[['district_name', 'state_name', column]]
        bottom_districts = valid_data.nsmallest(5, column)[['district_name', 'state_name', column]]
        
        # Plot on main axis
        im = valid_data.plot(
            column=column,
            ax=ax_main,
            legend=True,
            cmap=config['cmap'],
            edgecolor='#333333',
            linewidth=0.3,
            vmin=vmin,
            vmax=vmax,
            legend_kwds={
                'label': f'{config["unit"]}',
                'orientation': 'vertical',
                'shrink': 0.5,
                'pad': 0.02,
                'ax': ax_main,
                'anchor': (1.15, 0.5)
            }
        )
        
        # Add title
        title = f'{config["title"]}\n'
        title += f'{config["description"]}\n\n'
        title += f'Statistics: Mean={mean_val:.1f} | Median={median_val:.1f} | Std={std_val:.1f}\n'
        title += f'Range: {vmin:.1f} - {vmax:.1f} | Coverage: {len(valid_data)}/{len(merged_gdf)} districts'
        
        ax_main.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax_main.axis('off')
        
        # Create statistics table in inset
        ax_inset.axis('off')
        
        # Prepare text for inset
        stats_text = f'TOP 5 DISTRICTS\n{"="*25}\n'
        for idx, row in top_districts.iterrows():
            dist_name = row['district_name'][:15] + '...' if len(row['district_name']) > 15 else row['district_name']
            stats_text += f"{dist_name}: {row[column]:.1f}\n"
        
        stats_text += f'\nBOTTOM 5 DISTRICTS\n{"="*25}\n'
        for idx, row in bottom_districts.iterrows():
            dist_name = row['district_name'][:15] + '...' if len(row['district_name']) > 15 else row['district_name']
            stats_text += f"{dist_name}: {row[column]:.1f}\n"
        
        # Add text to inset with background
        ax_inset.text(0.5, 0.5, stats_text, transform=ax_inset.transAxes,
                     fontsize=8, ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    if output_file:
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    plt.show()

def generate_summary_statistics(merged_gdf):
    """Generate summary statistics for the water disruption indicators"""
    
    columns_to_analyze = [
        'disruption_rate_pct',
        'piped_water_coverage_pct',
        'reliability_gap',
        'mean_wvi_score',
        'mean_cci_score',
        'mean_idi_score'
    ]
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS FOR WATER DISRUPTION INDICATORS")
    print("="*60)
    
    for column in columns_to_analyze:
        if column in merged_gdf.columns:
            valid_data = merged_gdf[merged_gdf[column].notna()][column]
            
            if len(valid_data) > 0:
                print(f"\n{column.replace('_', ' ').title()}:")
                print(f"  Count: {len(valid_data)}")
                print(f"  Mean: {valid_data.mean():.2f}")
                print(f"  Median: {valid_data.median():.2f}")
                print(f"  Std Dev: {valid_data.std():.2f}")
                print(f"  Min: {valid_data.min():.2f}")
                print(f"  Max: {valid_data.max():.2f}")
                print(f"  25th percentile: {valid_data.quantile(0.25):.2f}")
                print(f"  75th percentile: {valid_data.quantile(0.75):.2f}")
    
    # Correlation analysis
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    correlation_data = merged_gdf[columns_to_analyze].dropna()
    if len(correlation_data) > 0:
        corr_matrix = correlation_data.corr()
        
        print("\nKey Correlations:")
        print(f"  Disruption Rate vs Piped Coverage: {corr_matrix.loc['disruption_rate_pct', 'piped_water_coverage_pct']:.3f}")
        print(f"  Disruption Rate vs WVI: {corr_matrix.loc['disruption_rate_pct', 'mean_wvi_score']:.3f}")
        print(f"  Disruption Rate vs CCI: {corr_matrix.loc['disruption_rate_pct', 'mean_cci_score']:.3f}")
        print(f"  Disruption Rate vs IDI: {corr_matrix.loc['disruption_rate_pct', 'mean_idi_score']:.3f}")
        print(f"  Reliability Gap vs IDI: {corr_matrix.loc['reliability_gap', 'mean_idi_score']:.3f}")

def main():
    try:
        # Load and fix shapefile
        gdf = load_and_fix_shapefile(SHAPEFILE_PATH)
        
        # Merge with CSV
        merged = merge_with_csv(gdf, CSV_PATH)
        
        if merged is not None:
            # Generate summary statistics
            generate_summary_statistics(merged)
            
            # Create combined heatmap figure
            print("\nCreating water disruption heatmaps...")
            create_water_disruption_heatmaps(merged)
            
            # Create reliability gap focused map
            print("\nCreating reliability gap focused map...")
            create_reliability_gap_focused_map(merged)
            
            # Create individual heatmaps
            create_individual = input("\nDo you want to create individual heatmaps for each indicator? (y/n): ")
            if create_individual.lower() == 'y':
                columns_config = {
                    'disruption_rate_pct': {
                        'title': 'Water Disruption Rate',
                        'cmap': 'Reds',
                        'unit': '%',
                        'description': 'Percentage of households experiencing water disruption'
                    },
                    'piped_water_coverage_pct': {
                        'title': 'Piped Water Coverage',
                        'cmap': 'Blues',
                        'unit': '%',
                        'description': 'Percentage of households with piped water access'
                    },
                    'reliability_gap': {
                        'title': 'Reliability Gap',
                        'cmap': 'RdYlGn_r',
                        'unit': 'pp',
                        'description': 'Difference between actual and expected disruption'
                    },
                    'mean_wvi_score': {
                        'title': 'Water Vulnerability Index',
                        'cmap': 'YlOrRd',
                        'unit': 'score',
                        'description': 'Traditional vulnerability to water insecurity'
                    },
                    'mean_cci_score': {
                        'title': 'Coping Capacity Index',
                        'cmap': 'Greens',
                        'unit': 'score',
                        'description': 'Household capacity to cope with disruptions'
                    },
                    'mean_idi_score': {
                        'title': 'Infrastructure Dependency Index',
                        'cmap': 'Purples',
                        'unit': 'score',
                        'description': 'Dependency on centralized water infrastructure'
                    }
                }
                
                for column, config in columns_config.items():
                    if column in merged.columns:
                        print(f"\nCreating individual heatmap for: {column}")
                        output_file = f'heatmap_{column}.png'
                        # Use the version with inset for individual maps
                        create_single_indicator_heatmap_with_inset(merged, column, config, output_file)
            
            # Save merged data for verification
            output_cols = ['District_Original', 'District_Mapped', 'district_name', 'state_name',
                          'disruption_rate_pct', 'piped_water_coverage_pct', 'reliability_gap',
                          'mean_wvi_score', 'mean_cci_score', 'mean_idi_score', 'district_typology']
            
            save_cols = [col for col in output_cols if col in merged.columns]
            merged[save_cols].to_csv('district_water_disruption_merged.csv', index=False)
            print("\nMerged data saved to 'district_water_disruption_merged.csv'")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
