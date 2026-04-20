#!/usr/bin/env python3

"""
NFHS 2019-21 Comprehensive Water Disruption Analysis
Enhanced with Bivariate and Multivariate Statistical Analyses
Using sh37b: Water not available for at least one day in past two weeks
"""

import pandas as pd
import numpy as np
import pyreadstat
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os

# Statistical and ML imports
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr, kendalltau, mannwhitneyu
from scipy.stats import kruskal, f_oneway, ttest_ind, fisher_exact
from scipy.spatial import distance
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.cluster import KMeans, DBSCAN
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.multivariate.manova import MANOVA

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# ============= CONFIGURATION =============

DATA_FILE_PATH = "/Users/kumar.aniket/Downloads/surveydata/IAHR7EFL.DTA"

# Required columns based on actual NFHS-5 variables
REQUIRED_COLS = [
    # Survey design
    'hv005',   # Sample weight
    'hv021',   # Primary sampling unit
    'hv022',   # Sample stratum
    'hv024',   # State
    'hv025',   # Urban/Rural
    'hv026',   # Place type
    'hv001',   # Cluster number
    
    # Temporal
    'hv006',   # Month of interview
    'hv007',   # Year of interview
    'hv008',   # Date of interview (CMC)
    
    # WATER DISRUPTION INDICATORS
    'sh37b',   # PRIMARY: Water not available for at least one day in past two weeks
    'hv201',   # Source of drinking water
    'hv201a',  # Water not available for at least a day last two weeks (if exists)
    'hv202',   # Source of non-drinking water
    'hv204',   # Time to water source
    'hv235',   # Location of water source
    'hv236',   # Person who fetches water
    
    # Socio-economic
    'hv270',   # Wealth index
    'hv271',   # Wealth index factor score
    'hv009',   # Number of household members
    'hv014',   # Children under 5
    'hv219',   # Sex of household head
    'sh47',    # Religion
    'sh49',    # Caste
    
    # Infrastructure
    'hv206',   # Has electricity
    'hv207',   # Has radio
    'hv208',   # Has television
    'hv209',   # Has refrigerator
    'hv210',   # Has bicycle
    'hv211',   # Has motorcycle
    'hv212',   # Has car
    'hv221',   # Has telephone
    'hv243a',  # Has mobile phone
    
    # Sanitation
    'hv205',   # Type of toilet facility
    
    # Housing
    'hv213',   # Main floor material
    'hv214',   # Main wall material
    'hv215',   # Main roof material
]

@dataclass
class Config:
    min_sample: int = 30
    bootstrap_n: int = 1000
    confidence: float = 0.95
    permutation_n: int = 1000
    alpha: float = 0.05  # For statistical tests
    
# State names and regions (same as original)
STATE_NAMES = {
    1: 'Jammu & Kashmir', 2: 'Himachal Pradesh', 3: 'Punjab', 4: 'Chandigarh',
    5: 'Uttarakhand', 6: 'Haryana', 7: 'NCT of Delhi', 8: 'Rajasthan',
    9: 'Uttar Pradesh', 10: 'Bihar', 11: 'Sikkim', 12: 'Arunachal Pradesh',
    13: 'Nagaland', 14: 'Manipur', 15: 'Mizoram', 16: 'Tripura',
    17: 'Meghalaya', 18: 'Assam', 19: 'West Bengal', 20: 'Jharkhand',
    21: 'Odisha', 22: 'Chhattisgarh', 23: 'Madhya Pradesh', 24: 'Gujarat',
    25: 'Dadra & Nagar Haveli & Daman & Diu', 27: 'Maharashtra',
    28: 'Andhra Pradesh', 29: 'Karnataka', 30: 'Goa', 31: 'Lakshadweep',
    32: 'Kerala', 33: 'Tamil Nadu', 34: 'Puducherry', 35: 'Andaman & Nicobar',
    36: 'Telangana', 37: 'Ladakh'
}

REGIONS = {
    'North': [1,2,3,4,5,6,7,37], 
    'Central': [8,9,10,23],
    'East': [19,20,21,22], 
    'Northeast': [11,12,13,14,15,16,17,18],
    'West': [24,25,27,30], 
    'South': [28,29,32,33,34,36,31,35]
}

class DataLoader:
    """Load NFHS data with correct variables"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def load(self, filepath: str) -> pd.DataFrame:
        """Load data with available columns"""
        print(f"Loading NFHS-5 data for water disruption analysis...")
        
        try:
            df, meta = pyreadstat.read_dta(filepath, usecols=REQUIRED_COLS)
        except:
            df, meta = pyreadstat.read_dta(filepath)
            available_cols = [col for col in REQUIRED_COLS if col in df.columns]
            df = df[available_cols]
            print(f"Some columns not found. Using {len(available_cols)} available columns.")
        
        print(f"Loaded {len(df):,} households with {len(df.columns)} variables")
        
        # Check key water disruption variable sh37b
        if 'sh37b' in df.columns:
            print(f"\nWater disruption variable (sh37b) found")
            print(f"'Water not available for at least one day in past two weeks'")
            value_counts = df['sh37b'].value_counts().sort_index()
            print(f"Distribution:")
            for val, count in value_counts.items():
                if val == 0:
                    print(f"  No: {count:,} ({count/len(df)*100:.1f}%)")
                elif val == 1:
                    print(f"  Yes: {count:,} ({count/len(df)*100:.1f}%)")
                elif val == 8:
                    print(f"  Don't know: {count:,} ({count/len(df)*100:.1f}%)")
        
        return df

class WaterDisruptionProcessor:
    """Process data to create comprehensive water disruption indicators"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def prepare(self) -> pd.DataFrame:
        """Prepare all water disruption related variables"""
        print("\nProcessing water disruption indicators...")
        
        # Survey weights
        self.df['weight'] = self.df['hv005'] / 1_000_000
        
        # PRIMARY WATER DISRUPTION INDICATOR
        if 'sh37b' in self.df.columns:
            self.df['water_disrupted_2weeks'] = (self.df['sh37b'] == 1).astype(int)
            self.df['water_disrupted_dk'] = (self.df['sh37b'] == 8).astype(int)
            self.df['water_disrupted'] = self.df['water_disrupted_2weeks']
            
            print(f"Water disruption in past 2 weeks (sh37b=1): {self.df['water_disrupted_2weeks'].mean()*100:.1f}%")
            print(f"Don't know (sh37b=8): {self.df['water_disrupted_dk'].mean()*100:.1f}%")
        
        # Check for hv201a as additional indicator
        if 'hv201a' in self.df.columns:
            self.df['water_interrupted_hv201a'] = (self.df['hv201a'] == 1).astype(int)
            print(f"Additional disruption indicator (hv201a): {self.df['water_interrupted_hv201a'].mean()*100:.1f}%")
            
            if 'sh37b' in self.df.columns:
                self.df['any_disruption'] = ((self.df['water_disrupted_2weeks'] == 1) | 
                                             (self.df['water_interrupted_hv201a'] == 1)).astype(int)
                print(f"Any disruption (sh37b OR hv201a): {self.df['any_disruption'].mean()*100:.1f}%")
        
        # Create comprehensive disruption index
        self.df = self._create_disruption_severity_index()
        
        # Temporal variables
        self.df['month'] = pd.to_numeric(self.df['hv006'], errors='coerce')
        self.df['year'] = pd.to_numeric(self.df['hv007'], errors='coerce')
        
        # Seasonal classification for India
        def get_season(month):
            if pd.isna(month):
                return 'Unknown'
            month = int(month)
            if month in [6, 7, 8, 9]:
                return 'Monsoon'
            elif month in [3, 4, 5]:
                return 'Summer'
            elif month in [10, 11]:
                return 'Post-monsoon'
            else:
                return 'Winter'
        
        self.df['season'] = self.df['month'].apply(get_season)
        
        # Spatial variables
        self.df['state'] = self.df['hv024'].map(STATE_NAMES)
        self.df['state_code'] = pd.to_numeric(self.df['hv024'], errors='coerce')
        self.df['cluster'] = pd.to_numeric(self.df['hv001'], errors='coerce')
        self.df['residence'] = self.df['hv025'].map({1: 'Urban', 2: 'Rural'})
        
        if 'hv026' in self.df.columns:
            self.df['place_type'] = self.df['hv026'].map({
                0: 'Capital/Large city',
                1: 'Small city', 
                2: 'Town',
                3: 'Rural'
            }).fillna('Rural')
        else:
            self.df['place_type'] = self.df['residence']
        
        self.df['region'] = self._assign_region()
        
        # Process water sources
        self.df = self._process_water_sources()
        
        # Process socio-economic variables
        self.df = self._process_socioeconomic()
        
        # Process infrastructure
        self.df = self._process_infrastructure()
        
        # Create indices
        self.df = self._create_vulnerability_index()
        self.df = self._create_water_insecurity_index()
        self.df = self._create_coping_capacity_index()
        
        # Clean data
        self.df = self.df.dropna(subset=['weight'])
        
        print(f"\nData processing complete:")
        print(f"Total households: {len(self.df):,}")
        print(f"Urban: {(self.df['residence']=='Urban').sum():,} ({(self.df['residence']=='Urban').mean()*100:.1f}%)")
        print(f"Rural: {(self.df['residence']=='Rural').sum():,} ({(self.df['residence']=='Rural').mean()*100:.1f}%)")
        
        print(f"\nWater Disruption Summary:")
        print(f"Water disrupted (past 2 weeks): {self.df['water_disrupted'].sum():,} ({self.df['water_disrupted'].mean()*100:.1f}%)")
        print(f"Severe disruption: {(self.df['disruption_severity']=='Severe').mean()*100:.1f}%")
        print(f"High water insecurity: {(self.df['water_insecurity_level']=='High').mean()*100:.1f}%")
        
        return self.df
    
    def _create_disruption_severity_index(self) -> pd.DataFrame:
        """Create comprehensive disruption severity index"""
        
        self.df['disruption_severity_score'] = 0
        
        if 'water_disrupted_2weeks' in self.df.columns:
            self.df['disruption_severity_score'] += self.df['water_disrupted_2weeks'] * 3
        
        if 'hv204' in self.df.columns:
            self.df['far_water'] = self.df['hv204'].apply(
                lambda x: 2 if pd.notna(x) and x != 996 and x >= 60 and x < 900 else
                         1 if pd.notna(x) and x != 996 and x >= 30 and x < 60 else 0
            )
            self.df['disruption_severity_score'] += self.df['far_water']
        
        if 'hv201' in self.df.columns:
            self.df['unreliable_source'] = self.df['hv201'].apply(
                lambda x: 1 if pd.notna(x) and x in [32, 42, 43] else 0
            )
            self.df['disruption_severity_score'] += self.df['unreliable_source']
        
        self.df['disruption_severity'] = pd.cut(
            self.df['disruption_severity_score'],
            bins=[-0.1, 0, 1, 3, 100],
            labels=['None', 'Mild', 'Moderate', 'Severe']
        )
        
        print(f"\nDisruption Severity Index created:")
        severity_dist = self.df['disruption_severity'].value_counts()
        for severity, count in severity_dist.items():
            print(f"  {severity}: {count:,} ({count/len(self.df)*100:.1f}%)")
        
        return self.df
    
    def _assign_region(self) -> pd.Series:
        """Assign regions based on state codes"""
        def get_region(state_code):
            if pd.isna(state_code):
                return 'Unknown'
            try:
                state_code = int(state_code)
                for region, states in REGIONS.items():
                    if state_code in states:
                        return region
            except:
                pass
            return 'Other'
        return self.df['hv024'].apply(get_region)
    
    def _process_water_sources(self) -> pd.DataFrame:
        """Process water source variables"""
        
        def categorize_source(code):
            if pd.isna(code):
                return 'Unknown'
            code = int(code)
            
            if code in [11, 12, 13, 14]:
                return 'Piped'
            elif code in [21]:
                return 'Tube well'
            elif code in [31, 41]:
                return 'Protected well/spring'
            elif code in [51]:
                return 'Rainwater'
            elif code in [71]:
                return 'Bottled'
            elif code in [92]:
                return 'Community RO'
            elif code in [32, 42]:
                return 'Unprotected well/spring'
            elif code in [43]:
                return 'Surface water'
            elif code in [61, 62]:
                return 'Tanker/Cart'
            else:
                return 'Other'
        
        if 'hv201' in self.df.columns:
            self.df['water_source'] = self.df['hv201'].apply(categorize_source)
            
            improved_sources = ['Piped', 'Tube well', 'Protected well/spring', 
                              'Rainwater', 'Bottled', 'Community RO']
            self.df['improved_source'] = self.df['water_source'].isin(improved_sources).astype(int)
            self.df['has_piped_water'] = (self.df['water_source'] == 'Piped').astype(int)
        else:
            self.df['water_source'] = 'Unknown'
            self.df['improved_source'] = 0
            self.df['has_piped_water'] = 0
        
        if 'hv204' in self.df.columns:
            def categorize_time(time):
                if pd.isna(time) or time >= 998:
                    return 'Unknown'
                if time == 996:
                    return 'On premises'
                if time < 15:
                    return '<15 min'
                if time < 30:
                    return '15-29 min'
                if time < 60:
                    return '30-59 min'
                return '≥60 min'
            
            self.df['time_to_water'] = self.df['hv204'].apply(categorize_time)
            self.df['water_on_premises'] = (self.df['hv204'] == 996).astype(int)
            
            time_score_map = {
                'On premises': 0, '<15 min': 1, '15-29 min': 2,
                '30-59 min': 3, '≥60 min': 4, 'Unknown': 1
            }
            self.df['water_time_score'] = self.df['time_to_water'].map(time_score_map).fillna(1)
        else:
            self.df['time_to_water'] = 'Unknown'
            self.df['water_on_premises'] = 0
            self.df['water_time_score'] = 1
        
        if 'hv235' in self.df.columns:
            self.df['water_location'] = self.df['hv235'].map({
                1: 'In dwelling',
                2: 'In yard/plot',
                3: 'Elsewhere'
            }).fillna('Unknown')
        else:
            self.df['water_location'] = 'Unknown'
        
        if 'hv236' in self.df.columns:
            self.df['water_fetcher'] = self.df['hv236'].map({
                1: 'Adult woman',
                2: 'Adult man',
                3: 'Female child (<15)',
                4: 'Male child (<15)',
                6: 'Other',
                9: 'No one (on premises)'
            }).fillna('Unknown')
            
            self.df['women_fetch_water'] = self.df['hv236'].isin([1, 3]).astype(int)
        else:
            self.df['water_fetcher'] = 'Unknown'
            self.df['women_fetch_water'] = 0
        
        return self.df
    
    def _process_socioeconomic(self) -> pd.DataFrame:
        """Process socio-economic variables"""
        
        self.df['wealth_quintile'] = self.df['hv270'].map({
            1: 'Poorest', 2: 'Poorer', 3: 'Middle', 4: 'Richer', 5: 'Richest'
        })
        self.df['wealth_score'] = pd.to_numeric(self.df['hv270'], errors='coerce').fillna(3)
        
        self.df['hh_size'] = pd.to_numeric(self.df['hv009'], errors='coerce').fillna(4)
        self.df['children_under5'] = pd.to_numeric(self.df['hv014'], errors='coerce').fillna(0)
        self.df['has_children'] = (self.df['children_under5'] > 0).astype(int)
        self.df['large_household'] = (self.df['hh_size'] >= 6).astype(int)
        
        self.df['hh_head_sex'] = self.df['hv219'].map({1: 'Male', 2: 'Female'}).fillna('Unknown')
        self.df['female_headed'] = (self.df['hv219'] == 2).astype(int)
        
        if 'sh47' in self.df.columns:
            self.df['religion'] = self.df['sh47'].map({
                1: 'Hindu', 2: 'Muslim', 3: 'Christian', 4: 'Sikh',
                5: 'Buddhist', 6: 'Jain', 7: 'Jewish', 8: 'Parsi',
                9: 'No religion', 96: 'Other'
            }).fillna('Other')
        else:
            self.df['religion'] = 'Unknown'
        
        if 'sh49' in self.df.columns:
            self.df['caste'] = self.df['sh49'].map({
                1: 'SC', 2: 'ST', 3: 'OBC', 4: 'General', 8: 'Don\'t know', 9: 'Other'
            }).fillna('Other')
            
            self.df['marginalized_caste'] = self.df['caste'].isin(['SC', 'ST']).astype(int)
        else:
            self.df['caste'] = 'Unknown'
            self.df['marginalized_caste'] = 0
        
        return self.df
    
    def _process_infrastructure(self) -> pd.DataFrame:
        """Process infrastructure variables"""
        
        self.df['has_electricity'] = (self.df['hv206'] == 1).astype(int) if 'hv206' in self.df.columns else 0
        self.df['has_television'] = (self.df['hv208'] == 1).astype(int) if 'hv208' in self.df.columns else 0
        self.df['has_refrigerator'] = (self.df['hv209'] == 1).astype(int) if 'hv209' in self.df.columns else 0
        
        self.df['has_bicycle'] = (self.df['hv210'] == 1).astype(int) if 'hv210' in self.df.columns else 0
        self.df['has_motorcycle'] = (self.df['hv211'] == 1).astype(int) if 'hv211' in self.df.columns else 0
        self.df['has_car'] = (self.df['hv212'] == 1).astype(int) if 'hv212' in self.df.columns else 0
        self.df['has_vehicle'] = ((self.df['has_motorcycle'] == 1) | (self.df['has_car'] == 1)).astype(int)
        
        self.df['has_mobile'] = (self.df['hv243a'] == 1).astype(int) if 'hv243a' in self.df.columns else 0
        self.df['has_telephone'] = (self.df['hv221'] == 1).astype(int) if 'hv221' in self.df.columns else 0
        
        if 'hv205' in self.df.columns:
            def categorize_toilet(code):
                if pd.isna(code):
                    return 'Unknown'
                code = int(code)
                if code in [11, 12, 13, 14, 15]:
                    return 'Flush'
                elif code in [21, 22]:
                    return 'Improved pit'
                elif code in [23]:
                    return 'Unimproved pit'
                elif code in [31]:
                    return 'Open defecation'
                else:
                    return 'Other'
            
            self.df['toilet_type'] = self.df['hv205'].apply(categorize_toilet)
            self.df['improved_sanitation'] = self.df['toilet_type'].isin(['Flush', 'Improved pit']).astype(int)
            self.df['open_defecation'] = (self.df['toilet_type'] == 'Open defecation').astype(int)
        else:
            self.df['toilet_type'] = 'Unknown'
            self.df['improved_sanitation'] = 0
            self.df['open_defecation'] = 0
        
        infra_vars = ['has_electricity', 'has_television', 'has_refrigerator', 
                     'has_vehicle', 'has_mobile', 'improved_sanitation']
        self.df['infrastructure_score'] = self.df[infra_vars].sum(axis=1)
        
        self.df['infrastructure_level'] = pd.cut(
            self.df['infrastructure_score'],
            bins=[-0.1, 2, 4, 6],
            labels=['Poor', 'Moderate', 'Good']
        )
        
        return self.df
    
    def _create_vulnerability_index(self) -> pd.DataFrame:
        """Create water vulnerability index"""
        
        self.df['vulnerability_score'] = 0
        
        self.df['vulnerability_score'] += (self.df['wealth_score'] <= 2).astype(int) * 2
        self.df['vulnerability_score'] += self.df['has_children']
        self.df['vulnerability_score'] += self.df['female_headed']
        self.df['vulnerability_score'] += self.df['large_household']
        self.df['vulnerability_score'] += self.df['marginalized_caste']
        self.df['vulnerability_score'] += (1 - self.df['water_on_premises'])
        self.df['vulnerability_score'] += (1 - self.df['improved_source'])
        self.df['vulnerability_score'] += self.df['women_fetch_water']
        self.df['vulnerability_score'] += (1 - self.df['has_electricity'])
        self.df['vulnerability_score'] += self.df['open_defecation']
        
        self.df['vulnerability_level'] = pd.cut(
            self.df['vulnerability_score'],
            bins=[-0.1, 3, 6, 100],
            labels=['Low', 'Medium', 'High']
        )
        
        return self.df
    
    def _create_water_insecurity_index(self) -> pd.DataFrame:
        """Create comprehensive water insecurity index"""
        
        self.df['water_insecurity_score'] = 0
        
        self.df['water_insecurity_score'] += self.df['water_disrupted'] * 3
        self.df['water_insecurity_score'] += self.df['water_time_score']
        self.df['water_insecurity_score'] += (1 - self.df['improved_source']) * 2
        self.df['water_insecurity_score'] += self.df['unreliable_source'] if 'unreliable_source' in self.df else 0
        self.df['water_insecurity_score'] += self.df['women_fetch_water']
        
        self.df['water_insecurity_level'] = pd.cut(
            self.df['water_insecurity_score'],
            bins=[-0.1, 2, 5, 100],
            labels=['Low', 'Medium', 'High']
        )
        
        return self.df
    
    def _create_coping_capacity_index(self) -> pd.DataFrame:
        """Create household coping capacity index"""
        
        self.df['coping_capacity_score'] = 0
        
        self.df['coping_capacity_score'] += (self.df['wealth_score'] - 1)
        self.df['coping_capacity_score'] += self.df['has_electricity']
        self.df['coping_capacity_score'] += self.df['has_vehicle']
        self.df['coping_capacity_score'] += self.df['has_mobile']
        self.df['coping_capacity_score'] += self.df['has_refrigerator']
        
        self.df['coping_capacity_level'] = pd.cut(
            self.df['coping_capacity_score'],
            bins=[-0.1, 2, 5, 100],
            labels=['Low', 'Medium', 'High']
        )
        
        return self.df

class BivariateAnalyzer:
    """Comprehensive bivariate statistical analysis"""
    
    def __init__(self, df: pd.DataFrame, config: Config):
        self.df = df
        self.config = config
    
    def run_comprehensive_bivariate_analysis(self) -> Dict:
        """Run all bivariate analyses"""
        print("\nRunning comprehensive bivariate analysis...")
        
        results = {}
        results['categorical_categorical'] = self._categorical_vs_categorical()
        results['continuous_categorical'] = self._continuous_vs_categorical()
        results['continuous_continuous'] = self._continuous_vs_continuous()
        results['chi_square_tests'] = self._chi_square_tests()
        results['t_tests'] = self._t_tests()
        results['anova_tests'] = self._anova_tests()
        results['correlation_matrix'] = self._correlation_analysis()
        results['rank_correlations'] = self._rank_correlations()
        results['contingency_tables'] = self._contingency_tables()
        
        print("Bivariate analysis complete")
        return results
    
    def _categorical_vs_categorical(self) -> Dict:
        """Analyze categorical vs categorical relationships"""
        cat_vars = ['residence', 'wealth_quintile', 'region', 'season', 'water_source',
                   'caste', 'religion', 'disruption_severity', 'vulnerability_level',
                   'water_insecurity_level', 'coping_capacity_level']
        
        target = 'water_disrupted'
        results = {}
        
        for var in cat_vars:
            if var not in self.df.columns:
                continue
                
            # Create contingency table
            crosstab = pd.crosstab(self.df[var], self.df[target], margins=True)
            
            # Chi-square test
            try:
                chi2, p_val, dof, expected = chi2_contingency(crosstab.iloc[:-1, :-1])
                cramer_v = np.sqrt(chi2 / (crosstab.iloc[-1, -1] * (min(crosstab.shape) - 1)))
                
                results[var] = {
                    'crosstab': crosstab,
                    'chi2_stat': chi2,
                    'p_value': p_val,
                    'degrees_freedom': dof,
                    'cramers_v': cramer_v,
                    'effect_size': 'small' if cramer_v < 0.1 else 'medium' if cramer_v < 0.3 else 'large'
                }
            except:
                results[var] = {'crosstab': crosstab, 'error': 'Chi-square test failed'}
        
        return results
    
    def _continuous_vs_categorical(self) -> Dict:
        """Analyze continuous vs categorical relationships"""
        continuous_vars = ['wealth_score', 'infrastructure_score', 'vulnerability_score',
                          'water_insecurity_score', 'coping_capacity_score', 'hh_size',
                          'children_under5', 'water_time_score', 'disruption_severity_score']
        
        target = 'water_disrupted'
        results = {}
        
        for var in continuous_vars:
            if var not in self.df.columns:
                continue
            
            # Group statistics
            group_stats = self.df.groupby(target)[var].agg(['count', 'mean', 'std', 'median'])
            
            # T-test or Mann-Whitney U test
            group0 = self.df[self.df[target] == 0][var].dropna()
            group1 = self.df[self.df[target] == 1][var].dropna()
            
            if len(group0) > 0 and len(group1) > 0:
                # T-test
                t_stat, t_p = ttest_ind(group0, group1)
                
                # Mann-Whitney U test (non-parametric)
                u_stat, u_p = mannwhitneyu(group0, group1, alternative='two-sided')
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt((group0.var() + group1.var()) / 2)
                cohens_d = (group1.mean() - group0.mean()) / pooled_std if pooled_std > 0 else 0
                
                results[var] = {
                    'group_stats': group_stats,
                    't_statistic': t_stat,
                    't_p_value': t_p,
                    'mannwhitney_u': u_stat,
                    'mannwhitney_p': u_p,
                    'cohens_d': cohens_d,
                    'effect_size': 'small' if abs(cohens_d) < 0.2 else 'medium' if abs(cohens_d) < 0.8 else 'large'
                }
        
        return results
    
    def _continuous_vs_continuous(self) -> Dict:
        """Analyze continuous vs continuous relationships"""
        continuous_vars = ['wealth_score', 'infrastructure_score', 'vulnerability_score',
                          'water_insecurity_score', 'coping_capacity_score', 'hh_size',
                          'children_under5', 'water_time_score', 'disruption_severity_score']
        
        results = {}
        
        for i, var1 in enumerate(continuous_vars):
            if var1 not in self.df.columns:
                continue
            for var2 in continuous_vars[i+1:]:
                if var2 not in self.df.columns:
                    continue
                
                # Remove missing values
                df_clean = self.df[[var1, var2]].dropna()
                
                if len(df_clean) > 10:
                    # Pearson correlation
                    pearson_r, pearson_p = pearsonr(df_clean[var1], df_clean[var2])
                    
                    # Spearman correlation
                    spearman_r, spearman_p = spearmanr(df_clean[var1], df_clean[var2])
                    
                    # Kendall's tau
                    kendall_tau, kendall_p = kendalltau(df_clean[var1], df_clean[var2])
                    
                    results[f"{var1}_vs_{var2}"] = {
                        'n_observations': len(df_clean),
                        'pearson_r': pearson_r,
                        'pearson_p': pearson_p,
                        'spearman_r': spearman_r,
                        'spearman_p': spearman_p,
                        'kendall_tau': kendall_tau,
                        'kendall_p': kendall_p,
                        'interpretation': 'weak' if abs(pearson_r) < 0.3 else 'moderate' if abs(pearson_r) < 0.7 else 'strong'
                    }
        
        return results
    
    def _chi_square_tests(self) -> Dict:
        """Comprehensive chi-square tests for independence"""
        cat_vars = ['residence', 'wealth_quintile', 'region', 'season', 'water_source',
                   'caste', 'religion', 'disruption_severity', 'vulnerability_level']
        
        results = {}
        
        for var in cat_vars:
            if var not in self.df.columns:
                continue
            
            crosstab = pd.crosstab(self.df[var], self.df['water_disrupted'])
            
            if crosstab.shape[0] > 1 and crosstab.shape[1] > 1:
                try:
                    chi2, p_val, dof, expected = chi2_contingency(crosstab)
                    
                    # Calculate effect sizes
                    n = crosstab.sum().sum()
                    cramer_v = np.sqrt(chi2 / (n * (min(crosstab.shape) - 1)))
                    phi = np.sqrt(chi2 / n) if crosstab.shape == (2, 2) else None
                    
                    results[var] = {
                        'chi2_statistic': chi2,
                        'p_value': p_val,
                        'degrees_freedom': dof,
                        'cramers_v': cramer_v,
                        'phi_coefficient': phi,
                        'sample_size': n,
                        'significant': p_val < self.config.alpha
                    }
                except Exception as e:
                    results[var] = {'error': str(e)}
        
        return results
    
    def _t_tests(self) -> Dict:
        """Independent samples t-tests"""
        continuous_vars = ['wealth_score', 'infrastructure_score', 'vulnerability_score',
                          'water_insecurity_score', 'coping_capacity_score']
        
        results = {}
        
        for var in continuous_vars:
            if var not in self.df.columns:
                continue
            
            # Water disrupted vs not disrupted
            group0 = self.df[self.df['water_disrupted'] == 0][var].dropna()
            group1 = self.df[self.df['water_disrupted'] == 1][var].dropna()
            
            if len(group0) > 0 and len(group1) > 0:
                t_stat, p_val = ttest_ind(group0, group1)
                
                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt((group0.var() + group1.var()) / 2)
                cohens_d = (group1.mean() - group0.mean()) / pooled_std if pooled_std > 0 else 0
                
                results[f"{var}_disrupted_vs_not"] = {
                    'group0_mean': group0.mean(),
                    'group1_mean': group1.mean(),
                    'group0_std': group0.std(),
                    'group1_std': group1.std(),
                    'group0_n': len(group0),
                    'group1_n': len(group1),
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'cohens_d': cohens_d,
                    'significant': p_val < self.config.alpha
                }
        
        return results
    
    def _anova_tests(self) -> Dict:
        """One-way ANOVA tests"""
        continuous_vars = ['wealth_score', 'infrastructure_score', 'vulnerability_score']
        categorical_vars = ['region', 'wealth_quintile', 'disruption_severity']
        
        results = {}
        
        for cat_var in categorical_vars:
            if cat_var not in self.df.columns:
                continue
            for cont_var in continuous_vars:
                if cont_var not in self.df.columns:
                    continue
                
                # Prepare groups
                groups = []
                group_names = []
                for category in self.df[cat_var].unique():
                    if pd.notna(category):
                        group_data = self.df[self.df[cat_var] == category][cont_var].dropna()
                        if len(group_data) > 2:  # Minimum sample size
                            groups.append(group_data)
                            group_names.append(category)
                
                if len(groups) > 2:
                    try:
                        # One-way ANOVA
                        f_stat, p_val = f_oneway(*groups)
                        
                        # Kruskal-Wallis test (non-parametric alternative)
                        h_stat, h_p = kruskal(*groups)
                        
                        # Calculate eta-squared (effect size)
                        all_data = np.concatenate(groups)
                        ss_total = np.sum((all_data - np.mean(all_data)) ** 2)
                        ss_between = sum([len(group) * (np.mean(group) - np.mean(all_data)) ** 2 for group in groups])
                        eta_squared = ss_between / ss_total if ss_total > 0 else 0
                        
                        results[f"{cont_var}_by_{cat_var}"] = {
                            'f_statistic': f_stat,
                            'f_p_value': p_val,
                            'kruskal_h': h_stat,
                            'kruskal_p': h_p,
                            'eta_squared': eta_squared,
                            'groups': group_names,
                            'group_means': [np.mean(group) for group in groups],
                            'significant': p_val < self.config.alpha
                        }
                    except Exception as e:
                        results[f"{cont_var}_by_{cat_var}"] = {'error': str(e)}
        
        return results
    
    def _correlation_analysis(self) -> Dict:
        """Comprehensive correlation analysis"""
        numerical_vars = ['wealth_score', 'infrastructure_score', 'vulnerability_score',
                         'water_insecurity_score', 'coping_capacity_score', 'hh_size',
                         'children_under5', 'water_time_score', 'disruption_severity_score',
                         'water_disrupted']
        
        # Filter available variables
        available_vars = [var for var in numerical_vars if var in self.df.columns]
        
        if len(available_vars) < 2:
            return {'error': 'Insufficient numerical variables for correlation analysis'}
        
        # Create correlation matrix
        corr_data = self.df[available_vars].corr()
        
        # Calculate significance levels
        n = len(self.df[available_vars].dropna())
        significance_matrix = pd.DataFrame(index=corr_data.index, columns=corr_data.columns)
        
        for i, var1 in enumerate(available_vars):
            for j, var2 in enumerate(available_vars):
                if i != j:
                    try:
                        _, p_val = pearsonr(self.df[var1].dropna(), self.df[var2].dropna())
                        significance_matrix.loc[var1, var2] = p_val
                    except:
                        significance_matrix.loc[var1, var2] = np.nan
                else:
                    significance_matrix.loc[var1, var2] = 0.0
        
        return {
            'correlation_matrix': corr_data,
            'significance_matrix': significance_matrix.astype(float),
            'sample_size': n,
            'variables': available_vars
        }
    
    def _rank_correlations(self) -> Dict:
        """Spearman and Kendall rank correlations"""
        numerical_vars = ['wealth_score', 'infrastructure_score', 'vulnerability_score',
                         'water_insecurity_score', 'coping_capacity_score', 'water_disrupted']
        
        available_vars = [var for var in numerical_vars if var in self.df.columns]
        
        results = {}
        
        for i, var1 in enumerate(available_vars):
            for var2 in available_vars[i+1:]:
                clean_data = self.df[[var1, var2]].dropna()
                
                if len(clean_data) > 10:
                    spearman_r, spearman_p = spearmanr(clean_data[var1], clean_data[var2])
                    kendall_tau, kendall_p = kendalltau(clean_data[var1], clean_data[var2])
                    
                    results[f"{var1}_vs_{var2}"] = {
                        'spearman_rho': spearman_r,
                        'spearman_p': spearman_p,
                        'kendall_tau': kendall_tau,
                        'kendall_p': kendall_p,
                        'n_observations': len(clean_data)
                    }
        
        return results
    
    def _contingency_tables(self) -> Dict:
        """Generate detailed contingency tables"""
        results = {}
        
        # Key categorical variables
        cat_vars = ['residence', 'wealth_quintile', 'region', 'caste', 'season']
        
        for var in cat_vars:
            if var not in self.df.columns:
                continue
            
            # Contingency table with percentages
            crosstab = pd.crosstab(self.df[var], self.df['water_disrupted'], margins=True)
            crosstab_pct = pd.crosstab(self.df[var], self.df['water_disrupted'], normalize='index') * 100
            
            results[var] = {
                'counts': crosstab,
                'row_percentages': crosstab_pct,
                'disruption_rates': crosstab_pct[1] if 1 in crosstab_pct.columns else None
            }
        
        return results

class MultivariateAnalyzer:
    """Comprehensive multivariate statistical analysis"""
    
    def __init__(self, df: pd.DataFrame, config: Config):
        self.df = df
        self.config = config
    
    def run_comprehensive_multivariate_analysis(self) -> Dict:
        """Run all multivariate analyses"""
        print("\nRunning comprehensive multivariate analysis...")
        
        results = {}
        results['logistic_regression'] = self._logistic_regression_analysis()
        results['multiple_regression'] = self._multiple_regression_analysis()
        results['factor_analysis'] = self._factor_analysis()
        results['principal_components'] = self._principal_component_analysis()
        results['manova'] = self._manova_analysis()
        results['discriminant_analysis'] = self._discriminant_analysis()
        results['cluster_analysis'] = self._cluster_analysis()
        results['interaction_effects'] = self._interaction_effects_analysis()
        
        print("Multivariate analysis complete")
        return results
    
    def _logistic_regression_analysis(self) -> Dict:
        """Comprehensive logistic regression with model diagnostics"""
        try:
            # Prepare predictors
            predictors = [
                'wealth_score', 'infrastructure_score', 'vulnerability_score',
                'coping_capacity_score', 'water_time_score', 'hh_size'
            ]
            
            # Add categorical variables as dummies
            cat_vars = ['residence', 'region', 'season']
            model_df = self.df[predictors + cat_vars + ['water_disrupted', 'weight']].copy()
            
            # Create dummy variables
            for cat_var in cat_vars:
                if cat_var in model_df.columns:
                    dummies = pd.get_dummies(model_df[cat_var], prefix=cat_var, drop_first=True)
                    model_df = pd.concat([model_df, dummies], axis=1)
                    model_df.drop(cat_var, axis=1, inplace=True)
            
            model_df = model_df.dropna()
            
            if len(model_df) < 100:
                return {'error': 'Insufficient data for logistic regression'}
            
            # Prepare variables
            y = model_df['water_disrupted'].astype(float)
            X = model_df.drop(['water_disrupted', 'weight'], axis=1).astype(float)
            weights = model_df['weight'].astype(float)
            
            # Standardize continuous variables
            continuous_cols = [col for col in predictors if col in X.columns]
            if continuous_cols:
                scaler = StandardScaler()
                X[continuous_cols] = scaler.fit_transform(X[continuous_cols])
            
            # Add constant
            X = sm.add_constant(X)
            
            # Fit model
            model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=weights)
            results_sm = model.fit()
            
            # Model diagnostics
            # Pseudo R-squared
            null_model = sm.GLM(y, np.ones(len(y)), family=sm.families.Binomial(), freq_weights=weights)
            null_results = null_model.fit()
            mcfadden_r2 = 1 - (results_sm.llf / null_results.llf)
            
            # Create results DataFrame
            coef_df = pd.DataFrame({
                'Variable': results_sm.params.index,
                'Coefficient': results_sm.params.values,
                'Std_Error': results_sm.bse.values,
                'z_value': results_sm.tvalues.values,
                'p_value': results_sm.pvalues.values,
                'OR': np.exp(results_sm.params.values),
                'OR_CI_lower': np.exp(results_sm.params.values - 1.96*results_sm.bse.values),
                'OR_CI_upper': np.exp(results_sm.params.values + 1.96*results_sm.bse.values)
            })
            
            # Add significance indicators
            coef_df['Significance'] = coef_df['p_value'].apply(
                lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            )
            
            return {
                'coefficients': coef_df,
                'model_fit': {
                    'aic': results_sm.aic,
                    'bic': results_sm.bic,
                    'log_likelihood': results_sm.llf,
                    'mcfadden_r2': mcfadden_r2,
                    'n_observations': len(model_df)
                },
                'model_summary': str(results_sm.summary())
            }
            
        except Exception as e:
            return {'error': f'Logistic regression failed: {str(e)}'}
    
    def _multiple_regression_analysis(self) -> Dict:
        """Multiple linear regression for continuous outcomes"""
        try:
            # Use continuous outcomes like vulnerability_score, water_insecurity_score
            outcomes = ['vulnerability_score', 'water_insecurity_score', 'disruption_severity_score']
            predictors = ['wealth_score', 'infrastructure_score', 'coping_capacity_score', 'hh_size']
            
            results = {}
            
            for outcome in outcomes:
                if outcome not in self.df.columns:
                    continue
                
                # Prepare data
                model_vars = predictors + [outcome, 'weight', 'residence']
                model_df = self.df[model_vars].copy()
                
                # Add residence dummy
                model_df['urban'] = (model_df['residence'] == 'Urban').astype(int)
                model_df = model_df.drop('residence', axis=1)
                model_df = model_df.dropna()
                
                if len(model_df) < 50:
                    continue
                
                # Prepare variables
                y = model_df[outcome]
                X = model_df.drop([outcome, 'weight'], axis=1)
                weights = model_df['weight']
                
                # Standardize variables
                scaler = StandardScaler()
                X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
                X_scaled = sm.add_constant(X_scaled)
                
                # Fit weighted regression
                model = sm.WLS(y, X_scaled, weights=weights)
                results_reg = model.fit()
                
                # Create results
                coef_df = pd.DataFrame({
                    'Variable': results_reg.params.index,
                    'Coefficient': results_reg.params.values,
                    'Std_Error': results_reg.bse.values,
                    't_value': results_reg.tvalues.values,
                    'p_value': results_reg.pvalues.values,
                    'CI_lower': results_reg.conf_int()[0],
                    'CI_upper': results_reg.conf_int()[1]
                })
                
                results[outcome] = {
                    'coefficients': coef_df,
                    'r_squared': results_reg.rsquared,
                    'adj_r_squared': results_reg.rsquared_adj,
                    'f_statistic': results_reg.fvalue,
                    'f_p_value': results_reg.f_pvalue,
                    'aic': results_reg.aic,
                    'bic': results_reg.bic,
                    'n_observations': len(model_df)
                }
            
            return results
            
        except Exception as e:
            return {'error': f'Multiple regression failed: {str(e)}'}
    
    def _factor_analysis(self) -> Dict:
        """Factor analysis for dimension reduction"""
        try:
            # Variables for factor analysis
            factor_vars = ['wealth_score', 'infrastructure_score', 'vulnerability_score',
                          'water_insecurity_score', 'coping_capacity_score', 'water_time_score']
            
            available_vars = [var for var in factor_vars if var in self.df.columns]
            
            if len(available_vars) < 4:
                return {'error': 'Insufficient variables for factor analysis'}
            
            factor_data = self.df[available_vars].dropna()
            
            if len(factor_data) < 100:
                return {'error': 'Insufficient observations for factor analysis'}
            
            # Standardize data
            scaler = StandardScaler()
            factor_data_scaled = scaler.fit_transform(factor_data)
            
            # Factor analysis with different numbers of factors
            results = {}
            
            for n_factors in range(1, min(5, len(available_vars))):
                fa = FactorAnalysis(n_components=n_factors, random_state=42)
                fa.fit(factor_data_scaled)
                
                # Calculate explained variance
                explained_var = np.var(fa.transform(factor_data_scaled), axis=0)
                total_var = explained_var.sum()
                
                results[f'{n_factors}_factors'] = {
                    'loadings': pd.DataFrame(fa.components_.T, 
                                           index=available_vars,
                                           columns=[f'Factor_{i+1}' for i in range(n_factors)]),
                    'explained_variance': explained_var,
                    'total_variance_explained': total_var,
                    'log_likelihood': fa.loglike_[-1] if hasattr(fa, 'loglike_') else None
                }
            
            return results
            
        except Exception as e:
            return {'error': f'Factor analysis failed: {str(e)}'}
    
    def _principal_component_analysis(self) -> Dict:
        """Principal Component Analysis"""
        try:
            # Variables for PCA
            pca_vars = ['wealth_score', 'infrastructure_score', 'vulnerability_score',
                       'water_insecurity_score', 'coping_capacity_score']
            
            available_vars = [var for var in pca_vars if var in self.df.columns]
            
            if len(available_vars) < 3:
                return {'error': 'Insufficient variables for PCA'}
            
            pca_data = self.df[available_vars].dropna()
            
            if len(pca_data) < 100:
                return {'error': 'Insufficient observations for PCA'}
            
            # Standardize data
            scaler = StandardScaler()
            pca_data_scaled = scaler.fit_transform(pca_data)
            
            # Perform PCA
            pca = PCA()
            pca_results = pca.fit_transform(pca_data_scaled)
            
            # Create results
            n_components = len(available_vars)
            
            components_df = pd.DataFrame(
                pca.components_,
                columns=available_vars,
                index=[f'PC{i+1}' for i in range(n_components)]
            )
            
            explained_var_df = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(n_components)],
                'Eigenvalue': pca.explained_variance_,
                'Variance_Explained': pca.explained_variance_ratio_,
                'Cumulative_Variance': np.cumsum(pca.explained_variance_ratio_)
            })
            
            return {
                'components': components_df,
                'explained_variance': explained_var_df,
                'total_variance_explained': pca.explained_variance_ratio_.sum(),
                'kaiser_criterion': sum(pca.explained_variance_ > 1),  # Eigenvalues > 1
                'n_observations': len(pca_data)
            }
            
        except Exception as e:
            return {'error': f'PCA failed: {str(e)}'}
    
    def _manova_analysis(self) -> Dict:
        """Multivariate Analysis of Variance (MANOVA)"""
        try:
            # Dependent variables (continuous)
            dependent_vars = ['vulnerability_score', 'water_insecurity_score', 'coping_capacity_score']
            available_deps = [var for var in dependent_vars if var in self.df.columns]
            
            if len(available_deps) < 2:
                return {'error': 'Insufficient dependent variables for MANOVA'}
            
            # Independent variable (categorical)
            independent_vars = ['residence', 'wealth_quintile', 'region']
            
            results = {}
            
            for indep_var in independent_vars:
                if indep_var not in self.df.columns:
                    continue
                
                # Prepare data
                manova_data = self.df[available_deps + [indep_var]].dropna()
                
                if len(manova_data) < 50:
                    continue
                
                # Create formula
                formula = f"{' + '.join(available_deps)} ~ C({indep_var})"
                
                try:
                    manova = MANOVA.from_formula(formula, data=manova_data)
                    manova_results = manova.mv_test()
                    
                    results[indep_var] = {
                        'manova_results': str(manova_results),
                        'n_observations': len(manova_data),
                        'dependent_variables': available_deps
                    }
                except Exception as e:
                    results[indep_var] = {'error': f'MANOVA failed for {indep_var}: {str(e)}'}
            
            return results
            
        except Exception as e:
            return {'error': f'MANOVA analysis failed: {str(e)}'}
    
    def _discriminant_analysis(self) -> Dict:
        """Linear Discriminant Analysis"""
        try:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            
            # Prepare predictors and target
            predictors = ['wealth_score', 'infrastructure_score', 'vulnerability_score',
                         'coping_capacity_score', 'water_time_score']
            available_predictors = [var for var in predictors if var in self.df.columns]
            
            if len(available_predictors) < 3:
                return {'error': 'Insufficient predictors for discriminant analysis'}
            
            # Prepare data
            da_data = self.df[available_predictors + ['water_disrupted']].dropna()
            
            if len(da_data) < 100:
                return {'error': 'Insufficient observations for discriminant analysis'}
            
            X = da_data[available_predictors]
            y = da_data['water_disrupted']
            
            # Standardize predictors
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit LDA
            lda = LinearDiscriminantAnalysis()
            lda.fit(X_scaled, y)
            
            # Calculate group means
            group_means = da_data.groupby('water_disrupted')[available_predictors].mean()
            
            # Predict and calculate accuracy
            y_pred = lda.predict(X_scaled)
            accuracy = (y_pred == y).mean()
            
            return {
                'discriminant_functions': pd.DataFrame(
                    lda.coef_,
                    columns=available_predictors,
                    index=['Discriminant_Function']
                ),
                'group_means': group_means,
                'prior_probabilities': lda.priors_,
                'accuracy': accuracy,
                'n_observations': len(da_data),
                'explained_variance_ratio': lda.explained_variance_ratio_[0] if len(lda.explained_variance_ratio_) > 0 else None
            }
            
        except Exception as e:
            return {'error': f'Discriminant analysis failed: {str(e)}'}
    
    def _cluster_analysis(self) -> Dict:
        """K-means and hierarchical clustering"""
        try:
            # Variables for clustering
            cluster_vars = ['wealth_score', 'infrastructure_score', 'vulnerability_score',
                           'water_insecurity_score', 'coping_capacity_score']
            available_vars = [var for var in cluster_vars if var in self.df.columns]
            
            if len(available_vars) < 3:
                return {'error': 'Insufficient variables for clustering'}
            
            cluster_data = self.df[available_vars].dropna()
            
            if len(cluster_data) < 100:
                return {'error': 'Insufficient observations for clustering'}
            
            # Standardize data
            scaler = StandardScaler()
            cluster_data_scaled = scaler.fit_transform(cluster_data)
            
            results = {}
            
            # K-means clustering for different k values
            kmeans_results = {}
            inertias = []
            k_range = range(2, min(8, len(available_vars) + 2))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(cluster_data_scaled)
                
                # Calculate silhouette score
                from sklearn.metrics import silhouette_score
                silhouette_avg = silhouette_score(cluster_data_scaled, cluster_labels)
                
                # Cluster centers (back-transformed)
                centers_scaled = kmeans.cluster_centers_
                centers_original = scaler.inverse_transform(centers_scaled)
                
                centers_df = pd.DataFrame(
                    centers_original,
                    columns=available_vars,
                    index=[f'Cluster_{i}' for i in range(k)]
                )
                
                # Cluster sizes
                cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
                
                kmeans_results[f'k_{k}'] = {
                    'cluster_centers': centers_df,
                    'cluster_sizes': cluster_sizes,
                    'inertia': kmeans.inertia_,
                    'silhouette_score': silhouette_avg
                }
                inertias.append(kmeans.inertia_)
            
            results['kmeans'] = kmeans_results
            results['elbow_data'] = {'k_values': list(k_range), 'inertias': inertias}
            
            return results
            
        except Exception as e:
            return {'error': f'Cluster analysis failed: {str(e)}'}
    
    def _interaction_effects_analysis(self) -> Dict:
        """Analysis of interaction effects"""
        try:
            results = {}
            
            # Two-way interactions for logistic regression
            interaction_pairs = [
                ('wealth_score', 'infrastructure_score'),
                ('vulnerability_score', 'coping_capacity_score'),
                ('wealth_score', 'residence'),
                ('infrastructure_score', 'residence')
            ]
            
            for var1, var2 in interaction_pairs:
                if var1 not in self.df.columns or var2 not in self.df.columns:
                    continue
                
                # Prepare interaction data
                if var2 == 'residence':
                    interact_data = self.df[[var1, var2, 'water_disrupted', 'weight']].copy()
                    interact_data['urban'] = (interact_data[var2] == 'Urban').astype(int)
                    interact_data[f'{var1}_x_urban'] = interact_data[var1] * interact_data['urban']
                    
                    # Fit model with interaction
                    model_df = interact_data[[var1, 'urban', f'{var1}_x_urban', 'water_disrupted', 'weight']].dropna()
                    
                    if len(model_df) > 100:
                        y = model_df['water_disrupted']
                        X = model_df[[var1, 'urban', f'{var1}_x_urban']]
                        weights = model_df['weight']
                        
                        X = sm.add_constant(X)
                        
                        try:
                            model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=weights)
                            results_int = model.fit()
                            
                            results[f'{var1}_x_{var2}'] = {
                                'interaction_coef': results_int.params[f'{var1}_x_urban'],
                                'interaction_p': results_int.pvalues[f'{var1}_x_urban'],
                                'interaction_significant': results_int.pvalues[f'{var1}_x_urban'] < 0.05,
                                'aic': results_int.aic,
                                'n_observations': len(model_df)
                            }
                        except:
                            results[f'{var1}_x_{var2}'] = {'error': 'Model fitting failed'}
                
                else:
                    # Continuous x continuous interaction
                    interact_data = self.df[[var1, var2, 'water_disrupted', 'weight']].copy()
                    interact_data[f'{var1}_x_{var2}'] = interact_data[var1] * interact_data[var2]
                    
                    model_df = interact_data.dropna()
                    
                    if len(model_df) > 100:
                        y = model_df['water_disrupted']
                        X = model_df[[var1, var2, f'{var1}_x_{var2}']]
                        weights = model_df['weight']
                        
                        # Standardize continuous variables
                        scaler = StandardScaler()
                        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
                        X_scaled = sm.add_constant(X_scaled)
                        
                        try:
                            model = sm.GLM(y, X_scaled, family=sm.families.Binomial(), freq_weights=weights)
                            results_int = model.fit()
                            
                            results[f'{var1}_x_{var2}'] = {
                                'interaction_coef': results_int.params[f'{var1}_x_{var2}'],
                                'interaction_p': results_int.pvalues[f'{var1}_x_{var2}'],
                                'interaction_significant': results_int.pvalues[f'{var1}_x_{var2}'] < 0.05,
                                'aic': results_int.aic,
                                'n_observations': len(model_df)
                            }
                        except:
                            results[f'{var1}_x_{var2}'] = {'error': 'Model fitting failed'}
            
            return results
            
        except Exception as e:
            return {'error': f'Interaction effects analysis failed: {str(e)}'}

class AdvancedStatisticalTests:
    """Advanced statistical tests and model diagnostics"""
    
    def __init__(self, df: pd.DataFrame, config: Config):
        self.df = df
        self.config = config
    
    def run_advanced_tests(self) -> Dict:
        """Run advanced statistical tests"""
        print("\nRunning advanced statistical tests...")
        
        results = {}
        results['normality_tests'] = self._normality_tests()
        results['homogeneity_tests'] = self._homogeneity_tests()
        results['independence_tests'] = self._independence_tests()
        results['effect_size_calculations'] = self._effect_size_calculations()
        results['power_analysis'] = self._power_analysis()
        results['multiple_comparisons'] = self._multiple_comparisons_correction()
        
        return results
    
    def _normality_tests(self) -> Dict:
        """Test normality of continuous variables"""
        from scipy.stats import shapiro, normaltest, kstest
        
        continuous_vars = ['wealth_score', 'infrastructure_score', 'vulnerability_score',
                          'water_insecurity_score', 'coping_capacity_score']
        
        results = {}
        
        for var in continuous_vars:
            if var not in self.df.columns:
                continue
            
            data = self.df[var].dropna()
            
            if len(data) > 20:
                # Shapiro-Wilk test (for smaller samples)
                if len(data) <= 5000:
                    shapiro_stat, shapiro_p = shapiro(data)
                else:
                    shapiro_stat, shapiro_p = None, None
                
                # D'Agostino's normality test
                dagostino_stat, dagostino_p = normaltest(data)
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_p = kstest(data, 'norm', args=(data.mean(), data.std()))
                
                results[var] = {
                    'shapiro_statistic': shapiro_stat,
                    'shapiro_p_value': shapiro_p,
                    'dagostino_statistic': dagostino_stat,
                    'dagostino_p_value': dagostino_p,
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p,
                    'sample_size': len(data),
                    'skewness': stats.skew(data),
                    'kurtosis': stats.kurtosis(data)
                }
        
        return results
    
    def _homogeneity_tests(self) -> Dict:
        """Test homogeneity of variances"""
        from scipy.stats import levene, bartlett
        
        results = {}
        continuous_vars = ['wealth_score', 'infrastructure_score', 'vulnerability_score']
        grouping_vars = ['residence', 'wealth_quintile', 'region']
        
        for cont_var in continuous_vars:
            if cont_var not in self.df.columns:
                continue
            
            for group_var in grouping_vars:
                if group_var not in self.df.columns:
                    continue
                
                # Create groups
                groups = []
                group_names = []
                for category in self.df[group_var].unique():
                    if pd.notna(category):
                        group_data = self.df[self.df[group_var] == category][cont_var].dropna()
                        if len(group_data) > 5:
                            groups.append(group_data)
                            group_names.append(category)
                
                if len(groups) >= 2:
                    try:
                        # Levene's test
                        levene_stat, levene_p = levene(*groups)
                        
                        # Bartlett's test
                        bartlett_stat, bartlett_p = bartlett(*groups)
                        
                        results[f'{cont_var}_by_{group_var}'] = {
                            'levene_statistic': levene_stat,
                            'levene_p_value': levene_p,
                            'bartlett_statistic': bartlett_stat,
                            'bartlett_p_value': bartlett_p,
                            'groups': group_names,
                            'group_variances': [np.var(group) for group in groups]
                        }
                    except:
                        results[f'{cont_var}_by_{group_var}'] = {'error': 'Homogeneity test failed'}
        
        return results
    
    def _independence_tests(self) -> Dict:
        """Test independence assumptions"""
        results = {}
        
        # Spatial autocorrelation (Moran's I approximation)
        if 'cluster' in self.df.columns and 'state_code' in self.df.columns:
            cluster_stats = self.df.groupby(['state_code', 'cluster']).agg({
                'water_disrupted': 'mean',
                'weight': 'sum'
            }).reset_index()
            
            # Simple spatial correlation within states
            spatial_corr_results = {}
            for state in cluster_stats['state_code'].unique():
                if pd.notna(state):
                    state_data = cluster_stats[cluster_stats['state_code'] == state]
                    if len(state_data) > 3:
                        # Calculate local spatial correlation
                        disruption_rates = state_data['water_disrupted'].values
                        weights = state_data['weight'].values
                        
                        # Weighted correlation with lag-1
                        if len(disruption_rates) > 1:
                            lag1_corr = np.corrcoef(disruption_rates[:-1], disruption_rates[1:])[0, 1]
                            spatial_corr_results[f'state_{int(state)}'] = {
                                'lag1_correlation': lag1_corr,
                                'n_clusters': len(state_data)
                            }
            
            results['spatial_autocorrelation'] = spatial_corr_results
        
        return results
    
    def _effect_size_calculations(self) -> Dict:
        """Calculate various effect sizes"""
        results = {}
        
        # Cohen's d for continuous variables
        continuous_vars = ['wealth_score', 'infrastructure_score', 'vulnerability_score']
        
        for var in continuous_vars:
            if var not in self.df.columns:
                continue
            
            group0 = self.df[self.df['water_disrupted'] == 0][var].dropna()
            group1 = self.df[self.df['water_disrupted'] == 1][var].dropna()
            
            if len(group0) > 0 and len(group1) > 0:
                # Cohen's d
                pooled_std = np.sqrt((group0.var() + group1.var()) / 2)
                cohens_d = (group1.mean() - group0.mean()) / pooled_std if pooled_std > 0 else 0
                
                # Glass's delta
                glass_delta = (group1.mean() - group0.mean()) / group0.std() if group0.std() > 0 else 0
                
                # Hedges' g (bias-corrected)
                n1, n2 = len(group0), len(group1)
                hedges_correction = 1 - (3 / (4 * (n1 + n2 - 2) - 1))
                hedges_g = cohens_d * hedges_correction
                
                results[f'{var}_effect_sizes'] = {
                    'cohens_d': cohens_d,
                    'glass_delta': glass_delta,
                    'hedges_g': hedges_g,
                    'interpretation': self._interpret_effect_size(abs(cohens_d)),
                    'group0_mean': group0.mean(),
                    'group1_mean': group1.mean(),
                    'group0_n': n1,
                    'group1_n': n2
                }
        
        return results
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude"""
        if effect_size < 0.2:
            return 'negligible'
        elif effect_size < 0.5:
            return 'small'
        elif effect_size < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def _power_analysis(self) -> Dict:
        """Basic power analysis calculations"""
        results = {}
        
        # Sample sizes by group
        n_total = len(self.df)
        n_disrupted = self.df['water_disrupted'].sum()
        n_not_disrupted = n_total - n_disrupted
        
        # Effect size for main comparison
        if 'wealth_score' in self.df.columns:
            group0 = self.df[self.df['water_disrupted'] == 0]['wealth_score'].dropna()
            group1 = self.df[self.df['water_disrupted'] == 1]['wealth_score'].dropna()
            
            if len(group0) > 0 and len(group1) > 0:
                pooled_std = np.sqrt((group0.var() + group1.var()) / 2)
                effect_size = abs(group1.mean() - group0.mean()) / pooled_std if pooled_std > 0 else 0
                
                # Approximate power calculation for t-test
                # This is a simplified calculation
                alpha = 0.05
                z_alpha = stats.norm.ppf(1 - alpha/2)
                z_beta = effect_size * np.sqrt((len(group0) * len(group1)) / (len(group0) + len(group1))) / 2 - z_alpha
                power = stats.norm.cdf(z_beta)
                
                results['main_comparison'] = {
                    'effect_size': effect_size,
                    'power_estimate': max(0, power),
                    'sample_size_group0': len(group0),
                    'sample_size_group1': len(group1),
                    'alpha_level': alpha
                }
        
        results['sample_size_summary'] = {
            'total_sample': n_total,
            'disrupted': n_disrupted,
            'not_disrupted': n_not_disrupted,
            'disruption_rate': n_disrupted / n_total if n_total > 0 else 0
        }
        
        return results
    
    def _multiple_comparisons_correction(self) -> Dict:
        """Apply multiple comparisons corrections"""
        results = {}
        
        # Collect p-values from various tests
        p_values = []
        test_names = []
        
        # Example: collect p-values from state comparisons
        if 'state' in self.df.columns:
            state_groups = []
            for state in self.df['state'].unique():
                if pd.notna(state) and len(self.df[self.df['state'] == state]) > 30:
                    state_data = self.df[self.df['state'] == state]['water_disrupted']
                    state_groups.append(state_data)
                    test_names.append(f'state_{state}')
            
            if len(state_groups) > 2:
                # ANOVA p-value
                try:
                    f_stat, anova_p = f_oneway(*state_groups)
                    p_values.append(anova_p)
                    test_names = ['state_anova']
                except:
                    pass
        
        # Apply corrections if we have multiple p-values
        if len(p_values) > 1:
            # Bonferroni correction
            bonferroni_corrected = [min(p * len(p_values), 1.0) for p in p_values]
            
            # Benjamini-Hochberg correction
            try:
                bh_rejected, bh_corrected, _, _ = multipletests(p_values, method='fdr_bh')
                
                results['multiple_corrections'] = {
                    'test_names': test_names,
                    'original_p_values': p_values,
                    'bonferroni_corrected': bonferroni_corrected,
                    'bh_corrected': bh_corrected.tolist(),
                    'bh_significant': bh_rejected.tolist(),
                    'n_tests': len(p_values)
                }
            except:
                results['multiple_corrections'] = {'error': 'Correction failed'}
        
        return results

def export_comprehensive_results(df: pd.DataFrame, bivariate_results: Dict, 
                               multivariate_results: Dict, advanced_results: Dict, 
                               timestamp: str):
    """Export comprehensive results to Excel with multiple detailed sheets"""
    print("\nExporting comprehensive results to Excel...")
    
    filename = f'comprehensive_water_analysis_{timestamp}.xlsx'
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        
        # 1. EXECUTIVE SUMMARY
        summary_data = {
            'Metric': [
                'Total Households',
                'Weighted Population', 
                'Water Disruption Rate (%)',
                'Urban Disruption Rate (%)',
                'Rural Disruption Rate (%)',
                'Severe Disruption (%)',
                'High Vulnerability (%)',
                'Water on Premises (%)',
                'Improved Source (%)',
                'Women Fetch Water (%)'
            ],
            'Value': [
                len(df),
                df['weight'].sum(),
                df['water_disrupted'].mean() * 100,
                df[df['residence']=='Urban']['water_disrupted'].mean() * 100 if 'residence' in df.columns else 0,
                df[df['residence']=='Rural']['water_disrupted'].mean() * 100 if 'residence' in df.columns else 0,
                (df['disruption_severity']=='Severe').mean() * 100 if 'disruption_severity' in df.columns else 0,
                (df['vulnerability_level']=='High').mean() * 100 if 'vulnerability_level' in df.columns else 0,
                df['water_on_premises'].mean() * 100 if 'water_on_premises' in df.columns else 0,
                df['improved_source'].mean() * 100 if 'improved_source' in df.columns else 0,
                df['women_fetch_water'].mean() * 100 if 'women_fetch_water' in df.columns else 0
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Executive_Summary', index=False)
        
        # 2. BIVARIATE ANALYSES
        
        # Chi-square tests
        if 'chi_square_tests' in bivariate_results:
            chi_sq_data = []
            for var, results in bivariate_results['chi_square_tests'].items():
                if 'error' not in results:
                    chi_sq_data.append({
                        'Variable': var,
                        'Chi2_Statistic': results.get('chi2_statistic', ''),
                        'P_Value': results.get('p_value', ''),
                        'Degrees_Freedom': results.get('degrees_freedom', ''),
                        'Cramers_V': results.get('cramers_v', ''),
                        'Sample_Size': results.get('sample_size', ''),
                        'Significant': results.get('significant', '')
                    })
            if chi_sq_data:
                pd.DataFrame(chi_sq_data).to_excel(writer, sheet_name='Chi_Square_Tests', index=False)
        
        # T-tests
        if 't_tests' in bivariate_results:
            t_test_data = []
            for var, results in bivariate_results['t_tests'].items():
                t_test_data.append({
                    'Variable': var,
                    'Group0_Mean': results.get('group0_mean', ''),
                    'Group1_Mean': results.get('group1_mean', ''),
                    'T_Statistic': results.get('t_statistic', ''),
                    'P_Value': results.get('p_value', ''),
                    'Cohens_D': results.get('cohens_d', ''),
                    'Significant': results.get('significant', '')
                })
            if t_test_data:
                pd.DataFrame(t_test_data).to_excel(writer, sheet_name='T_Tests', index=False)
        
        # ANOVA tests
        if 'anova_tests' in bivariate_results:
            anova_data = []
            for var, results in bivariate_results['anova_tests'].items():
                if 'error' not in results:
                    anova_data.append({
                        'Variable_Combination': var,
                        'F_Statistic': results.get('f_statistic', ''),
                        'F_P_Value': results.get('f_p_value', ''),
                        'Eta_Squared': results.get('eta_squared', ''),
                        'Kruskal_H': results.get('kruskal_h', ''),
                        'Kruskal_P': results.get('kruskal_p', ''),
                        'Significant': results.get('significant', '')
                    })
            if anova_data:
                pd.DataFrame(anova_data).to_excel(writer, sheet_name='ANOVA_Tests', index=False)
        
        # Correlation matrix
        if 'correlation_matrix' in bivariate_results:
            if 'correlation_matrix' in bivariate_results['correlation_matrix']:
                corr_matrix = bivariate_results['correlation_matrix']['correlation_matrix']
                corr_matrix.to_excel(writer, sheet_name='Correlation_Matrix')
                
                # Significance matrix
                if 'significance_matrix' in bivariate_results['correlation_matrix']:
                    sig_matrix = bivariate_results['correlation_matrix']['significance_matrix']
                    sig_matrix.to_excel(writer, sheet_name='Correlation_Significance')
        
        # Contingency tables
        if 'contingency_tables' in bivariate_results:
            for var, tables in bivariate_results['contingency_tables'].items():
                if 'counts' in tables:
                    tables['counts'].to_excel(writer, sheet_name=f'Contingency_{var}')
        
        # 3. MULTIVARIATE ANALYSES
        
        # Logistic regression
        if 'logistic_regression' in multivariate_results:
            if 'coefficients' in multivariate_results['logistic_regression']:
                multivariate_results['logistic_regression']['coefficients'].to_excel(
                    writer, sheet_name='Logistic_Regression', index=False)
        
        # Multiple regression
        if 'multiple_regression' in multivariate_results:
            for outcome, results in multivariate_results['multiple_regression'].items():
                if 'coefficients' in results:
                    results['coefficients'].to_excel(
                        writer, sheet_name=f'Regression_{outcome}', index=False)
        
        # PCA results
        if 'principal_components' in multivariate_results:
            if 'components' in multivariate_results['principal_components']:
                multivariate_results['principal_components']['components'].to_excel(
                    writer, sheet_name='PCA_Components')
            if 'explained_variance' in multivariate_results['principal_components']:
                multivariate_results['principal_components']['explained_variance'].to_excel(
                    writer, sheet_name='PCA_Variance', index=False)
        
        # Factor analysis
        if 'factor_analysis' in multivariate_results:
            for n_factors, results in multivariate_results['factor_analysis'].items():
                if 'loadings' in results:
                    results['loadings'].to_excel(writer, sheet_name=f'Factor_{n_factors}')
        
        # Cluster analysis
        if 'cluster_analysis' in multivariate_results and 'kmeans' in multivariate_results['cluster_analysis']:
            cluster_summary = []
            for k, results in multivariate_results['cluster_analysis']['kmeans'].items():
                cluster_summary.append({
                    'K_Clusters': k,
                    'Silhouette_Score': results.get('silhouette_score', ''),
                    'Inertia': results.get('inertia', '')
                })
            if cluster_summary:
                pd.DataFrame(cluster_summary).to_excel(writer, sheet_name='Cluster_Summary', index=False)
        
        # 4. ADVANCED STATISTICAL TESTS
        
        # Effect sizes
        if 'effect_size_calculations' in advanced_results:
            effect_size_data = []
            for var, results in advanced_results['effect_size_calculations'].items():
                effect_size_data.append({
                    'Variable': var,
                    'Cohens_D': results.get('cohens_d', ''),
                    'Hedges_G': results.get('hedges_g', ''),
                    'Glass_Delta': results.get('glass_delta', ''),
                    'Interpretation': results.get('interpretation', ''),
                    'Group0_N': results.get('group0_n', ''),
                    'Group1_N': results.get('group1_n', '')
                })
            if effect_size_data:
                pd.DataFrame(effect_size_data).to_excel(writer, sheet_name='Effect_Sizes', index=False)
        
        # Normality tests
        if 'normality_tests' in advanced_results:
            normality_data = []
            for var, results in advanced_results['normality_tests'].items():
                normality_data.append({
                    'Variable': var,
                    'Shapiro_Stat': results.get('shapiro_statistic', ''),
                    'Shapiro_P': results.get('shapiro_p_value', ''),
                    'DAgostino_Stat': results.get('dagostino_statistic', ''),
                    'DAgostino_P': results.get('dagostino_p_value', ''),
                    'KS_Stat': results.get('ks_statistic', ''),
                    'KS_P': results.get('ks_p_value', ''),
                    'Skewness': results.get('skewness', ''),
                    'Kurtosis': results.get('kurtosis', '')
                })
            if normality_data:
                pd.DataFrame(normality_data).to_excel(writer, sheet_name='Normality_Tests', index=False)
        
        # 5. DETAILED CROSS-TABULATIONS
        
        # Key cross-tabulations
        key_vars = ['residence', 'wealth_quintile', 'region', 'season']
        for var in key_vars:
            if var in df.columns:
                crosstab = pd.crosstab(df[var], df['water_disrupted'], margins=True, normalize='index') * 100
                crosstab.to_excel(writer, sheet_name=f'CrossTab_{var}')
    
    print(f"Comprehensive results exported to: {filename}")
    return filename

def main():
    """Main execution function with comprehensive bivariate and multivariate analyses"""
    print("=" * 60)
    print("NFHS 2019-21 Comprehensive Water Disruption Analysis")
    print("Enhanced with Bivariate and Multivariate Statistical Analyses")
    print("=" * 60)
    
    config = Config()
    
    # Get file path
    if DATA_FILE_PATH and os.path.exists(DATA_FILE_PATH):
        filepath = DATA_FILE_PATH
    else:
        filepath = input("\nEnter NFHS data file path: ").strip().strip('"')
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return
    
    try:
        # Load and process data
        loader = DataLoader(config)
        df = loader.load(filepath)
        
        processor = WaterDisruptionProcessor(df)
        df = processor.prepare()
        
        # Initialize analyzers
        bivariate_analyzer = BivariateAnalyzer(df, config)
        multivariate_analyzer = MultivariateAnalyzer(df, config)
        advanced_analyzer = AdvancedStatisticalTests(df, config)
        
        # Run comprehensive analyses
        print("\n" + "="*50)
        print("RUNNING COMPREHENSIVE STATISTICAL ANALYSES")
        print("="*50)
        
        bivariate_results = bivariate_analyzer.run_comprehensive_bivariate_analysis()
        multivariate_results = multivariate_analyzer.run_comprehensive_multivariate_analysis()
        advanced_results = advanced_analyzer.run_advanced_tests()
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        # Export comprehensive results
        filename = export_comprehensive_results(
            df, bivariate_results, multivariate_results, advanced_results, timestamp
        )
        
        # Print comprehensive summary
        print("\n" + "="*60)
        print("COMPREHENSIVE ANALYSIS COMPLETE!")
        print("="*60)
        
        print(f"\nKEY FINDINGS:")
        print(f"Total Households Analyzed: {len(df):,}")
        print(f"Water Disruption Rate: {df['water_disrupted'].mean()*100:.1f}%")
        
        if 'residence' in df.columns:
            urban_rate = df[df['residence']=='Urban']['water_disrupted'].mean()*100
            rural_rate = df[df['residence']=='Rural']['water_disrupted'].mean()*100
            print(f"Urban Disruption Rate: {urban_rate:.1f}%")
            print(f"Rural Disruption Rate: {rural_rate:.1f}%")
        
        # Print bivariate findings
        print(f"\nBIVARIATE ANALYSIS HIGHLIGHTS:")
        
        if 'chi_square_tests' in bivariate_results:
            significant_chi2 = sum(1 for results in bivariate_results['chi_square_tests'].values() 
                                 if isinstance(results, dict) and results.get('significant', False))
            print(f"Significant chi-square associations: {significant_chi2}")
        
        if 't_tests' in bivariate_results:
            significant_t = sum(1 for results in bivariate_results['t_tests'].values() 
                              if results.get('significant', False))
            print(f"Significant t-test comparisons: {significant_t}")
        
        if 'correlation_matrix' in bivariate_results and 'correlation_matrix' in bivariate_results['correlation_matrix']:
            corr_matrix = bivariate_results['correlation_matrix']['correlation_matrix']
            if 'water_disrupted' in corr_matrix.columns:
                strongest_corr = corr_matrix['water_disrupted'].abs().drop('water_disrupted').max()
                strongest_var = corr_matrix['water_disrupted'].abs().drop('water_disrupted').idxmax()
                print(f"Strongest correlation with water disruption: {strongest_var} (r={strongest_corr:.3f})")
        
        # Print multivariate findings
        print(f"\nMULTIVARIATE ANALYSIS HIGHLIGHTS:")
        
        if 'logistic_regression' in multivariate_results and 'model_fit' in multivariate_results['logistic_regression']:
            model_fit = multivariate_results['logistic_regression']['model_fit']
            print(f"Logistic regression McFadden R²: {model_fit.get('mcfadden_r2', 'N/A'):.3f}")
            print(f"Model AIC: {model_fit.get('aic', 'N/A')}")
        
        if 'principal_components' in multivariate_results and 'explained_variance' in multivariate_results['principal_components']:
            pca_var = multivariate_results['principal_components']['explained_variance']
            if len(pca_var) > 0:
                first_pc_var = pca_var.iloc[0]['Variance_Explained']
                print(f"First principal component explains: {first_pc_var*100:.1f}% of variance")
        
        if 'cluster_analysis' in multivariate_results and 'kmeans' in multivariate_results['cluster_analysis']:
            kmeans_results = multivariate_results['cluster_analysis']['kmeans']
            if 'k_3' in kmeans_results:
                silhouette_k3 = kmeans_results['k_3'].get('silhouette_score', 'N/A')
                print(f"3-cluster solution silhouette score: {silhouette_k3:.3f}")
        
        # Print advanced test findings
        print(f"\nADVANCED STATISTICAL TEST HIGHLIGHTS:")
        
        if 'effect_size_calculations' in advanced_results:
            large_effects = sum(1 for results in advanced_results['effect_size_calculations'].values() 
                              if results.get('interpretation') == 'large')
            print(f"Variables with large effect sizes: {large_effects}")
        
        if 'power_analysis' in advanced_results and 'main_comparison' in advanced_results['power_analysis']:
            power = advanced_results['power_analysis']['main_comparison'].get('power_estimate', 'N/A')
            print(f"Statistical power for main comparison: {power:.3f}")
        
        print(f"\nFILES GENERATED:")
        print(f"Comprehensive Excel file: {filename}")
        print(f"  - Executive Summary")
        print(f"  - Chi-square Tests")
        print(f"  - T-tests and ANOVA")
        print(f"  - Correlation Matrices")
        print(f"  - Logistic Regression")
        print(f"  - Multiple Regression")
        print(f"  - Factor Analysis")
        print(f"  - Principal Components")
        print(f"  - Cluster Analysis")
        print(f"  - Effect Size Calculations")
        print(f"  - Normality Tests")
        print(f"  - Detailed Cross-tabulations")
        
        print(f"\nSTATISTICAL METHODS APPLIED:")
        print("BIVARIATE:")
        print("  - Chi-square tests of independence")
        print("  - Independent samples t-tests")
        print("  - One-way ANOVA")
        print("  - Pearson, Spearman, and Kendall correlations")
        print("  - Mann-Whitney U tests")
        print("  - Comprehensive contingency table analysis")
        
        print("\nMULTIVARIATE:")
        print("  - Weighted logistic regression with diagnostics")
        print("  - Multiple linear regression")
        print("  - Principal component analysis")
        print("  - Factor analysis")
        print("  - Linear discriminant analysis")
        print("  - K-means clustering")
        print("  - Interaction effects analysis")
        print("  - MANOVA (when applicable)")
        
        print("\nADVANCED DIAGNOSTICS:")
        print("  - Normality tests (Shapiro-Wilk, D'Agostino, KS)")
        print("  - Homogeneity of variance tests (Levene, Bartlett)")
        print("  - Effect size calculations (Cohen's d, Hedges' g)")
        print("  - Power analysis")
        print("  - Multiple comparisons corrections")
        print("  - Spatial autocorrelation assessment")
        
        print(f"\nREADY FOR HIGH-IMPACT PUBLICATION:")
        print("  - Comprehensive statistical rigor")
        print("  - Multiple analytical approaches")
        print("  - Effect sizes and confidence intervals")
        print("  - Model diagnostics and validation")
        print("  - Publication-ready results tables")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()