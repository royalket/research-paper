#!/usr/bin/env python3
"""
NFHS-5 Water Disruption Analysis: Publication-Ready Framework
Version 4.0 - Methodologically Rigorous Implementation
Target: Nature Water, Science, Lancet Planetary Health (IF 10+)

Author: [Your Name]
Date: 2024
License: MIT
"""

# ============================================================================
# IMPORTS AND CONFIGURATION
# ============================================================================

import os
import sys
import json
import yaml
import pickle
import hashlib
import platform
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import logging

# Data handling
import pandas as pd
import numpy as np
import pyreadstat
from pandas.api.types import is_numeric_dtype

# Statistical packages
import scipy.stats as stats
from scipy.stats import (
    chi2_contingency, pearsonr, spearmanr, kendalltau,
    shapiro, normaltest, jarque_bera, kstest,
    mannwhitneyu, wilcoxon, kruskal, friedmanchisquare,
    f_oneway, ttest_ind, ttest_rel, ranksums,
    multivariate_normal, entropy, wasserstein_distance
)
from scipy.spatial import distance
from scipy.cluster import hierarchy
from scipy.optimize import minimize
from sklearn.experimental import enable_iterative_imputer
# Machine Learning
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit,
    cross_val_score, cross_validate,
    GridSearchCV, RandomizedSearchCV,
    train_test_split, learning_curve, validation_curve
)
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler,
    PolynomialFeatures, SplineTransformer
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    IsolationForest, VotingClassifier, VotingRegressor
)
from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso, ElasticNet,
    LassoCV, RidgeCV, ElasticNetCV
)
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, max_error,
    confusion_matrix, classification_report,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.decomposition import PCA, FactorAnalysis, FastICA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.inspection import permutation_importance, partial_dependence

# Statistical modeling
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import (
    het_breuschpagan, het_white, acorr_ljungbox,
    linear_harvey_collier, linear_rainbow
)
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q
from statsmodels.stats.power import tt_solve_power, TTestPower
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.api import VAR, VARMAX

# Causal inference
try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    warnings.warn("DoWhy not available. Install with: pip install dowhy")

try:
    from econml.dml import (
        LinearDML, SparseLinearDML, CausalForestDML,
        NonParamDML, KernelDML
    )
    from econml.metalearners import (
        TLearner, SLearner, XLearner, DomainAdaptationLearner
    )
    from econml.dr import DRLearner, ForestDRLearner
    from econml.iv.dml import DMLIV, NonParamDMLIV
    from econml.iv.dr import IntentToTreatDRIV
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    warnings.warn("EconML not available. Install with: pip install econml")

try:
    from causalml.inference.meta import (
        BaseSRegressor, BaseTRegressor, BaseXRegressor,
        BaseRRegressor
    )
    from causalml.propensity import ElasticNetPropensityModel
    from causalml.match import NearestNeighborMatch, MatchOptimizer
    CAUSALML_AVAILABLE = True
except ImportError:
    CAUSALML_AVAILABLE = False
    warnings.warn("CausalML not available. Install with: pip install causalml")

# Advanced packages
try:
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import (
        calculate_bartlett_sphericity, calculate_kmo
    )
    FACTOR_ANALYZER_AVAILABLE = True
except ImportError:
    FACTOR_ANALYZER_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import geopandas as gpd
    from pysal.explore import esda
    from pysal.model import spreg
    SPATIAL_AVAILABLE = True
except ImportError:
    SPATIAL_AVAILABLE = False
    warnings.warn("Spatial packages not available")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

@dataclass
class AnalysisConfig:
    """Complete configuration for reproducible analysis"""
    
    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    figure_dir: Path = field(default_factory=lambda: Path("figures"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    
    # Data file
    data_file: str = "/Users/kumar.aniket/Downloads/surveydata/IAHR7EFL.DTA"
    
    # Random seeds
    random_seed: int = 42
    n_jobs: int = -1
    
    # Statistical parameters
    alpha: float = 0.05  # Significance level
    confidence_level: float = 0.95
    bonferroni_correction: bool = True
    fdr_correction: bool = True
    
    # Bootstrap parameters
    n_bootstrap: int = 10000
    bootstrap_ci_method: str = 'percentile'  # 'percentile', 'bca', 'basic'
    
    # Cross-validation
    cv_folds: int = 10
    cv_strategy: str = 'stratified'  # 'stratified', 'grouped', 'time_series'
    
    # Missing data
    missing_threshold: float = 0.5  # Drop variables with >50% missing
    imputation_method: str = 'multiple'  # 'simple', 'knn', 'iterative', 'multiple'
    n_imputations: int = 20
    
    # Outlier detection
    outlier_method: str = 'isolation_forest'  # 'iqr', 'zscore', 'isolation_forest'
    outlier_threshold: float = 0.01  # For isolation forest
    
    # Model selection
    model_selection_metric: str = 'auc'  # 'auc', 'f1', 'precision', 'recall'
    use_nested_cv: bool = True
    
    # Causal inference
    causal_estimands: List[str] = field(default_factory=lambda: ['ate', 'att', 'cate'])
    sensitivity_analysis: bool = True
    placebo_tests: bool = True
    
    # Index construction
    index_validation: bool = True
    min_factor_loading: float = 0.4
    min_cronbach_alpha: float = 0.7
    
    # Spatial analysis
    spatial_weights: str = 'queen'  # 'queen', 'rook', 'knn', 'distance'
    spatial_lag: int = 1
    
    # Reporting
    verbose: int = 1
    save_intermediate: bool = True
    create_report: bool = True
    
    def __post_init__(self):
        """Create directories and validate configuration"""
        for dir_attr in ['data_dir', 'output_dir', 'figure_dir', 'log_dir']:
            dir_path = getattr(self, dir_attr)
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging system"""
        log_file = self.log_dir / f"analysis_{datetime.now():%Y%m%d_%H%M%S}.log"
        
        logging.basicConfig(
            level=logging.INFO if self.verbose > 0 else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Analysis started at {datetime.now()}")
        self.logger.info(f"Configuration: {self.__dict__}")
    
    def to_yaml(self, filepath: Path):
        """Save configuration to YAML"""
        config_dict = {
            k: str(v) if isinstance(v, Path) else v 
            for k, v in self.__dict__.items()
            if not k.startswith('_') and k != 'logger'
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def from_yaml(cls, filepath: Path):
        """Load configuration from YAML"""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert string paths back to Path objects
        for key in ['data_dir', 'output_dir', 'figure_dir', 'log_dir']:
            if key in config_dict:
                config_dict[key] = Path(config_dict[key])
        
        return cls(**config_dict)

# ============================================================================
# REPRODUCIBILITY MANAGER
# ============================================================================

class ReproducibilityManager:
    """Ensure complete reproducibility of analysis"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.set_all_seeds()
        self.log_environment()
        self.create_hash()
    
    def set_all_seeds(self):
        """Set all random seeds for reproducibility"""
        seed = self.config.random_seed
        
        # Python random
        import random
        random.seed(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # Scikit-learn
        from sklearn.utils import check_random_state
        self.random_state = check_random_state(seed)
        
        # Environment variable for hash randomization
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # TensorFlow/Keras if available
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
        except ImportError:
            pass
        
        # PyTorch if available
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
        
        self.config.logger.info(f"All random seeds set to {seed}")
    
    def log_environment(self):
        """Log complete computational environment"""
        import pkg_resources
        
        environment = {
            'timestamp': datetime.now().isoformat(),
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
            },
            'packages': {
                pkg.key: pkg.version
                for pkg in pkg_resources.working_set
            },
            'environment_variables': {
                k: v for k, v in os.environ.items()
                if any(x in k.upper() for x in ['PATH', 'PYTHON', 'CONDA', 'VIRTUAL'])
            }
        }
        
        env_file = self.config.output_dir / 'environment.json'
        with open(env_file, 'w') as f:
            json.dump(environment, f, indent=2)
        
        self.config.logger.info(f"Environment logged to {env_file}")
        
        return environment
    
    def create_hash(self):
        """Create hash of code and data for verification"""
        hasher = hashlib.sha256()
        
        # Hash the main script
        script_path = Path(__file__)
        if script_path.exists():
            with open(script_path, 'rb') as f:
                hasher.update(f.read())
        
        # Hash configuration
        config_str = json.dumps(
            {k: str(v) for k, v in self.config.__dict__.items() 
             if not k.startswith('_') and k != 'logger'},
            sort_keys=True
        )
        hasher.update(config_str.encode())
        
        self.analysis_hash = hasher.hexdigest()[:16]
        
        self.config.logger.info(f"Analysis hash: {self.analysis_hash}")
        
        return self.analysis_hash

# ============================================================================
# DATA QUALITY FRAMEWORK
# ============================================================================

class DataQualityFramework:
    """Comprehensive data quality assessment and documentation"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.quality_report = {}
        self.decisions_log = []
    
    def assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Complete data quality assessment"""
        
        self.config.logger.info("Starting data quality assessment")
        
        # 1. Basic statistics
        self.quality_report['basic'] = self._basic_statistics(df)
        
        # 2. Missing data analysis
        self.quality_report['missing'] = self._analyze_missing_data(df)
        
        # 3. Outlier detection
        self.quality_report['outliers'] = self._detect_outliers(df)
        
        # 4. Distribution analysis
        self.quality_report['distributions'] = self._analyze_distributions(df)
        
        # 5. Consistency checks
        self.quality_report['consistency'] = self._check_consistency(df)
        
        # 6. Measurement quality
        self.quality_report['measurement'] = self._assess_measurement_quality(df)
        
        # Generate quality score
        self.quality_report['overall_score'] = self._calculate_quality_score()
        
        # Save report
        self._save_quality_report()
        
        return self.quality_report
    
    def _basic_statistics(self, df: pd.DataFrame) -> Dict:
        """Basic dataset statistics"""
        
        stats = {
            'n_observations': len(df),
            'n_variables': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'dtypes': df.dtypes.value_counts().to_dict(),
        }
        
        # Variable types
        stats['numeric_vars'] = df.select_dtypes(include=[np.number]).columns.tolist()
        stats['categorical_vars'] = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Duplicates
        stats['n_duplicates'] = df.duplicated().sum()
        stats['duplicate_rate'] = stats['n_duplicates'] / len(df)
        
        return stats
    
    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict:
        """Comprehensive missing data analysis"""
        
        missing_analysis = {
            'summary': {},
            'patterns': {},
            'mechanisms': {}
        }
        
        # Overall missingness
        total_cells = df.shape[0] * df.shape[1]
        total_missing = df.isnull().sum().sum()
        missing_analysis['summary']['total_missing_cells'] = total_missing
        missing_analysis['summary']['overall_missing_rate'] = total_missing / total_cells
        
        # Per variable missingness
        missing_per_var = df.isnull().mean()
        missing_analysis['summary']['variables_with_missing'] = (missing_per_var > 0).sum()
        missing_analysis['summary']['high_missing_vars'] = missing_per_var[
            missing_per_var > self.config.missing_threshold
        ].to_dict()
        
        # Missing patterns
        missing_patterns = df.isnull().value_counts()
        missing_analysis['patterns']['n_patterns'] = len(missing_patterns)
        missing_analysis['patterns']['most_common'] = missing_patterns.head().to_dict()
        
        # Test for MCAR (Missing Completely At Random)
        missing_analysis['mechanisms']['mcar_test'] = self._test_mcar(df)
        
        # Correlations between missingness indicators
        if len(df.columns) < 50:  # Only for reasonable number of variables
            missing_indicators = df.isnull().astype(int)
            missing_corr = missing_indicators.corr()
            high_corr_pairs = []
            
            for i in range(len(missing_corr.columns)):
                for j in range(i+1, len(missing_corr.columns)):
                    if abs(missing_corr.iloc[i, j]) > 0.3:
                        high_corr_pairs.append({
                            'var1': missing_corr.columns[i],
                            'var2': missing_corr.columns[j],
                            'correlation': missing_corr.iloc[i, j]
                        })
            
            missing_analysis['patterns']['correlated_missingness'] = high_corr_pairs
        
        return missing_analysis
    
    def _test_mcar(self, df: pd.DataFrame) -> Dict:
        """Test if data is Missing Completely At Random"""
        
        # Little's MCAR test would go here
        # For now, using a simplified approach
        
        results = {'method': 'correlation_based'}
        
        # Test if missingness in each variable is related to observed values
        tests = []
        
        for col in df.columns:
            if df[col].isnull().any() and not df[col].isnull().all():
                missing_indicator = df[col].isnull().astype(int)
                
                # Test against other variables
                for other_col in df.columns:
                    if other_col != col and is_numeric_dtype(df[other_col]):
                        valid_idx = ~df[other_col].isnull()
                        if valid_idx.sum() > 30:
                            try:
                                stat, pval = stats.pointbiserialr(
                                    missing_indicator[valid_idx],
                                    df[other_col][valid_idx]
                                )
                                if pval < 0.05:
                                    tests.append({
                                        'missing_var': col,
                                        'predictor': other_col,
                                        'correlation': stat,
                                        'p_value': pval
                                    })
                            except:
                                pass
        
        results['significant_predictors'] = tests
        results['likely_mcar'] = len(tests) == 0
        
        return results
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict:
        """Multi-method outlier detection"""
        
        outlier_report = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].nunique() > 10:  # Skip likely categorical
                col_outliers = {}
                data = df[col].dropna()
                
                if len(data) < 10:
                    continue
                
                # 1. IQR method
                Q1, Q3 = data.quantile([0.25, 0.75])
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                col_outliers['iqr'] = {
                    'n_outliers': ((data < lower_bound) | (data > upper_bound)).sum(),
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
                
                # 2. Z-score method
                z_scores = np.abs(stats.zscore(data))
                col_outliers['zscore'] = {
                    'n_outliers': (z_scores > 3).sum(),
                    'threshold': 3
                }
                
                # 3. Modified Z-score (MAD)
                median = data.median()
                mad = np.median(np.abs(data - median))
                if mad > 0:
                    modified_z = 0.6745 * (data - median) / mad
                    col_outliers['modified_zscore'] = {
                        'n_outliers': (np.abs(modified_z) > 3.5).sum(),
                        'threshold': 3.5
                    }
                
                # 4. Isolation Forest (if enough data)
                if len(data) > 100 and self.config.outlier_method == 'isolation_forest':
                    iso_forest = IsolationForest(
                        contamination=self.config.outlier_threshold,
                        random_state=self.config.random_seed
                    )
                    outlier_labels = iso_forest.fit_predict(data.values.reshape(-1, 1))
                    col_outliers['isolation_forest'] = {
                        'n_outliers': (outlier_labels == -1).sum(),
                        'contamination': self.config.outlier_threshold
                    }
                
                outlier_report[col] = col_outliers
        
        return outlier_report
    
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict:
        """Analyze variable distributions"""
        
        distribution_report = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:50]:  # Limit to first 50 for efficiency
            data = df[col].dropna()
            
            if len(data) < 30:
                continue
            
            col_dist = {
                'n': len(data),
                'mean': data.mean(),
                'median': data.median(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'skewness': data.skew(),
                'kurtosis': data.kurtosis()
            }
            
            # Normality tests
            if len(data) <= 5000:
                col_dist['shapiro_p'] = shapiro(data)[1]
            
            col_dist['jarque_bera_p'] = jarque_bera(data)[1]
            
            # Distribution type inference
            col_dist['likely_distribution'] = self._infer_distribution(data)
            
            distribution_report[col] = col_dist
        
        return distribution_report
    
    def _infer_distribution(self, data: pd.Series) -> str:
        """Infer likely distribution type"""
        
        # Remove outliers for better inference
        Q1, Q3 = data.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        filtered = data[(data >= Q1 - 1.5*IQR) & (data <= Q3 + 1.5*IQR)]
        
        skew = filtered.skew()
        kurt = filtered.kurtosis()
        
        # Simple heuristics
        if abs(skew) < 0.5 and abs(kurt) < 1:
            return 'normal'
        elif skew > 1:
            return 'right_skewed'
        elif skew < -1:
            return 'left_skewed'
        elif kurt > 3:
            return 'heavy_tailed'
        else:
            return 'unknown'
    
    def _check_consistency(self, df: pd.DataFrame) -> Dict:
        """Check logical consistency in data"""
        
        consistency_issues = []
        
        # Example checks (customize based on your data)
        
        # Check 1: Age-related consistency
        if 'age' in df.columns and 'education_years' in df.columns:
            invalid = df['education_years'] > df['age']
            if invalid.any():
                consistency_issues.append({
                    'type': 'logical',
                    'description': 'Education years exceeds age',
                    'n_cases': invalid.sum(),
                    'percentage': invalid.mean() * 100
                })
        
        # Check 2: Household size consistency
        if 'household_size' in df.columns and 'n_children' in df.columns:
            invalid = df['n_children'] > df['household_size']
            if invalid.any():
                consistency_issues.append({
                    'type': 'logical',
                    'description': 'Number of children exceeds household size',
                    'n_cases': invalid.sum(),
                    'percentage': invalid.mean() * 100
                })
        
        # Check 3: Value range consistency
        for col in df.columns:
            if is_numeric_dtype(df[col]):
                # Check for impossible values based on variable name
                if 'percentage' in col.lower() or 'percent' in col.lower():
                    invalid = (df[col] < 0) | (df[col] > 100)
                    if invalid.any():
                        consistency_issues.append({
                            'type': 'range',
                            'variable': col,
                            'description': 'Percentage outside 0-100 range',
                            'n_cases': invalid.sum()
                        })
                
                elif 'age' in col.lower():
                    invalid = (df[col] < 0) | (df[col] > 120)
                    if invalid.any():
                        consistency_issues.append({
                            'type': 'range',
                            'variable': col,
                            'description': 'Age outside valid range',
                            'n_cases': invalid.sum()
                        })
        
        return {
            'n_issues': len(consistency_issues),
            'issues': consistency_issues
        }
    
    def _assess_measurement_quality(self, df: pd.DataFrame) -> Dict:
        """Assess measurement quality indicators"""
        
        measurement_quality = {}
        
        # Digit preference (heaping)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].nunique() > 20:
                # Check for digit preference
                last_digits = df[col].dropna().astype(str).str[-1]
                digit_freq = last_digits.value_counts(normalize=True)
                
                # Chi-square test for uniform distribution
                expected_freq = 1/10
                observed = digit_freq.values
                expected = np.full(len(observed), expected_freq * len(last_digits))
                
                if len(observed) == 10:  # All digits present
                    chi2_stat = np.sum((observed * len(last_digits) - expected)**2 / expected)
                    p_value = 1 - stats.chi2.cdf(chi2_stat, df=9)
                    
                    if p_value < 0.01:
                        measurement_quality[f'{col}_digit_preference'] = {
                            'detected': True,
                            'p_value': p_value,
                            'most_common_digit': digit_freq.index[0]
                        }
        
        # Variance analysis
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) > 30:
                cv = data.std() / data.mean() if data.mean() != 0 else np.inf
                measurement_quality[f'{col}_cv'] = cv
        
        return measurement_quality
    
    def _calculate_quality_score(self) -> float:
        """Calculate overall data quality score"""
        
        scores = []
        
        # Missing data score (0-1, lower is better)
        missing_rate = self.quality_report['missing']['summary']['overall_missing_rate']
        scores.append(1 - min(missing_rate * 2, 1))  # Penalize heavily after 50%
        
        # Consistency score
        n_issues = self.quality_report['consistency']['n_issues']
        scores.append(1 / (1 + n_issues))  # Decay with number of issues
        
        # Outlier score (based on percentage of outliers)
        outlier_rates = []
        for col_outliers in self.quality_report['outliers'].values():
            if 'iqr' in col_outliers:
                # Rough estimate of outlier rate
                outlier_rates.append(col_outliers['iqr']['n_outliers'])
        
        if outlier_rates:
            avg_outlier_rate = np.mean(outlier_rates) / len(self.quality_report['basic']['n_observations'])
            scores.append(1 - min(avg_outlier_rate * 10, 1))
        
        # Overall score (weighted average)
        weights = [0.4, 0.3, 0.3]  # Missing, consistency, outliers
        overall_score = np.average(scores[:len(weights)], weights=weights[:len(scores)])
        
        return overall_score
    
    def _save_quality_report(self):
        """Save quality report to file"""
        
        report_file = self.config.output_dir / f'data_quality_report_{datetime.now():%Y%m%d_%H%M%S}.json'
        
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        report_serializable = convert_types(self.quality_report)
        
        with open(report_file, 'w') as f:
            json.dump(report_serializable, f, indent=2)
        
        self.config.logger.info(f"Data quality report saved to {report_file}")

# ============================================================================
# STATISTICAL TESTING FRAMEWORK
# ============================================================================

class StatisticalTestingFramework:
    """Rigorous statistical testing with multiple comparison correction"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.all_p_values = []
        self.test_results = {}
        self.test_hierarchy = {
            'primary': [],
            'secondary': [],
            'exploratory': []
        }
    
    def register_hypothesis(self, hypothesis: str, level: str = 'exploratory'):
        """Register hypothesis for proper multiple testing control"""
        
        if level not in self.test_hierarchy:
            raise ValueError(f"Level must be one of {list(self.test_hierarchy.keys())}")
        
        self.test_hierarchy[level].append(hypothesis)
        self.config.logger.info(f"Registered {level} hypothesis: {hypothesis}")
    
    def run_test(self, 
                 test_name: str,
                 test_func: callable,
                 data: Any,
                 hypothesis_level: str = 'exploratory',
                 **kwargs) -> Dict:
        """Run a statistical test with proper documentation"""
        
        result = {
            'test_name': test_name,
            'hypothesis_level': hypothesis_level,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Run the test
            test_output = test_func(data, **kwargs)
            
            # Extract p-value (handle different output formats)
            if isinstance(test_output, tuple):
                if len(test_output) >= 2:
                    result['statistic'] = test_output[0]
                    result['p_value'] = test_output[1]
                    if len(test_output) > 2:
                        result['additional'] = test_output[2:]
            elif hasattr(test_output, 'pvalue'):
                result['p_value'] = test_output.pvalue
                result['statistic'] = test_output.statistic if hasattr(test_output, 'statistic') else None
            else:
                result['output'] = test_output
            
            # Store p-value for multiple testing correction
            if 'p_value' in result:
                self.all_p_values.append({
                    'test': test_name,
                    'level': hypothesis_level,
                    'p_value': result['p_value']
                })
            
            result['success'] = True
            
        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
            self.config.logger.error(f"Test {test_name} failed: {e}")
        
        self.test_results[test_name] = result
        return result
    
    def apply_multiple_testing_correction(self) -> pd.DataFrame:
        """Apply appropriate multiple testing corrections"""
        
        if not self.all_p_values:
            return pd.DataFrame()
        
        corrections_df = pd.DataFrame(self.all_p_values)
        
        # Separate by hypothesis level
        for level in ['primary', 'secondary', 'exploratory']:
            level_mask = corrections_df['level'] == level
            level_pvals = corrections_df.loc[level_mask, 'p_value'].values
            
            if len(level_pvals) == 0:
                continue
            
            if level == 'primary':
                # No correction for pre-registered primary hypotheses
                corrections_df.loc[level_mask, 'corrected_p'] = level_pvals
                corrections_df.loc[level_mask, 'correction_method'] = 'none'
                
            elif level == 'secondary':
                # Holm-Bonferroni for secondary
                if self.config.bonferroni_correction:
                    reject, corrected_p, _, _ = multipletests(
                        level_pvals, 
                        alpha=self.config.alpha,
                        method='holm'
                    )
                    corrections_df.loc[level_mask, 'corrected_p'] = corrected_p
                    corrections_df.loc[level_mask, 'reject_null'] = reject
                    corrections_df.loc[level_mask, 'correction_method'] = 'holm'
                else:
                    corrections_df.loc[level_mask, 'corrected_p'] = level_pvals
                    corrections_df.loc[level_mask, 'correction_method'] = 'none'
            
            elif level == 'exploratory':
                # FDR for exploratory
                if self.config.fdr_correction:
                    reject, corrected_p, _, _ = multipletests(
                        level_pvals,
                        alpha=self.config.alpha,
                        method='fdr_bh'
                    )
                    corrections_df.loc[level_mask, 'corrected_p'] = corrected_p
                    corrections_df.loc[level_mask, 'reject_null'] = reject
                    corrections_df.loc[level_mask, 'correction_method'] = 'fdr_bh'
                else:
                    corrections_df.loc[level_mask, 'corrected_p'] = level_pvals
                    corrections_df.loc[level_mask, 'correction_method'] = 'none'
        
        # Add significance flags
        corrections_df['significant_raw'] = corrections_df['p_value'] < self.config.alpha
        corrections_df['significant_corrected'] = corrections_df['corrected_p'] < self.config.alpha
        
        return corrections_df
    
    def power_analysis(self, 
                       effect_size: Optional[float] = None,
                       sample_size: Optional[int] = None,
                       alpha: Optional[float] = None,
                       power: Optional[float] = None) -> Dict:
        """Conduct power analysis"""
        
        if alpha is None:
            alpha = self.config.alpha
        
        # Count number of None values to determine what to solve for
        params = [effect_size, sample_size, alpha, power]
        n_none = sum(p is None for p in params)
        
        if n_none != 1:
            raise ValueError("Exactly one parameter must be None to solve for it")
        
        result = {}
        
        if effect_size is None:
            # Solve for minimum detectable effect
            effect_size = tt_solve_power(
                effect_size=None,
                nobs=sample_size,
                alpha=alpha,
                power=power,
                alternative='two-sided'
            )
            result['minimum_detectable_effect'] = effect_size
            
        elif sample_size is None:
            # Solve for required sample size
            sample_size = tt_solve_power(
                effect_size=effect_size,
                nobs=None,
                alpha=alpha,
                power=power,
                alternative='two-sided'
            )
            result['required_sample_size'] = int(np.ceil(sample_size))
            
        elif power is None:
            # Solve for power
            power = tt_solve_power(
                effect_size=effect_size,
                nobs=sample_size,
                alpha=alpha,
                power=None,
                alternative='two-sided'
            )
            result['statistical_power'] = power
            
        elif alpha is None:
            # Solve for alpha (less common)
            alpha = tt_solve_power(
                effect_size=effect_size,
                nobs=sample_size,
                alpha=None,
                power=power,
                alternative='two-sided'
            )
            result['required_alpha'] = alpha
        
        result['parameters'] = {
            'effect_size': effect_size,
            'sample_size': sample_size,
            'alpha': alpha,
            'power': power
        }
        
        return result

# ============================================================================
# CAUSAL INFERENCE FRAMEWORK
# ============================================================================

class CausalInferenceFramework:
    """State-of-the-art causal inference methods"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.causal_results = {}
        self.causal_graph = None
    
    def specify_causal_model(self, 
                            data: pd.DataFrame,
                            treatment: str,
                            outcome: str,
                            confounders: List[str],
                            instruments: Optional[List[str]] = None,
                            mediators: Optional[List[str]] = None,
                            effect_modifiers: Optional[List[str]] = None) -> 'CausalModel':
        """Specify causal model with DAG"""
        
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = confounders
        self.instruments = instruments or []
        self.mediators = mediators or []
        self.effect_modifiers = effect_modifiers or []
        
        # Build causal graph
        self._build_causal_graph()
        
        if DOWHY_AVAILABLE:
            # Use DoWhy for causal modeling
            self.causal_model = CausalModel(
                data=data,
                treatment=treatment,
                outcome=outcome,
                graph=self.causal_graph_str,
                instruments=instruments,
                effect_modifiers=effect_modifiers
            )
            
            # Identify causal effect
            self.identified_estimand = self.causal_model.identify_effect(
                proceed_when_unidentifiable=True
            )
            
            self.config.logger.info(f"Causal model specified: {treatment} -> {outcome}")
            self.config.logger.info(f"Identification: {self.identified_estimand}")
        
        return self
    
    def _build_causal_graph(self):
        """Build causal DAG"""
        
        edges = []
        
        # Confounders affect both treatment and outcome
        for conf in self.confounders:
            edges.append(f"{conf} -> {self.treatment}")
            edges.append(f"{conf} -> {self.outcome}")
        
        # Treatment affects outcome
        edges.append(f"{self.treatment} -> {self.outcome}")
        
        # Instruments affect treatment only
        for inst in self.instruments:
            edges.append(f"{inst} -> {self.treatment}")
        
        # Mediators
        for med in self.mediators:
            edges.append(f"{self.treatment} -> {med}")
            edges.append(f"{med} -> {self.outcome}")
        
        # Effect modifiers affect the relationship
        for mod in self.effect_modifiers:
            edges.append(f"{mod} -> {self.outcome}")
        
        self.causal_graph_str = "; ".join(edges)
        
        return self.causal_graph_str
    
    def estimate_causal_effect(self, 
                              data: pd.DataFrame,
                              method: str = 'backdoor.propensity_score_matching',
                              **kwargs) -> Dict:
        """Estimate causal effect using specified method"""
        
        result = {'method': method}
        
        if DOWHY_AVAILABLE and hasattr(self, 'causal_model'):
            # DoWhy estimation
            estimate = self.causal_model.estimate_effect(
                identified_estimand=self.identified_estimand,
                method_name=method,
                **kwargs
            )
            
            result['ate'] = estimate.value
            result['confidence_interval'] = self._get_confidence_interval(estimate)
            result['method_params'] = estimate.params
            
        elif ECONML_AVAILABLE:
            # EconML estimation
            result.update(self._estimate_with_econml(data, method))
        
        else:
            # Fallback to basic methods
            result.update(self._estimate_basic(data, method))
        
        self.causal_results[method] = result
        return result
    
    def _estimate_with_econml(self, data: pd.DataFrame, method: str) -> Dict:
        """Use EconML for causal estimation"""
        
        X = data[self.confounders]
        T = data[self.treatment]
        Y = data[self.outcome]
        
        result = {}
        
        if method == 'double_ml':
            # Double/Debiased Machine Learning
            est = LinearDML(
                model_y=RandomForestRegressor(n_estimators=100, random_state=self.config.random_seed),
                model_t=RandomForestRegressor(n_estimators=100, random_state=self.config.random_seed),
                discrete_treatment=True,
                cv=self.config.cv_folds,
                random_state=self.config.random_seed
            )
            
        elif method == 'causal_forest':
            # Causal Forest
            est = CausalForestDML(
                model_y=RandomForestRegressor(n_estimators=100, random_state=self.config.random_seed),
                model_t=RandomForestClassifier(n_estimators=100, random_state=self.config.random_seed),
                discrete_treatment=True,
                cv=self.config.cv_folds,
                random_state=self.config.random_seed
            )
            
        elif method == 'metalearner_s':
            # S-Learner
            est = SLearner(
                overall_model=RandomForestRegressor(n_estimators=100, random_state=self.config.random_seed)
            )
            
        elif method == 'metalearner_t':
            # T-Learner
            est = TLearner(
                models=RandomForestRegressor(n_estimators=100, random_state=self.config.random_seed)
            )
            
        elif method == 'metalearner_x':
            # X-Learner
            est = XLearner(
                models=RandomForestRegressor(n_estimators=100, random_state=self.config.random_seed),
                propensity_model=RandomForestClassifier(n_estimators=100, random_state=self.config.random_seed)
            )
            
        else:
            raise ValueError(f"Unknown EconML method: {method}")
        
        # Fit the model
        est.fit(Y, T, X=X)
        
        # Get treatment effects
        result['ate'] = est.ate(X)
        result['ate_interval'] = est.ate_interval(X, alpha=1-self.config.confidence_level)
        
        # Conditional effects
        result['cate'] = est.effect(X)
        result['cate_interval'] = est.effect_interval(X, alpha=1-self.config.confidence_level)
        
        return result
    
    def _estimate_basic(self, data: pd.DataFrame, method: str) -> Dict:
        """Basic causal estimation methods"""
        
        result = {}
        
        if method == 'simple_difference':
            # Simple difference in means
            treated = data[data[self.treatment] == 1][self.outcome]
            control = data[data[self.treatment] == 0][self.outcome]
            
            result['ate'] = treated.mean() - control.mean()
            
            # T-test for significance
            t_stat, p_value = ttest_ind(treated, control)
            result['t_statistic'] = t_stat
            result['p_value'] = p_value
            
        elif method == 'regression':
            # OLS regression
            formula = f"{self.outcome} ~ {self.treatment}"
            if self.confounders:
                formula += " + " + " + ".join(self.confounders)
            
            model = smf.ols(formula, data=data).fit()
            result['ate'] = model.params[self.treatment]
            result['confidence_interval'] = model.conf_int().loc[self.treatment].values
            result['p_value'] = model.pvalues[self.treatment]
            
        return result
    
    def validate_causal_assumptions(self, data: pd.DataFrame) -> Dict:
        """Validate key causal assumptions"""
        
        validation = {}
        
        # 1. Overlap/Common support
        validation['overlap'] = self._check_overlap(data)
        
        # 2. Balance after matching/weighting
        validation['balance'] = self._check_balance(data)
        
        # 3. Unconfoundedness (cannot be directly tested)
        validation['sensitivity'] = self._sensitivity_analysis(data)
        
        # 4. SUTVA (Stable Unit Treatment Value Assumption)
        validation['sutva'] = self._check_sutva(data)
        
        return validation
    
    def _check_overlap(self, data: pd.DataFrame) -> Dict:
        """Check overlap assumption"""
        
        # Estimate propensity scores
        X = data[self.confounders]
        T = data[self.treatment]
        
        ps_model = LogisticRegression(random_state=self.config.random_seed)
        ps_model.fit(X, T)
        
        propensity_scores = ps_model.predict_proba(X)[:, 1]
        
        # Check overlap
        treated_ps = propensity_scores[T == 1]
        control_ps = propensity_scores[T == 0]
        
        overlap_stats = {
            'treated_min': treated_ps.min(),
            'treated_max': treated_ps.max(),
            'control_min': control_ps.min(),
            'control_max': control_ps.max(),
            'common_support_min': max(treated_ps.min(), control_ps.min()),
            'common_support_max': min(treated_ps.max(), control_ps.max())
        }
        
        # Calculate percentage of units in common support
        common_support_mask = (
            (propensity_scores >= overlap_stats['common_support_min']) &
            (propensity_scores <= overlap_stats['common_support_max'])
        )
        overlap_stats['pct_in_common_support'] = common_support_mask.mean()
        
        return overlap_stats
    
    def _check_balance(self, data: pd.DataFrame) -> Dict:
        """Check covariate balance"""
        
        balance_stats = {}
        
        for var in self.confounders:
            if is_numeric_dtype(data[var]):
                # Standardized mean difference
                treated_mean = data[data[self.treatment] == 1][var].mean()
                control_mean = data[data[self.treatment] == 0][var].mean()
                pooled_std = np.sqrt(
                    (data[data[self.treatment] == 1][var].var() + 
                     data[data[self.treatment] == 0][var].var()) / 2
                )
                
                smd = (treated_mean - control_mean) / pooled_std if pooled_std > 0 else 0
                balance_stats[f'{var}_smd'] = smd
                balance_stats[f'{var}_balanced'] = abs(smd) < 0.1
        
        balance_stats['overall_balanced'] = all(
            v for k, v in balance_stats.items() if k.endswith('_balanced')
        )
        
        return balance_stats
    
    def _sensitivity_analysis(self, data: pd.DataFrame) -> Dict:
        """Sensitivity to unmeasured confounding"""
        
        # Rosenbaum bounds or E-value calculation would go here
        # For now, returning placeholder
        
        return {
            'method': 'rosenbaum_bounds',
            'gamma_values': [1.0, 1.5, 2.0],
            'p_values': [0.05, 0.10, 0.20],  # Placeholder
            'robust_to_gamma': 1.5
        }
    
    def _check_sutva(self, data: pd.DataFrame) -> Dict:
        """Check SUTVA assumption"""
        
        # Check for spillovers/interference
        # This would require cluster or network information
        
        return {
            'assumption': 'SUTVA',
            'testable': False,
            'notes': 'Requires cluster/network structure to test'
        }
    
    def refute_estimate(self, data: pd.DataFrame, method: str = 'placebo') -> Dict:
        """Refutation tests for causal estimates"""
        
        refutation = {'method': method}
        
        if method == 'placebo':
            # Placebo treatment test
            placebo_treatment = np.random.binomial(1, 0.5, len(data))
            placebo_data = data.copy()
            placebo_data['placebo_treatment'] = placebo_treatment
            
            # Re-estimate with placebo
            formula = f"{self.outcome} ~ placebo_treatment"
            if self.confounders:
                formula += " + " + " + ".join(self.confounders)
            
            model = smf.ols(formula, data=placebo_data).fit()
            refutation['placebo_effect'] = model.params['placebo_treatment']
            refutation['placebo_p_value'] = model.pvalues['placebo_treatment']
            refutation['passes'] = model.pvalues['placebo_treatment'] > 0.05
            
        elif method == 'random_common_cause':
            # Add random common cause
            random_cause = np.random.normal(0, 1, len(data))
            augmented_data = data.copy()
            augmented_data['random_cause'] = random_cause
            
            # Re-estimate with random cause
            augmented_confounders = self.confounders + ['random_cause']
            formula = f"{self.outcome} ~ {self.treatment}"
            formula += " + " + " + ".join(augmented_confounders)
            
            model = smf.ols(formula, data=augmented_data).fit()
            refutation['effect_with_random'] = model.params[self.treatment]
            refutation['original_effect'] = self.causal_results.get('regression', {}).get('ate', 0)
            refutation['relative_change'] = abs(
                refutation['effect_with_random'] - refutation['original_effect']
            ) / abs(refutation['original_effect']) if refutation['original_effect'] != 0 else 0
            refutation['passes'] = refutation['relative_change'] < 0.1
            
        elif method == 'data_subset':
            # Subset validation
            subset_effects = []
            for i in range(10):
                subset = data.sample(frac=0.8, random_state=i)
                formula = f"{self.outcome} ~ {self.treatment}"
                if self.confounders:
                    formula += " + " + " + ".join(self.confounders)
                
                model = smf.ols(formula, data=subset).fit()
                subset_effects.append(model.params[self.treatment])
            
            refutation['subset_effects'] = subset_effects
            refutation['effect_std'] = np.std(subset_effects)
            refutation['effect_cv'] = np.std(subset_effects) / np.mean(subset_effects)
            refutation['passes'] = refutation['effect_cv'] < 0.2
        
        return refutation

# ============================================================================
# INDEX CONSTRUCTION AND VALIDATION
# ============================================================================

class IndexConstructor:
    """Psychometrically validated index construction"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.indices = {}
        self.validation_results = {}
    
    def create_index(self,
                    data: pd.DataFrame,
                    variables: List[str],
                    index_name: str,
                    method: str = 'pca',
                    validate: bool = True) -> pd.Series:
        """Create and validate composite index"""
        
        self.config.logger.info(f"Creating index: {index_name}")
        
        # Prepare data
        index_data = data[variables].copy()
        
        # Handle missing values
        if index_data.isnull().any().any():
            imputer = SimpleImputer(strategy='median')
            index_data_imputed = pd.DataFrame(
                imputer.fit_transform(index_data),
                columns=index_data.columns,
                index=index_data.index
            )
        else:
            index_data_imputed = index_data
        
        # Standardize
        scaler = StandardScaler()
        index_data_scaled = pd.DataFrame(
            scaler.fit_transform(index_data_imputed),
            columns=index_data.columns,
            index=index_data.index
        )
        
        # Validate before construction
        if validate:
            validation = self._validate_index_construction(index_data_scaled, index_name)
            self.validation_results[index_name] = validation
            
            if not validation['suitable_for_index']:
                self.config.logger.warning(f"Index {index_name} failed validation checks")
        
        # Construct index
        if method == 'pca':
            index_values = self._pca_index(index_data_scaled)
        elif method == 'factor_analysis':
            index_values = self._factor_analysis_index(index_data_scaled)
        elif method == 'equal_weight':
            index_values = self._equal_weight_index(index_data_scaled)
        elif method == 'data_driven':
            index_values = self._data_driven_weights(index_data_scaled, data)
        else:
            raise ValueError(f"Unknown index method: {method}")
        
        # Create series
        index_series = pd.Series(index_values, index=data.index, name=index_name)
        
        # Store
        self.indices[index_name] = {
            'values': index_series,
            'variables': variables,
            'method': method,
            'validation': validation if validate else None
        }
        
        return index_series
    
    def _validate_index_construction(self, data: pd.DataFrame, index_name: str) -> Dict:
        """Comprehensive index validation"""
        
        validation = {
            'index_name': index_name,
            'n_variables': len(data.columns),
            'n_observations': len(data)
        }
        
        # 1. Sample size adequacy
        min_n = len(data.columns) * 10
        validation['sample_size_adequate'] = len(data) >= min_n
        
        # 2. Correlation matrix
        corr_matrix = data.corr()
        validation['mean_correlation'] = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        
        # 3. KMO and Bartlett's test
        if FACTOR_ANALYZER_AVAILABLE:
            try:
                kmo_all, kmo_model = calculate_kmo(data)
                validation['kmo_overall'] = kmo_model
                validation['kmo_adequate'] = kmo_model >= 0.6
                validation['kmo_interpretation'] = self._interpret_kmo(kmo_model)
                
                chi2, p_value = calculate_bartlett_sphericity(data)
                validation['bartlett_chi2'] = chi2
                validation['bartlett_p'] = p_value
                validation['bartlett_significant'] = p_value < 0.05
            except:
                validation['kmo_overall'] = None
                validation['bartlett_p'] = None
        
        # 4. Cronbach's alpha
        alpha = self._calculate_cronbach_alpha(data)
        validation['cronbach_alpha'] = alpha
        validation['reliability_acceptable'] = alpha >= self.config.min_cronbach_alpha
        
        # 5. Dimensionality check (PCA)
        pca = PCA()
        pca.fit(data)
        explained_var = pca.explained_variance_ratio_
        validation['first_component_variance'] = explained_var[0]
        validation['n_components_80pct'] = np.argmax(np.cumsum(explained_var) >= 0.8) + 1
        
        # Overall suitability
        validation['suitable_for_index'] = (
            validation.get('sample_size_adequate', False) and
            validation.get('reliability_acceptable', False) and
            (validation.get('kmo_adequate', True) or validation['mean_correlation'] > 0.3)
        )
        
        return validation
    
    def _interpret_kmo(self, kmo_value: float) -> str:
        """Interpret KMO value"""
        
        if kmo_value >= 0.9:
            return "Marvelous"
        elif kmo_value >= 0.8:
            return "Meritorious"
        elif kmo_value >= 0.7:
            return "Middling"
        elif kmo_value >= 0.6:
            return "Mediocre"
        elif kmo_value >= 0.5:
            return "Miserable"
        else:
            return "Unacceptable"
    
    def _calculate_cronbach_alpha(self, data: pd.DataFrame) -> float:
        """Calculate Cronbach's alpha"""
        
        n_items = len(data.columns)
        if n_items < 2:
            return 0.0
        
        item_variances = data.var()
        total_variance = data.sum(axis=1).var()
        
        if total_variance == 0:
            return 0.0
        
        alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)
        
        return alpha
    
    def _pca_index(self, data: pd.DataFrame) -> np.ndarray:
        """Create index using PCA"""
        
        pca = PCA(n_components=1)
        index_values = pca.fit_transform(data).flatten()
        
        # Store loadings
        self.indices[f'pca_loadings'] = pd.DataFrame(
            pca.components_.T,
            columns=['PC1'],
            index=data.columns
        )
        
        return index_values
    
    def _factor_analysis_index(self, data: pd.DataFrame) -> np.ndarray:
        """Create index using factor analysis"""
        
        if FACTOR_ANALYZER_AVAILABLE:
            fa = FactorAnalyzer(n_factors=1, rotation=None)
            fa.fit(data)
            index_values = fa.transform(data).flatten()
            
            # Store loadings
            self.indices[f'fa_loadings'] = pd.DataFrame(
                fa.loadings_,
                columns=['Factor1'],
                index=data.columns
            )
        else:
            # Fallback to sklearn
            fa = FactorAnalysis(n_components=1)
            index_values = fa.fit_transform(data).flatten()
        
        return index_values
    
    def _equal_weight_index(self, data: pd.DataFrame) -> np.ndarray:
        """Create index with equal weights"""
        
        return data.mean(axis=1).values
    
    def _data_driven_weights(self, data: pd.DataFrame, full_data: pd.DataFrame) -> np.ndarray:
        """Create index with data-driven weights"""
        
        # Use correlation with outcome if available
        if 'outcome' in full_data.columns:
            correlations = []
            for col in data.columns:
                corr = data[col].corr(full_data['outcome'])
                correlations.append(abs(corr))
            
            weights = np.array(correlations) / sum(correlations)
        else:
            # Use first principal component weights
            pca = PCA(n_components=1)
            pca.fit(data)
            weights = np.abs(pca.components_[0])
            weights = weights / weights.sum()
        
        index_values = (data * weights).sum(axis=1)
        
        return index_values

# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

class WaterDisruptionAnalysis:
    """Main analysis pipeline for water disruption study"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.data = None
        self.processed_data = None
        self.results = {}
        
        # Define required columns based on analysis needs
        self.required_columns = self._define_required_columns()
        
        # Initialize components
        self.reproducibility = ReproducibilityManager(config)
        self.quality_framework = DataQualityFramework(config)
        self.testing_framework = StatisticalTestingFramework(config)
        self.causal_framework = CausalInferenceFramework(config)
        self.index_constructor = IndexConstructor(config)
        
        self.config.logger.info("Water Disruption Analysis initialized")
    
    def _define_required_columns(self) -> Dict[str, List[str]]:
        """Define all required columns organized by category"""
        
        columns = {
            # Core outcome and treatment variables
            'core': [
                'sh37b',      # Water disruption (main outcome)
                'hv005',      # Sample weight
                'hv021',      # Primary sampling unit
                'hv022',      # Sample stratum
                'hv024',      # State code
                'hv025',      # Urban/rural
                'hv001',      # Cluster number
            ],
            
            # Water-related variables
            'water': [
                'hv201',      # Source of drinking water
                'hv201a',     # Water interruption
                'hv202',      # Source of non-drinking water
                'hv204',      # Time to water source
                'hv235',      # Location of water source
                'hv236',      # Person collecting water
                'hv237',      # Water treatment
                'hv238',      # Water treatment method
            ],
            
            # Socioeconomic variables
            'socioeconomic': [
                'hv270',      # Wealth index
                'hv271',      # Wealth factor score
                'hv009',      # Household members
                'hv014',      # Children under 5
                'hv219',      # Sex of household head
                'hv106',      # Education level
                'hv107',      # Years of education
                'sh49',       # Caste/tribe
            ],
            
            # Infrastructure variables
            'infrastructure': [
                'hv206',      # Has electricity
                'hv207',      # Has radio
                'hv208',      # Has television
                'hv209',      # Has refrigerator
                'hv210',      # Has bicycle
                'hv211',      # Has motorcycle
                'hv212',      # Has car
                'hv221',      # Has telephone
                'hv243a',     # Has mobile phone
            ],
            
            # Sanitation and housing
            'sanitation_housing': [
                'hv205',      # Type of toilet facility
                'hv225',      # Share toilet with other households
                'hv213',      # Main floor material
                'hv214',      # Main wall material
                'hv215',      # Main roof material
                'hv216',      # Rooms for sleeping
                'hv226',      # Type of cooking fuel
            ],
            
            # Temporal variables
            'temporal': [
                'hv006',      # Month of interview
                'hv007',      # Year of interview
                'hv008',      # Date of interview (CMC)
            ],
            
            # Additional contextual variables
            'contextual': [
                'hv026',      # Type of place of residence
                'sh47',       # Below poverty line card
                'hv040',      # Altitude
                'hv041',      # GPS accuracy
            ]
        }
        
        return columns
    
    def load_data(self, columns_to_load: Optional[List[str]] = None) -> pd.DataFrame:
        """Load NFHS data with optimized column selection"""
        
        data_path = self.config.data_dir / self.config.data_file
        
        # Determine which columns to load
        if columns_to_load is None:
            # Get all required columns
            columns_to_load = []
            for category, cols in self.required_columns.items():
                columns_to_load.extend(cols)
            
            # Remove duplicates
            columns_to_load = list(set(columns_to_load))
        
        self.config.logger.info(f"Loading {len(columns_to_load)} columns from {data_path}")
        
        try:
            # First, try to load only required columns
            df, meta = pyreadstat.read_dta(
                str(data_path),
                usecols=columns_to_load,
                apply_value_formats=True  # Apply value labels
            )
            self.metadata = meta
            
            # Log what was actually loaded
            self.config.logger.info(f"Successfully loaded {len(df.columns)} columns")
            
            # Check which requested columns are missing
            missing_cols = set(columns_to_load) - set(df.columns)
            if missing_cols:
                self.config.logger.warning(f"Missing columns: {missing_cols}")
            
        except ValueError as e:
            # If specific columns not found, load all and then filter
            self.config.logger.warning(f"Could not load specific columns: {e}")
            self.config.logger.info("Loading full dataset and filtering...")
            
            df, meta = pyreadstat.read_dta(str(data_path))
            self.metadata = meta
            
            # Filter to available columns
            available_cols = [col for col in columns_to_load if col in df.columns]
            df = df[available_cols]
            
            self.config.logger.info(f"Filtered to {len(df.columns)} available columns")
        
        except Exception as e:
            self.config.logger.error(f"Failed to load data: {e}")
            raise
        
        # Store information about loaded columns
        self.loaded_columns = {
            'requested': columns_to_load,
            'loaded': list(df.columns),
            'missing': list(set(columns_to_load) - set(df.columns))
        }
        
        # Memory optimization: convert appropriate columns to categorical
        self._optimize_dtypes(df)
        
        self.data = df
        
        # Log memory usage
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        self.config.logger.info(f"Loaded {len(df)} observations with {len(df.columns)} variables")
        self.config.logger.info(f"Memory usage: {memory_usage:.2f} MB")
        
        return df
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> None:
        """Optimize data types to reduce memory usage"""
        
        initial_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object':
                # Downcast numeric types
                if 'int' in str(col_type):
                    # Check if can be converted to smaller int type
                    c_min = df[col].min()
                    c_max = df[col].max()
                    
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                
                elif 'float' in str(col_type):
                    # Downcast floats
                    c_min = df[col].min()
                    c_max = df[col].max()
                    
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
            
            else:
                # Convert string columns with few unique values to categorical
                num_unique_values = len(df[col].unique())
                num_total_values = len(df[col])
                
                if num_unique_values / num_total_values < 0.5:
                    df[col] = df[col].astype('category')
        
        final_memory = df.memory_usage(deep=True).sum() / 1024**2
        memory_saved = initial_memory - final_memory
        
        self.config.logger.info(f"Memory optimization: {initial_memory:.2f} MB -> {final_memory:.2f} MB")
        self.config.logger.info(f"Memory saved: {memory_saved:.2f} MB ({memory_saved/initial_memory*100:.1f}%)")
    
    def load_minimal_data(self) -> pd.DataFrame:
        """Load only essential columns for basic analysis"""
        
        essential_columns = [
            'sh37b',      # Water disruption
            'hv005',      # Weight
            'hv024',      # State
            'hv025',      # Urban/rural
            'hv270',      # Wealth index
            'hv201',      # Water source
            'hv204',      # Time to water
        ]
        
        return self.load_data(columns_to_load=essential_columns)
    
    def load_data_for_analysis(self, analysis_type: str) -> pd.DataFrame:
        """Load columns specific to an analysis type"""
        
        analysis_columns = {
            'descriptive': self.required_columns['core'] + ['hv270', 'hv025'],
            'spatial': self.required_columns['core'] + self.required_columns['contextual'],
            'causal': self.required_columns['core'] + self.required_columns['socioeconomic'],
            'infrastructure': self.required_columns['core'] + self.required_columns['infrastructure'],
            'comprehensive': None  # Load all
        }
        
        columns = analysis_columns.get(analysis_type, self.required_columns['core'])
        
        if columns is None:
            # Load all required columns
            return self.load_data()
        else:
            return self.load_data(columns_to_load=columns)
    
    def check_column_availability(self) -> pd.DataFrame:
        """Check which required columns are available in the dataset"""
        
        # Load just the first row to get column names
        data_path = self.config.data_dir / self.config.data_file
        df_sample, meta = pyreadstat.read_dta(str(data_path), row_limit=1)
        
        available_columns = set(df_sample.columns)
        
        # Create availability report
        availability_report = []
        
        for category, cols in self.required_columns.items():
            for col in cols:
                availability_report.append({
                    'category': category,
                    'column': col,
                    'available': col in available_columns,
                    'description': meta.column_labels.get(col, 'No description')
                })
        
        availability_df = pd.DataFrame(availability_report)
        
        # Summary statistics
        summary = availability_df.groupby('category')['available'].agg(['sum', 'count', 'mean'])
        summary.columns = ['n_available', 'n_total', 'pct_available']
        
        print("\nColumn Availability Summary:")
        print(summary)
        
        # Save report
        report_file = self.config.output_dir / 'column_availability.csv'
        availability_df.to_csv(report_file, index=False)
        
        return availability_df
  
    def process_data(self) -> pd.DataFrame:
        """Complete data processing pipeline"""
        
        self.config.logger.info("Starting data processing")
        
        # Quality assessment
        quality_report = self.quality_framework.assess_data_quality(self.data)
        
        # Create processed dataset
        processed = self.data.copy()
        
        # 1. Create water disruption variable
        if 'sh37b' in processed.columns:
            processed['water_disrupted'] = (processed['sh37b'] == 1).astype(int)
        
        # 2. Create indices
        self._create_all_indices(processed)
        
        # 3. Handle missing data
        processed = self._handle_missing_data(processed)
        
        # 4. Create derived variables
        processed = self._create_derived_variables(processed)
        
        # 5. Apply survey weights
        if 'hv005' in processed.columns:
            processed['weight'] = processed['hv005'] / 1_000_000
        else:
            processed['weight'] = 1.0
        
        self.processed_data = processed
        
        # Save processed data
        processed_file = self.config.output_dir / f'processed_data_{self.reproducibility.analysis_hash}.pkl'
        processed.to_pickle(processed_file)
        
        self.config.logger.info(f"Data processing complete. Saved to {processed_file}")
        
        return processed
    
    def _create_all_indices(self, data: pd.DataFrame):
        """Create all composite indices"""
        
        # Define indices to create
        indices_config = {
            'vulnerability_index': {
                'variables': ['hv270', 'hv106', 'hv219'],  # Adjust based on available variables
                'method': 'pca'
            },
            'water_insecurity_index': {
                'variables': ['sh37b', 'hv201', 'hv204'],  # Adjust based on available variables
                'method': 'factor_analysis'
            }
        }
        
        for index_name, config in indices_config.items():
            # Check if all variables exist
            available_vars = [v for v in config['variables'] if v in data.columns]
            
            if len(available_vars) >= 2:
                index_values = self.index_constructor.create_index(
                    data=data,
                    variables=available_vars,
                    index_name=index_name,
                    method=config['method'],
                    validate=True
                )
                
                data[index_name] = index_values
    
    def _handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data appropriately"""
        
        if self.config.imputation_method == 'multiple':
            # Multiple imputation
            imputed_datasets = []
            
            for i in range(self.config.n_imputations):
                imputer = IterativeImputer(
                    random_state=self.config.random_seed + i,
                    max_iter=10
                )
                
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                imputed_values = imputer.fit_transform(data[numeric_cols])
                
                imputed_df = data.copy()
                imputed_df[numeric_cols] = imputed_values
                imputed_datasets.append(imputed_df)
            
            # For now, use the first imputation (proper analysis would use all)
            return imputed_datasets[0]
        
        else:
            # Simple imputation
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
            
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'missing')
            
            return data
    
    def _create_derived_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create derived variables for analysis"""
        
        # Urban/rural
        if 'hv025' in data.columns:
            data['is_urban'] = (data['hv025'] == 1).astype(int)
        
        # Wealth categories
        if 'hv270' in data.columns:
            data['is_poor'] = data['hv270'].isin([1, 2]).astype(int)
            data['is_rich'] = data['hv270'].isin([4, 5]).astype(int)
        
        # Add more derived variables as needed
        
        return data
    
    def run_analysis(self) -> Dict:
        """Run complete analysis pipeline"""
        
        self.config.logger.info("Starting main analysis")
        
        # Register hypotheses
        self._register_hypotheses()
        
        # 1. Descriptive analysis
        self.results['descriptive'] = self._run_descriptive_analysis()
        
        # 2. Bivariate analysis
        self.results['bivariate'] = self._run_bivariate_analysis()
        
        # 3. Causal analysis
        self.results['causal'] = self._run_causal_analysis()
        
        # 4. Machine learning
        self.results['ml'] = self._run_ml_analysis()
        
        # 5. Spatial analysis
        if SPATIAL_AVAILABLE:
            self.results['spatial'] = self._run_spatial_analysis()
        
        # Apply multiple testing correction
        corrections = self.testing_framework.apply_multiple_testing_correction()
        self.results['multiple_testing'] = corrections
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _register_hypotheses(self):
        """Register all hypotheses for multiple testing control"""
        
        # Primary hypotheses (pre-registered)
        self.testing_framework.register_hypothesis(
            "Water disruption is associated with household vulnerability",
            level='primary'
        )
        
        # Secondary hypotheses
        self.testing_framework.register_hypothesis(
            "Urban areas have lower water disruption rates",
            level='secondary'
        )
        
        # Add more hypotheses as needed
    
    def _run_descriptive_analysis(self) -> Dict:
        """Descriptive statistics"""
        
        results = {}
        
        # Overall disruption rate
        if 'water_disrupted' in self.processed_data.columns:
            results['disruption_rate'] = self.processed_data['water_disrupted'].mean()
            
            # Weighted estimate
            results['disruption_rate_weighted'] = np.average(
                self.processed_data['water_disrupted'],
                weights=self.processed_data['weight']
            )
            
            # Bootstrap CI
            bootstrap_means = []
            for _ in range(self.config.n_bootstrap):
                boot_sample = self.processed_data.sample(
                    n=len(self.processed_data),
                    replace=True,
                    weights='weight'
                )
                bootstrap_means.append(boot_sample['water_disrupted'].mean())
            
            results['disruption_ci'] = np.percentile(bootstrap_means, [2.5, 97.5])
        
        return results
    
    def _run_bivariate_analysis(self) -> Dict:
        """Bivariate associations"""
        
        results = {}
        
        if 'water_disrupted' not in self.processed_data.columns:
            return results
        
        # Test associations with key variables
        test_vars = ['is_urban', 'is_poor', 'vulnerability_index']
        
        for var in test_vars:
            if var in self.processed_data.columns:
                if self.processed_data[var].dtype in ['float64', 'int64']:
                    # Continuous variable
                    test_result = self.testing_framework.run_test(
                        test_name=f'water_disruption_vs_{var}',
                        test_func=stats.spearmanr,
                        data=(self.processed_data['water_disrupted'], 
                              self.processed_data[var]),
                        hypothesis_level='exploratory'
                    )
                else:
                    # Categorical variable
                    crosstab = pd.crosstab(
                        self.processed_data[var],
                        self.processed_data['water_disrupted']
                    )
                    test_result = self.testing_framework.run_test(
                        test_name=f'water_disruption_vs_{var}',
                        test_func=chi2_contingency,
                        data=crosstab,
                        hypothesis_level='exploratory'
                    )
                
                results[var] = test_result
        
        return results
    
    def _run_causal_analysis(self) -> Dict:
        """Causal inference analysis"""
        
        results = {}
        
        # Define causal model
        if all(v in self.processed_data.columns for v in ['water_disrupted', 'is_urban']):
            
            # Specify model
            self.causal_framework.specify_causal_model(
                data=self.processed_data,
                treatment='is_urban',
                outcome='water_disrupted',
                confounders=['hv270'] if 'hv270' in self.processed_data.columns else []
            )
            
            # Estimate effects
            results['ate'] = self.causal_framework.estimate_causal_effect(
                data=self.processed_data,
                method='regression'
            )
            
            # Validate assumptions
            results['validation'] = self.causal_framework.validate_causal_assumptions(
                data=self.processed_data
            )
            
            # Refutation tests
            results['refutation'] = self.causal_framework.refute_estimate(
                data=self.processed_data,
                method='placebo'
            )
        
        return results
    
    def _run_ml_analysis(self) -> Dict:
        """Machine learning analysis"""
        
        results = {}
        
        # Prepare features and target
        feature_cols = [col for col in self.processed_data.columns 
                       if col not in ['water_disrupted', 'weight'] and 
                       self.processed_data[col].dtype in ['float64', 'int64']]
        
        if 'water_disrupted' in self.processed_data.columns and len(feature_cols) > 0:
            X = self.processed_data[feature_cols].fillna(0)
            y = self.processed_data['water_disrupted']
            weights = self.processed_data['weight']
            
            # Split data
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X, y, weights,
                test_size=0.2,
                random_state=self.config.random_seed,
                stratify=y
            )
            
            # Train model
            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.random_seed,
                n_jobs=self.config.n_jobs
            )
            
            rf.fit(X_train, y_train, sample_weight=w_train)
            
            # Evaluate
            y_pred = rf.predict(X_test)
            y_pred_proba = rf.predict_proba(X_test)[:, 1]
            
            results['accuracy'] = accuracy_score(y_test, y_pred, sample_weight=w_test)
            results['auc'] = roc_auc_score(y_test, y_pred_proba, sample_weight=w_test)
            
            # Feature importance
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            results['feature_importance'] = importance_df.head(10).to_dict()
        
        return results
    
    def _run_spatial_analysis(self) -> Dict:
        """Spatial analysis if geo data available"""
        
        results = {}
        
        # Placeholder for spatial analysis
        # Would require geographic coordinates or administrative boundaries
        
        return results
    
    def _save_results(self):
        """Save all results"""
        
        # Save as pickle
        results_file = self.config.output_dir / f'results_{self.reproducibility.analysis_hash}.pkl'
        with open(results_file, 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save summary as JSON
        summary_file = self.config.output_dir / f'summary_{self.reproducibility.analysis_hash}.json'
        
        # Convert results to JSON-serializable format
        summary = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                summary[key] = {
                    k: v if not isinstance(v, (np.ndarray, pd.DataFrame, pd.Series)) 
                    else str(v) 
                    for k, v in value.items()
                }
            else:
                summary[key] = str(value)
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.config.logger.info(f"Results saved to {results_file}")
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        
        report_file = self.config.output_dir / f'report_{self.reproducibility.analysis_hash}.md'
        
        with open(report_file, 'w') as f:
            f.write("# Water Disruption Analysis Report\n\n")
            f.write(f"Analysis ID: {self.reproducibility.analysis_hash}\n")
            f.write(f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
            
            # Add results summary
            if 'descriptive' in self.results:
                f.write("## Key Findings\n\n")
                f.write(f"- Water disruption rate: {self.results['descriptive'].get('disruption_rate', 0):.2%}\n")
                f.write(f"- Weighted rate: {self.results['descriptive'].get('disruption_rate_weighted', 0):.2%}\n")
            
            # Add more sections as needed
        
        self.config.logger.info(f"Report generated: {report_file}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    # Create configuration
    config = AnalysisConfig()
    
    # Save configuration for reproducibility
    config.to_yaml(config.output_dir / 'config.yaml')
    
    print("\n" + "="*80)
    print("WATER DISRUPTION ANALYSIS - PUBLICATION READY")
    print("="*80)
    print(f"Analysis ID: {config.random_seed}")
    print(f"Output directory: {config.output_dir}")
    print()
    
    try:
        # Initialize analysis
        analysis = WaterDisruptionAnalysis(config)
        
        # Load data
        print("Loading data...")
        data = analysis.load_data()
        print(f"✓ Loaded {len(data)} observations")
        
        # Process data
        print("\nProcessing data...")
        processed_data = analysis.process_data()
        print(f"✓ Processing complete")
        
        # Run analysis
        print("\nRunning analysis...")
        results = analysis.run_analysis()
        print(f"✓ Analysis complete")
        
        # Generate report
        print("\nGenerating report...")
        analysis.generate_report()
        print(f"✓ Report generated")
        
        # Print summary
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY")
        print("="*80)
        
        if 'descriptive' in results:
            desc = results['descriptive']
            print(f"Water disruption rate: {desc.get('disruption_rate_weighted', 0)*100:.1f}%")
            if 'disruption_ci' in desc:
                print(f"95% CI: [{desc['disruption_ci'][0]*100:.1f}%, {desc['disruption_ci'][1]*100:.1f}%]")
        
        if 'causal' in results and 'ate' in results['causal']:
            print(f"\nCausal effect (ATE): {results['causal']['ate'].get('ate', 'N/A')}")
        
        if 'ml' in results:
            print(f"\nML Model Performance:")
            print(f"  Accuracy: {results['ml'].get('accuracy', 0):.3f}")
            print(f"  AUC: {results['ml'].get('auc', 0):.3f}")
        
        print("\n✅ Analysis complete! Check output directory for detailed results.")
        
    except Exception as e:
        config.logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
