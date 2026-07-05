import os

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────

DATA_DIR = 'data'
OUTPUT_DIR = 'visualisations/outputs'
DATA_PATH = os.path.join(DATA_DIR, 'synthetic_nhs_data.csv')

# ─────────────────────────────────────────
# DATA CONFIGURATION
# ─────────────────────────────────────────

FEATURES = [
    'risk_score',
    'referral_length',
    'previous_contacts',
    'age'
]

TARGET = 'true_outcome'
GROUP_COL = 'group'
GROUP_A = 'A'  # underrepresented population
GROUP_B = 'B'  # majority population

# Train/test split
TEST_SIZE = 0.3
RANDOM_STATE = 42

# ─────────────────────────────────────────
# SYNTHETIC DATA PARAMETERS
# ─────────────────────────────────────────

# Population sizes
N_A = 200  # underrepresented group
N_B = 800  # majority group

# Base rates P(T=1|G)
P_A = 0.30  # genuine need rate - Group A
P_B = 0.50  # genuine need rate - Group B

# Beta distribution parameters for risk scores
ALPHA_A, BETA_A = 1.5, 4.0  # Group A - uncertain, lower
ALPHA_B, BETA_B = 2.5, 3.0  # Group B - confident, higher

# Referral letter length (words)
MEAN_LENGTH_A, STD_LENGTH_A = 150, 50   # shorter, vaguer
MEAN_LENGTH_B, STD_LENGTH_B = 300, 80   # longer, detailed

# Previous contacts with services
LAMBDA_A = 0.5  # Poisson parameter - Group A
LAMBDA_B = 2.0  # Poisson parameter - Group B

# Age
MEAN_AGE_A, STD_AGE_A = 22, 8   # younger - young carers
MEAN_AGE_B, STD_AGE_B = 35, 12  # broader age range

# ─────────────────────────────────────────
# MODEL PARAMETERS
# ─────────────────────────────────────────

# Shared parameters for all models
N_ESTIMATORS = 100

# Baseline model (default parameters)
BASELINE_PARAMS = {
    'n_estimators': N_ESTIMATORS,
    'max_depth': 5,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced'
}

# Grid Search best parameters
GRID_PARAMS = {
    'n_estimators': N_ESTIMATORS,
    'max_features': 1,
    'max_samples': 0.6,
    'bootstrap': True,
    'max_depth': 3,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced'
}

# Bayesian Optimisation best parameters
BAYES_PARAMS = {
    'n_estimators': N_ESTIMATORS,
    'max_features': 1,
    'max_samples': 0.734,
    'bootstrap': True,
    'max_depth': 2,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced'
}

# ────────────────────────────────────────
# FAIRNESS AND DECISION PARAMETERS
# ─────────────────────────────────────────

UNIFORM_THRESHOLD = 0.4

# Bayesian loss function cost parameters
C_FN = 5.0
C_FP = 1.0 

# Fairness tolerance
FAIRNESS_TOLERANCE = 0.05

# ─────────────────────────────────────────
# OPTIMISATION PARAMETERS
# ─────────────────────────────────────────

# Grid Search parameter grid
GRID_SEARCH_PARAMS = {
    'max_features': [1, 2, 3, 'sqrt'],
    'max_samples':  [0.5, 0.6, 0.7, 1.0],
    'bootstrap':    [True, False],
    'max_depth':    [3, 5, 7]
}

# Bayesian Optimisation
N_BAYES_CALLS = 50
N_BAYES_INITIAL = 10

# Permutation importance
N_PERM_TREES = 50  # trees to use for permutation importance

# ─────────────────────────────────────────
# VISUALISATION
# ──────────────────────────────────────

# Colours
COLOR_A = 'steelblue' 
COLOR_B = 'coral'        
COLOR_BASELINE = 'coral'
COLOR_GRID = 'steelblue'
COLOR_BAYES = 'purple'

DPI = 150
FIGURE_TITLE_SUFFIX = 'From Probabilities to Decisions'

# ─────────────────────────────────────────
# FEATURE INDEX
# ─────────────────────────────────────────

FEATURE_INDEX = {feat: i for i, feat in enumerate(FEATURES)}
REF_LENGTH_IDX = FEATURE_INDEX['referral_length']
RISK_SCORE_IDX = FEATURE_INDEX['risk_score']