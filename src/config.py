"""
config.py

Central configuration for the equitable triage framework.
Supports multiple modes:
  CHRONOSIG - Adult secondary mental health triage
  CAMHS     - Child and adolescent mental health triage
  COMBINED  - Both systems (future)

Change CURRENT_MODE to switch between datasets and features.
All output files are automatically suffixed with mode name.

Author: Nitya Thota
Institution: KCLMS
Date: 2026
"""

import os

# ─────────────────────────────────────────
# MODE SELECTION
# Change this to switch between systems
# ─────────────────────────────────────────

CURRENT_MODE = "CAMHS"  # "CHRONOSIG" | "CAMHS" | "COMBINED"

# ─────────────────────────────────────────
# MODE-SPECIFIC CONFIGURATION
# ─────────────────────────────────────────

MODE_CONFIG = {

    "CHRONOSIG": {
        "description": "Adult secondary mental health triage",
        "system": "CHRONOSIG / Limbic",
        "population": "Adults 18+",
        "data_file": "synthetic_nhs_data.csv",
        "features": [
            'risk_score',
            'referral_length',
            'previous_contacts',
            'age'
        ],
        "primary_bias_feature": "referral_length",
        "bias_description": (
            "Short referral letters systematically "
            "disadvantage underrepresented patients"
        ),
        "group_A_description": "Young carers / underrepresented",
        "group_B_description": "Majority population",
        "base_rate_A": 0.30,
        "base_rate_B": 0.50,
    },

    "CAMHS": {
        "description": "Child and adolescent mental health triage",
        "system": "CAMHS triage",
        "population": "Children and young people 5-17",
        "data_file": "synthetic_camhs_data.csv",
        "features": [
            # Child self-report
            'sdq_emotional_c',
            'sdq_conduct_c',
            'sdq_hyperactivity_c',
            'sdq_peer_c',
            'sdq_prosocial_c',
            'sdq_total_c',
            'rcads_separation_anxiety_c',
            'rcads_social_phobia_c',
            'rcads_gad_c',
            'rcads_panic_c',
            'rcads_ocd_c',
            'rcads_depression_c',
            'rcads_total_c',
            # Parent report (may be missing)
            'sdq_total_p',
            'rcads_total_p',
            # Discrepancy and derived
            'sdq_discrepancy',
            'impact_overall_severity_c',
            'impact_duration_c',
            'impact_distress_c',
            'impact_overall_severity_p',
            'impact_family_burden_p',
            # Demographics
            'age',
            'gender',
            'parent_data_missing',
            'duration_months_c',
        ],
        "primary_bias_feature": "sdq_discrepancy",
        "bias_description": (
            "Underreported SDQ/RCADS scores and missing "
            "parent data systematically disadvantage "
            "young carers in CAMHS triage"
        ),
        "group_A_description": "Young carers / single parent households",
        "group_B_description": "Majority population",
        "base_rate_A": 0.45,
        "base_rate_B": 0.60,
    },

    "COMBINED": {
        "description": "Combined CHRONOSIG and CAMHS analysis",
        "system": "Multiple NHS systems",
        "population": "All ages",
        "data_file": "synthetic_combined_data.csv",
        "features": [],  # defined when implemented
        "primary_bias_feature": None,
        "bias_description": "Framework generalises across care settings",
        "group_A_description": "Underrepresented populations",
        "group_B_description": "Majority population",
        "base_rate_A": 0.35,
        "base_rate_B": 0.55,
    }
}

# ─────────────────────────────────────────
# ACTIVE CONFIGURATION
# Everything below uses CURRENT_MODE
# ─────────────────────────────────────────

ACTIVE = MODE_CONFIG[CURRENT_MODE]

# Data
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data'
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'visualisations', 'outputs'
)
DATA_PATH = os.path.join(DATA_DIR, ACTIVE["data_file"])

# Features and target
FEATURES = ACTIVE["features"]
TARGET = 'true_outcome'
GROUP_COL = 'group'
GROUP_A = 'A'
GROUP_B = 'B'

# Train/test split
TEST_SIZE = 0.3
RANDOM_STATE = 42

# ─────────────────────────────────────────
# SYNTHETIC DATA PARAMETERS - CHRONOSIG
# ─────────────────────────────────────────

N_A = 200
N_B = 800
P_A = ACTIVE["base_rate_A"]
P_B = ACTIVE["base_rate_B"]

# Beta distribution parameters (CHRONOSIG only)
ALPHA_A, BETA_A = 1.5, 4.0
ALPHA_B, BETA_B = 2.5, 3.0
MEAN_LENGTH_A, STD_LENGTH_A = 150, 50
MEAN_LENGTH_B, STD_LENGTH_B = 300, 80
LAMBDA_A = 0.5
LAMBDA_B = 2.0
MEAN_AGE_A, STD_AGE_A = 22, 8
MEAN_AGE_B, STD_AGE_B = 35, 12

# ─────────────────────────────────────────
# MODEL PARAMETERS
# ─────────────────────────────────────────

N_ESTIMATORS = 100

BASELINE_PARAMS = {
    'n_estimators': N_ESTIMATORS,
    'max_depth': 5,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced'
}

GRID_PARAMS = {
    'n_estimators': N_ESTIMATORS,
    'max_features': 1,
    'max_samples': 0.6,
    'bootstrap': True,
    'max_depth': 3,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced'
}

BAYES_PARAMS = {
    'n_estimators': N_ESTIMATORS,
    'max_features': 1,
    'max_samples': 0.734,
    'bootstrap': True,
    'max_depth': 2,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced'
}

# ─────────────────────────────────────────
# FAIRNESS PARAMETERS
# ─────────────────────────────────────────

UNIFORM_THRESHOLD = 0.4
C_FN = 5.0
C_FP = 1.0
FAIRNESS_TOLERANCE = 0.05

# ─────────────────────────────────────────
# OPTIMISATION PARAMETERS
# ─────────────────────────────────────────

GRID_SEARCH_PARAMS = {
    'max_features': [1, 2, 3, 'sqrt'],
    'max_samples':  [0.5, 0.6, 0.7, 1.0],
    'bootstrap':    [True, False],
    'max_depth':    [3, 5, 7]
}

N_BAYES_CALLS = 50
N_BAYES_INITIAL = 10
N_PERM_TREES = 50

# ─────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────

COLOR_A = 'steelblue'
COLOR_B = 'coral'
COLOR_BASELINE = 'coral'
COLOR_GRID = 'steelblue'
COLOR_BAYES = 'purple'
DPI = 150
FIGURE_TITLE_SUFFIX = (
    f'From Probabilities to Decisions [{CURRENT_MODE}] - '
    f'Nitya Thota, KCLMS 2026'
)

# ─────────────────────────────────────────
# FILE NAMING - MODE AWARE
# Outputs automatically suffixed with mode
# so CHRONOSIG and CAMHS never overwrite each other
# ─────────────────────────────────────────

def get_output_path(filename):
    """
    Get mode-aware output path.
    e.g. 'roc_analysis.png' ->
         'visualisations/outputs/roc_analysis_CHRONOSIG.png'

    Ensures outputs from different modes never overwrite.
    """
    name, ext = os.path.splitext(filename)
    suffixed = f"{name}_{CURRENT_MODE}{ext}"
    return os.path.join(OUTPUT_DIR, suffixed)


def get_data_path(filename=None):
    """Get mode-aware data path."""
    if filename is None:
        return DATA_PATH
    return os.path.join(DATA_DIR, filename)

# ─────────────────────────────────────────
# FEATURE INDEX
# ─────────────────────────────────────────

FEATURE_INDEX = {feat: i for i, feat in enumerate(FEATURES)}

# Primary bias feature index (mode-specific)
PRIMARY_BIAS_FEATURE = ACTIVE["primary_bias_feature"]
if PRIMARY_BIAS_FEATURE and PRIMARY_BIAS_FEATURE in FEATURE_INDEX:
    PRIMARY_BIAS_IDX = FEATURE_INDEX[PRIMARY_BIAS_FEATURE]
else:
    PRIMARY_BIAS_IDX = 0

# Keep backward compatibility
REF_LENGTH_IDX = FEATURE_INDEX.get('referral_length', 0)
RISK_SCORE_IDX = FEATURE_INDEX.get('risk_score', 0)

# ─────────────────────────────────────────
# PRINT ACTIVE CONFIG ON IMPORT
# ─────────────────────────────────────────

print(f"[CONFIG] Mode: {CURRENT_MODE}")
print(f"[CONFIG] System: {ACTIVE['system']}")
print(f"[CONFIG] Population: {ACTIVE['population']}")
print(f"[CONFIG] Features: {len(FEATURES)}")
print(f"[CONFIG] Data: {ACTIVE['data_file']}")
print(f"[CONFIG] Primary bias: {PRIMARY_BIAS_FEATURE}")