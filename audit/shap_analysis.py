"""
shap_analysis.py
SHAP value analysis - works in CHRONOSIG and CAMHS mode.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import shap

from src.config import (
    FEATURES, PRIMARY_BIAS_FEATURE, PRIMARY_BIAS_IDX,
    COLOR_A, COLOR_B, FIGURE_TITLE_SUFFIX,
    get_output_path, CURRENT_MODE
)
from src.data_loader import load_and_split
from src.models import (
    get_baseline_model, get_grid_model,
    get_bayes_model, train_model
)
from src.utils import print_header, ensure_output_dir

# ─────────────────────────────────────────
# 1. LOAD DATA AND TRAIN ALL THREE MODELS
# ─────────────────────────────────────────

(df, X_train, X_test, y_train, y_test,
 groups, X_A, y_A, X_B, y_B) = load_and_split()

rf_baseline = train_model(get_baseline_model(), X_train, y_train)
rf_grid = train_model(get_grid_model(), X_train, y_train)
rf_bayes = train_model(get_bayes_model(), X_train, y_train)

mask_A = groups == 'A'
mask_B = groups == 'B'

primary_feature = PRIMARY_BIAS_FEATURE
j_bias = PRIMARY_BIAS_IDX

print_header("SHAP VALUE ANALYSIS")
print(f"Mode: {CURRENT_MODE}")
print(f"Primary bias feature: {primary_feature}")
print("Three models trained: Baseline, Grid Search, Bayesian")

# ─────────────────────────────────────────
# 2. COMPUTE SHAP VALUES
# ─────────────────────────────────────────

print("\nComputing SHAP values...")

explainer_baseline = shap.TreeExplainer(rf_baseline)
explainer_grid = shap.TreeExplainer(rf_grid)
explainer_bayes = shap.TreeExplainer(rf_bayes)

shap_baseline = explainer_baseline.shap_values(X_test)
shap_grid = explainer_grid.shap_values(X_test)
shap_bayes = explainer_bayes.shap_values(X_test)

# For binary classification take class 1
if isinstance(shap_baseline, list):
    shap_baseline_pos = shap_baseline[1]
    shap_grid_pos = shap_grid[1]
    shap_bayes_pos = shap_bayes[1]
else:
    shap_baseline_pos = shap_baseline[:, :, 1]
    shap_grid_pos = shap_grid[:, :, 1]
    shap_bayes_pos = shap_bayes[:, :, 1]

print(f"SHAP computed. Shape: {shap_baseline_pos.shape}")

# ─────────────────────────────────────────
# 3. GLOBAL SHAP IMPORTANCE
# ─────────────────────────────────────────

print_header("GLOBAL SHAP IMPORTANCE (Mean |SHAP value|)")

def global_importance(shap_values, feature_names):
    return {f: np.abs(shap_values).mean(axis=0)[i]
            for i, f in enumerate(feature_names)}

imp_b = global_importance(shap_baseline_pos, FEATURES)
imp_g = global_importance(shap_grid_pos, FEATURES)
imp_bay = global_importance(shap_bayes_pos, FEATURES)

print(f"\n{'Feature':<30} {'Baseline':>10} {'Grid':>10} {'Bayesian':>10}")
print(f"{'─'*62}")
for feat in FEATURES:
    marker = " <-- PRIMARY BIAS" if feat == primary_feature else ""
    print(f"{feat:<30} {imp_b[feat]:>10.4f} "
          f"{imp_g[feat]:>10.4f} {imp_bay[feat]:>10.4f}{marker}")

# ─────────────────────────────────────────
# 4. GROUP-SPECIFIC SHAP ANALYSIS
# ─────────────────────────────────────────

print_header("GROUP-SPECIFIC SHAP ANALYSIS")

shap_A_b = shap_baseline_pos[mask_A]
shap_B_b = shap_baseline_pos[mask_B]
shap_A_bay = shap_bayes_pos[mask_A]
shap_B_bay = shap_bayes_pos[mask_B]

print(f"\nBASELINE MODEL - Mean SHAP by group:")
print(f"(+ve = pushes toward referral, -ve = pushes away)")
print(f"\n{'Feature':<30} {'Group A':>10} {'Group B':>10} {'Gap':>10}")
print(f"{'─'*62}")
for i, feat in enumerate(FEATURES):
    a = shap_A_b[:, i].mean()
    b = shap_B_b[:, i].mean()
    marker = " <--" if feat == primary_feature else ""
    print(f"{feat:<30} {a:>10.4f} {b:>10.4f} {a-b:>10.4f}{marker}")

# ─────────────────────────────────────────
# 5. PRIMARY BIAS FEATURE DEEP DIVE
# ─────────────────────────────────────────

print_header(f"PRIMARY BIAS FEATURE DEEP DIVE: {primary_feature}")

shap_bias_A = shap_baseline_pos[mask_A, j_bias]
shap_bias_B = shap_baseline_pos[mask_B, j_bias]

print(f"""
{primary_feature} SHAP values (Baseline Model):

Group A (underrepresented):
  Mean SHAP:   {shap_bias_A.mean():.4f}
  Std SHAP:    {shap_bias_A.std():.4f}
  % negative:  {(shap_bias_A < 0).mean()*100:.1f}%
  (negative = pushes AWAY from referral = harmful bias)

Group B (majority):
  Mean SHAP:   {shap_bias_B.mean():.4f}
  Std SHAP:    {shap_bias_B.std():.4f}
  % negative:  {(shap_bias_B < 0).mean()*100:.1f}%

Gap: {shap_bias_A.mean() - shap_bias_B.mean():.4f}
""")

# ─────────────────────────────────────────
# 6. VISUALISATIONS
# ─────────────────────────────────────────

ensure_output_dir()
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Global importance comparison
features_sorted = sorted(FEATURES,
                          key=lambda f: imp_b[f], reverse=True)
x = np.arange(len(features_sorted))
width = 0.25

imp_b_vals = [imp_b[f] for f in features_sorted]
imp_g_vals = [imp_g[f] for f in features_sorted]
imp_bay_vals = [imp_bay[f] for f in features_sorted]

axes[0, 0].barh(features_sorted, imp_b_vals,
                color='coral', alpha=0.8, label='Baseline')
axes[0, 0].barh(features_sorted, imp_g_vals,
                color='steelblue', alpha=0.7, label='Grid')
axes[0, 0].barh(features_sorted, imp_bay_vals,
                color='purple', alpha=0.6, label='Bayesian')
axes[0, 0].set_xlabel('Mean |SHAP value|')
axes[0, 0].set_title(f'Global SHAP Importance\nAll Three Models [{CURRENT_MODE}]')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Group SHAP baseline
shap_A_means = [shap_A_b[:, i].mean() for i in range(len(FEATURES))]
shap_B_means = [shap_B_b[:, i].mean() for i in range(len(FEATURES))]

x2 = np.arange(len(FEATURES))
w = 0.35
axes[0, 1].bar(x2 - w/2, shap_A_means, w,
               color=COLOR_A, alpha=0.8, label='Group A')
axes[0, 1].bar(x2 + w/2, shap_B_means, w,
               color=COLOR_B, alpha=0.8, label='Group B')
axes[0, 1].axhline(y=0, color='black', linewidth=1)
axes[0, 1].set_xticks(x2)
axes[0, 1].set_xticklabels(FEATURES, rotation=45,
                             ha='right', fontsize=7)
axes[0, 1].set_ylabel('Mean SHAP value')
axes[0, 1].set_title(f'Group SHAP Values\nBaseline [{CURRENT_MODE}]')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Group SHAP bayesian
shap_A_bay_means = [shap_A_bay[:, i].mean()
                    for i in range(len(FEATURES))]
shap_B_bay_means = [shap_B_bay[:, i].mean()
                    for i in range(len(FEATURES))]

axes[0, 2].bar(x2 - w/2, shap_A_bay_means, w,
               color=COLOR_A, alpha=0.8, label='Group A')
axes[0, 2].bar(x2 + w/2, shap_B_bay_means, w,
               color=COLOR_B, alpha=0.8, label='Group B')
axes[0, 2].axhline(y=0, color='black', linewidth=1)
axes[0, 2].set_xticks(x2)
axes[0, 2].set_xticklabels(FEATURES, rotation=45,
                             ha='right', fontsize=7)
axes[0, 2].set_ylabel('Mean SHAP value')
axes[0, 2].set_title(f'Group SHAP Values\nBayesian [{CURRENT_MODE}]')
axes[0, 2].legend(fontsize=8)
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Primary bias feature SHAP distribution
axes[1, 0].hist(shap_bias_A, bins=20, color=COLOR_A,
                alpha=0.7,
                label=f'Group A (mean={shap_bias_A.mean():.4f})')
axes[1, 0].hist(shap_bias_B, bins=20, color=COLOR_B,
                alpha=0.7,
                label=f'Group B (mean={shap_bias_B.mean():.4f})')
axes[1, 0].axvline(x=0, color='black', linewidth=2)
axes[1, 0].axvline(x=shap_bias_A.mean(),
                    color=COLOR_A, linestyle='--')
axes[1, 0].axvline(x=shap_bias_B.mean(),
                    color=COLOR_B, linestyle='--')
axes[1, 0].set_xlabel(f'SHAP value for {primary_feature}')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title(f'{primary_feature} SHAP Distribution\n'
                      f'Negative = pushes away from referral')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Feature value vs SHAP value
bias_vals_A = X_test.values[mask_A, j_bias]
bias_vals_B = X_test.values[mask_B, j_bias]

axes[1, 1].scatter(bias_vals_A, shap_bias_A,
                    color=COLOR_A, alpha=0.6, s=30,
                    label='Group A')
axes[1, 1].scatter(bias_vals_B, shap_bias_B,
                    color=COLOR_B, alpha=0.4, s=30,
                    label='Group B')
axes[1, 1].axhline(y=0, color='black', linewidth=1)
axes[1, 1].set_xlabel(f'{primary_feature} value')
axes[1, 1].set_ylabel('SHAP value')
axes[1, 1].set_title(f'{primary_feature} Value vs SHAP\n'
                      f'Low values -> negative SHAP -> missed')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

# ─────────────────────────────────────────
# 7. SUMMARY
# ─────────────────────────────────────────

print_header("SHAP ANALYSIS SUMMARY")
print(f"""
Mode: {CURRENT_MODE}
Primary bias feature: {primary_feature}

Directional bias:
  Group A mean SHAP: {shap_bias_A.mean():.4f}
  Group B mean SHAP: {shap_bias_B.mean():.4f}
  Gap: {shap_bias_A.mean() - shap_bias_B.mean():.4f}

  {(shap_bias_A < 0).mean()*100:.1f}% of Group A have NEGATIVE SHAP
  -> {primary_feature} actively pushes Group A AWAY from referral

Global importance reduction (baseline -> bayesian):
  {primary_feature}: {imp_b[primary_feature]:.4f} -> {imp_bay[primary_feature]:.4f}
""")