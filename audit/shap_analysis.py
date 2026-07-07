import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# ─────────────────────────────────────────
# 1. LOAD DATA AND TRAIN MODEL
# ─────────────────────────────────────────

df = pd.read_csv('data/synthetic_nhs_data.csv')

# Blind classifier - no demographic group feature
FEATURES = ['risk_score', 'referral_length',
            'previous_contacts', 'age']
TARGET = 'true_outcome'

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Get group membership for test set
groups_test = df.loc[X_test.index, 'group'].values
X_test_A = X_test[groups_test == 'A']
X_test_B = X_test[groups_test == 'B']
y_test_A = y_test[groups_test == 'A']
y_test_B = y_test[groups_test == 'B']

print("=" * 65)
print("SHAP VALUE ANALYSIS")
print("Shapley Additive Explanations for Bias Detection")
print("=" * 65)
print(f"""
SHAP vs Permutation Importance:
""")

# ─────────────────────────────────────────
# 2. TRAIN THREE MODELS FOR COMPARISON
# ─────────────────────────────────────────

# Baseline model
rf_baseline = RandomForestClassifier(
    n_estimators=100, max_depth=5,
    random_state=42, class_weight='balanced'
)
rf_baseline.fit(X_train, y_train)

# Grid Search tuned model
rf_grid = RandomForestClassifier(
    n_estimators=100, max_features=1,
    max_samples=0.6, bootstrap=True,
    max_depth=3, random_state=42,
    class_weight='balanced'
)
rf_grid.fit(X_train, y_train)

# Bayesian tuned model
rf_bayes = RandomForestClassifier(
    n_estimators=100, max_features=1,
    max_samples=0.734, bootstrap=True,
    max_depth=2, random_state=42,
    class_weight='balanced'
)
rf_bayes.fit(X_train, y_train)

print("Three models trained:")
print("  1. Baseline (default parameters)")
print("  2. Grid Search tuned")
print("  3. Bayesian Optimisation tuned")
print()

# ─────────────────────────────────────────
# 3. COMPUTE SHAP VALUES
# ─────────────────────────────────────────

print("Computing SHAP values...")
print("(Using TreeExplainer - optimised for Random Forests)")
print()

# Use TreeExplainer - most efficient for tree-based models
explainer_baseline = shap.TreeExplainer(rf_baseline)
explainer_grid = shap.TreeExplainer(rf_grid)
explainer_bayes = shap.TreeExplainer(rf_bayes)

# Compute SHAP values for test set
# shap_values shape: (n_samples, n_features, n_classes)
shap_baseline = explainer_baseline.shap_values(X_test)
shap_grid = explainer_grid.shap_values(X_test)
shap_bayes = explainer_bayes.shap_values(X_test)

# For binary classification take class 1 (referral = positive)
if isinstance(shap_baseline, list):
    shap_baseline_pos = shap_baseline[1]
    shap_grid_pos = shap_grid[1]
    shap_bayes_pos = shap_bayes[1]
else:
    shap_baseline_pos = shap_baseline[:, :, 1]
    shap_grid_pos = shap_grid[:, :, 1]
    shap_bayes_pos = shap_bayes[:, :, 1]

print("SHAP values computed successfully")
print(f"Shape: {shap_baseline_pos.shape} "
      f"(patients x features)")
print()

# ─────────────────────────────────────────
# 4. GLOBAL SHAP IMPORTANCE
# Mean absolute SHAP value per feature
# ─────────────────────────────────────────

print("=" * 65)
print("GLOBAL SHAP IMPORTANCE (Mean |SHAP value|)")
print("=" * 65)

def global_shap_importance(shap_values, feature_names):
    """Mean absolute SHAP value per feature."""
    return pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)

imp_baseline = global_shap_importance(
    shap_baseline_pos, FEATURES)
imp_grid = global_shap_importance(shap_grid_pos, FEATURES)
imp_bayes = global_shap_importance(shap_bayes_pos, FEATURES)

print(f"\n{'Feature':<25} {'Baseline':>10} "
      f"{'Grid':>10} {'Bayesian':>10}")
print("-" * 58)
for feat in FEATURES:
    b = imp_baseline[imp_baseline['feature']==feat]['importance'].values[0]
    g = imp_grid[imp_grid['feature']==feat]['importance'].values[0]
    bay = imp_bayes[imp_bayes['feature']==feat]['importance'].values[0]
    print(f"{feat:<25} {b:>10.4f} {g:>10.4f} {bay:>10.4f}")


# ─────────────────────────────────────────
# 5. GROUP-SPECIFIC SHAP ANALYSIS
# Core fairness insight - how features affect each group
# ─────────────────────────────────────────

print("=" * 65)
print("GROUP-SPECIFIC SHAP ANALYSIS")
print("How features affect Group A vs Group B differently")
print("=" * 65)

# Split SHAP values by group
mask_A = groups_test == 'A'
mask_B = groups_test == 'B'

def group_shap_stats(shap_values, mask, feature_names):
    """Mean SHAP value per feature for a demographic group."""
    group_shap = shap_values[mask]
    return pd.DataFrame({
        'feature': feature_names,
        'mean_shap': group_shap.mean(axis=0),
        'mean_abs_shap': np.abs(group_shap).mean(axis=0)
    })

# Baseline model group SHAP
shap_A_baseline = group_shap_stats(
    shap_baseline_pos, mask_A, FEATURES)
shap_B_baseline = group_shap_stats(
    shap_baseline_pos, mask_B, FEATURES)

# Bayesian model group SHAP
shap_A_bayes = group_shap_stats(
    shap_bayes_pos, mask_A, FEATURES)
shap_B_bayes = group_shap_stats(
    shap_bayes_pos, mask_B, FEATURES)

print(f"\nBASELINE MODEL - Mean SHAP values by group:")
print(f"(Positive = pushes toward referral, "
      f"Negative = pushes away)")
print(f"\n{'Feature':<25} {'Group A':>10} "
      f"{'Group B':>10} {'Gap':>10}")
print("-" * 58)
for feat in FEATURES:
    a = shap_A_baseline[
        shap_A_baseline['feature']==feat]['mean_shap'].values[0]
    b = shap_B_baseline[
        shap_B_baseline['feature']==feat]['mean_shap'].values[0]
    print(f"{feat:<25} {a:>10.4f} {b:>10.4f} {a-b:>10.4f}")

print(f"\nBAYESIAN TUNED MODEL - Mean SHAP values by group:")
print(f"\n{'Feature':<25} {'Group A':>10} "
      f"{'Group B':>10} {'Gap':>10}")
print("-" * 58)
for feat in FEATURES:
    a = shap_A_bayes[
        shap_A_bayes['feature']==feat]['mean_shap'].values[0]
    b = shap_B_bayes[
        shap_B_bayes['feature']==feat]['mean_shap'].values[0]
    print(f"{feat:<25} {a:>10.4f} {b:>10.4f} {a-b:>10.4f}")

# ─────────────────────────────────────────
# 6. REFERRAL LENGTH SHAP DEEP DIVE
# ─────────────────────────────────────────

print()
print("=" * 65)
print("REFERRAL LENGTH SHAP DEEP DIVE")
print("Direction of proxy variable bias")
print("=" * 65)

j_ref = FEATURES.index('referral_length')

shap_ref_A = shap_baseline_pos[mask_A, j_ref]
shap_ref_B = shap_baseline_pos[mask_B, j_ref]

print(f"""
referral_length SHAP values (Baseline Model):

Group A (underrepresented):
  Mean SHAP:     {shap_ref_A.mean():.4f}
  Std SHAP:      {shap_ref_A.std():.4f}
  % negative:    {(shap_ref_A < 0).mean()*100:.1f}%
  (negative = pushes AWAY from referral = harmful bias)

Group B (majority):
  Mean SHAP:     {shap_ref_B.mean():.4f}
  Std SHAP:      {shap_ref_B.std():.4f}
  % negative:    {(shap_ref_B < 0).mean()*100:.1f}%

Interpretation:
  For Group A: referral_length SHAP is {shap_ref_A.mean():.4f}
""")

# ─────────────────────────────────────────
# 7. VISUALISATIONS
# ─────────────────────────────────────────

os.makedirs('visualisations/outputs', exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Global SHAP importance comparison
features_sorted = imp_baseline.sort_values(
    'importance', ascending=True)['feature'].tolist()
x = np.arange(len(features_sorted))
width = 0.25

imp_b_vals = [imp_baseline[imp_baseline['feature']==f]
              ['importance'].values[0] for f in features_sorted]
imp_g_vals = [imp_grid[imp_grid['feature']==f]
              ['importance'].values[0] for f in features_sorted]
imp_bay_vals = [imp_bayes[imp_bayes['feature']==f]
                ['importance'].values[0] for f in features_sorted]

axes[0, 0].barh(x - width, imp_b_vals, width,
                color='coral', alpha=0.8, label='Baseline')
axes[0, 0].barh(x, imp_g_vals, width,
                color='steelblue', alpha=0.8, label='Grid Search')
axes[0, 0].barh(x + width, imp_bay_vals, width,
                color='purple', alpha=0.8,
                label='Bayesian Opt')
axes[0, 0].set_yticks(x)
axes[0, 0].set_yticklabels(features_sorted)
axes[0, 0].set_xlabel('Mean |SHAP value|')
axes[0, 0].set_title('Global SHAP Importance\nAll Three Models')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Group A vs Group B SHAP (baseline)
feat_labels = FEATURES
shap_A_means = [shap_A_baseline[
    shap_A_baseline['feature']==f]['mean_shap'].values[0]
    for f in feat_labels]
shap_B_means = [shap_B_baseline[
    shap_B_baseline['feature']==f]['mean_shap'].values[0]
    for f in feat_labels]

x2 = np.arange(len(feat_labels))
axes[0, 1].bar(x2 - width/2, shap_A_means, width,
               color='steelblue', alpha=0.8,
               label='Group A (underrepresented)')
axes[0, 1].bar(x2 + width/2, shap_B_means, width,
               color='coral', alpha=0.8,
               label='Group B (majority)')
axes[0, 1].axhline(y=0, color='black', linewidth=1)
axes[0, 1].set_xticks(x2)
axes[0, 1].set_xticklabels(feat_labels, rotation=15, fontsize=9)
axes[0, 1].set_ylabel('Mean SHAP value')
axes[0, 1].set_title('Group-Specific SHAP Values\nBaseline Model\n'
                      '(+ve = pushes toward referral)')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Group A vs Group B SHAP (Bayesian)
shap_A_bayes_means = [shap_A_bayes[
    shap_A_bayes['feature']==f]['mean_shap'].values[0]
    for f in feat_labels]
shap_B_bayes_means = [shap_B_bayes[
    shap_B_bayes['feature']==f]['mean_shap'].values[0]
    for f in feat_labels]

axes[0, 2].bar(x2 - width/2, shap_A_bayes_means, width,
               color='steelblue', alpha=0.8,
               label='Group A (underrepresented)')
axes[0, 2].bar(x2 + width/2, shap_B_bayes_means, width,
               color='coral', alpha=0.8,
               label='Group B (majority)')
axes[0, 2].axhline(y=0, color='black', linewidth=1)
axes[0, 2].set_xticks(x2)
axes[0, 2].set_xticklabels(feat_labels, rotation=15, fontsize=9)
axes[0, 2].set_ylabel('Mean SHAP value')
axes[0, 2].set_title('Group-Specific SHAP Values\nBayesian Model\n'
                      '(+ve = pushes toward referral)')
axes[0, 2].legend(fontsize=8)
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: referral_length SHAP distribution by group
axes[1, 0].hist(shap_ref_A, bins=20,
                color='steelblue', alpha=0.7,
                label=f'Group A (mean={shap_ref_A.mean():.4f})')
axes[1, 0].hist(shap_ref_B, bins=20,
                color='coral', alpha=0.7,
                label=f'Group B (mean={shap_ref_B.mean():.4f})')
axes[1, 0].axvline(x=0, color='black', linewidth=2,
                    label='Zero (neutral)')
axes[1, 0].axvline(x=shap_ref_A.mean(),
                    color='steelblue', linestyle='--',
                    linewidth=2)
axes[1, 0].axvline(x=shap_ref_B.mean(),
                    color='coral', linestyle='--',
                    linewidth=2)
axes[1, 0].set_xlabel('SHAP value for referral_length')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('referral_length SHAP Distribution\n'
                      'by Demographic Group (Baseline)\n'
                      'Negative = pushes away from referral')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: SHAP scatter - referral_length value vs SHAP value
ref_vals_A = X_test[mask_A]['referral_length'].values
ref_vals_B = X_test[mask_B]['referral_length'].values

axes[1, 1].scatter(ref_vals_A, shap_ref_A,
                    color='steelblue', alpha=0.6, s=30,
                    label='Group A')
axes[1, 1].scatter(ref_vals_B, shap_ref_B,
                    color='coral', alpha=0.4, s=30,
                    label='Group B')
axes[1, 1].axhline(y=0, color='black', linewidth=1)
axes[1, 1].set_xlabel('referral_length (words)')
axes[1, 1].set_ylabel('SHAP value')
axes[1, 1].set_title('referral_length Value vs SHAP Value\n'
                      'by Demographic Group\n'
                      'Short letters → negative SHAP → missed')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Permutation importance vs SHAP comparison
perm_imp = [0.0085, 0.0034, -0.0014, 0.0052]
shap_imp = [imp_baseline[imp_baseline['feature']==f]
            ['importance'].values[0] for f in FEATURES]

axes[1, 2].scatter(perm_imp, shap_imp,
                    color='purple', s=100, zorder=5)
for i, feat in enumerate(FEATURES):
    axes[1, 2].annotate(feat,
                         (perm_imp[i], shap_imp[i]),
                         textcoords='offset points',
                         xytext=(5, 5), fontsize=9)
axes[1, 2].axhline(y=0, color='black', linewidth=0.5)
axes[1, 2].axvline(x=0, color='black', linewidth=0.5)
axes[1, 2].set_xlabel('Permutation Importance I_j')
axes[1, 2].set_ylabel('Mean |SHAP value|')
axes[1, 2].set_title('Permutation Importance vs SHAP\n'
                      'Do both methods agree on bias source?')
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle('SHAP Value Analysis: Feature Bias Detection\n'
             'From Probabilities to Decisions - ',
             fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('visualisations/outputs/shap_analysis.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to visualisations/outputs/shap_analysis.png")

# ─────────────────────────────────────────
# 8. FINAL SUMMARY
# ─────────────────────────────────────────

print()
print("=" * 65)
print("SHAP ANALYSIS SUMMARY")
print("=" * 65)

ref_idx = FEATURES.index('referral_length')
shap_ref_reduction = (
    imp_baseline[imp_baseline['feature']=='referral_length']
    ['importance'].values[0] -
    imp_bayes[imp_bayes['feature']=='referral_length']
    ['importance'].values[0]
)

print(f"""
1. GLOBAL IMPORTANCE:
   referral_length SHAP importance reduced by {shap_ref_reduction:.4f}
   from baseline to Bayesian model - confirming bias reduction

2. DIRECTIONAL BIAS (key SHAP advantage over permutation):
   Group A referral_length mean SHAP: {shap_ref_A.mean():.4f}
   Group B referral_length mean SHAP: {shap_ref_B.mean():.4f}
   
   For Group A: {(shap_ref_A < 0).mean()*100:.1f}% of patients have
   NEGATIVE referral_length SHAP.
""")