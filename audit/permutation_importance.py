import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

from src.config import (
    FEATURES, PRIMARY_BIAS_FEATURE, PRIMARY_BIAS_IDX,
    COLOR_A, COLOR_B, FIGURE_TITLE_SUFFIX,
    get_output_path, CURRENT_MODE
)
from src.data_loader import load_and_split
from src.models import get_baseline_model, train_model
from src.permutation import (
    permutation_importance_all,
    permutation_importance_single,
    group_permutation_importance
)
from src.utils import print_header, ensure_output_dir

# ─────────────────────────────────────────
# 1. LOAD DATA AND TRAIN
# ─────────────────────────────────────────

(df, X_train, X_test, y_train, y_test,
 groups, X_A, y_A, X_B, y_B) = load_and_split()

rf = train_model(get_baseline_model(), X_train, y_train)
X_test_arr = X_test.values
y_test_arr = y_test.values
mask_A = groups == 'A'
mask_B = groups == 'B'

j_bias = PRIMARY_BIAS_IDX
primary_feature = PRIMARY_BIAS_FEATURE

print_header("PERMUTATION IMPORTANCE")
print(f"Mode: {CURRENT_MODE}")
print(f"Primary bias feature: {primary_feature}")
print(f"Formula: I_j = (1/T) * sum(E_t^j - E_t)")
print(f"Trained: {rf.n_estimators} trees, max_depth={rf.max_depth}")

# ─────────────────────────────────────────
# 2. COMPUTE PERMUTATION IMPORTANCE
# ─────────────────────────────────────────

print("\nComputing permutation importance")
importance_results = permutation_importance_all(
    rf, X_test, y_test, FEATURES, n_trees=50
)

# ─────────────────────────────────────────
# 3. DISPLAY RESULTS
# ─────────────────────────────────────────

print_header("PERMUTATION IMPORTANCE RESULTS")
print(f"\n{'Feature':<30} {'I_j':>10} {'Std':>10} {'Level'}")
print(f"{'─'*65}")

sorted_results = sorted(importance_results.items(),
                         key=lambda x: x[1]['I_j'],
                         reverse=True)

for feature, scores in sorted_results:
    I_j = scores['I_j']
    std = scores['std']
    level = "HIGH" if I_j > 0.05 else "MEDIUM" if I_j > 0.02 else "LOW"
    marker = " <-- PRIMARY BIAS" if feature == primary_feature else ""
    print(f"{feature:<30} {I_j:>10.4f} {std:>10.4f} {level}{marker}")

bias_importance = importance_results[primary_feature]['I_j']
print(f"""
Key finding:
  {primary_feature} importance: {bias_importance:.4f}
  This is the primary variable driving bias in {CURRENT_MODE} mode.
""")

# ─────────────────────────────────────────
# 4. TREE-BY-TREE BREAKDOWN
# ─────────────────────────────────────────

print_header(f"TREE-BY-TREE BREAKDOWN: {primary_feature}")
print(f"(First 10 trees - like KC project table)")
print(f"\n{'Tree t':<10} {'E_t':<12} {'E_t^(j)':<12} {'E_t^(j)-E_t'}")
print(f"{'─'*50}")

_, diffs_bias = permutation_importance_single(
    rf, X_test_arr, y_test_arr, j_bias, n_trees=10
)

from src.metrics import classification_error
for t, tree in enumerate(rf.estimators_[:10]):
    y_pred_orig = tree.predict(X_test_arr)
    E_t = classification_error(y_test_arr, y_pred_orig)
    E_t_j = E_t + diffs_bias[t]
    print(f"{t+1:<10} {E_t:<12.4f} {E_t_j:<12.4f} {diffs_bias[t]:<12.4f}")

print(f"\n{'':10} {'':12} Sum = {sum(diffs_bias):.4f}")
print(f"I_j ({primary_feature}, 10 trees) = {sum(diffs_bias)/10:.4f}")

# ─────────────────────────────────────────
# 5. GROUP-SPECIFIC IMPORTANCE
# ─────────────────────────────────────────

print_header("GROUP-SPECIFIC PERMUTATION IMPORTANCE")

X_A_arr = X_test_arr[mask_A]
y_A_arr = y_test_arr[mask_A]
X_B_arr = X_test_arr[mask_B]
y_B_arr = y_test_arr[mask_B]

imp_bias_A = group_permutation_importance(
    rf, X_A_arr, y_A_arr, j_bias)
imp_bias_B = group_permutation_importance(
    rf, X_B_arr, y_B_arr, j_bias)

# Also check risk score or second feature if available
j_second = 1 if j_bias != 1 else 0
second_feature = FEATURES[j_second]
imp_second_A = group_permutation_importance(
    rf, X_A_arr, y_A_arr, j_second)
imp_second_B = group_permutation_importance(
    rf, X_B_arr, y_B_arr, j_second)

print(f"\n{'Feature':<30} {'Group A':>10} {'Group B':>10} {'Gap'}")
print(f"{'─'*58}")
print(f"{primary_feature:<30} {imp_bias_A:>10.4f} "
      f"{imp_bias_B:>10.4f} {imp_bias_A-imp_bias_B:>10.4f}")
print(f"{second_feature:<30} {imp_second_A:>10.4f} "
      f"{imp_second_B:>10.4f} {imp_second_A-imp_second_B:>10.4f}")

print(f"""
Interpretation:
  Higher importance for Group A means the model relies MORE
  on {primary_feature} for underrepresented patients.
  Gap: {imp_bias_A-imp_bias_B:.4f}
""")

# ─────────────────────────────────────────
# 6. VISUALISATIONS
# ─────────────────────────────────────────

ensure_output_dir()
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Overall importance
features_sorted = [f for f, _ in sorted_results]
imps_sorted = [s['I_j'] for _, s in sorted_results]
stds_sorted = [s['std'] for _, s in sorted_results]
colors_sorted = ['red' if f == primary_feature
                 else COLOR_A for f in features_sorted]

axes[0].barh(features_sorted, imps_sorted,
             xerr=stds_sorted, color=colors_sorted,
             alpha=0.8, capsize=3)
axes[0].set_xlabel('Permutation Importance I_j')
axes[0].set_title(f'Permutation Importance\n'
                   f'Red = primary bias ({primary_feature})')
axes[0].axvline(x=0, color='black', linewidth=0.5)
axes[0].grid(True, alpha=0.3)

# Plot 2: Tree-by-tree
axes[1].bar(range(1, 11), diffs_bias,
            color=['red' if d > 0 else COLOR_A
                   for d in diffs_bias], alpha=0.8)
axes[1].axhline(y=0, color='black', linewidth=1)
axes[1].axhline(y=np.mean(diffs_bias), color='red',
                linestyle='--', linewidth=2,
                label=f'Mean I_j={np.mean(diffs_bias):.4f}')
axes[1].set_xlabel('Tree t')
axes[1].set_ylabel('E_t^(j) - E_t')
axes[1].set_title(f'Tree-by-Tree Breakdown\n{primary_feature}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Group-specific
group_features = [primary_feature, second_feature]
imp_A_vals = [imp_bias_A, imp_second_A]
imp_B_vals = [imp_bias_B, imp_second_B]

x = np.arange(len(group_features))
width = 0.35
axes[2].bar(x - width/2, imp_A_vals, width,
            label='Group A (underrepresented)',
            color=COLOR_A, alpha=0.8)
axes[2].bar(x + width/2, imp_B_vals, width,
            label='Group B (majority)',
            color=COLOR_B, alpha=0.8)
axes[2].set_xticks(x)
axes[2].set_xticklabels(group_features, rotation=15, fontsize=9)
axes[2].set_ylabel('Permutation Importance I_j')
axes[2].set_title('Group-Specific Importance\nDoes bias affect Group A more?')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle(f'Permutation Importance: Bias Detection\n{FIGURE_TITLE_SUFFIX}',
             fontsize=12, fontweight='bold')
plt.tight_layout()

filepath = get_output_path('permutation_importance.png')
plt.savefig(filepath, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved to {filepath}")

# ─────────────────────────────────────────
# 7. SUMMARY
# ─────────────────────────────────────────

print_header("SUMMARY")
print(f"""
Mode: {CURRENT_MODE}
Primary bias feature: {primary_feature}
  I_j = {bias_importance:.4f}

Group-specific analysis:
  Group A importance: {imp_bias_A:.4f}
  Group B importance: {imp_bias_B:.4f}
  Gap: {imp_bias_A-imp_bias_B:.4f}

Next: hyperparameter tuning to reduce {primary_feature} dependence
""")