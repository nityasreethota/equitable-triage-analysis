"""
hyperparameter_tuning.py
Grid Search hyperparameter tuning - CHRONOSIG and CAMHS mode.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    FEATURES, PRIMARY_BIAS_FEATURE, PRIMARY_BIAS_IDX,
    GRID_SEARCH_PARAMS, COLOR_A, COLOR_B,
    FIGURE_TITLE_SUFFIX, get_output_path,
    CURRENT_MODE, DATA_DIR, RANDOM_STATE
)
from src.data_loader import load_and_split
from src.models import get_baseline_model, train_model
from src.metrics import compute_tpr_gap
from src.permutation import permutation_importance_single
from src.utils import print_header, ensure_output_dir

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────

(df, X_train, X_test, y_train, y_test,
 groups, X_A, y_A, X_B, y_B) = load_and_split()

X_test_arr = X_test.values
y_test_arr = y_test.values
j_bias = PRIMARY_BIAS_IDX
primary_feature = PRIMARY_BIAS_FEATURE

print_header("HYPERPARAMETER TUNING - GRID SEARCH")
print(f"Mode: {CURRENT_MODE}")
print(f"Primary bias feature: {primary_feature}")

# ─────────────────────────────────────────
# 2. HELPER FUNCTIONS
# ─────────────────────────────────────────

from src.metrics import classification_error

def perm_imp_bias(rf, n_trees=30):
    I_j, _ = permutation_importance_single(
        rf, X_test_arr, y_test_arr, j_bias, n_trees
    )
    return I_j

def tpr_gap_uniform(rf, threshold=0.4):
    gap, tpr_a, tpr_b = compute_tpr_gap(
        rf, X_A, y_A, X_B, y_B, threshold
    )
    return gap, tpr_a, tpr_b

# ─────────────────────────────────────────
# 3. BASELINE
# ─────────────────────────────────────────

rf_baseline = train_model(get_baseline_model(), X_train, y_train)
baseline_imp = perm_imp_bias(rf_baseline)
baseline_gap, baseline_tpr_A, baseline_tpr_B = tpr_gap_uniform(rf_baseline)

print(f"\nBaseline:")
print(f"  {primary_feature} I_j: {baseline_imp:.4f}")
print(f"  TPR gap: {baseline_gap:.3f}")
print(f"  TPR A: {baseline_tpr_A:.3f}, TPR B: {baseline_tpr_B:.3f}")

# ─────────────────────────────────────────
# 4. GRID SEARCH
# ─────────────────────────────────────────

keys = list(GRID_SEARCH_PARAMS.keys())
values = list(GRID_SEARCH_PARAMS.values())
combinations = list(itertools.product(*values))

print(f"\nGrid search: {len(combinations)} combinations...")

results = []
for i, combo in enumerate(combinations):
    params = dict(zip(keys, combo))
    if not params['bootstrap'] and params['max_samples'] != 1.0:
        continue
    try:
        rf = RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE,
            class_weight='balanced', **params
        )
        rf.fit(X_train, y_train)
        imp = perm_imp_bias(rf, n_trees=20)
        gap, tpr_a, tpr_b = tpr_gap_uniform(rf)
        results.append({
            'max_features': params['max_features'],
            'max_samples': params['max_samples'],
            'bootstrap': params['bootstrap'],
            'max_depth': params['max_depth'],
            'referral_importance': imp,
            'tpr_gap': gap,
            'tpr_A': tpr_a,
            'tpr_B': tpr_b
        })
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(combinations)} done...")
    except Exception:
        continue

results_df = pd.DataFrame(results)
print(f"Completed: {len(results_df)} valid combinations")

# ─────────────────────────────────────────
# 5. BEST PARAMETERS
# ─────────────────────────────────────────

results_df['combined_score'] = (
    results_df['referral_importance'] /
    results_df['referral_importance'].max() +
    results_df['tpr_gap'] /
    results_df['tpr_gap'].max()
)
best = results_df.loc[results_df['combined_score'].idxmin()]

print_header("GRID SEARCH RESULTS")
print(f"\nBest parameters (combined score):")
print(f"  max_features: {best['max_features']}")
print(f"  max_samples:  {best['max_samples']:.3f}")
print(f"  bootstrap:    {best['bootstrap']}")
print(f"  max_depth:    {int(best['max_depth'])}")
print(f"\n{'Metric':<35} {'Baseline':>10} {'Tuned':>10}")
print(f"{'─'*55}")
print(f"  {primary_feature} I_j {'':10} "
      f"{baseline_imp:>10.4f} {best['referral_importance']:>10.4f}")
print(f"  TPR Group A {'':13} "
      f"{baseline_tpr_A:>10.3f} {best['tpr_A']:>10.3f}")
print(f"  TPR Group B {'':13} "
      f"{baseline_tpr_B:>10.3f} {best['tpr_B']:>10.3f}")
print(f"  TPR Gap {'':17} "
      f"{baseline_gap:>10.3f} {best['tpr_gap']:>10.3f}")

# ─────────────────────────────────────────
# 6. SAVE RESULTS
# ─────────────────────────────────────────

results_path = os.path.join(
    DATA_DIR, f'grid_search_results_{CURRENT_MODE}.csv'
)
results_df.to_csv(results_path, index=False)
print(f"\nResults saved to {results_path}")

# ─────────────────────────────────────────
# 7. VISUALISATIONS
# ─────────────────────────────────────────

ensure_output_dir()
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Distribution of bias importance
axes[0, 0].hist(results_df['referral_importance'],
                bins=20, color='steelblue', alpha=0.8)
axes[0, 0].axvline(x=baseline_imp, color='red',
                    linestyle='--',
                    label=f'Baseline={baseline_imp:.4f}')
axes[0, 0].axvline(x=best['referral_importance'],
                    color='green', linestyle='--',
                    label=f"Best={best['referral_importance']:.4f}")
axes[0, 0].set_xlabel(f'{primary_feature} Importance I_j')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title(f'{primary_feature} Importance Distribution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Distribution of TPR gap
axes[0, 1].hist(results_df['tpr_gap'],
                bins=20, color='coral', alpha=0.8)
axes[0, 1].axvline(x=baseline_gap, color='red',
                    linestyle='--',
                    label=f'Baseline={baseline_gap:.3f}')
axes[0, 1].axvline(x=best['tpr_gap'], color='green',
                    linestyle='--',
                    label=f"Best={best['tpr_gap']:.3f}")
axes[0, 1].set_xlabel('TPR Gap')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('TPR Gap Distribution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Trade-off scatter
scatter = axes[1, 0].scatter(
    results_df['referral_importance'],
    results_df['tpr_gap'],
    c=results_df['max_depth'],
    cmap='viridis', alpha=0.6, s=50
)
plt.colorbar(scatter, ax=axes[1, 0], label='max_depth')
axes[1, 0].scatter([baseline_imp], [baseline_gap],
                    color='red', s=200, marker='*',
                    zorder=5, label='Baseline')
axes[1, 0].scatter([best['referral_importance']],
                    [best['tpr_gap']],
                    color='green', s=200, marker='*',
                    zorder=5, label='Best')
axes[1, 0].set_xlabel(f'{primary_feature} I_j')
axes[1, 0].set_ylabel('TPR Gap')
axes[1, 0].set_title('Trade-off: Proxy Reliance vs Fairness Gap')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Before vs after
metrics = [f'{primary_feature[:12]}\nI_j',
           'TPR\nGroup A', 'TPR\nGroup B', 'TPR\nGap']
baseline_vals = [baseline_imp, baseline_tpr_A,
                 baseline_tpr_B, baseline_gap]
tuned_vals = [best['referral_importance'], best['tpr_A'],
              best['tpr_B'], best['tpr_gap']]

x = np.arange(len(metrics))
w = 0.35
axes[1, 1].bar(x - w/2, baseline_vals, w,
               label='Baseline', color='coral', alpha=0.8)
axes[1, 1].bar(x + w/2, tuned_vals, w,
               label='Grid Search Tuned',
               color='steelblue', alpha=0.8)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(metrics, fontsize=9)
axes[1, 1].set_ylabel('Value')
axes[1, 1].set_title('Before vs After Tuning')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
for i, (b, t) in enumerate(zip(baseline_vals, tuned_vals)):
    axes[1, 1].text(i-w/2, b+0.005, f'{b:.3f}',
                     ha='center', fontsize=8)
    axes[1, 1].text(i+w/2, t+0.005, f'{t:.3f}',
                     ha='center', fontsize=8)

plt.suptitle(f'Hyperparameter Tuning: Grid Search\n'
             f'{FIGURE_TITLE_SUFFIX}',
             fontsize=12, fontweight='bold')
plt.tight_layout()

filepath = get_output_path('hyperparameter_tuning.png')
plt.savefig(filepath, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved to {filepath}")

print_header("SUMMARY")
print(f"""
Mode: {CURRENT_MODE}
Primary bias feature: {primary_feature}

Grid Search found best params:
  max_features={best['max_features']},
  max_samples={best['max_samples']:.3f},
  max_depth={int(best['max_depth'])}

Stage 1 results:
  {primary_feature} I_j: {baseline_imp:.4f} -> {best['referral_importance']:.4f}
  TPR gap: {baseline_gap:.3f} -> {best['tpr_gap']:.3f}

Stage 2 (threshold correction) still required.
Results saved to {results_path}
""")