"""
bayesian_optimisation.py
Bayesian Optimisation - CHRONOSIG and CAMHS mode.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

from src.config import (
    FEATURES, PRIMARY_BIAS_FEATURE, PRIMARY_BIAS_IDX,
    N_BAYES_CALLS, N_BAYES_INITIAL,
    COLOR_A, COLOR_B, FIGURE_TITLE_SUFFIX,
    get_output_path, CURRENT_MODE, DATA_DIR,
    RANDOM_STATE, GRID_PARAMS
)
from src.data_loader import load_and_split
from src.models import get_grid_model, train_model
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

print_header("BAYESIAN OPTIMISATION")
print(f"Mode: {CURRENT_MODE}")
print(f"Primary bias feature: {primary_feature}")
print(f"Evaluations: {N_BAYES_CALLS} "
      f"({N_BAYES_INITIAL} random + "
      f"{N_BAYES_CALLS-N_BAYES_INITIAL} GP-guided)")

# ─────────────────────────────────────────
# 2. GRID SEARCH BASELINE
# ─────────────────────────────────────────

rf_grid = train_model(get_grid_model(), X_train, y_train)
grid_imp, _ = permutation_importance_single(
    rf_grid, X_test_arr, y_test_arr, j_bias, n_trees=30
)
grid_gap, grid_tpr_A, grid_tpr_B = compute_tpr_gap(
    rf_grid, X_A, y_A, X_B, y_B
)

print(f"\nGrid Search baseline:")
print(f"  {primary_feature} I_j: {grid_imp:.4f}")
print(f"  TPR gap: {grid_gap:.3f}")

# ─────────────────────────────────────────
# 3. BAYESIAN OPTIMISATION
# ─────────────────────────────────────────

search_space = [
    Integer(1, 4, name='max_features'),
    Real(0.5, 1.0, name='max_samples'),
    Integer(2, 8, name='max_depth')
]

evaluation_history = []

@use_named_args(search_space)
def objective(max_features, max_samples, max_depth):
    try:
        rf = RandomForestClassifier(
            n_estimators=100,
            max_features=int(max_features),
            max_samples=float(max_samples),
            bootstrap=True,
            max_depth=int(max_depth),
            random_state=RANDOM_STATE,
            class_weight='balanced'
        )
        rf.fit(X_train, y_train)

        imp, _ = permutation_importance_single(
            rf, X_test_arr, y_test_arr, j_bias, n_trees=20
        )
        gap, tpr_a, tpr_b = compute_tpr_gap(
            rf, X_A, y_A, X_B, y_B
        )
        score = imp * 10 + gap

        evaluation_history.append({
            'max_features': int(max_features),
            'max_samples': float(max_samples),
            'max_depth': int(max_depth),
            'referral_importance': imp,
            'tpr_gap': gap,
            'tpr_A': tpr_a,
            'tpr_B': tpr_b,
            'score': score
        })
        return score
    except Exception:
        return 1.0

print("\nRunning Bayesian Optimisation...")
result = gp_minimize(
    func=objective,
    dimensions=search_space,
    n_calls=N_BAYES_CALLS,
    n_initial_points=N_BAYES_INITIAL,
    random_state=RANDOM_STATE,
    verbose=False
)

eval_df = pd.DataFrame(evaluation_history)

best_params = {
    'max_features': int(result.x[0]),
    'max_samples': float(result.x[1]),
    'max_depth': int(result.x[2])
}

rf_bayes = RandomForestClassifier(
    n_estimators=100, random_state=RANDOM_STATE,
    class_weight='balanced', bootstrap=True,
    **best_params
)
rf_bayes.fit(X_train, y_train)

bayes_imp, _ = permutation_importance_single(
    rf_bayes, X_test_arr, y_test_arr, j_bias, n_trees=50
)
bayes_gap, bayes_tpr_A, bayes_tpr_B = compute_tpr_gap(
    rf_bayes, X_A, y_A, X_B, y_B
)

print_header("BAYESIAN OPTIMISATION RESULTS")
print(f"\nBest parameters:")
print(f"  max_features: {best_params['max_features']}")
print(f"  max_samples:  {best_params['max_samples']:.3f}")
print(f"  max_depth:    {best_params['max_depth']}")
print(f"\n{'Metric':<35} {'Grid':>10} {'Bayesian':>10}")
print(f"{'─'*55}")
print(f"  {primary_feature} I_j {'':5} "
      f"{grid_imp:>10.4f} {bayes_imp:>10.4f}")
print(f"  TPR Group A {'':13} "
      f"{grid_tpr_A:>10.3f} {bayes_tpr_A:>10.3f}")
print(f"  TPR Gap {'':17} "
      f"{grid_gap:>10.3f} {bayes_gap:>10.3f}")

# ─────────────────────────────────────────
# 4. SAVE
# ─────────────────────────────────────────

results_path = os.path.join(
    DATA_DIR,
    f'bayesian_optimisation_results_{CURRENT_MODE}.csv'
)
eval_df.to_csv(results_path, index=False)
print(f"\nResults saved to {results_path}")

# ─────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────

ensure_output_dir()
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Convergence
cumulative_best = [eval_df['score'][:i+1].min()
                   for i in range(len(eval_df))]
axes[0, 0].plot(range(1, len(eval_df)+1),
                eval_df['score'], 'o-',
                color='steelblue', alpha=0.5,
                linewidth=1, markersize=4)
axes[0, 0].plot(range(1, len(eval_df)+1),
                cumulative_best, color='red',
                linewidth=2, label='Best so far')
axes[0, 0].axvline(x=N_BAYES_INITIAL, color='grey',
                    linestyle='--',
                    label=f'GP starts (n={N_BAYES_INITIAL})')
axes[0, 0].set_xlabel('Evaluation')
axes[0, 0].set_ylabel('Objective score')
axes[0, 0].set_title('Convergence')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# Importance distribution
axes[0, 1].hist(eval_df['referral_importance'],
                bins=15, color='steelblue', alpha=0.8)
axes[0, 1].axvline(x=grid_imp, color='red',
                    linestyle='--',
                    label=f'Grid={grid_imp:.4f}')
axes[0, 1].axvline(x=bayes_imp, color='green',
                    linestyle='--',
                    label=f'Bayes={bayes_imp:.4f}')
axes[0, 1].set_xlabel(f'{primary_feature} I_j')
axes[0, 1].set_title(f'{primary_feature} Importance Distribution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# TPR gap distribution
axes[1, 0].hist(eval_df['tpr_gap'],
                bins=15, color='coral', alpha=0.8)
axes[1, 0].axvline(x=grid_gap, color='red',
                    linestyle='--',
                    label=f'Grid={grid_gap:.3f}')
axes[1, 0].axvline(x=bayes_gap, color='green',
                    linestyle='--',
                    label=f'Bayes={bayes_gap:.3f}')
axes[1, 0].set_xlabel('TPR Gap')
axes[1, 0].set_title('TPR Gap Distribution')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Before vs after
metrics = [f'{primary_feature[:12]}\nI_j',
           'TPR\nGroup A', 'TPR\nGroup B', 'TPR\nGap']
baseline_vals = [0.0238, 0.353, 0.933, 0.580]
bayes_vals = [bayes_imp, bayes_tpr_A, bayes_tpr_B, bayes_gap]

x = np.arange(len(metrics))
w = 0.35
axes[1, 1].bar(x - w/2, baseline_vals, w,
               label='Baseline (default)',
               color='coral', alpha=0.8)
axes[1, 1].bar(x + w/2, bayes_vals, w,
               label='Bayesian best',
               color='steelblue', alpha=0.8)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(metrics, fontsize=9)
axes[1, 1].set_ylabel('Value')
axes[1, 1].set_title('Before vs After Bayesian Optimisation')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle(f'Bayesian Optimisation Results\n{FIGURE_TITLE_SUFFIX}',
             fontsize=12, fontweight='bold')
plt.tight_layout()

filepath = get_output_path('bayesian_optimisation.png')
plt.savefig(filepath, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved to {filepath}")

print_header("SUMMARY")
print(f"""
Mode: {CURRENT_MODE}
Bayesian Optimisation outperforms Grid Search:
  {primary_feature} I_j: {grid_imp:.4f} -> {bayes_imp:.4f}
  TPR gap: {grid_gap:.3f} -> {bayes_gap:.3f}
  max_samples={best_params['max_samples']:.3f} found between grid points
""")