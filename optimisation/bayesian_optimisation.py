import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
import os

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────

df = pd.read_csv('data/synthetic_nhs_data.csv')

FEATURES = ['risk_score', 'referral_length',
            'previous_contacts', 'age']
TARGET = 'true_outcome'

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

groups_test = df.loc[X_test.index, 'group'].values
X_test_A = X_test.values[groups_test == 'A']
y_test_A = y_test.values[groups_test == 'A']
X_test_B = X_test.values[groups_test == 'B']
y_test_B = y_test.values[groups_test == 'B']

print("=" * 65)
print("BAYESIAN OPTIMISATION")
print("=" * 65)
print(f"""
Comparing two hyperparameter optimisation strategies:
Grid Search (discrete) vs Bayesian Optimisation (continuous)
""")

# ─────────────────────────────────────────
# 2. HELPER FUNCTIONS
# ─────────────────────────────────────────

def classification_error(y_true, y_pred):
    return np.mean(y_true != y_pred)

def permutation_importance_referral(rf, X_arr, y_arr,
                                     feature_idx, n_trees=30):
    """Formula: I_j = (1/T) * sum(E_t^j - E_t)"""
    diffs = []
    for t, tree in enumerate(rf.estimators_[:n_trees]):
        y_pred_orig = tree.predict(X_arr)
        E_t = classification_error(y_arr, y_pred_orig)
        X_shuf = X_arr.copy()
        np.random.seed(t)
        np.random.shuffle(X_shuf[:, feature_idx])
        y_pred_shuf = tree.predict(X_shuf)
        E_t_j = classification_error(y_arr, y_pred_shuf)
        diffs.append(E_t_j - E_t)
    return np.mean(diffs)

def compute_tpr_gap(rf, X_A, y_A, X_B, y_B, threshold=0.4):
    prob_A = rf.predict_proba(X_A)[:, 1]
    prob_B = rf.predict_proba(X_B)[:, 1]
    D_A = (prob_A >= threshold).astype(int)
    D_B = (prob_B >= threshold).astype(int)
    TP_A = ((D_A == 1) & (y_A == 1)).sum()
    FN_A = ((D_A == 0) & (y_A == 1)).sum()
    TP_B = ((D_B == 1) & (y_B == 1)).sum()
    FN_B = ((D_B == 0) & (y_B == 1)).sum()
    TPR_A = TP_A / (TP_A + FN_A) if (TP_A + FN_A) > 0 else 0
    TPR_B = TP_B / (TP_B + FN_B) if (TP_B + FN_B) > 0 else 0
    return abs(TPR_B - TPR_A), TPR_A, TPR_B

X_test_arr = X_test.values
j_ref = FEATURES.index('referral_length')

# ─────────────────────────────────────────
# 3. GRID SEARCH BASELINE
# ─────────────────────────────────────────

print("Reconfirming Grid Search best result...")
rf_grid = RandomForestClassifier(
    n_estimators=100,
    max_features=1,
    max_samples=0.6,
    bootstrap=True,
    max_depth=3,
    random_state=42,
    class_weight='balanced'
)
rf_grid.fit(X_train, y_train)

grid_imp = permutation_importance_referral(
    rf_grid, X_test_arr, y_test.values, j_ref
)
grid_gap, grid_tpr_A, grid_tpr_B = compute_tpr_gap(
    rf_grid, X_test_A, y_test_A, X_test_B, y_test_B
)

print(f"Grid Search best:")
print(f"  referral_length I_j: {grid_imp:.4f}")
print(f"  TPR gap:             {grid_gap:.3f}")
print(f"  TPR Group A:         {grid_tpr_A:.3f}")
print()

# ─────────────────────────────────────────
# 4. DEFINE BAYESIAN OPTIMISATION SEARCH SPACE
# Continuous rather than discrete grid
# ─────────────────────────────────────────

search_space = [
    Integer(1, 4, name='max_features'),
    Real(0.5, 1.0, name='max_samples'),
    Integer(2, 8, name='max_depth')
]

# ─────────────────────────────────────────
# 5. DEFINE OBJECTIVE FUNCTION
# Minimise combined score:
# referral_length importance + TPR gap
# ─────────────────────────────────────────

# Track all evaluations
evaluation_history = []

@use_named_args(search_space)
def objective(max_features, max_samples, max_depth):
    """
    Objective function for Bayesian Optimisation.
    Minimises combined bias score:
      score = referral_importance + tpr_gap

    Lower is better.
    """
    try:
        rf = RandomForestClassifier(
            n_estimators=100,
            max_features=int(max_features),
            max_samples=float(max_samples),
            bootstrap=True,
            max_depth=int(max_depth),
            random_state=42,
            class_weight='balanced'
        )
        rf.fit(X_train, y_train)

        imp = permutation_importance_referral(
            rf, X_test_arr, y_test.values, j_ref, n_trees=20
        )
        gap, tpr_a, tpr_b = compute_tpr_gap(
            rf, X_test_A, y_test_A, X_test_B, y_test_B
        )

        # Combined objective - normalise both to similar scale
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

    except Exception as e:
        return 1.0

# ─────────────────────────────────────────
# 6. RUN BAYESIAN OPTIMISATION
# ─────────────────────────────────────────

print("Running Bayesian Optimisation...")
print("(Gaussian Process surrogate model)")
print()

N_CALLS = 50  # number of evaluations
N_INITIAL = 10  # random initial evaluations before GP kicks in

result = gp_minimize(
    func=objective,
    dimensions=search_space,
    n_calls=N_CALLS,
    n_initial_points=N_INITIAL,
    random_state=42,
    verbose=False
)

print(f"Bayesian Optimisation complete")
print(f"Total evaluations: {N_CALLS}")
print(f"Initial random: {N_INITIAL}")
print(f"GP-guided: {N_CALLS - N_INITIAL}")
print()

# ─────────────────────────────────────────
# 7. EXTRACT BEST PARAMETERS
# ─────────────────────────────────────────

best_params_bayes = {
    'max_features': int(result.x[0]),
    'max_samples': float(result.x[1]),
    'max_depth': int(result.x[2])
}

print("=" * 65)
print("BAYESIAN OPTIMISATION RESULTS")
print("=" * 65)
print(f"\nBest parameters found:")
print(f"  max_features: {best_params_bayes['max_features']}")
print(f"  max_samples:  {best_params_bayes['max_samples']:.3f}")
print(f"  max_depth:    {best_params_bayes['max_depth']}")
print(f"  bootstrap:    True (fixed)")
print(f"\nBest objective score: {result.fun:.4f}")

# ─────────────────────────────────────────
# 8. RETRAIN WITH BAYESIAN BEST PARAMETERS
# ─────────────────────────────────────────

rf_bayes = RandomForestClassifier(
    n_estimators=100,
    max_features=best_params_bayes['max_features'],
    max_samples=best_params_bayes['max_samples'],
    bootstrap=True,
    max_depth=best_params_bayes['max_depth'],
    random_state=42,
    class_weight='balanced'
)
rf_bayes.fit(X_train, y_train)

bayes_imp = permutation_importance_referral(
    rf_bayes, X_test_arr, y_test.values, j_ref, n_trees=50
)
bayes_gap, bayes_tpr_A, bayes_tpr_B = compute_tpr_gap(
    rf_bayes, X_test_A, y_test_A, X_test_B, y_test_B
)

# ─────────────────────────────────────────
# 9. COMPARISON TABLE
# ─────────────────────────────────────────

print()
print("=" * 65)
print("BAYESIAN OPTIMISATION")
print("=" * 65)
print(f"""
{'Method':<30} {'Grid Search':>15} {'Bayesian Opt':>15}
{'-'*60}
{'max_features':<30} {'1':>15} {best_params_bayes['max_features']:>15}
{'max_samples':<30} {'0.6':>15} {best_params_bayes['max_samples']:>15.3f}
{'max_depth':<30} {'3':>15} {best_params_bayes['max_depth']:>15}
{'bootstrap':<30} {'True':>15} {'True':>15}
{'-'*60}
{'referral_length I_j':<30} {grid_imp:>15.4f} {bayes_imp:>15.4f}
{'TPR Group A':<30} {grid_tpr_A:>15.3f} {bayes_tpr_A:>15.3f}
{'TPR Group B':<30} {grid_tpr_B:>15.3f} {bayes_tpr_B:>15.3f}
{'TPR Gap':<30} {grid_gap:>15.3f} {bayes_gap:>15.3f}
{'-'*60}
{'Combinations tested':<30} {'48':>15} {N_CALLS:>15}
{'Search strategy':<30} {'Exhaustive':>15} {'GP-guided':>15}
""")

# Determine winner
if bayes_gap < grid_gap:
    print("Bayesian Optimisation achieves lower TPR gap")
elif bayes_gap == grid_gap:
    print("Result: Both methods achieve equivalent TPR gap")
else:
    print("Result: Grid Search achieves lower TPR gap in this case")
    print("Note: Bayesian Optimisation explores continuous space")
    print("      and may find better solutions with more evaluations")

# ─────────────────────────────────────────
# 10. EVALUATION HISTORY ANALYSIS
# ─────────────────────────────────────────

eval_df = pd.DataFrame(evaluation_history)

print()
print("=" * 65)
print("BAYESIAN OPTIMISATION CONVERGENCE ANALYSIS")
print("=" * 65)
print(f"\nEvaluation history ({len(eval_df)} evaluations):")
print(f"\n{'Eval':<6} {'max_feat':<10} {'max_samp':<10} "
      f"{'max_dep':<10} {'ref_imp':<12} {'tpr_gap':<10} {'score'}")
print("-" * 65)

for i, row in eval_df.iterrows():
    print(f"{i+1:<6} {int(row['max_features']):<10} "
          f"{row['max_samples']:<10.3f} "
          f"{int(row['max_depth']):<10} "
          f"{row['referral_importance']:<12.4f} "
          f"{row['tpr_gap']:<10.3f} "
          f"{row['score']:.4f}")

print()
print(f"Best evaluation:")
best_eval = eval_df.loc[eval_df['score'].idxmin()]
print(f"  max_features: {int(best_eval['max_features'])}")
print(f"  max_samples:  {best_eval['max_samples']:.3f}")
print(f"  max_depth:    {int(best_eval['max_depth'])}")
print(f"  ref_imp:      {best_eval['referral_importance']:.4f}")
print(f"  tpr_gap:      {best_eval['tpr_gap']:.3f}")

# ─────────────────────────────────────────
# 11. SAVE RESULTS
# ─────────────────────────────────────────

eval_df.to_csv('data/bayesian_optimisation_results.csv', index=False)
print(f"\nBayesian optimisation results saved to "
      f"data/bayesian_optimisation_results.csv")

# ─────────────────────────────────────────
# 12. VISUALISATIONS
# ─────────────────────────────────────────

os.makedirs('visualisations/outputs', exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Convergence - score over evaluations
cumulative_best = [eval_df['score'][:i+1].min()
                   for i in range(len(eval_df))]
axes[0, 0].plot(range(1, len(eval_df)+1), eval_df['score'],
                'o-', color='steelblue', alpha=0.5,
                linewidth=1, markersize=4,
                label='Each evaluation')
axes[0, 0].plot(range(1, len(eval_df)+1), cumulative_best,
                color='red', linewidth=2,
                label='Best so far')
axes[0, 0].axvline(x=N_INITIAL, color='grey', linestyle='--',
                    label=f'GP starts (n={N_INITIAL})')
axes[0, 0].axhline(y=result.fun, color='green', linestyle=':',
                    label=f'Best={result.fun:.4f}')
axes[0, 0].set_xlabel('Evaluation number')
axes[0, 0].set_ylabel('Objective score')
axes[0, 0].set_title('Bayesian Optimisation Convergence\n'
                      'Score over evaluations')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Distribution of referral_length importance
axes[0, 1].hist(eval_df['referral_importance'],
                bins=20, color='steelblue', alpha=0.8)
axes[0, 1].axvline(x=0.0238, color='red', linestyle='--',
                    linewidth=2,
                    label=f'Baseline I_j=0.0238')
axes[0, 1].axvline(x=bayes_imp, color='green', linestyle='--',
                    linewidth=2,
                    label=f'Best I_j={bayes_imp:.4f}')
axes[0, 1].set_xlabel('referral_length Permutation Importance I_j')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Distribution of referral_length Importance\n'
                      'Across Bayesian Evaluations')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Distribution of TPR gap
axes[1, 0].hist(eval_df['tpr_gap'],
                bins=20, color='coral', alpha=0.8)
axes[1, 0].axvline(x=0.580, color='red', linestyle='--',
                    linewidth=2, label=f'Baseline gap=0.580')
axes[1, 0].axvline(x=bayes_gap, color='green', linestyle='--',
                    linewidth=2,
                    label=f'Best gap={bayes_gap:.3f}')
axes[1, 0].set_xlabel('TPR Gap')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Distribution of TPR Gap\n'
                      'Across Bayesian Evaluations')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Before vs after comparison
metrics = ['referral_length\nImportance I_j',
           'TPR\nGroup A', 'TPR\nGroup B', 'TPR\nGap']
baseline_vals = [0.0238, 0.353, 0.933, 0.580]
bayes_vals = [bayes_imp, bayes_tpr_A, bayes_tpr_B, bayes_gap]

x = np.arange(len(metrics))
width = 0.35
axes[1, 1].bar(x - width/2, baseline_vals, width,
               label='Baseline (default params)',
               color='coral', alpha=0.8)
axes[1, 1].bar(x + width/2, bayes_vals, width,
               label=f'Bayesian best params',
               color='steelblue', alpha=0.8)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(metrics, fontsize=9)
axes[1, 1].set_ylabel('Value')
axes[1, 1].set_title('Before vs After Bayesian Optimisation\n'
                      'Stage 1: Internal Bias Reduction')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

for i, (b, t) in enumerate(zip(baseline_vals, bayes_vals)):
    axes[1, 1].text(i - width/2, b + 0.005,
                     f'{b:.3f}', ha='center', fontsize=8)
    axes[1, 1].text(i + width/2, t + 0.005,
                     f'{t:.3f}', ha='center', fontsize=8)

plt.suptitle('Bayesian Optimisation: Results\n'
             'From Probabilities to Decisions',
             fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('visualisations/outputs/bayesian_optimisation.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to visualisations/outputs/bayesian_optimisation.png")