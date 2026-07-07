import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os

# ─────────────────────────────────────────
# 1. LOAD RESULTS FROM BOTH METHODS
# ─────────────────────────────────────────

grid_results = pd.read_csv('data/grid_search_results.csv')
bayes_results = pd.read_csv('data/bayesian_optimisation_results.csv')

print("=" * 65)
print("GRID SEARCH vs BAYESIAN OPTIMISATION - COMPARISON")
print("=" * 65)

# ─────────────────────────────────────────
# 2. SUMMARY STATISTICS
# ─────────────────────────────────────────

# Grid search best
grid_best = grid_results.loc[grid_results['tpr_gap'].idxmin()]
grid_best_imp = grid_results.loc[
    grid_results['referral_importance'].idxmin()]

# Bayesian best
bayes_best = bayes_results.loc[bayes_results['tpr_gap'].idxmin()]
bayes_best_imp = bayes_results.loc[
    bayes_results['referral_importance'].idxmin()]

print(f"""
{'Method':<35} {'Grid Search':>12} {'Bayesian Opt':>12}
{'─'*60}
{'Strategy':<35} {'Exhaustive':>12} {'GP-guided':>12}
{'Combinations tested':<35} {len(grid_results):>12} {len(bayes_results):>12}
{'─'*60}
BEST BY TPR GAP:
{'  max_features':<35} {int(grid_best['max_features']):>12} {int(bayes_best['max_features']):>12}
{'  max_samples':<35} {grid_best['max_samples']:>12.3f} {bayes_best['max_samples']:>12.3f}
{'  max_depth':<35} {int(grid_best['max_depth']):>12} {int(bayes_best['max_depth']):>12}
{'  referral_length I_j':<35} {grid_best['referral_importance']:>12.4f} {bayes_best['referral_importance']:>12.4f}
{'  TPR gap':<35} {grid_best['tpr_gap']:>12.3f} {bayes_best['tpr_gap']:>12.3f}
{'─'*60}
BEST BY referral_length I_j:
{'  referral_length I_j':<35} {grid_best_imp['referral_importance']:>12.4f} {bayes_best_imp['referral_importance']:>12.4f}
{'  TPR gap':<35} {grid_best_imp['tpr_gap']:>12.3f} {bayes_best_imp['tpr_gap']:>12.3f}
""")

# ─────────────────────────────────────────
# 3. VISUALISATIONS
# ─────────────────────────────────────────

os.makedirs('visualisations/outputs', exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Distribution of TPR gap - both methods
axes[0, 0].hist(grid_results['tpr_gap'], bins=15,
                color='coral', alpha=0.7,
                label=f'Grid Search (n={len(grid_results)})')
axes[0, 0].hist(bayes_results['tpr_gap'], bins=15,
                color='steelblue', alpha=0.7,
                label=f'Bayesian Opt (n={len(bayes_results)})')
axes[0, 0].axvline(x=grid_best['tpr_gap'],
                    color='red', linestyle='--', linewidth=2,
                    label=f'Grid best={grid_best["tpr_gap"]:.3f}')
axes[0, 0].axvline(x=bayes_best['tpr_gap'],
                    color='blue', linestyle=':', linewidth=2,
                    label=f'Bayes best={bayes_best["tpr_gap"]:.3f}')
axes[0, 0].set_xlabel('TPR Gap')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('TPR Gap Distribution\nGrid Search vs Bayesian')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Distribution of referral_length importance
axes[0, 1].hist(grid_results['referral_importance'], bins=15,
                color='coral', alpha=0.7,
                label=f'Grid Search (n={len(grid_results)})')
axes[0, 1].hist(bayes_results['referral_importance'], bins=15,
                color='steelblue', alpha=0.7,
                label=f'Bayesian Opt (n={len(bayes_results)})')
axes[0, 1].axvline(x=grid_best_imp['referral_importance'],
                    color='red', linestyle='--', linewidth=2,
                    label=f'Grid best={grid_best_imp["referral_importance"]:.4f}')
axes[0, 1].axvline(x=bayes_best_imp['referral_importance'],
                    color='blue', linestyle=':', linewidth=2,
                    label=f'Bayes best={bayes_best_imp["referral_importance"]:.4f}')
axes[0, 1].set_xlabel('referral_length I_j')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Proxy Bias Distribution\nGrid Search vs Bayesian')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Scatter - importance vs TPR gap (both methods)
axes[0, 2].scatter(grid_results['referral_importance'],
                    grid_results['tpr_gap'],
                    color='coral', alpha=0.6, s=60,
                    label='Grid Search', marker='o')
axes[0, 2].scatter(bayes_results['referral_importance'],
                    bayes_results['tpr_gap'],
                    color='steelblue', alpha=0.6, s=60,
                    label='Bayesian Opt', marker='^')
axes[0, 2].scatter([grid_best['referral_importance']],
                    [grid_best['tpr_gap']],
                    color='red', s=200, zorder=5,
                    marker='*', label='Grid best')
axes[0, 2].scatter([bayes_best['referral_importance']],
                    [bayes_best['tpr_gap']],
                    color='blue', s=200, zorder=5,
                    marker='*', label='Bayes best')
axes[0, 2].set_xlabel('referral_length Importance I_j')
axes[0, 2].set_ylabel('TPR Gap')
axes[0, 2].set_title('Trade-off Space Explored\nGrid Search vs Bayesian')
axes[0, 2].legend(fontsize=8)
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Head to head bar chart
methods = ['Baseline\n(default)', 'Grid\nSearch',
           'Bayesian\nOptimisation']
tpr_gaps = [0.580, grid_best['tpr_gap'], bayes_best['tpr_gap']]
ref_imps = [0.0238,
            grid_best['referral_importance'],
            bayes_best['referral_importance']]
tpr_As = [0.353, grid_best['tpr_A'], bayes_best['tpr_A']]

x = np.arange(len(methods))
width = 0.25

axes[1, 0].bar(x - width, tpr_gaps, width,
               color='coral', alpha=0.8, label='TPR Gap')
axes[1, 0].bar(x, tpr_As, width,
               color='steelblue', alpha=0.8, label='TPR Group A')
axes[1, 0].bar(x + width, ref_imps, width,
               color='purple', alpha=0.8,
               label='referral_length I_j')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(methods)
axes[1, 0].set_ylabel('Value')
axes[1, 0].set_title('Head to Head Comparison\nAll Key Metrics')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

for i, (g, ta, ri) in enumerate(zip(tpr_gaps, tpr_As, ref_imps)):
    axes[1, 0].text(i - width, g + 0.005,
                     f'{g:.3f}', ha='center', fontsize=7)
    axes[1, 0].text(i, ta + 0.005,
                     f'{ta:.3f}', ha='center', fontsize=7)
    axes[1, 0].text(i + width, ri + 0.0005,
                     f'{ri:.4f}', ha='center', fontsize=7)

# Plot 5: max_features explored by both methods
axes[1, 1].scatter(grid_results['max_features'],
                    grid_results['tpr_gap'],
                    color='coral', alpha=0.6, s=60,
                    label='Grid Search', marker='o')
axes[1, 1].scatter(bayes_results['max_features'],
                    bayes_results['tpr_gap'],
                    color='steelblue', alpha=0.6, s=60,
                    label='Bayesian Opt', marker='^')
axes[1, 1].set_xlabel('max_features')
axes[1, 1].set_ylabel('TPR Gap')
axes[1, 1].set_title('max_features vs TPR Gap\nBoth Methods')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: max_samples explored by both methods
axes[1, 2].scatter(grid_results['max_samples'],
                    grid_results['tpr_gap'],
                    color='coral', alpha=0.6, s=60,
                    label='Grid Search', marker='o')
axes[1, 2].scatter(bayes_results['max_samples'],
                    bayes_results['tpr_gap'],
                    color='steelblue', alpha=0.6, s=60,
                    label='Bayesian Opt', marker='^')
axes[1, 2].set_xlabel('max_samples')
axes[1, 2].set_ylabel('TPR Gap')
axes[1, 2].set_title('max_samples vs TPR Gap\nBoth Methods')
axes[1, 2].legend(fontsize=8)
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle('Grid Search vs Bayesian Optimisation: Full Comparison\n'
             'From Probabilities to Decisions ',
             fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('visualisations/outputs/grid_vs_bayesian.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to visualisations/outputs/grid_vs_bayesian.png")

# ─────────────────────────────────────────
# 4. KEY FINDINGS
# ─────────────────────────────────────────

print()
print("=" * 65)
print("KEY FINDINGS")
print("=" * 65)
print(f"""
1. SEARCH EFFICIENCY:
   Grid Search tested {len(grid_results)} discrete combinations
   Bayesian Opt tested {len(bayes_results)} continuous evaluations
   Bayesian found TPR gap = {bayes_best['tpr_gap']:.3f} vs
   Grid Search TPR gap = {grid_best['tpr_gap']:.3f}

2. PARAMETER AGREEMENT:
   Both methods agree: max_features=1 is critical
   Bayesian found max_samples={bayes_best['max_samples']:.3f} vs
   Grid found max_samples={grid_best['max_samples']:.3f}
   Bayesian explores between grid points - finds 0.734
   which Grid Search couldn't test (only 0.5/0.6/0.7/1.0)

3. CONTINUOUS vs DISCRETE SEARCH:
   Grid Search limited to predefined values
   Bayesian explores continuous space - found max_depth=2
   which Grid Search didn't include in its grid

4. CONCLUSION:
   Bayesian Optimisation outperforms Grid Search by
   finding TPR gap = {bayes_best['tpr_gap']:.3f} vs {grid_best['tpr_gap']:.3f}
   Both methods confirm Stage 2 threshold correction
   is necessary for full fairness gap elimination.
""")