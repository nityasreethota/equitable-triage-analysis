"""
grid_vs_bayesian.py
Comparison of Grid Search vs Bayesian Optimisation.
Works in CHRONOSIG and CAMHS mode.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    PRIMARY_BIAS_FEATURE, COLOR_A, COLOR_B,
    FIGURE_TITLE_SUFFIX, get_output_path,
    CURRENT_MODE, DATA_DIR
)
from src.utils import print_header, ensure_output_dir

# ─────────────────────────────────────────
# 1. LOAD RESULTS
# ─────────────────────────────────────────

grid_path = os.path.join(
    DATA_DIR, f'grid_search_results_{CURRENT_MODE}.csv'
)
bayes_path = os.path.join(
    DATA_DIR,
    f'bayesian_optimisation_results_{CURRENT_MODE}.csv'
)

try:
    grid_results = pd.read_csv(grid_path)
    bayes_results = pd.read_csv(bayes_path)
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Run hyperparameter_tuning.py and "
          "bayesian_optimisation.py first.")
    sys.exit(1)

primary_feature = PRIMARY_BIAS_FEATURE

print_header("GRID SEARCH vs BAYESIAN OPTIMISATION")
print(f"Mode: {CURRENT_MODE}")
print(f"Primary bias feature: {primary_feature}")
print(f"Grid Search: {len(grid_results)} combinations")
print(f"Bayesian Opt: {len(bayes_results)} evaluations")

# ─────────────────────────────────────────
# 2. BEST RESULTS
# ─────────────────────────────────────────

grid_best = grid_results.loc[grid_results['tpr_gap'].idxmin()]
bayes_best = bayes_results.loc[bayes_results['tpr_gap'].idxmin()]

print_header("COMPARISON")
print(f"\n{'Method':<30} {'Grid Search':>15} {'Bayesian':>15}")
print(f"{'─'*60}")
print(f"{'Combinations tested':<30} "
      f"{len(grid_results):>15} {len(bayes_results):>15}")
print(f"{'Strategy':<30} {'Exhaustive':>15} {'GP-guided':>15}")
print(f"{'─'*60}")
print(f"{'max_features':<30} "
      f"{str(grid_best['max_features']):>15} "
      f"{int(bayes_best['max_features']):>15}")
print(f"{'max_samples':<30} "
      f"{grid_best['max_samples']:>15.3f} "
      f"{bayes_best['max_samples']:>15.3f}")
print(f"{'max_depth':<30} "
      f"{int(grid_best['max_depth']):>15} "
      f"{int(bayes_best['max_depth']):>15}")
print(f"{'─'*60}")
print(f"{primary_feature+' I_j':<30} "
      f"{grid_best['referral_importance']:>15.4f} "
      f"{bayes_best['referral_importance']:>15.4f}")
print(f"{'TPR gap':<30} "
      f"{grid_best['tpr_gap']:>15.3f} "
      f"{bayes_best['tpr_gap']:>15.3f}")

# ─────────────────────────────────────────
# 3. VISUALISATIONS
# ─────────────────────────────────────────

ensure_output_dir()
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: TPR gap distribution
axes[0, 0].hist(grid_results['tpr_gap'], bins=15,
                color='coral', alpha=0.7,
                label=f'Grid (n={len(grid_results)})')
axes[0, 0].hist(bayes_results['tpr_gap'], bins=15,
                color='steelblue', alpha=0.7,
                label=f'Bayesian (n={len(bayes_results)})')
axes[0, 0].axvline(x=grid_best['tpr_gap'],
                    color='red', linestyle='--',
                    label=f"Grid best={grid_best['tpr_gap']:.3f}")
axes[0, 0].axvline(x=bayes_best['tpr_gap'],
                    color='blue', linestyle=':',
                    label=f"Bayes best={bayes_best['tpr_gap']:.3f}")
axes[0, 0].set_xlabel('TPR Gap')
axes[0, 0].set_title('TPR Gap Distribution')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Bias importance distribution
axes[0, 1].hist(grid_results['referral_importance'],
                bins=15, color='coral', alpha=0.7,
                label='Grid Search')
axes[0, 1].hist(bayes_results['referral_importance'],
                bins=15, color='steelblue', alpha=0.7,
                label='Bayesian Opt')
axes[0, 1].axvline(
    x=grid_results['referral_importance'].min(),
    color='red', linestyle='--',
    label=f"Grid best={grid_results['referral_importance'].min():.4f}")
axes[0, 1].axvline(
    x=bayes_results['referral_importance'].min(),
    color='blue', linestyle=':',
    label=f"Bayes best={bayes_results['referral_importance'].min():.4f}")
axes[0, 1].set_xlabel(f'{primary_feature} I_j')
axes[0, 1].set_title('Proxy Bias Distribution')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Trade-off scatter
axes[0, 2].scatter(grid_results['referral_importance'],
                    grid_results['tpr_gap'],
                    color='coral', alpha=0.6, s=60,
                    marker='o', label='Grid Search')
axes[0, 2].scatter(bayes_results['referral_importance'],
                    bayes_results['tpr_gap'],
                    color='steelblue', alpha=0.6, s=60,
                    marker='^', label='Bayesian Opt')
axes[0, 2].scatter([grid_best['referral_importance']],
                    [grid_best['tpr_gap']],
                    color='red', s=200, marker='*',
                    zorder=5, label='Grid best')
axes[0, 2].scatter([bayes_best['referral_importance']],
                    [bayes_best['tpr_gap']],
                    color='blue', s=200, marker='*',
                    zorder=5, label='Bayes best')
axes[0, 2].set_xlabel(f'{primary_feature} I_j')
axes[0, 2].set_ylabel('TPR Gap')
axes[0, 2].set_title('Trade-off Space Explored')
axes[0, 2].legend(fontsize=8)
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Head to head
methods = ['Baseline\n(default)', 'Grid\nSearch',
           'Bayesian\nOpt']
tpr_gaps = [
    grid_results['tpr_gap'].min(),
            grid_best['tpr_gap'],
            bayes_best['tpr_gap']]
ref_imps = [0.0238,
            grid_best['referral_importance'],
            bayes_best['referral_importance']]
tpr_As = [0.353,
          grid_best.get('tpr_A', 0),
          bayes_best.get('tpr_A', 0)]

x = np.arange(len(methods))
w = 0.25
axes[1, 0].bar(x - w, tpr_gaps, w,
               color='coral', alpha=0.8,
               label='TPR Gap')
axes[1, 0].bar(x, tpr_As, w,
               color='steelblue', alpha=0.8,
               label='TPR Group A')
axes[1, 0].bar(x + w, ref_imps, w,
               color='purple', alpha=0.8,
               label=f'{primary_feature[:10]} I_j')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(methods)
axes[1, 0].set_ylabel('Value')
axes[1, 0].set_title('Head to Head Comparison')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: max_features vs TPR gap
axes[1, 1].scatter(grid_results['max_features'].astype(str),
                    grid_results['tpr_gap'],
                    color='coral', alpha=0.6,
                    label='Grid Search')
axes[1, 1].scatter(bayes_results['max_features'],
                    bayes_results['tpr_gap'],
                    color='steelblue', alpha=0.6,
                    label='Bayesian Opt')
axes[1, 1].set_xlabel('max_features')
axes[1, 1].set_ylabel('TPR Gap')
axes[1, 1].set_title('max_features vs TPR Gap')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: max_samples vs TPR gap
axes[1, 2].scatter(grid_results['max_samples'],
                    grid_results['tpr_gap'],
                    color='coral', alpha=0.6,
                    label='Grid Search')
axes[1, 2].scatter(bayes_results['max_samples'],
                    bayes_results['tpr_gap'],
                    color='steelblue', alpha=0.6,
                    label='Bayesian Opt')
axes[1, 2].set_xlabel('max_samples')
axes[1, 2].set_ylabel('TPR Gap')
axes[1, 2].set_title('max_samples vs TPR Gap')
axes[1, 2].legend(fontsize=8)
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle(f'Grid Search vs Bayesian Optimisation\n'
             f'{FIGURE_TITLE_SUFFIX}',
             fontsize=12, fontweight='bold')
plt.tight_layout()

filepath = get_output_path('grid_vs_bayesian.png')
plt.savefig(filepath, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved to {filepath}")

print_header("KEY FINDINGS")
winner = "Bayesian" if bayes_best['tpr_gap'] < grid_best['tpr_gap'] \
    else "Grid Search"
print(f"""
Mode: {CURRENT_MODE}
Winner: {winner}

Grid Search: TPR gap={grid_best['tpr_gap']:.3f},
             max_samples={grid_best['max_samples']:.3f}
Bayesian:    TPR gap={bayes_best['tpr_gap']:.3f},
             max_samples={bayes_best['max_samples']:.3f}

Bayesian explores continuous space - finds values
between discrete grid points.
Both confirm Stage 2 threshold correction still required.
""")