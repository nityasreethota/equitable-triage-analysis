import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# # ─────────────────────────────────────────
# # 1. LOAD DATA AND TRAIN MODEL
# # ─────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    FEATURES, P_A, P_B, UNIFORM_THRESHOLD, COLOR_A, COLOR_B,
    FIGURE_TITLE_SUFFIX, OUTPUT_DIR
)
from src.data_loader import load_and_split
from src.models import get_grid_model, train_model
from src.metrics import (
    compute_metrics_at_threshold,
    find_optimal_threshold
)
from src.utils import (
    print_header, save_plot, ensure_output_dir
)

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────

(df, X_train, X_test, y_train, y_test,
 groups, X_A, y_A, X_B, y_B) = load_and_split()

print_header("THRESHOLD OPTIMISATION ANALYSIS")
print(f"Training set: {len(X_train)} patients")
print(f"Test set:     {len(X_test)} patients")

# ─────────────────────────────────────────
# 2. TRAIN MODEL
# ─────────────────────────────────────────

rf = train_model(get_grid_model(), X_train, y_train)
print(f"\n  treshold optimisation- GRID model trained")
print(f"Trees: {rf.n_estimators}, max_depth: {rf.max_depth}")

#------------------------------------------------
#3. COMPUTE METRICS
#------------------------------------------------
m_A = compute_metrics_at_threshold(X_A, y_A, rf)
m_B = compute_metrics_at_threshold(X_B, y_B, rf)

# ─────────────────────────────────────────
# 2. BAYESIAN LOSS FRAMEWORK
# ─────────────────────────────────────────

print("=" * 65)
print("BAYESIAN LOSS FRAMEWORK")
print("=" * 65)
# ─────────────────────────────────────────
# 3. COST SENSITIVITY ANALYSIS
# Show how optimal threshold changes with cost ratio
# ─────────────────────────────────────────

print("=" * 65)
print("COST SENSITIVITY ANALYSIS")
print("=" * 65)

m_A = compute_metrics_at_threshold(X_A, y_A, rf)
m_B = compute_metrics_at_threshold(X_B, y_B, rf)

# Test different cost ratios
cost_ratios = [1, 2, 3, 5, 10, 20]
C_FP = 1.0

print(f"\n{'Cost Ratio':<12} {'τ_A (optimal)':<16} "
      f"{'τ_B (optimal)':<16} {'Difference':<12}")
print("-" * 58)

for ratio in cost_ratios:
    C_FN = ratio * C_FP
    thresholds_A, losses_A, tpr_A, fpr_A = find_optimal_threshold(rf, X_A, y_A, C_FN, C_FP)
    thresholds_B, losses_B, tpr_B, fpr_B = find_optimal_threshold(rf, X_B, y_B, C_FN, C_FP)
        
    print(f"c_FN/c_FP={ratio:<3} {thresholds_A:<16.3f} "
          f"{thresholds_B:<16.3f} {thresholds_B-thresholds_A:<12.3f}")

print("""
Observation: As c_FN/c_FP increases (missing someone costs more),
both thresholds decrease - but τ_A decreases faster than τ_B.
This reflects the higher cost of missing Group A patients.
""")

# ─────────────────────────────────────────
# 4. SELECTED COST RATIO: c_FN = 5 * c_FP
# ─────────────────────────────────────────

C_FN = 5.0
C_FP = 1.0

thresholds_A_opt, losses_A, tpr_A_opt, fpr_A_opt = find_optimal_threshold(
    rf, X_A, y_A, C_FN, C_FP
)
thresholds_B_opt, losses_B, tpr_B_opt, fpr_B_opt = find_optimal_threshold(
    rf, X_B, y_B, C_FN, C_FP
)

print("=" * 65)
print(f"SELECTED COST PARAMETERS: c_FN={C_FN}, c_FP={C_FP}")
print("=" * 65)
print(f"""
Group A:
  Base rate P_A:          {P_A:.3f}
  Optimal threshold τ_A:  {thresholds_A_opt:.3f}
  TPR at τ_A:             {tpr_A_opt:.3f}
  FPR at τ_A:             {fpr_A_opt:.3f}
  Minimum expected loss:  {losses_A:.4f}

Group B:
  Base rate P_B:          {P_B:.3f}
  Optimal threshold τ_B:  {thresholds_B_opt:.3f}
  TPR at τ_B:             {tpr_B_opt:.3f}
  FPR at τ_B:             {fpr_B_opt:.3f}
  Minimum expected loss:  {losses_B:.4f}

Uniform threshold τ=0.4 for comparison:
  τ_A - τ_uniform:        {thresholds_A_opt - 0.4:.3f}
  τ_B - τ_uniform:        {thresholds_B_opt - 0.4:.3f}
""")

# ─────────────────────────────────────────
# 5. COMPUTE METRICS ACROSS ALL THRESHOLDS
# ─────────────────────────────────────────

thresholds_range = np.linspace(0.1, 0.9, 100)

def metrics_at_threshold(df_group, tau):
    D = (df_group['predicted_prob'] >= tau).astype(int)
    T = df_group['true_outcome'].values
    TP = ((D == 1) & (T == 1)).sum()
    FP = ((D == 1) & (T == 0)).sum()
    TN = ((D == 0) & (T == 0)).sum()
    FN = ((D == 0) & (T == 1)).sum()
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FNR = 1 - TPR
    return TPR, FPR, FNR

# Compute losses across threshold range
losses_A_range = []
losses_B_range = []
tpr_A_range = []
tpr_B_range = []
fpr_A_range = []
fpr_B_range = []

for tau in thresholds_range:
    m_A_optimal = compute_metrics_at_threshold(X_A, y_A, rf, tau)
    m_B_optimal = compute_metrics_at_threshold(X_B, y_B, rf, tau)

    loss_a = C_FN * m_A_optimal['FNR'] * P_A + C_FP * m_A_optimal['FPR'] * (1 - P_A)
    loss_b = C_FN * m_B_optimal['FNR'] * P_B + C_FP * m_B_optimal['FPR'] * (1 - P_B)

    losses_A_range.append(loss_a)
    losses_B_range.append(loss_b)
    tpr_A_range.append(m_A_optimal['TPR'])
    tpr_B_range.append(m_B_optimal['TPR'])
    fpr_A_range.append(m_A_optimal['FPR'])
    fpr_B_range.append(m_B_optimal['FPR'])

# ─────────────────────────────────────────
# 6. VISUALISATIONS
# ─────────────────────────────────────────

os.makedirs('visualisations/outputs', exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Expected loss curves
axes[0, 0].plot(thresholds_range, losses_A_range,
                color='steelblue', linewidth=2,
                label='Group A expected loss')
axes[0, 0].plot(thresholds_range, losses_B_range,
                color='coral', linewidth=2,
                label='Group B expected loss')
axes[0, 0].axvline(x=0.4, color='black', linestyle='--',
                    linewidth=2, label='Uniform τ=0.4')
axes[0, 0].axvline(x=thresholds_A_opt, color='steelblue',
                    linestyle=':', linewidth=2,
                    label=f'Optimal τ_A={thresholds_A_opt:.3f}')
axes[0, 0].axvline(x=thresholds_B_opt, color='coral',
                    linestyle=':', linewidth=2,
                    label=f'Optimal τ_B={thresholds_B_opt:.3f}')
axes[0, 0].scatter([thresholds_A_opt], [min(losses_A_range)],
                    color='steelblue', s=100, zorder=5)
axes[0, 0].scatter([thresholds_B_opt], [min(losses_B_range)],
                    color='coral', s=100, zorder=5)
axes[0, 0].set_xlabel('Threshold τ')
axes[0, 0].set_ylabel('Expected Loss E[L]')
axes[0, 0].set_title(f'Bayesian Expected Loss Curves\n'
                      f'c_FN={C_FN}, c_FP={C_FP}')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Cost sensitivity - how optimal threshold changes
tau_A_by_ratio = []
tau_B_by_ratio = []
cost_ratios_plot = np.linspace(1, 20, 50)

for ratio in cost_ratios_plot:
    t_A, losses_A, tpr_A_opt, fpr_A_opt = find_optimal_threshold(
    rf, X_A, y_A, C_FN, C_FP
    )
    t_B, losses_B, tpr_B_opt, fpr_B_opt = find_optimal_threshold(
        rf, X_B, y_B, C_FN, C_FP
    )
    tau_A_by_ratio.append(t_A)
    tau_B_by_ratio.append(t_B)

axes[0, 1].plot(cost_ratios_plot, tau_A_by_ratio,
                color='steelblue', linewidth=2,
                label='Group A optimal threshold')
axes[0, 1].plot(cost_ratios_plot, tau_B_by_ratio,
                color='coral', linewidth=2,
                label='Group B optimal threshold')
axes[0, 1].axhline(y=0.4, color='black', linestyle='--',
                    label='Uniform threshold τ=0.4')
axes[0, 1].axvline(x=5, color='grey', linestyle=':',
                    label='Selected ratio (c_FN/c_FP=5)')
axes[0, 1].set_xlabel('Cost Ratio c_FN / c_FP')
axes[0, 1].set_ylabel('Optimal Threshold τ*')
axes[0, 1].set_title('Cost Sensitivity Analysis:\n'
                      'How Optimal Threshold Varies with Cost Ratio')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: TPR across thresholds
axes[1, 0].plot(thresholds_range, tpr_A_range,
                color='steelblue', linewidth=2,
                label='Group A TPR')
axes[1, 0].plot(thresholds_range, tpr_B_range,
                color='coral', linewidth=2,
                label='Group B TPR')
axes[1, 0].axvline(x=0.4, color='black', linestyle='--',
                    linewidth=2, label='Uniform τ=0.4')
axes[1, 0].axvline(x=thresholds_A_opt, color='steelblue',
                    linestyle=':', linewidth=2,
                    label=f'Optimal τ_A={thresholds_A_opt:.3f}')
axes[1, 0].axvline(x=thresholds_B_opt, color='coral',
                    linestyle=':', linewidth=2,
                    label=f'Optimal τ_B={thresholds_B_opt:.3f}')
axes[1, 0].set_xlabel('Threshold τ')
axes[1, 0].set_ylabel('True Positive Rate (TPR)')
axes[1, 0].set_title('TPR by Group Across All Thresholds')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: TPR gap across thresholds
tpr_gap = np.abs(np.array(tpr_B_range) - np.array(tpr_A_range))
axes[1, 1].plot(thresholds_range, tpr_gap,
                color='purple', linewidth=2,
                label='TPR Gap |TPR_B - TPR_A|')
axes[1, 1].axvline(x=0.4, color='black', linestyle='--',
                    linewidth=2,
                    label=f'Uniform τ=0.4 (gap={tpr_gap[np.argmin(np.abs(thresholds_range-0.4))]:.3f})')
axes[1, 1].axvline(x=thresholds_A_opt, color='steelblue',
                    linestyle=':', linewidth=2,
                    label=f'Optimal τ_A={thresholds_A_opt:.3f}')
axes[1, 1].scatter([thresholds_A_opt],
                    [tpr_gap[np.argmin(np.abs(
                        thresholds_range - thresholds_A_opt))]],
                    color='green', s=100, zorder=5,
                    label='Minimum gap point')
axes[1, 1].set_xlabel('Threshold τ')
axes[1, 1].set_ylabel('TPR Gap')
axes[1, 1].set_title('Equalised Odds Violation:\n'
                      'TPR Gap Across All Thresholds')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle(f'Random Forest ResultsThreshold Optimisation: Bayesian Loss Framework\n{FIGURE_TITLE_SUFFIX}',
             fontsize=12, fontweight='bold')

save_plot('threshold_optimisation.png')
plt.show()

# ─────────────────────────────────────────
# 7. FINAL SUMMARY
# ─────────────────────────────────────────

print()
print("=" * 65)
print("THRESHOLD OPTIMISATION SUMMARY")
print("=" * 65)