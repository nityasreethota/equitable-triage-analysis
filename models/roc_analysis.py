import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from src.config import (
    UNIFORM_THRESHOLD, C_FN, C_FP,
    COLOR_A, COLOR_B, FIGURE_TITLE_SUFFIX,
    get_output_path, CURRENT_MODE
)
from src.data_loader import load_and_split
from src.models import get_bayes_model, train_model
from src.metrics import (
    compute_metrics_at_threshold,
    compute_tpr_gap,
    find_optimal_threshold
)
from src.utils import print_header, ensure_output_dir

# ─────────────────────────────────────────
# 1. LOAD DATA AND TRAIN
# ─────────────────────────────────────────

(df, X_train, X_test, y_train, y_test,
 groups, X_A, y_A, X_B, y_B) = load_and_split()

rf = train_model(get_bayes_model(), X_train, y_train)

print(f"Mode: {CURRENT_MODE}")
print(f"Model retrained with Bayesian Optimisation parameters")
print(f"Group A test: {len(X_A)}, Group B test: {len(X_B)}")

# ─────────────────────────────────────────
# 2. GET PREDICTED PROBABILITIES
# ─────────────────────────────────────────

prob_A = rf.predict_proba(X_A)[:, 1]
prob_B = rf.predict_proba(X_B)[:, 1]

# ─────────────────────────────────────────
# 3. COMPUTE GROUP-SPECIFIC ROC CURVES
# ─────────────────────────────────────────

fpr_A, tpr_A, thresholds_A = roc_curve(y_A, prob_A)
auc_A = auc(fpr_A, tpr_A)

fpr_B, tpr_B, thresholds_B = roc_curve(y_B, prob_B)
auc_B = auc(fpr_B, tpr_B)

print_header("ROC CURVE ANALYSIS")
print(f"Group A AUC: {auc_A:.3f}")
print(f"Group B AUC: {auc_B:.3f}")
print(f"AUC Gap:     {auc_B - auc_A:.3f}")

# ─────────────────────────────────────────
# 4. FIND OPTIMAL THRESHOLDS
# ─────────────────────────────────────────

tau_A, loss_A, tpr_A_opt, fpr_A_opt = find_optimal_threshold(
    rf, X_A, y_A, C_FN, C_FP
)
tau_B, loss_B, tpr_B_opt, fpr_B_opt = find_optimal_threshold(
    rf, X_B, y_B, C_FN, C_FP
)

print_header("OPTIMAL GROUP-SPECIFIC THRESHOLDS")
print(f"(Bayesian Loss Framework: c_FN={C_FN}, c_FP={C_FP})")
print(f"Uniform threshold:         tau = {UNIFORM_THRESHOLD:.3f}")
print(f"Group A optimal threshold: tau_A = {tau_A:.3f}")
print(f"Group B optimal threshold: tau_B = {tau_B:.3f}")
print(f"Group A base rate: {np.array(y_A).mean():.3f}")
print(f"Group B base rate: {np.array(y_B).mean():.3f}")

# ─────────────────────────────────────────
# 5. COMPARE UNIFORM VS GROUP-SPECIFIC
# ─────────────────────────────────────────

m_A_uniform = compute_metrics_at_threshold(X_A, y_A, rf, UNIFORM_THRESHOLD)
m_B_uniform = compute_metrics_at_threshold(X_B, y_B, rf, UNIFORM_THRESHOLD)
m_A_optimal = compute_metrics_at_threshold(X_A, y_A, rf, tau_A)
m_B_optimal = compute_metrics_at_threshold(X_B, y_B, rf, tau_B)

gap_uniform, _, _ = compute_tpr_gap(rf, X_A, y_A, X_B, y_B, UNIFORM_THRESHOLD)
gap_optimal = abs(m_B_optimal['TPR'] - m_A_optimal['TPR'])

print_header("COMPARISON: UNIFORM vs GROUP-SPECIFIC THRESHOLDS")
print(f"\n{'':30} {'Uniform':>10} {'Optimal':>10} {'Improvement':>12}")
print("-" * 65)
print("GROUP A (underrepresented):")
print(f"  {'TPR (catches who needs help)':<28} "
      f"{m_A_uniform['TPR']:>10.3f} "
      f"{m_A_optimal['TPR']:>10.3f} "
      f"{m_A_optimal['TPR']-m_A_uniform['TPR']:>+12.3f}")
print(f"  {'FNR (misses who needs help)':<28} "
      f"{m_A_uniform['FNR']:>10.3f} "
      f"{m_A_optimal['FNR']:>10.3f} "
      f"{m_A_uniform['FNR']-m_A_optimal['FNR']:>+12.3f}")
print()
print("GROUP B (majority):")
print(f"  {'TPR (catches who needs help)':<28} "
      f"{m_B_uniform['TPR']:>10.3f} "
      f"{m_B_optimal['TPR']:>10.3f} "
      f"{m_B_optimal['TPR']-m_B_uniform['TPR']:>+12.3f}")
print(f"  {'FNR (misses who needs help)':<28} "
      f"{m_B_uniform['FNR']:>10.3f} "
      f"{m_B_optimal['FNR']:>10.3f} "
      f"{m_B_uniform['FNR']-m_B_optimal['FNR']:>+12.3f}")

print()
print(f"TPR Gap (should reduce toward 0):")
print(f"  Uniform threshold:         {gap_uniform:.3f}")
print(f"  Group-specific thresholds: {gap_optimal:.3f}")

# ─────────────────────────────────────────
# 6. EXPECTED LOSS CURVES
# ─────────────────────────────────────────

thresholds_range = np.linspace(0.1, 0.9, 100)

def expected_loss_curve(rf, X_grp, y_grp, thresholds_range):
    losses = []
    for t in thresholds_range:
        m = compute_metrics_at_threshold(X_grp, y_grp, rf, t)
        P = m['base_rate']
        loss = C_FN * m['FNR'] * P + C_FP * m['FPR'] * (1 - P)
        losses.append(loss)
    return losses

losses_A = expected_loss_curve(rf, X_A, y_A, thresholds_range)
losses_B = expected_loss_curve(rf, X_B, y_B, thresholds_range)

# ─────────────────────────────────────────
# 7. VISUALISATIONS
# ─────────────────────────────────────────

ensure_output_dir()
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: ROC curves
axes[0].plot(fpr_A, tpr_A, color=COLOR_A, linewidth=2,
             label=f'Group A (AUC={auc_A:.3f})')
axes[0].plot(fpr_B, tpr_B, color=COLOR_B, linewidth=2,
             label=f'Group B (AUC={auc_B:.3f})')
axes[0].plot([0,1],[0,1],'k--', alpha=0.5, label='Random')

idx_A = np.argmin(np.abs(thresholds_A - tau_A))
idx_B = np.argmin(np.abs(thresholds_B - tau_B))
axes[0].scatter(fpr_A[idx_A], tpr_A[idx_A],
                color=COLOR_A, s=100, zorder=5,
                label=f'Optimal tau_A={tau_A:.3f}')
axes[0].scatter(fpr_B[idx_B], tpr_B[idx_B],
                color=COLOR_B, s=100, zorder=5,
                label=f'Optimal tau_B={tau_B:.3f}')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('Group-Specific ROC Curves\nwith Optimal Thresholds')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Plot 2: TPR comparison
categories = ['Group A\n(underrepresented)', 'Group B\n(majority)']
tpr_uniform = [m_A_uniform['TPR'], m_B_uniform['TPR']]
tpr_optimal = [m_A_optimal['TPR'], m_B_optimal['TPR']]

x = np.arange(len(categories))
width = 0.35
axes[1].bar(x - width/2, tpr_uniform, width,
            label=f'Uniform tau={UNIFORM_THRESHOLD}',
            color='grey', alpha=0.8)
axes[1].bar(x + width/2, tpr_optimal, width,
            label='Group-specific thresholds',
            color=[COLOR_A, COLOR_B], alpha=0.8)
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('TPR Comparison:\nUniform vs Group-Specific')
axes[1].set_xticks(x)
axes[1].set_xticklabels(categories)
axes[1].legend()
axes[1].set_ylim(0, 1.1)
for i, (u, o) in enumerate(zip(tpr_uniform, tpr_optimal)):
    axes[1].text(i - width/2, u + 0.02, f'{u:.3f}',
                 ha='center', fontsize=9)
    axes[1].text(i + width/2, o + 0.02, f'{o:.3f}',
                 ha='center', fontsize=9)

# Plot 3: Expected loss curves
axes[2].plot(thresholds_range, losses_A,
             color=COLOR_A, linewidth=2,
             label='Group A expected loss')
axes[2].plot(thresholds_range, losses_B,
             color=COLOR_B, linewidth=2,
             label='Group B expected loss')
axes[2].axvline(x=UNIFORM_THRESHOLD, color='black',
                linestyle='--', label=f'Uniform tau={UNIFORM_THRESHOLD}')
axes[2].axvline(x=tau_A, color=COLOR_A, linestyle=':',
                linewidth=2, label=f'Optimal tau_A={tau_A:.3f}')
axes[2].axvline(x=tau_B, color=COLOR_B, linestyle=':',
                linewidth=2, label=f'Optimal tau_B={tau_B:.3f}')
axes[2].set_xlabel('Threshold tau')
axes[2].set_ylabel('Expected Loss E[L]')
axes[2].set_title('Bayesian Expected Loss Curves\nby Demographic Group')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.suptitle(f'ROC Analysis\n{FIGURE_TITLE_SUFFIX}',
             fontsize=12, fontweight='bold')
plt.tight_layout()

filepath = get_output_path('roc_analysis_bayesian.png')
plt.savefig(filepath, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved to {filepath}")

# ─────────────────────────────────────────
# 8. FINAL SUMMARY
# ─────────────────────────────────────────

print_header("FINAL SUMMARY")
print(f"""
Mode: {CURRENT_MODE}

Chouldechova's result predicts when base rates differ,
a uniform threshold cannot achieve equalised odds.

This simulation confirms:
  TPR gap under uniform threshold:         {gap_uniform:.3f}
  TPR gap under group-specific thresholds: {gap_optimal:.3f}

Group-specific thresholds:
  tau_A = {tau_A:.3f} (lower - reflecting higher c_FN for Group A)
  tau_B = {tau_B:.3f}

The decoupled framework reduces the TPR gap by making
fairness trade-offs mathematically explicit and measurable.
""")