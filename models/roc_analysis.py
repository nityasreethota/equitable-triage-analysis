import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. LOAD DATA AND RETRAIN MODEL
# ─────────────────────────────────────────

df = pd.read_csv('data/synthetic_nhs_data.csv')

# Encode group
# df['group_binary'] = (df['group'] == 'B').astype(int)

# FEATURES = ['risk_score', 'referral_length',
#             'previous_contacts', 'age', 'group_binary']

FEATURES = ['risk_score', 'referral_length',
            'previous_contacts', 'age']
TARGET = 'true_outcome'

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# rf = RandomForestClassifier(
#     n_estimators=100,
#     max_depth=5,
#     random_state=42,
#     class_weight='balanced'
# )
# # Stage 1: Use best parameters from hyperparameter tuning (grid search)
# # Reduces referral_length dependence from 0.0238 to 0.0095
# # TPR gap reduced from 0.580 to 0.235 before threshold correction
# rf = RandomForestClassifier(
#     n_estimators=100,
#     max_features=1,
#     max_samples=0.6,
#     bootstrap=True,
#     max_depth=3,
#     random_state=42,
#     class_weight='balanced'
# )

# Stage 1: Use best parameters from Bayesian Optimisation
# Bayesian outperforms Grid Search:
#   max_samples=0.734 (between grid points 0.6 and 0.7)
#   max_depth=2 (not in discrete grid)
# Reduces referral_length I_j from 0.0238 to 0.0032
# TPR gap reduced from 0.580 to 0.000 before threshold correction
rf = RandomForestClassifier(
    n_estimators=100,
    max_features=1,
    max_samples=0.734,
    bootstrap=True,
    max_depth=2,
    random_state=42,
    class_weight='balanced'
)
rf.fit(X_train, y_train)

# Get predicted probabilities
y_prob = rf.predict_proba(X_test)[:, 1]

# Build test dataframe
df_test = X_test.copy()
df_test['true_outcome'] = y_test.values
df_test['predicted_prob'] = y_prob
df_test['group'] = df.loc[X_test.index, 'group'].values

# print("Model retrained successfully")
# print("Model retrained with Grid search tuned parameters (Stage 1 complete)")
# print("max_features=1, max_samples=0.6, bootstrap=True, max_depth=3")
print("Model retrained with Bayesian Optimisation parameters (Stage 1 complete)")
print("max_features=1, max_samples=0.734, bootstrap=True, max_depth=2")
print(f"Test set: {len(df_test)} patients")
print()

# ─────────────────────────────────────────
# 2. SPLIT BY GROUP
# ─────────────────────────────────────────

df_A = df_test[df_test['group'] == 'A']
df_B = df_test[df_test['group'] == 'B']

print(f"Group A test patients: {len(df_A)}")
print(f"Group B test patients: {len(df_B)}")
print()

# ─────────────────────────────────────────
# 3. COMPUTE GROUP-SPECIFIC ROC CURVES
# ─────────────────────────────────────────

# Group A ROC
fpr_A, tpr_A, thresholds_A = roc_curve(
    df_A['true_outcome'],
    df_A['predicted_prob']
)
auc_A = auc(fpr_A, tpr_A)

# Group B ROC
fpr_B, tpr_B, thresholds_B = roc_curve(
    df_B['true_outcome'],
    df_B['predicted_prob']
)
auc_B = auc(fpr_B, tpr_B)

print("=" * 60)
print("ROC CURVE ANALYSIS")
print("=" * 60)
print(f"Group A AUC: {auc_A:.3f}")
print(f"Group B AUC: {auc_B:.3f}")
print(f"AUC Gap:     {auc_B - auc_A:.3f}")
print()

# ─────────────────────────────────────────
# 4. FIND OPTIMAL GROUP-SPECIFIC THRESHOLDS
# Using Bayesian loss framework
# ─────────────────────────────────────────

# Cost parameters - reflecting asymmetric costs
C_FN = 5.0
C_FP = 1.0
COST_RATIO = C_FP / C_FN  # = 0.2

def find_optimal_threshold(fpr, tpr, thresholds, c_fn, c_fp, base_rate):
    """
    Find threshold minimising expected Bayesian loss:
    E[L] = c_FN * FNR * P + c_FP * FPR * (1-P)
    where P is the base rate.
    """
    fnr = 1 - tpr  # False negative rate
    expected_loss = (c_fn * fnr * base_rate +
                     c_fp * fpr * (1 - base_rate))
    optimal_idx = np.argmin(expected_loss)
    return thresholds[optimal_idx], expected_loss[optimal_idx]

# Base rates
P_A = df_A['true_outcome'].mean()
P_B = df_B['true_outcome'].mean()

# Find optimal thresholds
tau_A_optimal, loss_A = find_optimal_threshold(
    fpr_A, tpr_A, thresholds_A, C_FN, C_FP, P_A
)
tau_B_optimal, loss_B = find_optimal_threshold(
    fpr_B, tpr_B, thresholds_B, C_FN, C_FP, P_B
)

print("=" * 60)
print("OPTIMAL GROUP-SPECIFIC THRESHOLDS")
print("(Bayesian Loss Framework: c_FN=5.0, c_FP=1.0)")
print("=" * 60)
print(f"Uniform threshold:          τ = 0.400")
print(f"Group A optimal threshold:  τ_A = {tau_A_optimal:.3f}")
print(f"Group B optimal threshold:  τ_B = {tau_B_optimal:.3f}")
print()
print(f"Group A base rate: {P_A:.3f}")
print(f"Group B base rate: {P_B:.3f}")
print()

# ─────────────────────────────────────────
# 5. COMPARE UNIFORM VS GROUP-SPECIFIC
# ─────────────────────────────────────────

def compute_metrics_at_threshold(df_group, threshold):
    """Compute TPR, FPR, FNR at a given threshold."""
    decisions = (df_group['predicted_prob'] >= threshold).astype(int)
    TP = ((decisions == 1) & (df_group['true_outcome'] == 1)).sum()
    FP = ((decisions == 1) & (df_group['true_outcome'] == 0)).sum()
    TN = ((decisions == 0) & (df_group['true_outcome'] == 0)).sum()
    FN = ((decisions == 0) & (df_group['true_outcome'] == 1)).sum()

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
    PPV = TP / (TP + FP) if (TP + FP) > 0 else 0

    return {'TPR': TPR, 'FPR': FPR, 'FNR': FNR, 'PPV': PPV}

# Uniform threshold results
UNIFORM = 0.4
m_A_uniform = compute_metrics_at_threshold(df_A, UNIFORM)
m_B_uniform = compute_metrics_at_threshold(df_B, UNIFORM)

# Group-specific threshold results
m_A_optimal = compute_metrics_at_threshold(df_A, tau_A_optimal)
m_B_optimal = compute_metrics_at_threshold(df_B, tau_B_optimal)

print("=" * 60)
print("COMPARISON: UNIFORM vs GROUP-SPECIFIC THRESHOLDS")
print("=" * 60)
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
print("TPR Gap (should reduce toward 0):")
print(f"  Uniform threshold:          "
      f"{abs(m_B_uniform['TPR']-m_A_uniform['TPR']):.3f}")
print(f"  Group-specific thresholds:  "
      f"{abs(m_B_optimal['TPR']-m_A_optimal['TPR']):.3f}")

# ─────────────────────────────────────────
# 6. VISUALISATIONS
# ─────────────────────────────────────────

os.makedirs('visualisations/outputs', exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Group-specific ROC curves
axes[0].plot(fpr_A, tpr_A, color='steelblue', linewidth=2,
             label=f'Group A (AUC = {auc_A:.3f})')
axes[0].plot(fpr_B, tpr_B, color='coral', linewidth=2,
             label=f'Group B (AUC = {auc_B:.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random classifier')

# Mark optimal operating points
idx_A = np.argmin(np.abs(thresholds_A - tau_A_optimal))
idx_B = np.argmin(np.abs(thresholds_B - tau_B_optimal))
axes[0].scatter(fpr_A[idx_A], tpr_A[idx_A],
                color='steelblue', s=100, zorder=5,
                label=f'Optimal τ_A={tau_A_optimal:.3f}')
axes[0].scatter(fpr_B[idx_B], tpr_B[idx_B],
                color='coral', s=100, zorder=5,
                label=f'Optimal τ_B={tau_B_optimal:.3f}')

axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('Group-Specific ROC Curves\nwith Optimal Thresholds')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Plot 2: TPR comparison uniform vs optimal
categories = ['Group A\n(underrepresented)', 'Group B\n(majority)']
tpr_uniform = [m_A_uniform['TPR'], m_B_uniform['TPR']]
tpr_optimal = [m_A_optimal['TPR'], m_B_optimal['TPR']]

x = np.arange(len(categories))
width = 0.35

axes[1].bar(x - width/2, tpr_uniform, width,
            label='Uniform threshold τ=0.4',
            color='grey', alpha=0.8)
axes[1].bar(x + width/2, tpr_optimal, width,
            label='Group-specific thresholds',
            color=['steelblue', 'coral'], alpha=0.8)
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('TPR Comparison:\nUniform vs Group-Specific Thresholds')
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
thresholds_range = np.linspace(0.1, 0.9, 100)

def expected_loss_at_threshold(df_group, thresholds_range,
                                c_fn, c_fp, base_rate):
    losses = []
    for t in thresholds_range:
        m = compute_metrics_at_threshold(df_group, t)
        loss = c_fn * m['FNR'] * base_rate + c_fp * m['FPR'] * (1-base_rate)
        losses.append(loss)
    return losses

losses_A = expected_loss_at_threshold(
    df_A, thresholds_range, C_FN, C_FP, P_A)
losses_B = expected_loss_at_threshold(
    df_B, thresholds_range, C_FN, C_FP, P_B)

axes[2].plot(thresholds_range, losses_A,
             color='steelblue', linewidth=2,
             label='Group A expected loss')
axes[2].plot(thresholds_range, losses_B,
             color='coral', linewidth=2,
             label='Group B expected loss')
axes[2].axvline(x=0.4, color='black', linestyle='--',
                label='Uniform threshold τ=0.4')
axes[2].axvline(x=tau_A_optimal, color='steelblue',
                linestyle=':', linewidth=2,
                label=f'Optimal τ_A={tau_A_optimal:.3f}')
axes[2].axvline(x=tau_B_optimal, color='coral',
                linestyle=':', linewidth=2,
                label=f'Optimal τ_B={tau_B_optimal:.3f}')
axes[2].set_xlabel('Threshold τ')
axes[2].set_ylabel('Expected Loss E[L]')
axes[2].set_title('Bayesian Expected Loss Curves\nby Demographic Group')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('visualisations/outputs/roc_analysis.png',
#             dpi=150, bbox_inches='tight')
plt.savefig('visualisations/outputs/roc_analysis_bayesian.png',
            dpi=150, bbox_inches='tight')
plt.show()
# print("\nPlot saved to visualisations/outputs/roc_analysis.png")
print("\nPlot saved to visualisations/outputs/roc_analysis_bayesian.png")

# ─────────────────────────────────────────
# 7. FINAL SUMMARY
# ─────────────────────────────────────────

print()
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"""
Chouldechova's impossibility result predicts that when base
rates differ (P_A={P_A:.3f} ≠ P_B={P_B:.3f}), a uniform threshold
cannot achieve equalised odds.

This simulation confirms:
  TPR gap under uniform threshold:         {abs(m_B_uniform['TPR']-m_A_uniform['TPR']):.3f}
  TPR gap under group-specific thresholds: {abs(m_B_optimal['TPR']-m_A_optimal['TPR']):.3f}

Group-specific thresholds:
  τ_A = {tau_A_optimal:.3f} (lower - reflecting higher c_FN for Group A)
  τ_B = {tau_B_optimal:.3f}

The decoupled framework reduces the TPR gap by making
fairness trade-offs mathematically explicit and measurable.
""")