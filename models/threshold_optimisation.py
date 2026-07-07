import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. LOAD DATA AND TRAIN MODEL
# ─────────────────────────────────────────

df = pd.read_csv('data/synthetic_nhs_data.csv')
# df['group_binary'] = (df['group'] == 'B').astype(int)

# FEATURES = ['risk_score', 'referral_length',
#             'previous_contacts', 'age', 'group_binary']

FEATURES = ['risk_score', 'referral_length',
            'previous_contacts', 'age' ]

TARGET = 'true_outcome'

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# rf = RandomForestClassifier(
#     n_estimators=100, max_depth=5,
#     random_state=42, class_weight='balanced'
# )
rf = RandomForestClassifier(
    n_estimators=100,
    max_features=1,
    max_samples=0.6,
    bootstrap=True,
    max_depth=3,
    random_state=42,
    class_weight='balanced'
)
rf.fit(X_train, y_train)

y_prob = rf.predict_proba(X_test)[:, 1]

df_test = X_test.copy()
df_test['true_outcome'] = y_test.values
df_test['predicted_prob'] = y_prob
df_test['group'] = df.loc[X_test.index, 'group'].values

df_A = df_test[df_test['group'] == 'A']
df_B = df_test[df_test['group'] == 'B']

print("Model trained successfully")
print(f"Group A test patients: {len(df_A)}")
print(f"Group B test patients: {len(df_B)}")
print()

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

# Get ROC curves
fpr_A, tpr_A, thresholds_A = roc_curve(
    df_A['true_outcome'], df_A['predicted_prob']
)
fpr_B, tpr_B, thresholds_B = roc_curve(
    df_B['true_outcome'], df_B['predicted_prob']
)

P_A = df_A['true_outcome'].mean()
P_B = df_B['true_outcome'].mean()

def expected_loss(fpr, tpr, thresholds, c_fn, c_fp, base_rate):
    """Compute expected loss at each threshold."""
    fnr = 1 - tpr
    return c_fn * fnr * base_rate + c_fp * fpr * (1 - base_rate)

def optimal_threshold(fpr, tpr, thresholds, c_fn, c_fp, base_rate):
    """Find threshold minimising expected loss."""
    losses = expected_loss(fpr, tpr, thresholds, c_fn, c_fp, base_rate)
    idx = np.argmin(losses)
    return thresholds[idx], losses[idx], tpr[idx], fpr[idx]

# Test different cost ratios
cost_ratios = [1, 2, 3, 5, 10, 20]
C_FP = 1.0

print(f"\n{'Cost Ratio':<12} {'τ_A (optimal)':<16} "
      f"{'τ_B (optimal)':<16} {'Difference':<12}")
print("-" * 58)

for ratio in cost_ratios:
    C_FN = ratio * C_FP
    tau_A, _, tpr_a, fpr_a = optimal_threshold(
        fpr_A, tpr_A, thresholds_A, C_FN, C_FP, P_A
    )
    tau_B, _, tpr_b, fpr_b = optimal_threshold(
        fpr_B, tpr_B, thresholds_B, C_FN, C_FP, P_B
    )
    print(f"c_FN/c_FP={ratio:<3} {tau_A:<16.3f} "
          f"{tau_B:<16.3f} {tau_B-tau_A:<12.3f}")

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

tau_A_opt, loss_A, tpr_A_opt, fpr_A_opt = optimal_threshold(
    fpr_A, tpr_A, thresholds_A, C_FN, C_FP, P_A
)
tau_B_opt, loss_B, tpr_B_opt, fpr_B_opt = optimal_threshold(
    fpr_B, tpr_B, thresholds_B, C_FN, C_FP, P_B
)

print("=" * 65)
print(f"SELECTED COST PARAMETERS: c_FN={C_FN}, c_FP={C_FP}")
print("=" * 65)
print(f"""
Group A:
  Base rate P_A:          {P_A:.3f}
  Optimal threshold τ_A:  {tau_A_opt:.3f}
  TPR at τ_A:             {tpr_A_opt:.3f}
  FPR at τ_A:             {fpr_A_opt:.3f}
  Minimum expected loss:  {loss_A:.4f}

Group B:
  Base rate P_B:          {P_B:.3f}
  Optimal threshold τ_B:  {tau_B_opt:.3f}
  TPR at τ_B:             {tpr_B_opt:.3f}
  FPR at τ_B:             {fpr_B_opt:.3f}
  Minimum expected loss:  {loss_B:.4f}

Uniform threshold τ=0.4 for comparison:
  τ_A - τ_uniform:        {tau_A_opt - 0.4:.3f}
  τ_B - τ_uniform:        {tau_B_opt - 0.4:.3f}
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
    tpr_a, fpr_a, fnr_a = metrics_at_threshold(df_A, tau)
    tpr_b, fpr_b, fnr_b = metrics_at_threshold(df_B, tau)

    loss_a = C_FN * fnr_a * P_A + C_FP * fpr_a * (1 - P_A)
    loss_b = C_FN * fnr_b * P_B + C_FP * fpr_b * (1 - P_B)

    losses_A_range.append(loss_a)
    losses_B_range.append(loss_b)
    tpr_A_range.append(tpr_a)
    tpr_B_range.append(tpr_b)
    fpr_A_range.append(fpr_a)
    fpr_B_range.append(fpr_b)

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
axes[0, 0].axvline(x=tau_A_opt, color='steelblue',
                    linestyle=':', linewidth=2,
                    label=f'Optimal τ_A={tau_A_opt:.3f}')
axes[0, 0].axvline(x=tau_B_opt, color='coral',
                    linestyle=':', linewidth=2,
                    label=f'Optimal τ_B={tau_B_opt:.3f}')
axes[0, 0].scatter([tau_A_opt], [min(losses_A_range)],
                    color='steelblue', s=100, zorder=5)
axes[0, 0].scatter([tau_B_opt], [min(losses_B_range)],
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
    t_A, _, _, _ = optimal_threshold(
        fpr_A, tpr_A, thresholds_A, ratio, C_FP, P_A
    )
    t_B, _, _, _ = optimal_threshold(
        fpr_B, tpr_B, thresholds_B, ratio, C_FP, P_B
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
axes[1, 0].axvline(x=tau_A_opt, color='steelblue',
                    linestyle=':', linewidth=2,
                    label=f'Optimal τ_A={tau_A_opt:.3f}')
axes[1, 0].axvline(x=tau_B_opt, color='coral',
                    linestyle=':', linewidth=2,
                    label=f'Optimal τ_B={tau_B_opt:.3f}')
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
axes[1, 1].axvline(x=tau_A_opt, color='steelblue',
                    linestyle=':', linewidth=2,
                    label=f'Optimal τ_A={tau_A_opt:.3f}')
axes[1, 1].scatter([tau_A_opt],
                    [tpr_gap[np.argmin(np.abs(
                        thresholds_range - tau_A_opt))]],
                    color='green', s=100, zorder=5,
                    label='Minimum gap point')
axes[1, 1].set_xlabel('Threshold τ')
axes[1, 1].set_ylabel('TPR Gap')
axes[1, 1].set_title('Equalised Odds Violation:\n'
                      'TPR Gap Across All Thresholds')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Threshold Optimisation: Bayesian Loss Framework\n'
             'From Probabilities to Decisions',
             fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('visualisations/outputs/threshold_optimisation.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to visualisations/outputs/threshold_optimisation.png")

# ─────────────────────────────────────────
# 7. FINAL SUMMARY
# ─────────────────────────────────────────

print()
print("=" * 65)
print("THRESHOLD OPTIMISATION SUMMARY")
print("=" * 65)