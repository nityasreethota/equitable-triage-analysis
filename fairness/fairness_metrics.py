# Demographic parity, equalised odds
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. LOAD DATA AND RETRAIN MODEL
# ─────────────────────────────────────────

df = pd.read_csv('data/synthetic_nhs_data.csv')
# df['group_binary'] = (df['group'] == 'B').astype(int)

# FEATURES = ['risk_score', 'referral_length',
#             'previous_contacts', 'age', 'group_binary']

FEATURES = ['risk_score', 'referral_length',
            'previous_contacts', 'age']
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
# Using best parameters from hyperparameter tuning (Stage 1)
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

print("Data loaded and model trained successfully")
print()

# ─────────────────────────────────────────
# 2. FAIRNESS CRITERIA DEFINITIONS
# ─────────────────────────────────────────

def compute_all_metrics(df_group, threshold):
    """
    Compute all fairness-relevant metrics for a group
    at a given threshold.
    """
    decisions = (df_group['predicted_prob'] >= threshold).astype(int)
    T = df_group['true_outcome'].values
    D = decisions.values

    TP = ((D == 1) & (T == 1)).sum()
    FP = ((D == 1) & (T == 0)).sum()
    TN = ((D == 0) & (T == 0)).sum()
    FN = ((D == 0) & (T == 1)).sum()

    # Base rate
    P = T.mean()

    # Fairness metrics
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0

    # Predictive values
    PPV = TP / (TP + FP) if (TP + FP) > 0 else 0  # Precision
    NPV = TN / (TN + FN) if (TN + FN) > 0 else 0

    # Selection rate (demographic parity)
    selection_rate = D.mean()

    return {
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'TPR': TPR, 'FPR': FPR, 'FNR': FNR, 'TNR': TNR,
        'PPV': PPV, 'NPV': NPV,
        'selection_rate': selection_rate,
        'base_rate': P,
        'n': len(T)
    }

# ─────────────────────────────────────────
# 3. COMPUTE METRICS UNDER BOTH THRESHOLDS
# ─────────────────────────────────────────

UNIFORM_TAU = 0.4
TAU_A = 0.260
TAU_B = 0.378

# Under uniform threshold
m_A_uni = compute_all_metrics(df_A, UNIFORM_TAU)
m_B_uni = compute_all_metrics(df_B, UNIFORM_TAU)

# Under group-specific thresholds
m_A_opt = compute_all_metrics(df_A, TAU_A)
m_B_opt = compute_all_metrics(df_B, TAU_B)

# ─────────────────────────────────────────
# 4. EVALUATE THREE FAIRNESS CRITERIA
# ─────────────────────────────────────────

print("=" * 70)
print("FAIRNESS CRITERIA EVALUATION")
print("=" * 70)


print("-" * 70)
print("UNDER UNIFORM THRESHOLD (τ=0.4)")
print("-" * 70)

print(f"\n{'Criterion':<35} {'Group A':>10} {'Group B':>10} "
      f"{'Satisfied?':>12}")
print("-" * 70)

# Demographic parity
dp_A = m_A_uni['selection_rate']
dp_B = m_B_uni['selection_rate']
dp_satisfied = abs(dp_A - dp_B) < 0.05
print(f"{'1. Demographic Parity':<35} "
      f"{dp_A:>10.3f} {dp_B:>10.3f} "
      f"{'✓' if dp_satisfied else '✗ VIOLATED':>12}")

# Equalised odds - TPR
eo_tpr_A = m_A_uni['TPR']
eo_tpr_B = m_B_uni['TPR']
eo_tpr_satisfied = abs(eo_tpr_A - eo_tpr_B) < 0.05
print(f"{'2a. Equalised Odds (TPR)':<35} "
      f"{eo_tpr_A:>10.3f} {eo_tpr_B:>10.3f} "
      f"{'✓' if eo_tpr_satisfied else '✗ VIOLATED':>12}")

# Equalised odds - FPR
eo_fpr_A = m_A_uni['FPR']
eo_fpr_B = m_B_uni['FPR']
eo_fpr_satisfied = abs(eo_fpr_A - eo_fpr_B) < 0.05
print(f"{'2b. Equalised Odds (FPR)':<35} "
      f"{eo_fpr_A:>10.3f} {eo_fpr_B:>10.3f} "
      f"{'✓' if eo_fpr_satisfied else '✗ VIOLATED':>12}")

# Calibration
cal_A = m_A_uni['PPV']
cal_B = m_B_uni['PPV']
cal_satisfied = abs(cal_A - cal_B) < 0.05
print(f"{'3. Calibration (PPV)':<35} "
      f"{cal_A:>10.3f} {cal_B:>10.3f} "
      f"{'✓' if cal_satisfied else '✗ VIOLATED':>12}")

print(f"\nBase rates: P_A={m_A_uni['base_rate']:.3f}, "
      f"P_B={m_B_uni['base_rate']:.3f}")
print(f"Base rates differ: P_A ≠ P_B → "
      f"Chouldechova's result applies")

print()
print("-" * 70)
print("UNDER GROUP-SPECIFIC THRESHOLDS (τ_A=0.260, τ_B=0.378)")
print("-" * 70)

print(f"\n{'Criterion':<35} {'Group A':>10} {'Group B':>10} "
      f"{'Satisfied?':>12}")
print("-" * 70)

# Demographic parity
dp_A_opt = m_A_opt['selection_rate']
dp_B_opt = m_B_opt['selection_rate']
dp_sat_opt = abs(dp_A_opt - dp_B_opt) < 0.05
print(f"{'1. Demographic Parity':<35} "
      f"{dp_A_opt:>10.3f} {dp_B_opt:>10.3f} "
      f"{'✓' if dp_sat_opt else '✗ VIOLATED':>12}")

# Equalised odds TPR
eo_tpr_A_opt = m_A_opt['TPR']
eo_tpr_B_opt = m_B_opt['TPR']
eo_tpr_sat_opt = abs(eo_tpr_A_opt - eo_tpr_B_opt) < 0.05
print(f"{'2a. Equalised Odds (TPR)':<35} "
      f"{eo_tpr_A_opt:>10.3f} {eo_tpr_B_opt:>10.3f} "
      f"{'✓' if eo_tpr_sat_opt else '✗ VIOLATED':>12}")

# Equalised odds FPR
eo_fpr_A_opt = m_A_opt['FPR']
eo_fpr_B_opt = m_B_opt['FPR']
eo_fpr_sat_opt = abs(eo_fpr_A_opt - eo_fpr_B_opt) < 0.05
print(f"{'2b. Equalised Odds (FPR)':<35} "
      f"{eo_fpr_A_opt:>10.3f} {eo_fpr_B_opt:>10.3f} "
      f"{'✓' if eo_fpr_sat_opt else '✗ VIOLATED':>12}")

# Calibration
cal_A_opt = m_A_opt['PPV']
cal_B_opt = m_B_opt['PPV']
cal_sat_opt = abs(cal_A_opt - cal_B_opt) < 0.05
print(f"{'3. Calibration (PPV)':<35} "
      f"{cal_A_opt:>10.3f} {cal_B_opt:>10.3f} "
      f"{'✓' if cal_sat_opt else '✗ VIOLATED':>12}")

# ─────────────────────────────────────────
# 5. CHOULDECHOVA IMPOSSIBILITY DEMONSTRATION
# ─────────────────────────────────────────

print()
print("=" * 70)
print("CHOULDECHOVA IMPOSSIBILITY RESULT - NUMERICAL DEMONSTRATION")
print("=" * 70)

P_A = m_A_uni['base_rate']
P_B = m_B_uni['base_rate']
FNR = 0.1  # hypothetical equal FNR
FPR = 0.1  # hypothetical equal FPR

PPV_A_theoretical = ((1 - FNR) * P_A /
                     ((1 - FNR) * P_A + FPR * (1 - P_A)))
PPV_B_theoretical = ((1 - FNR) * P_B /
                     ((1 - FNR) * P_B + FPR * (1 - P_B)))

print(f"""
Chouldechova's formula:
PPV = (1-FNR)·P / [(1-FNR)·P + FPR·(1-P)]

Assume equalised odds satisfied: FNR_A = FNR_B = {FNR}
                                  FPR_A = FPR_B = {FPR}

Base rates: P_A = {P_A:.3f}, P_B = {P_B:.3f}

PPV_A = (1-{FNR})×{P_A:.3f} / [(1-{FNR})×{P_A:.3f} + {FPR}×{1-P_A:.3f}]
      = {PPV_A_theoretical:.3f}

PPV_B = (1-{FNR})×{P_B:.3f} / [(1-{FNR})×{P_B:.3f} + {FPR}×{1-P_B:.3f}]
      = {PPV_B_theoretical:.3f}

PPV_A ({PPV_A_theoretical:.3f}) ≠ PPV_B ({PPV_B_theoretical:.3f})

∴ When equalised odds is satisfied, calibration is violated.
  When P_A ≠ P_B, all three criteria CANNOT be satisfied simultaneously.
  This confirms Chouldechova's (2017) impossibility result.
""")

# ─────────────────────────────────────────
# 6. VISUALISATIONS
# ─────────────────────────────────────────

os.makedirs('visualisations/outputs', exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Fairness criteria comparison - uniform threshold
criteria = ['Demographic\nParity', 'Equalised\nOdds (TPR)',
            'Equalised\nOdds (FPR)', 'Calibration\n(PPV)']
vals_A = [m_A_uni['selection_rate'], m_A_uni['TPR'],
          m_A_uni['FPR'], m_A_uni['PPV']]
vals_B = [m_B_uni['selection_rate'], m_B_uni['TPR'],
          m_B_uni['FPR'], m_B_uni['PPV']]

x = np.arange(len(criteria))
width = 0.35
axes[0].bar(x - width/2, vals_A, width,
            label='Group A', color='steelblue', alpha=0.8)
axes[0].bar(x + width/2, vals_B, width,
            label='Group B', color='coral', alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(criteria, fontsize=9)
axes[0].set_ylabel('Value')
axes[0].set_title('Fairness Criteria\nUniform Threshold τ=0.4')
axes[0].legend()
axes[0].set_ylim(0, 1.2)
for i in range(len(criteria)):
    axes[0].text(i - width/2, vals_A[i] + 0.02,
                 f'{vals_A[i]:.2f}', ha='center', fontsize=8)
    axes[0].text(i + width/2, vals_B[i] + 0.02,
                 f'{vals_B[i]:.2f}', ha='center', fontsize=8)

# Plot 2: Fairness criteria comparison - group specific
vals_A_opt = [m_A_opt['selection_rate'], m_A_opt['TPR'],
              m_A_opt['FPR'], m_A_opt['PPV']]
vals_B_opt = [m_B_opt['selection_rate'], m_B_opt['TPR'],
              m_B_opt['FPR'], m_B_opt['PPV']]

axes[1].bar(x - width/2, vals_A_opt, width,
            label='Group A', color='steelblue', alpha=0.8)
axes[1].bar(x + width/2, vals_B_opt, width,
            label='Group B', color='coral', alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(criteria, fontsize=9)
axes[1].set_ylabel('Value')
axes[1].set_title('Fairness Criteria\nGroup-Specific Thresholds')
axes[1].legend()
axes[1].set_ylim(0, 1.2)
for i in range(len(criteria)):
    axes[1].text(i - width/2, vals_A_opt[i] + 0.02,
                 f'{vals_A_opt[i]:.2f}', ha='center', fontsize=8)
    axes[1].text(i + width/2, vals_B_opt[i] + 0.02,
                 f'{vals_B_opt[i]:.2f}', ha='center', fontsize=8)

# Plot 3: Chouldechova impossibility visualisation
base_rates = np.linspace(0.1, 0.9, 100)
FNR_fixed = 0.1
FPR_fixed = 0.1

ppv_values = ((1 - FNR_fixed) * base_rates /
              ((1 - FNR_fixed) * base_rates +
               FPR_fixed * (1 - base_rates)))

axes[2].plot(base_rates, ppv_values,
             color='purple', linewidth=2)
axes[2].axvline(x=P_A, color='steelblue', linestyle='--',
                linewidth=2, label=f'P_A = {P_A:.3f}')
axes[2].axvline(x=P_B, color='coral', linestyle='--',
                linewidth=2, label=f'P_B = {P_B:.3f}')
axes[2].scatter([P_A], [PPV_A_theoretical],
                color='steelblue', s=100, zorder=5,
                label=f'PPV_A = {PPV_A_theoretical:.3f}')
axes[2].scatter([P_B], [PPV_B_theoretical],
                color='coral', s=100, zorder=5,
                label=f'PPV_B = {PPV_B_theoretical:.3f}')
axes[2].set_xlabel('Base Rate P(T=1|G)')
axes[2].set_ylabel('PPV (Calibration)')
axes[2].set_title("Chouldechova's Impossibility:\nPPV varies with Base Rate\n"
                  "(even when FNR=FPR equal across groups)")
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualisations/outputs/fairness_metrics.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to visualisations/outputs/fairness_metrics.png")