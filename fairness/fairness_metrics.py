import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

from src.config import (
    UNIFORM_THRESHOLD, COLOR_A, COLOR_B,
    FIGURE_TITLE_SUFFIX, get_output_path, CURRENT_MODE
)
from src.data_loader import load_and_split
from src.models import get_bayes_model, train_model
from src.metrics import (
    compute_metrics_at_threshold,
    find_optimal_threshold
)
from src.utils import print_header, ensure_output_dir

# ─────────────────────────────────────────
# 1. LOAD DATA AND TRAIN
# ─────────────────────────────────────────

(df, X_train, X_test, y_train, y_test,
 groups, X_A, y_A, X_B, y_B) = load_and_split()

rf = train_model(get_bayes_model(), X_train, y_train)
print("Data loaded and model trained successfully")

# Compute optimal thresholds dynamically
TAU_A, _, _, _ = find_optimal_threshold(rf, X_A, y_A)
TAU_B, _, _, _ = find_optimal_threshold(rf, X_B, y_B)
print(f"Optimal thresholds: tau_A={TAU_A:.3f}, tau_B={TAU_B:.3f}")

# ─────────────────────────────────────────
# 2. COMPUTE METRICS UNDER BOTH THRESHOLDS
# ─────────────────────────────────────────

# Under uniform threshold
m_A_uni = compute_metrics_at_threshold(X_A, y_A, rf, UNIFORM_THRESHOLD)
m_B_uni = compute_metrics_at_threshold(X_B, y_B, rf, UNIFORM_THRESHOLD)

# Under group-specific thresholds
m_A_opt = compute_metrics_at_threshold(X_A, y_A, rf, TAU_A)
m_B_opt = compute_metrics_at_threshold(X_B, y_B, rf, TAU_B)

P_A = m_A_uni['base_rate']
P_B = m_B_uni['base_rate']

# ─────────────────────────────────────────
# 3. EVALUATE FAIRNESS CRITERIA
# ─────────────────────────────────────────

print_header("FAIRNESS CRITERIA EVALUATION")
print(f"Mode: {CURRENT_MODE}")
print("""
Three fairness criteria evaluated:
1. DEMOGRAPHIC PARITY: P(D=1|G=A) = P(D=1|G=B)
2. EQUALISED ODDS:     Equal TPR AND FPR across groups
3. CALIBRATION:        Equal PPV across groups
""")

def check(a, b, tol=0.05):
    return "SATISFIED" if abs(a-b) < tol else "VIOLATED"

print(f"\n{'─'*65}")
print(f"UNDER UNIFORM THRESHOLD (tau={UNIFORM_THRESHOLD})")
print(f"{'─'*65}")
print(f"\n{'Criterion':<35} {'Group A':>10} {'Group B':>10} {'Status':>12}")
print(f"{'─'*65}")
print(f"{'1. Demographic Parity':<35} "
      f"{m_A_uni['selection_rate']:>10.3f} "
      f"{m_B_uni['selection_rate']:>10.3f} "
      f"{check(m_A_uni['selection_rate'], m_B_uni['selection_rate']):>12}")
print(f"{'2a. Equalised Odds (TPR)':<35} "
      f"{m_A_uni['TPR']:>10.3f} "
      f"{m_B_uni['TPR']:>10.3f} "
      f"{check(m_A_uni['TPR'], m_B_uni['TPR']):>12}")
print(f"{'2b. Equalised Odds (FPR)':<35} "
      f"{m_A_uni['FPR']:>10.3f} "
      f"{m_B_uni['FPR']:>10.3f} "
      f"{check(m_A_uni['FPR'], m_B_uni['FPR']):>12}")
print(f"{'3. Calibration (PPV)':<35} "
      f"{m_A_uni['PPV']:>10.3f} "
      f"{m_B_uni['PPV']:>10.3f} "
      f"{check(m_A_uni['PPV'], m_B_uni['PPV']):>12}")
print(f"\nBase rates: P_A={P_A:.3f}, P_B={P_B:.3f}")
print(f"P_A != P_B -> Chouldechova's result applies")

print(f"\n{'─'*65}")
print(f"UNDER GROUP-SPECIFIC THRESHOLDS "
      f"(tau_A={TAU_A:.3f}, tau_B={TAU_B:.3f})")
print(f"{'─'*65}")
print(f"\n{'Criterion':<35} {'Group A':>10} {'Group B':>10} {'Status':>12}")
print(f"{'─'*65}")
print(f"{'1. Demographic Parity':<35} "
      f"{m_A_opt['selection_rate']:>10.3f} "
      f"{m_B_opt['selection_rate']:>10.3f} "
      f"{check(m_A_opt['selection_rate'], m_B_opt['selection_rate']):>12}")
print(f"{'2a. Equalised Odds (TPR)':<35} "
      f"{m_A_opt['TPR']:>10.3f} "
      f"{m_B_opt['TPR']:>10.3f} "
      f"{check(m_A_opt['TPR'], m_B_opt['TPR']):>12}")
print(f"{'2b. Equalised Odds (FPR)':<35} "
      f"{m_A_opt['FPR']:>10.3f} "
      f"{m_B_opt['FPR']:>10.3f} "
      f"{check(m_A_opt['FPR'], m_B_opt['FPR']):>12}")
print(f"{'3. Calibration (PPV)':<35} "
      f"{m_A_opt['PPV']:>10.3f} "
      f"{m_B_opt['PPV']:>10.3f} "
      f"{check(m_A_opt['PPV'], m_B_opt['PPV']):>12}")

# ─────────────────────────────────────────
# 4. CHOULDECHOVA DEMONSTRATION
# ─────────────────────────────────────────

print_header("CHOULDECHOVA IMPOSSIBILITY - NUMERICAL DEMONSTRATION")

FNR_hyp = 0.1
FPR_hyp = 0.1

PPV_A = ((1-FNR_hyp)*P_A /
         ((1-FNR_hyp)*P_A + FPR_hyp*(1-P_A)))
PPV_B = ((1-FNR_hyp)*P_B /
         ((1-FNR_hyp)*P_B + FPR_hyp*(1-P_B)))

print(f"""
Assume equalised odds satisfied:
  FNR_A = FNR_B = {FNR_hyp}
  FPR_A = FPR_B = {FPR_hyp}

Base rates: P_A={P_A:.3f}, P_B={P_B:.3f}

PPV_A = (1-{FNR_hyp})*{P_A:.3f} / [...] = {PPV_A:.3f}
PPV_B = (1-{FNR_hyp})*{P_B:.3f} / [...] = {PPV_B:.3f}

PPV_A ({PPV_A:.3f}) != PPV_B ({PPV_B:.3f})

When equalised odds is satisfied, calibration is violated.
When P_A != P_B, all three criteria CANNOT be satisfied.
This confirms Chouldechova's (2017) impossibility result.
""")

# ─────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────

ensure_output_dir()
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Uniform threshold
criteria = ['Demographic\nParity', 'Equalised\nOdds (TPR)',
            'Equalised\nOdds (FPR)', 'Calibration\n(PPV)']
vals_A_uni = [m_A_uni['selection_rate'], m_A_uni['TPR'],
              m_A_uni['FPR'], m_A_uni['PPV']]
vals_B_uni = [m_B_uni['selection_rate'], m_B_uni['TPR'],
              m_B_uni['FPR'], m_B_uni['PPV']]

x = np.arange(len(criteria))
width = 0.35
axes[0].bar(x - width/2, vals_A_uni, width,
            label='Group A', color=COLOR_A, alpha=0.8)
axes[0].bar(x + width/2, vals_B_uni, width,
            label='Group B', color=COLOR_B, alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(criteria, fontsize=9)
axes[0].set_ylabel('Value')
axes[0].set_title(f'Fairness Criteria\nUniform Threshold tau={UNIFORM_THRESHOLD}')
axes[0].legend()
axes[0].set_ylim(0, 1.2)
for i in range(len(criteria)):
    axes[0].text(i-width/2, vals_A_uni[i]+0.02,
                 f'{vals_A_uni[i]:.2f}', ha='center', fontsize=8)
    axes[0].text(i+width/2, vals_B_uni[i]+0.02,
                 f'{vals_B_uni[i]:.2f}', ha='center', fontsize=8)

# Plot 2: Group-specific thresholds
vals_A_opt = [m_A_opt['selection_rate'], m_A_opt['TPR'],
              m_A_opt['FPR'], m_A_opt['PPV']]
vals_B_opt = [m_B_opt['selection_rate'], m_B_opt['TPR'],
              m_B_opt['FPR'], m_B_opt['PPV']]

axes[1].bar(x - width/2, vals_A_opt, width,
            label='Group A', color=COLOR_A, alpha=0.8)
axes[1].bar(x + width/2, vals_B_opt, width,
            label='Group B', color=COLOR_B, alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(criteria, fontsize=9)
axes[1].set_ylabel('Value')
axes[1].set_title(f'Fairness Criteria\nGroup-Specific Thresholds')
axes[1].legend()
axes[1].set_ylim(0, 1.2)
for i in range(len(criteria)):
    axes[1].text(i-width/2, vals_A_opt[i]+0.02,
                 f'{vals_A_opt[i]:.2f}', ha='center', fontsize=8)
    axes[1].text(i+width/2, vals_B_opt[i]+0.02,
                 f'{vals_B_opt[i]:.2f}', ha='center', fontsize=8)

# Plot 3: Chouldechova curve
base_rates = np.linspace(0.05, 0.95, 200)
ppv_vals = ((1-FNR_hyp)*base_rates /
            ((1-FNR_hyp)*base_rates + FPR_hyp*(1-base_rates)))

axes[2].plot(base_rates, ppv_vals, color='purple', linewidth=2)
axes[2].axvline(x=P_A, color=COLOR_A, linestyle='--',
                linewidth=2, label=f'P_A={P_A:.3f}')
axes[2].axvline(x=P_B, color=COLOR_B, linestyle='--',
                linewidth=2, label=f'P_B={P_B:.3f}')
axes[2].scatter([P_A], [PPV_A], color=COLOR_A, s=100, zorder=5,
                label=f'PPV_A={PPV_A:.3f}')
axes[2].scatter([P_B], [PPV_B], color=COLOR_B, s=100, zorder=5,
                label=f'PPV_B={PPV_B:.3f}')
axes[2].set_xlabel('Base Rate P(T=1|G)')
axes[2].set_ylabel('PPV (Calibration)')
axes[2].set_title("Chouldechova's Impossibility:\nPPV varies with Base Rate")
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.suptitle(f'Fairness Metrics\n{FIGURE_TITLE_SUFFIX}',
             fontsize=12, fontweight='bold')
plt.tight_layout()

filepath = get_output_path('fairness_metrics.png')
plt.savefig(filepath, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved to {filepath}")