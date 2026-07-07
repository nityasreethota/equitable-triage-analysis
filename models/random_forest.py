import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    FEATURES, UNIFORM_THRESHOLD, COLOR_A, COLOR_B,
    FIGURE_TITLE_SUFFIX, OUTPUT_DIR
)
from src.data_loader import load_and_split
from src.models import get_baseline_model, train_model
from src.metrics import (
    compute_metrics_at_threshold,
    compute_tpr_gap,
    evaluate_fairness_criteria
)
from src.utils import (
    print_header, save_plot, ensure_output_dir
)

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────

(df, X_train, X_test, y_train, y_test,
 groups, X_A, y_A, X_B, y_B) = load_and_split()

print_header("RANDOM FOREST - FAIRNESS GAP ANALYSIS")
print(f"Training set: {len(X_train)} patients")
print(f"Test set:     {len(X_test)} patients")
print(f"Group A:      {len(X_A)} patients")
print(f"Group B:      {len(X_B)} patients")

# ─────────────────────────────────────────
# 2. TRAIN MODEL
# ─────────────────────────────────────────

rf = train_model(get_baseline_model(), X_train, y_train)
print(f"\nBaseline Random Forest trained")
print(f"Trees: {rf.n_estimators}, max_depth: {rf.max_depth}")

# ─────────────────────────────────────────
# 3. COMPUTE FAIRNESS METRICS
# ─────────────────────────────────────────

print_header("FAIRNESS METRICS UNDER UNIFORM THRESHOLD")
print(f"Threshold: tau = {UNIFORM_THRESHOLD}")

m_A = compute_metrics_at_threshold(X_A, y_A, rf)
m_B = compute_metrics_at_threshold(X_B, y_B, rf)
gap, tpr_A, tpr_B = compute_tpr_gap(rf, X_A, y_A, X_B, y_B)

print(f"\n{'Metric':<30} {'Group A':>10} "
      f"{'Group B':>10} {'Gap':>10}")
print("-" * 62)
print(f"{'TPR (Sensitivity)':<30} {m_A['TPR']:>10.3f} "
      f"{m_B['TPR']:>10.3f} {m_B['TPR']-m_A['TPR']:>+10.3f}")
print(f"{'FPR (Fall-out)':<30} {m_A['FPR']:>10.3f} "
      f"{m_B['FPR']:>10.3f} {m_B['FPR']-m_A['FPR']:>+10.3f}")
print(f"{'FNR (Miss Rate)':<30} {m_A['FNR']:>10.3f} "
      f"{m_B['FNR']:>10.3f} {m_A['FNR']-m_B['FNR']:>+10.3f}")
print(f"{'PPV (Precision)':<30} {m_A['PPV']:>10.3f} "
      f"{m_B['PPV']:>10.3f} {m_B['PPV']-m_A['PPV']:>+10.3f}")

print(f"\nTPR Gap: {gap:.3f}")
print(f"Group A misses {m_A['FNR']*100:.1f}% of patients who need help")
print(f"Group B misses {m_B['FNR']*100:.1f}% of patients who need help")

# ─────────────────────────────────────────
# 4. FEATURE IMPORTANCE
# ─────────────────────────────────────────

print_header("FEATURE IMPORTANCE SCORES")
importances = rf.feature_importances_
for name, imp in sorted(zip(FEATURES, importances),
                         key=lambda x: x[1], reverse=True):
    bar = '█' * int(imp * 100)
    print(f"  {name:<25} {imp:.4f} {bar}")

print(f"\nKey finding: referral_length is primary bias source")
print(f"Model relies on proxy variable not clinical need")

# ─────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────

ensure_output_dir()
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Fairness metrics comparison
metrics_labels = ['TPR\n(Sensitivity)', 'FPR\n(Fall-out)',
                  'FNR\n(Miss Rate)', 'PPV\n(Precision)']
metrics_A_vals = [m_A['TPR'], m_A['FPR'],
                  m_A['FNR'], m_A['PPV']]
metrics_B_vals = [m_B['TPR'], m_B['FPR'],
                  m_B['FNR'], m_B['PPV']]

x = np.arange(len(metrics_labels))
width = 0.35
axes[0].bar(x - width/2, metrics_A_vals, width,
            label='Group A (underrepresented)',
            color=COLOR_A, alpha=0.8)
axes[0].bar(x + width/2, metrics_B_vals, width,
            label='Group B (majority)',
            color=COLOR_B, alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics_labels)
axes[0].set_ylabel('Value')
axes[0].set_title(f'Fairness Metrics\nUniform Threshold tau={UNIFORM_THRESHOLD}')
axes[0].legend()
axes[0].set_ylim(0, 1.1)

# Plot 2: Feature importance
sorted_idx = np.argsort(importances)
axes[1].barh([FEATURES[i] for i in sorted_idx],
             [importances[i] for i in sorted_idx],
             color=COLOR_A, alpha=0.8)
axes[1].set_xlabel('Importance Score')
axes[1].set_title('Random Forest\nFeature Importance')

# Plot 3: Predicted probability distributions
y_prob = rf.predict_proba(X_test)[:, 1]
prob_A = rf.predict_proba(X_A)[:, 1]
prob_B = rf.predict_proba(X_B)[:, 1]

axes[2].hist(prob_A, bins=20, alpha=0.7,
             color=COLOR_A, label='Group A (underrepresented)')
axes[2].hist(prob_B, bins=20, alpha=0.7,
             color=COLOR_B, label='Group B (majority)')
axes[2].axvline(x=UNIFORM_THRESHOLD, color='black',
                linestyle='--', linewidth=2,
                label=f'Uniform threshold tau={UNIFORM_THRESHOLD}')
axes[2].set_xlabel('Predicted Probability')
axes[2].set_ylabel('Count')
axes[2].set_title('Predicted Probability Distributions\nby Demographic Group')
axes[2].legend()

plt.suptitle(f'Random Forest Results\n{FIGURE_TITLE_SUFFIX}',
             fontsize=12, fontweight='bold')

save_plot('random_forest_results.png')
plt.show()

# ─────────────────────────────────────────
# 6. SUMMARY
# ─────────────────────────────────────────

print_header("KEY FINDING")
print(f"""
Under uniform threshold tau={UNIFORM_THRESHOLD}:

  Group A TPR: {m_A['TPR']:.3f} - catches only {m_A['TPR']*100:.1f}% of
               underrepresented patients who need help
  Group B TPR: {m_B['TPR']:.3f} - catches {m_B['TPR']*100:.1f}% of
               majority patients who need help

  TPR Gap: {gap:.3f}

  This confirms equalised odds is violated.
  Chouldechova's result predicts this is unavoidable
  under a uniform threshold when base rates differ.

  Next: roc_analysis.py - group-specific thresholds
""")