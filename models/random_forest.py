
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                             classification_report)
import os

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────

df = pd.read_csv('data/synthetic_nhs_data.csv')
print("Dataset loaded successfully")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print()

# ─────────────────────────────────────────
# 2. PREPARE FEATURES
# ─────────────────────────────────────────

# Encode group as binary
# df['group_binary'] = (df['group'] == 'B').astype(int)

# Features and target
# FEATURES = ['risk_score', 'referral_length',
#             'previous_contacts', 'age', 'group_binary']
FEATURES = ['risk_score', 'referral_length',
            'previous_contacts', 'age']
TARGET = 'true_outcome'

X = df[FEATURES]
y = df[TARGET]

# Train/test split - stratified to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print(f"Training set: {X_train.shape[0]} patients")
print(f"Test set: {X_test.shape[0]} patients")
print()

# ─────────────────────────────────────────
# 3. TRAIN RANDOM FOREST
# ─────────────────────────────────────────

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    class_weight='balanced'  # accounts for class imbalance
)

rf.fit(X_train, y_train)
print("Random Forest trained successfully")
print(f"Number of trees: {rf.n_estimators}")
print()

# ─────────────────────────────────────────
# 4. PREDICT PROBABILITIES
# ─────────────────────────────────────────

# Get predicted probabilities for test set
y_prob = rf.predict_proba(X_test)[:, 1]

# Add predictions back to test dataframe
df_test = X_test.copy()
df_test['true_outcome'] = y_test
df_test['predicted_prob'] = y_prob
df_test['group'] = df.loc[X_test.index, 'group'].values

# ─────────────────────────────────────────
# 5. APPLY UNIFORM THRESHOLD
# ─────────────────────────────────────────

UNIFORM_THRESHOLD = 0.4

df_test['decision_uniform'] = (
    df_test['predicted_prob'] >= UNIFORM_THRESHOLD
).astype(int)

# ─────────────────────────────────────────
# 6. MEASURE FAIRNESS GAP
# ─────────────────────────────────────────

def compute_metrics(df_group, decision_col='decision_uniform'):
    """Compute TPR, FPR, FNR, PPV for a group."""
    TP = ((df_group[decision_col] == 1) &
          (df_group['true_outcome'] == 1)).sum()
    FP = ((df_group[decision_col] == 1) &
          (df_group['true_outcome'] == 0)).sum()
    TN = ((df_group[decision_col] == 0) &
          (df_group['true_outcome'] == 0)).sum()
    FN = ((df_group[decision_col] == 0) &
          (df_group['true_outcome'] == 1)).sum()

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # Sensitivity
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # Fall-out
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0  # Miss rate
    PPV = TP / (TP + FP) if (TP + FP) > 0 else 0  # Precision

    return {
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'TPR': TPR, 'FPR': FPR, 'FNR': FNR, 'PPV': PPV
    }

# Split test set by group
df_test_A = df_test[df_test['group'] == 'A']
df_test_B = df_test[df_test['group'] == 'B']

metrics_A = compute_metrics(df_test_A)
metrics_B = compute_metrics(df_test_B)

print("=" * 60)
print("FAIRNESS METRICS UNDER UNIFORM THRESHOLD (τ=0.4)")
print("=" * 60)
print(f"\n{'Metric':<30} {'Group A':>10} {'Group B':>10} {'Gap':>10}")
print("-" * 60)
print(f"{'True Positive Rate (TPR)':<30} "
      f"{metrics_A['TPR']:>10.3f} "
      f"{metrics_B['TPR']:>10.3f} "
      f"{metrics_B['TPR']-metrics_A['TPR']:>10.3f}")
print(f"{'False Positive Rate (FPR)':<30} "
      f"{metrics_A['FPR']:>10.3f} "
      f"{metrics_B['FPR']:>10.3f} "
      f"{metrics_B['FPR']-metrics_A['FPR']:>10.3f}")
print(f"{'False Negative Rate (FNR)':<30} "
      f"{metrics_A['FNR']:>10.3f} "
      f"{metrics_B['FNR']:>10.3f} "
      f"{metrics_A['FNR']-metrics_B['FNR']:>10.3f}")
print(f"{'Positive Predictive Value':<30} "
      f"{metrics_A['PPV']:>10.3f} "
      f"{metrics_B['PPV']:>10.3f} "
      f"{metrics_B['PPV']-metrics_A['PPV']:>10.3f}")
print()
print(f"Group A patients in test set: {len(df_test_A)}")
print(f"Group B patients in test set: {len(df_test_B)}")

# ─────────────────────────────────────────
# 7. FEATURE IMPORTANCE
# ─────────────────────────────────────────

importances = rf.feature_importances_
feature_names = FEATURES

print()
print("=" * 60)
print("FEATURE IMPORTANCE SCORES")
print("=" * 60)
for name, imp in sorted(zip(feature_names, importances),
                         key=lambda x: x[1], reverse=True):
    print(f"  {name:<25} {imp:.4f}")

# ─────────────────────────────────────────
# 8. VISUALISATIONS
# ─────────────────────────────────────────

os.makedirs('visualisations/outputs', exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: TPR and FNR comparison
metrics_labels = ['TPR\n(Sensitivity)', 'FPR\n(Fall-out)',
                  'FNR\n(Miss Rate)', 'PPV\n(Precision)']
metrics_A_vals = [metrics_A['TPR'], metrics_A['FPR'],
                  metrics_A['FNR'], metrics_A['PPV']]
metrics_B_vals = [metrics_B['TPR'], metrics_B['FPR'],
                  metrics_B['FNR'], metrics_B['PPV']]

x = np.arange(len(metrics_labels))
width = 0.35

axes[0].bar(x - width/2, metrics_A_vals, width,
            label='Group A (underrepresented)',
            color='steelblue', alpha=0.8)
axes[0].bar(x + width/2, metrics_B_vals, width,
            label='Group B (majority)',
            color='coral', alpha=0.8)
axes[0].set_xlabel('Metric')
axes[0].set_ylabel('Value')
axes[0].set_title('Fairness Metrics Under\nUniform Threshold τ=0.4')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics_labels)
axes[0].legend()
axes[0].set_ylim(0, 1)

# Plot 2: Feature importance
sorted_idx = np.argsort(importances)
axes[1].barh([feature_names[i] for i in sorted_idx],
             [importances[i] for i in sorted_idx],
             color='steelblue', alpha=0.8)
axes[1].set_xlabel('Importance Score')
axes[1].set_title('Random Forest\nFeature Importance')

# Plot 3: Predicted probability distributions by group
axes[2].hist(df_test_A['predicted_prob'], bins=20,
             alpha=0.7, color='steelblue',
             label='Group A (underrepresented)')
axes[2].hist(df_test_B['predicted_prob'], bins=20,
             alpha=0.7, color='coral',
             label='Group B (majority)')
axes[2].axvline(x=UNIFORM_THRESHOLD, color='black',
                linestyle='--', linewidth=2,
                label=f'Uniform threshold τ={UNIFORM_THRESHOLD}')
axes[2].set_xlabel('Predicted Probability')
axes[2].set_ylabel('Count')
axes[2].set_title('Predicted Probability Distributions\nby Demographic Group')
axes[2].legend()

plt.tight_layout()
plt.savefig('visualisations/outputs/random_forest_results.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved to visualisations/outputs/random_forest_results.png")

# ─────────────────────────────────────────
# 9. KEY FINDING SUMMARY
# ─────────────────────────────────────────

print()
print("=" * 60)
print("KEY FINDING")
print("=" * 60)
print(f"""
Under a uniform threshold of τ={UNIFORM_THRESHOLD}:

Group A (underrepresented) TPR: {metrics_A['TPR']:.3f}
Group B (majority) TPR:         {metrics_B['TPR']:.3f}
TPR Gap:                        {metrics_B['TPR']-metrics_A['TPR']:.3f}

Group A (underrepresented) FNR: {metrics_A['FNR']:.3f}
Group B (majority) FNR:         {metrics_B['FNR']:.3f}
FNR Gap:                        {metrics_A['FNR']-metrics_B['FNR']:.3f}
""")