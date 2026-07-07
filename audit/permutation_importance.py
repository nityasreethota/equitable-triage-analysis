import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# ─────────────────────────────────────────
# 1. LOAD DATA
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

print("=" * 65)
print("PERMUTATION IMPORTANCE")
print("=" * 65)
print(f"""
Formula:
    I_j = (1/T) * Σ (E_t^(j) - E_t)
""")

# ─────────────────────────────────────────
# 2. TRAIN INITIAL RANDOM FOREST
# Default parameters
# ─────────────────────────────────────────

T = 100

rf_initial = RandomForestClassifier(
    n_estimators=T,
    max_depth=5,
    random_state=42,
    class_weight='balanced'
)
rf_initial.fit(X_train, y_train)

print(f"Initial Random Forest trained")
print(f"Parameters: n_estimators={T}, max_depth=5")
print(f"Default: bootstrap=True, max_features='sqrt'")
print()

# ─────────────────────────────────────────
# 3. PERMUTATION IMPORTANCE
# Implementing formula: I_j = (1/T) * sum(E_t^j - E_t)
# ─────────────────────────────────────────

def classification_error(y_true, y_pred):
    """
    Classification error rate = proportion of wrong predictions.
    Equivalent to 1 - accuracy.
    """
    return np.mean(y_true != y_pred)

def permutation_importance_from_scratch(rf, X_test, y_test,
                                         feature_names, T=100):
    """
    Calculate permutation importance for each feature using formula:
        I_j = (1/T) * sum_{t=1}^{T} (E_t^(j) - E_t)
    """
    X_test_array = X_test.values
    y_test_array = y_test.values

    importance_scores = {}

    for j, feature in enumerate(feature_names):
        print(f"  Computing permutation importance for: {feature}")

        differences = []  # stores E_t^(j) - E_t for each tree

        for t, tree in enumerate(rf.estimators_):
            # Original error E_t
            y_pred_original = tree.predict(X_test_array)
            E_t = classification_error(y_test_array, y_pred_original)

            # Shuffle feature j
            X_shuffled = X_test_array.copy()
            np.random.seed(t)  # reproducible shuffle per tree
            np.random.shuffle(X_shuffled[:, j])

            # Error after shuffling E_t^(j)
            y_pred_shuffled = tree.predict(X_shuffled)
            E_t_j = classification_error(y_test_array, y_pred_shuffled)

            differences.append(E_t_j - E_t)

        # I_j = (1/T) * sum(E_t^(j) - E_t)
        I_j = np.mean(differences)
        importance_scores[feature] = {
            'I_j': I_j,
            'differences': differences,
            'mean': np.mean(differences),
            'std': np.std(differences)
        }

    return importance_scores

print("Computing permutation importance from scratch...")
print()

importance_initial = permutation_importance_from_scratch(
    rf_initial, X_test, y_test, FEATURES, T
)

# ─────────────────────────────────────────
# 4. DISPLAY RESULTS
# ─────────────────────────────────────────

print()
print("=" * 65)
print("PERMUTATION IMPORTANCE RESULTS")
print("=" * 65)
print(f"\n{'Feature':<25} {'I_j (mean)':<15} "
      f"{'Std':<10} {'Bias Level'}")
print("-" * 65)

sorted_features = sorted(importance_initial.items(),
                          key=lambda x: x[1]['I_j'],
                          reverse=True)

for feature, scores in sorted_features:
    I_j = scores['I_j']
    std = scores['std']
    if I_j > 0.05:
        bias = "HIGH - proxy variable"
    elif I_j > 0.02:
        bias = "MEDIUM"
    else:
        bias = "LOW"
    print(f"{feature:<25} {I_j:<15.4f} {std:<10.4f} {bias}")

ref_importance = importance_initial['referral_length']['I_j']
print(f"""
Key finding:
  referral_length importance: {ref_importance:.4f}
  
  This means shuffling referral_length increases classification
  error by {ref_importance:.4f} on average across all {T} trees.
  
  referral_length is a PROXY variable - it reflects data quality
  and demographic group membership, not clinical need.
  This is the primary source of demographic bias in the model.
""")

# ─────────────────────────────────────────
# 5. SHOW TREE-BY-TREE TABLE FOR referral_length
# ─────────────────────────────────────────

print("=" * 65)
print("TREE-BY-TREE BREAKDOWN: referral_length")
print("Showing first 10 trees")
print("=" * 65)

X_test_array = X_test.values
y_test_array = y_test.values
j_ref = FEATURES.index('referral_length')

print(f"\n{'Tree t':<10} {'E_t':<12} {'E_t^(j)':<12} "
      f"{'E_t^(j) - E_t':<15}")
print("-" * 50)

differences_ref = []
for t, tree in enumerate(rf_initial.estimators_[:10]):
    y_pred_orig = tree.predict(X_test_array)
    E_t = classification_error(y_test_array, y_pred_orig)

    X_shuffled = X_test_array.copy()
    np.random.seed(t)
    np.random.shuffle(X_shuffled[:, j_ref])

    y_pred_shuf = tree.predict(X_shuffled)
    E_t_j = classification_error(y_test_array, y_pred_shuf)

    diff = E_t_j - E_t
    differences_ref.append(diff)
    print(f"{t+1:<10} {E_t:<12.4f} {E_t_j:<12.4f} {diff:<15.4f}")

print(f"\n{'':10} {'':12} {'':12} "
      f"Sum = {sum(differences_ref):.4f}")
print(f"\nI_j (referral_length, first 10 trees) = "
      f"{sum(differences_ref)/10:.4f}")

# ─────────────────────────────────────────
# 6. GROUP-SPECIFIC PERMUTATION IMPORTANCE
# ─────────────────────────────────────────

print()
print("=" * 65)
print("GROUP-SPECIFIC PERMUTATION IMPORTANCE")
print("Does bias affect Group A more than Group B?")
print("=" * 65)

groups_test = df.loc[X_test.index, 'group'].values
X_test_A = X_test.values[groups_test == 'A']
y_test_A = y_test.values[groups_test == 'A']
X_test_B = X_test.values[groups_test == 'B']
y_test_B = y_test.values[groups_test == 'B']

def group_permutation_importance(rf, X_grp, y_grp,
                                  feature_idx, n_trees=10):
    """Permutation importance calculated within a demographic group."""
    diffs = []
    for t, tree in enumerate(rf.estimators_[:n_trees]):
        y_pred_orig = tree.predict(X_grp)
        E_t = classification_error(y_grp, y_pred_orig)

        X_shuf = X_grp.copy()
        np.random.seed(t)
        np.random.shuffle(X_shuf[:, feature_idx])

        y_pred_shuf = tree.predict(X_shuf)
        E_t_j = classification_error(y_grp, y_pred_shuf)
        diffs.append(E_t_j - E_t)
    return np.mean(diffs)

j_ref = FEATURES.index('referral_length')
j_risk = FEATURES.index('risk_score')

imp_ref_A = group_permutation_importance(
    rf_initial, X_test_A, y_test_A, j_ref)
imp_ref_B = group_permutation_importance(
    rf_initial, X_test_B, y_test_B, j_ref)
imp_risk_A = group_permutation_importance(
    rf_initial, X_test_A, y_test_A, j_risk)
imp_risk_B = group_permutation_importance(
    rf_initial, X_test_B, y_test_B, j_risk)

print(f"\n{'Feature':<25} {'Group A':<12} {'Group B':<12} {'Gap'}")
print("-" * 55)
print(f"{'referral_length':<25} {imp_ref_A:<12.4f} "
      f"{imp_ref_B:<12.4f} {imp_ref_A-imp_ref_B:<12.4f}")
print(f"{'risk_score':<25} {imp_risk_A:<12.4f} "
      f"{imp_risk_B:<12.4f} {imp_risk_A-imp_risk_B:<12.4f}")

print(f"""
Interpretation:
  If referral_length importance is higher for Group A than Group B,
  it means the model relies MORE on this proxy variable when
  making decisions about underrepresented patients
""")

# ─────────────────────────────────────────
# 7. VISUALISATIONS
# ─────────────────────────────────────────

os.makedirs('visualisations/outputs', exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Overall permutation importance
features = [f for f, _ in sorted_features]
importances = [s['I_j'] for _, s in sorted_features]
stds = [s['std'] for _, s in sorted_features]
colors = ['red' if f == 'referral_length' else 'steelblue'
          for f in features]

axes[0].barh(features, importances, xerr=stds,
             color=colors, alpha=0.8, capsize=5)
axes[0].set_xlabel('Permutation Importance I_j')
axes[0].set_title('Permutation Importance (Initial Model)\n'
                   'Red = primary bias source')
axes[0].axvline(x=0, color='black', linewidth=0.5)
axes[0].grid(True, alpha=0.3)

# Plot 2: Tree-by-tree for referral_length
trees = list(range(1, 11))
axes[1].bar(trees, differences_ref,
            color=['red' if d > 0 else 'steelblue'
                   for d in differences_ref],
            alpha=0.8)
axes[1].axhline(y=0, color='black', linewidth=1)
axes[1].axhline(y=np.mean(differences_ref), color='red',
                linestyle='--', linewidth=2,
                label=f'Mean I_j = {np.mean(differences_ref):.4f}')
axes[1].set_xlabel('Tree t')
axes[1].set_ylabel('E_t^(j) - E_t')
axes[1].set_title('Tree-by-Tree Permutation Importance\n'
                   'Feature: referral_length')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Group-specific importance
group_features = ['referral_length', 'risk_score']
imp_A = [imp_ref_A, imp_risk_A]
imp_B = [imp_ref_B, imp_risk_B]

x = np.arange(len(group_features))
width = 0.35
axes[2].bar(x - width/2, imp_A, width,
            label='Group A (underrepresented)',
            color='steelblue', alpha=0.8)
axes[2].bar(x + width/2, imp_B, width,
            label='Group B (majority)',
            color='coral', alpha=0.8)
axes[2].set_xticks(x)
axes[2].set_xticklabels(group_features)
axes[2].set_ylabel('Permutation Importance I_j')
axes[2].set_title('Group-Specific Permutation Importance\n'
                   'Does bias affect Group A more?')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle('Permutation Importance: Identifying Bias Sources\n'
             'From Probabilities to Decisions',
             fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('visualisations/outputs/permutation_importance.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to visualisations/outputs/permutation_importance.png")

# ─────────────────────────────────────────
# 8. SUMMARY
# ─────────────────────────────────────────

print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"""
Initial model permutation importance for referral_length:
  I_j = {ref_importance:.4f}
""")