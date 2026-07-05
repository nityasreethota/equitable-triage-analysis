import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────
# 1. RANDOM SEED (for reproducibility)
# ─────────────────────────────────────────
np.random.seed(42)

# ─────────────────────────────────────────
# 2. POPULATION PARAMETERS
# Justified by Joyce et al. (2021) and NHS Digital (2024)
# ─────────────────────────────────────────

# Population sizes
N_A = 200   # Group A: underrepresented (young carers, ethnic minorities)
N_B = 800   # Group B: majority population

# Base rates P(T=1 | G) - probability of genuinely needing help
# Group A lower because historically underrepresented in training data
P_A = 0.30
P_B = 0.50

# Beta distribution parameters for risk scores
# Group A: more uncertain, lower scores - model has less signal
ALPHA_A, BETA_A = 1.5, 4.0
# Group B: more confident, higher scores - model trained on similar patients
ALPHA_B, BETA_B = 2.5, 3.0

# ─────────────────────────────────────────
# 3. GENERATE GROUP A (underrepresented)
# ─────────────────────────────────────────

# Risk scores from Beta distribution
scores_A = np.random.beta(ALPHA_A, BETA_A, N_A)

# True clinical status (genuine need)
T_A = np.random.binomial(1, P_A, N_A)

# Referral letter length - shorter and vaguer for Group A
# Reflecting data sparsity documented in Joyce et al. (2021)
length_A = np.clip(np.random.normal(150, 50, N_A), 50, 400)

# Previous contacts with services - fewer for underrepresented group
contacts_A = np.random.poisson(0.5, N_A)

# Age - younger on average (young carers)
age_A = np.clip(np.random.normal(22, 8, N_A), 14, 65)

# ─────────────────────────────────────────
# 4. GENERATE GROUP B (majority)
# ─────────────────────────────────────────

# Risk scores from Beta distribution
scores_B = np.random.beta(ALPHA_B, BETA_B, N_B)

# True clinical status
T_B = np.random.binomial(1, P_B, N_B)

# Referral letter length - longer and more detailed
length_B = np.clip(np.random.normal(300, 80, N_B), 100, 600)

# Previous contacts - more contacts, better known to services
contacts_B = np.random.poisson(2.0, N_B)

# Age
age_B = np.clip(np.random.normal(35, 12, N_B), 18, 75)

# ─────────────────────────────────────────
# 5. BUILD DATAFRAMES
# ─────────────────────────────────────────

df_A = pd.DataFrame({
    'group': 'A',
    'risk_score': scores_A,
    'true_outcome': T_A,
    'referral_length': length_A,
    'previous_contacts': contacts_A,
    'age': age_A
})

df_B = pd.DataFrame({
    'group': 'B',
    'risk_score': scores_B,
    'true_outcome': T_B,
    'referral_length': length_B,
    'previous_contacts': contacts_B,
    'age': age_B
})

# Combine into single dataset
df = pd.concat([df_A, df_B], ignore_index=True)

# ─────────────────────────────────────────
# 6. SUMMARY STATISTICS
# ─────────────────────────────────────────

print("=" * 50)
print("SYNTHETIC DATASET SUMMARY")
print("=" * 50)
print(f"Total patients: {len(df)}")
print(f"Group A (underrepresented): {N_A}")
print(f"Group B (majority): {N_B}")
print()
print("Base rates (true clinical need):")
print(f"  Group A: {T_A.mean():.3f} (target: {P_A})")
print(f"  Group B: {T_B.mean():.3f} (target: {P_B})")
print()
print("Mean risk scores:")
print(f"  Group A: {scores_A.mean():.3f}")
print(f"  Group B: {scores_B.mean():.3f}")
print()
print("Mean referral letter length:")
print(f"  Group A: {length_A.mean():.1f} words")
print(f"  Group B: {length_B.mean():.1f} words")
print()
print(df.head(10))

# ─────────────────────────────────────────
# 7. VISUALISE RISK SCORE DISTRIBUTIONS
# ─────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Risk score distributions
axes[0].hist(scores_A, bins=30, alpha=0.7,
             color='steelblue', label='Group A (underrepresented)')
axes[0].hist(scores_B, bins=30, alpha=0.7,
             color='coral', label='Group B (majority)')
axes[0].axvline(x=0.4, color='black', linestyle='--',
                linewidth=2, label='Uniform threshold τ=0.4')
axes[0].set_xlabel('Risk Score')
axes[0].set_ylabel('Count')
axes[0].set_title('Risk Score Distributions by Demographic Group')
axes[0].legend()

# Base rate comparison
groups = ['Group A\n(underrepresented)', 'Group B\n(majority)']
base_rates = [T_A.mean(), T_B.mean()]
colors = ['steelblue', 'coral']
axes[1].bar(groups, base_rates, color=colors, alpha=0.8)
axes[1].axhline(y=P_A, color='steelblue', linestyle='--', alpha=0.5)
axes[1].axhline(y=P_B, color='coral', linestyle='--', alpha=0.5)
axes[1].set_ylabel('Base Rate P(T=1|G)')
axes[1].set_title('True Clinical Need by Demographic Group\n(Base Rates)')
axes[1].set_ylim(0, 0.7)
for i, v in enumerate(base_rates):
    axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()

# Save visualisation
os.makedirs('visualisations/outputs', exist_ok=True)
plt.savefig('visualisations/outputs/risk_distributions.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to visualisations/outputs/risk_distributions.png")

# ─────────────────────────────────────────
# 8. SAVE DATASET
# ─────────────────────────────────────────

os.makedirs('data', exist_ok=True)
df.to_csv('data/synthetic_nhs_data.csv', index=False)
print(f"\nDataset saved to data/synthetic_nhs_data.csv")
print(f"Shape: {df.shape}")