import numpy as np
import pandas as pd
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

np.random.seed(42)

# ─────────────────────────────────────────
# 1. QUESTIONNAIRE DEFINITIONS
# Raw items exactly as they appear in the forms
# ─────────────────────────────────────────

# SDQ Items - 25 questions, scored 0/1/2
SDQ_ITEMS = [
    "considerate",        # 1  - prosocial
    "restless",           # 2  - hyperactivity
    "somatic",            # 3  - emotional
    "shares",             # 4  - prosocial
    "temper",             # 5  - conduct
    "solitary",           # 6  - peer
    "obedient",           # 7  - conduct (reversed)
    "worry",              # 8  - emotional
    "helpful",            # 9  - prosocial
    "fidgety",            # 10 - hyperactivity
    "friend",             # 11 - peer (reversed)
    "fights",             # 12 - conduct
    "unhappy",            # 13 - emotional
    "liked",              # 14 - peer (reversed)
    "distracted",         # 15 - hyperactivity
    "nervous",            # 16 - emotional
    "kind",               # 17 - prosocial
    "lies",               # 18 - conduct
    "bullied",            # 19 - peer
    "volunteers",         # 20 - prosocial
    "thinks",             # 21 - hyperactivity (reversed)
    "steals",             # 22 - conduct
    "adults",             # 23 - peer
    "fears",              # 24 - emotional
    "attention",          # 25 - hyperactivity (reversed)
]

# SDQ Subscale mapping (1-indexed item numbers)
SDQ_SUBSCALES = {
    "emotional":      [3, 8, 13, 16, 24],
    "conduct":        [5, 7, 12, 18, 22],
    "hyperactivity":  [2, 10, 15, 21, 25],
    "peer":           [6, 11, 14, 19, 23],
    "prosocial":      [1, 4, 9, 17, 20],
}

# Items that need reversing (higher score = better)
SDQ_REVERSED = [7, 11, 14, 21, 25]

# RCADS Items - 47 questions, scored 0/1/2/3
RCADS_ITEMS = [
    "worries_things",           # 1  - GAD
    "sad_empty",                # 2  - Depression
    "stomach_funny",            # 3  - Panic
    "poorly_at_something",      # 4  - GAD
    "afraid_alone",             # 5  - Sep Anxiety
    "nothing_fun",              # 6  - Depression
    "scared_test",              # 7  - Social Phobia
    "someone_angry",            # 8  - Social Phobia
    "away_from_parent",         # 9  - Sep Anxiety
    "bad_thoughts",             # 10 - OCD
    "trouble_sleeping",         # 11 - Depression
    "bad_school_work",          # 12 - GAD
    "awful_family",             # 13 - Sep Anxiety
    "cant_breathe",             # 14 - Panic
    "appetite_problems",        # 15 - Depression
    "keep_checking",            # 16 - OCD
    "scared_sleep_alone",       # 17 - Sep Anxiety
    "trouble_school_morning",   # 18 - Sep Anxiety
    "no_energy",                # 19 - Depression
    "look_foolish",             # 20 - Social Phobia
    "tired_lot",                # 21 - Depression
    "bad_things_happen_me",     # 22 - GAD
    "bad_thoughts_out_of_head", # 23 - OCD
    "heart_beats_fast",         # 24 - Panic
    "cannot_think_clearly",     # 25 - Depression
    "tremble_shake",            # 26 - Panic
    "something_bad_happen",     # 27 - GAD
    "feels_shaky",              # 28 - Panic
    "worthless",                # 29 - Depression
    "making_mistakes",          # 30 - Social Phobia
    "special_thoughts",         # 31 - OCD
    "what_people_think",        # 32 - Social Phobia
    "crowded_places",           # 33 - Social Phobia
    "scared_no_reason",         # 34 - Sep Anxiety
    "what_is_going_to_happen",  # 35 - GAD
    "dizzy_faint",              # 36 - Panic
    "think_about_death",        # 37 - Depression
    "talk_front_class",         # 38 - Social Phobia
    "heart_too_quickly",        # 39 - Panic
    "dont_want_move",           # 40 - Depression
    "suddenly_get_scared",      # 41 - Panic
    "over_and_over",            # 42 - OCD
    "make_fool_of_myself",      # 43 - Social Phobia
    "just_the_right_way",       # 44 - OCD
    "worries_in_bed",           # 45 - GAD
    "away_home_overnight",      # 46 - Sep Anxiety
    "feels_restless",           # 47 - Depression
]

# RCADS Subscale mapping (1-indexed)
RCADS_SUBSCALES = {
    "separation_anxiety": [5, 9, 13, 17, 18, 34, 46],
    "social_phobia":      [7, 8, 20, 30, 32, 33, 38, 43],
    "gad":                [1, 4, 12, 22, 27, 35, 45],
    "panic":              [3, 14, 24, 26, 28, 36, 39, 41],
    "ocd":                [10, 16, 23, 31, 42, 44],
    "depression":         [2, 6, 11, 15, 19, 21, 25, 29, 37, 40, 47],
}

# Impact supplement items
IMPACT_ITEMS_PARENT = [
    "overall_severity",    # 0=No / 1=Minor / 2=Definite / 3=Severe
    "duration",            # 0=<1mo / 1=1-5mo / 2=6-12mo / 3=>1yr
    "distress",            # 0-3
    "home_life",           # 0-3
    "friendships",         # 0-3
    "classroom_learning",  # 0-3
    "leisure",             # 0-3
    "family_burden",       # 0-3
]

IMPACT_ITEMS_CHILD = [
    "overall_severity",    # 0-3
    "duration",            # 0-3
    "distress",            # 0-3
    "home_life",           # 0-3
    "friendships",         # 0-3
    "classroom_learning",  # 0-3
    "leisure",             # 0-3
    "social_burden",       # 0-3
]

print("=" * 65)
print("CAMHS SYNTHETIC DATA GENERATION")
print("From raw questionnaire items to model-ready features")
print("=" * 65)
print(f"""
Questionnaires:
  SDQ:    {len(SDQ_ITEMS)} items x 3 options (0/1/2) x 2 perspectives
  RCADS:  {len(RCADS_ITEMS)} items x 4 options (0/1/2/3) x 2 perspectives
  Impact: {len(IMPACT_ITEMS_PARENT)} items x 4 options x 2 perspectives

Total raw items per patient: {len(SDQ_ITEMS)*2 + len(RCADS_ITEMS)*2 + len(IMPACT_ITEMS_PARENT) + len(IMPACT_ITEMS_CHILD)}
""")

# ─────────────────────────────────────────
# 2. POPULATION PARAMETERS
# ─────────────────────────────────────────

N_A = 200   # Group A: young carers / underrepresented
N_B = 800   # Group B: majority population

# Base rates - probability of genuinely needing CAMHS
P_A = 0.45  # higher than CHRONOSIG - CAMHS referrals already filtered
P_B = 0.60  # majority group better represented in training data

# Age distributions
AGE_MEAN_A, AGE_STD_A = 13, 2.5  # young carers tend older
AGE_MEAN_B, AGE_STD_B = 11, 3.0  # broader range

# Probability of parent data being missing
# Higher for single parents / young carers
P_MISSING_A = 0.25  # 25% of Group A have missing parent data
P_MISSING_B = 0.05  # 5% of Group B

print(f"Population:")
print(f"  Group A (young carers/underrepresented): n={N_A}")
print(f"  Group B (majority): n={N_B}")
print(f"  Base rate P_A={P_A}, P_B={P_B}")
print(f"  Parent missing: {P_MISSING_A*100}% Group A, "
      f"{P_MISSING_B*100}% Group B")
print()

# ─────────────────────────────────────────
# 3. ITEM GENERATION FUNCTIONS
# Generate raw questionnaire responses
# ─────────────────────────────────────────

def generate_sdq_items(n, severity_mean, severity_std,
                        underreport_factor=1.0):
    """
    Generate SDQ item responses (0/1/2) for n patients.
    severity_mean: average latent severity (0-10)
    underreport_factor: <1.0 means patient underreports
    """
    items = np.zeros((n, len(SDQ_ITEMS)), dtype=int)

    for i in range(n):
        # Latent severity for this patient
        severity = np.clip(
            np.random.normal(severity_mean, severity_std), 0, 10
        )

        # Apply underreporting
        reported_severity = severity * underreport_factor

        # Generate each item based on severity
        for j in range(len(SDQ_ITEMS)):
            # Base probability of scoring 1 or 2
            p_endorse = np.clip(reported_severity / 10, 0.05, 0.95)

            # Generate ordinal response
            r = np.random.random()
            if r < (1 - p_endorse):
                items[i, j] = 0
            elif r < (1 - p_endorse * 0.4):
                items[i, j] = 1
            else:
                items[i, j] = 2

    return items


def generate_rcads_items(n, anxiety_mean, depression_mean,
                          severity_std, underreport_factor=1.0):
    """
    Generate RCADS item responses (0/1/2/3) for n patients.

    Different means for anxiety and depression subscales.
    underreport_factor: key bias mechanism
    """
    items = np.zeros((n, len(RCADS_ITEMS)), dtype=int)

    # Which items belong to depression vs anxiety
    depression_items = set(
        [i-1 for i in RCADS_SUBSCALES['depression']]
    )

    for i in range(n):
        anx = np.clip(
            np.random.normal(anxiety_mean, severity_std), 0, 3
        )
        dep = np.clip(
            np.random.normal(depression_mean, severity_std), 0, 3
        )

        # Apply underreporting
        anx_rep = anx * underreport_factor
        dep_rep = dep * underreport_factor

        for j in range(len(RCADS_ITEMS)):
            if j in depression_items:
                base_score = dep_rep
            else:
                base_score = anx_rep

            # Generate 0/1/2/3 response
            p = np.clip(base_score / 3, 0.05, 0.95)
            r = np.random.random()
            if r < (1 - p):
                items[i, j] = 0
            elif r < (1 - p * 0.5):
                items[i, j] = 1
            elif r < (1 - p * 0.2):
                items[i, j] = 2
            else:
                items[i, j] = 3

    return items


def generate_impact_items(n, severity_mean,
                           underreport_factor=1.0):
    """
    Generate impact supplement responses (0/1/2/3).
    """
    items = np.zeros((n, len(IMPACT_ITEMS_PARENT)), dtype=int)

    for i in range(n):
        sev = np.clip(
            np.random.normal(severity_mean, 0.8), 0, 3
        ) * underreport_factor

        # Duration is independent of underreporting
        # Young carers often have longer duration
        # but this gets missed
        duration = np.random.choice([0, 1, 2, 3],
                                     p=[0.1, 0.2, 0.3, 0.4])

        items[i, 0] = int(np.clip(np.round(sev), 0, 3))
        items[i, 1] = duration
        for k in range(2, len(IMPACT_ITEMS_PARENT)):
            val = np.clip(
                np.random.normal(sev * 0.8, 0.5), 0, 3
            )
            items[i, k] = int(np.round(val))

    return items


# ─────────────────────────────────────────
# 4. SCORING FUNCTIONS
# Convert raw items to validated subscale scores
# ─────────────────────────────────────────

def score_sdq(items):
    """
    Score SDQ items into subscale scores.

    Reversed items: 7, 11, 14, 21, 25 (1-indexed)
    Score = sum of 5 items per subscale (0-10)
    Total difficulties = emotional + conduct +
                         hyperactivity + peer (NOT prosocial)
    """
    n = items.shape[0]
    scores = {}

    # Apply reversals (convert to 0-indexed)
    items_scored = items.copy()
    for rev_item in SDQ_REVERSED:
        idx = rev_item - 1
        items_scored[:, idx] = 2 - items_scored[:, idx]

    for subscale, item_nums in SDQ_SUBSCALES.items():
        indices = [i - 1 for i in item_nums]
        scores[f"sdq_{subscale}"] = items_scored[:, indices].sum(axis=1)

    # Total difficulties (sum of 4 subscales, not prosocial)
    scores["sdq_total"] = (
        scores["sdq_emotional"] +
        scores["sdq_conduct"] +
        scores["sdq_hyperactivity"] +
        scores["sdq_peer"]
    )

    return scores


def score_rcads(items):
    """
    Score RCADS items into subscale scores.
    Each item scored 0-3.
    Subscale score = sum of items in subscale.
    """
    scores = {}

    for subscale, item_nums in RCADS_SUBSCALES.items():
        indices = [i - 1 for i in item_nums]
        scores[f"rcads_{subscale}"] = items[:, indices].sum(axis=1)

    # Total anxiety + depression
    scores["rcads_total_anxiety"] = (
        scores["rcads_separation_anxiety"] +
        scores["rcads_social_phobia"] +
        scores["rcads_gad"] +
        scores["rcads_panic"] +
        scores["rcads_ocd"]
    )
    scores["rcads_total"] = (
        scores["rcads_total_anxiety"] +
        scores["rcads_depression"]
    )

    return scores


# ─────────────────────────────────────────
# 5. GENERATE GROUP A (Young carers)
# ─────────────────────────────────────────

print("Generating Group A (young carers/underrepresented)...")

# Group A parameters
# Higher true severity but lower reported severity
# due to underreporting mechanism
TRUE_SEVERITY_A = 6.5   # genuine clinical severity (hidden)
REPORT_FACTOR_CHILD_A = 0.40 #0.55 #0.65  # child underreports significantly
REPORT_FACTOR_PARENT_A = 0.45 #0.60 #0.75  # parent also underreports

# Ages
ages_A = np.clip(
    np.random.normal(AGE_MEAN_A, AGE_STD_A, N_A),
    5, 17
).astype(int)

# Gender (0=male, 1=female, 2=other)
gender_A = np.random.choice([0, 1, 2], N_A, p=[0.40, 0.55, 0.05])

# Generate SDQ - Child perspective (underreporting)
sdq_child_A = generate_sdq_items(
    N_A, TRUE_SEVERITY_A, 1.5,
    underreport_factor=REPORT_FACTOR_CHILD_A
)

# Generate RCADS - Child perspective (underreporting)
rcads_child_A = generate_rcads_items(
    N_A, anxiety_mean=1.8, depression_mean=1.5,
    severity_std=0.6,
    underreport_factor=REPORT_FACTOR_CHILD_A
)

# Impact supplement - Child
impact_child_A = generate_impact_items(
    N_A, severity_mean=1.8,
    underreport_factor=REPORT_FACTOR_CHILD_A
)

# Parent data - may be missing for young carers
parent_missing_A = np.random.binomial(1, P_MISSING_A, N_A)

# Generate parent data where available
sdq_parent_A = generate_sdq_items(
    N_A, TRUE_SEVERITY_A, 1.5,
    underreport_factor=REPORT_FACTOR_PARENT_A
)
rcads_parent_A = generate_rcads_items(
    N_A, anxiety_mean=1.8, depression_mean=1.5,
    severity_std=0.6,
    underreport_factor=REPORT_FACTOR_PARENT_A
)
impact_parent_A = generate_impact_items(
    N_A, severity_mean=1.8,
    underreport_factor=REPORT_FACTOR_PARENT_A
)

# Set parent data to NaN where missing
sdq_parent_A = sdq_parent_A.astype(float)
rcads_parent_A = rcads_parent_A.astype(float)
impact_parent_A = impact_parent_A.astype(float)

for i in range(N_A):
    if parent_missing_A[i] == 1:
        sdq_parent_A[i, :] = np.nan
        rcads_parent_A[i, :] = np.nan
        impact_parent_A[i, :] = np.nan

# True clinical outcome (independent of reporting)
true_outcome_A = np.random.binomial(1, P_A, N_A)

print(f"  Generated {N_A} Group A patients")
print(f"  Parent data missing: {parent_missing_A.sum()} "
      f"({parent_missing_A.mean()*100:.1f}%)")
print(f"  True positive rate: {true_outcome_A.mean():.3f}")

# ─────────────────────────────────────────
# 6. GENERATE GROUP B (Majority)
# ─────────────────────────────────────────

print("\nGenerating Group B (majority)...")

TRUE_SEVERITY_B = 6.0   # similar true severity
REPORT_FACTOR_CHILD_B = 0.95 #0.90  # more accurate reporting
REPORT_FACTOR_PARENT_B = 0.95 #0.92

ages_B = np.clip(
    np.random.normal(AGE_MEAN_B, AGE_STD_B, N_B),
    5, 17
).astype(int)

gender_B = np.random.choice([0, 1, 2], N_B, p=[0.48, 0.49, 0.03])

# SDQ - Child
sdq_child_B = generate_sdq_items(
    N_B, TRUE_SEVERITY_B, 1.5,
    underreport_factor=REPORT_FACTOR_CHILD_B
)

# RCADS - Child
rcads_child_B = generate_rcads_items(
    N_B, anxiety_mean=1.8, depression_mean=1.5,
    severity_std=0.6,
    underreport_factor=REPORT_FACTOR_CHILD_B
)

# Impact - Child
impact_child_B = generate_impact_items(
    N_B, severity_mean=2.0,
    underreport_factor=REPORT_FACTOR_CHILD_B
)

# Parent data - rarely missing
parent_missing_B = np.random.binomial(1, P_MISSING_B, N_B)

sdq_parent_B = generate_sdq_items(
    N_B, TRUE_SEVERITY_B, 1.5,
    underreport_factor=REPORT_FACTOR_PARENT_B
)
rcads_parent_B = generate_rcads_items(
    N_B, anxiety_mean=1.8, depression_mean=1.5,
    severity_std=0.6,
    underreport_factor=REPORT_FACTOR_PARENT_B
)
impact_parent_B = generate_impact_items(
    N_B, severity_mean=2.0,
    underreport_factor=REPORT_FACTOR_PARENT_B
)

sdq_parent_B = sdq_parent_B.astype(float)
rcads_parent_B = rcads_parent_B.astype(float)
impact_parent_B = impact_parent_B.astype(float)

for i in range(N_B):
    if parent_missing_B[i] == 1:
        sdq_parent_B[i, :] = np.nan
        rcads_parent_B[i, :] = np.nan
        impact_parent_B[i, :] = np.nan

true_outcome_B = np.random.binomial(1, P_B, N_B)

print(f"  Generated {N_B} Group B patients")
print(f"  Parent data missing: {parent_missing_B.sum()} "
      f"({parent_missing_B.mean()*100:.1f}%)")
print(f"  True positive rate: {true_outcome_B.mean():.3f}")

# ─────────────────────────────────────────
# 7. SCORE RAW ITEMS INTO SUBSCALES
# ─────────────────────────────────────────

print("\nScoring raw items into validated subscales...")

def score_group(sdq_c, rcads_c, impact_c,
                sdq_p, rcads_p, impact_p,
                parent_missing, ages, genders,
                true_outcomes, group_label, n):
    """Score all items and build feature dataframe."""

    records = []

    for i in range(n):
        record = {
            'patient_id': f"{group_label}_{i:04d}",
            'group': group_label,
            'age': ages[i],
            'gender': genders[i],
            'true_outcome': true_outcomes[i],
            'parent_data_missing': int(parent_missing[i])
        }

        # ── Score SDQ Child ──
        sdq_c_items = sdq_c[i:i+1]
        sdq_c_scores = score_sdq(sdq_c_items)
        for k, v in sdq_c_scores.items():
            record[f"{k}_c"] = int(v[0])

        # ── Score SDQ Parent ──
        if parent_missing[i] == 0:
            sdq_p_items = sdq_p[i:i+1]
            sdq_p_scores = score_sdq(sdq_p_items)
            for k, v in sdq_p_scores.items():
                record[f"{k}_p"] = int(v[0])
        else:
            for k in score_sdq(sdq_c[0:1]).keys():
                record[f"{k}_p"] = np.nan

        # ── Score RCADS Child ──
        rcads_c_items = rcads_c[i:i+1]
        rcads_c_scores = score_rcads(rcads_c_items)
        for k, v in rcads_c_scores.items():
            record[f"{k}_c"] = int(v[0])

        # ── Score RCADS Parent ──
        if parent_missing[i] == 0:
            rcads_p_items = rcads_p[i:i+1]
            rcads_p_scores = score_rcads(rcads_p_items)
            for k, v in rcads_p_scores.items():
                record[f"{k}_p"] = int(v[0])
        else:
            for k in score_rcads(rcads_c[0:1]).keys():
                record[f"{k}_p"] = np.nan

        # ── Impact Supplement Child ──
        for j, item_name in enumerate(IMPACT_ITEMS_CHILD):
            record[f"impact_{item_name}_c"] = int(impact_c[i, j])

        # ── Impact Supplement Parent ──
        if parent_missing[i] == 0:
            for j, item_name in enumerate(IMPACT_ITEMS_PARENT):
                record[f"impact_{item_name}_p"] = int(impact_p[i, j])
        else:
            for item_name in IMPACT_ITEMS_PARENT:
                record[f"impact_{item_name}_p"] = np.nan

        # ── Derived Features ──

        # SDQ discrepancy (key bias indicator)
        if parent_missing[i] == 0:
            record['sdq_discrepancy'] = abs(
                record['sdq_total_c'] - record['sdq_total_p']
            )
        else:
            record['sdq_discrepancy'] = np.nan

        # Impact discrepancy
        if parent_missing[i] == 0:
            record['impact_discrepancy'] = abs(
                record['impact_overall_severity_c'] -
                record['impact_overall_severity_p']
            )
        else:
            record['impact_discrepancy'] = np.nan

        # Duration in numeric months (midpoint of range)
        duration_map = {0: 0.5, 1: 3, 2: 9, 3: 18}
        record['duration_months_c'] = duration_map[
            int(impact_c[i, 1])
        ]
        if parent_missing[i] == 0:
            record['duration_months_p'] = duration_map[
                int(impact_p[i, 1])
            ]
        else:
            record['duration_months_p'] = np.nan

        records.append(record)

    return pd.DataFrame(records)


df_A = score_group(
    sdq_child_A, rcads_child_A, impact_child_A,
    sdq_parent_A, rcads_parent_A, impact_parent_A,
    parent_missing_A, ages_A, gender_A,
    true_outcome_A, 'A', N_A
)

df_B = score_group(
    sdq_child_B, rcads_child_B, impact_child_B,
    sdq_parent_B, rcads_parent_B, impact_parent_B,
    parent_missing_B, ages_B, gender_B,
    true_outcome_B, 'B', N_B
)

df = pd.concat([df_A, df_B], ignore_index=True)

print(f"  Scoring complete")
print(f"  Total patients: {len(df)}")
print(f"  Total features: {len(df.columns)}")

# ─────────────────────────────────────────
# 8. SUMMARY STATISTICS
# ─────────────────────────────────────────

print()
print("=" * 65)
print("SUMMARY STATISTICS")
print("=" * 65)

print(f"""
Group A (young carers):
  n = {N_A}
  Age: {ages_A.mean():.1f} ± {ages_A.std():.1f}
  Parent missing: {parent_missing_A.sum()} ({parent_missing_A.mean()*100:.1f}%)
  SDQ Total (child): {df_A['sdq_total_c'].mean():.1f} ± {df_A['sdq_total_c'].std():.1f}
  SDQ Total (parent): {df_A['sdq_total_p'].mean():.1f} ± {df_A['sdq_total_p'].std():.1f}
  RCADS Total (child): {df_A['rcads_total_c'].mean():.1f}
  True positive rate: {true_outcome_A.mean():.3f}

Group B (majority):
  n = {N_B}
  Age: {ages_B.mean():.1f} ± {ages_B.std():.1f}
  Parent missing: {parent_missing_B.sum()} ({parent_missing_B.mean()*100:.1f}%)
  SDQ Total (child): {df_B['sdq_total_c'].mean():.1f} ± {df_B['sdq_total_c'].std():.1f}
  SDQ Total (parent): {df_B['sdq_total_p'].mean():.1f} ± {df_B['sdq_total_p'].std():.1f}
  RCADS Total (child): {df_B['rcads_total_c'].mean():.1f}
  True positive rate: {true_outcome_B.mean():.3f}

Key bias indicator - SDQ discrepancy:
  Group A mean: {df_A['sdq_discrepancy'].mean():.2f}
  Group B mean: {df_B['sdq_discrepancy'].mean():.2f}
  (lower discrepancy for Group A = both perspectives
   underreport = harder for model to detect need)
""")

# ─────────────────────────────────────────
# 9. SAVE DATASETS
# ─────────────────────────────────────────

os.makedirs('data', exist_ok=True)

# Full dataset with all features
df.to_csv('data/synthetic_camhs_data.csv', index=False)
print(f"Full dataset saved: data/synthetic_camhs_data.csv")
print(f"Shape: {df.shape}")

# Also save raw items separately for transparency
raw_sdq_child = pd.DataFrame(
    np.vstack([sdq_child_A, sdq_child_B]),
    columns=[f"sdq_child_{item}" for item in SDQ_ITEMS]
)
raw_sdq_child['patient_id'] = df['patient_id'].values
raw_sdq_child['group'] = df['group'].values

raw_rcads_child = pd.DataFrame(
    np.vstack([rcads_child_A, rcads_child_B]),
    columns=[f"rcads_child_{item}" for item in RCADS_ITEMS]
)
raw_rcads_child['patient_id'] = df['patient_id'].values

raw_sdq_child.to_csv('data/raw_sdq_child_items.csv', index=False)
raw_rcads_child.to_csv('data/raw_rcads_child_items.csv', index=False)

print(f"Raw SDQ items saved: data/raw_sdq_child_items.csv")
print(f"Raw RCADS items saved: data/raw_rcads_child_items.csv")

print()
print("=" * 65)
print("DATA GENERATION COMPLETE")
print("=" * 65)
print(f"""
Pipeline demonstrated:
  1. Defined all {len(SDQ_ITEMS)} SDQ items with exact clinical wording
  2. Defined all {len(RCADS_ITEMS)} RCADS items with exact clinical wording
  3. Generated item-level responses with clinical distributions
  4. Applied underreporting bias (Group A factor: {REPORT_FACTOR_CHILD_A})
  5. Handled missing parent data ({P_MISSING_A*100}% Group A)
  6. Scored items into validated subscales
  7. Computed discrepancy features
  8. Generated true clinical outcome independently


""")