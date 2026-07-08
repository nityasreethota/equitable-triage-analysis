import numpy as np
import pandas as pd


def engineer_camhs_features(df):
    """
    Transform raw CAMHS subscale scores into
    aggregated composite features.
    """

    engineered = pd.DataFrame()
    engineered['patient_id'] = df['patient_id']
    engineered['group'] = df['group']
    engineered['true_outcome'] = df['true_outcome']

    # ── Child perspective aggregates ──

    engineered['child_total_difficulties'] = df['sdq_total_c']

    engineered['child_prosocial'] = df['sdq_prosocial_c']

    engineered['child_total_anxiety'] = (
        df['rcads_separation_anxiety_c'] +
        df['rcads_social_phobia_c'] +
        df['rcads_gad_c'] +
        df['rcads_panic_c'] +
        df['rcads_ocd_c']
    )

    engineered['child_depression'] = df['rcads_depression_c']

    engineered['child_total_mental_health'] = df['rcads_total_c']

    # Child impact total (sum all impact items)
    child_impact_cols = [
        'impact_overall_severity_c',
        'impact_distress_c',
        'impact_home_life_c',
        'impact_friendships_c',
        'impact_classroom_learning_c',
        'impact_leisure_c'
    ]
    # Use available columns
    available_impact_c = [
        c for c in child_impact_cols if c in df.columns
    ]
    engineered['child_impact_total'] = df[available_impact_c].sum(axis=1)

    engineered['child_impact_severity'] = df['impact_overall_severity_c']
    engineered['child_duration'] = df['impact_duration_c']

    # ── Parent perspective aggregates ──
    # Fill -1 already done in data_loader for missing

    engineered['parent_total_difficulties'] = df['sdq_total_p'].fillna(-1)
    engineered['parent_total_mental_health'] = df['rcads_total_p'].fillna(-1)

    # Parent impact total
    parent_impact_cols = [
        'impact_overall_severity_p',
        'impact_family_burden_p'
    ]
    available_impact_p = [
        c for c in parent_impact_cols if c in df.columns
    ]
    engineered['parent_impact_total'] = df[available_impact_p].fillna(0).sum(axis=1)

    # ── Cross-perspective features ──

    engineered['parent_data_missing'] = df['parent_data_missing']

    # Perspective agreement (0=disagree, 1=agree)
    # Only meaningful when parent data present
    max_sdq = 40  # maximum SDQ total
    sdq_diff = np.abs(
        df['sdq_total_c'] - df['sdq_total_p'].fillna(df['sdq_total_c'])
    )
    engineered['perspective_agreement'] = 1 - (sdq_diff / max_sdq)

    # SDQ discrepancy (raw)
    engineered['sdq_discrepancy'] = df['sdq_discrepancy'].fillna(-1)

    # Underreporting index
    # High value = likely underreporting = model should
    # lower its threshold for this patient
    age_expected_sdq = df['age'].apply(
        lambda a: 12 if a < 11 else 15 if a < 14 else 13
    )
    child_underreport = np.clip(
        (age_expected_sdq - df['sdq_total_c']) / age_expected_sdq,
        0, 1
    )
    engineered['underreporting_index'] = (
        child_underreport * 0.5 +
        df['parent_data_missing'] * 0.3 +
        (1 - engineered['perspective_agreement']) * 0.2
    )

    # Overall severity - use best available perspective
    # If parent present: average of both
    # If parent missing: child report only (but flagged)
    parent_present = 1 - df['parent_data_missing']
    engineered['overall_severity'] = np.where(
        parent_present,
        (df['impact_overall_severity_c'] +
         df['impact_overall_severity_p'].fillna(0)) / 2,
        df['impact_overall_severity_c'] * 0.8
        # discount child-only by 20% acknowledging uncertainty
    )

    # Chronicity - duration in months
    engineered['chronicity'] = df['duration_months_c']

    # Functional impairment
    impairment_cols = [
        'impact_home_life_c',
        'impact_friendships_c',
        'impact_classroom_learning_c',
        'impact_leisure_c'
    ]
    available_imp = [c for c in impairment_cols if c in df.columns]
    if available_imp:
        engineered['functional_impairment'] = (
            df[available_imp].mean(axis=1)
        )
    else:
        engineered['functional_impairment'] = (
            df['impact_overall_severity_c']
        )

    # ── Demographics ──
    engineered['age'] = df['age']
    engineered['gender'] = df['gender']

    return engineered


def get_engineered_feature_names():
    """Return list of engineered feature names."""
    return [
        'child_total_difficulties',
        'child_prosocial',
        'child_total_anxiety',
        'child_depression',
        'child_total_mental_health',
        'child_impact_total',
        'child_impact_severity',
        'child_duration',
        'parent_total_difficulties',
        'parent_total_mental_health',
        'parent_impact_total',
        'parent_data_missing',
        'perspective_agreement',
        'sdq_discrepancy',
        'underreporting_index',
        'overall_severity',
        'chronicity',
        'functional_impairment',
        'age',
        'gender',
    ]