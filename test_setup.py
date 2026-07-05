"""
test_setup.py

Quick test to verify src/ modules load correctly
and all functions work as expected.

Run: python test_setup.py
"""

import warnings
warnings.filterwarnings('ignore')

print("Testing src/ module imports...")
print()

# ─────────────────────────────────────────
# TEST 1: Config
# ─────────────────────────────────────────
try:
    from src.config import (
        FEATURES, TARGET, BASELINE_PARAMS,
        GRID_PARAMS, BAYES_PARAMS, C_FN, C_FP,
        UNIFORM_THRESHOLD, DATA_PATH
    )
    print(f"[OK] config.py loaded")
    print(f"  Features: {FEATURES}")
    print(f"  Threshold: {UNIFORM_THRESHOLD}")
    print(f"  c_FN={C_FN}, c_FP={C_FP}")
except Exception as e:
    print(f"[FAIL] config.py FAILED: {e}")

print()

# ─────────────────────────────────────────
# TEST 2: Data Loader
# ─────────────────────────────────────────
try:
    from src.data_loader import load_and_split
    (df, X_train, X_test, y_train, y_test,
     groups, X_A, y_A, X_B, y_B) = load_and_split()
    print(f"[OK] data_loader.py loaded")
    print(f"  Total patients: {len(df)}")
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  Group A test: {len(X_A)}, Group B test: {len(X_B)}")
except Exception as e:
    print(f"[FAIL] data_loader.py FAILED: {e}")

print()

# ─────────────────────────────────────────
# TEST 3: Models
# ─────────────────────────────────────────
try:
    from src.models import (
        get_baseline_model, get_grid_model,
        get_bayes_model, train_model
    )
    rf_baseline = train_model(get_baseline_model(), X_train, y_train)
    rf_grid = train_model(get_grid_model(), X_train, y_train)
    rf_bayes = train_model(get_bayes_model(), X_train, y_train)
    print(f"[OK] models.py loaded")
    print(f"  Baseline trained: {rf_baseline.n_estimators} trees")
    print(f"  Grid trained: max_features={rf_grid.max_features}")
    print(f"  Bayes trained: max_samples={rf_bayes.max_samples}")
except Exception as e:
    print(f"[FAIL] models.py FAILED: {e}")

print()

# ─────────────────────────────────────────
# TEST 4: Metrics
# ─────────────────────────────────────────
try:
    from src.metrics import (
        compute_metrics_at_threshold,
        compute_tpr_gap,
        find_optimal_threshold,
        evaluate_fairness_criteria
    )

    # Test metrics computation
    m_A = compute_metrics_at_threshold(X_A, y_A, rf_baseline)
    m_B = compute_metrics_at_threshold(X_B, y_B, rf_baseline)
    gap, tpr_a, tpr_b = compute_tpr_gap(
        rf_baseline, X_A, y_A, X_B, y_B
    )

    print(f"[OK] metrics.py loaded")
    print(f"  Group A TPR: {m_A['TPR']:.3f}")
    print(f"  Group B TPR: {m_B['TPR']:.3f}")
    print(f"  TPR Gap: {gap:.3f}")
except Exception as e:
    print(f"[FAIL] metrics.py FAILED: {e}")

print()

# ─────────────────────────────────────────
# TEST 5: Permutation Importance
# ─────────────────────────────────────────
try:
    from src.permutation import (
        permutation_importance_all,
        group_permutation_importance
    )
    from src.config import REF_LENGTH_IDX

    # Test on small subset for speed
    import numpy as np
    X_test_arr = X_test.values
    y_test_arr = y_test.values

    # Just test referral_length importance
    from src.permutation import permutation_importance_single
    I_j, diffs = permutation_importance_single(
        rf_baseline, X_test_arr, y_test_arr,
        REF_LENGTH_IDX, n_trees=10
    )
    print(f"[OK] permutation.py loaded")
    print(f"  referral_length I_j (10 trees): {I_j:.4f}")
except Exception as e:
    print(f"[FAIL] permutation.py FAILED: {e}")

print()

# ─────────────────────────────────────────
# TEST 6: Utils
# ─────────────────────────────────────────
try:
    from src.utils import (
        ensure_output_dir, print_header,
        check_satisfied, format_comparison_row
    )
    ensure_output_dir()
    print(f"[OK] utils.py loaded")
    print(f"  check_satisfied(0.8, 0.82): "
          f"{check_satisfied(0.8, 0.82)}")
    print(f"  check_satisfied(0.3, 0.9):  "
          f"{check_satisfied(0.3, 0.9)}")
except Exception as e:
    print(f"[FAIL] utils.py FAILED: {e}")

print()

# ─────────────────────────────────────────
# TEST 7: Full fairness evaluation
# ─────────────────────────────────────────
try:
    fairness = evaluate_fairness_criteria(
        rf_baseline, X_A, y_A, X_B, y_B
    )
    print(f"[OK] Full fairness evaluation working")
    print(f"  Demographic Parity satisfied: "
          f"{fairness['demographic_parity']['satisfied']}")
    print(f"  Equalised Odds TPR satisfied: "
          f"{fairness['equalised_odds_tpr']['satisfied']}")
    print(f"  Calibration satisfied: "
          f"{fairness['calibration']['satisfied']}")
except Exception as e:
    print(f"[FAIL] Fairness evaluation FAILED: {e}")

print()
print("=" * 50)
print("Setup test complete!")
print("If all [OK] - ready to update scripts to use src/")
print("If any [FAIL] - fix that module before proceeding")
print("=" * 50)