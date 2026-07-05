"""
Permutation importance:
    I_j = (1/T) * sum_{t=1}^{T} (E_t^(j) - E_t)
"""

import numpy as np
from src.metrics import classification_error
from src.config import FEATURES, N_PERM_TREES


def permutation_importance_single(rf, X_arr, y_arr,
                                   feature_idx,
                                   n_trees=N_PERM_TREES):
    differences = []

    for t, tree in enumerate(rf.estimators_[:n_trees]):
        # Original error E_t
        y_pred_orig = tree.predict(X_arr)
        E_t = classification_error(y_arr, y_pred_orig)

        # Shuffle feature j
        X_shuffled = X_arr.copy()
        np.random.seed(t)
        np.random.shuffle(X_shuffled[:, feature_idx])

        # Error after shuffling E_t^(j)
        y_pred_shuf = tree.predict(X_shuffled)
        E_t_j = classification_error(y_arr, y_pred_shuf)

        differences.append(E_t_j - E_t)

    I_j = np.mean(differences)
    return I_j, differences


def permutation_importance_all(rf, X_test, y_test,
                                feature_names=FEATURES,
                                n_trees=N_PERM_TREES):
    if hasattr(X_test, 'values'):
        X_arr = X_test.values
    else:
        X_arr = X_test

    y_arr = np.array(y_test)

    results = {}
    for j, feature in enumerate(feature_names):
        I_j, diffs = permutation_importance_single(
            rf, X_arr, y_arr, j, n_trees
        )
        results[feature] = {
            'I_j': I_j,
            'differences': diffs,
            'std': np.std(diffs)
        }

    return results


def group_permutation_importance(rf, X_group, y_group,
                                  feature_idx,
                                  n_trees=N_PERM_TREES):
    if hasattr(X_group, 'values'):
        X_arr = X_group.values
    else:
        X_arr = X_group

    y_arr = np.array(y_group)
    I_j, _ = permutation_importance_single(
        rf, X_arr, y_arr, feature_idx, n_trees
    )
    return I_j