import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from src.config import C_FN, C_FP, UNIFORM_THRESHOLD, FAIRNESS_TOLERANCE


def classification_error(y_true, y_pred):
    return np.mean(y_true != y_pred)


def compute_confusion(decisions, true_outcomes):
    D = np.array(decisions)
    T = np.array(true_outcomes)
    TP = ((D == 1) & (T == 1)).sum()
    FP = ((D == 1) & (T == 0)).sum()
    TN = ((D == 0) & (T == 0)).sum()
    FN = ((D == 0) & (T == 1)).sum()
    return TP, FP, TN, FN


def compute_rates(TP, FP, TN, FN):
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
    PPV = TP / (TP + FP) if (TP + FP) > 0 else 0
    return TPR, FPR, FNR, TNR, PPV


def compute_metrics_at_threshold(X_group, y_group,
                                  rf, threshold=UNIFORM_THRESHOLD):
    prob = rf.predict_proba(X_group)[:, 1]
    decisions = (prob >= threshold).astype(int)
    true_outcomes = np.array(y_group)

    TP, FP, TN, FN = compute_confusion(decisions, true_outcomes)
    TPR, FPR, FNR, TNR, PPV = compute_rates(TP, FP, TN, FN)

    return {
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'TPR': TPR, 'FPR': FPR, 'FNR': FNR,
        'TNR': TNR, 'PPV': PPV,
        'selection_rate': decisions.mean(),
        'base_rate': true_outcomes.mean(),
        'n': len(true_outcomes)
    }


def compute_tpr_gap(rf, X_A, y_A, X_B, y_B,
                     threshold=UNIFORM_THRESHOLD):
    m_A = compute_metrics_at_threshold(X_A, y_A, rf, threshold)
    m_B = compute_metrics_at_threshold(X_B, y_B, rf, threshold)
    gap = abs(m_B['TPR'] - m_A['TPR'])
    return gap, m_A['TPR'], m_B['TPR']


def expected_loss_at_threshold(rf, X_group, y_group,
                                threshold, c_fn=C_FN, c_fp=C_FP):
    m = compute_metrics_at_threshold(X_group, y_group, rf, threshold)
    P = m['base_rate']
    return c_fn * m['FNR'] * P + c_fp * m['FPR'] * (1 - P)


def compute_roc(rf, X_group, y_group):
    prob = rf.predict_proba(X_group)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_group, prob)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, thresholds, auc_score


def find_optimal_threshold(rf, X_group, y_group,
                            c_fn=C_FN, c_fp=C_FP):
    fpr, tpr, thresholds = roc_curve(y_group,
        rf.predict_proba(X_group)[:, 1])
    P = np.array(y_group).mean()
    fnr = 1 - tpr
    losses = c_fn * fnr * P + c_fp * fpr * (1 - P)
    idx = np.argmin(losses)
    return thresholds[idx], losses[idx], tpr[idx], fpr[idx]


def evaluate_fairness_criteria(rf, X_A, y_A, X_B, y_B,
                                threshold=UNIFORM_THRESHOLD,
                                tol=FAIRNESS_TOLERANCE):
    m_A = compute_metrics_at_threshold(X_A, y_A, rf, threshold)
    m_B = compute_metrics_at_threshold(X_B, y_B, rf, threshold)

    return {
        'demographic_parity': {
            'A': m_A['selection_rate'],
            'B': m_B['selection_rate'],
            'satisfied': abs(m_A['selection_rate'] -
                            m_B['selection_rate']) < tol
        },
        'equalised_odds_tpr': {
            'A': m_A['TPR'],
            'B': m_B['TPR'],
            'satisfied': abs(m_A['TPR'] - m_B['TPR']) < tol
        },
        'equalised_odds_fpr': {
            'A': m_A['FPR'],
            'B': m_B['FPR'],
            'satisfied': abs(m_A['FPR'] - m_B['FPR']) < tol
        },
        'calibration': {
            'A': m_A['PPV'],
            'B': m_B['PPV'],
            'satisfied': abs(m_A['PPV'] - m_B['PPV']) < tol
        },
        'base_rates': {
            'A': m_A['base_rate'],
            'B': m_B['base_rate']
        }
    }