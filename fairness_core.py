import numpy as np
from sklearn.metrics import roc_curve, auc
from dataclasses import dataclass
 
 
# ─────────────────────────────────────────
# 1. SYNTHETIC SCORE GENERATION
# Generates realistic risk-score distributions per group
# from Beta distributions, calibrated to base rate P_G.
# ─────────────────────────────────────────
 
def generate_group_scores(n, base_rate, quality, seed=0):
    """
    Generate synthetic predicted probabilities and true labels
    for one demographic group.
    """
    rng = np.random.default_rng(seed)
    labels = rng.binomial(1, base_rate, n)
 
    # Positive-class scores: Beta shifted toward 1, scaled by quality
    # Negative-class scores: Beta shifted toward 0
    alpha_pos = 2 + 4 * quality
    beta_pos = 2
    alpha_neg = 2
    beta_neg = 2 + 4 * quality
 
    scores = np.zeros(n)
    pos_mask = labels == 1
    scores[pos_mask] = rng.beta(alpha_pos, beta_pos, pos_mask.sum())
    scores[~pos_mask] = rng.beta(alpha_neg, beta_neg, (~pos_mask).sum())
 
    return scores, labels
 
 
# ─────────────────────────────────────────
# 2. CHOULDECHOVA'S IMPOSSIBILITY RESULT
# Derived from Bayes' theorem (see paper Section 3.3)
# ─────────────────────────────────────────
 
def ppv_from_bayes(fnr, fpr, base_rate):
    """
    PPV_G = (1-FNR)*P_G / [(1-FNR)*P_G + FPR*(1-P_G)]
    Works for scalars or numpy arrays.
    """
    fnr = np.asarray(fnr, dtype=float)
    fpr = np.asarray(fpr, dtype=float)
    base_rate = np.asarray(base_rate, dtype=float)
 
    numerator = (1 - fnr) * base_rate
    denominator = (1 - fnr) * base_rate + fpr * (1 - base_rate)
    return np.divide(numerator, denominator,
                      out=np.zeros_like(numerator, dtype=float),
                      where=denominator > 0)
 
 
def chouldechova_gap(base_rate_a, base_rate_b, fnr=0.1, fpr=0.1):
    """
    Numerical confirmation of Chouldechova's (2017) impossibility
    result: even under perfectly equalised odds, PPV differs
    whenever base rates differ.
    """
    ppv_a = ppv_from_bayes(fnr, fpr, base_rate_a)
    ppv_b = ppv_from_bayes(fnr, fpr, base_rate_b)
    return float(ppv_a), float(ppv_b), float(abs(ppv_b - ppv_a))
 
 
def minimum_ppv_gap(base_rate_a, base_rate_b, n_points=60):
    """
    Sweep across all FNR, FPR in [0.05, 0.5] and confirm the
    PPV gap is never zero when base rates differ. Returns the
    minimum PPV gap found across the entire sweep.
    """
    fnr_range = np.linspace(0.05, 0.5, n_points)
    fpr_range = np.linspace(0.05, 0.5, n_points)
    FNR, FPR = np.meshgrid(fnr_range, fpr_range)
 
    ppv_a = ppv_from_bayes(FNR, FPR, base_rate_a)
    ppv_b = ppv_from_bayes(FNR, FPR, base_rate_b)
    gap = np.abs(ppv_b - ppv_a)
    return float(gap.min())
 
 
# ─────────────────────────────────────────
# 3. ROC CURVE + BAYESIAN LOSS
# ─────────────────────────────────────────
 
@dataclass
class GroupResult:
    group: str
    base_rate: float
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc: float
    optimal_threshold: float
    optimal_tpr: float
    optimal_fpr: float
    optimal_slope: float
    target_slope: float
    min_expected_loss: float
 
 
def expected_loss(fpr, tpr, base_rate, c_fn, c_fp):
    """E[L](tau) = c_FN * FNR * P + c_FP * FPR * (1-P)"""
    fnr = 1 - tpr
    return c_fn * fnr * base_rate + c_fp * fpr * (1 - base_rate)
 
 
def optimal_threshold_search(scores, labels, base_rate, c_fn, c_fp):
    """
    Find the threshold minimising expected Bayesian loss,
    implementing Theorem 1 :
        dTPR/dFPR |_tau* = c_FP*(1-P) / (c_FN*P)
    """
    fpr, tpr, thresh = roc_curve(labels, scores)
    thresh = np.clip(thresh, 0, 1)
 
    loss = expected_loss(fpr, tpr, base_rate, c_fn, c_fp)
    idx = int(np.argmin(loss))
 
    roc_auc = auc(fpr, tpr)
    target_slope = (c_fp * (1 - base_rate)) / (c_fn * base_rate)
 
    if 0 < idx < len(fpr) - 1:
        dfpr = fpr[idx + 1] - fpr[idx - 1]
        dtpr = tpr[idx + 1] - tpr[idx - 1]
        empirical_slope = dtpr / dfpr if abs(dfpr) > 1e-9 else np.nan
    else:
        empirical_slope = np.nan
 
    return GroupResult(
        group="", base_rate=base_rate,
        fpr=fpr, tpr=tpr, thresholds=thresh, auc=roc_auc,
        optimal_threshold=float(thresh[idx]),
        optimal_tpr=float(tpr[idx]),
        optimal_fpr=float(fpr[idx]),
        optimal_slope=float(empirical_slope),
        target_slope=float(target_slope),
        min_expected_loss=float(loss[idx]),
    )
 
 
def metrics_at_threshold(scores, labels, tau):
    """Compute TPR, FPR, FNR, PPV, selection rate at a threshold."""
    d = (scores >= tau).astype(int)
    t = np.asarray(labels)
 
    tp = int(((d == 1) & (t == 1)).sum())
    fp = int(((d == 1) & (t == 0)).sum())
    tn = int(((d == 0) & (t == 0)).sum())
    fn = int(((d == 0) & (t == 1)).sum())
 
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = 1 - tpr
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    selection_rate = d.mean()
 
    return dict(tpr=tpr, fpr=fpr, fnr=fnr, ppv=ppv,
                selection_rate=selection_rate,
                tp=tp, fp=fp, tn=tn, fn=fn)
 
 
# ─────────────────────────────────────────
# 4. FULL PIPELINE
# ─────────────────────────────────────────
 
def run_full_analysis(pa, pb, cfn, cfp, ma,
                       n_a=300, n_b=700,
                       quality_a=0.45, quality_b=0.55,
                       uniform_threshold=0.4, seed=42):
    """
    End-to-end pipeline using synthetic data distributions.
    """
    cfn_a = cfn * ma
 
    scores_a, labels_a = generate_group_scores(n_a, pa, quality_a, seed=seed)
    scores_b, labels_b = generate_group_scores(n_b, pb, quality_b, seed=seed + 1)
 
    result_a = optimal_threshold_search(scores_a, labels_a, pa, cfn_a, cfp)
    result_a.group = "A"
    result_b = optimal_threshold_search(scores_b, labels_b, pb, cfn, cfp)
    result_b.group = "B"
 
    m_a_uniform = metrics_at_threshold(scores_a, labels_a, uniform_threshold)
    m_b_uniform = metrics_at_threshold(scores_b, labels_b, uniform_threshold)
 
    m_a_opt = metrics_at_threshold(scores_a, labels_a, result_a.optimal_threshold)
    m_b_opt = metrics_at_threshold(scores_b, labels_b, result_b.optimal_threshold)
 
    ppv_a_theory, ppv_b_theory, ppv_gap_theory = chouldechova_gap(pa, pb)
    min_gap = minimum_ppv_gap(pa, pb)
 
    tpr_gap_uniform = abs(m_b_uniform["tpr"] - m_a_uniform["tpr"])
    tpr_gap_optimal = abs(m_b_opt["tpr"] - m_a_opt["tpr"])
 
    return {
        "result_a": result_a,
        "result_b": result_b,
        "m_a_uniform": m_a_uniform,
        "m_b_uniform": m_b_uniform,
        "m_a_opt": m_a_opt,
        "m_b_opt": m_b_opt,
        "tpr_gap_uniform": tpr_gap_uniform,
        "tpr_gap_optimal": tpr_gap_optimal,
        "ppv_a_theory": ppv_a_theory,
        "ppv_b_theory": ppv_b_theory,
        "ppv_gap_theory": ppv_gap_theory,
        "min_ppv_gap": min_gap,
        "cfn_a": cfn_a,
        "scores_a": scores_a, "labels_a": labels_a,
        "scores_b": scores_b, "labels_b": labels_b,
    }
 
 
# ─────────────────────────────────────────
# 5. CSV-BASED ANALYSIS (real uploaded data)
# ─────────────────────────────────────────
 
def run_analysis_from_dataframe(df, cfn, cfp, ma, uniform_threshold=0.4):
    """
    Same pipeline but using real uploaded data instead of synthetic generation.
    """
    df_a = df[df["group"] == "A"]
    df_b = df[df["group"] == "B"]
 
    scores_a = df_a["prob"].to_numpy()
    labels_a = df_a["true_label"].to_numpy()
    scores_b = df_b["prob"].to_numpy()
    labels_b = df_b["true_label"].to_numpy()
 
    pa = float(labels_a.mean())
    pb = float(labels_b.mean())
    cfn_a = cfn * ma
 
    result_a = optimal_threshold_search(scores_a, labels_a, pa, cfn_a, cfp)
    result_a.group = "A"
    result_b = optimal_threshold_search(scores_b, labels_b, pb, cfn, cfp)
    result_b.group = "B"
 
    m_a_uniform = metrics_at_threshold(scores_a, labels_a, uniform_threshold)
    m_b_uniform = metrics_at_threshold(scores_b, labels_b, uniform_threshold)
    m_a_opt = metrics_at_threshold(scores_a, labels_a, result_a.optimal_threshold)
    m_b_opt = metrics_at_threshold(scores_b, labels_b, result_b.optimal_threshold)
 
    ppv_a_theory, ppv_b_theory, ppv_gap_theory = chouldechova_gap(pa, pb)
    min_gap = minimum_ppv_gap(pa, pb)
 
    return {
        "pa": pa, "pb": pb,
        "n_a": len(scores_a), "n_b": len(scores_b),
        "result_a": result_a, "result_b": result_b,
        "m_a_uniform": m_a_uniform, "m_b_uniform": m_b_uniform,
        "m_a_opt": m_a_opt, "m_b_opt": m_b_opt,
        "tpr_gap_uniform": abs(m_b_uniform["tpr"] - m_a_uniform["tpr"]),
        "tpr_gap_optimal": abs(m_b_opt["tpr"] - m_a_opt["tpr"]),
        "ppv_a_theory": ppv_a_theory, "ppv_b_theory": ppv_b_theory,
        "ppv_gap_theory": ppv_gap_theory, "min_ppv_gap": min_gap,
        "cfn_a": cfn_a,
        "scores_a": scores_a, "labels_a": labels_a,
        "scores_b": scores_b, "labels_b": labels_b,
    }
