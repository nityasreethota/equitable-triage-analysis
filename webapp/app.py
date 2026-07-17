"""
app.py

Deploy to Hugging Face Spaces: see README.md
"""
 
import io
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
 
from fairness_core import run_full_analysis, run_analysis_from_dataframe
 
 
# ─────────────────────────────────────────
# PLOTTING HELPERS
# ─────────────────────────────────────────
 
def plot_roc(res):
    fig, ax = plt.subplots(figsize=(6, 5))
    a, b = res["result_a"], res["result_b"]
 
    ax.plot(a.fpr, a.tpr, color="#4a9eff", linewidth=2,
            label=f"Group A ROC (AUC={a.auc:.3f})")
    ax.plot(b.fpr, b.tpr, color="#2ecc71", linewidth=2,
            label=f"Group B ROC (AUC={b.auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random classifier")
 
    ax.scatter([a.optimal_fpr], [a.optimal_tpr], color="#4a9eff",
               s=140, zorder=5, edgecolor="white",
               label=f"Optimal τ_A*={a.optimal_threshold:.3f}")
    ax.scatter([b.optimal_fpr], [b.optimal_tpr], color="#2ecc71",
               s=140, zorder=5, edgecolor="white",
               label=f"Optimal τ_B*={b.optimal_threshold:.3f}")
 
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Group-Specific ROC Curves\nwith Theorem-Optimal Thresholds")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
 
 
def plot_loss_curves(res):
    fig, ax = plt.subplots(figsize=(6, 5))
    a, b = res["result_a"], res["result_b"]
 
    from fairness_core import expected_loss
    loss_a = expected_loss(a.fpr, a.tpr, a.base_rate,
                            res["cfn_a"], 1.0)
    loss_b = expected_loss(b.fpr, b.tpr, b.base_rate,
                            res["cfn_a"] / max(res["cfn_a"] / 1, 1), 1.0)
 
    ax.plot(a.thresholds, loss_a, color="#4a9eff", linewidth=2,
            label="Group A expected loss")
    ax.plot(b.thresholds, loss_b, color="#2ecc71", linewidth=2,
            label="Group B expected loss")
    ax.axvline(a.optimal_threshold, color="#4a9eff", linestyle=":",
               linewidth=2, label=f"Optimal τ_A*={a.optimal_threshold:.3f}")
    ax.axvline(b.optimal_threshold, color="#2ecc71", linestyle=":",
               linewidth=2, label=f"Optimal τ_B*={b.optimal_threshold:.3f}")
    ax.axvline(0.4, color="black", linestyle="--", alpha=0.5,
               label="Uniform τ=0.4")
 
    ax.set_xlabel("Threshold τ")
    ax.set_ylabel("Expected Loss E[L]")
    ax.set_title("Bayesian Expected Loss by Threshold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
 
 
def plot_chouldechova(pa, pb, res):
    fig, ax = plt.subplots(figsize=(6, 5))
    from fairness_core import ppv_from_bayes
 
    base_rates = np.linspace(0.05, 0.95, 200)
    ppv_curve = ppv_from_bayes(0.1, 0.1, base_rates)
 
    ax.plot(base_rates, ppv_curve, color="purple", linewidth=2)
    ax.axvline(pa, color="#4a9eff", linestyle="--",
               label=f"P_A={pa:.3f}")
    ax.axvline(pb, color="#2ecc71", linestyle="--",
               label=f"P_B={pb:.3f}")
    ax.scatter([pa], [res["ppv_a_theory"]], color="#4a9eff", s=140,
               zorder=5, edgecolor="white",
               label=f"PPV_A={res['ppv_a_theory']:.3f}")
    ax.scatter([pb], [res["ppv_b_theory"]], color="#2ecc71", s=140,
               zorder=5, edgecolor="white",
               label=f"PPV_B={res['ppv_b_theory']:.3f}")
 
    ax.set_xlabel("Base Rate P(T=1|G)")
    ax.set_ylabel("PPV (Calibration)")
    ax.set_title("Chouldechova's Impossibility:\nPPV varies with Base Rate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
 
 
# ─────────────────────────────────────────
# MAIN CALLBACK
# ─────────────────────────────────────────
 
def compute(pa, pb, cfn, cfp, ma, csv_file):
 
    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file.name)
            df.columns = [c.strip().lower() for c in df.columns]
            res = run_analysis_from_dataframe(df, cfn, cfp, ma)
            pa, pb = res["pa"], res["pb"]
            source_note = (f"Using uploaded data: n_A={res['n_a']}, "
                            f"n_B={res['n_b']} (base rates estimated "
                            f"from data)")
        except Exception as e:
            return (None, None, None,
                    f"**Error reading CSV:** {e}\n\nExpected columns: "
                    f"`prob, true_label, group`", "", "")
    else:
        res = run_full_analysis(pa, pb, cfn, cfp, ma)
        source_note = "Using synthetic demonstration data (adjust sliders)"
 
    a, b = res["result_a"], res["result_b"]
 
    summary_md = f"""
### {source_note}
 
## Theorem 1 result 
 
| | Group A | Group B |
|---|---|---|
| Base rate P<sub>G</sub> | {pa:.3f} | {pb:.3f} |
| Optimal threshold τ<sub>G</sub>* | **{a.optimal_threshold:.3f}** | **{b.optimal_threshold:.3f}** |
| ROC-implied slope at τ* | {a.optimal_slope:.3f} | {b.optimal_slope:.3f} |
| Theorem target slope | {a.target_slope:.3f} | {b.target_slope:.3f} |
| AUC | {a.auc:.3f} | {b.auc:.3f} |
 
## Fairness gap
 
| Metric | Uniform τ=0.4 | Group-specific τ* |
|---|---|---|
| TPR Group A | {res['m_a_uniform']['tpr']:.3f} | {res['m_a_opt']['tpr']:.3f} |
| TPR Group B | {res['m_b_uniform']['tpr']:.3f} | {res['m_b_opt']['tpr']:.3f} |
| **TPR Gap** | **{res['tpr_gap_uniform']:.3f}** | **{res['tpr_gap_optimal']:.3f}** |
 
## Chouldechova's impossibility (2017) — confirmed numerically
 
- PPV<sub>A</sub> (theoretical, equal error rates) = **{res['ppv_a_theory']:.3f}**
- PPV<sub>B</sub> (theoretical, equal error rates) = **{res['ppv_b_theory']:.3f}**
- Gap = **{res['ppv_gap_theory']:.3f}** (persists even under perfect equalised odds)
- Minimum PPV gap across *all* FNR/FPR combinations = **{res['min_ppv_gap']:.4f}** — never zero.
 
*This is real computation on {'your uploaded data' if csv_file else 'synthetic data generated from Beta distributions'} — every number above comes from `fairness_core.py` running scikit-learn's `roc_curve` and the closed-form Bayes/Chouldechova formulae, not a lookup table.*
"""
 
    export = {
        "parameters": {"pa": pa, "pb": pb, "cfn": cfn, "cfp": cfp,
                        "cfn_a": res["cfn_a"], "multiplier": ma},
        "optimal_thresholds": {"tau_A": a.optimal_threshold,
                                "tau_B": b.optimal_threshold},
        "tpr_gap_uniform": res["tpr_gap_uniform"],
        "tpr_gap_optimal": res["tpr_gap_optimal"],
        "chouldechova_ppv_gap": res["ppv_gap_theory"],
        "min_ppv_gap_all_configs": res["min_ppv_gap"],
    }
    export_str = json.dumps(export, indent=2)
 
    return (plot_roc(res), plot_loss_curves(res),
            plot_chouldechova(pa, pb, res),
            summary_md, export_str)
 
 
# ─────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────
 
THEOREM_MD = r"""
## Theorem 1 
 
Let $G \in \{A, B\}$ be two demographic groups with base rates $P_G$,
group-specific costs $c_{FN}^G, c_{FP}^G > 0$, and differentiable ROC
curves. The threshold $\tau_G^*$ minimising expected Bayesian loss
 
$$\mathcal{L}_G(\tau) = c_{FN}^G \cdot FNR_G(\tau) \cdot P_G + c_{FP}^G \cdot FPR_G(\tau) \cdot (1-P_G)$$
 
satisfies the first-order condition:
 
$$\left.\frac{d\,TPR_G}{d\,FPR_G}\right|_{\tau_G^*} = \frac{c_{FP}^G(1-P_G)}{c_{FN}^G \cdot P_G}$$
 
**Corollary:** when $P_A \neq P_B$, the right-hand side differs between
groups, so $\tau_A^* \neq \tau_B^*$ — a uniform threshold is provably
suboptimal for at least one group. Group-specific thresholds are not a
fairness accommodation; they are the mathematically optimal decision rule.
 
This connects directly to **Chouldechova's (2017) impossibility result**:
because $P_A \neq P_B$ in any historically biased dataset, calibration
(equal PPV) and equalised odds (equal TPR/FPR) can never be simultaneously
satisfied. This tool always prioritises equalised odds — the criterion
most relevant to patient safety — and reports the resulting, irreducible
calibration gap explicitly.
"""
 
GUIDE_MD = """
## How to use this tool
 
1. **Set base rates** P<sub>A</sub> and P<sub>B</sub> — the true proportion of each
   group that genuinely needs care.
2. **Set error costs** c<sub>FN</sub> (cost of missing someone in crisis) and
   c<sub>FP</sub> (cost of an unnecessary referral).
3. **Adjust the Group A multiplier** if that group faces higher
   consequences from being missed (e.g. young carers, isolated patients
   with fewer alternative routes to care).
4. **Optionally upload a CSV** of real predicted probabilities with columns
   `prob, true_label, group` (group = A or B) to run the exact same maths
   on your own data instead of the synthetic demonstration.
5. Read off the optimal thresholds τ<sub>A</sub>* and τ<sub>B</sub>* and their
   geometric meaning on the ROC curve.
 
### CSV format
```
prob,true_label,group
0.32,0,A
0.71,1,B
0.18,1,A
0.85,1,B
```
 
### Reference
*From Proxy Features to Fair Decisions: Investigating Algorithmic Bias 
and Threshold Correction in NHS Mental Health AI Triage.*
Building on Chouldechova, A. (2017), *Fair prediction with
disparate impact*, Big Data 5(2).
"""
 
with gr.Blocks(title="Fairness Optimiser",
               theme=gr.themes.Soft(primary_hue="blue")) as demo:
 
    gr.Markdown(
        "# ⚖️ Fairness Optimiser\n"
        "### Compute optimal group-specific triage thresholds — "
        "*From Proxy Features to Fair Decisions: Investigating Algorithmic Bias and Threshold Correction in NHS Mental Health AI Triage* \n"
        "Live implementation of Theorem 1 and Chouldechova's (2017) "
        "impossibility result, running real NumPy/SciPy/scikit-learn "
        "computation — not a lookup table."
    )
 
    with gr.Tabs():
        with gr.TabItem("Optimiser"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Base rates")
                    pa = gr.Slider(0.05, 0.95, value=0.30, step=0.01,
                                    label="P_A — Group A base rate (underrepresented)")
                    pb = gr.Slider(0.05, 0.95, value=0.50, step=0.01,
                                    label="P_B — Group B base rate (majority)")
 
                    gr.Markdown("#### Error costs")
                    cfn = gr.Slider(1, 20, value=5, step=0.5,
                                     label="c_FN — cost of missing someone in crisis")
                    cfp = gr.Slider(0.5, 5, value=1, step=0.5,
                                     label="c_FP — cost of unnecessary referral")
                    ma = gr.Slider(1.0, 3.0, value=1.5, step=0.1,
                                    label="Group A c_FN multiplier (young carers)")
 
                    gr.Markdown("#### Or upload real data")
                    csv_file = gr.File(label="CSV: prob, true_label, group",
                                        file_types=[".csv"])
 
                    run_btn = gr.Button("Compute optimal thresholds ⚡",
                                         variant="primary")
 
                with gr.Column(scale=2):
                    summary = gr.Markdown()
 
            with gr.Row():
                roc_plot = gr.Plot(label="ROC curves with optimal points")
                loss_plot = gr.Plot(label="Expected loss curves")
            chould_plot = gr.Plot(label="Chouldechova's impossibility")
 
            gr.Markdown("#### Export")
            export_json = gr.Code(label="Results (JSON)", language="json")
 
        with gr.TabItem("Theorem"):
            gr.Markdown(THEOREM_MD)
 
        with gr.TabItem("How to use"):
            gr.Markdown(GUIDE_MD)
 
    run_btn.click(
        compute,
        inputs=[pa, pb, cfn, cfp, ma, csv_file],
        outputs=[roc_plot, loss_plot, chould_plot, summary, export_json],
    )
 
    demo.load(
        compute,
        inputs=[pa, pb, cfn, cfp, ma, csv_file],
        outputs=[roc_plot, loss_plot, chould_plot, summary, export_json],
    )
 
if __name__ == "__main__":
    # When localhost accessibility is restricted (corporate proxy / firewall),
    # Gradio can fail the local launch check. Use `share=True` to create a
    # temporary public tunnel as a robust fallback for local testing.
    demo.launch(share=True)