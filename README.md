---
title: Fairness Optimiser
emoji: ⚖️
colorFrom: blue
colorTo: green
sdk: static
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: mit
---

# Fairness Optimiser

An interactive demonstration accompanying the research project:
**From Proxy Features to Fair Decisions: Investigating Algorithmic Bias and Threshold Correction in NHS Mental Health AI Triage**
The application illustrates how algorithmic bias can arise in AI-assisted mental health triage and explores the trade-offs involved in mitigating it through threshold adjustment.

## Features

- Explore fairness metrics across demographic groups.
- Compare uniform and group-specific decision thresholds.
- Visualise trade-offs between:
  - True Positive Rate (TPR)
  - False Positive Rate (FPR)
  - Positive Predictive Value (PPV)
  - Expected decision cost
- Demonstrate why fairness criteria cannot always be satisfied simultaneously when groups have different base rates (Chouldechova, 2017).
- Upload your own prediction data (`prob`, `true_label`, `group`) to evaluate fairness metrics.

The accompanying research investigates:
- Detection of proxy feature reliance using permutation importance.
- Hyperparameter optimisation using Grid Search and Bayesian Optimisation.
- Threshold correction to investigate and reduce disparities in NHS mental health triage.
- Synthetic case studies based on CHRONOSIG and CAMHS triage scenarios.

## Running locally
```bash
pip install -r requirements.txt
python app.py

Then open:
http://127.0.0.1:7860

## Disclaimer
This project uses synthetic data for research and educational purposes only. It is not intended for clinical decision-making.