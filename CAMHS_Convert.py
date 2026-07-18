# run from your project root, CURRENT_MODE = "CAMHS" in src/config.py
from src.data_loader import load_and_split
from src.models import get_bayes_model, train_model
import pandas as pd

df, X_train, X_test, y_train, y_test, groups, X_A, y_A, X_B, y_B = load_and_split()
rf = train_model(get_bayes_model(), X_train, y_train)

prob = rf.predict_proba(X_test)[:, 1]
out = pd.DataFrame({
    "prob": prob,
    "true_label": y_test.values,
    "group": groups
})
out.to_csv("webapp/example_data_camhs.csv", index=False)