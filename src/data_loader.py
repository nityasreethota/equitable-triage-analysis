import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import (
    DATA_PATH, FEATURES, TARGET, GROUP_COL,
    GROUP_A, GROUP_B, TEST_SIZE, RANDOM_STATE
)


def load_data():
    """
    Load dataset based on current mode.
    Applies feature engineering for CAMHS mode.
    """
    from src.config import CURRENT_MODE, DATA_PATH, FEATURES, TARGET

    df_raw = pd.read_csv(DATA_PATH)

    if CURRENT_MODE == "CAMHS":
        from src.feature_engineering import engineer_camhs_features
        df = engineer_camhs_features(df_raw)
        # Use engineered features
        available = [f for f in FEATURES if f in df.columns]
        X = df[available].fillna(-1)
    else:
        df = df_raw
        X = df[FEATURES].fillna(-1)

    y = df[TARGET]
    return df, X, y


def split_data(X, y):
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


def get_group_labels(df, X_test):
    return df.loc[X_test.index, GROUP_COL].values


def split_by_group(X_test, y_test, groups):
    mask_A = groups == GROUP_A
    mask_B = groups == GROUP_B

    if isinstance(X_test, pd.DataFrame):
        X_A = X_test[mask_A]
        X_B = X_test[mask_B]
    else:
        X_A = X_test[mask_A]
        X_B = X_test[mask_B]

    y_A = y_test[mask_A] if hasattr(y_test, '__getitem__') \
        else y_test.values[mask_A]
    y_B = y_test[mask_B] if hasattr(y_test, '__getitem__') \
        else y_test.values[mask_B]

    return X_A, y_A, X_B, y_B


def load_and_split():
    df, X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    groups = get_group_labels(df, X_test)
    X_A, y_A, X_B, y_B = split_by_group(X_test, y_test, groups)

    return (df, X_train, X_test, y_train, y_test,
            groups, X_A, y_A, X_B, y_B)