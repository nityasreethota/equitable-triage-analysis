from sklearn.ensemble import RandomForestClassifier
from src.config import BASELINE_PARAMS, GRID_PARAMS, BAYES_PARAMS


def get_baseline_model():
    """Default Random Forest"""
    return RandomForestClassifier(**BASELINE_PARAMS)


def get_grid_model():
    return RandomForestClassifier(**GRID_PARAMS)


def get_bayes_model():
    return RandomForestClassifier(**BAYES_PARAMS)


def train_model(rf, X_train, y_train):
    """Train a model and return it."""
    rf.fit(X_train, y_train)
    return rf


def get_all_models():
    return {
        'baseline': get_baseline_model(),
        'grid': get_grid_model(),
        'bayesian': get_bayes_model()
    }