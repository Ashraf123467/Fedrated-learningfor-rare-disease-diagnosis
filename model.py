"""
model.py - Machine Learning Model Definition & Parameter Serialization.

Implements a Scikit-Learn LogisticRegression diagnostic classifier.
Extracts and injects parameters (weights & bias vectors) to enable
Federated Averaging (FedAvg) without exposing local patient records.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from data import FEATURE_NAMES

class RareDiseaseClassifier:
    """
    Local / Global model wrapper around Scikit-Learn Logistic Regression.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = LogisticRegression(
            max_iter=500,
            solver="lbfgs",
            random_state=random_state,
            warm_start=True
        )
        self.is_fitted = False

    def get_parameters(self) -> dict:
        """
        Extracts model weights and bias for transmission to server.
        """
        if not self.is_fitted:
            # Default zero initialization if not yet trained
            n_features = len(FEATURE_NAMES)
            return {
                "coef": np.zeros((1, n_features)),
                "intercept": np.zeros((1,))
            }
        return {
            "coef": self.model.coef_.copy(),
            "intercept": self.model.intercept_.copy()
        }

    def set_parameters(self, parameters: dict):
        """
        Injects aggregated global model parameters into local model instance.
        """
        n_features = len(FEATURE_NAMES)
        self.model.classes_ = np.array([0, 1])
        self.model.coef_ = parameters["coef"].reshape(1, n_features)
        self.model.intercept_ = parameters["intercept"].reshape(1,)
        self.is_fitted = True

    def fit(self, X, y):
        """
        Trains model locally on hospital dataset.
        """
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X):
        """
        Predicts binary diagnosis label (0: Control, 1: Positive).
        """
        if not self.is_fitted:
            return np.zeros(len(X))
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predicts disease probability [P(Negative), P(Positive)].
        """
        if not self.is_fitted:
            return np.tile([0.5, 0.5], (len(X), 1))
        return self.model.predict_proba(X)

    def evaluate(self, X, y) -> dict:
        """
        Computes diagnostic metrics: Accuracy, Precision, Recall, F1, AUC, Confusion Matrix.
        """
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)[:, 1] if self.is_fitted else np.zeros(len(y))

        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, y_prob)) if len(np.unique(y)) > 1 else 0.5,
            "confusion_matrix": cm.tolist(),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        }