"""
Spam E-mail Detection - classifier training pipeline.

This module trains five supervised classifiers on the UCI Spambase corpus and
persists each one as a self-contained scikit-learn Pipeline (StandardScaler +
estimator). Bundling the scaler inside the pipeline means the serving layer only
has to call ``pipeline.predict`` - no separate preprocessing step is required.

Run:  python src/build_classifiers.py
Author: Bhavani S (BITS ID: 2026AJ05027)
"""

from __future__ import annotations

import pathlib
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics

SEED = 7
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "_spam" / "spambase_full.csv"
ARTIFACT_DIR = PROJECT_ROOT / "model"
LABEL = "is_spam"
HOLDOUT_FRACTION = 0.25


@dataclass
class Scored:
    """Container for one model's six evaluation scores."""
    accuracy: float
    auc: float
    precision: float
    recall: float
    f1: float
    mcc: float

    def as_row(self) -> dict:
        return {
            "Accuracy": round(self.accuracy, 4),
            "AUC": round(self.auc, 4),
            "Precision": round(self.precision, 4),
            "Recall": round(self.recall, 4),
            "F1": round(self.f1, 4),
            "MCC": round(self.mcc, 4),
        }


def slugify(label: str) -> str:
    """Turn a model label into a filesystem-safe artifact slug."""
    return label.lower().replace(" ", "_").replace("-", "_")


def classifier_zoo() -> dict[str, Pipeline]:
    """Return the five classifiers, each wrapped in a scaling pipeline."""
    estimators = {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=5000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, random_state=SEED),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=11, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(C=2.0, kernel="rbf", probability=True, random_state=SEED),
    }
    return {
        label: Pipeline([("scaler", StandardScaler()), ("clf", est)])
        for label, est in estimators.items()
    }


def score_model(pipe: Pipeline, feats: pd.DataFrame, truth: np.ndarray) -> Scored:
    """Evaluate a fitted pipeline on a feature/label pair (binary task)."""
    preds = pipe.predict(feats)
    try:
        proba = pipe.predict_proba(feats)[:, 1]
        auc = metrics.roc_auc_score(truth, proba)
    except (AttributeError, ValueError):
        auc = float("nan")

    return Scored(
        accuracy=metrics.accuracy_score(truth, preds),
        auc=auc,
        precision=metrics.precision_score(truth, preds, zero_division=0),
        recall=metrics.recall_score(truth, preds, zero_division=0),
        f1=metrics.f1_score(truth, preds, zero_division=0),
        mcc=metrics.matthews_corrcoef(truth, preds),
    )


def persist(obj, filename: str) -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    with open(ARTIFACT_DIR / filename, "wb") as fh:
        pickle.dump(obj, fh)


def run() -> None:
    frame = pd.read_csv(RAW_CSV)
    feature_cols = [c for c in frame.columns if c != LABEL]
    print(f"Corpus loaded -> {frame.shape[0]} messages, {len(feature_cols)} features")

    X = frame[feature_cols]
    y = frame[LABEL].astype(int).to_numpy()

    X_fit, X_hold, y_fit, y_hold = train_test_split(
        X, y, test_size=HOLDOUT_FRACTION, random_state=SEED, stratify=y
    )

    zoo = classifier_zoo()
    scoreboard: dict[str, dict] = {}

    for label, pipe in zoo.items():
        pipe.fit(X_fit, y_fit)
        scoreboard[label] = score_model(pipe, X_hold, y_hold).as_row()
        slug = slugify(label)
        persist(pipe, f"pipeline_{slug}.pkl")
        print(f"  fitted -> {label}")

    # Persist the schema the serving app needs.
    schema = {
        "features": feature_cols,
        "label": LABEL,
        "positive_class": 1,
        "class_labels": {"0": "Not Spam", "1": "Spam"},
        "pipelines": {
            lbl: f"pipeline_{slugify(lbl)}.pkl"
            for lbl in zoo
        },
    }
    persist(schema, "schema.pkl")

    board = pd.DataFrame(scoreboard).T[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    print("\nHold-out performance\n" + "-" * 60)
    print(board.to_string())
    board.to_csv(ARTIFACT_DIR / "scoreboard.csv")

    # Export the hold-out split so the Streamlit app has test data to upload.
    holdout = X_hold.copy()
    holdout[LABEL] = y_hold
    holdout.to_csv(PROJECT_ROOT / "sample_test.csv", index=False)
    print(f"\nWrote sample_test.csv ({len(holdout)} rows)")

    champion = board["F1"].astype(float).idxmax()
    print(f"Top model by F1: {champion}  (F1={board.loc[champion, 'F1']})")


if __name__ == "__main__":
    run()
