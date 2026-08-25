"""
classifier.py

Model: RandomForestClassifier as a solid, interpretable baseline.
[ASSUMPTION — README does not specify a classifier architecture for
downstream use; you may swap this for CSP+LDA, EEGNet, etc. as a project
extension. This module is written so the model is easily swappable.]

Train/val/test split: 60/20/20, split BY TRIAL, not by session and not by
random window shuffle. [ASSUMPTION / design choice, updated per user
request from an earlier by-session split.]

Why by trial: windows within one trial overlap 50% (config.WINDOW_OVERLAP),
so a random window-level shuffle would put near-duplicate windows on both
sides of a split boundary and inflate accuracy artificially. Splitting by
whole trial keeps every window of a given trial on one side only, while
still allowing an exact 60/20/20 ratio (unlike splitting by session, which
only gives 4 discrete blocks and confounds decoder identity with split —
Se01 runs a different decoder set than Se02-Se04, see config.py).

This must be reported explicitly as a methodology choice in your report.
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def trial_val_test_split(X: np.ndarray, y: list, meta: list,
                          train_frac: float = 0.6, val_frac: float = 0.2,
                          test_frac: float = 0.2, seed: int = 42) -> dict:
    """
    Split data 60/20/20 (default) by TRIAL, so that no window from one
    trial appears in more than one of train/val/test. This avoids the
    window-overlap leakage documented above.

    IMPORTANT (test isolation): the returned test arrays are meant to be
    touched ONLY after model training and validation are fully complete.
    Do not call evaluate_classifier() on the test split during development
    or hyperparameter tuning — use the validation split for that. See
    main.py for the enforced ordering (train -> validate -> test, in that
    sequence).

    Parameters
    ----------
    train_frac, val_frac, test_frac : must sum to 1.0
    seed : fixes the trial shuffle order for reproducibility.

    Returns
    -------
    dict with keys: X_train, y_train, X_val, y_val, X_test, y_test,
    and train_trials/val_trials/test_trials (lists of trial_id, for
    traceability/reporting).
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError(
            f"train_frac + val_frac + test_frac must sum to 1.0, got "
            f"{train_frac + val_frac + test_frac}"
        )

    # Group window indices by trial_id.
    trial_to_indices = {}
    for i, m in enumerate(meta):
        trial_to_indices.setdefault(m["trial_id"], []).append(i)

    trial_ids = sorted(trial_to_indices.keys())  # sorted first for determinism
    rng = np.random.default_rng(seed)
    rng.shuffle(trial_ids)

    n_windows_total = len(meta)
    train_idx, val_idx, test_idx = [], [], []
    train_trials, val_trials, test_trials = [], [], []
    running_total = 0

    for tid in trial_ids:
        idxs = trial_to_indices[tid]
        # Assign each whole trial to whichever split is furthest below its
        # target fraction, based on windows placed so far (greedy balancing).
        n_train, n_val, n_test = len(train_idx), len(val_idx), len(test_idx)
        n_so_far = max(n_train + n_val + n_test, 1)
        train_deficit = train_frac - (n_train / n_so_far)
        val_deficit = val_frac - (n_val / n_so_far)
        test_deficit = test_frac - (n_test / n_so_far)

        target = max(
            [("train", train_deficit), ("val", val_deficit), ("test", test_deficit)],
            key=lambda kv: kv[1],
        )[0]

        if target == "train":
            train_idx.extend(idxs); train_trials.append(tid)
        elif target == "val":
            val_idx.extend(idxs); val_trials.append(tid)
        else:
            test_idx.extend(idxs); test_trials.append(tid)

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"Trial split produced an empty split (train={len(train_idx)}, "
            f"val={len(val_idx)}, test={len(test_idx)}). "
            f"Total trials available: {len(trial_ids)}. "
            f"You need at least 3 distinct trials (one per split) for this to work."
        )

    return {
        "X_train": X[train_idx], "y_train": [y[i] for i in train_idx],
        "X_val": X[val_idx], "y_val": [y[i] for i in val_idx],
        "X_test": X[test_idx], "y_test": [y[i] for i in test_idx],
        "train_trials": train_trials, "val_trials": val_trials, "test_trials": test_trials,
        "n_windows_total": n_windows_total,
    }


def train_classifier(X_train: np.ndarray, y_train: list):
    """Fit a StandardScaler + RandomForest pipeline."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
    clf.fit(X_train_scaled, y_train)
    return clf, scaler


def evaluate_classifier(clf, scaler, X_test: np.ndarray, y_test: list) -> dict:
    """Evaluate on held-out data. Returns dict with accuracy, report, confusion matrix."""
    X_test_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, labels=config.CLASSES, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=config.CLASSES)

    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_test": y_test,
    }


def save_model(clf, scaler, path_prefix: str):
    """Save classifier and scaler to disk."""
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    joblib.dump(clf, f"{path_prefix}_clf.joblib")
    joblib.dump(scaler, f"{path_prefix}_scaler.joblib")


def load_model(path_prefix: str):
    """Load classifier and scaler from disk."""
    clf = joblib.load(f"{path_prefix}_clf.joblib")
    scaler = joblib.load(f"{path_prefix}_scaler.joblib")
    return clf, scaler
