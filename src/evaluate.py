# src/evaluate.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)


@dataclass
class CVResults:
    n_samples: int
    n_features: int
    n_splits: int
    seed: int
    roc_auc_mean: float
    roc_auc_std: float
    accuracy_mean: float
    accuracy_std: float
    bal_accuracy_mean: float
    bal_accuracy_std: float
    f1_mean: float
    f1_std: float
    confusion_matrix_sum: List[List[int]]  # [[tn, fp], [fn, tp]] summed over folds


def load_npz(npz_path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)

    if "X" not in data or "y" not in data:
        raise ValueError(f"{npz_path} must contain arrays 'X' and 'y'.")

    X = data["X"]
    y = data["y"]
    ids = data["ids"] if "ids" in data else None

    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features). Got shape {X.shape}")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(f"y must be 1D of length n_samples. Got shape {y.shape}")
    if ids is not None and (ids.ndim != 1 or ids.shape[0] != X.shape[0]):
        raise ValueError(f"ids must be 1D of length n_samples. Got shape {ids.shape}")

    y = y.astype(int)
    uniq = set(np.unique(y).tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"y must be binary 0/1. Found labels: {sorted(uniq)}")

    return X, y, ids


def logistic_regression_fit_predict_proba(train_X, train_y, test_X):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    clf.fit(train_X, train_y)
    proba = clf.predict_proba(test_X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return proba, pred


def evaluate_model_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> CVResults:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    aucs, accs, baccs, f1s = [], [], [], []
    cm_sum = np.zeros((2, 2), dtype=int)

    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        proba, pred = logistic_regression_fit_predict_proba(Xtr, ytr, Xte)

        try:
            auc = roc_auc_score(yte, proba)
        except ValueError:
            auc = float("nan")

        aucs.append(auc)
        accs.append(accuracy_score(yte, pred))
        baccs.append(balanced_accuracy_score(yte, pred))
        f1s.append(f1_score(yte, pred, zero_division=0))
        cm_sum += confusion_matrix(yte, pred, labels=[0, 1])

    def mean_std(vals: List[float]) -> tuple[float, float]:
        arr = np.array(vals, dtype=float)
        return float(np.nanmean(arr)), float(np.nanstd(arr))

    auc_m, auc_s = mean_std(aucs)
    acc_m, acc_s = mean_std(accs)
    bacc_m, bacc_s = mean_std(baccs)
    f1_m, f1_s = mean_std(f1s)

    return CVResults(
        n_samples=int(X.shape[0]),
        n_features=int(X.shape[1]),
        n_splits=int(n_splits),
        seed=int(seed),
        roc_auc_mean=auc_m,
        roc_auc_std=auc_s,
        accuracy_mean=acc_m,
        accuracy_std=acc_s,
        bal_accuracy_mean=bacc_m,
        bal_accuracy_std=bacc_s,
        f1_mean=f1_m,
        f1_std=f1_s,
        confusion_matrix_sum=cm_sum.tolist(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="Path to features_*.npz containing X,y,(ids).")
    parser.add_argument("--out", default=None, help="Optional path to save JSON results.")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y, _ = load_npz(args.npz)
    results = evaluate_model_cv(X, y, n_splits=args.splits, seed=args.seed)

    print(json.dumps(asdict(results), indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(asdict(results), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
