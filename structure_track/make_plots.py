from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

NPZ_PATH = Path("features_structure.npz")
OUT_DIR = Path("figures")


def boxplot_feature(X: np.ndarray, y: np.ndarray, feat_idx: int, title: str, ylabel: str, out_path: Path):
    x0 = X[y == 0, feat_idx]
    x1 = X[y == 1, feat_idx]

    plt.figure()
    plt.boxplot([x0, x1], tick_labels=["0", "1"], showfliers=True)
    plt.title(title)
    plt.xlabel("Label (0 vs 1)")
    plt.ylabel(ylabel)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

def roc_plot_cv(X: np.ndarray, y: np.ndarray, out_path: Path) -> float:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = np.zeros(len(y), dtype=float)

    for tr, te in skf.split(X, y):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
        model.fit(X[tr], y[tr])
        oof_proba[te] = model.predict_proba(X[te])[:, 1]

    auc = roc_auc_score(y, oof_proba)
    fpr, tpr, _ = roc_curve(y, oof_proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f"Structure model (CV OOF) AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (5-fold CV, out-of-fold predictions)")
    plt.legend(loc="lower right")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    return auc


def main():
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"Missing {NPZ_PATH}. Run: python structure_track/build_features.py")

    data = np.load(NPZ_PATH, allow_pickle=True)
    X = data["X"].astype(float)
    y = data["y"].astype(int)

    auc = roc_plot_cv(X, y, OUT_DIR / "roc_structure_cv.png")
    print(f"Saved ROC curve, AUC = {auc:.3f}")

    # Feature order from build_features.py:
    # [plddt, plddt_window_mean, contact_count_8A, mean_neighbor_plddt]
    boxplot_feature(
        X, y,
        feat_idx=0,
        title="pLDDT at mutated residue by label",
        ylabel="pLDDT (B-factor field)",
        out_path=OUT_DIR / "box_plddt.png"
    )

    boxplot_feature(
        X, y,
        feat_idx=2,
        title="Local packing proxy (CA neighbors within 8Å) by label",
        ylabel="Neighbor count (8Å)",
        out_path=OUT_DIR / "box_contact_count_8A.png"
    )

    # Optional: also save a quick text summary for convenience
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "plot_info.txt"
    summary_path.write_text(
        "Saved:\n"
        "- box_plddt.png\n"
        "- box_contact_count_8A.png\n",
        encoding="utf-8"
    )
    print(f"Wrote plots to {OUT_DIR}/")


if __name__ == "__main__":
    main()
