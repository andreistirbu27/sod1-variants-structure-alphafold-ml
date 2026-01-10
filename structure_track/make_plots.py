from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


NPZ_PATH = Path("structure_track/features_structure.npz")
OUT_DIR = Path("structure_track/figures")


def boxplot_feature(X: np.ndarray, y: np.ndarray, feat_idx: int, title: str, ylabel: str, out_path: Path):
    x0 = X[y == 0, feat_idx]
    x1 = X[y == 1, feat_idx]

    plt.figure()
    plt.boxplot([x0, x1], labels=["0", "1"], showfliers=True)
    plt.title(title)
    plt.xlabel("Label (0 vs 1)")
    plt.ylabel(ylabel)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"Missing {NPZ_PATH}. Run: python structure_track/build_features.py")

    data = np.load(NPZ_PATH, allow_pickle=True)
    X = data["X"].astype(float)
    y = data["y"].astype(int)

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
