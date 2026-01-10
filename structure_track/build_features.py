# structure_track/build_features.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser


# Paths (relative to repo root)
PDB_PATH = Path("data/raw/AF-P00441-F1-model_v6.pdb")
VARIANTS_CSV = Path("data/variants.csv")
OUT_NPZ = Path("structure_track/features_structure.npz")

# Feature hyperparams
CONTACT_RADIUS_A = 8.0   # CA-CA neighbor radius in Angstrom
WINDOW = 2               # +/- residues for local pLDDT mean


def load_plddt_and_ca_coords(pdb_path: Path) -> tuple[dict[int, float], dict[int, np.ndarray]]:
    """
    AlphaFold PDB: pLDDT is stored in the B-factor field.
    We take the CA atom for each residue.

    Returns:
      pos_to_plddt: residue position (1-based) -> pLDDT
      pos_to_ca:    residue position (1-based) -> CA xyz coords (np.array shape (3,))
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("AF", str(pdb_path))

    model = next(structure.get_models())
    chain = next(model.get_chains())

    pos_to_plddt: dict[int, float] = {}
    pos_to_ca: dict[int, np.ndarray] = {}

    for res in chain.get_residues():
        hetfield, resseq, icode = res.get_id()
        if hetfield != " ":
            continue
        if "CA" not in res:
            continue
        ca = res["CA"]
        pos = int(resseq)
        pos_to_ca[pos] = ca.coord.astype(float)
        pos_to_plddt[pos] = float(ca.get_bfactor())

    if not pos_to_plddt:
        raise RuntimeError("No residues parsed from PDB. Is the file correct?")

    return pos_to_plddt, pos_to_ca


def compute_pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """coords: (N,3) -> dists: (N,N)"""
    diffs = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diffs * diffs, axis=-1))


def local_window_mean(pos: int, pos_to_val: dict[int, float], w: int) -> float:
    vals = [pos_to_val[p] for p in range(pos - w, pos + w + 1) if p in pos_to_val]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    if not VARIANTS_CSV.exists():
        raise FileNotFoundError(f"Missing {VARIANTS_CSV}. Generate it first.")
    if not PDB_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PDB_PATH}. Download it with:\n"
            f'curl -L -o {PDB_PATH} "https://alphafold.ebi.ac.uk/files/AF-P00441-F1-model_v4.pdb"'
        )

    df = pd.read_csv(VARIANTS_CSV)
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)

    pos_to_plddt, pos_to_ca = load_plddt_and_ca_coords(PDB_PATH)

    # Prepare arrays for distance-based features
    positions = np.array(sorted(pos_to_ca.keys()), dtype=int)
    coords = np.stack([pos_to_ca[p] for p in positions], axis=0)  # (N,3)
    dists = compute_pairwise_distances(coords)

    # Map residue position -> index in positions/coords
    pos_to_idx = {int(p): i for i, p in enumerate(positions)}

    feats = []
    ids = []
    y = []
    skipped = 0

    for _, row in df.iterrows():
        vid = str(row["variant_id"])
        pos = int(row["pos"])
        label = int(row["label"])

        if pos not in pos_to_idx or pos not in pos_to_plddt:
            skipped += 1
            continue

        idx = pos_to_idx[pos]
        plddt = pos_to_plddt[pos]
        plddt_win = local_window_mean(pos, pos_to_plddt, WINDOW)

        # CA neighbors within radius (excluding itself)
        mask = (dists[idx] <= CONTACT_RADIUS_A) & (dists[idx] > 1e-6)
        neighbor_positions = positions[mask]
        contact_count = int(mask.sum())

        # Mean pLDDT of neighbors
        neigh_plddt_vals = [pos_to_plddt.get(int(p), np.nan) for p in neighbor_positions]
        neigh_plddt_vals = [v for v in neigh_plddt_vals if not np.isnan(v)]
        mean_neighbor_plddt = float(np.mean(neigh_plddt_vals)) if neigh_plddt_vals else float("nan")

        # Simple 4-feature vector
        feats.append([plddt, plddt_win, contact_count, mean_neighbor_plddt])
        ids.append(vid)
        y.append(label)

    X = np.asarray(feats, dtype=float)
    y = np.asarray(y, dtype=int)
    ids = np.asarray(ids, dtype=object)

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, X=X, y=y, ids=ids)

    print(f"Wrote {OUT_NPZ} with X shape {X.shape}, y shape {y.shape}")
    if skipped:
        print(f"Skipped {skipped} variants (pos not found in PDB numbering).")
    unique, counts = np.unique(y, return_counts=True)
    print("Label counts:", dict(zip(unique.tolist(), counts.tolist())))
    print("Feature columns: [plddt, plddt_window_mean, contact_count_8A, mean_neighbor_plddt]")


if __name__ == "__main__":
    main()
