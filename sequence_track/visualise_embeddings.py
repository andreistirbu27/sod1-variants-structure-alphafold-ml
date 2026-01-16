import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# Paths
INPUT_NPZ = "sequence_track/features_sequence.npz"
FIGURE_DIR = "sequence_track/figures"

def main():
    # 1. Load Data
    if not os.path.exists(INPUT_NPZ):
        print(f"File {INPUT_NPZ} not found. Run generate_embeddings.py first.")
        return

    data = np.load(INPUT_NPZ)
    X = data['X']
    y = data['y']
    
    # 2. Run PCA
    print("Running PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # 3. Plot
    plt.figure(figsize=(8, 6))
    
    # Plot Benign (0)
    plt.scatter(
        X_pca[y == 0, 0], X_pca[y == 0, 1],
        c='blue', alpha=0.6, label='Benign', edgecolors='k'
    )
    
    # Plot Pathogenic (1)
    plt.scatter(
        X_pca[y == 1, 0], X_pca[y == 1, 1],
        c='red', alpha=0.6, label='Pathogenic', edgecolors='k'
    )
    
    plt.title("PCA of SOD1 Variant Embeddings (ESM-2 Difference)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 4. Save
    os.makedirs(FIGURE_DIR, exist_ok=True)
    out_path = os.path.join(FIGURE_DIR, "pca_embeddings.png")
    plt.savefig(out_path, dpi=300)
    print(f"Figure saved to {out_path}")

if __name__ == "__main__":
    main()