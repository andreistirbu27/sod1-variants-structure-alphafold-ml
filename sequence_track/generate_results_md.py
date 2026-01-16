import json
import numpy as np
import os

# Paths
INPUT_JSON = "sequence_track/results.json"
OUTPUT_MD = "sequence_track/RESULTS.md"

def main():
    if not os.path.exists(INPUT_JSON):
        print(f"Error: {INPUT_JSON} not found.")
        return

    with open(INPUT_JSON, 'r') as f:
        results = json.load(f)

    # --- FIX 1: Match the keys exactly to your JSON output ---
    auc_mean = results.get("roc_auc_mean", 0.0)
    auc_std = results.get("roc_auc_std", 0.0)
    
    acc_mean = results.get("accuracy_mean", 0.0)
    acc_std = results.get("accuracy_std", 0.0)
    
    # Get Confusion Matrix details
    cm = results.get("confusion_matrix_sum", [[0,0],[0,0]])
    tn, fp = cm[0]
    fn, tp = cm[1]
    
    n_total = tn + fp + fn + tp
    n_pos = fn + tp
    n_neg = tn + fp

    # --- FIX 2: Update description to match the 150M model ---
    md_content = f"""## Sequence track baseline (ESM-2 Embeddings, Logistic Regression)

Dataset:
- n={n_total} variants ({n_pos} label=1, {n_neg} label=0)
- Task: Pathogenic (1) vs Benign (0)
- Model: Logistic Regression on ESM-2 (t30_150M) embedding features

Features:
- Concatenated Vector: [WT_Embedding, Difference]
- Embedding dimension: 1280 (640 WT + 640 Diff)

Cross-validation results:
- ROC-AUC: {auc_mean:.3f} ± {auc_std:.3f}
- Accuracy: {acc_mean:.3f} ± {acc_std:.3f}

Confusion matrix (sum over folds):
- TN={tn}, FP={fp}
- FN={fn}, TP={tp}
"""

    with open(OUTPUT_MD, 'w') as f:
        f.write(md_content)
    
    print(f"Successfully generated {OUTPUT_MD}")
    print(md_content)

if __name__ == "__main__":
    main()