import json
import numpy as np

# Paths
INPUT_JSON = "sequence_track/results.json"  # Output from evaluate.py
OUTPUT_MD = "sequence_track/RESULTS.md"

def main():
    # 1. Load the JSON results
    try:
        with open(INPUT_JSON, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_JSON} not found. Did you run evaluate.py?")
        return

    # 2. Extract Metrics
    # Note: Adjust keys based on exactly what evaluate.py outputs.
    # Assuming standard sklearn keys or the structure likely used by your partner.
    
    auc_mean = results.get("test_roc_auc_mean", 0.0)
    auc_std = results.get("test_roc_auc_std", 0.0)
    
    acc_mean = results.get("test_accuracy_mean", 0.0)
    acc_std = results.get("test_accuracy_std", 0.0)
    
    # Calculate dataset stats (if available in JSON, otherwise hardcoded/estimated)
    # If the JSON contains confusion matrix, we can derive counts
    cm = results.get("confusion_matrix_sum", [[0,0],[0,0]])
    tn, fp = cm[0]
    fn, tp = cm[1]
    
    n_total = tn + fp + fn + tp
    n_pos = fn + tp
    n_neg = tn + fp

    # 3. Format the Markdown Content
    md_content = f"""## Sequence track baseline (ESM-2 Embeddings, Logistic Regression)

Dataset:
- n={n_total} variants ({n_pos} label=1, {n_neg} label=0)
- Task: Pathogenic (1) vs Benign (0)
- Model: Logistic Regression on ESM-2 (t6_8M) embedding differences

Features:
- Vector difference (Mutant - Wildtype)
- Embedding dimension: 320 (from esm2_t6_8M_UR50D)

Cross-validation results:
- ROC-AUC: {auc_mean:.3f} ± {auc_std:.3f}
- Accuracy: {acc_mean:.3f} ± {acc_std:.3f}

Confusion matrix (sum over folds):
- TN={tn}, FP={fp}
- FN={fn}, TP={tp}
"""

    # 4. Save to File
    with open(OUTPUT_MD, 'w') as f:
        f.write(md_content)
    
    print(f"Successfully generated {OUTPUT_MD}")
    print(md_content)

if __name__ == "__main__":
    main()