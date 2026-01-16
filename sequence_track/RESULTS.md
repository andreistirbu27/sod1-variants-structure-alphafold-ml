## Sequence track baseline (ESM-2 Embeddings, Logistic Regression)

Dataset:
- n=162 variants (110 label=1, 52 label=0)
- Task: Pathogenic (1) vs Benign (0)
- Model: Logistic Regression on ESM-2 (t30_150M) embedding features

Features:
- Concatenated Vector: [WT_Embedding, Difference]
- Embedding dimension: 1280 (640 WT + 640 Diff)

Cross-validation results:
- ROC-AUC: 0.649 ± 0.142
- Accuracy: 0.618 ± 0.104

Confusion matrix (sum over folds):
- TN=22, FP=30
- FN=32, TP=78
