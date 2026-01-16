## Sequence track baseline (ESM-2 Embeddings, Logistic Regression)

Dataset:
- n=162 variants (110 label=1, 52 label=0)
- Task: Pathogenic (1) vs Benign (0)
- Model: Logistic Regression on ESM-2 (t6_8M) embedding differences

Features:
- Vector difference (Mutant - Wildtype)
- Embedding dimension: 320 (from esm2_t6_8M_UR50D)

Cross-validation results:
- ROC-AUC: 0.000 ± 0.000
- Accuracy: 0.000 ± 0.000

Confusion matrix (sum over folds):
- TN=14, FP=38
- FN=48, TP=62
