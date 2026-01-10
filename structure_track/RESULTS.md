## Structure track baseline (logistic regression, 5-fold CV, seed=42)

Dataset:
- n=162 variants (110 label=1, 52 label=0)
- Task: label=1 vs label=0 (Pathogenic/Likely pathogenic vs VUS)

Features (4):
- pLDDT at mutated residue
- mean pLDDT in window ±2 residues
- CA-neighbor count within 8Å
- mean neighbor pLDDT within 8Å

Cross-validation results:
- ROC-AUC: 0.689 ± 0.067
- Accuracy: 0.667 ± 0.030
- Balanced accuracy: 0.625 ± 0.049
- F1: 0.748 ± 0.035

Confusion matrix (sum over folds):
- TN=27, FP=25
- FN=29, TP=81
