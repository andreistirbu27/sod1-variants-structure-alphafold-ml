# Analysis of SOD1 Variants for ALS Prediction using Sequence and Structure Features

## Executive Summary

Amyotrophic Lateral Sclerosis (ALS) is a progressive neurodegenerative disease often linked to mutations in the *SOD1* gene. This study investigates whether computational methods can predict the pathogenicity of *SOD1* missense variants, thereby aiding in the assessment of ALS risk. We employed two distinct approaches: a **Sequence Track** using protein language model embeddings (ESM-2) and a **Structure Track** using AlphaFold-derived structural features.

Our analysis of 162 variants (110 Pathogenic/Likely Pathogenic, 52 Benign/Likely Benign/VUS) yields the following key findings:
- **Predictive performance is modest**: The best-performing single modality (Structure) achieved an ROC-AUC of **0.689**, suggesting that while there is a signal, simple linear models on these features are not yet sufficient for clinical diagnosis.
- **Structural features are slightly superior**: AlphaFold-derived features (specifically local packing and pLDDT) outperformed raw sequence embeddings (ROC-AUC 0.689 vs 0.649).
- **Data limitations**: The inclusion of "Uncertain Significance" variants in the negative class may introduce label noise, complicating the training process.

## 1. Introduction

Superoxide dismutase 1 (SOD1) is a critical antioxidant enzyme. Mutations in the *SOD1* gene are a known cause of familial ALS. With the proliferation of genetic testing, many variants of uncertain significance (VUS) are identified, creating a need for computational tools to predict their potential pathogenicity.

This project aims to leverage modern protein representation learning (ESM-2) and structural prediction (AlphaFold) to classify SOD1 variants as Pathogenic (ALS-associated) or Non-Pathogenic/VUS.

## 2. Methodology

### 2.1 Dataset
The dataset consists of **162 unique missense variants** in the human SOD1 protein (UniProt P00441), derived from ClinVar.
- **Positive Class (1)**: Pathogenic, Likely Pathogenic (n=110)
- **Negative Class (0)**: Benign, Likely Benign, Uncertain Significance (n=52)

*Note: The negative class includes VUS, which poses a challenge as some may be pathogenic but lack sufficient clinical evidence.*

### 2.2 Sequence Track
- **Model**: ESM-2 (t30_150M), a protein language model trained on millions of sequences.
- **Feature Extraction**: For each variant, we extracted the embedding of the wild-type (WT) residue and the mutant residue at the specific position.
- **Feature Vector**: A concatenation of the WT embedding and the difference vector (Mutant - WT), resulting in a 1280-dimensional vector.
- **Classifier**: Logistic Regression with balanced class weights.

### 2.3 Structure Track
- **Model**: AlphaFold (v2) predicted structure for Wild-Type SOD1 (P00441).
- **Feature Extraction**: We extracted local structural properties at the mutation site:
    1.  **pLDDT**: Predicted Local Distance Difference Test score (confidence/disorder).
    2.  **pLDDT Window**: Mean pLDDT of the residue and its neighbors (±2).
    3.  **Neighbor Count**: Number of C-alpha atoms within 8Å (local packing density).
    4.  **Neighbor pLDDT**: Mean pLDDT of the neighboring residues.
- **Classifier**: Logistic Regression with balanced class weights.

### 2.4 Evaluation Strategy
- **Validation**: 5-fold Stratified Cross-Validation (seed=42).
- **Metrics**: ROC-AUC, Accuracy, Balanced Accuracy, F1 Score.

## 3. Results and Analysis

### 3.1 Sequence Track Findings
- **ROC-AUC**: 0.649 ± 0.142
- **Accuracy**: 0.618 ± 0.104
- **True Positives**: 78 | **False Positives**: 30
- **False Negatives**: 32 | **True Negatives**: 22

**Analysis**: The ESM-2 embeddings provide a baseline predictive power significantly better than random (AUC=0.5) but suffer from high variance across folds. This suggests that while evolutionary context (captured by ESM) is informative, the generic embeddings might not fully capture the specific destabilizing effects of SOD1 mutations without fine-tuning or larger models.

### 3.2 Structure Track Findings
- **ROC-AUC**: 0.689 ± 0.067
- **Accuracy**: 0.667 ± 0.030
- **True Positives**: 81 | **False Positives**: 25
- **False Negatives**: 29 | **True Negatives**: 27

**Analysis**: The structural features performed more consistently (lower std dev) and slightly better overall. This aligns with the biological understanding that many ALS-causing SOD1 mutations lead to protein instability or aggregation, which are directly related to local packing (neighbor count) and structural disorder (pLDDT).

### 3.3 Comparative Analysis
| Metric | Sequence Track (ESM-2) | Structure Track (AlphaFold) |
| :--- | :--- | :--- |
| **ROC-AUC** | 0.649 | **0.689** |
| **Accuracy** | 0.618 | **0.667** |
| **Sensitivity (Recall)** | 0.71 (78/110) | **0.74** (81/110) |
| **Specificity** | 0.42 (22/52) | **0.52** (27/52) |

The Structure track offers better specificity, meaning it is better at correctly identifying non-pathogenic (or VUS) variants, reducing false alarms.

## 4. Discussion regarding "Can we predict ALS?"

The short answer is **"Partially, but not with high enough confidence for clinical use yet."**

1.  **Biological Complexity**: SOD1 ALS is often gain-of-function (aggregation), not just loss-of-function. Simple embeddings or local stability metrics might miss long-range allosteric effects or specific aggregation interfaces.
2.  **Label Noise**: Treating "Uncertain Significance" as "Benign" (0) forces the model to classify potentially pathogenic variants as negative. This confusing signal likely lowers the reported performance. A cleaner dataset (excluding VUS) might show higher performance but would be smaller.
3.  **Feature Fusion**: While a fusion model was conceptualized, the individual strength of structural features suggests they are the primary driver. Combining them might add noise unless the dimensionality of the sequence embeddings (1280) is reduced to avoid overfitting.

## 5. Conclusion

We demonstrated that machine learning models using AlphaFold-derived structural features can predict ALS-associated SOD1 mutations with an AUC of ~0.69. While this confirms a structure-function relationship, the predictive performance is limited. 

**Recommendation for Future Work**:
To improve prediction accuracy towards a clinically useful range (>0.9 AUC):
1.  **Refine Labels**: Re-curate the VUS class or use semi-supervised learning.
2.  **Advanced Features**: Calculate $\Delta\Delta G$ (change in stability) explicitly using tools like FoldX or Rosetta, rather than just using static WT features.
3.  **Non-Linear Models**: Move from Logistic Regression to Random Forests or Gradient Boosting to capture non-linear interactions between packing and hydrophobicity.

**Final Verdict**: Current computational methods provide a valuable *prior* or *filter* for prioritizing variants for experimental validation, but cannot yet replace functional assays for ALS risk prediction.
