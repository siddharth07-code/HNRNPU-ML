# HNRNPU-ML

## Machine Learning-Based Classification of HNRNPU Genetic Variants Using ClinVar

This project explores the application of **machine learning for the classification of genetic variants in the HNRNPU gene** using data derived from the **ClinVar** database.

The goal is to develop a reproducible computational pipeline for extracting HNRNPU-specific variants, preprocessing genetic data, engineering relevant features, and training machine learning models for variant classification.

---

## 🧬 Project Overview

**HNRNPU (Heterogeneous Nuclear Ribonucleoprotein U)** is a gene associated with human genetic disorders. Variants within this gene can have different clinical interpretations, including benign, pathogenic, uncertain, and conflicting classifications.

This project investigates whether machine learning can identify patterns within annotated HNRNPU variants and assist computational analysis of variant classifications.

### Research Pipeline

    ClinVar Dataset
          │
          ▼
    HNRNPU Variant Extraction
          │
          ▼
    Data Cleaning & Deduplication
          │
          ▼
    Variant Classification
          │
          ▼
    Feature Engineering
          │
          ▼
    ML Training Dataset
          │
          ▼
    Machine Learning Models
          │
          ▼
    Model Evaluation

---

## 🎯 Objectives

- Extract HNRNPU-specific variants from ClinVar.
- Clean and normalize variant representations.
- Remove duplicate variant records.
- Analyze clinical significance annotations.
- Engineer machine-learning features from variant information.
- Train supervised machine-learning models.
- Evaluate model performance using appropriate classification metrics.
- Investigate the potential of ML for computational genetic variant analysis.

---

## 📊 Dataset

The primary data source for this project is **ClinVar**, a publicly accessible database containing information about human genetic variants and their relationships with diseases and phenotypes.

The analysis focuses specifically on variants associated with **HNRNPU**.

### Dataset Processing

The current preprocessing pipeline produced approximately:

| Stage | Records |
|---|---:|
| HNRNPU ClinVar records | 2,487 |
| After duplicate representation removal | 1,247 |
| Missense variants | 518 |
| Variants selected for ML training | 146 |
| VUS variants | 326 |
| Conflicting variants | 41 |
| Benign training labels | 100 |
| Pathogenic training labels | 46 |

> These values represent the current state of the dataset and may change as the preprocessing and feature-engineering pipeline is refined.

### Raw ClinVar Dataset

The complete ClinVar `variant_summary.txt.gz` file is **not included in this repository** because the file is approximately **430 MB**, exceeding GitHub's 100 MB individual file limit.

The raw dataset should be obtained separately and placed locally at:

    data/variant_summary.txt.gz

The file is intentionally excluded from Git using `.gitignore`.

---

## 🧠 Machine Learning Task

The initial supervised learning task is to classify HNRNPU variants into two primary categories:

- **Benign**
- **Pathogenic**

The current training dataset contains:

- 100 Benign variants
- 46 Pathogenic variants

This results in a binary classification task:

    Variant
       │
       ├── Benign
       │
       └── Pathogenic

Variants classified as VUS or having conflicting interpretations are handled separately during dataset preparation rather than being automatically assigned to either class.

---

## 🔬 Feature Engineering

The project converts genetic variant information into machine-learning features.

Feature categories may include:

- Variant type
- Molecular consequence
- Protein change
- Amino-acid substitution
- Variant position
- Allele information
- Clinical significance
- Other available ClinVar annotations

Feature engineering is implemented within the `src/` directory.

---

## 📁 Project Structure

    HNRNPU-ML/
    │
    ├── data/
    │   ├── hnrnpu_variants.csv
    │   └── ...
    │
    ├── results/
    │   └── ...
    │
    ├── src/
    │   ├── clinvar_exploration.py
    │   ├── dataset_preparation.py
    │   ├── feature_engineering.py
    │   └── ...
    │
    ├── .gitignore
    ├── README.md
    └── requirements.txt

> The complete raw ClinVar dataset is intentionally excluded from the repository.

---

## ⚙️ Installation

### 1. Clone the Repository

    git clone https://github.com/siddharth07-code/HNRNPU-ML.git

### 2. Navigate to the Project

    cd HNRNPU-ML

### 3. Create a Virtual Environment

    python3 -m venv .venv

### 4. Activate the Virtual Environment

macOS/Linux:

    source .venv/bin/activate

### 5. Install Dependencies

    pip install -r requirements.txt

---

## 🚀 Running the Pipeline

### 1. ClinVar Exploration

    python src/clinvar_exploration.py

This step extracts and explores HNRNPU-associated records from the ClinVar dataset.

### 2. Dataset Preparation

    python src/dataset_preparation.py

This step performs data cleaning, normalization, duplicate handling, and preparation of the training dataset.

### 3. Feature Engineering

    python src/feature_engineering.py

This step converts the processed variant data into a machine-learning-ready feature dataset.

---

## 📈 Model Evaluation

The models are evaluated using classification metrics suitable for biomedical datasets.

Key metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC
- PR-AUC

Because the dataset contains relatively few pathogenic variants compared with benign variants, accuracy alone should not be used to assess model performance.

Particular attention should be given to precision, recall, and F1-score for pathogenic variant classification.

---

## 🧪 Research Context

This project is an exploratory computational research project investigating the application of machine learning to genetic variant interpretation.

Machine-learning predictions generated by this project should **not be considered clinical diagnoses or definitive clinical interpretations**.

ClinVar contains evidence from different sources and may include uncertain or conflicting classifications. Therefore, model predictions must be interpreted within the context of the underlying dataset and biological evidence.

---

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Machine Learning
- ClinVar
- Genomic Variant Analysis
- Feature Engineering

---

## 📌 Current Status

### Completed

- [x] ClinVar data exploration
- [x] HNRNPU variant extraction
- [x] Dataset cleaning
- [x] Duplicate variant handling
- [x] Variant categorization
- [x] Feature engineering
- [x] Initial ML training dataset
- [x] Initial ML model training

### Planned

- [ ] Compare multiple machine-learning algorithms
- [ ] Hyperparameter optimization
- [ ] Cross-validation
- [ ] Address class imbalance
- [ ] Feature importance analysis
- [ ] SHAP-based model interpretation
- [ ] External validation
- [ ] Evaluation on unseen variants
- [ ] Biological interpretation of important features
- [ ] Research paper preparation

---

## 🔮 Future Work

### Larger Training Dataset

Increase the number of high-confidence HNRNPU variants available for model training.

### Advanced Variant Features

Incorporate additional sequence, protein, conservation, and functional annotations.

### Explainable AI

Apply techniques such as SHAP to determine which features contribute most strongly to model predictions.

### Model Comparison

Compare multiple machine-learning algorithms and evaluate their generalization performance.

### External Validation

Evaluate the trained models on independent variant datasets to determine how well they generalize beyond the training data.

### Biological Interpretation

Investigate whether patterns identified by the models correspond to known biological mechanisms associated with HNRNPU.

---

## ⚠️ Limitations

- ClinVar contains variants with different levels of supporting evidence.
- Some variants have conflicting clinical interpretations.
- The current labeled training dataset is relatively small.
- Class imbalance can affect model performance.
- Variant representations may differ between records.
- Model performance on ClinVar data does not automatically establish clinical validity.
- A model trained specifically on HNRNPU variants may not generalize to other genes.

---

## 📚 Data Source

The project uses data derived from **ClinVar**, maintained by the National Center for Biotechnology Information (NCBI).

The raw ClinVar dataset is not stored in this repository due to its large size.

Users reproducing the project should obtain the required ClinVar dataset separately and place it in the `data/` directory.

---

## 👨‍💻 Author

**Siddharth Lal**

B.Tech Computer Science Engineering  
Manipal University Jaipur

---

## 📄 License

This repository contains original source code and analysis developed for this research project.

Dataset licensing and attribution requirements should be followed according to the terms of the respective data providers.