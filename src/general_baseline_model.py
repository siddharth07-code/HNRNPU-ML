import os
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GroupShuffleSplit,
    StratifiedKFold,
    cross_validate
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/general_features.csv"
RESULTS_DIR = "results"

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("GENERAL BASELINE MODEL")
print("=" * 70)

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"Records: {len(df):,}")
print(f"Genes: {df['GeneSymbol'].nunique():,}")


# ============================================================
# FEATURE SELECTION
# ============================================================

feature_columns = [
    "ProteinPosition",

    "OriginalHydrophobicity",
    "NewHydrophobicity",
    "HydrophobicityDifference",

    "OriginalVolume",
    "NewVolume",
    "VolumeDifference",

    "OriginalPolarity",
    "NewPolarity",
    "PolarityDifference",

    "OriginalCharge",
    "NewCharge",
    "ChargeDifference",

    "SameCharge",
    "SamePolarity",
    "SameHydrophobicityClass",
    "SameGroup"
]


X = df[feature_columns].copy()
y = df["Label"].astype(int)

groups = df["GeneSymbol"].astype(str)


print(f"\nFeatures used: {len(feature_columns)}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

print("\nClass distribution:")
print(
    pd.Series(y)
    .value_counts()
    .rename(index={0: "Benign", 1: "Pathogenic"})
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
}


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    results = {
        "Accuracy": accuracy_score(y_test, predictions),

        "Balanced Accuracy": balanced_accuracy_score(
            y_test,
            predictions
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "F1": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y_test,
            probabilities
        ),

        "PR-AUC": average_precision_score(
            y_test,
            probabilities
        )
    }

    return results, predictions, probabilities


# ============================================================
# RANDOM STRATIFIED SPLIT
# ============================================================

print("\n" + "=" * 70)
print("RANDOM STRATIFIED SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

print(f"Training variants: {len(X_train):,}")
print(f"Testing variants : {len(X_test):,}")


random_results = []


for name, model in models.items():

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    results, predictions, probabilities = evaluate_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    for metric, value in results.items():
        print(f"{metric:20s}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    random_results.append({
        "Model": name,
        "Split": "Random",
        **results
    })


# ============================================================
# GENE-AWARE SPLIT
# ============================================================

print("\n" + "=" * 70)
print("GENE-AWARE SPLIT")
print("=" * 70)

print(
    "\nEntire genes are kept together so that "
    "the test set contains genes unseen during training."
)

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_indices, test_indices = next(
    gss.split(
        X,
        y,
        groups=groups
    )
)

X_train_gene = X.iloc[train_indices]
X_test_gene = X.iloc[test_indices]

y_train_gene = y.iloc[train_indices]
y_test_gene = y.iloc[test_indices]

genes_train = groups.iloc[train_indices]
genes_test = groups.iloc[test_indices]

print(f"\nTraining variants: {len(X_train_gene):,}")
print(f"Testing variants : {len(X_test_gene):,}")

print(f"Training genes: {genes_train.nunique():,}")
print(f"Testing genes : {genes_test.nunique():,}")

overlap = set(genes_train) & set(genes_test)

print(f"Gene overlap    : {len(overlap)}")

if len(overlap) != 0:
    raise ValueError(
        "Gene leakage detected between training and test sets!"
    )


gene_results = []


for name, model in models.items():

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    results, predictions, probabilities = evaluate_model(
        model,
        X_train_gene,
        X_test_gene,
        y_train_gene,
        y_test_gene
    )

    for metric, value in results.items():
        print(f"{metric:20s}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test_gene,
            predictions
        )
    )

    gene_results.append({
        "Model": name,
        "Split": "Gene-aware",
        **results
    })


# ============================================================
# 5-FOLD RANDOM CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 70)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

cv_results = []


scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision"
}


for name, model in models.items():

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    for metric_name in scoring:

        values = scores[
            f"test_{metric_name}"
        ]

        mean = values.mean()
        std = values.std()

        print(
            f"{metric_name:20s}: "
            f"{mean:.4f} ± {std:.4f}"
        )

    cv_results.append({
        "Model": name,
        "Split": "5-fold CV",

        "Accuracy": scores["test_accuracy"].mean(),
        "Balanced Accuracy":
            scores["test_balanced_accuracy"].mean(),
        "Precision":
            scores["test_precision"].mean(),
        "Recall":
            scores["test_recall"].mean(),
        "F1":
            scores["test_f1"].mean(),
        "ROC-AUC":
            scores["test_roc_auc"].mean(),
        "PR-AUC":
            scores["test_pr_auc"].mean()
    })


# ============================================================
# SAVE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

os.makedirs(RESULTS_DIR, exist_ok=True)

all_results = pd.DataFrame(
    random_results
    + gene_results
    + cv_results
)

output_file = os.path.join(
    RESULTS_DIR,
    "general_baseline_results.csv"
)

all_results.to_csv(
    output_file,
    index=False
)

print(f"Saved to: {output_file}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    all_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 70)
print("GENERAL BASELINE COMPLETE")
print("=" * 70)