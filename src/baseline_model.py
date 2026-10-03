import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

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
# Configuration
# ============================================================

INPUT_FILE = "data/hnrnpu_features.csv"

RANDOM_STATE = 42


# ============================================================
# Load dataset
# ============================================================

print("=" * 60)
print("LOADING HNRNPU FEATURE DATASET")
print("=" * 60)

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(
    f"\nTotal variants: {len(df):,}"
)


# ============================================================
# Display label distribution
# ============================================================

print("\n" + "=" * 60)
print("LABEL DISTRIBUTION")
print("=" * 60)

print(
    df["Label"]
    .value_counts()
    .sort_index()
    .rename(
        {
            0: "Benign",
            1: "Pathogenic"
        }
    )
    .to_string()
)


# ============================================================
# Select numerical features
# ============================================================

print("\n" + "=" * 60)
print("SELECTING NUMERICAL FEATURES")
print("=" * 60)

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
    "SameHydrophobicityClass"
]


available_features = [
    column
    for column in feature_columns
    if column in df.columns
]

print("\nFeatures used:")

for index, column in enumerate(
    available_features,
    start=1
):
    print(
        f"{index:2}. {column}"
    )


# ============================================================
# Build X and y
# ============================================================

X = df[
    available_features
].copy()

y = df[
    "Label"
].astype(int)


print(
    f"\nFeature matrix shape: {X.shape}"
)

print(
    f"Target shape        : {y.shape}"
)


# ============================================================
# Train/test split
# ============================================================

print("\n" + "=" * 60)
print("CREATING TRAIN / TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y
)

print(
    f"\nTraining samples: {len(X_train):,}"
)

print(
    f"Testing samples : {len(X_test):,}"
)

print("\nTraining labels:")

print(
    y_train
    .value_counts()
    .sort_index()
    .rename(
        {
            0: "Benign",
            1: "Pathogenic"
        }
    )
    .to_string()
)

print("\nTesting labels:")

print(
    y_test
    .value_counts()
    .sort_index()
    .rename(
        {
            0: "Benign",
            1: "Pathogenic"
        }
    )
    .to_string()
)


# ============================================================
# Evaluation function
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name
):

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    average_precision = average_precision_score(
        y_test,
        probabilities
    )

    print(
        f"\nAccuracy           : {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy  : "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"Precision          : {precision:.4f}"
    )

    print(
        f"Recall             : {recall:.4f}"
    )

    print(
        f"F1 Score           : {f1:.4f}"
    )

    print(
        f"ROC-AUC            : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC             : "
        f"{average_precision:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Benign",
                "Pathogenic"
            ],
            zero_division=0
        )
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "BalancedAccuracy": balanced_accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": average_precision
    }


# ============================================================
# Logistic Regression
# ============================================================

print("\n" + "=" * 60)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 60)

logistic_model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                random_state=RANDOM_STATE
            )
        )
    ]
)

logistic_model.fit(
    X_train,
    y_train
)

logistic_results = evaluate_model(
    logistic_model,
    X_test,
    y_test,
    "LOGISTIC REGRESSION"
)


# ============================================================
# Random Forest
# ============================================================

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

random_forest_model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)

random_forest_model.fit(
    X_train,
    y_train
)

random_forest_results = evaluate_model(
    random_forest_model,
    X_test,
    y_test,
    "RANDOM FOREST"
)


# ============================================================
# Compare models
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results = pd.DataFrame(
    [
        logistic_results,
        random_forest_results
    ]
)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# Cross-validation
# ============================================================

print("\n" + "=" * 60)
print("STRATIFIED CROSS-VALIDATION")
print("=" * 60)

# Only 11 pathogenic examples exist.
# Therefore, we cannot use a large number of folds.
#
# Five folds means each validation fold receives roughly
# two to three pathogenic examples.

cross_validation = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# Logistic Regression CV
# ============================================================

print("\nLogistic Regression CV:")

logistic_scores = cross_validate(
    logistic_model,
    X,
    y,
    cv=cross_validation,
    scoring=[
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ],
    n_jobs=-1
)

print(
    f"Balanced Accuracy: "
    f"{logistic_scores['test_balanced_accuracy'].mean():.4f}"
)

print(
    f"Precision: "
    f"{logistic_scores['test_precision'].mean():.4f}"
)

print(
    f"Recall: "
    f"{logistic_scores['test_recall'].mean():.4f}"
)

print(
    f"F1: "
    f"{logistic_scores['test_f1'].mean():.4f}"
)

print(
    f"ROC-AUC: "
    f"{logistic_scores['test_roc_auc'].mean():.4f}"
)


# ============================================================
# Random Forest CV
# ============================================================

print("\nRandom Forest CV:")

random_forest_scores = cross_validate(
    random_forest_model,
    X,
    y,
    cv=cross_validation,
    scoring=[
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ],
    n_jobs=-1
)

print(
    f"Balanced Accuracy: "
    f"{random_forest_scores['test_balanced_accuracy'].mean():.4f}"
)

print(
    f"Precision: "
    f"{random_forest_scores['test_precision'].mean():.4f}"
)

print(
    f"Recall: "
    f"{random_forest_scores['test_recall'].mean():.4f}"
)

print(
    f"F1: "
    f"{random_forest_scores['test_f1'].mean():.4f}"
)

print(
    f"ROC-AUC: "
    f"{random_forest_scores['test_roc_auc'].mean():.4f}"
)


# ============================================================
# Final note
# ============================================================

print("\n" + "=" * 60)
print("BASELINE EXPERIMENT COMPLETE")
print("=" * 60)

print(
    "\nImportant:"
)

print(
    "Only 11 pathogenic HNRNPU missense variants are "
    "available in this dataset."
)

print(
    "Therefore, these results should be treated as a "
    "small-sample baseline rather than a validated "
    "clinical predictor."
)

print(
    "\nThe next research stage is to increase biological "
    "feature richness and evaluate generalization."
)