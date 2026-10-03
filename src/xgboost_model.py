import os
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

INPUT_FILE = "data/general_features.csv"
RESULTS_DIR = "results"
RANDOM_STATE = 42

print("=" * 60)
print("XGBOOST GENERAL VARIANT MODEL")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"\nRecords: {len(df):,}")
print(f"Genes: {df['GeneSymbol'].nunique():,}")

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

X = df[feature_columns]
y = df["Label"].astype(int)
groups = df["GeneSymbol"].astype(str)

print(f"Features: {len(feature_columns)}")
print(f"X shape: {X.shape}")

benign = (y == 0).sum()
pathogenic = (y == 1).sum()
scale_pos_weight = benign / pathogenic

print(f"\nBenign: {benign:,}")
print(f"Pathogenic: {pathogenic:,}")
print(f"Scale pos weight: {scale_pos_weight:.4f}")


def create_model():
    return XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Balanced Accuracy": balanced_accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "PR-AUC": average_precision_score(y_test, probabilities)
    }

    return results, predictions


# ============================================================
# RANDOM SPLIT
# ============================================================

print("\n" + "=" * 60)
print("RANDOM STRATIFIED SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

model = create_model()
random_results, random_predictions = evaluate_model(
    model, X_train, X_test, y_train, y_test
)

for metric, value in random_results.items():
    print(f"{metric:20s}: {value:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, random_predictions))


# ============================================================
# GENE-AWARE SPLIT
# ============================================================

print("\n" + "=" * 60)
print("GENE-AWARE SPLIT")
print("=" * 60)

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    gss.split(X, y, groups=groups)
)

X_train_gene = X.iloc[train_idx]
X_test_gene = X.iloc[test_idx]
y_train_gene = y.iloc[train_idx]
y_test_gene = y.iloc[test_idx]

genes_train = groups.iloc[train_idx]
genes_test = groups.iloc[test_idx]

print(f"Training variants: {len(X_train_gene):,}")
print(f"Testing variants: {len(X_test_gene):,}")
print(f"Training genes: {genes_train.nunique():,}")
print(f"Testing genes: {genes_test.nunique():,}")

overlap = set(genes_train) & set(genes_test)
print(f"Gene overlap: {len(overlap)}")

if overlap:
    raise ValueError("Gene leakage detected!")

model_gene = create_model()

gene_results, gene_predictions = evaluate_model(
    model_gene,
    X_train_gene,
    X_test_gene,
    y_train_gene,
    y_test_gene
)

for metric, value in gene_results.items():
    print(f"{metric:20s}: {value:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_gene, gene_predictions))


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model_gene.feature_importances_
}).sort_values("Importance", ascending=False)

print(importance.to_string(index=False))


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(RESULTS_DIR, exist_ok=True)

results_df = pd.DataFrame([
    {
        "Model": "XGBoost",
        "Split": "Random",
        **random_results
    },
    {
        "Model": "XGBoost",
        "Split": "Gene-aware",
        **gene_results
    }
])

results_file = os.path.join(
    RESULTS_DIR,
    "xgboost_results.csv"
)

importance_file = os.path.join(
    RESULTS_DIR,
    "xgboost_feature_importance.csv"
)

results_df.to_csv(results_file, index=False)
importance.to_csv(importance_file, index=False)

print("\n" + "=" * 60)
print("XGBOOST COMPLETE")
print("=" * 60)

print(f"Results saved: {results_file}")
print(f"Feature importance saved: {importance_file}")

print("\nResults:")
print(results_df.to_string(
    index=False,
    float_format=lambda x: f"{x:.4f}"
))

print("\nTop 10 features:")
print(importance.head(10).to_string(index=False))