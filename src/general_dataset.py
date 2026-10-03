import re
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "data/variant_summary.txt.gz"

OUTPUT_FILE = "data/general_missense.csv"

TARGET_GENE = "HNRNPU"


# ============================================================
# Load ClinVar
# ============================================================

print("=" * 60)
print("LOADING CLINVAR")
print("=" * 60)

df = pd.read_csv(
    INPUT_FILE,
    sep="\t",
    compression="gzip",
    low_memory=False
)

print(
    f"\nTotal ClinVar records: {len(df):,}"
)


# ============================================================
# Select required columns
# ============================================================

print("\n" + "=" * 60)
print("SELECTING REQUIRED COLUMNS")
print("=" * 60)

required_columns = [
    "#AlleleID",
    "VariationID",
    "Type",
    "Name",
    "GeneSymbol",
    "ClinicalSignificance",
    "Assembly",
    "Chromosome",
    "Start",
    "Stop",
    "ReferenceAllele",
    "AlternateAllele",
    "ReferenceAlleleVCF",
    "AlternateAlleleVCF",
    "ReviewStatus",
    "NumberSubmitters"
]

available_columns = [
    column
    for column in required_columns
    if column in df.columns
]

df = df[
    available_columns
].copy()

print(
    f"Columns retained: {len(df.columns)}"
)


# ============================================================
# Remove duplicate allele representations
# ============================================================

print("\n" + "=" * 60)
print("REMOVING DUPLICATE ALLELES")
print("=" * 60)

before = len(df)

df = df.drop_duplicates(
    subset=["#AlleleID"],
    keep="first"
).copy()

after = len(df)

print(
    f"Before: {before:,}"
)

print(
    f"After : {after:,}"
)

print(
    f"Removed: {before - after:,}"
)


# ============================================================
# Extract protein-level HGVS
# ============================================================

print("\n" + "=" * 60)
print("EXTRACTING PROTEIN CHANGES")
print("=" * 60)


def extract_protein_change(name):

    if pd.isna(name):
        return None

    match = re.search(
        r"\((p\.[^)]+)\)",
        str(name)
    )

    if match:
        return match.group(1)

    return None


df["ProteinChange"] = df[
    "Name"
].apply(
    extract_protein_change
)

protein_count = df[
    "ProteinChange"
].notna().sum()

print(
    f"Protein changes found: {protein_count:,}"
)


# ============================================================
# Identify true missense variants
# ============================================================

print("\n" + "=" * 60)
print("IDENTIFYING MISSENSE VARIANTS")
print("=" * 60)


def is_missense(protein_change):

    if pd.isna(protein_change):
        return False

    pattern = (
        r"^p\.[A-Z][a-z]{2}"
        r"\d+"
        r"[A-Z][a-z]{2}$"
    )

    match = re.match(
        pattern,
        str(protein_change)
    )

    if not match:
        return False

    # Stop codon is not an amino acid substitution.
    if "Ter" in str(protein_change):
        return False

    return True


df["IsMissense"] = df[
    "ProteinChange"
].apply(
    is_missense
)

missense = df[
    df["IsMissense"]
].copy()

print(
    f"Missense variants: {len(missense):,}"
)


# ============================================================
# Normalize clinical significance
# ============================================================

print("\n" + "=" * 60)
print("NORMALIZING CLINICAL SIGNIFICANCE")
print("=" * 60)


def classify_significance(value):

    if pd.isna(value):
        return "Other"

    value = str(
        value
    ).strip().lower()

    # Conflicting interpretations are excluded.
    if "conflicting" in value:
        return "Conflicting"

    # Combined classifications.
    if value == "pathogenic/likely pathogenic":
        return "Pathogenic"

    if value == "benign/likely benign":
        return "Benign"

    # Clear classifications.
    if value == "pathogenic":
        return "Pathogenic"

    if value == "likely pathogenic":
        return "Pathogenic"

    if value == "benign":
        return "Benign"

    if value == "likely benign":
        return "Benign"

    if value == "uncertain significance":
        return "VUS"

    return "Other"


missense["Classification"] = missense[
    "ClinicalSignificance"
].apply(
    classify_significance
)


# ============================================================
# Classification distribution
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION DISTRIBUTION")
print("=" * 60)

print(
    missense[
        "Classification"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# Keep only clear labels
# ============================================================

print("\n" + "=" * 60)
print("CREATING SUPERVISED DATASET")
print("=" * 60)

training = missense[
    missense["Classification"].isin(
        [
            "Benign",
            "Pathogenic"
        ]
    )
].copy()

training["Label"] = training[
    "Classification"
].map(
    {
        "Benign": 0,
        "Pathogenic": 1
    }
)


# ============================================================
# Remove HNRNPU from training
# ============================================================

print("\n" + "=" * 60)
print("HOLDING OUT HNRNPU")
print("=" * 60)

hnrnpu_mask = (
    training[
        "GeneSymbol"
    ]
    .astype(str)
    .str.upper()
    == TARGET_GENE
)

hnrnpu_holdout = training[
    hnrnpu_mask
].copy()

general_training = training[
    ~hnrnpu_mask
].copy()

print(
    f"\nHNRNPU variants removed from training: "
    f"{len(hnrnpu_holdout):,}"
)

print(
    f"General training variants: "
    f"{len(general_training):,}"
)


# ============================================================
# Label distribution
# ============================================================

print("\n" + "=" * 60)
print("GENERAL TRAINING LABELS")
print("=" * 60)

print(
    general_training[
        "Classification"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# Gene statistics
# ============================================================

print("\n" + "=" * 60)
print("GENE STATISTICS")
print("=" * 60)

print(
    f"Unique genes: "
    f"{general_training['GeneSymbol'].nunique():,}"
)


# ============================================================
# Select final columns
# ============================================================

final_columns = [
    "#AlleleID",
    "VariationID",
    "GeneSymbol",
    "Type",
    "Name",
    "ProteinChange",
    "ClinicalSignificance",
    "Classification",
    "Label",
    "Assembly",
    "Chromosome",
    "Start",
    "Stop",
    "ReferenceAllele",
    "AlternateAllele",
    "ReferenceAlleleVCF",
    "AlternateAlleleVCF",
    "ReviewStatus",
    "NumberSubmitters"
]

available_final_columns = [
    column
    for column in final_columns
    if column in general_training.columns
]

general_training = general_training[
    available_final_columns
].copy()


# ============================================================
# Save general dataset
# ============================================================

general_training.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("GENERAL DATASET SAVED")
print("=" * 60)

print(
    f"\nSaved to: {OUTPUT_FILE}"
)

print(
    f"Records: {len(general_training):,}"
)

print(
    f"Genes: "
    f"{general_training['GeneSymbol'].nunique():,}"
)


# ============================================================
# Verify HNRNPU exclusion
# ============================================================

remaining_hnrnpu = (
    general_training[
        "GeneSymbol"
    ]
    .astype(str)
    .str.upper()
    .eq(TARGET_GENE)
    .sum()
)

print(
    f"\nHNRNPU records remaining in training: "
    f"{remaining_hnrnpu}"
)

if remaining_hnrnpu == 0:

    print(
        "HNRNPU successfully held out."
    )

else:

    print(
        "WARNING: HNRNPU was not completely removed."
    )


# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 60)
print("GENERAL DATASET PREPARATION COMPLETE")
print("=" * 60)

print(
    f"\nGeneral training variants: "
    f"{len(general_training):,}"
)

print(
    f"Unique genes: "
    f"{general_training['GeneSymbol'].nunique():,}"
)

print("\nLabels:")

print(
    general_training[
        "Classification"
    ]
    .value_counts()
    .to_string()
)

print(
    "\nHNRNPU is completely excluded from the "
    "general training dataset."
)