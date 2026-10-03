import re
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "data/hnrnpu_variants.csv"

MISSENSE_OUTPUT = "data/hnrnpu_missense.csv"
TRAINING_OUTPUT = "data/hnrnpu_training.csv"
VUS_OUTPUT = "data/hnrnpu_vus.csv"
CONFLICTING_OUTPUT = "data/hnrnpu_conflicting.csv"


# ============================================================
# Load HNRNPU ClinVar dataset
# ============================================================

print("=" * 60)
print("Loading HNRNPU ClinVar dataset...")
print("=" * 60)

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(f"\nOriginal records: {len(df):,}")


# ============================================================
# Remove duplicate assembly representations
# ============================================================

print("\n" + "=" * 60)
print("REMOVING DUPLICATE VARIANT REPRESENTATIONS")
print("=" * 60)

before = len(df)

df = df.drop_duplicates(
    subset=["#AlleleID"],
    keep="first"
).copy()

after = len(df)

print(f"Records before: {before:,}")
print(f"Records after : {after:,}")
print(f"Removed       : {before - after:,}")


# ============================================================
# Extract protein-level change
# ============================================================

print("\n" + "=" * 60)
print("EXTRACTING PROTEIN CHANGES")
print("=" * 60)


def extract_protein_change(name):
    """
    Extract protein HGVS notation.

    Example:
        NM_031844.3(HNRNPU):c.607G>A (p.Ala203Thr)

    Returns:
        p.Ala203Thr
    """

    if pd.isna(name):
        return None

    match = re.search(
        r"\((p\.[^)]+)\)",
        str(name)
    )

    if match:
        return match.group(1)

    return None


df["ProteinChange"] = df["Name"].apply(
    extract_protein_change
)

print(
    f"Variants with protein changes: "
    f"{df['ProteinChange'].notna().sum():,}"
)


# ============================================================
# Classify protein consequence
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFYING PROTEIN CONSEQUENCES")
print("=" * 60)


def classify_protein_change(protein_change):
    """
    Classify protein-level HGVS changes.

    Missense:
        p.Ala203Thr

    Stop-gain:
        p.Arg23Ter
        p.Gln171*

    Frameshift:
        p.Gly769fs

    Synonymous:
        p.Cys389=

    In-frame deletion:
        p.Glu279del
    """

    if pd.isna(protein_change):
        return "Unknown"

    protein_change = str(
        protein_change
    ).strip()

    # --------------------------------------------------------
    # Missense
    # --------------------------------------------------------

    missense_pattern = (
        r"^p\.[A-Z][a-z]{2}"
        r"\d+"
        r"[A-Z][a-z]{2}$"
    )

    if re.match(
        missense_pattern,
        protein_change
    ):

        # Ter is NOT a valid amino acid.
        if "Ter" not in protein_change:

            return "Missense"

    # --------------------------------------------------------
    # Stop gained
    # --------------------------------------------------------

    if re.search(
        r"(Ter|\*)$",
        protein_change
    ):

        return "Stop_gained"

    # --------------------------------------------------------
    # Frameshift
    # --------------------------------------------------------

    if "fs" in protein_change:

        return "Frameshift"

    # --------------------------------------------------------
    # Synonymous
    # --------------------------------------------------------

    if protein_change.endswith("="):

        return "Synonymous"

    # --------------------------------------------------------
    # In-frame deletion
    # --------------------------------------------------------

    if "del" in protein_change:

        return "In_frame_deletion"

    # --------------------------------------------------------
    # Insertion
    # --------------------------------------------------------

    if "ins" in protein_change:

        return "Insertion"

    # --------------------------------------------------------
    # Unknown
    # --------------------------------------------------------

    return "Other"


df["ProteinConsequence"] = df[
    "ProteinChange"
].apply(
    classify_protein_change
)


# ============================================================
# Consequence distribution
# ============================================================

print("\n" + "=" * 60)
print("PROTEIN CONSEQUENCE DISTRIBUTION")
print("=" * 60)

consequence_counts = df[
    "ProteinConsequence"
].value_counts(
    dropna=False
)

print(
    consequence_counts.to_string()
)


# ============================================================
# Keep ONLY true missense variants
# ============================================================

print("\n" + "=" * 60)
print("EXTRACTING TRUE MISSENSE VARIANTS")
print("=" * 60)

missense = df[
    df["ProteinConsequence"] == "Missense"
].copy()

print(
    f"True missense variants: {len(missense):,}"
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

    # Conflicting must be checked first.
    if "conflicting" in value:
        return "Conflicting"

    # Combined classifications.
    if value == "pathogenic/likely pathogenic":
        return "Pathogenic"

    if value == "benign/likely benign":
        return "Benign"

    # Individual classifications.
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
print("MISSENSE CLASSIFICATION")
print("=" * 60)

classification_counts = missense[
    "Classification"
].value_counts(
    dropna=False
)

print(
    classification_counts.to_string()
)


# ============================================================
# Select useful columns
# ============================================================

columns = [
    "#AlleleID",
    "VariationID",
    "GeneSymbol",
    "Type",
    "Name",
    "ProteinChange",
    "ProteinConsequence",
    "ClinicalSignificance",
    "Classification",
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
    for column in columns
    if column in missense.columns
]

missense_clean = missense[
    available_columns
].copy()


# ============================================================
# Create supervised training dataset
# ============================================================

print("\n" + "=" * 60)
print("CREATING TRAINING DATASET")
print("=" * 60)

training = missense_clean[
    missense_clean["Classification"].isin(
        [
            "Pathogenic",
            "Benign"
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

print(
    f"\nTraining variants: {len(training):,}"
)

print("\nTraining label distribution:")

print(
    training["Label"]
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
# Extract VUS
# ============================================================

print("\n" + "=" * 60)
print("EXTRACTING VUS")
print("=" * 60)

vus = missense_clean[
    missense_clean["Classification"] == "VUS"
].copy()

print(
    f"VUS missense variants: {len(vus):,}"
)


# ============================================================
# Extract conflicting variants
# ============================================================

print("\n" + "=" * 60)
print("EXTRACTING CONFLICTING VARIANTS")
print("=" * 60)

conflicting = missense_clean[
    missense_clean["Classification"] == "Conflicting"
].copy()

print(
    f"Conflicting missense variants: "
    f"{len(conflicting):,}"
)


# ============================================================
# Save datasets
# ============================================================

print("\n" + "=" * 60)
print("SAVING DATASETS")
print("=" * 60)

missense_clean.to_csv(
    MISSENSE_OUTPUT,
    index=False
)

training.to_csv(
    TRAINING_OUTPUT,
    index=False
)

vus.to_csv(
    VUS_OUTPUT,
    index=False
)

conflicting.to_csv(
    CONFLICTING_OUTPUT,
    index=False
)

print("\nSaved:")

print(
    f" - {MISSENSE_OUTPUT}"
)

print(
    f" - {TRAINING_OUTPUT}"
)

print(
    f" - {VUS_OUTPUT}"
)

print(
    f" - {CONFLICTING_OUTPUT}"
)


# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETE")
print("=" * 60)

print(
    f"\nUnique ClinVar alleles : {len(df):,}"
)

print(
    f"Missense variants      : {len(missense_clean):,}"
)

print(
    f"Training variants      : {len(training):,}"
)

print(
    f"VUS variants           : {len(vus):,}"
)

print(
    f"Conflicting variants   : {len(conflicting):,}"
)

print("\nTraining labels:")

print(
    training[
        "Classification"
    ]
    .value_counts()
    .to_string()
)

print(
    "\nDataset preparation complete."
)