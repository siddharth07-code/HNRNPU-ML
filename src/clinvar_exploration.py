import pandas as pd


# ============================================================
# Configuration
# ============================================================

DATASET = "data/variant_summary.txt.gz"
TARGET_GENE = "HNRNPU"


# ============================================================
# Load ClinVar dataset
# ============================================================

print("=" * 60)
print("Loading ClinVar dataset...")
print("=" * 60)

df = pd.read_csv(
    DATASET,
    sep="\t",
    compression="gzip",
    low_memory=False
)

print(f"\nTotal records: {len(df):,}")
print(f"Total columns: {len(df.columns)}")


# ============================================================
# Display all available columns
# ============================================================

print("\n" + "=" * 60)
print("AVAILABLE COLUMNS")
print("=" * 60)

for index, column in enumerate(df.columns, start=1):
    print(f"{index:3}. {column}")


# ============================================================
# Extract HNRNPU variants
# ============================================================

print("\n" + "=" * 60)
print("HNRNPU VARIANTS")
print("=" * 60)

hnrnpu = df[
    df["GeneSymbol"]
    .astype(str)
    .str.upper()
    .eq(TARGET_GENE)
].copy()

print(f"\nHNRNPU records: {len(hnrnpu):,}")


# ============================================================
# Important ClinVar columns
# ============================================================

print("\n" + "=" * 60)
print("IMPORTANT CLINVAR COLUMNS")
print("=" * 60)

important_keywords = [
    "Name",
    "Position",
    "Chromosome",
    "Reference",
    "Alternate",
    "Assembly",
    "Type",
    "Gene",
    "Allele",
    "Condition"
]

for column in hnrnpu.columns:

    if any(
        keyword.lower() in column.lower()
        for keyword in important_keywords
    ):
        print(f" - {column}")


# ============================================================
# Clinical significance
# ============================================================

print("\n" + "=" * 60)
print("CLINICAL SIGNIFICANCE")
print("=" * 60)

clinical = hnrnpu[
    "ClinicalSignificance"
].value_counts(
    dropna=False
)

print(clinical.to_string())


# ============================================================
# Variant types
# ============================================================

print("\n" + "=" * 60)
print("VARIANT TYPES")
print("=" * 60)

if "Type" in hnrnpu.columns:

    variant_types = hnrnpu[
        "Type"
    ].value_counts(
        dropna=False
    )

    print(variant_types.to_string())

else:

    print("\nType column was not found.")


# ============================================================
# Sample HNRNPU variants
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE HNRNPU VARIANTS")
print("=" * 60)

sample_columns = [
    "#AlleleID",
    "Type",
    "Name",
    "ClinicalSignificance",
    "Assembly",
    "Chromosome",
    "PositionVCF",
    "ReferenceAlleleVCF",
    "AlternateAlleleVCF"
]

available_columns = [
    column
    for column in sample_columns
    if column in hnrnpu.columns
]

print(
    hnrnpu[
        available_columns
    ]
    .head(30)
    .to_string(index=False)
)


# ============================================================
# Inspect variant names
# ============================================================

print("\n" + "=" * 60)
print("VARIANT NAMES")
print("=" * 60)

if "Name" in hnrnpu.columns:

    names = hnrnpu[
        "Name"
    ].dropna()

    for index, name in enumerate(
        names.head(30),
        start=1
    ):
        print(f"{index:2}. {name}")

else:

    print("\nName column was not found.")


# ============================================================
# Save HNRNPU dataset
# ============================================================

output = "data/hnrnpu_variants.csv"

hnrnpu.to_csv(
    output,
    index=False
)


# ============================================================
# Completion summary
# ============================================================

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)

print(f"\nSaved to: {output}")
print(f"Records : {len(hnrnpu):,}")
print(f"Columns : {len(hnrnpu.columns):,}")

print("\nNext step:")
print("Inspect the Type and Name fields before preparing")
print("the machine-learning dataset.")