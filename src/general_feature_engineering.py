import os
import re
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/general_missense.csv"
OUTPUT_FILE = "data/general_features.csv"


# ============================================================
# AMINO ACID PROPERTIES
# ============================================================

HYDROPHOBICITY = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2
}

VOLUME = {
    "A": 88.6,
    "R": 173.4,
    "N": 114.1,
    "D": 111.1,
    "C": 108.5,
    "Q": 143.8,
    "E": 138.4,
    "G": 60.1,
    "H": 153.2,
    "I": 166.7,
    "L": 166.7,
    "K": 168.6,
    "M": 162.9,
    "F": 189.9,
    "P": 112.7,
    "S": 89.0,
    "T": 116.1,
    "W": 227.8,
    "Y": 193.6,
    "V": 140.0
}

POLARITY = {
    "A": 8.1,
    "R": 10.5,
    "N": 11.6,
    "D": 13.0,
    "C": 5.5,
    "Q": 10.5,
    "E": 12.3,
    "G": 9.0,
    "H": 10.4,
    "I": 5.2,
    "L": 4.9,
    "K": 11.3,
    "M": 5.7,
    "F": 5.2,
    "P": 8.0,
    "S": 9.2,
    "T": 8.6,
    "W": 5.4,
    "Y": 6.2,
    "V": 5.9
}

CHARGE = {
    "A": 0,
    "R": 1,
    "N": 0,
    "D": -1,
    "C": 0,
    "Q": 0,
    "E": -1,
    "G": 0,
    "H": 1,
    "I": 0,
    "L": 0,
    "K": 1,
    "M": 0,
    "F": 0,
    "P": 0,
    "S": 0,
    "T": 0,
    "W": 0,
    "Y": 0,
    "V": 0
}


# ============================================================
# AMINO ACID GROUPS
# ============================================================

HYDROPHOBIC = set("AVILMFWY")
POLAR = set("STNQ")
POSITIVELY_CHARGED = set("KRH")
NEGATIVELY_CHARGED = set("DE")
SPECIAL = set("CGP")


def amino_acid_group(aa):
    """
    Assign a broad biochemical group to an amino acid.
    """

    if aa in HYDROPHOBIC:
        return "Hydrophobic"

    if aa in POLAR:
        return "Polar"

    if aa in POSITIVELY_CHARGED:
        return "Positive"

    if aa in NEGATIVELY_CHARGED:
        return "Negative"

    if aa in SPECIAL:
        return "Special"

    return "Other"


# ============================================================
# PARSE PROTEIN CHANGE
# ============================================================

def parse_protein_change(protein_change):
    """
    Convert:

        p.Ala203Thr

    into:

        OriginalAA = A
        Position   = 203
        NewAA      = T
    """

    if pd.isna(protein_change):
        return None, None, None

    protein_change = str(protein_change).strip()

    match = re.fullmatch(
        r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})",
        protein_change
    )

    if not match:
        return None, None, None

    original_three = match.group(1)
    position = int(match.group(2))
    new_three = match.group(3)

    aa_map = {
        "Ala": "A",
        "Arg": "R",
        "Asn": "N",
        "Asp": "D",
        "Cys": "C",
        "Gln": "Q",
        "Glu": "E",
        "Gly": "G",
        "His": "H",
        "Ile": "I",
        "Leu": "L",
        "Lys": "K",
        "Met": "M",
        "Phe": "F",
        "Pro": "P",
        "Ser": "S",
        "Thr": "T",
        "Trp": "W",
        "Tyr": "Y",
        "Val": "V"
    }

    original_aa = aa_map.get(original_three)
    new_aa = aa_map.get(new_three)

    if original_aa is None or new_aa is None:
        return None, None, None

    return original_aa, position, new_aa


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("GENERAL FEATURE ENGINEERING")
print("=" * 60)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("LOADING GENERAL DATASET")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"Records loaded: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ------------------------------------------------------------
# VERIFY CLASSIFICATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("VERIFYING CLASSIFICATION")
print("=" * 60)

print(df["Classification"].value_counts())


# Only supervised labels
df = df[df["Classification"].isin(["Benign", "Pathogenic"])].copy()

print(f"\nSupervised records: {len(df):,}")


# ------------------------------------------------------------
# VERIFY HNRNPU IS ABSENT
# ------------------------------------------------------------

if "GeneSymbol" in df.columns:

    hnrnpu_count = (
        df["GeneSymbol"]
        .astype(str)
        .str.upper()
        .eq("HNRNPU")
        .sum()
    )

    print(f"HNRNPU records present: {hnrnpu_count}")

    if hnrnpu_count != 0:
        raise ValueError(
            "HNRNPU contamination detected in general dataset!"
        )


# ------------------------------------------------------------
# PARSE PROTEIN CHANGES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PARSING PROTEIN CHANGES")
print("=" * 60)

parsed = df["ProteinChange"].apply(parse_protein_change)

df["OriginalAA"] = parsed.apply(lambda x: x[0])
df["ProteinPosition"] = parsed.apply(lambda x: x[1])
df["NewAA"] = parsed.apply(lambda x: x[2])


valid = (
    df["OriginalAA"].notna()
    & df["ProteinPosition"].notna()
    & df["NewAA"].notna()
)

df = df[valid].copy()

print(f"Valid missense variants: {len(df):,}")


# ------------------------------------------------------------
# PHYSICOCHEMICAL FEATURES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CREATING PHYSICOCHEMICAL FEATURES")
print("=" * 60)

df["OriginalHydrophobicity"] = (
    df["OriginalAA"].map(HYDROPHOBICITY)
)

df["NewHydrophobicity"] = (
    df["NewAA"].map(HYDROPHOBICITY)
)

df["HydrophobicityDifference"] = (
    df["NewHydrophobicity"]
    - df["OriginalHydrophobicity"]
)


df["OriginalVolume"] = (
    df["OriginalAA"].map(VOLUME)
)

df["NewVolume"] = (
    df["NewAA"].map(VOLUME)
)

df["VolumeDifference"] = (
    df["NewVolume"]
    - df["OriginalVolume"]
)


df["OriginalPolarity"] = (
    df["OriginalAA"].map(POLARITY)
)

df["NewPolarity"] = (
    df["NewAA"].map(POLARITY)
)

df["PolarityDifference"] = (
    df["NewPolarity"]
    - df["OriginalPolarity"]
)


df["OriginalCharge"] = (
    df["OriginalAA"].map(CHARGE)
)

df["NewCharge"] = (
    df["NewAA"].map(CHARGE)
)

df["ChargeDifference"] = (
    df["NewCharge"]
    - df["OriginalCharge"]
)


# ------------------------------------------------------------
# BIOCHEMICAL GROUP FEATURES
# ------------------------------------------------------------

df["OriginalGroup"] = (
    df["OriginalAA"].apply(amino_acid_group)
)

df["NewGroup"] = (
    df["NewAA"].apply(amino_acid_group)
)

df["SameGroup"] = (
    df["OriginalGroup"] == df["NewGroup"]
).astype(int)


df["SameCharge"] = (
    df["OriginalCharge"] == df["NewCharge"]
).astype(int)

df["SamePolarity"] = (
    (df["OriginalPolarity"] - df["NewPolarity"]).abs() < 1e-9
).astype(int)


df["SameHydrophobicityClass"] = (
    df["OriginalAA"].isin(HYDROPHOBIC)
    == df["NewAA"].isin(HYDROPHOBIC)
).astype(int)


# ------------------------------------------------------------
# AMINO ACID CHANGE
# ------------------------------------------------------------

df["AminoAcidChange"] = (
    df["OriginalAA"] + ">" + df["NewAA"]
)


# ------------------------------------------------------------
# LABEL
# ------------------------------------------------------------

df["Label"] = (
    df["Classification"] == "Pathogenic"
).astype(int)


# ------------------------------------------------------------
# CLEAN NUMERICAL FEATURES
# ------------------------------------------------------------

numerical_features = [
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


df[numerical_features] = (
    df[numerical_features]
    .apply(pd.to_numeric, errors="coerce")
)


# ------------------------------------------------------------
# REMOVE MISSING VALUES
# ------------------------------------------------------------

before = len(df)

df = df.dropna(
    subset=numerical_features + ["Label"]
).copy()

after = len(df)

print(f"\nRows before cleaning: {before:,}")
print(f"Rows after cleaning : {after:,}")
print(f"Rows removed        : {before - after:,}")


# ------------------------------------------------------------
# FINAL FEATURE LIST
# ------------------------------------------------------------

metadata_columns = [
    "#AlleleID",
    "VariationID",
    "GeneSymbol",
    "ProteinChange",
    "OriginalAA",
    "NewAA",
    "ProteinPosition",
    "OriginalGroup",
    "NewGroup",
    "AminoAcidChange",
    "Classification",
    "Label"
]

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
    "SameGroup",

    "OriginalGroup",
    "NewGroup",
    "AminoAcidChange"
]


# Keep only columns that actually exist
final_columns = [
    col
    for col in metadata_columns + feature_columns
    if col in df.columns
]

df = df[final_columns].copy()


# ------------------------------------------------------------
# DATA QUALITY CHECKS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATA QUALITY CHECK")
print("=" * 60)

print(f"Final records: {len(df):,}")

print("\nMissing values:")
print(
    df.isnull()
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nLabel distribution:")
print(df["Classification"].value_counts())

print("\nLabel percentages:")
print(
    df["Classification"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ------------------------------------------------------------
# GENE STATISTICS
# ------------------------------------------------------------

if "GeneSymbol" in df.columns:

    print("\n" + "=" * 60)
    print("GENE STATISTICS")
    print("=" * 60)

    print(
        f"Unique genes: "
        f"{df['GeneSymbol'].nunique():,}"
    )


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

os.makedirs("data", exist_ok=True)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("GENERAL FEATURE ENGINEERING COMPLETE")
print("=" * 60)

print(f"Saved to: {OUTPUT_FILE}")
print(f"Records: {len(df):,}")

print(
    f"Features available: "
    f"{len(feature_columns)}"
)

print("\nLabel distribution:")
print(df["Classification"].value_counts())

print("\nHNRNPU contamination check:")

if "GeneSymbol" in df.columns:
    remaining_hnrnpu = (
        df["GeneSymbol"]
        .astype(str)
        .str.upper()
        .eq("HNRNPU")
        .sum()
    )

    print(
        f"HNRNPU records: {remaining_hnrnpu}"
    )

print("=" * 60)