import re
import pandas as pd
import numpy as np


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "data/hnrnpu_training.csv"
OUTPUT_FILE = "data/hnrnpu_features.csv"


# ============================================================
# Amino-acid properties
# ============================================================

AMINO_ACIDS = {

    "Ala": {
        "one": "A",
        "hydrophobicity": 1.8,
        "volume": 88.6,
        "polarity": 0,
        "charge": 0
    },

    "Arg": {
        "one": "R",
        "hydrophobicity": -4.5,
        "volume": 173.4,
        "polarity": 1,
        "charge": 1
    },

    "Asn": {
        "one": "N",
        "hydrophobicity": -3.5,
        "volume": 114.1,
        "polarity": 1,
        "charge": 0
    },

    "Asp": {
        "one": "D",
        "hydrophobicity": -3.5,
        "volume": 111.1,
        "polarity": 1,
        "charge": -1
    },

    "Cys": {
        "one": "C",
        "hydrophobicity": 2.5,
        "volume": 108.5,
        "polarity": 1,
        "charge": 0
    },

    "Gln": {
        "one": "Q",
        "hydrophobicity": -3.5,
        "volume": 143.8,
        "polarity": 1,
        "charge": 0
    },

    "Glu": {
        "one": "E",
        "hydrophobicity": -3.5,
        "volume": 138.4,
        "polarity": 1,
        "charge": -1
    },

    "Gly": {
        "one": "G",
        "hydrophobicity": -0.4,
        "volume": 60.1,
        "polarity": 0,
        "charge": 0
    },

    "His": {
        "one": "H",
        "hydrophobicity": -3.2,
        "volume": 153.2,
        "polarity": 1,
        "charge": 1
    },

    "Ile": {
        "one": "I",
        "hydrophobicity": 4.5,
        "volume": 166.7,
        "polarity": 0,
        "charge": 0
    },

    "Leu": {
        "one": "L",
        "hydrophobicity": 3.8,
        "volume": 166.7,
        "polarity": 0,
        "charge": 0
    },

    "Lys": {
        "one": "K",
        "hydrophobicity": -3.9,
        "volume": 168.6,
        "polarity": 1,
        "charge": 1
    },

    "Met": {
        "one": "M",
        "hydrophobicity": 1.9,
        "volume": 162.9,
        "polarity": 0,
        "charge": 0
    },

    "Phe": {
        "one": "F",
        "hydrophobicity": 2.8,
        "volume": 189.9,
        "polarity": 0,
        "charge": 0
    },

    "Pro": {
        "one": "P",
        "hydrophobicity": -1.6,
        "volume": 112.7,
        "polarity": 0,
        "charge": 0
    },

    "Ser": {
        "one": "S",
        "hydrophobicity": -0.8,
        "volume": 89.0,
        "polarity": 1,
        "charge": 0
    },

    "Thr": {
        "one": "T",
        "hydrophobicity": -0.7,
        "volume": 116.1,
        "polarity": 1,
        "charge": 0
    },

    "Trp": {
        "one": "W",
        "hydrophobicity": -0.9,
        "volume": 227.8,
        "polarity": 0,
        "charge": 0
    },

    "Tyr": {
        "one": "Y",
        "hydrophobicity": -1.3,
        "volume": 193.6,
        "polarity": 1,
        "charge": 0
    },

    "Val": {
        "one": "V",
        "hydrophobicity": 4.2,
        "volume": 140.0,
        "polarity": 0,
        "charge": 0
    }
}


# ============================================================
# Load training data
# ============================================================

print("=" * 60)
print("LOADING HNRNPU MISSENSE TRAINING DATA")
print("=" * 60)

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(
    f"\nTraining records: {len(df):,}"
)


# ============================================================
# Safety check
# ============================================================

print("\n" + "=" * 60)
print("VALIDATING MISSENSE DATA")
print("=" * 60)

if "ProteinConsequence" in df.columns:

    print(
        df[
            "ProteinConsequence"
        ]
        .value_counts()
        .to_string()
    )

    non_missense = (
        df["ProteinConsequence"]
        != "Missense"
    ).sum()

    if non_missense > 0:

        print(
            f"\nWARNING: {non_missense} "
            "non-missense variants detected."
        )

        df = df[
            df["ProteinConsequence"]
            == "Missense"
        ].copy()

        print(
            f"Remaining variants: {len(df):,}"
        )


# ============================================================
# Parse protein HGVS
# ============================================================

print("\n" + "=" * 60)
print("PARSING PROTEIN CHANGES")
print("=" * 60)


def parse_protein_change(protein_change):

    if pd.isna(protein_change):
        return None, None, None

    pattern = (
        r"^p\.([A-Z][a-z]{2})"
        r"(\d+)"
        r"([A-Z][a-z]{2})$"
    )

    match = re.match(
        pattern,
        str(protein_change)
    )

    if not match:
        return None, None, None

    original = match.group(1)
    position = int(match.group(2))
    new = match.group(3)

    return original, position, new


parsed = df[
    "ProteinChange"
].apply(
    parse_protein_change
)

df[
    [
        "OriginalAA",
        "ProteinPosition",
        "NewAA"
    ]
] = pd.DataFrame(
    parsed.tolist(),
    index=df.index
)


# ============================================================
# Validate amino acids
# ============================================================

valid_amino_acids = set(
    AMINO_ACIDS.keys()
)

df = df[
    df["OriginalAA"].isin(
        valid_amino_acids
    )
    &
    df["NewAA"].isin(
        valid_amino_acids
    )
].copy()

print(
    f"Valid missense variants: {len(df):,}"
)


# ============================================================
# Amino-acid property helper
# ============================================================

def get_property(
    amino_acid,
    property_name
):

    return AMINO_ACIDS[
        amino_acid
    ][property_name]


# ============================================================
# Original amino-acid properties
# ============================================================

df["OriginalHydrophobicity"] = df[
    "OriginalAA"
].apply(
    lambda x: get_property(
        x,
        "hydrophobicity"
    )
)

df["OriginalVolume"] = df[
    "OriginalAA"
].apply(
    lambda x: get_property(
        x,
        "volume"
    )
)

df["OriginalPolarity"] = df[
    "OriginalAA"
].apply(
    lambda x: get_property(
        x,
        "polarity"
    )
)

df["OriginalCharge"] = df[
    "OriginalAA"
].apply(
    lambda x: get_property(
        x,
        "charge"
    )
)


# ============================================================
# New amino-acid properties
# ============================================================

df["NewHydrophobicity"] = df[
    "NewAA"
].apply(
    lambda x: get_property(
        x,
        "hydrophobicity"
    )
)

df["NewVolume"] = df[
    "NewAA"
].apply(
    lambda x: get_property(
        x,
        "volume"
    )
)

df["NewPolarity"] = df[
    "NewAA"
].apply(
    lambda x: get_property(
        x,
        "polarity"
    )
)

df["NewCharge"] = df[
    "NewAA"
].apply(
    lambda x: get_property(
        x,
        "charge"
    )
)


# ============================================================
# Calculate differences
# ============================================================

df["HydrophobicityDifference"] = (
    df["NewHydrophobicity"]
    - df["OriginalHydrophobicity"]
)

df["VolumeDifference"] = (
    df["NewVolume"]
    - df["OriginalVolume"]
)

df["PolarityDifference"] = (
    df["NewPolarity"]
    - df["OriginalPolarity"]
)

df["ChargeDifference"] = (
    df["NewCharge"]
    - df["OriginalCharge"]
)


# ============================================================
# Position features
# ============================================================

df["ProteinPosition"] = (
    df["ProteinPosition"]
    .astype(int)
)


# ============================================================
# Broad protein regions
# ============================================================

def classify_region(position):

    if position <= 200:
        return "N_terminal"

    if position <= 400:
        return "N_mid"

    if position <= 600:
        return "C_mid"

    return "C_terminal"


df["ProteinRegion"] = df[
    "ProteinPosition"
].apply(
    classify_region
)


# ============================================================
# Biological similarity indicators
# ============================================================

df["SameCharge"] = (
    df["OriginalCharge"]
    ==
    df["NewCharge"]
).astype(int)

df["SamePolarity"] = (
    df["OriginalPolarity"]
    ==
    df["NewPolarity"]
).astype(int)

df["SameHydrophobicityClass"] = (
    (
        df["OriginalHydrophobicity"] > 0
    )
    ==
    (
        df["NewHydrophobicity"] > 0
    )
).astype(int)


# ============================================================
# Amino-acid change
# ============================================================

df["AminoAcidChange"] = (
    df["OriginalAA"]
    + "_to_"
    + df["NewAA"]
)


# ============================================================
# Display engineered features
# ============================================================

print("\n" + "=" * 60)
print("ENGINEERED FEATURES")
print("=" * 60)

display_columns = [
    "ProteinChange",
    "OriginalAA",
    "ProteinPosition",
    "NewAA",
    "HydrophobicityDifference",
    "VolumeDifference",
    "PolarityDifference",
    "ChargeDifference",
    "SameCharge",
    "SamePolarity",
    "SameHydrophobicityClass",
    "AminoAcidChange",
    "Classification",
    "Label"
]

print(
    df[
        display_columns
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# Missing-value check
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = df[
    display_columns
].isnull().sum()

missing = missing[
    missing > 0
]

if len(missing) == 0:

    print(
        "No missing values found."
    )

else:

    print(
        missing.to_string()
    )


# ============================================================
# Final feature dataset
# ============================================================

print("\n" + "=" * 60)
print("CREATING FINAL FEATURE DATASET")
print("=" * 60)

final_columns = [
    "#AlleleID",
    "VariationID",

    "ProteinChange",
    "OriginalAA",
    "NewAA",
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

    "AminoAcidChange",
    "ProteinRegion",

    "Classification",
    "Label"
]

available_columns = [
    column
    for column in final_columns
    if column in df.columns
]

features = df[
    available_columns
].copy()


# ============================================================
# Save feature dataset
# ============================================================

features.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nSaved feature dataset to:"
)

print(
    f"  {OUTPUT_FILE}"
)


# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 60)

print(
    f"\nTotal variants : {len(features):,}"
)

print(
    f"Total features: {len(features.columns):,}"
)

print("\nClassification distribution:")

print(
    features[
        "Classification"
    ]
    .value_counts()
    .to_string()
)

print("\nFeature columns:")

for index, column in enumerate(
    features.columns,
    start=1
):

    print(
        f"{index:2}. {column}"
    )

print(
    "\nFeature engineering complete."
)