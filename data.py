"""
data.py - Synthetic Dataset Generator for Rare Disease Diagnosis in Federated Learning.


Demonstrates Non-IID (Independent and Identically Distributed) clinical data across sites.
Raw data is encapsulated within hospital instances and NEVER transmitted to the central server.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_NAMES = [
    "age",
    "genetic_mutation_score",   # 0.0 - 10.0 pathogenic load
    "serum_protein_pg_ml",       # Biomarker protein level (pg/mL)
    "inflammatory_index",        # Biomarker score 0.0 - 5.0
    "gene_expression_alpha",     # RNA-seq expression z-score (-3.0 to 3.0)
    "cellular_decay_rate",       # Metabolic decay rate (0.0 to 1.0)
    "family_history_score"       # Ordinal risk score (0 to 3)
]

TARGET_NAME = "diagnosis"  # 0: Negative/Control, 1: Positive for Rare Disease X

HOSPITAL_METADATA = {
    "hospital_1": {
        "name": "Hospital A",
        "location": "Memphis, TN",
        "specialty": "Pediatric & Genetic Subtypes",
        "samples": 300,
        "color": "#1f77b4"
    },
    "hospital_2": {
        "name": "Hospital B",
        "location": "Rochester, MN",
        "specialty": "Adult Biomarker & Protein Profiles",
        "samples": 400,
        "color": "#ff7f0e"
    },
    "hospital_3": {
        "name": "Hospital C",
        "location": "Baltimore, MD",
        "specialty": "General Clinical Population",
        "samples": 250,
        "color": "#2ca02c"
    }
}


def generate_hospital_dataset(hospital_id: str, seed: int = 42) -> pd.DataFrame:
    """
    Generates realistic non-IID clinical patient features for a given hospital.
    """

    HOSPITAL_SEEDS = {
        "hospital_1": 101,
        "hospital_2": 202,
        "hospital_3": 303
    }

    np.random.seed(seed + HOSPITAL_SEEDS[hospital_id])
    meta = HOSPITAL_METADATA[hospital_id]
    n = meta["samples"]

    if hospital_id == "hospital_1":
        # St. Jude: Younger demographic, higher genetic mutation prevalence
        age = np.random.normal(loc=28, scale=10, size=n).clip(5, 60)
        mutation = np.random.beta(a=3, b=2, size=n) * 10.0
        protein = np.random.normal(loc=120, scale=35, size=n).clip(20, 250)
        inflam = np.random.exponential(scale=1.2, size=n).clip(0, 5)
        expression = np.random.normal(loc=0.8, scale=1.2, size=n)
        decay = np.random.uniform(0.1, 0.9, size=n)
        fam_hist = np.random.choice([0, 1, 2, 3], size=n, p=[0.2, 0.3, 0.3, 0.2])

    elif hospital_id == "hospital_2":
        # Mayo Clinic: Older demographic, higher serum protein levels
        age = np.random.normal(loc=58, scale=12, size=n).clip(25, 85)
        mutation = np.random.beta(a=1.5, b=3, size=n) * 10.0
        protein = np.random.normal(loc=210, scale=45, size=n).clip(50, 350)
        inflam = np.random.normal(loc=2.8, scale=1.0, size=n).clip(0, 5)
        expression = np.random.normal(loc=-0.2, scale=1.0, size=n)
        decay = np.random.uniform(0.3, 0.95, size=n)
        fam_hist = np.random.choice([0, 1, 2, 3], size=n, p=[0.4, 0.3, 0.2, 0.1])

    else: # hospital_3
        # Johns Hopkins: Moderate demographic, wide variability
        age = np.random.normal(loc=42, scale=15, size=n).clip(18, 75)
        mutation = np.random.beta(a=2, b=2, size=n) * 10.0
        protein = np.random.normal(loc=150, scale=40, size=n).clip(30, 280)
        inflam = np.random.normal(loc=1.8, scale=1.1, size=n).clip(0, 5)
        expression = np.random.normal(loc=0.1, scale=1.1, size=n)
        decay = np.random.uniform(0.05, 0.8, size=n)
        fam_hist = np.random.choice([0, 1, 2, 3], size=n, p=[0.35, 0.35, 0.2, 0.1])

    # Latent true clinical risk formula for ground truth label assignment
    logit = (
        0.35 * (mutation - 5.0) +
        0.015 * (protein - 150.0) +
        0.4 * inflam +
        0.45 * expression +
        0.8 * (fam_hist - 1.0) +
        0.02 * (age - 40.0)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    diagnosis = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        "age": np.round(age, 1),
        "genetic_mutation_score": np.round(mutation, 2),
        "serum_protein_pg_ml": np.round(protein, 1),
        "inflammatory_index": np.round(inflam, 2),
        "gene_expression_alpha": np.round(expression, 2),
        "cellular_decay_rate": np.round(decay, 3),
        "family_history_score": fam_hist,
        TARGET_NAME: diagnosis
    })

    return df


def load_all_hospital_datasets(seed: int = 42):
    """
    Creates dataset splits for all 3 hospital clients plus a pooled global test set.
    """
    hospital_data = {}
    test_splits = []

    for h_id in HOSPITAL_METADATA.keys():
        df = generate_hospital_dataset(h_id, seed=seed)
        X = df[FEATURE_NAMES]
        y = df[TARGET_NAME]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y
        )
        hospital_data[h_id] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_test,
            "y_val": y_test,
            "full_df": df
        }
        test_splits.append((X_test, y_test))

    # Global benchmark test set pooled from all hospital validation splits
    X_global_test = pd.concat([ts[0] for ts in test_splits], axis=0)
    y_global_test = pd.concat([ts[1] for ts in test_splits], axis=0)

    return hospital_data, X_global_test, y_global_test