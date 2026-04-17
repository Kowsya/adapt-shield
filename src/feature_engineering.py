"""
STEP 3: FEATURE ENGINEERING
============================
Cleans, encodes, and scales the dataset for model training.
This is called automatically by train.py — you don't need to run it separately.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/combined_data.parquet")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Columns to DROP (zero variance or irrelevant)
COLS_TO_DROP = [
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "CWE Flag Count",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
]


def load_data() -> pd.DataFrame:
    """Load the combined parquet file."""
    print("📂 Loading combined data...")
    df = pd.read_parquet(DATA_PATH)
    print(f"   Loaded: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, handle nulls and infinities."""
    print("🧹 Cleaning data...")

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"   Removed {before - len(df):,} duplicates")

    # Drop low-value columns
    existing_drops = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=existing_drops, errors="ignore")

    # Replace infinite values with NaN, then fill with column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Clip extreme outliers using 99.9th percentile
    for col in numeric_cols:
        cap = df[col].quantile(0.999)
        df[col] = df[col].clip(upper=cap)

    print(f"   Clean shape: {df.shape}")
    return df


def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """Convert string labels to integers. Returns df + encoder."""
    print("🏷️  Encoding labels...")
    le = LabelEncoder()
    df["Label"] = le.fit_transform(df["Label"])
    print(f"   Classes: {list(le.classes_)}")
    # Save encoder
    joblib.dump(le, MODELS_DIR / "label_encoder.pkl")
    print(f"   Saved label encoder → models/label_encoder.pkl")
    return df, le


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Standardize features (mean=0, std=1)."""
    print("📏 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Save scaler for inference time
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print("   Saved scaler → models/scaler.pkl")
    return X_train_scaled, X_test_scaled, scaler


def prepare_data(sample_frac: float = 0.3) -> tuple:
    """
    Full pipeline: load → clean → encode → split → scale.
    
    sample_frac: Use fraction of data (0.3 = 30%) to speed up training.
                 Set to 1.0 for full dataset (takes longer).
    Returns: X_train, X_test, y_train, y_test, feature_names, label_encoder
    """
    # Load
    df = load_data()

    # Sample for speed (optional — remove for full training)
    if sample_frac < 1.0:
        print(f"📉 Sampling {sample_frac*100:.0f}% of data for speed...")
        df = df.groupby("Label", group_keys=False).apply(
            lambda x: x.sample(frac=sample_frac, random_state=42)
        )
        print(f"   Sampled shape: {df.shape}")

    # Clean
    df = clean_data(df)

    # Separate features and target
    X = df.drop(columns=["Label"])
    y_raw = df["Label"]

    # Encode labels (only if string type)
    if y_raw.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        joblib.dump(le, MODELS_DIR / "label_encoder.pkl")
    else:
        # Already encoded or needs fresh encoding from cleaned data
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
        joblib.dump(le, MODELS_DIR / "label_encoder.pkl")

    print(f"   Classes: {list(le.classes_)}")

    feature_names = list(X.columns)
    # Save feature names for inference
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")

    # Train/test split (80/20), stratified
    print("✂️  Splitting data 80% train / 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {X_train.shape} | Test: {X_test.shape}")

    # Scale
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, le


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features, le = prepare_data(sample_frac=0.1)
    print(f"\n✅ Feature engineering complete!")
    print(f"   Features: {len(features)}")
    print(f"   Train size: {len(X_train):,}")
    print(f"   Test size: {len(X_test):,}")