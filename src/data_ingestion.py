"""
STEP 1: DATA INGESTION
======================
This script loads all 8 parquet files from the attackdata folder,
combines them into one big dataframe, and saves it.

Run this first: python src/data_ingestion.py
"""

import os
import pandas as pd
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────
DATA_DIR = Path("data/attackdata")          # Folder with your parquet files
OUTPUT_PATH = Path("data/combined_data.parquet")  # Where to save combined data

# All 8 files
PARQUET_FILES = [
    "Benign-Monday-no-metadata.parquet",
    "Botnet-Friday-no-metadata.parquet",
    "Bruteforce-Tuesday-no-metadata.parquet",
    "DDoS-Friday-no-metadata.parquet",
    "DoS-Wednesday-no-metadata.parquet",
    "Infiltration-Thursday-no-metadata.parquet",
    "Portscan-Friday-no-metadata.parquet",
    "WebAttacks-Thursday-no-metadata.parquet",
]

# ── Label Mapping ───────────────────────────────────────────────
# Map raw labels → clean category names
LABEL_MAP = {
    "Benign": "BENIGN",
    "Bot": "Botnet",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "DDoS": "DDoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "Heartbleed": "DoS",
    "Infiltration": "Infiltration",
    "PortScan": "PortScan",
    "Web Attack \xef\xbf\xbd Brute Force": "WebAttack",
    "Web Attack \xef\xbf\xbd XSS": "WebAttack",
    "Web Attack \xef\xbf\xbd Sql Injection": "WebAttack",
    # Handle encoding variants
    "Web Attack  Brute Force": "WebAttack",
    "Web Attack  XSS": "WebAttack",
    "Web Attack  Sql Injection": "WebAttack",
}


def load_and_combine() -> pd.DataFrame:
    """Load all parquet files and combine into one dataframe."""
    dfs = []

    for filename in PARQUET_FILES:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"⚠️  WARNING: {filename} not found — skipping")
            continue

        print(f"📂 Loading {filename}...")
        df = pd.read_parquet(filepath)
        print(f"   Shape: {df.shape} | Labels: {df['Label'].unique().tolist()}")
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No parquet files found in {DATA_DIR}. "
            "Please copy your .parquet files there."
        )

    print(f"\n🔗 Combining {len(dfs)} files...")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"✅ Combined shape: {combined.shape}")
    return combined


def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize label names to clean categories."""
    # Handle encoding issues in Web Attack labels
    df["Label"] = df["Label"].str.replace(
        r"Web Attack.*?Brute Force", "WebAttack", regex=True
    )
    df["Label"] = df["Label"].str.replace(
        r"Web Attack.*?XSS", "WebAttack", regex=True
    )
    df["Label"] = df["Label"].str.replace(
        r"Web Attack.*?Sql Injection", "WebAttack", regex=True
    )

    # Apply label map
    df["Label"] = df["Label"].map(LABEL_MAP).fillna(df["Label"])
    return df


def show_class_distribution(df: pd.DataFrame):
    """Print how many samples each attack class has."""
    print("\n📊 CLASS DISTRIBUTION:")
    print("=" * 45)
    dist = df["Label"].value_counts()
    total = len(df)
    for label, count in dist.items():
        pct = (count / total) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:<15} {count:>8,}  ({pct:5.1f}%)  {bar}")
    print(f"\n  TOTAL: {total:,} records")
    print("=" * 45)


def main():
    print("=" * 50)
    print("  ADAPT-SHIELD: Data Ingestion")
    print("=" * 50)

    # Create output folder if needed
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load all files
    df = load_and_combine()

    # Clean labels
    print("\n🏷️  Cleaning labels...")
    df = clean_labels(df)

    # Show distribution
    show_class_distribution(df)

    # Save combined file
    print(f"\n💾 Saving combined data to {OUTPUT_PATH}...")
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"✅ Saved! File size: {OUTPUT_PATH.stat().st_size / 1e6:.1f} MB")

    return df


if __name__ == "__main__":
    main()