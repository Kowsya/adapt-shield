"""
STEP 2: DATA VALIDATION
========================
Checks the combined dataset for data quality issues.
Run: python src/data_validation.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/combined_data.parquet")


def validate(df: pd.DataFrame) -> dict:
    """Run all validation checks. Returns a dict of results."""
    results = {}

    print("\n" + "=" * 50)
    print("  ADAPT-SHIELD: Data Validation Report")
    print("=" * 50)

    # 1. Shape check
    results["rows"] = len(df)
    results["cols"] = len(df.columns)
    print(f"\n✅ Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # 2. Missing values
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    results["null_columns"] = null_cols.to_dict()
    if null_cols.empty:
        print("✅ No missing values found!")
    else:
        print(f"⚠️  Missing values in {len(null_cols)} columns:")
        for col, cnt in null_cols.items():
            print(f"     {col}: {cnt:,} nulls ({cnt/len(df)*100:.2f}%)")

    # 3. Infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(df[numeric_cols]).sum()
    inf_cols = inf_counts[inf_counts > 0]
    results["inf_columns"] = inf_cols.to_dict()
    if inf_cols.empty:
        print("✅ No infinite values found!")
    else:
        print(f"⚠️  Infinite values in {len(inf_cols)} columns:")
        for col, cnt in inf_cols.items():
            print(f"     {col}: {cnt:,} inf values")

    # 4. Duplicate rows
    dup_count = df.duplicated().sum()
    results["duplicates"] = int(dup_count)
    if dup_count == 0:
        print("✅ No duplicate rows!")
    else:
        print(f"⚠️  Found {dup_count:,} duplicate rows (will be removed)")

    # 5. Label column check
    if "Label" not in df.columns:
        print("❌ CRITICAL: 'Label' column missing!")
        results["label_ok"] = False
    else:
        labels = df["Label"].unique()
        results["labels"] = labels.tolist()
        results["label_ok"] = True
        print(f"✅ Label column found with {len(labels)} classes: {labels.tolist()}")

    # 6. Class imbalance warning
    if results["label_ok"]:
        dist = df["Label"].value_counts(normalize=True)
        majority = dist.max()
        minority = dist.min()
        ratio = majority / minority
        results["imbalance_ratio"] = round(ratio, 2)
        if ratio > 10:
            print(f"⚠️  HIGH IMBALANCE: majority/minority ratio = {ratio:.1f}x")
            print("   → SMOTE oversampling will be applied in training")
        else:
            print(f"✅ Class imbalance ratio: {ratio:.1f}x (acceptable)")

    # 7. Negative values in columns that shouldn't have them
    suspicious_neg = []
    non_neg_cols = [c for c in numeric_cols if "IAT" not in c and "Win" not in c]
    for col in non_neg_cols[:20]:  # check first 20 only
        if (df[col] < 0).sum() > 0:
            suspicious_neg.append(col)
    results["suspicious_negative"] = suspicious_neg
    if suspicious_neg:
        print(f"⚠️  Suspicious negative values in: {suspicious_neg}")
    else:
        print("✅ No suspicious negative values in key columns")

    # Summary
    print("\n" + "=" * 50)
    issues = (
        len(null_cols) + len(inf_cols) + (1 if dup_count > 0 else 0)
    )
    if issues == 0:
        print("🎉 VALIDATION PASSED: Data looks clean!")
    else:
        print(f"⚠️  VALIDATION: {issues} issue(s) found (will be handled in preprocessing)")
    print("=" * 50)

    return results


def main():
    if not DATA_PATH.exists():
        print(f"❌ {DATA_PATH} not found.")
        print("Run data_ingestion.py first: python src/data_ingestion.py")
        return

    df = pd.read_parquet(DATA_PATH)
    results = validate(df)
    return results


if __name__ == "__main__":
    main()