"""
STEP 8: DRIFT DETECTION & MONITORING
=======================================
Detects if incoming data has shifted from training data.
If drift is detected → retraining pipeline is triggered.

Run: python monitoring/drift_detection.py
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats as scipy_stats

MODELS_DIR = Path("models")
LOG_PATH = Path("monitoring/drift_log.json")
LOG_PATH.parent.mkdir(exist_ok=True)

# Threshold: p-value below this = drift detected
DRIFT_THRESHOLD = 0.05


class DriftDetector:
    """Detects statistical drift between training and new data."""

    def __init__(self):
        self.reference_stats = None
        self.alerts = []

    def compute_stats(self, df: pd.DataFrame) -> dict:
        """Compute mean and std for each numeric column."""
        numeric = df.select_dtypes(include=[np.number])
        return {
            col: {"mean": float(numeric[col].mean()), "std": float(numeric[col].std())}
            for col in numeric.columns
        }

    def fit_reference(self, df: pd.DataFrame):
        """Store stats of the training data as reference."""
        print("📊 Computing reference statistics from training data...")
        self.reference_stats = self.compute_stats(df)
        print(f"   Reference computed for {len(self.reference_stats)} features")

    def detect_drift(self, new_df: pd.DataFrame) -> dict:
        """
        Compare new data against reference using KS test.
        Returns dict of drifted columns.
        """
        if self.reference_stats is None:
            raise ValueError("Call fit_reference() first!")

        drift_results = {}
        drifted_cols = []

        numeric = new_df.select_dtypes(include=[np.number])

        for col in numeric.columns:
            if col not in self.reference_stats:
                continue

            ref = self.reference_stats[col]

            # Simulate reference distribution (normally we'd store actual training data)
            # In production: store a sample of training data
            ref_sample = np.random.normal(
                ref["mean"], ref["std"] + 1e-9, size=1000
            )
            new_sample = numeric[col].dropna().values

            if len(new_sample) < 10:
                continue

            # Kolmogorov-Smirnov test
            ks_stat, p_value = scipy_stats.ks_2samp(ref_sample, new_sample)

            is_drifted = p_value < DRIFT_THRESHOLD
            drift_results[col] = {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_value), 4),
                "drift_detected": is_drifted,
            }

            if is_drifted:
                drifted_cols.append(col)

        print(f"\n🔍 DRIFT DETECTION RESULTS:")
        print(f"   Columns checked: {len(drift_results)}")
        print(f"   Drifted columns: {len(drifted_cols)}")

        if drifted_cols:
            print(f"\n⚠️  DRIFT DETECTED in: {drifted_cols[:5]}...")
            self._log_drift(drifted_cols)
            self._trigger_retraining_alert(drifted_cols)
        else:
            print("✅ No significant drift detected")

        return drift_results

    def _log_drift(self, drifted_cols: list):
        """Log drift event to file."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "drifted_columns": drifted_cols,
            "num_drifted": len(drifted_cols),
        }
        logs = []
        if LOG_PATH.exists():
            with open(LOG_PATH) as f:
                logs = json.load(f)
        logs.append(entry)
        with open(LOG_PATH, "w") as f:
            json.dump(logs, f, indent=2)
        print(f"   Drift event logged → {LOG_PATH}")

    def _trigger_retraining_alert(self, drifted_cols: list):
        """Alert that retraining is needed."""
        print("\n🚨 RETRAINING ALERT!")
        print("   Data drift detected — model should be retrained")
        print("   In production: this would trigger your CI/CD pipeline")
        print("   Command: python src/train.py")


class APIMonitor:
    """Monitor API health metrics."""

    def __init__(self):
        self.metrics = {
            "requests": 0,
            "blocked": 0,
            "allowed": 0,
            "errors": 0,
            "latencies": [],
        }

    def record_request(self, is_blocked: bool, latency_ms: float, error: bool = False):
        self.metrics["requests"] += 1
        if error:
            self.metrics["errors"] += 1
        elif is_blocked:
            self.metrics["blocked"] += 1
        else:
            self.metrics["allowed"] += 1
        self.metrics["latencies"].append(latency_ms)

    def report(self):
        total = self.metrics["requests"]
        if total == 0:
            print("No requests recorded yet.")
            return

        latencies = self.metrics["latencies"]
        print("\n📈 API MONITORING REPORT")
        print("=" * 40)
        print(f"  Total Requests:   {total:,}")
        print(f"  Blocked:          {self.metrics['blocked']:,} ({self.metrics['blocked']/total*100:.1f}%)")
        print(f"  Allowed:          {self.metrics['allowed']:,} ({self.metrics['allowed']/total*100:.1f}%)")
        print(f"  Errors:           {self.metrics['errors']:,}")
        if latencies:
            print(f"  Avg Latency:      {np.mean(latencies):.1f} ms")
            print(f"  95th Percentile:  {np.percentile(latencies, 95):.1f} ms")
        print("=" * 40)


def main():
    print("=" * 50)
    print("  ADAPT-SHIELD: Drift Detection Demo")
    print("=" * 50)

    detector = DriftDetector()
    monitor = APIMonitor()

    # Simulate reference data (in production, use actual training data sample)
    print("\n📂 Simulating reference data (training distribution)...")
    np.random.seed(42)
    n_features = 10
    feature_names = [f"Feature_{i}" for i in range(n_features)]
    reference_df = pd.DataFrame(
        np.random.normal(0, 1, size=(1000, n_features)),
        columns=feature_names
    )
    detector.fit_reference(reference_df)

    # Simulate new data — some columns have shifted (drift)
    print("\n📂 Simulating new incoming data (with some drift)...")
    new_data = np.random.normal(0, 1, size=(500, n_features))
    new_data[:, :3] += 5   # Introduce drift in first 3 features
    new_df = pd.DataFrame(new_data, columns=feature_names)

    # Detect drift
    results = detector.detect_drift(new_df)

    # Simulate some API monitoring
    print("\n📊 Simulating API monitoring...")
    for i in range(100):
        is_blocked = np.random.random() < 0.15
        latency = np.random.normal(25, 5)
        monitor.record_request(is_blocked, latency)

    monitor.report()

    print("\n✅ Monitoring demo complete!")
    print("   In production: run this script on a schedule (cron job)")
    print("   or integrate with Prometheus + Grafana for live dashboards")


if __name__ == "__main__":
    main()