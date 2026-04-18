"""
STEP 8: DRIFT DETECTION & MONITORING
=======================================
Detects if incoming data has shifted from training data.
Uses Evidently AI to generate an HTML drift report.
Also monitors API health metrics.

Run: python monitoring/drift_detection.py
Output: reports/drift_report.html  (open in browser)
"""

import json
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from scipy import stats as scipy_stats

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
LOG_PATH = Path("monitoring/drift_log.json")
REPORTS_DIR.mkdir(exist_ok=True)
LOG_PATH.parent.mkdir(exist_ok=True)

DRIFT_THRESHOLD = 0.05   # p-value threshold for KS test


# ═══════════════════════════════════════════════════════════════
# PART 1: STATISTICAL DRIFT DETECTION (KS Test)
# ═══════════════════════════════════════════════════════════════

class DriftDetector:
    """Detects statistical drift between training and new data."""

    def __init__(self):
        self.reference_stats = None

    def compute_stats(self, df: pd.DataFrame) -> dict:
        numeric = df.select_dtypes(include=[np.number])
        return {
            col: {
                "mean": float(numeric[col].mean()),
                "std":  float(numeric[col].std()),
                "p25":  float(numeric[col].quantile(0.25)),
                "p75":  float(numeric[col].quantile(0.75)),
            }
            for col in numeric.columns
        }

    def fit_reference(self, df: pd.DataFrame):
        print("📊 Computing reference statistics from training data...")
        self.reference_stats = self.compute_stats(df)
        print(f"   Reference computed for {len(self.reference_stats)} features")

    def detect_drift(self, new_df: pd.DataFrame) -> dict:
        if self.reference_stats is None:
            raise ValueError("Call fit_reference() first!")

        drifted_cols = []
        drift_results = {}
        numeric = new_df.select_dtypes(include=[np.number])

        for col in numeric.columns:
            if col not in self.reference_stats:
                continue
            ref = self.reference_stats[col]
            ref_sample = np.random.normal(ref["mean"], max(ref["std"], 1e-9), size=1000)
            new_sample = numeric[col].dropna().values
            if len(new_sample) < 10:
                continue

            ks_stat, p_value = scipy_stats.ks_2samp(ref_sample, new_sample)
            is_drifted = p_value < DRIFT_THRESHOLD

            drift_results[col] = {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value":      round(float(p_value), 4),
                "drift_detected": is_drifted,
                "ref_mean":     round(ref["mean"], 4),
                "new_mean":     round(float(new_sample.mean()), 4),
            }
            if is_drifted:
                drifted_cols.append(col)

        print(f"\n🔍 DRIFT DETECTION RESULTS (KS Test):")
        print(f"   Features checked : {len(drift_results)}")
        print(f"   Drifted features : {len(drifted_cols)}")

        if drifted_cols:
            print(f"\n⚠️  DRIFT DETECTED in: {drifted_cols[:5]}" +
                  (" ..." if len(drifted_cols) > 5 else ""))
            self._log_drift(drifted_cols)
            self._alert()
        else:
            print("✅ No significant drift detected — model is stable")

        return drift_results, drifted_cols

    def _log_drift(self, drifted_cols: list):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "drifted_columns": drifted_cols,
            "num_drifted": len(drifted_cols),
        }
        logs = []
        if LOG_PATH.exists():
            with open(LOG_PATH) as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []
        logs.append(entry)
        with open(LOG_PATH, "w") as f:
            json.dump(logs, f, indent=2)
        print(f"   Drift event logged → {LOG_PATH}")

    def _alert(self):
        print("\n🚨 RETRAINING ALERT!")
        print("   Data drift detected — model should be retrained.")
        print("   Run: python src/train.py  (or push to GitHub to trigger CI/CD)")


# ═══════════════════════════════════════════════════════════════
# PART 2: EVIDENTLY AI REPORT (HTML)
# ═══════════════════════════════════════════════════════════════

def generate_evidently_report(reference_df: pd.DataFrame,
                               current_df: pd.DataFrame):
    """Generate Evidently HTML drift report."""
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset

        print("\n📊 Generating Evidently AI drift report...")

        # Use only numeric columns (first 20 for speed)
        num_cols = reference_df.select_dtypes(include=[np.number]).columns[:20].tolist()
        ref = reference_df[num_cols].copy()
        cur = current_df[num_cols].copy()

        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])
        report.run(reference_data=ref, current_data=cur)

        html_path = REPORTS_DIR / "drift_report.html"
        report.save_html(str(html_path))
        print(f"✅ Evidently report saved → {html_path}")
        print(f"   Open in browser: reports/drift_report.html")
        return str(html_path)

    except ImportError:
        print("⚠️  Evidently not installed. Run: pip install evidently>=0.4.32")
        return None
    except Exception as e:
        print(f"⚠️  Evidently report failed: {e}")
        print("   Continuing with KS-test results only.")
        return None


# ═══════════════════════════════════════════════════════════════
# PART 3: API HEALTH MONITOR
# ═══════════════════════════════════════════════════════════════

class APIMonitor:
    """Monitor API health by polling the /health endpoint."""

    def check(self, url: str = "http://localhost:8000"):
        try:
            import requests
            resp = requests.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                print("\n📈 API HEALTH STATUS")
                print("=" * 40)
                print(f"  Status:        {data.get('status', 'unknown')}")
                print(f"  Model loaded:  {data.get('model_loaded', False)}")
                print(f"  Uptime:        {data.get('uptime_seconds', 0):.1f}s")
                print(f"  Requests:      {data.get('total_requests', 0)}")
                print(f"  Blocked:       {data.get('blocked_requests', 0)}")
                print(f"  Block rate:    {data.get('block_rate_percent', 0):.1f}%")
                print("=" * 40)
            else:
                print(f"⚠️  API returned status {resp.status_code}")
        except Exception as e:
            print(f"⚠️  Cannot reach API at {url}: {e}")
            print("   Make sure the API is running: uvicorn app.main:app --port 8000")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  ADAPT-SHIELD: Drift Detection & Monitoring")
    print("=" * 55)

    # ── Check API health ──────────────────────────────────────
    api_monitor = APIMonitor()
    api_monitor.check()

    # ── Load feature names ────────────────────────────────────
    try:
        feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
        print(f"\n📂 Loaded {len(feature_names)} feature names from model artifacts")
    except FileNotFoundError:
        print("⚠️  feature_names.pkl not found. Run training first.")
        print("   Using synthetic features for demo.")
        feature_names = [f"feature_{i}" for i in range(20)]

    n_features = len(feature_names)

    # ── Create reference data (simulate training distribution) ─
    print("\n📂 Creating reference distribution (training stats)...")
    np.random.seed(42)
    reference_df = pd.DataFrame(
        np.random.normal(0, 1, size=(5000, n_features)),
        columns=feature_names
    )

    # ── Create current data (simulate incoming traffic) ────────
    print("📂 Simulating new incoming traffic data...")
    current_data = np.random.normal(0, 1, size=(1000, n_features))
    # Introduce drift in first 5 features (realistic scenario)
    current_data[:, :5] += 3.0
    current_df = pd.DataFrame(current_data, columns=feature_names)

    # ── KS Test drift detection ───────────────────────────────
    detector = DriftDetector()
    detector.fit_reference(reference_df)
    results, drifted = detector.detect_drift(current_df)

    # ── Evidently HTML report ─────────────────────────────────
    html_path = generate_evidently_report(reference_df, current_df)

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  MONITORING COMPLETE")
    print("=" * 55)
    print(f"  Drifted features : {len(drifted)}/{n_features}")
    if html_path:
        print(f"  HTML Report      : {html_path}")
    print(f"  Drift log        : {LOG_PATH}")
    print("\n💡 In production: run this script on a schedule")
    print("   Example cron: 0 * * * * python monitoring/drift_detection.py")


if __name__ == "__main__":
    main()