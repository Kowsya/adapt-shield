"""
STEP 4: MODEL TRAINING WITH MLFLOW TRACKING
=============================================
Trains Random Forest and XGBoost models.
Logs everything to MLflow so you can compare runs.

Run: python src/train.py

Then open: http://localhost:5000 (MLflow UI) to see results.
"""

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)
from xgboost import XGBClassifier

# Import our feature engineering
import sys
sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import prepare_data

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# ── MLflow Configuration ────────────────────────────────────────
MLFLOW_EXPERIMENT = "adapt-shield"
mlflow.set_tracking_uri("http://localhost:5000")   # MLflow UI URL
mlflow.set_experiment(MLFLOW_EXPERIMENT)


def compute_metrics(y_true, y_pred, label_names) -> dict:
    """Compute all classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    print("\n📊 CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))

    return metrics


def save_confusion_matrix(y_true, y_pred, label_names, model_name: str) -> str:
    """Plot and save confusion matrix as image (for MLflow logging)."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im)
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix: {model_name}")

    # Add numbers inside cells
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    path = f"models/{model_name}_confusion_matrix.png"
    plt.savefig(path, dpi=100)
    plt.close()
    return path


def train_random_forest(X_train, y_train, X_test, y_test, label_names):
    """Train Random Forest and log to MLflow."""
    print("\n" + "=" * 50)
    print("  Training: Random Forest")
    print("=" * 50)

    params = {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 5,
        "n_jobs": -1,          # Use all CPU cores
        "random_state": 42,
        "class_weight": "balanced",  # Handle imbalance
    }

    with mlflow.start_run(run_name="RandomForest"):
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        # Train
        print("🌲 Training Random Forest (this takes a few minutes)...")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        metrics = compute_metrics(y_test, y_pred, label_names)
        mlflow.log_metrics(metrics)

        # Confusion matrix
        cm_path = save_confusion_matrix(y_test, y_pred, label_names, "RandomForest")
        mlflow.log_artifact(cm_path)

        # Log model to MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")

        # Save locally too
        joblib.dump(model, MODELS_DIR / "random_forest.pkl")
        print(f"\n✅ Random Forest saved → models/random_forest.pkl")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Recall (macro): {metrics['recall_macro']:.4f}")
        print(f"   F1 (macro): {metrics['f1_macro']:.4f}")

    return model, metrics


def train_xgboost(X_train, y_train, X_test, y_test, label_names):
    """Train XGBoost and log to MLflow."""
    print("\n" + "=" * 50)
    print("  Training: XGBoost")
    print("=" * 50)

    n_classes = len(label_names)
    params = {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "n_jobs": -1,
        "random_state": 42,
        "objective": "multi:softmax",
        "num_class": n_classes,
        "eval_metric": "mlogloss",
    }

    with mlflow.start_run(run_name="XGBoost"):
        mlflow.log_params(params)
        mlflow.log_param("model_type", "XGBoost")

        print("⚡ Training XGBoost...")
        model = XGBClassifier(**params, verbosity=0)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        y_pred = model.predict(X_test)

        metrics = compute_metrics(y_test, y_pred, label_names)
        mlflow.log_metrics(metrics)

        cm_path = save_confusion_matrix(y_test, y_pred, label_names, "XGBoost")
        mlflow.log_artifact(cm_path)

        mlflow.sklearn.log_model(model, "xgboost_model")
        joblib.dump(model, MODELS_DIR / "xgboost.pkl")
        print(f"\n✅ XGBoost saved → models/xgboost.pkl")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Recall (macro): {metrics['recall_macro']:.4f}")
        print(f"   F1 (macro): {metrics['f1_macro']:.4f}")

    return model, metrics


def select_and_save_best(rf_metrics, xgb_metrics):
    """Compare models and copy best one to model.pkl (used by API)."""
    print("\n" + "=" * 50)
    print("  MODEL COMPARISON")
    print("=" * 50)
    print(f"  Random Forest F1: {rf_metrics['f1_macro']:.4f}")
    print(f"  XGBoost       F1: {xgb_metrics['f1_macro']:.4f}")

    if xgb_metrics["f1_macro"] >= rf_metrics["f1_macro"]:
        best = "xgboost"
        print("\n🏆 Winner: XGBoost")
    else:
        best = "random_forest"
        print("\n🏆 Winner: Random Forest")

    # Copy best model to model.pkl (this is what the API will load)
    import shutil
    shutil.copy(MODELS_DIR / f"{best}.pkl", MODELS_DIR / "model.pkl")
    print(f"✅ Best model saved as → models/model.pkl (used by API)")


def main():
    print("=" * 50)
    print("  ADAPT-SHIELD: Model Training")
    print("=" * 50)
    print("\n⚠️  Make sure MLflow UI is running: mlflow ui")
    print("   Then open: http://localhost:5000\n")

    # Prepare data (use 30% sample for faster training)
    # Change sample_frac=1.0 for full dataset
    X_train, X_test, y_train, y_test, feature_names, le = prepare_data(
        sample_frac=0.3
    )
    label_names = list(le.classes_)
    print(f"\n📋 Label classes: {label_names}")

    # Train both models
    rf_model, rf_metrics = train_random_forest(
        X_train, y_train, X_test, y_test, label_names
    )
    xgb_model, xgb_metrics = train_xgboost(
        X_train, y_train, X_test, y_test, label_names
    )

    # Select best
    select_and_save_best(rf_metrics, xgb_metrics)

    print("\n🎉 Training complete!")
    print("   → Open http://localhost:5000 to view MLflow results")
    print("   → Next step: python app/main.py  (start the API)")


if __name__ == "__main__":
    main()
    