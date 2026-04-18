"""
STEP 4b: MODEL EVALUATION + HYPERPARAMETER TUNING
===================================================
Loads the best model, runs detailed evaluation,
and performs hyperparameter tuning with GridSearchCV.

Run: python src/evaluate.py

Shows: confusion matrix, per-class report, and tuning results.
"""

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("adapt-shield")


def load_artifacts():
    """Load model and test data."""
    print("📂 Loading model artifacts...")
    model       = joblib.load(MODELS_DIR / "model.pkl")
    scaler      = joblib.load(MODELS_DIR / "scaler.pkl")
    le          = joblib.load(MODELS_DIR / "label_encoder.pkl")
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
    return model, scaler, le, feature_names


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   label_names: list) -> dict:
    """Run full evaluation and save confusion matrix."""
    print("\n📊 Evaluating model on test set...")
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":         accuracy_score(y_test, y_pred),
        "recall_macro":     recall_score(y_test, y_pred, average="macro",    zero_division=0),
        "precision_macro":  precision_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro":         f1_score(y_test, y_pred, average="macro",        zero_division=0),
        "f1_weighted":      f1_score(y_test, y_pred, average="weighted",     zero_division=0),
    }

    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
    for k, v in metrics.items():
        bar = "█" * int(v * 20)
        print(f"  {k:<20} {v:.4f}  {bar}")
    print("=" * 55)

    print("\n📋 Per-Class Report:")
    print(classification_report(y_test, y_pred,
                                target_names=label_names, zero_division=0))

    # ── Plot Confusion Matrix ──────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=label_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("ADAPT-SHIELD: Confusion Matrix (Best Model)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = REPORTS_DIR / "best_model_confusion_matrix.png"
    plt.savefig(cm_path, dpi=120)
    plt.close()
    print(f"✅ Confusion matrix saved → {cm_path}")

    return metrics, y_pred


def hyperparameter_tuning(X_train: np.ndarray, y_train: np.ndarray,
                          label_names: list):
    """
    Grid search over Random Forest hyperparameters.
    Logs best params + score to MLflow.
    Uses small sample for speed — increase for full tuning.
    """
    print("\n" + "=" * 55)
    print("  HYPERPARAMETER TUNING (Random Forest)")
    print("=" * 55)
    print("⚠️  Using 10% sample for speed. Set sample_frac=1.0 for full tuning.\n")

    # Sample to keep tuning fast
    n = min(20_000, len(X_train))
    idx = np.random.choice(len(X_train), n, replace=False)
    X_s, y_s = X_train[idx], y_train[idx]

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth":    [10, 20, None],
        "min_samples_split": [2, 5, 10],
    }

    rf = RandomForestClassifier(
        n_jobs=-1,
        random_state=42,
        class_weight="balanced"
    )

    print("🔍 Running GridSearchCV (3-fold CV)...")
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid_search.fit(X_s, y_s)

    best_params  = grid_search.best_params_
    best_score   = grid_search.best_score_
    best_model   = grid_search.best_estimator_

    print(f"\n🏆 Best Parameters: {best_params}")
    print(f"   Best CV F1 (macro): {best_score:.4f}")

    # Log tuning results to MLflow
    with mlflow.start_run(run_name="HyperparamTuning_RF"):
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_f1_macro_best", best_score)
        mlflow.sklearn.log_model(best_model, "tuned_rf_model")
        print("   Logged to MLflow ✅")

    # Save tuned model
    tuned_path = MODELS_DIR / "tuned_random_forest.pkl"
    joblib.dump(best_model, tuned_path)
    print(f"   Saved tuned model → {tuned_path}")

    return best_model, best_params, best_score


def cross_validate_best(model, X_train: np.ndarray, y_train: np.ndarray):
    """Run 5-fold cross-validation on the best model."""
    print("\n🔁 Running 5-Fold Cross-Validation...")
    n = min(30_000, len(X_train))
    idx = np.random.choice(len(X_train), n, replace=False)
    scores = cross_val_score(
        model, X_train[idx], y_train[idx],
        cv=5, scoring="f1_macro", n_jobs=-1
    )
    print(f"   CV F1 Scores:  {[round(s, 4) for s in scores]}")
    print(f"   Mean ± Std:    {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


def main():
    print("=" * 55)
    print("  ADAPT-SHIELD: Model Evaluation & Hyperparameter Tuning")
    print("=" * 55)

    # Load artifacts
    model, scaler, le, feature_names = load_artifacts()
    label_names = list(le.classes_)

    # Load test data (saved during feature engineering)
    # We recreate test data from a small sample for evaluation
    import sys
    sys.path.insert(0, "src")
    from feature_engineering import prepare_data

    print("\n📂 Preparing data for evaluation (10% sample)...")
    X_train, X_test, y_train, y_test, _, _ = prepare_data(sample_frac=0.1)

    # ── Evaluate current best model ───────────────────────────
    metrics, y_pred = evaluate_model(model, X_test, y_test, label_names)

    # ── Hyperparameter Tuning ─────────────────────────────────
    tuned_model, best_params, best_score = hyperparameter_tuning(
        X_train, y_train, label_names
    )

    # ── Cross-validation ──────────────────────────────────────
    cross_validate_best(tuned_model, X_train, y_train)

    # ── Compare: current vs tuned ─────────────────────────────
    print("\n" + "=" * 55)
    print("  COMPARISON: Current Best vs Tuned Model")
    print("=" * 55)
    print(f"  Current model F1: {metrics['f1_macro']:.4f}")
    print(f"  Tuned model  F1: {best_score:.4f}  (CV estimate)")

    if best_score > metrics["f1_macro"]:
        print("\n✅ Tuned model is better! Updating model.pkl...")
        joblib.dump(tuned_model, MODELS_DIR / "model.pkl")
        print("   model.pkl updated → API will use tuned model")
    else:
        print("\n✅ Current model is already optimal.")

    print("\n🎉 Evaluation complete!")
    print(f"   Reports saved in: {REPORTS_DIR}/")
    print("   Check MLflow at: http://localhost:5000")


if __name__ == "__main__":
    main()