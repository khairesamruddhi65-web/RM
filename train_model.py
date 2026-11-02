import argparse
import json
from pathlib import Path
import warnings

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

def main():
    parser = argparse.ArgumentParser(description="Train an experimental grant-outcome text classifier.")
    parser.add_argument("--data", default="data/sample_grants.csv")
    parser.add_argument("--text-col", default="abstract")
    parser.add_argument("--label-col", default="outcome")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    df = pd.read_csv(data_path)
    for col in [args.text_col, args.label_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found. Available: {list(df.columns)}")
    df = df[[args.text_col, args.label_col] + (["synthetic"] if "synthetic" in df.columns else [])].copy()
    df[args.text_col] = df[args.text_col].fillna("").astype(str).str.strip()
    df[args.label_col] = pd.to_numeric(df[args.label_col], errors="coerce")
    df = df[(df[args.text_col].str.len() > 20) & df[args.label_col].isin([0, 1])]
    df[args.label_col] = df[args.label_col].astype(int)
    df = df.drop_duplicates(subset=[args.text_col])
    if df[args.label_col].nunique() != 2 or df[args.label_col].value_counts().min() < 2:
        raise ValueError("Need at least two examples per class for a stratified train/test split.")
    X_train, X_test, y_train, y_test = train_test_split(
        df[args.text_col], df[args.label_col], test_size=args.test_size,
        random_state=args.seed, stratify=df[args.label_col]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000,
                                  sublinear_tf=True, stop_words="english")),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced",
                                          random_state=args.seed))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "n_total": int(len(df)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "class_counts": {str(k): int(v) for k, v in df[args.label_col].value_counts().items()},
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "average_precision_pr_auc": float(average_precision_score(y_test, prob)),
        "brier_score": float(brier_score_loss(y_test, prob)),
        "confusion_matrix_labels_0_1": confusion_matrix(y_test, pred, labels=[0,1]).tolist(),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "data_file": str(data_path),
        "synthetic_demo_only": bool(df["synthetic"].astype(str).str.lower().isin(["true","1","yes"]).all()) if "synthetic" in df.columns else False,
        "warning": "Random holdout is a smoke-test only. For publication, use temporal/grouped splits and independent calibration."
    }
    if len(set(y_test)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_test, prob))
    else:
        metrics["roc_auc"] = None
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "grant_outcome_model.joblib")
    (REPORT_DIR / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"text": X_test.values, "y_true": y_test.values, "y_pred": pred, "probability_class_1": prob}).to_csv(
        REPORT_DIR / "test_predictions.csv", index=False
    )
    print(json.dumps({k:v for k,v in metrics.items() if k != "classification_report"}, indent=2))
    if metrics["synthetic_demo_only"]:
        print("\nWARNING: trained on synthetic demonstration records. Do not cite these metrics as research evidence.")

if __name__ == "__main__":
    main()
