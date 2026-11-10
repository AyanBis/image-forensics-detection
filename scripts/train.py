#!/usr/bin/env python3
"""Train and evaluate a reproducible Random Forest baseline."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from image_forensics.features import load_image, extract_features
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "random_forest.joblib")
    parser.add_argument("--results", type=Path, default=ROOT / "outputs" / "evaluation.json")
    parser.add_argument("--test-size", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    samples, labels = [], []
    for label, folder in (("authentic", "authentic"), ("tampered", "tampered")):
        paths = sorted(p for p in (args.data / folder).rglob("*") if p.suffix.lower() in EXTS)
        for path in paths:
            try:
                samples.append(extract_features(load_image(path)))
                labels.append(label)
            except Exception as exc:
                print(f"Skipping {path}: {exc}", file=sys.stderr)
    counts = {name: labels.count(name) for name in ("authentic", "tampered")}
    if min(counts.values()) < 2:
        raise SystemExit(f"Need at least 2 readable images per class for a stratified split; found {counts}")
    X, y = np.asarray(samples), np.asarray(labels)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RandomForestClassifier(
        n_estimators=400, class_weight="balanced", random_state=args.seed, n_jobs=-1, min_samples_leaf=1))
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    metrics = {"dataset_counts": counts, "train_size": int(len(ytr)), "test_size": int(len(yte)),
               "seed": args.seed, "test_fraction": args.test_size,
               "accuracy": float(accuracy_score(yte, pred)),
               "precision_macro": float(precision_score(yte, pred, average="macro", zero_division=0)),
               "recall_macro": float(recall_score(yte, pred, average="macro", zero_division=0)),
               "f1_macro": float(f1_score(yte, pred, average="macro", zero_division=0)),
               "labels": ["authentic", "tampered"],
               "confusion_matrix": confusion_matrix(yte, pred, labels=["authentic", "tampered"]).tolist(),
               "classification_report": classification_report(yte, pred, labels=["authentic", "tampered"], output_dict=True, zero_division=0)}
    args.model.parent.mkdir(parents=True, exist_ok=True); args.results.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_version": 1, "labels": ["authentic", "tampered"]}, args.model)
    args.results.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2)); print(f"\nSaved model: {args.model}\nSaved results: {args.results}")
if __name__ == "__main__": main()
