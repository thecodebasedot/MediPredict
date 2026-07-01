"""বাস্তব ডেটাসেট ইন্টিগ্রেশন — UCI/Pima Indians Diabetes।

সিনথেটিক ডেটার পাশাপাশি একটি প্রকৃত মেডিকেল ডেটাসেটে XGBoost প্রশিক্ষণ দিয়ে
দেখায় যে পাইপলাইনটি বাস্তব ডেটায়ও কাজ করে। Pima ডেটাসেটের নিজস্ব ৮টি ফিচার
ব্যবহৃত হয় (সিনথেটিক স্কিমা থেকে আলাদা)।

উৎস: UCI Machine Learning Repository (Pima Indians Diabetes Database)।

ব্যবহার:
    python -m src.real_data          # ডাউনলোড + প্রশিক্ষণ + মূল্যায়ন
    python -m src.real_data --path local.csv   # স্থানীয় CSV থেকে
"""

from __future__ import annotations

import argparse
import json
import urllib.request

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from . import config

PIMA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)
PIMA_COLUMNS = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age", "outcome",
]

RAW_PATH = config.DATA_DIR / "pima_diabetes.csv"
REAL_MODEL_PATH = config.MODEL_DIR / "xgboost_real_diabetes.json"
REAL_METRICS_PATH = config.MODEL_DIR / "real_metrics.json"


def download(path=RAW_PATH) -> None:
    """Pima ডেটাসেট ডাউনলোড করে সেভ করে (আগে থেকে না থাকলে)।"""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"⬇️  Pima ডেটাসেট ডাউনলোড হচ্ছে: {PIMA_URL}")
    urllib.request.urlretrieve(PIMA_URL, path)


def load(path=RAW_PATH):
    """ডেটাসেট DataFrame হিসেবে লোড করে।"""
    import pandas as pd

    if not path.exists():
        download(path)
    return pd.read_csv(path, header=None, names=PIMA_COLUMNS)


def train(path=RAW_PATH) -> dict:
    """প্রকৃত Pima ডেটায় XGBoost প্রশিক্ষণ দিয়ে মূল্যায়ন ও সেভ করে।"""
    df = load(path)
    features = [c for c in PIMA_COLUMNS if c != "outcome"]
    X, y = df[features], df["outcome"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y,
    )
    model = XGBClassifier(**config.XGB_PARAMS)
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = float(accuracy_score(y_te, pred))
    auc = float(roc_auc_score(y_te, proba))

    print("🌍 প্রকৃত Pima Indians Diabetes ডেটাসেট")
    print(f"   নমুনা: {len(df)} | ফিচার: {len(features)}")
    print(f"   Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")
    print(classification_report(y_te, pred, target_names=["সুস্থ", "ডায়াবেটিস"], digits=3))

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(REAL_MODEL_PATH)
    metrics = {
        "dataset": "Pima Indians Diabetes (UCI)",
        "n_samples": int(len(df)),
        "features": features,
        "accuracy": round(acc, 4),
        "roc_auc": round(auc, 4),
    }
    with open(REAL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"💾 মডেল ও মেট্রিক সেভ হয়েছে: {config.MODEL_DIR}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pima বাস্তব ডেটায় প্রশিক্ষণ")
    parser.add_argument("--path", default=str(RAW_PATH), help="স্থানীয় CSV পাথ")
    args = parser.parse_args()
    from pathlib import Path

    train(Path(args.path))
