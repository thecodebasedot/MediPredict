"""XGBoost মাল্টি-ডিজিজ মডেল প্রশিক্ষণ স্ক্রিপ্ট।

প্রতিটি রোগের (ডায়াবেটিস, হৃদরোগ, উচ্চ রক্তচাপ) জন্য আলাদা XGBoost মডেল
প্রশিক্ষণ দেয়, মূল্যায়ন করে এবং মডেল ও একত্রিত মেটাডেটা সেভ করে।

ব্যবহার:
    python -m src.train
"""

from __future__ import annotations

import json

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from . import config
from .data import load_dataset


def train_one(df, disease: str) -> dict:
    """একটি রোগের জন্য মডেল প্রশিক্ষণ ও মূল্যায়ন; মেটাডেটা রিটার্ন করে।"""
    meta = config.DISEASES[disease]
    X = df[config.FEATURE_NAMES]
    y = df[disease]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y,
    )

    model = XGBClassifier(**config.XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    importance = dict(
        sorted(
            zip(config.FEATURE_NAMES, model.feature_importances_.astype(float)),
            key=lambda kv: kv[1], reverse=True,
        )
    )

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(config.model_path(disease))

    print(f"\n🩺 {meta['name_bn']} ({meta['name_en']})")
    print(f"   Accuracy: {accuracy:.4f} | ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["সুস্থ", "ঝুঁকিপূর্ণ"], digits=3))

    return {
        "accuracy": round(float(accuracy), 4),
        "roc_auc": round(float(auc), 4),
        "feature_importance": {k: round(v, 4) for k, v in importance.items()},
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }


def train() -> dict:
    """সব রোগের মডেল প্রশিক্ষণ দিয়ে মেটাডেটা সেভ করে।"""
    print("📊 ডেটাসেট লোড হচ্ছে...")
    df = load_dataset()
    print(f"   মোট নমুনা: {df.shape[0]}")

    metadata = {"features": config.FEATURE_NAMES, "diseases": {}}
    for disease in config.DISEASE_KEYS:
        result = train_one(df, disease)
        result["name_bn"] = config.DISEASES[disease]["name_bn"]
        result["name_en"] = config.DISEASES[disease]["name_en"]
        metadata["diseases"][disease] = result

    with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n💾 {len(config.DISEASE_KEYS)}টি মডেল ও মেটাডেটা সেভ হয়েছে: {config.MODEL_DIR}")
    return metadata


if __name__ == "__main__":
    train()
