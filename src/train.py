"""XGBoost মডেল প্রশিক্ষণ স্ক্রিপ্ট।

ডেটাসেট লোড করে, ট্রেন/টেস্ট বিভক্ত করে, XGBoost ক্লাসিফায়ার প্রশিক্ষণ দেয়,
মূল্যায়ন করে এবং মডেল ও মেটাডেটা সেভ করে।

ব্যবহার:
    python -m src.train
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from . import config
from .data import load_dataset


def train() -> XGBClassifier:
    """মডেল প্রশিক্ষণ দিয়ে সেভ করে এবং প্রশিক্ষিত মডেল রিটার্ন করে।"""
    print("📊 ডেটাসেট লোড হচ্ছে...")
    df = load_dataset()
    X = df[config.FEATURE_NAMES]
    y = df[config.TARGET_NAME]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y,
    )
    print(f"   ট্রেন: {X_train.shape[0]} | টেস্ট: {X_test.shape[0]}")

    print("🤖 XGBoost মডেল প্রশিক্ষণ হচ্ছে...")
    model = XGBClassifier(**config.XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ----------------------------- মূল্যায়ন -----------------------------
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("\n✅ মূল্যায়ন ফলাফল")
    print(f"   Accuracy : {accuracy:.4f}")
    print(f"   ROC-AUC  : {auc:.4f}")
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["সুস্থ", "ঝুঁকিপূর্ণ"]))
    print("   Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ফিচার গুরুত্ব
    importance = dict(
        sorted(
            zip(config.FEATURE_NAMES, model.feature_importances_.astype(float)),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )
    print("\n📈 ফিচার গুরুত্ব (গুরুত্বপূর্ণ থেকে কম):")
    for name, score in importance.items():
        print(f"   {name:18s}: {score:.4f}")

    # ------------------------- মডেল ও মেটাডেটা সেভ -------------------------
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(config.MODEL_PATH)

    metadata = {
        "features": config.FEATURE_NAMES,
        "metrics": {"accuracy": round(float(accuracy), 4), "roc_auc": round(float(auc), 4)},
        "feature_importance": {k: round(v, 4) for k, v in importance.items()},
        "params": config.XGB_PARAMS,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }
    with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n💾 মডেল সেভ হয়েছে: {config.MODEL_PATH}")
    print(f"💾 মেটাডেটা সেভ হয়েছে: {config.METADATA_PATH}")
    return model


if __name__ == "__main__":
    train()
