"""মডেল তুলনা — XGBoost বনাম Random Forest বনাম Logistic Regression।

প্রতিটি রোগের জন্য একই ডেটায় তিনটি মডেল প্রশিক্ষণ দিয়ে accuracy ও ROC-AUC
তুলনা করে, ফলাফল JSON-এ সেভ করে এবং একটি বার চার্ট তৈরি করে।

ব্যবহার:
    python -m src.compare
"""

from __future__ import annotations

import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from . import config
from .data import load_dataset


def _models():
    return {
        "XGBoost": XGBClassifier(**config.XGB_PARAMS),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=config.RANDOM_SEED
        ),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED),
        ),
    }


def compare() -> dict:
    """সব রোগ ও মডেলের তুলনা চালিয়ে ফলাফল রিটার্ন ও সেভ করে।"""
    df = load_dataset()
    results: dict = {}

    for disease in config.DISEASE_KEYS:
        X = df[config.FEATURE_NAMES]
        y = df[disease]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y,
        )
        disease_res = {}
        print(f"\n🩺 {config.DISEASES[disease]['name_bn']}")
        for name, model in _models().items():
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_te)[:, 1]
            pred = (proba >= 0.5).astype(int)
            acc = round(float(accuracy_score(y_te, pred)), 4)
            auc = round(float(roc_auc_score(y_te, proba)), 4)
            disease_res[name] = {"accuracy": acc, "roc_auc": auc}
            print(f"   {name:22s}: Accuracy {acc:.4f} | ROC-AUC {auc:.4f}")
        results[disease] = disease_res

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 তুলনা সেভ হয়েছে: {config.COMPARISON_PATH}")

    _plot(results)
    return results


def _plot(results: dict) -> None:
    """ROC-AUC তুলনার বার চার্ট docs/model_comparison.png-এ সেভ করে।"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("⚠️  matplotlib নেই — চার্ট এড়িয়ে যাওয়া হলো।")
        return

    model_names = list(next(iter(results.values())).keys())
    diseases = [config.DISEASES[d]["name_en"] for d in results]
    x = np.arange(len(diseases))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(model_names):
        scores = [results[d][m]["roc_auc"] for d in results]
        ax.bar(x + (i - 1) * width, scores, width, label=m)

    ax.set_ylabel("ROC-AUC")
    ax.set_title("MediPredict — Model Comparison (ROC-AUC)")
    ax.set_xticks(x)
    ax.set_xticklabels(diseases)
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = config.ROOT_DIR / "docs" / "model_comparison.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"📈 চার্ট সেভ হয়েছে: {out}")


if __name__ == "__main__":
    compare()
