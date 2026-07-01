"""হাইপারপ্যারামিটার টিউনিং — RandomizedSearchCV দিয়ে সেরা XGBoost প্যারামিটার খোঁজা।

নির্দিষ্ট রোগের জন্য ক্রস-ভ্যালিডেশন করে সেরা প্যারামিটার সেট বের করে এবং
ফলাফল models/best_params_<disease>.json-এ সেভ করে।

ব্যবহার:
    python -m src.tune                 # ডিফল্ট রোগ (diabetes)
    python -m src.tune heart           # নির্দিষ্ট রোগ
"""

from __future__ import annotations

import json
import sys

from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from . import config
from .data import load_dataset

PARAM_DISTRIBUTIONS = {
    "max_depth": randint(3, 8),
    "learning_rate": uniform(0.02, 0.28),
    "n_estimators": randint(100, 400),
    "subsample": uniform(0.7, 0.3),
    "colsample_bytree": uniform(0.7, 0.3),
    "min_child_weight": randint(1, 6),
    "gamma": uniform(0.0, 0.4),
}


def tune(disease: str = config.DEFAULT_DISEASE, n_iter: int = 25) -> dict:
    """নির্দিষ্ট রোগের জন্য হাইপারপ্যারামিটার টিউন করে সেরা প্যারামিটার রিটার্ন করে।"""
    if disease not in config.DISEASES:
        raise ValueError(f"অজানা রোগ: {disease}")

    df = load_dataset()
    X = df[config.FEATURE_NAMES]
    y = df[disease]

    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=config.RANDOM_SEED,
    )

    print(f"🔧 {config.DISEASES[disease]['name_bn']} — {n_iter} ইটারেশন টিউনিং চলছে...")
    search = RandomizedSearchCV(
        base,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=4,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X, y)

    best = {
        "disease": disease,
        "best_score_roc_auc": round(float(search.best_score_), 4),
        "best_params": search.best_params_,
        "baseline_params": config.XGB_PARAMS,
    }

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.MODEL_DIR / f"best_params_{disease}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2, default=str)

    print(f"✅ সেরা ROC-AUC (CV): {best['best_score_roc_auc']}")
    print("   সেরা প্যারামিটার:")
    for k, v in search.best_params_.items():
        print(f"     {k:18s}: {v}")
    print(f"💾 সেভ হয়েছে: {out_path}")
    return best


if __name__ == "__main__":
    disease = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_DISEASE
    tune(disease)
