"""প্রোবাবিলিটি ক্যালিব্রেশন ও অনিশ্চয়তা পরিমাপ।

দুটি কাজ করে:
1. প্রতিটি রোগের জন্য reliability (calibration) curve ও Brier score হিসাব করে
   এবং একটি চার্ট তৈরি করে — মডেলের সম্ভাবনা কতটা নির্ভরযোগ্য তা দেখায়।
2. bootstrap ensemble (K টি মডেল) প্রশিক্ষণ দিয়ে সেভ করে, যা দিয়ে প্রতিটি
   পূর্বাভাসের জন্য confidence interval (অনিশ্চয়তা) নির্ণয় করা যায়।

ব্যবহার:
    python -m src.calibrate
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from . import config
from .data import load_dataset

N_ENSEMBLE = 8  # bootstrap মডেলের সংখ্যা


def ensemble_path(disease: str, k: int):
    return config.MODEL_DIR / f"ens_{disease}_{k}.json"


def train_ensemble(df, disease: str) -> None:
    """bootstrap resample-এ K টি মডেল প্রশিক্ষণ দিয়ে সেভ করে।"""
    X = df[config.FEATURE_NAMES].to_numpy()
    y = df[disease].to_numpy()
    n = len(X)
    rng = np.random.default_rng(config.RANDOM_SEED)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for k in range(N_ENSEMBLE):
        idx = rng.integers(0, n, n)  # bootstrap নমুনা
        model = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            max_depth=4, learning_rate=0.1, n_estimators=120,
            subsample=0.9, colsample_bytree=0.9, random_state=k,
        )
        model.fit(X[idx], y[idx])
        model.save_model(ensemble_path(disease, k))


def calibrate() -> dict:
    """ক্যালিব্রেশন কার্ভ + Brier score হিসাব করে এবং ensemble প্রশিক্ষণ দেয়।"""
    df = load_dataset()
    results: dict = {}

    for disease in config.DISEASE_KEYS:
        X = df[config.FEATURE_NAMES]
        y = df[disease]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y,
        )
        model = XGBClassifier(**config.XGB_PARAMS)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]

        brier = float(brier_score_loss(y_te, proba))
        frac_pos, mean_pred = calibration_curve(y_te, proba, n_bins=10, strategy="quantile")
        results[disease] = {
            "name_en": config.DISEASES[disease]["name_en"],
            "brier_score": round(brier, 4),
            "curve": {
                "mean_predicted": [round(float(v), 4) for v in mean_pred],
                "fraction_positive": [round(float(v), 4) for v in frac_pos],
            },
        }
        print(f"🩺 {config.DISEASES[disease]['name_bn']}: Brier score {brier:.4f}")

        print(f"   {N_ENSEMBLE} টি bootstrap মডেল প্রশিক্ষণ হচ্ছে...")
        train_ensemble(df, disease)

    with open(config.MODEL_DIR / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 ক্যালিব্রেশন সেভ হয়েছে: {config.MODEL_DIR / 'calibration.json'}")

    _plot(results)
    return results


def _plot(results: dict) -> None:
    """reliability diagram docs/calibration.png-এ সেভ করে।"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib নেই — চার্ট এড়িয়ে যাওয়া হলো।")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    for disease, res in results.items():
        ax.plot(
            res["curve"]["mean_predicted"], res["curve"]["fraction_positive"],
            marker="o", label=f"{res['name_en']} (Brier {res['brier_score']})",
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("MediPredict — Calibration (Reliability) Curve")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = config.ROOT_DIR / "docs" / "calibration.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"📈 চার্ট সেভ হয়েছে: {out}")


if __name__ == "__main__":
    calibrate()
