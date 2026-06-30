"""প্রশিক্ষিত XGBoost মডেল দিয়ে রোগের সম্ভাবনা পূর্বাভাস।

ব্যবহার (CLI, ইন্টারঅ্যাক্টিভ):
    python -m src.predict

প্রোগ্রাম্যাটিক:
    from src.predict import MediPredictor
    predictor = MediPredictor()
    result = predictor.predict({"age": 55, "glucose": 160, ...})
"""

from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd
from xgboost import XGBClassifier

from . import config
from .explain import explain_prediction
from .recommend import get_recommendations


class MediPredictor:
    """মডেল লোড করে এবং একক রোগীর জন্য পূর্বাভাস দেয়।"""

    def __init__(self, model_path=config.MODEL_PATH, metadata_path=config.METADATA_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"মডেল পাওয়া যায়নি: {model_path}\n"
                "প্রথমে প্রশিক্ষণ দিন: python -m src.train"
            )
        self.model = XGBClassifier()
        self.model.load_model(model_path)

        self.metadata = {}
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                self.metadata = json.load(f)

    def predict(self, features: dict, explain: bool = False) -> dict:
        """ফিচার ডিকশনারি নিয়ে সম্ভাবনা ও ঝুঁকি স্তর রিটার্ন করে।

        explain=True হলে ফিচার অবদান ও স্বাস্থ্য পরামর্শও যুক্ত হয়।
        """
        row = {name: float(features[name]) for name in config.FEATURE_NAMES}
        X = pd.DataFrame([row], columns=config.FEATURE_NAMES)

        probability = float(self.model.predict_proba(X)[0, 1])
        prediction = int(probability >= 0.5)

        result = {
            "probability": round(probability, 4),
            "probability_percent": round(probability * 100, 1),
            "prediction": prediction,
            "risk_level": self._risk_level(probability),
            "label": "ঝুঁকিপূর্ণ" if prediction == 1 else "সুস্থ",
        }

        if explain:
            result["explanation"] = explain_prediction(self.model, row)
            result["recommendations"] = get_recommendations(row)

        return result

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """একাধিক রোগীর DataFrame নিয়ে প্রতিটির জন্য পূর্বাভাস রিটার্ন করে।"""
        missing = [c for c in config.FEATURE_NAMES if c not in df.columns]
        if missing:
            raise ValueError(f"অনুপস্থিত কলাম: {', '.join(missing)}")

        X = df[config.FEATURE_NAMES].astype(float)
        proba = self.model.predict_proba(X)[:, 1]

        out = df.copy()
        out["probability"] = proba.round(4)
        out["probability_percent"] = (proba * 100).round(1)
        out["prediction"] = (proba >= 0.5).astype(int)
        out["risk_level"] = [self._risk_level(p) for p in proba]
        return out

    @staticmethod
    def _risk_level(prob: float) -> str:
        """সম্ভাবনাকে বাংলা ঝুঁকি স্তরে রূপান্তর করে।"""
        if prob < 0.25:
            return "নিম্ন ঝুঁকি"
        if prob < 0.50:
            return "মাঝারি ঝুঁকি"
        if prob < 0.75:
            return "উচ্চ ঝুঁকি"
        return "অতি উচ্চ ঝুঁকি"


@lru_cache(maxsize=1)
def get_predictor() -> MediPredictor:
    """সিঙ্গলটন প্রেডিক্টর (ওয়েব অ্যাপে বারবার লোড এড়াতে)।"""
    return MediPredictor()


def _interactive() -> None:
    predictor = get_predictor()
    print("=== MediPredict — রোগের সম্ভাবনা পূর্বাভাস ===\n")
    features = {}
    for f in config.FEATURES:
        while True:
            raw = input(f"{f['label']} [{f['default']}]: ").strip()
            if raw == "":
                features[f["name"]] = f["default"]
                break
            try:
                features[f["name"]] = float(raw)
                break
            except ValueError:
                print("  ⚠️  অনুগ্রহ করে একটি সংখ্যা দিন।")

    result = predictor.predict(features, explain=True)
    print("\n--- ফলাফল ---")
    print(f"রোগের সম্ভাবনা : {result['probability_percent']}%")
    print(f"ঝুঁকি স্তর      : {result['risk_level']}")
    print(f"পূর্বাভাস       : {result['label']}")

    print("\n--- প্রধান প্রভাবক ফ্যাক্টর ---")
    for item in result["explanation"]:
        sign = "▲" if item["contribution"] > 0 else "▼"
        print(f"  {sign} {item['label']} ({item['value']}) — ঝুঁকি {item['direction']}")

    print("\n--- স্বাস্থ্য পরামর্শ ---")
    for tip in result["recommendations"]:
        print(f"  {tip}")

    print("\n⚠️  দ্রষ্টব্য: এটি কেবল একটি ML ডেমো; প্রকৃত চিকিৎসা পরামর্শের বিকল্প নয়।")


if __name__ == "__main__":
    _interactive()
