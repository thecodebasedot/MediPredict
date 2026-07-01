"""প্রশিক্ষিত XGBoost মডেল দিয়ে মাল্টি-ডিজিজ রোগের সম্ভাবনা পূর্বাভাস।

ব্যবহার (CLI, ইন্টারঅ্যাক্টিভ):
    python -m src.predict

প্রোগ্রাম্যাটিক:
    from src.predict import MediPredictor
    predictor = MediPredictor()
    result = predictor.predict({"age": 55, "glucose": 160, ...}, disease="diabetes")
    summary = predictor.predict_all({"age": 55, ...})
"""

from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd
from xgboost import XGBClassifier

from . import config
from .counterfactual import suggest_action_plan
from .explain import explain_prediction
from .recommend import get_recommendations


class MediPredictor:
    """প্রতিটি রোগের মডেল লোড করে এবং পূর্বাভাস দেয়।"""

    def __init__(self):
        self.models: dict[str, XGBClassifier] = {}
        self.ensembles: dict[str, list[XGBClassifier]] = {}
        missing = []
        for disease in config.DISEASE_KEYS:
            path = config.model_path(disease)
            if not path.exists():
                missing.append(disease)
                continue
            model = XGBClassifier()
            model.load_model(path)
            self.models[disease] = model
            self._load_ensemble(disease)

        if missing:
            raise FileNotFoundError(
                f"মডেল পাওয়া যায়নি: {', '.join(missing)}\n"
                "প্রথমে প্রশিক্ষণ দিন: python -m src.train"
            )

    def _load_ensemble(self, disease: str) -> None:
        """অনিশ্চয়তা পরিমাপের জন্য bootstrap ensemble লোড করে (থাকলে)।"""
        from .calibrate import N_ENSEMBLE, ensemble_path

        members = []
        for k in range(N_ENSEMBLE):
            path = ensemble_path(disease, k)
            if path.exists():
                m = XGBClassifier()
                m.load_model(path)
                members.append(m)
        if members:
            self.ensembles[disease] = members

        self.metadata = {}
        if config.METADATA_PATH.exists():
            with open(config.METADATA_PATH, encoding="utf-8") as f:
                self.metadata = json.load(f)

    # ------------------------------------------------------------------
    def predict(self, features: dict, disease: str = config.DEFAULT_DISEASE,
                explain: bool = False) -> dict:
        """একটি নির্দিষ্ট রোগের জন্য সম্ভাবনা, ঝুঁকি স্তর ও (ঐচ্ছিক) ব্যাখ্যা।"""
        if disease not in self.models:
            raise ValueError(f"অজানা রোগ: {disease}")

        row = {name: float(features[name]) for name in config.FEATURE_NAMES}
        X = pd.DataFrame([row], columns=config.FEATURE_NAMES)

        probability = float(self.models[disease].predict_proba(X)[0, 1])
        prediction = int(probability >= 0.5)
        meta = config.DISEASES[disease]

        result = {
            "disease": disease,
            "disease_name": meta["name_bn"],
            "disease_name_en": meta["name_en"],
            "probability": round(probability, 4),
            "probability_percent": round(probability * 100, 1),
            "prediction": prediction,
            "risk_level": self._risk_level(probability, "bn"),
            "risk_level_en": self._risk_level(probability, "en"),
            "label": "ঝুঁকিপূর্ণ" if prediction == 1 else "সুস্থ",
            "label_en": "At Risk" if prediction == 1 else "Healthy",
        }

        if explain:
            result["explanation"] = explain_prediction(self.models[disease], row)
            result["recommendations"] = get_recommendations(row)
            # ঝুঁকিপূর্ণ হলে অ্যাকশন প্ল্যান যোগ করা হয়
            if probability > 0.25:
                result["action_plan"] = suggest_action_plan(self.models[disease], row)
            # bootstrap ensemble থাকলে confidence interval যোগ করা হয়
            ci = self._confidence_interval(disease, X, probability)
            if ci:
                result.update(ci)

        return result

    def _confidence_interval(self, disease: str, X, probability: float):
        """bootstrap ensemble থেকে ৮০% confidence interval ও অনিশ্চয়তা নির্ণয়।"""
        members = self.ensembles.get(disease)
        if not members:
            return None
        import numpy as np

        probs = np.array([float(m.predict_proba(X)[0, 1]) for m in members])
        lo = min(float(np.percentile(probs, 10)), probability)
        hi = max(float(np.percentile(probs, 90)), probability)
        return {
            "confidence_interval_percent": [round(lo * 100, 1), round(hi * 100, 1)],
            "uncertainty_percent": round(float(probs.std()) * 100, 1),
        }

    def predict_all(self, features: dict) -> list[dict]:
        """সব রোগের জন্য সংক্ষিপ্ত পূর্বাভাস সারাংশ।"""
        return [self.predict(features, disease=d) for d in config.DISEASE_KEYS]

    def predict_batch(self, df: pd.DataFrame, disease: str = config.DEFAULT_DISEASE) -> pd.DataFrame:
        """একাধিক রোগীর DataFrame নিয়ে নির্দিষ্ট রোগের পূর্বাভাস।"""
        if disease not in self.models:
            raise ValueError(f"অজানা রোগ: {disease}")
        missing = [c for c in config.FEATURE_NAMES if c not in df.columns]
        if missing:
            raise ValueError(f"অনুপস্থিত কলাম: {', '.join(missing)}")

        X = df[config.FEATURE_NAMES].astype(float)
        proba = self.models[disease].predict_proba(X)[:, 1]

        out = df.copy()
        out["probability"] = proba.round(4)
        out["probability_percent"] = (proba * 100).round(1)
        out["prediction"] = (proba >= 0.5).astype(int)
        out["risk_level"] = [self._risk_level(p, "bn") for p in proba]
        return out

    @staticmethod
    def _risk_level(prob: float, lang: str = "bn") -> str:
        """সম্ভাবনাকে ঝুঁকি স্তরে রূপান্তর করে (bn/en)।"""
        for level in config.RISK_LEVELS:
            if prob < level["max"]:
                return level[lang]
        return config.RISK_LEVELS[-1][lang]


@lru_cache(maxsize=1)
def get_predictor() -> MediPredictor:
    """সিঙ্গলটন প্রেডিক্টর (ওয়েব অ্যাপে বারবার লোড এড়াতে)।"""
    return MediPredictor()


def _interactive() -> None:
    predictor = get_predictor()
    print("=== MediPredict — মাল্টি-ডিজিজ রোগের সম্ভাবনা পূর্বাভাস ===\n")

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

    print("\n--- সব রোগের সারাংশ ---")
    for r in predictor.predict_all(features):
        print(f"  {r['disease_name']:12s}: {r['probability_percent']:5.1f}%  ({r['risk_level']})")

    # ডিফল্ট রোগের বিস্তারিত ব্যাখ্যা
    detail = predictor.predict(features, explain=True)
    print(f"\n--- {detail['disease_name']} — প্রধান প্রভাবক ফ্যাক্টর ---")
    for item in detail["explanation"]:
        sign = "▲" if item["contribution"] > 0 else "▼"
        print(f"  {sign} {item['label']} ({item['value']}) — ঝুঁকি {item['direction']}")

    print("\n--- স্বাস্থ্য পরামর্শ ---")
    for tip in detail["recommendations"]:
        print(f"  {tip['bn']}")

    print("\n⚠️  দ্রষ্টব্য: এটি কেবল একটি ML ডেমো; প্রকৃত চিকিৎসা পরামর্শের বিকল্প নয়।")


if __name__ == "__main__":
    _interactive()
