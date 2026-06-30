"""Counterfactual অ্যাকশন প্ল্যান — ঝুঁকি কমাতে সর্বনিম্ন কী কী বদলাতে হবে।

একটি greedy coordinate-descent অ্যালগরিদম: প্রতিটি ধাপে যে পরিবর্তনযোগ্য
(modifiable) ফিচারটি ঝুঁকি সবচেয়ে বেশি কমায় সেটি "স্বাস্থ্যকর" লক্ষ্যমানে
সরিয়ে দেওয়া হয়, যতক্ষণ না ঝুঁকি লক্ষ্যমাত্রার নিচে নামে বা আর উন্নতি না হয়।

বয়স ও পারিবারিক ইতিহাস পরিবর্তনযোগ্য নয়, তাই বাদ দেওয়া হয়।
"""

from __future__ import annotations

import pandas as pd

from . import config

# পরিবর্তনযোগ্য ফিচার → স্বাস্থ্যকর লক্ষ্যমান এবং উন্নতির দিক
# direction: "lower" মানে বর্তমান মান লক্ষ্যের চেয়ে বেশি হলে কমানো উপকারী,
#            "higher" মানে কম হলে বাড়ানো উপকারী।
MODIFIABLE = {
    "glucose":           {"target": 95,  "direction": "lower"},
    "bmi":               {"target": 23,  "direction": "lower"},
    "blood_pressure":    {"target": 78,  "direction": "lower"},
    "cholesterol":       {"target": 170, "direction": "lower"},
    "insulin":           {"target": 85,  "direction": "lower"},
    "heart_rate":        {"target": 70,  "direction": "lower"},
    "smoking":           {"target": 0,   "direction": "lower"},
    "physical_activity": {"target": 8,   "direction": "higher"},
}

_LABELS = {f["name"]: f["label"] for f in config.FEATURES}
_LABELS_EN = {f["name"]: f["label_en"] for f in config.FEATURES}


def _beneficial_target(name: str, current: float):
    """ফিচারটি উপকারীভাবে সরানো গেলে নতুন (লক্ষ্য) মান রিটার্ন করে, নাহলে None।"""
    spec = MODIFIABLE[name]
    if spec["direction"] == "lower" and current > spec["target"]:
        return spec["target"]
    if spec["direction"] == "higher" and current < spec["target"]:
        return spec["target"]
    return None


def _proba(model, features: dict) -> float:
    X = pd.DataFrame([{n: features[n] for n in config.FEATURE_NAMES}], columns=config.FEATURE_NAMES)
    return float(model.predict_proba(X)[0, 1])


def suggest_action_plan(model, features: dict, target_prob: float = 0.25,
                        max_steps: int = 6) -> dict:
    """ঝুঁকি কমানোর greedy অ্যাকশন প্ল্যান তৈরি করে।

    Returns: শুরু/শেষ সম্ভাবনা এবং ক্রমানুসারে পরিবর্তনের তালিকা।
    """
    current = {n: float(features[n]) for n in config.FEATURE_NAMES}
    start_prob = _proba(model, current)
    steps: list[dict] = []
    changed: set[str] = set()

    for _ in range(max_steps):
        prob_now = _proba(model, current)
        if prob_now <= target_prob:
            break

        best = None  # (drop, name, new_value, new_prob)
        for name in MODIFIABLE:
            if name in changed:
                continue
            new_val = _beneficial_target(name, current[name])
            if new_val is None:
                continue
            trial = dict(current)
            trial[name] = float(new_val)
            new_prob = _proba(model, trial)
            drop = prob_now - new_prob
            if drop > 0 and (best is None or drop > best[0]):
                best = (drop, name, float(new_val), new_prob)

        if best is None or best[0] < 1e-4:
            break  # আর উন্নতি সম্ভব নয়

        _, name, new_val, new_prob = best
        steps.append({
            "feature": name,
            "label": _LABELS.get(name, name),
            "label_en": _LABELS_EN.get(name, name),
            "from": round(current[name], 1),
            "to": round(new_val, 1),
            "new_probability_percent": round(new_prob * 100, 1),
            "risk_drop_percent": round(best[0] * 100, 1),
        })
        current[name] = new_val
        changed.add(name)

    end_prob = _proba(model, current)
    return {
        "start_probability_percent": round(start_prob * 100, 1),
        "end_probability_percent": round(end_prob * 100, 1),
        "total_drop_percent": round((start_prob - end_prob) * 100, 1),
        "achieved_target": end_prob <= target_prob,
        "steps": steps,
    }
