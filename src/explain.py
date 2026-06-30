"""প্রেডিকশন ব্যাখ্যা — কোন ফিচার পূর্বাভাসে কতটা অবদান রাখল।

XGBoost-এর built-in SHAP-স্টাইল contribution (`pred_contribs`) ব্যবহার করে
প্রতিটি ফিচারের অবদান বের করা হয়। ধনাত্মক অবদান ঝুঁকি বাড়ায়, ঋণাত্মক কমায়।
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier

from . import config

# ফিচারের বাংলা লেবেল ম্যাপ (UI-তে দেখানোর জন্য)
_LABELS = {f["name"]: f["label"] for f in config.FEATURES}


def explain_prediction(model: XGBClassifier, features: dict, top_k: int = 5) -> list[dict]:
    """একক রোগীর প্রেডিকশনে প্রতিটি ফিচারের অবদান হিসাব করে।

    Returns: top_k অবদানের তালিকা (পরম মান অনুসারে সাজানো), প্রতিটিতে
    ফিচারের নাম, বাংলা লেবেল, রোগীর মান, এবং অবদান (ধনাত্মক=ঝুঁকি বাড়ায়)।
    """
    row = {name: float(features[name]) for name in config.FEATURE_NAMES}
    X = pd.DataFrame([row], columns=config.FEATURE_NAMES)

    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X, feature_names=config.FEATURE_NAMES)
    # pred_contribs শেষ কলামে bias দেয়, তাই সেটি বাদ দেওয়া হয়
    contribs = booster.predict(dmatrix, pred_contribs=True)[0][:-1]

    items = []
    for name, contrib in zip(config.FEATURE_NAMES, contribs):
        items.append(
            {
                "feature": name,
                "label": _LABELS.get(name, name),
                "value": row[name],
                "contribution": round(float(contrib), 4),
                "direction": "বাড়ায়" if contrib > 0 else "কমায়",
            }
        )

    items.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    return items[:top_k]
