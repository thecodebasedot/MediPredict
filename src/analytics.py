"""অ্যানালিটিক্স ও ডেটা-ড্রিফট মনিটরিং।

সংরক্ষিত হিস্টোরি থেকে সমষ্টিগত পরিসংখ্যান এবং ইনপুট বিতরণের পরিবর্তন
(data drift) হিসাব করে। ড্রিফট পরিমাপে Population Stability Index (PSI)
ব্যবহৃত হয় — রেফারেন্স (প্রশিক্ষণ) ডেটা ও সাম্প্রতিক ইনপুটের তুলনা।
"""

from __future__ import annotations

import numpy as np

from . import config
from .data import load_dataset
from .history import get_all


def summary() -> dict:
    """হিস্টোরির সমষ্টিগত সারাংশ: মোট, রোগভিত্তিক গড় ঝুঁকি, ঝুঁকি-স্তর বিতরণ।"""
    records = get_all()
    total = len(records)
    if total == 0:
        return {"total": 0, "by_disease": {}, "risk_distribution": {}}

    by_disease: dict[str, list[float]] = {}
    risk_dist: dict[str, int] = {}
    for r in records:
        by_disease.setdefault(r["disease"], []).append(r["probability_percent"])
        risk_dist[r["risk_level"]] = risk_dist.get(r["risk_level"], 0) + 1

    by_disease_summary = {
        d: {
            "count": len(probs),
            "avg_probability_percent": round(float(np.mean(probs)), 1),
            "name_bn": config.DISEASES.get(d, {}).get("name_bn", d),
        }
        for d, probs in by_disease.items()
    }
    return {
        "total": total,
        "by_disease": by_disease_summary,
        "risk_distribution": risk_dist,
    }


def _psi(reference: np.ndarray, observed: np.ndarray, bins: int = 10) -> float:
    """একটি ফিচারের জন্য Population Stability Index হিসাব করে।

    PSI < 0.1: স্থিতিশীল | 0.1–0.25: সামান্য ড্রিফট | > 0.25: উল্লেখযোগ্য ড্রিফট।
    """
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts = np.histogram(reference, bins=edges)[0].astype(float)
    obs_counts = np.histogram(observed, bins=edges)[0].astype(float)

    eps = 1e-6
    ref_pct = ref_counts / max(ref_counts.sum(), eps) + eps
    obs_pct = obs_counts / max(obs_counts.sum(), eps) + eps
    return float(np.sum((obs_pct - ref_pct) * np.log(obs_pct / ref_pct)))


def drift(min_samples: int = 10) -> dict:
    """প্রশিক্ষণ ডেটা ও সংরক্ষিত ইনপুটের মধ্যে প্রতি ফিচারে PSI ড্রিফট।"""
    records = get_all()
    inputs = [r["features"] for r in records]
    if len(inputs) < min_samples:
        return {
            "status": "insufficient_data",
            "message": f"ড্রিফট হিসাবে অন্তত {min_samples}টি রেকর্ড দরকার (আছে {len(inputs)})।",
            "features": {},
        }

    reference = load_dataset()
    feature_drift = {}
    for name in config.FEATURE_NAMES:
        ref = reference[name].to_numpy(dtype=float)
        obs = np.array([float(x.get(name, np.nan)) for x in inputs], dtype=float)
        obs = obs[~np.isnan(obs)]
        if len(obs) < min_samples:
            continue
        psi = round(_psi(ref, obs), 4)
        feature_drift[name] = {
            "psi": psi,
            "level": ("stable" if psi < 0.1 else "minor" if psi < 0.25 else "significant"),
        }

    max_psi = max((v["psi"] for v in feature_drift.values()), default=0.0)
    return {
        "status": "ok",
        "n_observed": len(inputs),
        "overall_level": ("stable" if max_psi < 0.1 else "minor" if max_psi < 0.25 else "significant"),
        "features": feature_drift,
    }
