"""সিনথেটিক মাল্টি-ডিজিজ মেডিকেল ডেটাসেট তৈরি ও লোড করার ইউটিলিটি।

বাস্তব রোগীর ডেটা সংবেদনশীল ও সীমিত। তাই এখানে চিকিৎসাগতভাবে যুক্তিসঙ্গত
নিয়মের উপর ভিত্তি করে একটি সিনথেটিক ডেটাসেট তৈরি করা হয়, যেখানে প্রতিটি
রোগের (ডায়াবেটিস, হৃদরোগ, উচ্চ রক্তচাপ) জন্য আলাদা টার্গেট কলাম থাকে।
বাস্তব ব্যবহারে এই ফাইলটিকে প্রকৃত ডেটাসেট লোডার দিয়ে প্রতিস্থাপন করুন।
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _risk_label(features: dict, weights: dict, rng) -> np.ndarray:
    """রোগ-নির্দিষ্ট ওজন প্রয়োগ করে বাইনারি লেবেল তৈরি করে।"""
    n = len(next(iter(features.values())))
    risk = np.zeros(n)
    for name, coef in weights.items():
        risk += coef * (features[name] - config.FEATURE_CENTERS[name])
    noise = rng.normal(0, 0.5, n)
    prob = 1 / (1 + np.exp(-(risk + noise) * 0.5))
    return (prob > 0.5).astype(int)


def generate_dataset(n_samples: int = 6000, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """চিকিৎসাগত নিয়মভিত্তিক সিনথেটিক ডেটাসেট তৈরি করে (প্রতি রোগে একটি টার্গেট)।"""
    rng = np.random.default_rng(seed)

    features = {
        "age": rng.integers(18, 90, n_samples).astype(float),
        "glucose": rng.normal(110, 30, n_samples).clip(50, 300),
        "blood_pressure": rng.normal(80, 15, n_samples).clip(40, 200),
        "bmi": rng.normal(26, 6, n_samples).clip(12, 55),
        "cholesterol": rng.normal(190, 40, n_samples).clip(100, 400),
        "insulin": rng.normal(85, 45, n_samples).clip(0, 300),
        "heart_rate": rng.normal(75, 12, n_samples).clip(45, 170),
        "smoking": rng.binomial(1, 0.3, n_samples).astype(float),
        "physical_activity": rng.integers(0, 11, n_samples).astype(float),
        "family_history": rng.binomial(1, 0.25, n_samples).astype(float),
    }

    df = pd.DataFrame({k: np.round(v, 1) for k, v in features.items()})

    # প্রতিটি রোগের জন্য টার্গেট কলাম
    for disease, meta in config.DISEASES.items():
        df[disease] = _risk_label(features, meta["weights"], rng)

    return df


def save_dataset(df: pd.DataFrame, path=config.DATASET_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_dataset(path=config.DATASET_PATH) -> pd.DataFrame:
    """CSV থেকে ডেটাসেট লোড করে; না থাকলে বা পুরনো ফরম্যাট হলে নতুন তৈরি করে।"""
    if path.exists():
        df = pd.read_csv(path)
        if all(d in df.columns for d in config.DISEASE_KEYS):
            return df
    df = generate_dataset()
    save_dataset(df, path)
    return df


if __name__ == "__main__":
    data = generate_dataset()
    save_dataset(data)
    print(f"ডেটাসেট তৈরি হয়েছে: {config.DATASET_PATH}")
    print(f"আকার: {data.shape}")
    for d, meta in config.DISEASES.items():
        pos = int(data[d].sum())
        print(f"  {meta['name_bn']:12s}: ঝুঁকিপূর্ণ {pos} / {len(data)}")
