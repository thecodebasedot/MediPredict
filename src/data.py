"""সিনথেটিক মেডিকেল ডেটাসেট তৈরি ও লোড করার ইউটিলিটি।

বাস্তব রোগীর ডেটা সংবেদনশীল ও সীমিত। তাই এখানে চিকিৎসাগতভাবে যুক্তিসঙ্গত
নিয়মের উপর ভিত্তি করে একটি সিনথেটিক ডেটাসেট তৈরি করা হয়, যা দিয়ে XGBoost
মডেল প্রশিক্ষণ ও ডেমো করা যায়। বাস্তব ব্যবহারে এই ফাইলটিকে প্রকৃত
ডেটাসেট লোডার দিয়ে প্রতিস্থাপন করুন।
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def generate_dataset(n_samples: int = 5000, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """চিকিৎসাগত নিয়মভিত্তিক সিনথেটিক ডেটাসেট তৈরি করে।

    রোগের ঝুঁকি গ্লুকোজ, বিএমআই, রক্তচাপ, বয়স, ধূমপান, পারিবারিক ইতিহাস
    ইত্যাদির সাথে বাড়ে এবং শারীরিক সক্রিয়তার সাথে কমে।
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 90, n_samples)
    glucose = rng.normal(110, 30, n_samples).clip(50, 300)
    blood_pressure = rng.normal(80, 15, n_samples).clip(40, 200)
    bmi = rng.normal(26, 6, n_samples).clip(12, 55)
    cholesterol = rng.normal(190, 40, n_samples).clip(100, 400)
    insulin = rng.normal(85, 45, n_samples).clip(0, 300)
    heart_rate = rng.normal(75, 12, n_samples).clip(45, 170)
    smoking = rng.binomial(1, 0.3, n_samples)
    physical_activity = rng.integers(0, 11, n_samples)
    family_history = rng.binomial(1, 0.25, n_samples)

    # ঝুঁকির রৈখিক স্কোর (চিকিৎসাগত অন্তর্দৃষ্টির আনুমানিক ওজন সহ)
    risk = (
        0.030 * (glucose - 110)
        + 0.040 * (bmi - 26)
        + 0.020 * (blood_pressure - 80)
        + 0.025 * (age - 50)
        + 0.010 * (cholesterol - 190)
        + 0.008 * (insulin - 85)
        + 0.6 * smoking
        + 0.7 * family_history
        - 0.12 * physical_activity
        + 0.010 * (heart_rate - 75)
    )

    # লজিস্টিক রূপান্তর + র‍্যান্ডম নয়েজ → সম্ভাবনা → বাইনারি লেবেল
    noise = rng.normal(0, 0.5, n_samples)
    prob = 1 / (1 + np.exp(-(risk + noise) * 0.5))
    disease = (prob > 0.5).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "glucose": glucose.round(1),
            "blood_pressure": blood_pressure.round(1),
            "bmi": bmi.round(1),
            "cholesterol": cholesterol.round(1),
            "insulin": insulin.round(1),
            "heart_rate": heart_rate.round(1),
            "smoking": smoking,
            "physical_activity": physical_activity,
            "family_history": family_history,
            config.TARGET_NAME: disease,
        }
    )
    return df


def save_dataset(df: pd.DataFrame, path=config.DATASET_PATH) -> None:
    """ডেটাসেট CSV হিসেবে সেভ করে।"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_dataset(path=config.DATASET_PATH) -> pd.DataFrame:
    """CSV থেকে ডেটাসেট লোড করে; না থাকলে তৈরি করে সেভ করে।"""
    if not path.exists():
        df = generate_dataset()
        save_dataset(df, path)
        return df
    return pd.read_csv(path)


if __name__ == "__main__":
    data = generate_dataset()
    save_dataset(data)
    print(f"ডেটাসেট তৈরি হয়েছে: {config.DATASET_PATH}")
    print(f"আকার: {data.shape}")
    print(f"রোগ বিতরণ:\n{data[config.TARGET_NAME].value_counts()}")
