"""MediPredict পাইপলাইনের বেসিক টেস্ট।

ব্যবহার:
    python -m pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data import generate_dataset


def test_dataset_shape():
    df = generate_dataset(n_samples=500)
    assert df.shape[0] == 500
    assert config.TARGET_NAME in df.columns
    for name in config.FEATURE_NAMES:
        assert name in df.columns


def test_dataset_balance():
    """উভয় শ্রেণি (সুস্থ ও ঝুঁকিপূর্ণ) উপস্থিত থাকতে হবে।"""
    df = generate_dataset(n_samples=2000)
    counts = df[config.TARGET_NAME].value_counts()
    assert set(counts.index) == {0, 1}
    # কোনো শ্রেণি যেন খুব বিরল না হয়
    assert counts.min() > 100


def test_predictor_output(tmp_path):
    """প্রশিক্ষণ ছাড়া predictor টেস্ট করতে একটি ছোট মডেল তৈরি করা হয়।"""
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    df = generate_dataset(n_samples=1000)
    X = df[config.FEATURE_NAMES]
    y = df[config.TARGET_NAME]
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(n_estimators=20, max_depth=3, random_state=42)
    model.fit(X_tr, y_tr)

    sample = {name: float(X_tr.iloc[0][name]) for name in config.FEATURE_NAMES}
    import pandas as pd
    prob = float(model.predict_proba(pd.DataFrame([sample]))[0, 1])
    assert 0.0 <= prob <= 1.0
