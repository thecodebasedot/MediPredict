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
    for disease in config.DISEASE_KEYS:
        assert disease in df.columns  # প্রতি রোগে একটি টার্গেট কলাম
    for name in config.FEATURE_NAMES:
        assert name in df.columns


def test_dataset_balance():
    """প্রতিটি রোগে উভয় শ্রেণি (সুস্থ ও ঝুঁকিপূর্ণ) উপস্থিত থাকতে হবে।"""
    df = generate_dataset(n_samples=2000)
    for disease in config.DISEASE_KEYS:
        counts = df[disease].value_counts()
        assert set(counts.index) == {0, 1}
        assert counts.min() > 50  # কোনো শ্রেণি যেন খুব বিরল না হয়


def test_recommendations():
    from src.recommend import get_recommendations

    risky = {"glucose": 200, "bmi": 33, "blood_pressure": 150,
             "cholesterol": 260, "smoking": 1, "physical_activity": 1, "heart_rate": 110}
    tips = get_recommendations(risky)
    assert len(tips) >= 5  # একাধিক ঝুঁকি ফ্যাক্টর শনাক্ত হওয়া উচিত
    assert all("bn" in t and "en" in t for t in tips)  # দ্বিভাষিক

    healthy = {"glucose": 90, "bmi": 22, "blood_pressure": 75,
               "cholesterol": 170, "smoking": 0, "physical_activity": 8, "heart_rate": 70}
    tips = get_recommendations(healthy)
    assert len(tips) == 1  # শুধু "স্বাভাবিক" বার্তা


def test_explanation():
    """প্রশিক্ষিত মডেল দিয়ে ব্যাখ্যা টেস্ট।"""
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    from src.explain import explain_prediction

    df = generate_dataset(n_samples=1000)
    X, y = df[config.FEATURE_NAMES], df[config.DEFAULT_DISEASE]
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(n_estimators=30, max_depth=3, random_state=42)
    model.fit(X_tr, y_tr)

    sample = {name: float(X_tr.iloc[0][name]) for name in config.FEATURE_NAMES}
    exp = explain_prediction(model, sample, top_k=3)
    assert len(exp) == 3
    assert all("contribution" in e and "label" in e and "label_en" in e for e in exp)
    # পরম মান অনুসারে সাজানো কিনা
    abs_vals = [abs(e["contribution"]) for e in exp]
    assert abs_vals == sorted(abs_vals, reverse=True)


def test_risk_levels():
    """ঝুঁকি স্তর রূপান্তর সঠিক কিনা।"""
    from src.predict import MediPredictor

    assert MediPredictor._risk_level(0.1, "bn") == "নিম্ন ঝুঁকি"
    assert MediPredictor._risk_level(0.1, "en") == "Low Risk"
    assert MediPredictor._risk_level(0.9, "bn") == "অতি উচ্চ ঝুঁকি"
    assert MediPredictor._risk_level(0.9, "en") == "Very High Risk"


def test_history(tmp_path, monkeypatch):
    """হিস্টোরি সংরক্ষণ ও পুনরুদ্ধার টেস্ট।"""
    from src import history

    monkeypatch.setattr(history, "DB_PATH", tmp_path / "test_history.db")
    history.clear_history()

    result = {"disease": "diabetes", "disease_name": "ডায়াবেটিস",
              "probability_percent": 87.5, "risk_level": "উচ্চ ঝুঁকি"}
    rid = history.save_prediction(result, {"glucose": 180})
    assert rid > 0

    hist = history.get_history(limit=5)
    assert len(hist) == 1
    assert hist[0]["disease"] == "diabetes"
    assert hist[0]["features"]["glucose"] == 180

    assert history.clear_history() == 1
    assert history.get_history() == []


def test_counterfactual():
    """অ্যাকশন প্ল্যান উচ্চ-ঝুঁকি কেসে ঝুঁকি কমায় ও বৈধ ধাপ দেয়।"""
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    from src.counterfactual import suggest_action_plan, MODIFIABLE

    df = generate_dataset(n_samples=1500)
    X, y = df[config.FEATURE_NAMES], df[config.DEFAULT_DISEASE]
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(n_estimators=60, max_depth=3, random_state=42)
    model.fit(X_tr, y_tr)

    risky = {"age": 60, "glucose": 200, "blood_pressure": 150, "bmi": 38,
             "cholesterol": 280, "insulin": 200, "heart_rate": 100,
             "smoking": 1, "physical_activity": 1, "family_history": 1}
    plan = suggest_action_plan(model, risky)
    assert plan["end_probability_percent"] <= plan["start_probability_percent"]
    # অপরিবর্তনযোগ্য ফিচার (age, family_history) প্ল্যানে থাকবে না
    for step in plan["steps"]:
        assert step["feature"] in MODIFIABLE


def test_assistant_offline():
    """API key ছাড়া সহকারীর আচরণ (নেটওয়ার্ক ছাড়াই)।"""
    from src import assistant

    msgs = assistant.build_messages("why?", {"disease": "diabetes", "probability_percent": 80})
    assert msgs[0]["role"] == "user"
    assert "diabetes" in assistant.build_context_text({"disease": "diabetes"}).lower()
    # খালি প্রশ্নে ValueError
    import pytest
    with pytest.raises(ValueError):
        assistant.ask("")


def test_disease_config():
    """রোগ কনফিগ ও ওজনের বৈধতা।"""
    assert config.DEFAULT_DISEASE in config.DISEASES
    for disease, meta in config.DISEASES.items():
        assert "name_bn" in meta and "name_en" in meta and "weights" in meta
        # ওজনের সব ফিচার বৈধ
        for feat in meta["weights"]:
            assert feat in config.FEATURE_NAMES
