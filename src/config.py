"""প্রজেক্ট কনফিগারেশন — পাথ, ফিচার, রোগ এবং মডেল হাইপারপ্যারামিটার।"""

from pathlib import Path

# ---------------------------------------------------------------------------
# ডিরেক্টরি পাথ
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"

DATASET_PATH = DATA_DIR / "medical_data.csv"
METADATA_PATH = MODEL_DIR / "metadata.json"
COMPARISON_PATH = MODEL_DIR / "comparison.json"


def model_path(disease: str) -> Path:
    """নির্দিষ্ট রোগের মডেল ফাইলের পাথ।"""
    return MODEL_DIR / f"xgboost_{disease}.json"


# ---------------------------------------------------------------------------
# ফিচার সংজ্ঞা — নাম, বাংলা ও ইংরেজি লেবেল, রেঞ্জ, ডিফল্ট, ও কেন্দ্রবিন্দু
# ---------------------------------------------------------------------------
FEATURES = [
    {"name": "age",          "label": "বয়স (বছর)",          "label_en": "Age (years)",          "min": 1,   "max": 120, "default": 45,  "center": 50},
    {"name": "glucose",      "label": "রক্তে গ্লুকোজ (mg/dL)", "label_en": "Glucose (mg/dL)",      "min": 50,  "max": 300, "default": 110, "center": 110},
    {"name": "blood_pressure","label": "রক্তচাপ (mm Hg)",    "label_en": "Blood Pressure (mmHg)","min": 40,  "max": 200, "default": 80,  "center": 80},
    {"name": "bmi",          "label": "বিএমআই (kg/m²)",      "label_en": "BMI (kg/m²)",          "min": 10,  "max": 60,  "default": 25,  "center": 26},
    {"name": "cholesterol",  "label": "কোলেস্টেরল (mg/dL)",   "label_en": "Cholesterol (mg/dL)",  "min": 100, "max": 400, "default": 190, "center": 190},
    {"name": "insulin",      "label": "ইনসুলিন (µU/mL)",     "label_en": "Insulin (µU/mL)",      "min": 0,   "max": 300, "default": 85,  "center": 85},
    {"name": "heart_rate",   "label": "হৃদস্পন্দন (bpm)",    "label_en": "Heart Rate (bpm)",     "min": 40,  "max": 180, "default": 75,  "center": 75},
    {"name": "smoking",      "label": "ধূমপান (০=না, ১=হ্যাঁ)", "label_en": "Smoking (0=No, 1=Yes)", "min": 0, "max": 1,   "default": 0,   "center": 0},
    {"name": "physical_activity", "label": "শারীরিক সক্রিয়তা (০-১০)", "label_en": "Physical Activity (0-10)", "min": 0, "max": 10, "default": 5, "center": 5},
    {"name": "family_history","label": "পারিবারিক ইতিহাস (০=না, ১=হ্যাঁ)", "label_en": "Family History (0=No, 1=Yes)", "min": 0, "max": 1, "default": 0, "center": 0},
]

FEATURE_NAMES = [f["name"] for f in FEATURES]
FEATURE_CENTERS = {f["name"]: f["center"] for f in FEATURES}

# ---------------------------------------------------------------------------
# রোগ সংজ্ঞা — প্রতিটি রোগের জন্য ফিচার-ভিত্তিক ঝুঁকি ওজন (সিনথেটিক ডেটার জন্য)
# ওজন = কেন্দ্রবিন্দু থেকে প্রতি একক বিচ্যুতির ঝুঁকি অবদান।
# ---------------------------------------------------------------------------
DISEASES = {
    "diabetes": {
        "name_bn": "ডায়াবেটিস",
        "name_en": "Diabetes",
        "weights": {
            "glucose": 0.032, "bmi": 0.040, "insulin": 0.012, "age": 0.015,
            "family_history": 0.9, "physical_activity": -0.12, "cholesterol": 0.005,
        },
    },
    "heart": {
        "name_bn": "হৃদরোগ",
        "name_en": "Heart Disease",
        "weights": {
            "blood_pressure": 0.030, "cholesterol": 0.016, "smoking": 0.9, "age": 0.032,
            "heart_rate": 0.016, "bmi": 0.025, "family_history": 0.6, "physical_activity": -0.10,
        },
    },
    "hypertension": {
        "name_bn": "উচ্চ রক্তচাপ",
        "name_en": "Hypertension",
        "weights": {
            "blood_pressure": 0.065, "bmi": 0.030, "age": 0.022, "smoking": 0.5,
            "physical_activity": -0.10, "heart_rate": 0.012, "glucose": 0.005,
        },
    },
}

DISEASE_KEYS = list(DISEASES.keys())
DEFAULT_DISEASE = "diabetes"

# ---------------------------------------------------------------------------
# ঝুঁকি স্তর (বাংলা ও ইংরেজি)
# ---------------------------------------------------------------------------
RISK_LEVELS = [
    {"max": 0.25, "bn": "নিম্ন ঝুঁকি",      "en": "Low Risk"},
    {"max": 0.50, "bn": "মাঝারি ঝুঁকি",     "en": "Moderate Risk"},
    {"max": 0.75, "bn": "উচ্চ ঝুঁকি",       "en": "High Risk"},
    {"max": 1.01, "bn": "অতি উচ্চ ঝুঁকি",   "en": "Very High Risk"},
]

# ---------------------------------------------------------------------------
# XGBoost হাইপারপ্যারামিটার
# ---------------------------------------------------------------------------
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 2,
    "gamma": 0.1,
    "random_state": 42,
}

RANDOM_SEED = 42
TEST_SIZE = 0.2
