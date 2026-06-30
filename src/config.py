"""প্রজেক্ট কনফিগারেশন — পাথ, ফিচার এবং মডেল হাইপারপ্যারামিটার।"""

from pathlib import Path

# ---------------------------------------------------------------------------
# ডিরেক্টরি পাথ
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"

DATASET_PATH = DATA_DIR / "medical_data.csv"
MODEL_PATH = MODEL_DIR / "xgboost_model.json"
METADATA_PATH = MODEL_DIR / "metadata.json"

# ---------------------------------------------------------------------------
# ফিচার সংজ্ঞা
# প্রতিটি ফিচারের নাম, বাংলা লেবেল, এবং বৈধ রেঞ্জ (UI ভ্যালিডেশনের জন্য)
# ---------------------------------------------------------------------------
FEATURES = [
    {"name": "age",          "label": "বয়স (বছর)",                 "min": 1,   "max": 120, "default": 45},
    {"name": "glucose",      "label": "রক্তে গ্লুকোজ (mg/dL)",       "min": 50,  "max": 300, "default": 110},
    {"name": "blood_pressure","label": "রক্তচাপ (mm Hg)",           "min": 40,  "max": 200, "default": 80},
    {"name": "bmi",          "label": "বিএমআই (kg/m²)",             "min": 10,  "max": 60,  "default": 25},
    {"name": "cholesterol",  "label": "কোলেস্টেরল (mg/dL)",          "min": 100, "max": 400, "default": 190},
    {"name": "insulin",      "label": "ইনসুলিন (µU/mL)",            "min": 0,   "max": 300, "default": 85},
    {"name": "heart_rate",   "label": "হৃদস্পন্দন (bpm)",           "min": 40,  "max": 180, "default": 75},
    {"name": "smoking",      "label": "ধূমপান (০=না, ১=হ্যাঁ)",      "min": 0,   "max": 1,   "default": 0},
    {"name": "physical_activity", "label": "শারীরিক সক্রিয়তা (০-১০)", "min": 0,  "max": 10,  "default": 5},
    {"name": "family_history","label": "পারিবারিক ইতিহাস (০=না, ১=হ্যাঁ)", "min": 0, "max": 1, "default": 0},
]

FEATURE_NAMES = [f["name"] for f in FEATURES]
TARGET_NAME = "disease"

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
