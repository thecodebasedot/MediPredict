"""SHAP ভিজ্যুয়ালাইজেশন — ইন্ডাস্ট্রি-স্ট্যান্ডার্ড ব্যাখ্যাযোগ্যতা।

`shap` লাইব্রেরি দিয়ে দুই ধরনের প্লট তৈরি করে:
1. Beeswarm summary — পুরো ডেটাসেটে কোন ফিচার সামগ্রিকভাবে কতটা প্রভাব ফেলে।
2. Waterfall — একটি নির্দিষ্ট রোগীর পূর্বাভাসে প্রতিটি ফিচারের অবদান।

প্লটগুলো docs/ ফোল্ডারে সেভ হয়।

ব্যবহার:
    python -m src.shap_explain            # ডিফল্ট রোগ (diabetes)
    python -m src.shap_explain heart      # নির্দিষ্ট রোগ
"""

from __future__ import annotations

import sys

from . import config
from .data import load_dataset


def generate(disease: str = config.DEFAULT_DISEASE, sample_size: int = 500) -> dict:
    """নির্দিষ্ট রোগের জন্য SHAP beeswarm ও waterfall প্লট তৈরি করে।"""
    if disease not in config.DISEASES:
        raise ValueError(f"অজানা রোগ: {disease}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("shap ও matplotlib দরকার: pip install shap matplotlib") from exc

    model_file = config.model_path(disease)
    if not model_file.exists():
        raise FileNotFoundError(f"মডেল নেই: {model_file} — প্রথমে: python -m src.train")

    model = XGBClassifier()
    model.load_model(model_file)

    df = load_dataset()
    X = df[config.FEATURE_NAMES].sample(
        min(sample_size, len(df)), random_state=config.RANDOM_SEED
    ).reset_index(drop=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    docs = config.ROOT_DIR / "docs"
    docs.mkdir(exist_ok=True)
    name_en = config.DISEASES[disease]["name_en"]
    outputs = {}

    # 1) Beeswarm summary
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False, max_display=10)
    plt.title(f"SHAP Summary — {name_en}")
    plt.tight_layout()
    beeswarm_path = docs / f"shap_{disease}.png"
    plt.savefig(beeswarm_path, dpi=120, bbox_inches="tight")
    plt.close()
    outputs["beeswarm"] = str(beeswarm_path)

    # 2) সর্বোচ্চ-ঝুঁকির রোগীর জন্য waterfall
    proba = model.predict_proba(X)[:, 1]
    top_idx = int(proba.argmax())
    plt.figure()
    shap.plots.waterfall(shap_values[top_idx], show=False, max_display=10)
    plt.title(f"SHAP Waterfall — {name_en} (highest-risk sample)")
    plt.tight_layout()
    waterfall_path = docs / f"shap_waterfall_{disease}.png"
    plt.savefig(waterfall_path, dpi=120, bbox_inches="tight")
    plt.close()
    outputs["waterfall"] = str(waterfall_path)

    print(f"📊 SHAP প্লট সেভ হয়েছে ({name_en}):")
    for kind, path in outputs.items():
        print(f"   {kind:9s}: {path}")
    return outputs


if __name__ == "__main__":
    disease = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_DISEASE
    generate(disease)
