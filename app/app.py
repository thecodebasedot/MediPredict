"""MediPredict — Flask ওয়েব অ্যাপ্লিকেশন (মাল্টি-ডিজিজ)।

ব্যবহার:
    python -m app.app
    তারপর ব্রাউজারে http://127.0.0.1:5000 খুলুন।
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

# প্যাকেজ ইম্পোর্ট নিশ্চিত করতে প্রজেক্ট রুট পাথে যোগ করা হয়
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src import history  # noqa: E402
from src.predict import get_predictor  # noqa: E402

app = Flask(__name__)


def _read_features(data: dict) -> dict:
    """ইনপুট ডিকশনারি থেকে ফিচার বের করে (অনুপস্থিত হলে ডিফল্ট)।"""
    features = {}
    for f in config.FEATURES:
        features[f["name"]] = float(data.get(f["name"], f["default"]))
    return features


@app.route("/")
def index():
    return render_template("index.html", features=config.FEATURES, diseases=config.DISEASES)


@app.route("/api/diseases")
def diseases():
    """সমর্থিত রোগের তালিকা।"""
    return jsonify([
        {"key": k, "name_bn": v["name_bn"], "name_en": v["name_en"]}
        for k, v in config.DISEASES.items()
    ])


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    disease = data.get("disease", config.DEFAULT_DISEASE)
    if disease not in config.DISEASES:
        return jsonify({"error": f"অজানা রোগ: {disease}"}), 400
    try:
        features = _read_features(data)
    except (TypeError, ValueError):
        return jsonify({"error": "অবৈধ ইনপুট। সব মান সংখ্যা হতে হবে।"}), 400

    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    result = predictor.predict(features, disease=disease, explain=True)
    result["all_diseases"] = predictor.predict_all(features)

    # হিস্টোরিতে সংরক্ষণ (ব্যর্থ হলে পূর্বাভাস তবু রিটার্ন হয়)
    try:
        history.save_prediction(result, features)
    except Exception:
        pass

    return jsonify(result)


@app.route("/api/history")
def api_history():
    """সর্বশেষ পূর্বাভাসের হিস্টোরি।"""
    try:
        limit = min(int(request.args.get("limit", 20)), 100)
    except ValueError:
        limit = 20
    return jsonify(history.get_history(limit))


@app.route("/api/history/clear", methods=["POST"])
def api_history_clear():
    """সব হিস্টোরি মুছে ফেলে।"""
    deleted = history.clear_history()
    return jsonify({"deleted": deleted})


@app.route("/api/model-info")
def model_info():
    """সব রোগের মডেল মেট্রিক ও ফিচার গুরুত্ব।"""
    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    return jsonify(predictor.metadata)


@app.route("/api/comparison")
def comparison():
    """মডেল তুলনার ফলাফল (যদি তৈরি থাকে)।"""
    if not config.COMPARISON_PATH.exists():
        return jsonify({"error": "তুলনা পাওয়া যায়নি। চালান: python -m src.compare"}), 404
    with open(config.COMPARISON_PATH, encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.route("/api/batch", methods=["POST"])
def api_batch():
    """CSV আপলোড নিয়ে একাধিক রোগীর পূর্বাভাস।"""
    if "file" not in request.files:
        return jsonify({"error": "কোনো ফাইল আপলোড করা হয়নি।"}), 400
    disease = request.form.get("disease", config.DEFAULT_DISEASE)
    if disease not in config.DISEASES:
        return jsonify({"error": f"অজানা রোগ: {disease}"}), 400

    file = request.files["file"]
    try:
        df = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8")))
    except Exception:
        return jsonify({"error": "CSV ফাইল পড়া যায়নি।"}), 400

    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    try:
        out = predictor.predict_batch(df, disease=disease)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({"count": int(out.shape[0]), "results": out.to_dict(orient="records")})


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "version": "3.0.0"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
