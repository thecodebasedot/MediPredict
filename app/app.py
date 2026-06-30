"""MediPredict — Flask ওয়েব অ্যাপ্লিকেশন।

ব্যবহার:
    python -m app.app
    তারপর ব্রাউজারে http://127.0.0.1:5000 খুলুন।
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

# প্যাকেজ ইম্পোর্ট নিশ্চিত করতে প্রজেক্ট রুট পাথে যোগ করা হয়
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.predict import get_predictor  # noqa: E402

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", features=config.FEATURES)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    try:
        features = {}
        for f in config.FEATURES:
            value = data.get(f["name"], f["default"])
            features[f["name"]] = float(value)
    except (TypeError, ValueError):
        return jsonify({"error": "অবৈধ ইনপুট। সব মান সংখ্যা হতে হবে।"}), 400

    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    result = predictor.predict(features, explain=True)
    return jsonify(result)


@app.route("/api/model-info")
def model_info():
    """মডেলের মেট্রিক ও ফিচার গুরুত্ব রিটার্ন করে।"""
    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    return jsonify(predictor.metadata)


@app.route("/api/batch", methods=["POST"])
def api_batch():
    """CSV আপলোড নিয়ে একাধিক রোগীর পূর্বাভাস রিটার্ন করে।"""
    if "file" not in request.files:
        return jsonify({"error": "কোনো ফাইল আপলোড করা হয়নি।"}), 400
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
        out = predictor.predict_batch(df)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "count": int(out.shape[0]),
            "results": out.to_dict(orient="records"),
        }
    )


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "version": "1.0.0"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
