"""MediPredict — Flask ওয়েব অ্যাপ্লিকেশন (মাল্টি-ডিজিজ)।

ব্যবহার:
    python -m app.app
    তারপর ব্রাউজারে http://127.0.0.1:5000 খুলুন।
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

# প্যাকেজ ইম্পোর্ট নিশ্চিত করতে প্রজেক্ট রুট পাথে যোগ করা হয়
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import analytics  # noqa: E402
from src import assistant  # noqa: E402
from src import config  # noqa: E402
from src import history  # noqa: E402
from src.predict import get_predictor  # noqa: E402

app = Flask(__name__)

# ---------------------------------------------------------------------------
# নিরাপত্তা (ঐচ্ছিক, পরিবেশ ভেরিয়েবল দিয়ে সক্রিয়)
#   MEDIPREDICT_API_KEY    — সেট থাকলে /api/* এ X-API-Key হেডার আবশ্যক
#   MEDIPREDICT_RATE_LIMIT — সেট থাকলে প্রতি IP প্রতি মিনিটে সর্বোচ্চ অনুরোধ
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("MEDIPREDICT_API_KEY")
RATE_LIMIT = int(os.environ.get("MEDIPREDICT_RATE_LIMIT", "0"))
_hits: dict[str, deque] = defaultdict(deque)


@app.before_request
def _guard():
    # শুধু /api/* সুরক্ষিত; healthcheck উন্মুক্ত রাখা হয়
    if not request.path.startswith("/api/") or request.path == "/api/health":
        return None
    if API_KEY and request.headers.get("X-API-Key") != API_KEY:
        return jsonify({"error": "অননুমোদিত — বৈধ X-API-Key দরকার।"}), 401
    if RATE_LIMIT > 0:
        now = time.time()
        dq = _hits[request.remote_addr or "?"]
        while dq and dq[0] < now - 60:
            dq.popleft()
        if len(dq) >= RATE_LIMIT:
            return jsonify({"error": "অনেক বেশি অনুরোধ — কিছুক্ষণ পরে চেষ্টা করুন।"}), 429
        dq.append(now)
    return None


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
        history.save_prediction(result, features, patient_id=str(data.get("patient_id", "")))
    except Exception:
        pass

    return jsonify(result)


@app.route("/api/patients")
def api_patients():
    """রোগীভিত্তিক সারাংশের তালিকা।"""
    return jsonify(history.list_patients())


@app.route("/api/patient/<patient_id>/trend")
def api_patient_trend(patient_id):
    """একজন রোগীর সময়ক্রমিক ঝুঁকি-ট্রেন্ড।"""
    disease = request.args.get("disease")
    return jsonify(history.get_patient_trend(patient_id, disease=disease))


@app.route("/api/analytics/summary")
def api_analytics_summary():
    """সব প্রেডিকশনের সমষ্টিগত সারাংশ।"""
    return jsonify(analytics.summary())


@app.route("/api/analytics/drift")
def api_analytics_drift():
    """ইনপুট ডেটা-ড্রিফট (PSI)।"""
    return jsonify(analytics.drift())


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


@app.route("/api/assistant/status")
def assistant_status():
    """AI সহকারী ব্যবহারযোগ্য কিনা।"""
    return jsonify({"available": assistant.is_available()})


@app.route("/api/assistant", methods=["POST"])
def api_assistant():
    """AI স্বাস্থ্য সহকারীকে প্রশ্ন করা।"""
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "প্রশ্ন খালি হতে পারে না।"}), 400
    try:
        answer = assistant.ask(question, context=data.get("context"))
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001 — যেকোনো API ত্রুটি বন্ধুত্বপূর্ণভাবে রিটার্ন
        return jsonify({"error": f"সহকারী ত্রুটি: {exc}"}), 502
    return jsonify({"answer": answer})


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "version": "4.0.0"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
