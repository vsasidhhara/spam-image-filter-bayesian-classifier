"""
Flask Web Application for Spam Image Filtering
DS307 Computer Vision - Project 12

Provides interactive web interface for:
- Single image upload & Bayesian classification
- Interactive prior probability and threshold adjustment
- Pixel-level Bayesian Spam Likelihood Heatmap visualization
- Model Performance Dashboard (Confusion Matrix, ROC Curve)
- Batch test set evaluation
"""

import os
import time
import json
import uuid
import glob
from flask import Flask, render_template, request, redirect, url_for, jsonify
import cv2
import numpy as np

from src.bayesian_classifier import BayesianImageClassifier
from src.heatmap_generator import HeatmapGenerator
from src.evaluator import ModelEvaluator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULTS_FOLDER = os.path.join(BASE_DIR, "static", "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max

# Initialize and load model
classifier = BayesianImageClassifier()
model_npz = os.path.join(MODELS_DIR, "bayesian_model.npz")
model_meta = os.path.join(MODELS_DIR, "model_metadata.json")

if os.path.exists(model_npz) and os.path.exists(model_meta):
    classifier.load_model(model_npz, model_meta)
    print("Bayesian Classifier model loaded into Flask app.")
else:
    print("Warning: Model files not found. Train the model using train.py first.")

heatmap_gen = HeatmapGenerator(classifier)
evaluator = ModelEvaluator(classifier)


@app.route("/")
def index():
    # Provide a few sample images for instant testing
    test_spam_samples = [
        f"/static/test_samples/spam/{os.path.basename(p)}"
        for p in glob.glob(os.path.join(BASE_DIR, "dataset", "test", "spam", "*.png"))[:4]
    ]
    test_ham_samples = [
        f"/static/test_samples/ham/{os.path.basename(p)}"
        for p in glob.glob(os.path.join(BASE_DIR, "dataset", "test", "ham", "*.png"))[:4]
    ]
    return render_template(
        "index.html",
        is_trained=classifier.is_trained,
        test_spam_samples=test_spam_samples,
        test_ham_samples=test_ham_samples
    )


@app.route("/classify", methods=["POST"])
def classify():
    if not classifier.is_trained:
        return jsonify({"error": "Classifier is not trained yet."}), 400

    prior_spam = float(request.form.get("prior_spam", 0.5))
    threshold = float(request.form.get("threshold", 0.5))
    strength_multiplier = float(request.form.get("strength_multiplier", 1.0))
    feature_balance = float(request.form.get("feature_balance", 0.5))
    filter_type = request.form.get("filter_type", "none")
    filter_strength = float(request.form.get("filter_strength", 1.0))
    heatmap_alpha = float(request.form.get("heatmap_alpha", 0.45))

    file = request.files.get("image")
    sample_path = request.form.get("sample_path")
    cached_filename = request.form.get("cached_filename")

    image_bgr = None
    save_filename = None

    if cached_filename:
        # Re-use existing uploaded/selected image for instantaneous slider recalculation
        cached_path = os.path.join(app.config["UPLOAD_FOLDER"], cached_filename)
        if os.path.exists(cached_path):
            save_filename = cached_filename
            image_bgr = cv2.imread(cached_path)

    if image_bgr is None:
        save_filename = f"{uuid.uuid4().hex}.png"
        orig_abs_path = os.path.join(app.config["UPLOAD_FOLDER"], save_filename)

        if file and file.filename != "":
            file.save(orig_abs_path)
            image_bgr = cv2.imread(orig_abs_path)
        elif sample_path:
            clean_sample = sample_path.lstrip("/").replace("static/", "")
            actual_path = os.path.join(BASE_DIR, "dataset", clean_sample.replace("test_samples/", "test/"))
            if os.path.exists(actual_path):
                image_bgr = cv2.imread(actual_path)
                cv2.imwrite(orig_abs_path, image_bgr)
            else:
                return jsonify({"error": "Sample image not found."}), 404
        else:
            return jsonify({"error": "No image provided."}), 400

    if image_bgr is None:
        return jsonify({"error": "Invalid image file format."}), 400

    start_time = time.time()
    
    # 1. Bayesian Prediction with Dynamic Strength & Additional Filters
    pred_result = classifier.predict(
        image_bgr,
        threshold=threshold,
        custom_prior_spam=prior_spam,
        strength_multiplier=strength_multiplier,
        feature_balance=feature_balance,
        filter_type=filter_type,
        filter_strength=filter_strength
    )

    # 2. Likelihood Heatmap and Blended Overlay
    post_map, colored_heat, overlay = heatmap_gen.generate_heatmap_and_overlay(
        image_bgr,
        custom_prior_spam=prior_spam,
        alpha_blend=heatmap_alpha,
        filter_type=filter_type,
        filter_strength=filter_strength
    )

    heat_filename = f"heat_{save_filename}"
    overlay_filename = f"overlay_{save_filename}"
    heat_abs_path = os.path.join(app.config["UPLOAD_FOLDER"], heat_filename)
    overlay_abs_path = os.path.join(app.config["UPLOAD_FOLDER"], overlay_filename)

    cv2.imwrite(heat_abs_path, colored_heat)
    cv2.imwrite(overlay_abs_path, overlay)

    # 3. Diagnostic 4-panel image
    diag_filename = f"diag_{save_filename}"
    diag_abs_path = os.path.join(app.config["UPLOAD_FOLDER"], diag_filename)
    heatmap_gen.save_diagnostic_report_image(
        image_bgr,
        diag_abs_path,
        custom_prior_spam=prior_spam,
        strength_multiplier=strength_multiplier,
        feature_balance=feature_balance,
        filter_type=filter_type,
        filter_strength=filter_strength
    )

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return jsonify({
        "status": "success",
        "cached_filename": save_filename,
        "prediction": pred_result["prediction"],
        "spam_probability": round(pred_result["spam_probability"] * 100, 2),
        "ham_probability": round(pred_result["ham_probability"] * 100, 2),
        "confidence": round(pred_result["confidence"], 2),
        "log_odds": round(pred_result["log_odds"], 3),
        "color_llr": round(pred_result["color_llr"], 3),
        "edge_llr": round(pred_result["edge_llr"], 3),
        "lap_llr": round(pred_result.get("lap_llr", 0.0), 3),
        "strength_used": pred_result.get("strength_used", strength_multiplier),
        "balance_used": pred_result.get("balance_used", feature_balance),
        "filter_type_used": filter_type,
        "filter_strength_used": filter_strength,
        "prior_used": pred_result["prior_spam_used"],
        "threshold_used": pred_result["threshold_used"],
        "elapsed_ms": elapsed_ms,
        "image_orig_url": url_for("static", filename=f"uploads/{save_filename}"),
        "image_heat_url": url_for("static", filename=f"uploads/{heat_filename}"),
        "image_overlay_url": url_for("static", filename=f"uploads/{overlay_filename}"),
        "image_diag_url": url_for("static", filename=f"uploads/{diag_filename}")
    })


@app.route("/dashboard")
def dashboard():
    metrics_path = os.path.join(RESULTS_FOLDER, "evaluation_metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    model_meta_data = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            model_meta_data = json.load(f)

    return render_template(
        "dashboard.html",
        metrics=metrics,
        meta=model_meta_data
    )


@app.route("/batch")
def batch():
    test_spam_dir = os.path.join(BASE_DIR, "dataset", "test", "spam")
    test_ham_dir = os.path.join(BASE_DIR, "dataset", "test", "ham")
    
    threshold = float(request.args.get("threshold", 0.5))
    metrics, detailed_results, _ = evaluator.evaluate_dataset(test_spam_dir, test_ham_dir, threshold=threshold)

    return render_template(
        "batch.html",
        metrics=metrics,
        results=detailed_results,
        threshold=threshold
    )


@app.route("/about")
def about():
    return render_template("about.html")


# Route to serve test samples directly
@app.route("/static/test_samples/<category>/<filename>")
def serve_test_samples(category, filename):
    file_path = os.path.join(BASE_DIR, "dataset", "test", category, filename)
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            from flask import Response
            return Response(f.read(), mimetype="image/png")
    return "File not found", 404


if __name__ == "__main__":
    print("Starting Flask Application on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
