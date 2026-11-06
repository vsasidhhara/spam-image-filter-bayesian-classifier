"""
Model Evaluation and Diagnostics Script
DS307 Computer Vision - Project 12: Spam Image Filter using Bayesian Classifier

Usage:
    python evaluate.py
"""

import os
import glob
import json
import sys
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.bayesian_classifier import BayesianImageClassifier
from src.evaluator import ModelEvaluator
from src.heatmap_generator import HeatmapGenerator

def main():
    print("=================================================================")
    print("  DS307 Computer Vision - Project 12: Evaluation & Metrics       ")
    print("=================================================================\n")

    test_spam_dir = os.path.join(BASE_DIR, "dataset", "test", "spam")
    test_ham_dir = os.path.join(BASE_DIR, "dataset", "test", "ham")
    models_dir = os.path.join(BASE_DIR, "models")
    results_dir = os.path.join(BASE_DIR, "static", "results")

    model_npz = os.path.join(models_dir, "bayesian_model.npz")
    model_meta = os.path.join(models_dir, "model_metadata.json")

    if not os.path.exists(model_npz):
        print("Model file not found. Please run 'python train.py' first.")
        return

    classifier = BayesianImageClassifier()
    classifier.load_model(model_npz, model_meta)

    evaluator = ModelEvaluator(classifier)
    metrics, detailed_results, roc_data = evaluator.evaluate_dataset(test_spam_dir, test_ham_dir, threshold=0.5)

    print("\n---------------- Performance Metrics ----------------")
    print(f"Total Test Samples:     {metrics['total_samples']}")
    print(f"Spam Test Samples:      {metrics['spam_samples']}")
    print(f"Ham Test Samples:       {metrics['ham_samples']}")
    print(f"Decision Threshold:     {metrics['threshold']}")
    print("-----------------------------------------------------")
    print(f"True Positives (TP):    {metrics['true_positives']}")
    print(f"True Negatives (TN):    {metrics['true_negatives']}")
    print(f"False Positives (FP):   {metrics['false_positives']}")
    print(f"False Negatives (FN):   {metrics['false_negatives']}")
    print("-----------------------------------------------------")
    print(f"Accuracy:               {metrics['accuracy']}%")
    print(f"Precision:              {metrics['precision']}%")
    print(f"Recall (Sensitivity):   {metrics['recall']}%")
    print(f"Specificity:            {metrics['specificity']}%")
    print(f"F1-Score:               {metrics['f1_score']}%")
    print(f"ROC Area Under Curve:   {metrics['auc']}")
    print("-----------------------------------------------------\n")

    # Save metrics JSON
    metrics_path = os.path.join(results_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

    # Generate Confusion Matrix plot
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    evaluator.plot_confusion_matrix(metrics, cm_path)
    print(f"Confusion Matrix plot saved to {cm_path}")

    # Generate ROC Curve plot
    roc_path = os.path.join(results_dir, "roc_curve.png")
    evaluator.plot_roc_curve(roc_data, metrics["auc"], roc_path)
    print(f"ROC Curve plot saved to {roc_path}")

    # Generate sample diagnostic heatmaps
    heatmap_gen = HeatmapGenerator(classifier)
    sample_spam_imgs = sorted(glob.glob(os.path.join(test_spam_dir, "*.png")))
    sample_ham_imgs = sorted(glob.glob(os.path.join(test_ham_dir, "*.png")))

    if sample_spam_imgs:
        spam_diag_path = os.path.join(results_dir, "sample_spam_diagnostic.png")
        sample_spam_bgr = cv2.imread(sample_spam_imgs[0])
        heatmap_gen.save_diagnostic_report_image(sample_spam_bgr, spam_diag_path)
        print(f"Sample Spam Diagnostic saved to {spam_diag_path}")

    if sample_ham_imgs:
        ham_diag_path = os.path.join(results_dir, "sample_ham_diagnostic.png")
        sample_ham_bgr = cv2.imread(sample_ham_imgs[0])
        heatmap_gen.save_diagnostic_report_image(sample_ham_bgr, ham_diag_path)
        print(f"Sample Ham Diagnostic saved to {ham_diag_path}")

    print("\nEvaluation successfully finished!")

if __name__ == "__main__":
    main()
