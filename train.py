"""
Model Training Script
DS307 Computer Vision - Project 12: Spam Image Filter using Bayesian Classifier

Usage:
    python train.py
"""

import os
import glob
import sys

# Ensure src is importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.bayesian_classifier import BayesianImageClassifier
from src.evaluator import ModelEvaluator

def main():
    print("=================================================================")
    print("  DS307 Computer Vision - Project 12: Bayesian Image Classifier  ")
    print("  Training Class Histograms for Spam Image Filtering            ")
    print("=================================================================\n")

    train_spam_dir = os.path.join(BASE_DIR, "dataset", "train", "spam")
    train_ham_dir = os.path.join(BASE_DIR, "dataset", "train", "ham")
    models_dir = os.path.join(BASE_DIR, "models")
    results_dir = os.path.join(BASE_DIR, "static", "results")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    spam_paths = sorted(glob.glob(os.path.join(train_spam_dir, "*.png")) +
                        glob.glob(os.path.join(train_spam_dir, "*.jpg")))
    ham_paths = sorted(glob.glob(os.path.join(train_ham_dir, "*.png")) +
                       glob.glob(os.path.join(train_ham_dir, "*.jpg")))

    if not spam_paths or not ham_paths:
        print("Error: Training data not found. Please run 'python dataset/generate_dataset.py' first.")
        return

    print(f"Found {len(spam_paths)} Spam training images.")
    print(f"Found {len(ham_paths)} Ham training images.")

    # Initialize Bayesian Classifier with Laplace smoothing
    classifier = BayesianImageClassifier(
        rgb_bins=16,
        h_bins=18,
        s_bins=16,
        edge_bins=16,
        alpha=1.0,
        prior_spam=0.5,
        edge_weight=0.7,
        scale_factor=7.5
    )

    # Train classifier
    classifier.train(spam_paths, ham_paths)

    # Save model
    model_npz = os.path.join(models_dir, "bayesian_model.npz")
    model_meta = os.path.join(models_dir, "model_metadata.json")
    classifier.save_model(model_npz, model_meta)

    # Visualize and save learned class distributions
    evaluator = ModelEvaluator(classifier)
    hist_plot_path = os.path.join(results_dir, "class_histograms.png")
    evaluator.plot_class_histograms(hist_plot_path)
    print(f"Class histogram comparisons saved to {hist_plot_path}")

    print("\nTraining completed successfully! Model is ready for evaluation and web demo.")

if __name__ == "__main__":
    main()
