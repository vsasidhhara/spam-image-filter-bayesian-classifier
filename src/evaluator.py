"""
Model Evaluation and Metrics Module
DS307 Computer Vision - Project 12

Computes:
- Confusion Matrix (TP, FP, TN, FN)
- Accuracy, Precision, Recall, Specificity, F1-Score
- ROC Curve & AUC (Area Under Curve)
- Precision-Recall Curve
- Class Histogram Visualizations
"""

import os
import glob
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ModelEvaluator:
    def __init__(self, classifier):
        self.classifier = classifier

    def evaluate_dataset(self, test_spam_dir, test_ham_dir, threshold=0.5):
        """
        Evaluates classifier on test directories and returns performance metrics.
        """
        spam_paths = sorted(glob.glob(os.path.join(test_spam_dir, "*.png")) +
                            glob.glob(os.path.join(test_spam_dir, "*.jpg")))
        ham_paths = sorted(glob.glob(os.path.join(test_ham_dir, "*.png")) +
                           glob.glob(os.path.join(test_ham_dir, "*.jpg")))

        y_true = []
        y_scores = []
        detailed_results = []

        # Ground truth: 1 for Spam, 0 for Ham
        for p in spam_paths:
            img = cv2.imread(p)
            if img is None:
                continue
            res = self.classifier.predict(img, threshold=threshold)
            y_true.append(1)
            y_scores.append(res["spam_probability"])
            detailed_results.append({
                "path": p,
                "filename": os.path.basename(p),
                "true_label": "SPAM",
                "pred_label": res["prediction"],
                "prob_spam": res["spam_probability"],
                "correct": res["prediction"] == "SPAM"
            })

        for p in ham_paths:
            img = cv2.imread(p)
            if img is None:
                continue
            res = self.classifier.predict(img, threshold=threshold)
            y_true.append(0)
            y_scores.append(res["spam_probability"])
            detailed_results.append({
                "path": p,
                "filename": os.path.basename(p),
                "true_label": "HAM",
                "pred_label": res["prediction"],
                "prob_spam": res["spam_probability"],
                "correct": res["prediction"] == "HAM"
            })

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        y_pred = (y_scores >= threshold).astype(int)

        # Confusion Matrix components
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        total = len(y_true)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Compute ROC Curve and AUC
        thresholds_roc = np.linspace(0.0, 1.0, 101)
        tpr_list = []
        fpr_list = []
        
        # Sort thresholds descending for trapezoidal AUC integration
        for th in sorted(thresholds_roc, reverse=True):
            pred_th = (y_scores >= th).astype(int)
            th_tp = np.sum((y_true == 1) & (pred_th == 1))
            th_fn = np.sum((y_true == 1) & (pred_th == 0))
            th_fp = np.sum((y_true == 0) & (pred_th == 1))
            th_tn = np.sum((y_true == 0) & (pred_th == 0))
            
            tpr = th_tp / (th_tp + th_fn) if (th_tp + th_fn) > 0 else 0.0
            fpr = th_fp / (th_fp + th_tn) if (th_fp + th_tn) > 0 else 0.0
            tpr_list.append(tpr)
            fpr_list.append(fpr)

        # Trapezoidal rule for AUC
        auc_score = 0.0
        for i in range(1, len(fpr_list)):
            auc_score += 0.5 * (tpr_list[i] + tpr_list[i - 1]) * (fpr_list[i] - fpr_list[i - 1])
        auc_score = abs(auc_score)

        metrics = {
            "total_samples": total,
            "spam_samples": len(spam_paths),
            "ham_samples": len(ham_paths),
            "threshold": threshold,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "specificity": round(specificity * 100, 2),
            "f1_score": round(f1_score * 100, 2),
            "auc": round(auc_score, 4)
        }

        roc_data = {
            "fpr": [float(x) for x in fpr_list],
            "tpr": [float(x) for x in tpr_list],
            "thresholds": [float(x) for x in sorted(thresholds_roc, reverse=True)]
        }

        return metrics, detailed_results, roc_data

    def plot_confusion_matrix(self, metrics, out_path):
        """Plots and saves the Confusion Matrix heatmap."""
        cm = np.array([
            [metrics["true_positives"], metrics["false_negatives"]],
            [metrics["false_positives"], metrics["true_negatives"]]
        ])
        
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
        fig.colorbar(cax)

        labels = [["TP", "FN"], ["FP", "TN"]]
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{labels[i][j]}\n{cm[i, j]}",
                    ha="center", va="center", color="black",
                    fontweight="bold", fontsize=14
                )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Spam (Pos)", "Ham (Neg)"], fontsize=11, fontweight="bold")
        ax.set_yticklabels(["Actual Spam", "Actual Ham"], fontsize=11, fontweight="bold")
        ax.set_title("Confusion Matrix - Bayesian Spam Classifier", fontsize=13, fontweight="bold", pad=20)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=180)
        plt.close()
        return out_path

    def plot_roc_curve(self, roc_data, auc_score, out_path):
        """Plots and saves the Receiver Operating Characteristic (ROC) curve."""
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(roc_data["fpr"], roc_data["tpr"], color="#d62728", lw=2.5,
                label=f"Bayesian Classifier (AUC = {auc_score:.4f})")
        ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
        
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, fontweight="bold")
        ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.7)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=180)
        plt.close()
        return out_path

    def plot_class_histograms(self, out_path):
        """Visualizes the learned Spam and Ham class histograms."""
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        plt.subplots_adjust(hspace=0.3, wspace=0.25)

        # 1. Edge gradient distribution comparison
        bins = np.arange(len(self.classifier.p_edge_spam))
        axes[0, 0].bar(bins - 0.2, self.classifier.p_edge_spam, width=0.4, color="crimson", alpha=0.85, label="Spam Class")
        axes[0, 0].bar(bins + 0.2, self.classifier.p_edge_ham, width=0.4, color="forestgreen", alpha=0.85, label="Ham Class")
        axes[0, 0].set_title("Edge Gradient Magnitude Likelihood P(Edge | Class)", fontsize=12, fontweight="bold")
        axes[0, 0].set_xlabel("Edge Gradient Bin")
        axes[0, 0].set_ylabel("Probability")
        axes[0, 0].legend()
        axes[0, 0].grid(True, linestyle=":", alpha=0.5)

        # 2. Hue distribution comparison (collapsed from 2D HS)
        hue_spam = np.sum(self.classifier.p_hs_spam, axis=1)
        hue_ham = np.sum(self.classifier.p_hs_ham, axis=1)
        h_bins = np.arange(len(hue_spam))
        axes[0, 1].bar(h_bins - 0.2, hue_spam, width=0.4, color="crimson", alpha=0.85, label="Spam Hue")
        axes[0, 1].bar(h_bins + 0.2, hue_ham, width=0.4, color="forestgreen", alpha=0.85, label="Ham Hue")
        axes[0, 1].set_title("Hue Chromaticity Likelihood P(Hue | Class)", fontsize=12, fontweight="bold")
        axes[0, 1].set_xlabel("Hue Bin (0-180°)")
        axes[0, 1].set_ylabel("Marginal Probability")
        axes[0, 1].legend()
        axes[0, 1].grid(True, linestyle=":", alpha=0.5)

        # 3. Saturation distribution comparison (collapsed from 2D HS)
        sat_spam = np.sum(self.classifier.p_hs_spam, axis=0)
        sat_ham = np.sum(self.classifier.p_hs_ham, axis=0)
        s_bins = np.arange(len(sat_spam))
        axes[1, 0].bar(s_bins - 0.2, sat_spam, width=0.4, color="crimson", alpha=0.85, label="Spam Saturation")
        axes[1, 0].bar(s_bins + 0.2, sat_ham, width=0.4, color="forestgreen", alpha=0.85, label="Ham Saturation")
        axes[1, 0].set_title("Color Saturation Likelihood P(Saturation | Class)", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Saturation Bin (0-255)")
        axes[1, 0].set_ylabel("Marginal Probability")
        axes[1, 0].legend()
        axes[1, 0].grid(True, linestyle=":", alpha=0.5)

        # 4. 2D Hue-Saturation Likelihood Ratio Map: log [P(H,S | Spam) / P(H,S | Ham)]
        ratio = np.log(self.classifier.p_hs_spam + 1e-12) - np.log(self.classifier.p_hs_ham + 1e-12)
        im = axes[1, 1].imshow(ratio, cmap="coolwarm", aspect="auto")
        axes[1, 1].set_title("2D Color Log-Likelihood Ratio Map (Red=Spam, Blue=Ham)", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("Saturation Bin")
        axes[1, 1].set_ylabel("Hue Bin")
        fig.colorbar(im, ax=axes[1, 1], label="Log-Likelihood Ratio")

        plt.suptitle("DS307 Computer Vision - Learned Class Histograms & Discrimination Maps", fontsize=14, fontweight="bold")
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=180)
        plt.close()
        return out_path
