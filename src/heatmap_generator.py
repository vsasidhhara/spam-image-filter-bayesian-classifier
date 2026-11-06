"""
Heatmap and Visual Diagnostics Generator
DS307 Computer Vision - Project 12

Generates visual spam evidence heatmaps and comparative diagnostic plots.
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server-safe rendering
import matplotlib.pyplot as plt


class HeatmapGenerator:
    def __init__(self, classifier):
        self.classifier = classifier

    def generate_heatmap_and_overlay(self, bgr_image, custom_prior_spam=None, alpha_blend=0.45,
                                     filter_type="none", filter_strength=1.0):
        """
        Generates:
        1. Posterior probability map (float 0-1)
        2. Colorized heatmap (BGR uint8)
        3. Blended overlay of original image and heatmap (BGR uint8)
        """
        active_img = bgr_image
        if filter_type != "none":
            active_img = self.classifier.extractor.apply_preprocessing_filter(bgr_image, filter_type, filter_strength)

        post_map = self.classifier.compute_pixel_likelihood_map(
            active_img, custom_prior_spam, filter_type="none"
        )
        
        # Scale to 0-255 uint8 for colormapping
        heat_uint8 = np.clip(post_map * 255.0, 0, 255).astype(np.uint8)
        # Apply Jet colormap (Blue = Low spam prob, Red = High spam prob)
        colored_heat = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        
        # Clamp alpha_blend
        alpha_blend = max(0.0, min(1.0, float(alpha_blend)))
        # Blend original with heatmap
        overlay = cv2.addWeighted(active_img, 1.0 - alpha_blend, colored_heat, alpha_blend, 0)
        
        return post_map, colored_heat, overlay

    def save_diagnostic_report_image(self, bgr_image, out_path, custom_prior_spam=None,
                                    strength_multiplier=1.0, feature_balance=0.5,
                                    filter_type="none", filter_strength=1.0):
        """
        Generates a 4-panel diagnostic figure:
        [1] Original Image
        [2] Spam Likelihood Heatmap
        [3] Blended Heatmap Overlay
        [4] Color Distribution Analysis vs Spam/Ham Classes
        """
        active_img = bgr_image
        if filter_type != "none":
            active_img = self.classifier.extractor.apply_preprocessing_filter(bgr_image, filter_type, filter_strength)

        post_map, colored_heat, overlay = self.generate_heatmap_and_overlay(
            active_img, custom_prior_spam=custom_prior_spam
        )
        pred_res = self.classifier.predict(
            active_img,
            custom_prior_spam=custom_prior_spam,
            strength_multiplier=strength_multiplier,
            feature_balance=feature_balance
        )

        rgb_orig = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb_heat = cv2.cvtColor(colored_heat, cv2.COLOR_BGR2RGB)
        rgb_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        plt.subplots_adjust(hspace=0.28, wspace=0.2)

        # Panel 1: Original Image
        axes[0, 0].imshow(rgb_orig)
        axes[0, 0].set_title("Input Image", fontsize=13, fontweight="bold")
        axes[0, 0].axis("off")

        # Panel 2: Likelihood Heatmap
        im2 = axes[0, 1].imshow(post_map, cmap="jet", vmin=0.0, vmax=1.0)
        axes[0, 1].set_title("Bayesian Likelihood Map P(Spam | Pixel)", fontsize=13, fontweight="bold")
        axes[0, 1].axis("off")
        fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04, label="P(Spam)")

        # Panel 3: Blended Overlay
        axes[1, 0].imshow(rgb_overlay)
        status_color = "red" if pred_res["prediction"] == "SPAM" else "green"
        axes[1, 0].set_title(
            f"Prediction: {pred_res['prediction']} ({pred_res['confidence']:.1f}%)\n"
            f"Log-Odds: {pred_res['log_odds']:.2f} | Prior: {pred_res['prior_spam_used']:.2f}",
            fontsize=12, fontweight="bold", color=status_color
        )
        axes[1, 0].axis("off")

        # Panel 4: Edge Gradient & Feature Comparison
        test_edge = self.classifier.extractor.compute_edge_histogram(bgr_image)
        norm_test_edge = test_edge / (np.sum(test_edge) + 1e-12)
        norm_spam_edge = self.classifier.p_edge_spam
        norm_ham_edge = self.classifier.p_edge_ham
        bins = np.arange(len(norm_test_edge))

        axes[1, 1].plot(bins, norm_test_edge, "k-o", label="Test Image Edge Dist", linewidth=2)
        axes[1, 1].plot(bins, norm_spam_edge, "r--", label="Trained Spam Class Hist", alpha=0.8)
        axes[1, 1].plot(bins, norm_ham_edge, "g--", label="Trained Ham Class Hist", alpha=0.8)
        axes[1, 1].set_title("Edge Gradient Magnitude Histogram Comparison", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("Gradient Magnitude Bin")
        axes[1, 1].set_ylabel("Normalized Probability")
        axes[1, 1].legend(loc="upper right", fontsize=9)
        axes[1, 1].grid(True, linestyle=":", alpha=0.6)

        plt.suptitle(
            f"DS307 Computer Vision - Project 12: Spam Image Bayesian Diagnosis\n"
            f"Verdict: {pred_res['prediction']} (P(Spam) = {pred_res['spam_probability']:.4f})",
            fontsize=15, fontweight="bold"
        )
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()
        return out_path
