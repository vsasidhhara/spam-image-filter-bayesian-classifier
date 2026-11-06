"""
Bayesian Classifier from Class Histograms
DS307 Computer Vision - Project 12

Implements Bayes' Decision Rule using Empirical Class-Conditional Histograms:
P(Spam | Image) = [P(Image | Spam) * P(Spam)] / P(Image)
Using Log-Odds formulation with Laplace smoothing.
"""

import os
import json
import numpy as np
import cv2
from .histogram_extractor import HistogramExtractor


class BayesianImageClassifier:
    def __init__(self, rgb_bins=16, h_bins=18, s_bins=16, edge_bins=16, 
                 alpha=1.0, prior_spam=0.5, edge_weight=0.6, scale_factor=6.0):
        """
        Initialize the Bayesian Image Classifier.
        
        :param rgb_bins: Bins per color channel for 3D RGB histogram
        :param h_bins: Bins for Hue
        :param s_bins: Bins for Saturation
        :param edge_bins: Bins for Edge Magnitude
        :param alpha: Laplace additive smoothing parameter
        :param prior_spam: Prior probability P(Spam)
        :param edge_weight: Weight of edge feature log-likelihood relative to color
        :param scale_factor: Temperature scaling for calibrated posterior probability
        """
        self.extractor = HistogramExtractor(
            rgb_bins=rgb_bins,
            h_bins=h_bins,
            s_bins=s_bins,
            edge_bins=edge_bins,
            alpha=alpha
        )
        self.prior_spam = prior_spam
        self.prior_ham = 1.0 - prior_spam
        self.edge_weight = edge_weight
        self.scale_factor = scale_factor
        
        # Trained class conditional distributions: P(feature | Spam) and P(feature | Ham)
        self.p_rgb_spam = None
        self.p_rgb_ham = None
        self.p_hs_spam = None
        self.p_hs_ham = None
        self.p_edge_spam = None
        self.p_edge_ham = None
        
        # Raw counts accumulated during training
        self.spam_rgb_counts = None
        self.ham_rgb_counts = None
        self.spam_hs_counts = None
        self.ham_hs_counts = None
        self.spam_edge_counts = None
        self.ham_edge_counts = None
        
        self.n_spam_train = 0
        self.n_ham_train = 0
        self.is_trained = False

    def train(self, spam_image_paths, ham_image_paths, auto_prior=False):
        """
        Trains the Bayesian classifier by accumulating class histograms
        from Spam and Ham training images and computing smoothed likelihoods.
        """
        self.n_spam_train = len(spam_image_paths)
        self.n_ham_train = len(ham_image_paths)
        
        if self.n_spam_train == 0 or self.n_ham_train == 0:
            raise ValueError("Training dataset must contain at least one spam and one ham image.")
            
        if auto_prior:
            total = self.n_spam_train + self.n_ham_train
            self.prior_spam = self.n_spam_train / total
            self.prior_ham = self.n_ham_train / total

        rgb_bins = self.extractor.rgb_bins
        h_bins = self.extractor.h_bins
        s_bins = self.extractor.s_bins
        edge_bins = self.extractor.edge_bins

        # Initialize accumulators
        self.spam_rgb_counts = np.zeros((rgb_bins, rgb_bins, rgb_bins), dtype=np.float64)
        self.ham_rgb_counts = np.zeros((rgb_bins, rgb_bins, rgb_bins), dtype=np.float64)
        
        self.spam_hs_counts = np.zeros((h_bins, s_bins), dtype=np.float64)
        self.ham_hs_counts = np.zeros((h_bins, s_bins), dtype=np.float64)
        
        self.spam_edge_counts = np.zeros(edge_bins, dtype=np.float64)
        self.ham_edge_counts = np.zeros(edge_bins, dtype=np.float64)

        # Accumulate Spam class histograms
        print(f"Training on {self.n_spam_train} Spam images...")
        for p in spam_image_paths:
            img = cv2.imread(p)
            if img is None:
                continue
            self.spam_rgb_counts += self.extractor.compute_rgb_3d_histogram(img)
            self.spam_hs_counts += self.extractor.compute_hs_2d_histogram(img)
            self.spam_edge_counts += self.extractor.compute_edge_histogram(img)

        # Accumulate Ham class histograms
        print(f"Training on {self.n_ham_train} Ham images...")
        for p in ham_image_paths:
            img = cv2.imread(p)
            if img is None:
                continue
            self.ham_rgb_counts += self.extractor.compute_rgb_3d_histogram(img)
            self.ham_hs_counts += self.extractor.compute_hs_2d_histogram(img)
            self.ham_edge_counts += self.extractor.compute_edge_histogram(img)

        # Compute smoothed conditional probability distributions P(feature | Class)
        self.p_rgb_spam = self.extractor.apply_laplace_smoothing(self.spam_rgb_counts)
        self.p_rgb_ham = self.extractor.apply_laplace_smoothing(self.ham_rgb_counts)

        self.p_hs_spam = self.extractor.apply_laplace_smoothing(self.spam_hs_counts)
        self.p_hs_ham = self.extractor.apply_laplace_smoothing(self.ham_hs_counts)

        self.p_edge_spam = self.extractor.apply_laplace_smoothing(self.spam_edge_counts)
        self.p_edge_ham = self.extractor.apply_laplace_smoothing(self.ham_edge_counts)

        self.is_trained = True
        print("Training completed successfully.")

    def compute_pixel_likelihood_map(self, bgr_image, custom_prior_spam=None, filter_type="none", filter_strength=1.0):
        """
        Computes the spatial pixel-level posterior map P(Spam | pixel)
        using the trained 3D RGB class histograms.
        Returns a 2D float array in [0.0, 1.0] of shape (H, W).
        """
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained before computing likelihood maps.")
            
        prior_s = custom_prior_spam if custom_prior_spam is not None else self.prior_spam
        prior_h = 1.0 - prior_s
        
        # Apply preprocessing filter if requested
        if filter_type != "none":
            bgr_image = self.extractor.apply_preprocessing_filter(bgr_image, filter_type, filter_strength)
        
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        bin_scale = 256.0 / self.extractor.rgb_bins

        # Map each pixel (r, g, b) to its quantized 3D bin index
        r_bin = np.clip((rgb[:, :, 0] / bin_scale).astype(np.int32), 0, self.extractor.rgb_bins - 1)
        g_bin = np.clip((rgb[:, :, 1] / bin_scale).astype(np.int32), 0, self.extractor.rgb_bins - 1)
        b_bin = np.clip((rgb[:, :, 2] / bin_scale).astype(np.int32), 0, self.extractor.rgb_bins - 1)

        # Look up likelihoods from class histograms
        p_c_spam = self.p_rgb_spam[r_bin, g_bin, b_bin]
        p_c_ham = self.p_rgb_ham[r_bin, g_bin, b_bin]

        # Bayes rule per pixel: P(Spam | c) = P(c | Spam)*P(Spam) / [P(c | Spam)*P(Spam) + P(c | Ham)*P(Ham)]
        num = p_c_spam * prior_s
        den = num + (p_c_ham * prior_h)
        posterior_map = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
        
        return posterior_map

    def predict(self, bgr_image, threshold=0.5, custom_prior_spam=None, 
                strength_multiplier=1.0, feature_balance=0.5,
                filter_type="none", filter_strength=1.0):
        """
        Classifies an input image into SPAM or HAM with dynamic filter controls.
        
        :param bgr_image: Input BGR image
        :param threshold: Decision boundary threshold tau in [0.05, 0.95]
        :param custom_prior_spam: Prior probability P(Spam) in [0.01, 0.99]
        :param strength_multiplier: Multiplier on Bayesian sensitivity (0.1x to 3.0x)
        :param feature_balance: Balance between Color (0.0) and Edge/Text (1.0)
        :param filter_type: Preprocessing filter ('none', 'gaussian_blur', 'sharpen', 'laplacian_edges', 'grayscale', 'contrast')
        :param filter_strength: Intensity of the preprocessing filter
        """
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained before predicting.")

        # Apply preprocessing filter if selected
        active_img = bgr_image
        if filter_type != "none":
            active_img = self.extractor.apply_preprocessing_filter(bgr_image, filter_type, filter_strength)

        prior_s = custom_prior_spam if custom_prior_spam is not None else self.prior_spam
        prior_h = 1.0 - prior_s
        prior_log_odds = np.log(prior_s / (prior_h + 1e-12))

        # 1. Color feature log-likelihood ratio using 2D HS and 3D RGB
        test_hs_hist = self.extractor.compute_hs_2d_histogram(active_img)
        norm_hs = test_hs_hist / (np.sum(test_hs_hist) + 1e-12)
        hs_llr = np.sum(norm_hs * (np.log(self.p_hs_spam) - np.log(self.p_hs_ham)))

        # 3D RGB average pixel log-likelihood ratio
        rgb = cv2.cvtColor(active_img, cv2.COLOR_BGR2RGB)
        bin_scale = 256.0 / self.extractor.rgb_bins
        r_bin = np.clip((rgb[:, :, 0] / bin_scale).astype(np.int32), 0, self.extractor.rgb_bins - 1)
        g_bin = np.clip((rgb[:, :, 1] / bin_scale).astype(np.int32), 0, self.extractor.rgb_bins - 1)
        b_bin = np.clip((rgb[:, :, 2] / bin_scale).astype(np.int32), 0, self.extractor.rgb_bins - 1)

        p_rgb_s = self.p_rgb_spam[r_bin, g_bin, b_bin]
        p_rgb_h = self.p_rgb_ham[r_bin, g_bin, b_bin]
        rgb_llr = np.mean(np.log(p_rgb_s) - np.log(p_rgb_h))

        color_llr = 0.5 * hs_llr + 0.5 * rgb_llr

        # 2. Edge feature log-likelihood ratio (Sobel magnitude)
        test_edge_hist = self.extractor.compute_edge_histogram(active_img)
        norm_edge = test_edge_hist / (np.sum(test_edge_hist) + 1e-12)
        edge_llr = np.sum(norm_edge * (np.log(self.p_edge_spam) - np.log(self.p_edge_ham)))

        # 3. High-frequency Laplacian text feature log-likelihood ratio
        test_lap = self.extractor.compute_laplacian_text_histogram(active_img, bins=self.extractor.edge_bins)
        norm_lap = test_lap / (np.sum(test_lap) + 1e-12)
        # Using edge distributions as calibrated proxy for high frequency transitions
        lap_llr = np.sum(norm_lap * (np.log(self.p_edge_spam) - np.log(self.p_edge_ham)))

        # Blend feature evidence based on feature_balance (0.0 = all color, 1.0 = all edge/text)
        w_color = 2.0 * (1.0 - feature_balance)
        w_edge = 2.0 * feature_balance
        total_evidence = (w_color * color_llr) + (w_edge * (0.7 * edge_llr + 0.3 * lap_llr))

        # Scale by calibrated scale_factor and user strength multiplier
        effective_scale = self.scale_factor * max(0.1, float(strength_multiplier))
        scaled_evidence = total_evidence * effective_scale
        total_log_odds = prior_log_odds + scaled_evidence

        # Sigmoid function for posterior spam probability
        # P(Spam | I) = 1 / (1 + exp(-log_odds))
        if total_log_odds > 30:
            spam_prob = 1.0
        elif total_log_odds < -30:
            spam_prob = 0.0
        else:
            spam_prob = float(1.0 / (1.0 + np.exp(-total_log_odds)))

        ham_prob = 1.0 - spam_prob
        prediction = "SPAM" if spam_prob >= threshold else "HAM"

        return {
            "prediction": prediction,
            "spam_probability": spam_prob,
            "ham_probability": ham_prob,
            "confidence": max(spam_prob, ham_prob) * 100.0,
            "log_odds": float(total_log_odds),
            "color_llr": float(color_llr),
            "edge_llr": float(edge_llr),
            "lap_llr": float(lap_llr),
            "strength_used": float(strength_multiplier),
            "balance_used": float(feature_balance),
            "filter_type_used": filter_type,
            "prior_spam_used": float(prior_s),
            "threshold_used": float(threshold)
        }

    def save_model(self, file_path_npz, file_path_meta):
        """Saves the trained class histograms and configuration parameters to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained model.")
            
        np.savez_compressed(
            file_path_npz,
            p_rgb_spam=self.p_rgb_spam,
            p_rgb_ham=self.p_rgb_ham,
            p_hs_spam=self.p_hs_spam,
            p_hs_ham=self.p_hs_ham,
            p_edge_spam=self.p_edge_spam,
            p_edge_ham=self.p_edge_ham,
            spam_rgb_counts=self.spam_rgb_counts,
            ham_rgb_counts=self.ham_rgb_counts,
            spam_hs_counts=self.spam_hs_counts,
            ham_hs_counts=self.ham_hs_counts,
            spam_edge_counts=self.spam_edge_counts,
            ham_edge_counts=self.ham_edge_counts
        )

        metadata = {
            "rgb_bins": self.extractor.rgb_bins,
            "h_bins": self.extractor.h_bins,
            "s_bins": self.extractor.s_bins,
            "edge_bins": self.extractor.edge_bins,
            "alpha": self.extractor.alpha,
            "prior_spam": self.prior_spam,
            "edge_weight": self.edge_weight,
            "scale_factor": self.scale_factor,
            "n_spam_train": self.n_spam_train,
            "n_ham_train": self.n_ham_train,
            "is_trained": True
        }

        with open(file_path_meta, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Model saved to {file_path_npz} and {file_path_meta}")

    def load_model(self, file_path_npz, file_path_meta):
        """Loads trained class histograms and metadata from disk."""
        if not os.path.exists(file_path_npz) or not os.path.exists(file_path_meta):
            raise FileNotFoundError("Model files not found.")

        with open(file_path_meta, "r") as f:
            meta = json.load(f)

        self.extractor = HistogramExtractor(
            rgb_bins=meta["rgb_bins"],
            h_bins=meta["h_bins"],
            s_bins=meta["s_bins"],
            edge_bins=meta["edge_bins"],
            alpha=meta["alpha"]
        )
        self.prior_spam = meta["prior_spam"]
        self.prior_ham = 1.0 - self.prior_spam
        self.edge_weight = meta["edge_weight"]
        self.scale_factor = meta["scale_factor"]
        self.n_spam_train = meta["n_spam_train"]
        self.n_ham_train = meta["n_ham_train"]

        data = np.load(file_path_npz)
        self.p_rgb_spam = data["p_rgb_spam"]
        self.p_rgb_ham = data["p_rgb_ham"]
        self.p_hs_spam = data["p_hs_spam"]
        self.p_hs_ham = data["p_hs_ham"]
        self.p_edge_spam = data["p_edge_spam"]
        self.p_edge_ham = data["p_edge_ham"]
        self.spam_rgb_counts = data["spam_rgb_counts"]
        self.ham_rgb_counts = data["ham_rgb_counts"]
        self.spam_hs_counts = data["spam_hs_counts"]
        self.ham_hs_counts = data["ham_hs_counts"]
        self.spam_edge_counts = data["spam_edge_counts"]
        self.ham_edge_counts = data["ham_edge_counts"]

        self.is_trained = True
        print(f"Model successfully loaded. Trained with {self.n_spam_train} Spam & {self.n_ham_train} Ham images.")
