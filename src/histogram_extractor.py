"""
Feature & Histogram Extraction Module
DS307 Computer Vision - Project 12

Extracts multidimensional color and gradient histograms from images:
1. 3D RGB Color Histogram (quantized color distribution)
2. 2D Hue-Saturation (HSV) Histogram (chromaticity invariant to illumination)
3. 1D Edge Gradient Magnitude Histogram (text/geometric sharpness)
Includes Laplace (additive) smoothing to prevent zero-probability issues.
"""

import numpy as np
import cv2


class HistogramExtractor:
    def __init__(self, rgb_bins=16, h_bins=18, s_bins=16, edge_bins=16, alpha=1.0):
        """
        Initialize the histogram extractor with bin resolutions and smoothing parameter.
        
        :param rgb_bins: Number of bins per color channel in 3D RGB space (e.g. 16 -> 16x16x16 = 4096 bins)
        :param h_bins: Number of Hue bins (0-180 in OpenCV)
        :param s_bins: Number of Saturation bins (0-256 in OpenCV)
        :param edge_bins: Number of Edge magnitude bins (0-255)
        :param alpha: Laplace additive smoothing parameter
        """
        self.rgb_bins = rgb_bins
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.edge_bins = edge_bins
        self.alpha = alpha

    def compute_rgb_3d_histogram(self, bgr_image):
        """
        Computes 3D RGB histogram with (rgb_bins x rgb_bins x rgb_bins) bins.
        Returns a 3D numpy array of counts.
        """
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist(
            [rgb],
            [0, 1, 2],
            None,
            [self.rgb_bins, self.rgb_bins, self.rgb_bins],
            [0, 256, 0, 256, 0, 256]
        )
        return hist

    def compute_hs_2d_histogram(self, bgr_image):
        """
        Computes 2D Hue-Saturation histogram in HSV color space.
        Returns a 2D numpy array of counts (h_bins x s_bins).
        """
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1],
            None,
            [self.h_bins, self.s_bins],
            [0, 180, 0, 256]
        )
        return hist

    def compute_edge_histogram(self, bgr_image):
        """
        Computes 1D gradient magnitude histogram using Sobel spatial derivatives.
        High edge counts correspond to text, banners, and sharp graphic borders.
        """
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        # Normalize magnitude to 0-255 range
        mag_clamped = np.clip(mag, 0, 255).astype(np.uint8)
        hist = cv2.calcHist([mag_clamped], [0], None, [self.edge_bins], [0, 256])
        return hist.flatten()

    def compute_laplacian_text_histogram(self, bgr_image, bins=16):
        """
        Computes 1D histogram of 2nd spatial derivatives using Laplacian operator.
        Detects high-frequency text stroke densities and exclamation typography.
        """
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        lap_abs = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
        hist = cv2.calcHist([lap_abs], [0], None, [bins], [0, 256])
        return hist.flatten()

    def compute_saturation_histogram(self, bgr_image, bins=16):
        """
        Computes 1D Saturation histogram in HSV color space.
        Spam images concentrate in high-saturation bins (vivid synthetic banners).
        """
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        hist = cv2.calcHist([sat], [0], None, [bins], [0, 256])
        return hist.flatten()

    @staticmethod
    def apply_preprocessing_filter(bgr_image, filter_type="none", strength=1.0):
        """
        Applies real-time image preprocessing filters to test Bayesian classifier robustness:
        - none: raw image
        - gaussian_blur: simulates low-res compression or blur evasion
        - sharpen: enhances text and graphic boundaries
        - laplacian_edges: high-pass edge-only representation
        - grayscale: strips color to isolate pure structural edge evidence
        - contrast: dynamically amplifies or compresses contrast
        """
        if filter_type == "none" or strength <= 0:
            return bgr_image.copy()

        img = bgr_image.copy()

        if filter_type == "gaussian_blur":
            # Map strength (0.1 - 5.0) to odd kernel size
            ksize = int(strength * 4) * 2 + 1
            ksize = max(3, min(ksize, 31))
            return cv2.GaussianBlur(img, (ksize, ksize), strength * 1.5)

        elif filter_type == "sharpen":
            # Unsharp masking: original + strength * (original - blur)
            gaussian = cv2.GaussianBlur(img, (9, 9), 2.0)
            unsharp = cv2.addWeighted(img, 1.0 + float(strength), gaussian, -float(strength), 0)
            return np.clip(unsharp, 0, 255).astype(np.uint8)

        elif filter_type == "laplacian_edges":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
            lap_norm = np.clip(np.abs(lap) * (1.0 + strength), 0, 255).astype(np.uint8)
            return cv2.cvtColor(lap_norm, cv2.COLOR_GRAY2BGR)

        elif filter_type == "grayscale":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        elif filter_type == "contrast":
            # contrast multiplier from 0.5 to 2.5
            alpha = float(strength)
            adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=10 * (alpha - 1.0))
            return adjusted

        return img

    def apply_laplace_smoothing(self, hist_counts, alpha=None):
        """
        Applies Laplace (additive) smoothing to convert raw histogram counts
        into a valid probability distribution P(bin | Class).
        
        P(b | C) = (count(b) + alpha) / (total_counts + alpha * total_bins)
        """
        if alpha is None:
            alpha = self.alpha
            
        total_counts = np.sum(hist_counts)
        total_bins = hist_counts.size
        
        smoothed_prob = (hist_counts + alpha) / (total_counts + alpha * total_bins)
        return smoothed_prob

    def extract_all_distributions(self, bgr_image):
        """
        Extracts smoothed probability distributions for an individual image:
        - 3D RGB distribution
        - 2D HS distribution
        - 1D Edge distribution
        """
        rgb_hist = self.compute_rgb_3d_histogram(bgr_image)
        hs_hist = self.compute_hs_2d_histogram(bgr_image)
        edge_hist = self.compute_edge_histogram(bgr_image)
        
        p_rgb = self.apply_laplace_smoothing(rgb_hist)
        p_hs = self.apply_laplace_smoothing(hs_hist)
        p_edge = self.apply_laplace_smoothing(edge_hist)
        
        return {
            "p_rgb": p_rgb,
            "p_hs": p_hs,
            "p_edge": p_edge,
            "rgb_counts": rgb_hist,
            "hs_counts": hs_hist,
            "edge_counts": edge_hist
        }
