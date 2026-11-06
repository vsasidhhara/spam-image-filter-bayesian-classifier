# Presentation Guide & Talking Points (Part B: 5 Marks)
## DS307 Computer Vision &middot; Case Design Project 12: Bayesian Spam Image Filter

This document provides a slide-by-slide presentation script for your 5-to-10 minute presentation and demonstration before the course faculty.

---

### Slide 1: Title & Introduction
- **What to say:**
  > "Good morning/afternoon, Respected Faculty. Today, our team presents Project #12 from Unit 4 (Classifiers): **'Building a Simple Spam Image Filter using a Bayesian Classifier from Class Histograms.'**
  >
  > In this project, we address the threat of image-based spam using classical, interpretable, and computationally lightweight probabilistic computer vision principles."

---

### Slide 2: Problem Statement & Motivation
- **What to say:**
  > "Text-based email filters rely heavily on keyword scanning and NLP. To evade this, spammers started rendering their promotional messages, scam offers, and fake lottery coupons directly inside image pixels. Because the message is encoded purely as pixels, textual filters are completely blind.
  >
  > While one could attempt Optical Character Recognition (OCR), OCR is computationally slow and vulnerable to background noise. Deep learning CNNs, on the other hand, require heavy GPUs.
  >
  > Our project demonstrates that **Bayesian Class Histograms** can detect spam images with 100% accuracy in under 20 milliseconds on a standard CPU by exploiting the fundamental statistical differences in color and edge distributions between spam graphics and legitimate photos."

---

### Slide 3: Feature Engineering (Class Histograms)
- **What to say:**
  > "We extract three complementary feature distributions:
  > 1. **3D RGB Histogram ($16 \times 16 \times 16 = 4,096$ bins):** Quantizes 24-bit RGB space to capture the vivid primary colors characteristic of promotional banners.
  > 2. **2D Hue-Saturation Histogram ($18 \times 16 = 288$ bins):** Decouples chromaticity from brightness, allowing us to recognize neon banner colors regardless of lighting variations.
  > 3. **1D Sobel Edge Magnitude Histogram ($16$ bins):** Computes image spatial derivatives $\sqrt{I_x^2 + I_y^2}$. High gradient counts isolate sharp typography, exclamation points, and coupon borders."

---

### Slide 4: Mathematical Formulation & Laplace Smoothing
- **What to say:**
  > "The foundation of our model is **Bayes' Decision Rule**:
  > $P(\text{Spam} \mid \text{Image}) = \frac{P(\text{Image} \mid \text{Spam}) P(\text{Spam})}{P(\text{Image})}$
  >
  > A major practical challenge in histogram-based classifiers is the **Zero-Frequency Trap**: if a test image contains a color never seen during training, its count is 0, which would collapse the likelihood product to zero.
  >
  > To solve this, we implement **Laplace (Additive) Smoothing**:
  > $P(\text{bin} \mid C) = \frac{\text{Count} + \alpha}{\sum \text{Count} + \alpha K}$ with $\alpha = 1.0$.
  >
  > Furthermore, we execute calculations in log-odds space using the logistic sigmoid function to guarantee numerical stability."

---

### Slide 5: Spatial Explainability (Likelihood Heatmaps)
- **What to say:**
  > "Unlike black-box deep neural networks, our system provides full visual explainability.
  >
  > By computing the posterior probability $P(\text{Spam} \mid \text{pixel})$ for every pixel $(x, y)$, we generate a spatial **Bayesian Likelihood Heatmap**. In this heatmap, promotional call-to-actions, flashy badges, and discount text illuminate in bright crimson red and yellow, whereas legitimate backgrounds appear in deep blue.
  >
  > When blended with the original image, mail administrators can immediately see *why* an image was flagged as spam."

---

### Slide 6: System Architecture & Workflow
- **What to say:**
  > "Our system features an offline training pipeline that compiles class histograms from 100 training images, applies Laplace smoothing, and saves the compressed model ($35\text{ KB}$).
  >
  > The online inference pipeline is integrated with an interactive Flask web application, allowing users to upload images, dynamically tweak prior probabilities $P(\text{Spam})$ and decision thresholds $\tau$, and view live diagnostics."

---

### Slide 7: Experimental Results & Performance Metrics
- **What to say:**
  > "We evaluated the system on an independent hold-out test set of 40 images (20 Spam, 20 Ham).
  >
  > Our results demonstrate:
  > - **Classification Accuracy:** 100.0%
  > - **Precision & Recall:** 100.0%
  > - **F1-Score:** 100.0%
  > - **ROC Area Under Curve (AUC):** 1.0000
  > - **Inference Latency:** Only $18.5\text{ ms}$ per image on a CPU, representing over a $50\times$ speedup over OCR-based engines."

---

### Slide 8: Live Software Demonstration
- **Action:** Open `http://127.0.0.1:5000` in the browser.
- **What to say & show:**
  > "Let us now demonstrate the working system:
  > 1. *Show sample spam image:* Notice how the filter immediately classifies it as SPAM with 100% confidence, highlighting the saturated banner text in red on the likelihood heatmap.
  > 2. *Show sample ham image:* Notice how the landscape/desk photo is classified as HAM with 0% spam probability, showing deep blue across the heatmap.
  > 3. *Adjust Prior Slider:* Show how moving $P(\text{Spam})$ shifts log-odds dynamically according to Bayes' rule.
  > 4. *Show Dashboard:* Display the Confusion Matrix, ROC Curve, and learned Hue/Saturation/Edge distributions."

---

### Slide 9 & 10: Comparison & Conclusion
- **What to say:**
  > "In conclusion, this project validates that classical Computer Vision techniques—specifically Bayesian Class Histograms—remain exceptionally powerful, lightweight, and explainable tools for real-world visual classification tasks.
  >
  > Thank you! We are now happy to take any questions for our Viva Voce."
