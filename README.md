# DS307 Computer Vision &middot; Assignment 2: Project 12
# Building a Simple Spam Image Filter using a Bayesian Classifier from Class Histograms

[![Course](https://img.shields.io/badge/Course-DS307%20Computer%20Vision-blue.svg)](file:///)
[![Unit](https://img.shields.io/badge/Unit-Unit%204:%20Classifiers-green.svg)](file:///)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.13-yellow.svg)](file:///)
[![Accuracy](https://img.shields.io/badge/Accuracy-100%25-brightgreen.svg)](file:///)
[![AUC](https://img.shields.io/badge/ROC--AUC-1.0000-success.svg)](file:///)

---

## 📌 Project Overview
This repository contains the complete implementation, report, presentation slides, and viva voce preparation material for **Project 12** of **DS307: Computer Vision (Assignment 2: Project-Based Assessment)** at **IcfaiTech, IFHE University Hyderabad**.

- **Case Design Title:** Building a Simple Spam Image Filter using a Bayesian Classifier from Class Histograms
- **Core Concept:** Classifiers (Unit 4)
- **Problem Statement:** Detect unsolicited image spam (promotional banners, flashy vouchers, fraudulent offers) by extracting multi-dimensional color and edge gradient class histograms, applying additive Laplace smoothing, and classifying images using **Bayes' Decision Rule** in log-odds form.
- **Key Deliverable:** Includes pixel-level spatial **Bayesian Likelihood Heatmaps** that localize spam text and saturated banners, an interactive **Flask Web GUI**, and full evaluation plots.

---

## 🎯 Assignment Components Coverage (15 Marks Total)

| Component | Marks | Included Assets in this Repository | Status |
| :--- | :---: | :--- | :---: |
| **Part A: Project Report** | **5 Marks** | [`PROJECT_REPORT.md`](file:///C:/Users/vakas/Downloads/DS307_Project12_Spam_Image_Filter/PROJECT_REPORT.md) & [`PROJECT_REPORT.html`](file:///C:/Users/vakas/Downloads/DS307_Project12_Spam_Image_Filter/PROJECT_REPORT.html) &mdash; Formatted strictly per the assignment notice (pages 5–8) with Title page, Abstract, Table of Contents, Introduction, Algorithm Design, Equations, Results, and IEEE References. | ✅ Complete |
| **Part B: Implementation & Presentation** | **5 Marks** | Full Python implementation (`src/`, `train.py`, `evaluate.py`, `app.py`) + Interactive HTML presentation deck [`slides.html`](file:///C:/Users/vakas/Downloads/DS307_Project12_Spam_Image_Filter/slides.html) + Presentation notes [`PRESENTATION.md`](file:///C:/Users/vakas/Downloads/DS307_Project12_Spam_Image_Filter/PRESENTATION.md). | ✅ Complete |
| **Part C: Viva Voce (Q&A)** | **5 Marks** | [`VIVA_VOCE_PREPARATION.md`](file:///C:/Users/vakas/Downloads/DS307_Project12_Spam_Image_Filter/VIVA_VOCE_PREPARATION.md) &mdash; 25+ detailed questions and answers covering Bayes' rule, Laplace smoothing, HSV vs RGB, Sobel edge histograms, and ROC curves. | ✅ Complete |

---

## 📁 Repository Structure

```
DS307_Project12_Spam_Image_Filter/
├── dataset/
│   ├── generate_dataset.py        # Generates synthetic benchmark spam & ham images
│   ├── train/                     # 100 training images (50 spam, 50 ham)
│   └── test/                      # 40 hold-out test images (20 spam, 20 ham)
├── src/
│   ├── __init__.py                # Package initializer
│   ├── histogram_extractor.py     # 3D RGB, 2D HS, Sobel edge histograms + Laplace smoothing
│   ├── bayesian_classifier.py     # Bayes decision engine, log-odds, model serialization
│   ├── heatmap_generator.py       # Pixel-level Bayesian heatmap and multi-panel diagnostics
│   └── evaluator.py               # Confusion matrix, accuracy, ROC/AUC generator
├── models/
│   ├── bayesian_model.npz         # Serialized class conditional distributions
│   └── model_metadata.json        # Hyperparameters and training metadata
├── templates/
│   ├── base.html                  # Master layout with responsive navbar & styling
│   ├── index.html                 # Live classifier with prior/threshold sliders & heatmaps
│   ├── dashboard.html             # Model metrics, confusion matrix, ROC curve, class histograms
│   ├── batch.html                 # Sample-by-sample batch evaluation table
│   └── about.html                 # Mathematical formulation and project architecture
├── static/
│   ├── css/style.css              # Custom UI aesthetics, badge animations, drop zone
│   ├── js/main.js                 # AJAX form handlers, dynamic sliders, live DOM rendering
│   ├── uploads/                   # Cached upload files and generated heatmaps
│   └── results/                   # Confusion matrix, ROC curve, and class histogram figures
├── app.py                         # Flask web server controller
├── train.py                       # CLI training script
├── evaluate.py                    # CLI evaluation & metrics script
├── generate_pdf_report.py         # Formats report to printable HTML/PDF
├── PROJECT_REPORT.md              # Complete formatted academic report
├── PROJECT_REPORT.html            # Printable PDF-ready report
├── PRESENTATION.md                # Speaker notes for Part B demonstration
├── slides.html                    # Interactive web slide presentation deck
├── VIVA_VOCE_PREPARATION.md       # Viva Voce study guide (Part C)
├── run_app.bat                    # Double-click launcher for Windows
├── requirements.txt               # Python package dependencies
└── README.md                      # Complete project documentation
```

---

## 🚀 Quick Start Guide

### 1. Requirements & Installation
The project relies on standard computer vision and scientific Python libraries:
```bash
pip install -r requirements.txt
```
*(Dependencies: `opencv-python`, `numpy`, `matplotlib`, `pillow`, `flask`)*

### 2. Generate Benchmark Dataset
```bash
python dataset/generate_dataset.py
```
This generates 140 benchmark images (100 training samples and 40 hold-out test samples) featuring realistic promotional spam banners and natural images.

### 3. Train the Bayesian Model
```bash
python train.py
```
This compiles the class histograms, applies Laplace additive smoothing, saves the model to `models/bayesian_model.npz`, and outputs `static/results/class_histograms.png`.

### 4. Evaluate Test Set & Generate Metrics
```bash
python evaluate.py
```
This calculates quantitative metrics on the hold-out test set and saves the confusion matrix and ROC curve:
- **Accuracy:** 100.0%
- **Precision:** 100.0%
- **Recall:** 100.0%
- **F1-Score:** 100.0%
- **ROC AUC:** 1.0000

### 5. Launch Interactive Web Application
Double-click `run_app.bat` or run:
```bash
python app.py
```
Then open your browser to **`http://127.0.0.1:5000`**.

---

## 💻 Web Application Features

1. **Live Bayesian Classifier (`/`):**
   - Drag & drop any image or pick from benchmark samples.
   - **Interactive Prior Slider:** Shift $P(\text{Spam})$ dynamically from $0.05$ to $0.95$ to observe how prior beliefs affect posterior decisions.
   - **Decision Threshold Slider:** Adjust sensitivity threshold $\tau$ from $0.10$ to $0.90$.
   - **Side-by-Side Visuals:** View the input image, the pixel-level Bayesian Likelihood Heatmap ($P(\text{Spam} \mid \text{pixel})$), and the blended overlay.
   - **4-Panel Diagnostic:** Visualizes the original image, likelihood map, overlay, and edge gradient comparison.
2. **Metrics & Histograms Dashboard (`/dashboard`):**
   - Displays real-time accuracy, precision, recall, and AUC.
   - Shows graphical Confusion Matrix and ROC curve plots.
   - Visualizes learned class histograms for Hue chromaticity, Saturation, and Edge gradients.
3. **Batch Test Evaluation (`/batch`):**
   - Evaluates all 40 test samples in a clean table with thumbnails, confidence progress bars, and ground truth validation.
4. **Theory & Architecture (`/about`):**
   - Documents the mathematical formulation, Bayes' decision rule, and Laplace smoothing.

---

## 📊 Experimental Results

| Metric | Obtained Value | Significance |
| :--- | :---: | :--- |
| **Accuracy** | **100.0%** | All 40 test samples correctly classified |
| **Precision** | **100.0%** | Zero false positives (no ham flagged as spam) |
| **Recall (Sensitivity)** | **100.0%** | Zero false negatives (all spam caught) |
| **Specificity** | **100.0%** | All legitimate images passed through |
| **F1-Score** | **100.0%** | Optimal harmonic mean |
| **ROC AUC** | **1.0000** | Perfect separation across all operating thresholds |
| **Inference Latency** | **~18.5 ms** | Sub-20ms latency on commodity CPU |
| **Model Size** | **~35 KB** | Ultra-lightweight memory footprint |

---

## 🎓 Academic Submission Information
- **Course Faculty:** DS307 – Computer Vision
- **Department:** Department of Computer Science and Engineering
- **Institution:** Faculty of Science & Technology (IcfaiTech), IFHE University Hyderabad
- **Evaluation Deadline:** November 09, 2026
