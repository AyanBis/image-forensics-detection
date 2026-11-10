# End-Semester Project Report

## 1. Title

**Image Forensics: AI-Generated and Tampered Image Detection Using Classical Computer Vision and Machine Learning**

**Group members:** [Add names of all group members]

## 2. Abstract

This project develops an educational image-forensics pipeline to classify an input image as authentic or suspicious (tampered or AI-generated). The system preprocesses images to a fixed size and extracts handcrafted features from Error Level Analysis (ELA), Discrete Cosine Transform (DCT) frequency summaries, Local Binary Patterns (LBP), and per-channel color statistics. A class-balanced Random Forest classifier is trained and evaluated with a stratified holdout split. The implementation reports accuracy, macro-averaged precision, recall and F1-score, per-class metrics, and a confusion matrix. At inference time, it returns class probabilities and saves an ELA visualization and a heuristic heatmap of locally elevated ELA contrast. The approach is intended as a reproducible baseline for study; it does not prove authenticity or precisely localize edits. Quantitative findings are pending execution on a documented, labeled dataset.

## 3. Approach

### Data and preprocessing

The training script reads supported image files from `data/authentic/` and `data/tampered/`, including nested directories. Images are decoded into RGB, resized to fit within 256 × 256 pixels while preserving aspect ratio, and centered on a zero-valued square canvas. The dataset is split into training and test portions with stratification and a fixed default random seed of 42.

### Feature extraction

Four feature groups are combined into a single vector:

1. **ELA:** The image is recompressed as JPEG at quality 90. Absolute pixel differences are summarized using regional means, standard deviations, 90th percentiles, and global summary statistics.
2. **DCT:** A two-dimensional DCT is applied to grayscale intensities. Log-magnitude coefficients are summarized over a 4 × 4 grid to represent frequency patterns.
3. **LBP texture:** An 8-neighbor, radius-one Local Binary Pattern is computed, then represented by a normalized 256-bin histogram.
4. **Color statistics:** Each RGB channel contributes its mean, standard deviation, and 10th and 90th percentiles.

### Classifier and evaluation

A scikit-learn pipeline imputes missing feature values with medians, standardizes features, and fits a Random Forest with 400 trees, balanced class weights, and seed 42. The held-out test partition is used to calculate accuracy, macro precision, macro recall, macro F1, per-class precision/recall/F1, and a confusion matrix. The matrix uses `authentic`, `tampered` order for both rows (actual) and columns (predicted).

### Inference and visualization

The inference script loads the saved model and reports the predicted label, its Random Forest probability, and probabilities for both labels. It writes an ELA image and a color heatmap based on locally elevated ELA contrast. The heatmap is a qualitative heuristic and must not be treated as a pixel-accurate tampering mask.

## 4. Results

### Measured results

**Pending dataset execution.** No labeled image dataset was present when this report was prepared, so no model was trained and no performance metrics have been measured. No accuracy, precision, recall, F1-score, or confusion-matrix values are claimed here. After placing images in the dataset folders and running `python scripts/train.py`, copy the measured values from `outputs/evaluation.json` into the table below.

| Metric | Measured value |
|---|---:|
| Accuracy | Pending |
| Macro precision | Pending |
| Macro recall | Pending |
| Macro F1-score | Pending |
| Confusion matrix (actual rows; predicted columns; authentic, tampered) | Pending |

**Evaluation configuration:** stratified random holdout; default test fraction 0.20; seed 42. Record actual class counts, train/test sample counts, dataset source/version, and any changed split configuration when filling in results. A random image-level split may leak near-duplicates or related source images across partitions; a source-grouped split is preferable for stronger claims.

### Qualitative output

For an inference example, insert the image filename, predicted class and probability, then include the saved heatmap and explain that the highlighted contrast is heuristic. Do not describe it as a verified manipulated region without pixel-level ground truth and a localization evaluation.

## 5. Conclusion

The project implements a complete, reproducible classical machine-learning baseline for image-level classification using compression, frequency, texture, and color cues. It produces standard classification metrics and supplementary ELA and heatmap visualizations. Its actual effectiveness remains unknown until evaluation on a documented dataset. Because handcrafted forensic cues are sensitive to compression and dataset bias, and because the heatmap is not a segmentation model, outputs should be treated as preliminary indicators rather than definitive authenticity judgments. Future work should evaluate on larger and more diverse data, use source-grouped and external test sets, assess probability calibration, distinguish AI-generated from edited images, and validate localization against pixel-level annotations.
