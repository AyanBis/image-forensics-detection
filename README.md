# Image Forensics: AI-Generated & Tampered Image Detection

A classical computer-vision baseline that extracts Error Level Analysis (ELA), DCT frequency summaries, Local Binary Pattern (LBP) texture histograms, and color statistics, then classifies images with a scikit-learn Random Forest. It reports image-level predictions and emits ELA and heuristic suspicious-region visualizations.

## Requirements and setup

Python 3.10 or newer is recommended. From this directory:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset layout

Place labeled image files in these folders (nested folders are supported):

```text
data/
  authentic/     # unaltered camera/benchmark images
  tampered/      # manipulated or AI-generated images
```

Supported extensions: JPG/JPEG, PNG, BMP, TIFF, and WebP. Keep related originals and edits together conceptually. Avoid near-duplicate or source-related images crossing the train/test split: this starter uses a stratified random image-level split and can overstate generalization when related images appear in both partitions. For a stronger evaluation, split by source image, subject, or dataset identity before training. Include diverse devices, compression settings, image sizes, and manipulation/generation methods; balance class counts where practical. Record dataset name, license, collection procedure, and exact sample counts for the report. Do not include private or copyrighted data without permission.

## Train and evaluate

```bash
python scripts/train.py
```

Defaults: 80/20 stratified holdout, seed 42, 400-tree Random Forest. Customize paths and split with `--data`, `--model`, `--results`, `--test-size`, and `--seed`. The script saves `models/random_forest.joblib` and `outputs/evaluation.json`, prints accuracy, macro precision/recall/F1, per-class metrics, and a confusion matrix (rows are actual labels; columns predicted; label order authentic, tampered). Metrics are measured on the held-out portion of the supplied dataset.

## Predict one image

After training:

```bash
python scripts/predict.py path/to/image.jpg
```

Optional flags: `--model PATH` and `--output-dir PATH`. The JSON output gives predicted class, Random Forest probability for that class, class probabilities, and output image paths. The output directory receives `<name>_ela.png` and `<name>_heatmap.png`.

## Method

Images are decoded to RGB, aspect-ratio-preserving resized, and padded to 256×256. ELA recompresses as JPEG at quality 90 and summarizes the absolute difference globally and over a 4×4 grid. DCT features summarize log-magnitude frequency values over the same grid. LBP uses 8 neighbors at radius one and a normalized 256-bin histogram. Per-channel color statistics include mean, standard deviation, and 10th/90th percentiles. Features feed a median imputer, standard scaler, and class-balanced Random Forest. The random seed controls splitting and estimator randomness.

## Limitations

This is an educational baseline, not a forensic-grade detector. ELA depends on image format, compression history, and processing; it is not reliable proof of manipulation. DCT, LBP, and color statistics can learn dataset/source artifacts rather than tampering. A model trained on one dataset may fail on unseen cameras, generators, edits, or social-media recompression. AI-generated and tampered are combined into one positive class here, so the model cannot distinguish their causes. Random Forest probabilities are not calibrated confidence guarantees. The suspicious-region map highlights local ELA contrast heuristically; it is not segmentation, does not establish the manipulated pixels, and may highlight legitimate edges or compression artifacts. Always interpret results with provenance and human review.

## Reproducibility

Record Python and dependency versions, dataset source/version and license, image counts per class, split seed/fraction, and the saved `outputs/evaluation.json`. Keep the same folder contents and seed to reproduce the split and model configuration. File traversal is sorted. Different library versions or image decoding implementations can cause small differences. Do not compare metrics across different dataset splits as if they were directly equivalent. No metrics are claimed until the training command has been run against a documented dataset.

## Project files

```text
src/image_forensics/features.py   preprocessing and feature extraction
scripts/train.py                  training and holdout evaluation
scripts/predict.py                inference and visualizations
reports/project_report.md         end-semester report draft
```
