# AraT5 QALB 2014 Grammatical Error Correction

This directory contains the codebase for training and evaluating an AraT5-based model for Arabic Grammatical Error Correction (GEC) on the QALB 2014 dataset.

## Files Overview

### 1. `qalb_pipeline_kaggel.ipynb`
**Purpose**: This notebook implements the complete training pipeline, optimized for a Kaggle Kernel environment with GPU support.

**Key Steps**:
- **Setup**: Installs necessary dependencies (`transformers`, `datasets`, `pyarabic`, etc.).
- **Data Preparation**: Downloads the QALB dataset and parses the M2 format files (Train + Dev) into a CSV format (`incorrect`, `correct` pairs).
- **Training**: Fine-tunes the `UBC-NLP/AraT5v2-base-1024` model on the processed data. It supports resuming training from checkpoints saved in Google Drive.
- **Export**: Saves the fine-tuned model checkpoints locally for later use.

**Usage**: Run this notebook to train the model from scratch or resume training. Ensure a GPU (T4 x2 or P100) is enabled.

### 2. `qalb_test_evaluation3.ipynb`
**Purpose**: This notebook handles the evaluation of the trained model on the QALB 2014 Test set.

**Key Steps**:
- **Setup**: Installs dependencies.
- **Data Loading**: Downloads and extracts the QALB 2014 Test data.
- **Model Loading**: Downloads your specific fine-tuned model checkpoint from Google Drive.
- **Inference**: Generates corrections for the test sentences.
- **Scoring**: Calculates **BLEU** and **GLEU** scores against the reference corrections.
- **Diacritization (Post-processing)**: Applies an optional "Tashkeel" step using `Abdou/arabic-tashkeel-flan-t5-small` to add diacritics to the corrected output.
- **Output**: Saves predictions to `qalb_test_predictions.txt` and displays sample comparisons (Input vs. Reference vs. Prediction).

**Usage**: Run this notebook to evaluate your trained model's performance and generate reportable metrics.

## Requirements
- Python 3.x
- `transformers`
- `datasets`
- `pyarabic`
- `gdown`
- `sentencepiece`
- `evaluate`
- `sacrebleu`
- `torch` (with GPU support recommended)

## Notes
- These notebooks are designed to interact with Google Drive for model storage and retrieval. Ensure you have the necessary file IDs and permissions if you are using your own checkpoints.
