## 📂 Project Script Registry
Here is a guide to every script in this folder and what it does.

| Script Name | Purpose | Status |
| :--- | :--- | :--- |
| **`train_gec_model.py`** | **The Final Model.** Trains AraBART on 20k sentences (Augmented). | ✅ **Use this.** |
| `train_trial1_arat5.py` | Trial 1 (Failed). Implementation of AraT5-base that failed due to vocab mismatch. | ❌ Archive. |
| `train_trial2_mt5.py` | Trial 2 (Failed). Implementation of mT5-base that failed due to hallucinations. | ❌ Archive. |
| `train_trial3_arabart_broken.py` | Trial 3 (Failed). AraBART implementation **without** weight repair (broken output). | ❌ Archive. |
| `evaluate_model_clean.py` | **Official Evaluation.** Calculates BLEU/GLEU/ROUGE on clean output. | 📊 Metrics. |
| `evaluate_model_raw.py` | Debug Evaluation. Shows raw model output (with hallucinations). | 🐞 Debug. |
---
*   **Link**: [Google Drive Folder](https://drive.google.com/drive/u/0/folders/1rBwuBdGMDc-1ndfaZhgkIXGFwgz1Sf4f)
*   **Description**: This folder contains the saved model output (weights + config) accessible by anyone. You can download this directory and point the `verify_model.py` script to it to run corrections.
---
## 1. Project Overview & Architecture
This project implements a **Grammatical Error Correction (GEC)** system for Arabic text using the **Sequence-to-Sequence (Seq2Seq)** paradigm.
The core idea is to treat "Error Correction" as a "Translation" task:
*   **Source Language**: Erroneous Arabic (from `.sent` files).
*   **Target Language**: Correct Arabic (from `.cor` files).

### The Model: AraBART
I chose **`moussaKam/AraBART`** as the backbone.
*   **Type**: BART (Bidirectional and Auto-Regressive Transformers).
*   **Why?**: Unlike T5 (which had vocabulary issues) and mBART (which was too large), AraBART is pretrained specifically on Arabic data and fits within the memory constraints of a local machine while delivering strong performance.

## 2. Implementation Logic (`train_gec_model.py`)

### A. Custom Dataset (`QALBDataset`)
I implemented a custom PyTorch `Dataset` class to handle the unique format of the QALB 2015 Shared Task data.
*   **Parsing**: The raw files contain metadata (e.g., `S001_T02 word...`). My code strips these IDs to extract the pure text.
*   **Tokenization**: It uses the `AraBART` tokenizer to convert text into numerical IDs. I explicitly handle padding (`<pad>`) and truncation to ensure uniform input size (`max_length=128`).
*   **Augmentation Logic**: The class supports lists of file paths, allowing me to seamlessly merge the QALB 2014 and 2015 datasets to create a larger training corpus.

### B. Training Configuration
I configured the `Seq2SeqTrainer` with specific parameters to optimize for both performance and hardware limitations (Mac MPS):
*   **Optimizer**: I utilized **Adafactor** instead of AdamW. Adafactor is designed to be memory-efficient, which is crucial when fine-tuning Transformers on consumer hardware.
*   **Batch Size Strategy**: To avoid Out-Of-Memory (OOM) errors, I set the batch size to **1** but used `gradient_accumulation_steps=8`.
    *   *Effect*: The model logically updates its weights every 8 samples, effectively simulating a batch size of 8 without the memory cost.
*   **Epochs**: I trained for **30 epochs**. Since GEC is a subtle task (changing one character in a sentence), the model required extensive exposure to the data to learn the nuances.

## 3. The "Repair" Utility (`repair_model.py`)
One of the major challenges encountered was a bug where the model's embedding weights were not saved correctly during checkpointing (a known issue with some Hugging Face models + Safetensors).
*   **Symptoms**: The model would output repetitive character noise (e.g., `تتتتت`).
*   **Solution**: I wrote `repair_model.py`. This script loads the "broken" checkpoint and the original pretrained model side-by-side, copies the missing embedding weights from the original to the checkpoint, and saves a "Fixed" version. This rescued the training run without needing to restart.

## 4. Evaluation Strategy
I implemented a robust evaluation pipeline to measure success using standard NLP metrics.
*   **Metric 1: BLEU**: Measures n-gram overlap. Good for general similarity.
*   **Metric 2: GLEU (Google BLEU)**: Specialized for sentence rewriting. It penalizes "bad edits" and rewards "good edits" more accurately than BLEU.
*   **Insight**: My "Augmented" model achieved a **GLEU score of ~36.7**, which is a strong baseline given the dataset size.

## 5. Summary of Results
By moving from a naive implementation (AraT5) to a robust, architecture-aware solution (AraBART + Adafactor + Augmentation), I successfully built a model that doesn't just copy text but actively corrects grammatical errors (e.g., fixing subject-verb agreement `أنا يحب` -> `أنا أحب`).
---

# Model Evaluation Documentation

This document describes the evaluation scripts and results for the Arabic GEC model. Two evaluation modes are available to assess model performance with and without special tokens.

## Evaluation Scripts

### 1. `evaluate_model_clean.py`
- **Purpose**: Evaluates the model's generated text in a human-readable format, similar to standard end-user output.
- **Behavior**: Hides special tokens (`</s>`, `<s>`, `<pad>`).
- **Output File**: `qalb_2015_l2_test_predictions_clean.txt`
- **Use Case**: Best for calculating standard metrics (BLEU, GLEU, ROUGE, BERTScore) as it aligns with the clean reference text.

### 2. `evaluate_model_raw.py`
- **Purpose**: Evaluates the raw output from the model's tokenizer.
- **Behavior**: Includes ALL tokens output by the model, including start/end and padding tokens.
- **Output File**: `qalb_2015_l2_test_predictions_raw.txt`
- **Use Case**: Debugging model behavior, inspecting sentence boundary generation, and analyzing tokenization artifacts.

## Evaluation Results (QALB 2015 Test Set)

Comparison of model performance between clean and raw outputs:

| Metric | Clean Output | Raw Output (with tokens) | Interpretation |
| :--- | :--- | :--- | :--- |
| **BLEU** | 30.73 | **32.56** | Raw output surprisingly scored higher in standard BLEU, possibly due to length penalties or specific n-gram overlap dynamics. |
| **GLEU** | **36.69** | 25.96 | Google BLEU favors the clean output significantly. |
| **ROUGE-L** | **0.1177** | 0.0483 | Clean output matches reference structure much better. |
| **BERTScore F1** | **0.8780** | 0.7907 | Semantic similarity is much higher for clean output. |

### Analysis
- **Standard Metrics (ROUGE, BERTScore)**: The "Clean" output performs drastically better. This is expected because the reference sentences are clean text. Including special tokens in the prediction is penalized as "incorrect" text by these metrics.
- **BLEU vs GLEU**: The discrepancy between BLEU and GLEU suggests that while n-gram overlap exists in the raw output, the overall fluency and structure (captured better by GLEU/BERTScore) are better represented in the clean output.

## Model Trials & Errors Comparison

| Model | Error / Issue | Diagnosis | Example Output (Bad) |
| :--- | :--- | :--- | :--- |
| **AraT5-base** | **Vocabulary Mismatch** | The code used a 32k vocab tokenizer, but the model expected 64k. | `<unk> <unk> <unk>` |
| **mt5-base** | **Hallucination** | Model was too massive and multilingual; it generated new stories instead of correcting. | `ذهب الطالب إلى الجامعة` (Completely changed meaning) |
| **AraBART** *(Initial)* | **Broken Weights** | "Missing Embeddings" bug in `safetensors` saving. Headless model. | `تتتتتتتتتتتت` (Noise repetition) |
| **AraBART** *(Augmented)* | **Success** ✅ | **Fixed!** Data Augmentation + Weight Repair script. | `أنا أحب اللغة العربية كثيرا .` |

## Model Evolution: From Garbage to Grammar

Here is the evolution of the model's performance on the same (or similar) input sentences, showing exactly how the output improved from "broken" to "grammatically correct."

| Trial | Model | Input Sentence | Actual Model Output | Diagnosis / Status |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **AraT5-base** | `ذهبت الى المدرسة` | `<unk> <unk> <unk>` | **Failed.** Vocabulary size mismatch (32k vs 64k) caused the model to panic. |
| **2** | **mt5-base** | `ذهبت الى المدرسة` | `ذهب الطالب إلى الجامعة` | **Hallucination.** Model was too big & multilingual; it ignored the input and wrote its own story. |
| **3** | **AraBART** *(Broken)* | `ذهبت الى المدرسة` | `تتتتتتتتتتتت` | **Broken Weights.** The "Missing Embeddings" bug disconnected the model's head, causing noise loops. |
| **4** | **AraBART** *(Small Data)* | `أنا يحب اللغة العربية` | `أنا يحب اللغة العربية` | **Partial Fix.** It learned to copy and fix spelling (`الى` -> `إلى`), but **ignored grammar** due to lack of data. |
| **5** | **AraBART** *(Augmented)* | `أنا يحب اللغة العربية` | `أنا أحب اللغة العربية` | **Success!** ✅ Adding 20k sentences finally taught the model to correct the verb agreement (`يحب` -> `أحب`). |

## Impact of Data Augmentation (Scores)

Comparison between training on **only 2015 data** (310 sentences) vs. **Augmented 2014+2015 data** (20,000 sentences).

| Metric | Before Augmentation (310 lines) | After Augmentation (20k lines) | Improvement | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **BLEU** | 29.91 | **30.73** | +0.82 | Slight improvement in n-gram overlap. |
| **GLEU** | 35.99 | **36.69** | **+0.70** | **Better grammar.** GLEU captures the fluency edits better. |

## How to Run
To reproduce these results:

```bash
# For clean evaluation
python3 evaluate_model_clean.py

# For raw evaluation
python3 evaluate_model_raw.py
```
---
## Grammar Correction and Diacritization Pipeline

This repository implements a **modular pipeline for Arabic grammatical error correction and diacritization**, grounded in QALB annotations.  
Each component is designed to solve a well-defined subtask, allowing interpretability, controlled evaluation, and independent improvement.

---

## 1️⃣ Grammar Correction Model (Seq2Seq)

### 🎯 Goal
Build a **sequence-to-sequence (Seq2Seq)** model that automatically corrects grammatical errors in Arabic text by transforming an **incorrect sentence** into its **correct form**.

- **Input:**  
  `ذهب الولد المدرسة`
- **Output:**  
  `ذهب الولد إلى المدرسة`

This component performs **full-sentence grammatical correction**.

---

### 🏆 Model Selection

**Selected Model:** `UBC-NLP/AraT5v2-base-1024`

- **Architecture:** Encoder–Decoder (T5-style)
- **Task:** Text-to-text grammatical correction
- **Rationale:**
  - Designed for Arabic text generation
  - Naturally fits sentence-level correction
  - Performs well in low-resource fine-tuning settings
  - Clean and interpretable Seq2Seq formulation

---

### 📁 Component Design

**File:** `correction_model.py`

#### 1. `read_text_file(path)`
Reads raw text lines from QALB `.sent` or `.cor` files.

- **Input:** Path to text file  
- **Output:** List of sentences

**Example Output:**
```python
[
  "ذهب الولد المدرسة",
  "أكلت التفاحة"
]

```
---
## Diacritization Model Implementation Plan

### Goal Description
Add the final touches to Arabic text by applying **Diacritics (Tashkeel)** to already corrected sentences.

**Example**
- **Input:**  
  `ذهب الولد إلى المدرسة`
- **Output:**  
  `ذَهَبَ الوَلَدُ إِلَى المَدْرَسَةِ`

---

## Strategy: Pre-trained Model 🧠

Training a diacritizer from scratch requires **massive fully-diacritized corpora**, which are not available in **QALB**.  
Therefore, we rely on an **open-source pre-trained model**.

### Recommended Models (Initial Candidates)
- `interpress/shakkala`
- `mannaa/tashkeela-bert`

We treat diacritization as a **translation task**:
> Non-Diacritized Arabic → Diacritized Arabic

---

## Component Design

**File:** `diacritization_model.py`

---

## 1. Model Selection 🏆

### Selected Model
**`glonor/byt5-arabic-diacritization`**  
(ByT5 – Seq2Seq)

### Architecture
- **Byte-Level T5**
- Text-to-Text (Seq2Seq)

### Reason for Selection
- Consumes **raw text directly**
- Outputs **fully diacritized text**
- Robust and easy to integrate

### Rejected Alternatives
- **AraT5 / Shakkala (BERT/RNN-based)**  
  Rejected because they output **raw class labels**, which require:
  - Complex decoding maps
  - Error-prone post-processing

- **Fine-Tashkeel**  
  Rejected due to:
  - Excessive model size (~3GB)
  - Download and deployment issues

**Decision:**  
➡️ Proceed with **ByT5 diacritization**

---

## 2. Implementation Steps

### Model Loading
- Use:
  - `AutoModelForSeq2SeqLM`
  - `AutoTokenizer`
- Load: `glonor/byt5-arabic-diacritization`

### Inference
- **Input:** Raw Arabic sentence
- **Process:**
```python
model.generate(input_ids)


