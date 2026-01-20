## 📂 Project Script Registry
Here is a guide to every script in this folder and what it does.

| Script Name | Purpose | Status |
| :--- | :--- | :--- |
| **`train_gec_model.py`** | **The Final Model.** Trains AraBART on 20k sentences (Augmented). | ✅ **Use this.** |
| `train_trial1_arat5.py` | Trial 1 (Failed). Implementation of AraT5-base that failed due to vocab mismatch. | ❌ Archive. |
| `train_trial2_mt5.py` | Trial 2 (Failed). Implementation of mT5-base that failed due to hallucinations. | ❌ Archive. |
| `train_trial3_arabart_broken.py` | Trial 3 (Failed). AraBART implementation **without** weight repair (broken output). | ❌ Archive. |
| `repair_model.py` | **The Surgeon.** Fixes the "Missing Embeddings" bug in checkpoints. | 🔧 Utility. |
| `verify_model.py` | **Quick Check.** Runs the model on sample sentences to see output. | 🧪 Testing. |
| `evaluate_model_clean.py` | **Official Evaluation.** Calculates BLEU/GLEU/ROUGE on clean output. | 📊 Metrics. |
| `evaluate_model_raw.py` | Debug Evaluation. Shows raw model output (with hallucinations). | 🐞 Debug. |
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

