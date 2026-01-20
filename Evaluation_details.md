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

## How to Run
To reproduce these results:

```bash
# For clean evaluation
python3 evaluate_model_clean.py

# For raw evaluation
python3 evaluate_model_raw.py
```


---

# Evaluation Scripts Details

This document provides a detailed walkthrough of the two evaluation scripts used to assess the Arabic GEC model.

## 1. Overview

There are two scripts:
1.  **`evaluate_model_clean.py`**: Produces clean, human-readable text. Used for standard scoring.
2.  **`evaluate_model_raw.py`**: Produces raw tokenizer output. Used for debugging boundaries and token generation.

Both scripts follow the same logic pipeline:
1.  **Load Resources**: Model, Tokenizer, and Data.
2.  **Preprocessing**: Clean input/reference text (remove IDs).
3.  **Generation**: Run the model on source sentences.
4.  **Scoring**: Compare generation vs reference using BLEU, GLEU, ROUGE, and BERTScore.
5.  **Saving**: Write results to a file.

---

## 2. Code Walkthrough

### Imports and Setup
```python
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
```
- **`torch`**: The engine for running the neural network.
- **`transformers`**: Hugging Face library to easily load the "AraBART" model.
- **`evaluate`**: Library that implements standard metrics like BLEU and ROUGE so we don't have to write the math ourselves.

### Data Loading
We load the Test Source (`.sent`) and Test Correction (`.cor`) files.
```python
# Cleaning the input data
parts = src.strip().split(maxsplit=1)
src_text = parts[1]
```
The raw data files contain IDs like `S001_...` at the start of every line. We strip these out so the model doesn't try to "correct" the ID itself.

### Model Generation (The Core)
```python
outputs = model.generate(
    **inputs, 
    max_length=128,          # Don't generate endless text
    num_beams=4,             # Beam Search: explore 4 possibilities together (better than picking just the top word)
    repetition_penalty=1.2,  # Discourage the model from saying "book book book"
    no_repeat_ngram_size=3,  # Hard rule: never repeat a 3-word phrase
    forced_bos_token_id=...  # Force start-of-sentence token
)
```
These parameters control "how" the model writes. 
- **High penalty/no_repeat** makes it less repetitive.
- **Num beams** makes it smarter but slower.

### Key Difference: Decoding
This is where the two scripts diverge.

**In `evaluate_model_clean.py`:**
```python
decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
```
- `skip_special_tokens=True` tells the tokenizer to REMOVE functionality tokens.
- Result: `ذهبت إلى المدرسة` (Clean text)

**In `evaluate_model_raw.py`:**
```python
decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
```
- `skip_special_tokens=False` keeps EVERYTHING.
- Result: `<s> ذهبت إلى المدرسة </s> <pad> <pad>`
- **`<s>`**: Start of Sentence.
- **`</s>`**: End of Sentence.
- **`<pad>`**: Padding (empty space to make all sentences in a batch the same length).

---

## 3. Metrics Explanation

| Metric | What it measures | Note |
| :--- | :--- | :--- |
| **BLEU** | n-gram overlap | Standard for translation. Checks if 1, 2, 3, or 4-word phrases match the reference. |
| **GLEU** | Google-BLEU | similar to BLEU but often correlates better with human judgement on individual sentences. |
| **ROUGE** | Recall | Checks "how much of the reference did the model manage to capture?". Important for summarization and correction. |
| **BERTScore** | **Meaning** (Semantics) | Uses a whole separate AI (BERT) to check if the *meaning* is similar, even if the exact words are different. **This is usually the most important metric for advanced models.** |

## 4. Usage

Run the scripts from the terminal:
```bash
python3 evaluate_model_clean.py
python3 evaluate_model_raw.py
```
