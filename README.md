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

