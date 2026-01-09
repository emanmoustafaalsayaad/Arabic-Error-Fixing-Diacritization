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

