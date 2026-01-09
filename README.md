## Grammar Correction Model (Seq2Seq) — Implementation Plan

### 🎯 Goal
Build a **sequence-to-sequence (Seq2Seq)** model that automatically corrects grammatical errors in Arabic text by transforming an **incorrect sentence** into its **correct form**.

- **Input:**  
  `ذهب الولد المدرسة`
- **Output:**  
  `ذهب الولد إلى المدرسة`

This component handles **full-sentence grammatical correction**.

---

## 📁 Component Reference

**File:** `correction_model.py`

---

## 1️⃣ `read_text_file(path)`

**Purpose**  
Reads raw text lines from QALB `.sent` or `.cor` files.

**Input**  
- `path`: Path to a text file

**Output**  
A list of sentences (strings)

**Example Output**
```python
[
  "ذهب الولد المدرسة",
  "أكلت التفاحة"
]

---

## 🏆 Model Selection

### ✅ Selected Model
**`glonor/byt5-arabic-diacritization`**

- **Architecture:** ByT5 (Byte-level T5, Seq2Seq)
- **Task:** Text-to-text diacritization
- **Key Advantages:**
  - Consumes raw Arabic text directly
  - Outputs fully diacritized text
  - No need for token-level or character-level decoding logic
  - Robust to spelling variation and OOV tokens
  - Very simple inference pipeline

**Decision:** ✔ Proceed with ByT5

---

### ❌ Rejected Alternatives

- **AraT5 / Shakkala (BERT / RNN-based):**
  - Output class labels instead of text
  - Require complex post-processing and decoding maps

- **Fine-Tashkeel:**
  - Very large model (~3GB)
  - Heavy storage and download overhead
  - Not practical for lightweight pipeline integration

---

## 🧩 Component Design

**File:** `diacritization_model.py`

### 1️⃣ Model Loading
- Use:
  - `AutoTokenizer`
  - `AutoModelForSeq2SeqLM`

### 2️⃣ Inference Procedure
- **Input:** Corrected Arabic sentence (no diacritics)
- **Processing:**  
  `model.generate(input_ids)`
- **Output:**  
  Decoded fully diacritized sentence

### 3️⃣ No Training Required
- This component runs in **inference-only mode**
- No fine-tuning or additional data needed

---

## 🔗 Integration with the Full Pipeline

The complete processing flow is:

