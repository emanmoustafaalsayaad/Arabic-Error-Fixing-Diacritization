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
| **BERTScore P** | **0.9082** | 0.7762 | Precision drops in raw output due to "irrelevant" special tokens. |
| **BERTScore R** | 0.8515 | 0.8254 | Recall is slightly better in clean output. |

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
