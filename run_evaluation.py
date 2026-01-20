import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import evaluate
import argparse
import zipfile
import gdown

def download_and_extract_test_data():
    """
    Downloads and extracts the QALB 2014 test set if not already present.
    """
    output_file = 'qalb_dataset.zip'
    test_data_folder = "QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test"

    if os.path.exists(test_data_folder):
        print("✅ Test data already exists.")
        return

    if not os.path.exists(output_file):
        print("📥 Downloading QALB dataset...")
        file_id = '1hvLiiMvvubyCEAZK4KIWgu7qHBNCHOp-'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_file, quiet=False)
        print("✅ Download complete!")

    print("📦 Extracting test data...")
    with zipfile.ZipFile(output_file, 'r') as zip_ref:
        for member in zip_ref.namelist():
            if 'data/2014/test/' in member:
                zip_ref.extract(member, '.')
    print("✅ Test data extracted!")

def run_evaluation(model_path: str, test_sent_path: str, test_cor_path: str, batch_size: int):
    """
    Evaluates the GEC model on the QALB 2014 test set.

    Args:
        model_path (str): Path to the fine-tuned model directory.
        test_sent_path (str): Path to the source sentences test file.
        test_cor_path (str): Path to the target (correct) sentences test file.
        batch_size (int): Batch size for inference.
    """
    # --- 1. Verify data paths ---
    if test_sent_path is None or test_cor_path is None:
        print("ℹ️ Test file paths not provided. Using default QALB 2014 test set.")
        download_and_extract_test_data()
        test_sent_path = "./QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test/QALB-2014-L1-Test.sent"
        test_cor_path = "./QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test/QALB-2014-L1-Test.cor"

    if not os.path.exists(model_path) or not os.path.exists(test_sent_path) or not os.path.exists(test_cor_path):
        print(f"❌ Error: One or more paths not found.")
        print(f"  Model path: {model_path} (Exists: {os.path.exists(model_path)})")
        print(f"  Test source path: {test_sent_path} (Exists: {os.path.exists(test_sent_path)})")
        print(f"  Test target path: {test_cor_path} (Exists: {os.path.exists(test_cor_path)})")
        return

    print(f"✅ Model path: {model_path}")
    print(f"✅ Test data: {test_sent_path}")

    # --- 2. Load Model & Tokenizer ---
    print("\n🔄 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded on {device}")

    # --- 3. Load Test Data ---
    print("\n📖 Loading test data...")
    with open(test_sent_path, 'r', encoding='utf-8') as f:
        src_sentences = [line.strip() for line in f.readlines()]

    with open(test_cor_path, 'r', encoding='utf-8') as f:
        ref_lines = f.readlines()
        ref_sentences = [line.strip() for line in ref_lines]

    print(f"✅ Loaded {len(src_sentences)} test sentences")

    # --- 4. Generate Predictions ---
    print(f"\n🚀 Generating predictions with batch size {batch_size}...")
    predictions = []
    prefix = "gec_arabic: "

    for i in tqdm(range(0, len(src_sentences), batch_size), desc="Evaluating"):
        batch = src_sentences[i:i + batch_size]
        batch_with_prefix = [prefix + sent for sent in batch]

        inputs = tokenizer(batch_with_prefix, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=256, num_beams=5, early_stopping=True)

        batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(batch_predictions)

    print(f"✅ Generated {len(predictions)} predictions")

    # --- 5. Calculate Scores ---
    print("\n📊 Calculating BLEU score...")
    bleu = evaluate.load("sacrebleu")
    bleu_result = bleu.compute(predictions=predictions, references=[[r] for r in ref_sentences])
    bleu_score = bleu_result['score']

    print("📊 Calculating GLEU score...")
    gleu = evaluate.load("google_bleu")
    gleu_result = gleu.compute(predictions=predictions, references=[[r] for r in ref_sentences])
    gleu_score = gleu_result['google_bleu'] * 100

    # --- 6. Display Results & Save ---
    print("\n" + "="*60)
    print("📈 EVALUATION RESULTS")
    print("="*60)
    print(f"BLEU Score:  {bleu_score:.2f}")
    print(f"GLEU Score:  {gleu_score:.2f}")
    print("="*60)

    output_file = "evaluation_predictions.txt"
    print(f"\n💾 Saving predictions to '{output_file}'...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for src, pred, ref in zip(src_sentences, predictions, ref_sentences):
            f.write(f"Source:    {src}\n")
            f.write(f"Predicted: {pred}\n")
            f.write(f"Reference: {ref}\n")
            f.write("-" * 30 + "\n")

    print(f"\n✅ Evaluation complete. Results saved to '{output_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Arabic GEC model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the directory containing the fine-tuned model and tokenizer."
    )
    parser.add_argument(
        "--test_source_file",
        type=str,
        default=None,
        help="Path to the file containing the source sentences for testing."
    )
    parser.add_argument(
        "--test_target_file",
        type=str,
        default=None,
        help="Path to the file containing the target (correct) sentences for testing."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for generating predictions."
    )
    args = parser.parse_args()
    run_evaluation(args.model_path, args.test_source_file, args.test_target_file, args.batch_size)
