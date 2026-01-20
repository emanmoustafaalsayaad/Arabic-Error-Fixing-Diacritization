
import os  # Standard library for operating system interactions
import torch  # PyTorch library for deep learning operations
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # Hugging Face Transformers for loading models
from tqdm import tqdm  # Library for displaying progress bars
import evaluate  # Hugging Face Evaluate library for metrics

# Paths configuration
MODEL_PATH = "./repaired_gec_model_arabart_augmented"  # Directory containing the fine-tuned model
TEST_SENT = "./test/QALB-2015-L2-Test.sent"  # Path to query/source file
TEST_COR = "./test/QALB-2015-L2-Test.cor"  # Path to reference/gold file
OUTPUT_FILE = "qalb_2015_l2_test_predictions_raw.txt"  # Output file name for RAW predictions

def evaluate_model():
    """
    Main function to load the model, generate RAW predictions (with special tokens), and calculate scores.
    """
    print(f"Loading model from {MODEL_PATH}...")  # Notify user
    try:
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        # Load the model
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    except Exception as e:
        # Handle loading errors
        print(f"Error loading model: {e}")
        return

    # Device selection: Use Mac MPS if available, else CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)  # Move model to device
    model.eval()  # Set to evaluation mode

    # Load Data
    print(f"Loading test data from {TEST_SENT}...")
    with open(TEST_SENT, 'r', encoding='utf-8') as f:
        src_lines = f.readlines()  # Read source lines
    
    with open(TEST_COR, 'r', encoding='utf-8') as f:
        ref_lines = f.readlines()  # Read reference lines

    # Preprocess Data
    src_sentences = []
    ref_sentences = []
    
    for src, ref in zip(src_lines, ref_lines):
        # Clean Source: Remove ID prefix
        parts = src.strip().split(maxsplit=1)
        src_text = parts[1] if len(parts) > 1 else ""
        src_sentences.append(src_text)

        # Clean Target: Remove "S " prefix if present
        ref_text = ref.strip()
        if ref_text.startswith("S "):
            ref_text = ref_text[2:]
        ref_sentences.append(ref_text)
    
    print(f"Loaded {len(src_sentences)} sentences.")
    
    # Generate Predictions
    print("Generating predictions...")
    predictions = []
    batch_size = 16  # Batch size for inference
    
    for i in tqdm(range(0, len(src_sentences), batch_size)):
        batch = src_sentences[i:i + batch_size]  # Get current batch
        
        # Tokenize batch
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            # Generate output. Note: These parameters (beams, repetition penalty) affect the quality of generation.
            outputs = model.generate(
                **inputs, 
                max_length=128,
                num_beams=4,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                forced_bos_token_id=tokenizer.bos_token_id
            )
            
        # Decode Key Difference: skip_special_tokens=False
        # This keeps tokens like <s> (start), </s> (end), and <pad> (padding) in the output string.
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        predictions.extend(decoded)
        
    # Calculate Scores
    # Note: Scores will likely be lower here because the Reference text does NOT have special tokens,
    # so metrics like BLEU will count the special tokens in Prediction as "wrong" words.
    
    print("Calculating BLEU...")
    bleu = evaluate.load("sacrebleu")
    bleu_result = bleu.compute(predictions=predictions, references=[[r] for r in ref_sentences])
    print(f"BLEU Score: {bleu_result['score']:.2f}")

    print("Calculating GLEU (Google BLEU)...")
    gleu = evaluate.load("google_bleu")
    gleu_result = gleu.compute(predictions=predictions, references=[[r] for r in ref_sentences])
    gleu_score = gleu_result['google_bleu'] * 100
    print(f"GLEU Score: {gleu_score:.2f}")

    print("Calculating ROUGE (Precision, Recall, F1)...")
    rouge = evaluate.load("rouge")
    rouge_result = rouge.compute(predictions=predictions, references=ref_sentences)
    print(f"ROUGE-1: {rouge_result['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_result['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_result['rougeL']:.4f}")
    
    print("Calculating BERTScore (Precision, Recall, F1)...")
    from bert_score import score
    # BERTScore uses embeddings, but the special tokens might still affect the contextual embedding of the sentence.
    P, R, F1 = score(predictions, ref_sentences, lang="ar", verbose=True)
    print(f"BERTScore P: {P.mean():.4f}")
    print(f"BERTScore R: {R.mean():.4f}")
    print(f"BERTScore F1: {F1.mean():.4f}")
    
    # Save Predictions
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for src, pred, ref in zip(src_sentences, predictions, ref_sentences):
            f.write(f"Source:    {src}\n")
            f.write(f"Ref:       {ref}\n")
            f.write(f"Pred:      {pred}\n")
            f.write("-" * 50 + "\n")
            
    print(f"Predictions saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    evaluate_model()
