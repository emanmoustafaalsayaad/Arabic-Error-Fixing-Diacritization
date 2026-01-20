
import os  # Standard library for operating system interactions (e.g., file paths)
import torch  # PyTorch library for deep learning operations and tensor manipulation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # Hugging Face Transformers for loading pre-trained models and tokenizers
from tqdm import tqdm  # Library for displaying progress bars during loops
import evaluate  # Hugging Face Evaluate library for calculating metrics like BLEU and ROUGE

# Paths configuration
MODEL_PATH = "./repaired_gec_model_arabart_augmented"  # Directory containing the fine-tuned model and tokenizer
TEST_SENT = "./test/QALB-2015-L2-Test.sent"  # Path to the source (input) text file for testing
TEST_COR = "./test/QALB-2015-L2-Test.cor"  # Path to the reference (corrected) text file for evaluation
OUTPUT_FILE = "qalb_2015_l2_test_predictions_clean.txt"  # File path where predictions will be saved

def evaluate_model():
    """
    Main function to load the model, generate predictions, and calculate evaluation scores.
    """
    print(f"Loading model from {MODEL_PATH}...")  # Notify user that model loading is starting
    try:
        # Load the tokenizer associated with the pre-trained model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        # Load the Sequence-to-Sequence language model (e.g., BART, T5)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    except Exception as e:
        # Catch and print any errors that occur during loading (e.g., path not found)
        print(f"Error loading model: {e}")
        return  # Exit the function if loading fails

    # Determine the computing device: use MPS (Metal Performance Shaders) for Mac if available, otherwise CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")  # Print the device being used
    model = model.to(device)  # Move the model to the selected device (GPU/CPU)
    model.eval()  # Set the model to evaluation mode (disables dropout, moving averages, etc.)

    # Load Data
    print(f"Loading test data from {TEST_SENT}...")  # Notify user that data loading is starting
    # Open the source sentences file for reading with UTF-8 encoding
    with open(TEST_SENT, 'r', encoding='utf-8') as f:
        src_lines = f.readlines()  # Read all lines from the source file into a list
    
    # Open the reference sentences file for reading with UTF-8 encoding
    with open(TEST_COR, 'r', encoding='utf-8') as f:
        ref_lines = f.readlines()  # Read all lines from the reference file into a list

    # Preprocess Data
    src_sentences = []  # Initialize empty list for processed source sentences
    ref_sentences = []  # Initialize empty list for processed reference sentences
    
    # Iterate through pairs of source and reference lines
    for src, ref in zip(src_lines, ref_lines):
        # Process Source: Input format is often "ID text". We need to strip the ID.
        parts = src.strip().split(maxsplit=1)  # Split the line into at most 2 parts using whitespace: [ID, Text]
        src_text = parts[1] if len(parts) > 1 else ""  # Take the text part (index 1), or empty string if no text
        src_sentences.append(src_text)  # Add the clean source text to the list

        # Process Target: Reference format might start with "S ".
        ref_text = ref.strip()  # Remove leading/trailing whitespace from reference
        if ref_text.startswith("S "):  # Check if the line starts with the strict "S " prefix
            ref_text = ref_text[2:]  # Remove the first 2 characters ("S ")
        ref_sentences.append(ref_text)  # Add the clean reference text to the list
    
    print(f"Loaded {len(src_sentences)} sentences.")  # Print the total number of sentences loaded
    
    # Generate Predictions
    print("Generating predictions...")  # Notify start of generation phase
    predictions = []  # Initialize empty list to store model predictions
    batch_size = 16 # Set batch size (number of sentences to process at once). Adjust down if out of memory.
    
    # Loop over the source sentences in chunks of `batch_size`, showing a progress bar with tqdm
    for i in tqdm(range(0, len(src_sentences), batch_size)):
        batch = src_sentences[i:i + batch_size]  # Slice the list to get the current batch of sentences
        
        # Tokenize the batch: 
        # padding=True: Pad sequences to the longest in the batch
        # truncation=True: Truncate sequences longer than max_length
        # return_tensors="pt": Return PyTorch tensors
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        
        # Disable gradient calculation for inference (saves memory and computation)
        with torch.no_grad():
            # Generate output sequences using the model
            outputs = model.generate(
                **inputs,  # Unpack input tensors (input_ids, attention_mask)
                max_length=128,  # Maximum length of the generated sequence
                num_beams=4,  # Use beam search with width 4 (explores 4 likely paths)
                repetition_penalty=1.2,  # Penalize repeating tokens to reduce getting stuck in loops
                no_repeat_ngram_size=3,  # Prevent repeating any 3-gram (3 consecutive words)
                forced_bos_token_id=tokenizer.bos_token_id  # Force the sequence to start with the BOS token
            )
            
        # Decode the generated token IDs back into text
        # skip_special_tokens=True: THIS IS THE "CLEAN" PART. It removes <s>, </s>, <pad> tokens automatically.
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)  # Add the decoded text batch to the all_predictions list
        
    # Calculate Scores
    # 1. BLEU (Bilingual Evaluation Understudy) - Standard machine translation metric
    print("Calculating BLEU...")
    bleu = evaluate.load("sacrebleu")  # Load the official SacreBLEU implementation
    # Compute BLEU. Note: references must be a list of lists (allowing multiple refs per sentence)
    bleu_result = bleu.compute(predictions=predictions, references=[[r] for r in ref_sentences])
    print(f"BLEU Score: {bleu_result['score']:.2f}")  # Print formatted BLEU score

    # 2. GLEU (Google BLEU) - Often better for single sentence evaluation
    print("Calculating GLEU (Google BLEU)...")
    gleu = evaluate.load("google_bleu")  # Load Google BLEU metric
    gleu_result = gleu.compute(predictions=predictions, references=[[r] for r in ref_sentences])
    gleu_score = gleu_result['google_bleu'] * 100  # Convert 0-1 score to 0-100 scale
    print(f"GLEU Score: {gleu_score:.2f}")

    # 3. ROUGE (Recall-Oriented Understudy for Gisting Evaluation) - Measures n-gram recall
    print("Calculating ROUGE (Precision, Recall, F1)...")
    rouge = evaluate.load("rouge")  # Load ROUGE metric
    rouge_result = rouge.compute(predictions=predictions, references=ref_sentences)
    print(f"ROUGE-1: {rouge_result['rouge1']:.4f}")  # Unigram overlap
    print(f"ROUGE-2: {rouge_result['rouge2']:.4f}")  # Bigram overlap
    print(f"ROUGE-L: {rouge_result['rougeL']:.4f}")  # Longest Common Subsequence

    # 4. BERTScore - Measures semantic similarity using BERT embeddings
    print("Calculating BERTScore (Precision, Recall, F1)...")
    from bert_score import score  # Import score function from bert_score library
    # Compute P, R, F1. lang="ar" ensures the right multilingual model is used. verbose=True shows progress.
    P, R, F1 = score(predictions, ref_sentences, lang="ar", verbose=True)
    print(f"BERTScore P: {P.mean():.4f}")  # Print mean Precision
    print(f"BERTScore R: {R.mean():.4f}")  # Print mean Recall
    print(f"BERTScore F1: {F1.mean():.4f}")  # Print mean F1 Score
    
    # Save Predictions to File
    # Open output file for writing
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # Iterate through source, prediction, and reference simultaneously
        for src, pred, ref in zip(src_sentences, predictions, ref_sentences):
            f.write(f"Source:    {src}\n")  # Write source
            f.write(f"Ref:       {ref}\n")  # Write reference
            f.write(f"Pred:      {pred}\n")  # Write prediction
            f.write("-" * 50 + "\n")  # Write a separator line
            
    print(f"Predictions saved to {OUTPUT_FILE}")  # Confirm file save

if __name__ == "__main__":
    evaluate_model()  # Execute the main function if script is run directly
