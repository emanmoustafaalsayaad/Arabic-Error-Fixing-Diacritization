import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys


def generate_correction(sentence: str, model, tokenizer):
    """
    Uses the fine-tuned model to correct a given Arabic sentence.
    """
    # Add the prefix and tokenize
    input_text = f"correct: {sentence}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)

    # Generate the corrected sentence
    output_sequences = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=128,
        num_beams=5,
        early_stopping=True,
    )

    # Decode and return the result
    corrected_sentence = tokenizer.decode(
        output_sequences[0], skip_special_tokens=True
    )
    return corrected_sentence


def main():
    parser = argparse.ArgumentParser(description="Correct Arabic sentences from a file or a single sentence using a fine-tuned model.")
    
    # --- Input Arguments ---
    # Make input arguments mutually exclusive so the user provides either a sentence or a file
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input_sentence", type=str, help="A single incorrect sentence to correct."
    )
    group.add_argument(
        "--input_file",
        type=str,
        help="Path to a file containing incorrect sentences (one per line).",
    )

    # --- Model and Output Arguments ---
    parser.add_argument(
        "--model_dir",
        type=str,
        default="UBC-NLP/AraT5-Base",
        help="Directory of the fine-tuned model.",
    )
    parser.add_argument(
        "--output_file", type=str, help="Optional path to save the corrected sentences."
    )
    
    args = parser.parse_args()

    # --- Load Model and Tokenizer ---
    print(f"Loading model from: {args.model_dir}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
        print("Model loaded successfully.")
    except OSError:
        print(f"Error: Model not found at {args.model_dir}", file=sys.stderr)
        print(
            "Please ensure you have trained the model using nlp.py and provided the correct path.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Process Input ---
    if args.input_sentence:
        # Handle a single sentence
        corrected = generate_correction(args.input_sentence, model, tokenizer)
        print("\n--- Correction Result ---")
        print(f"Incorrect: {args.input_sentence}")
        print(f"Corrected: {corrected}")

    elif args.input_file:
        # Handle a file of sentences
        from tqdm import tqdm

        try:
            with open(args.input_file, 'r', encoding='utf-8') as f_in:
                incorrect_sentences = [line.strip() for line in f_in if line.strip()]
            
            print(f"\nFound {len(incorrect_sentences)} sentences to correct in {args.input_file}.")
            
            corrected_sentences = []
            for sentence in tqdm(incorrect_sentences, desc="Correcting sentences"):
                corrected = generate_correction(sentence, model, tokenizer)
                corrected_sentences.append(corrected)

            # Save to output file if specified
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f_out:
                    for sent in corrected_sentences:
                        f_out.write(sent + '\n')
                print(f"\nSaved {len(corrected_sentences)} corrected sentences to {args.output_file}.")
            else:
                # Otherwise, print to console
                print("\n--- Correction Results ---")
                for inc, cor in zip(incorrect_sentences, corrected_sentences):
                    print(f"Incorrect: {inc}\nCorrected: {cor}\n")

        except FileNotFoundError:
            print(f"Error: Input file not found at {args.input_file}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()