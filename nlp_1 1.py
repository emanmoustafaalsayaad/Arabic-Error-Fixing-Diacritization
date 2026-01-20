import pandas as pd
from datasets import Dataset
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
 
def apply_m2_edits(m2_file_path: str):
    """
    Parses a .m2 file, applies the edits, and returns a list of 
    (incorrect_sentence, corrected_sentence) pairs.

    Args:
        m2_file_path: The path to the .m2 file.

    Returns:
        A list of dictionaries, where each dictionary contains the 
        'incorrect' and 'correct' sentence.
    """
    with open(m2_file_path, 'r', encoding='utf-8') as f:
        entries = f.read().strip().split('\n\n')

    data_pairs = []
    print(f"Processing {len(entries)} entries from {m2_file_path}...")

    for entry in entries:
        lines = entry.split('\n')
        source_sentence = lines[0][2:]  # Remove 'S ' prefix
        
        # Store the original sentence to apply edits on a tokenized copy
        original_source_sentence = source_sentence
        source_tokens = source_sentence.split()

        corrections = [line for line in lines if line.startswith('A')]
        
        if not corrections:
            data_pairs.append({"incorrect": source_sentence, "correct": source_sentence})
            continue

        offset = 0
        for correction in corrections:
            parts = correction.split('|||')
            span = parts[0][2:].split()
            start_idx, end_idx = int(span[0]), int(span[1])
            
            is_deletion = parts[2].strip() == ""

            start_idx += offset
            end_idx += offset

            if start_idx == -1:
                # Handle insertions at the beginning of the sentence
                if end_idx == 0:
                    source_tokens[0:0] = parts[2].split()
                    offset += len(parts[2].split())
                else:
                    # Insert after a specific token
                    source_tokens[end_idx:end_idx] = parts[2].split()
                    offset += len(parts[2].split())
            else: # Substitution or Deletion
                original_tokens_len = end_idx - start_idx
                replacement_tokens = [] if is_deletion else parts[2].split()
                source_tokens[start_idx:end_idx] = replacement_tokens 
                offset += len(replacement_tokens) - original_tokens_len

        corrected_sentence = " ".join(source_tokens)
        data_pairs.append({"incorrect": original_source_sentence, "correct": corrected_sentence})

    return data_pairs

def add_prefix(examples):
    prefix = "gec: "
    examples['incorrect'] = [prefix + text for text in examples['incorrect']]
    return examples

# Define the tokenization function
def tokenize_function(examples, tokenizer):
    # Tokenize the input (incorrect sentences)
    model_inputs = tokenizer(examples["incorrect"], max_length=128, truncation=True, padding="max_length")

    # Tokenize the output (correct sentences)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["correct"], max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_preds, tokenizer):
    """
    Computes SacreBLEU score for the generated predictions.
    """
    sacrebleu = evaluate.load("sacrebleu")
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    # Decode generated summaries, ignoring special tokens
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Sacrebleu expects a list of predictions and a list of lists of references
    result = sacrebleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
    return {"bleu": result["score"]}

def main(args):
    # --- Main Data Loading ---
    print("Loading and preparing data...")
    # Use the more robust .m2 file parser
    # Process one or more .m2 files and combine them
    all_train_data = []
    for m2_file in args.train_m2_file:
        all_train_data.extend(apply_m2_edits(m2_file))
    
    train_df = pd.DataFrame(all_train_data)
    train_dataset = Dataset.from_pandas(train_df)

    # Load development/validation data if provided, otherwise split from training
    if args.dev_m2_file:
        print("Loading dedicated development data...")
        all_dev_data = []
        for m2_file in args.dev_m2_file:
            all_dev_data.extend(apply_m2_edits(m2_file))
        dev_df = pd.DataFrame(all_dev_data)
        dev_dataset = Dataset.from_pandas(dev_df)
    else:
        print("No development data provided. Splitting from training set.")
        train_test_split = train_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = train_test_split['train']
        dev_dataset = train_test_split['test']

    train_dataset = train_dataset.map(add_prefix, batched=True)
    dev_dataset = dev_dataset.map(add_prefix, batched=True)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Development dataset size: {len(dev_dataset)}")
    print("Example:", train_dataset[0])

    # --- Tokenization ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    tokenized_train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_dev_dataset = dev_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # --- Model Training ---
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=True,
        report_to="none",
        push_to_hub=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
    )

    print("\n--- Starting model training ---")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir) # Explicitly save the tokenizer
    print(f"Model saved to {args.output_dir}")

    # --- Evaluation on Test Set ---
    print("\n--- Running evaluation on the test set ---")
    all_test_data = []
    if args.test_m2_file:
        for m2_file in args.test_m2_file:
            all_test_data.extend(apply_m2_edits(m2_file))
    
    test_df = pd.DataFrame(all_test_data)
    test_dataset = Dataset.from_pandas(test_df)

    test_dataset_prefixed = test_dataset.map(add_prefix, batched=True)
    tokenized_test_dataset = test_dataset_prefixed.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=['incorrect', 'correct'])

    print("Predicting on the test set...")
    test_predictions = trainer.predict(tokenized_test_dataset)
    decoded_preds = tokenizer.batch_decode(test_predictions.predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(tokenized_test_dataset["labels"] != -100, tokenized_test_dataset["labels"], tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    print("\n--- Test Set Correction Examples ---")
    for i in range(min(5, len(decoded_preds))):
        print(f"Example {i+1}:")
        # Access original incorrect text from the non-prefixed dataset
        print(f"  Incorrect: {test_dataset[i]['incorrect']}")
        print(f"  Reference: {decoded_labels[i]}")
        print(f"  Model-Corrected: {decoded_preds[i]}\n")

    # Save the corrected sentences to a file if specified
    if args.corrected_test_output_file:
        print(f"\nSaving corrected test sentences to {args.corrected_test_output_file}...")
        with open(args.corrected_test_output_file, 'w', encoding='utf-8') as f_out:
            for sentence in decoded_preds:
                f_out.write(sentence + '\n')
        print(f"Successfully saved {len(decoded_preds)} sentences.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune an AraT5 model for Arabic GEC.")
    parser.add_argument('--model_checkpoint', type=str, default="UBC-NLP/AraT5-Base", help='Hugging Face model checkpoint.')
    parser.add_argument('--train_m2_file', type=str, nargs='+', required=True, help='Path(s) to the training .m2 file(s).')    
    parser.add_argument('--dev_m2_file', type=str, nargs='+', help='Path(s) to the development/validation .m2 file(s).')
    parser.add_argument('--test_m2_file', type=str, nargs='+', help='Path(s) to the test .m2 file(s).')
    parser.add_argument('--output_dir', type=str, default="./my-arabic-gec-model_1", help='Directory to save the fine-tuned model.')
    parser.add_argument('--corrected_test_output_file', type=str, help='Optional path to save the corrected test sentences.')

    # # Set default paths from the original script
    # default_base_path = '/content/drive/MyDrive/Arabic_GEC/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014'
    # parser.set_defaults(
    #     train_m2_file=f'{default_base_path}/train/QALB-2014-L1-Train.m2',
    #     test_m2_file=f'{default_base_path}/dev/QALB-2014-L1-Dev.m2'
    # )

    args = parser.parse_args()
    main(args)
