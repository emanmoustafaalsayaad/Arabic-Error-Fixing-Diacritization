
import os
import re
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers.trainer_utils import get_last_checkpoint

# =============================================================================
# CONFIGURATION & PATHS
# =============================================================================
# Define the main directories for training and validation data.
TRAIN_DIR = "/Users/emanismail/Documents/Diploma/NLP/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/train"
DEV_DIR = "/Users/emanismail/Documents/Diploma/NLP/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev"

# =============================================================================
# CUSTOM DATASET CLASS
# =============================================================================
class QALBDataset(Dataset):
    """
    Custom PyTorch Dataset to handle loading and tokenizing QALB data.
    It pairs source sentences (.sent) with correction sentences (.cor).
    """
    def __init__(self, sent_paths, cor_paths, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # Support loading from a single string path or a list of paths (for augmentation)
        if isinstance(sent_paths, str):
            sent_paths = [sent_paths]
        if isinstance(cor_paths, str):
            cor_paths = [cor_paths]
            
        # sanity check: input and label files must match in count
        assert len(sent_paths) == len(cor_paths), "Mismatch in number of sent/cor files"

        # Loop through each pair of files (e.g., 2015 data, then 2014 data)
        for sent_p, cor_p in zip(sent_paths, cor_paths):
            print(f"Loading data from:\n  SENT: {sent_p}\n  COR:  {cor_p}")
            
            with open(sent_p, 'r', encoding='utf-8') as f_sent, \
                 open(cor_p, 'r', encoding='utf-8') as f_cor:
                
                sent_lines = f_sent.readlines()
                cor_lines = f_cor.readlines()
                
                # Ensure line counts match within the files themselves
                assert len(sent_lines) == len(cor_lines), f"Mismatch in line counts for {sent_p}"
                
                # Parse each line
                for sent_line, cor_line in zip(sent_lines, cor_lines):
                    # Data Cleaning: 
                    # 1. Source file (.sent) lines often start with an ID (e.g., "S001_T02 word word...")
                    #    We split by space and take everything after the first token.
                    sent_parts = sent_line.strip().split(maxsplit=1)
                    if len(sent_parts) > 1:
                        source_text = sent_parts[1]
                    else:
                        source_text = "" 

                    # 2. Target file (.cor) lines often start with "S " (e.g., "S Corrected sentence...")
                    target_text = cor_line.strip()
                    if target_text.startswith("S "):
                        target_text = target_text[2:]
                    elif target_text == "S": 
                        target_text = ""
                    
                    # Store the clean pair
                    self.samples.append((source_text, target_text))
                
        print(f"Total samples loaded: {len(self.samples)}")

    def __len__(self):
        # Required by PyTorch to know exactly how big the dataset is
        return len(self.samples)

    def __getitem__(self, idx):
        # Required by PyTorch to fetch a single data point by index
        source_text, target_text = self.samples[idx]
        
        # 1. Tokenize the erroneous input sentence
        model_inputs = self.tokenizer(
            source_text, 
            max_length=self.max_length, 
            padding="max_length",  # Pad short sentences to max_length
            truncation=True        # Cut long sentences
        )
        
        # 2. Tokenize the correct target sentence
        # Note: We use as_target_tokenizer context manager for correct label processing
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_text, 
                max_length=self.max_length, 
                padding="max_length", 
                truncation=True
            )
            
        model_inputs["labels"] = labels["input_ids"]
        
        # 3. Mask padding tokens in the labels
        # We replace the padding token ID with -100. This tells the Loss Function (CrossEntropy)
        # to IGNORE these tokens so we don't learn to predict "padding".
        model_inputs["labels"] = [
            (l if l != self.tokenizer.pad_token_id else -100) for l in model_inputs["labels"]
        ]
        
        return model_inputs

# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================
def train():
    # 1. Model Selection
    # AraBART is a Sequence-to-Sequence model pretrained specifically on Arabic text.
    # It understands Arabic grammar and semantic structure.
    model_name = "moussaKam/AraBART"
    print(f"Loading model: {model_name}")
    
    # Load tokenizer and model from Hugging Face Hub
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # 2. Dataset Preparation
    
    # Path setup for 2015 Data (The main task data)
    train_sent_2015 = os.path.join(TRAIN_DIR, "QALB-2015-L2-Train.sent")
    train_cor_2015 = os.path.join(TRAIN_DIR, "QALB-2015-L2-Train.cor")
    
    # Path setup for 2014 Data (Used for Augmentation)
    # This increases our training size from ~300 to ~20,000 sentences.
    data_root = os.path.dirname(os.path.dirname(TRAIN_DIR))
    train_sent_2014 = os.path.join(data_root, "2014/train/QALB-2014-L1-Train.sent")
    train_cor_2014 = os.path.join(data_root, "2014/train/QALB-2014-L1-Train.cor")
    
    # Combine paths into lists
    train_sents = [train_sent_2015, train_sent_2014]
    train_cors = [train_cor_2015, train_cor_2014]

    # Validation (Dev) Data
    dev_sent = os.path.join(DEV_DIR, "QALB-2015-L2-Dev.sent")
    dev_cor = os.path.join(DEV_DIR, "QALB-2015-L2-Dev.cor")
    
    # Instantiate Datasets
    train_dataset = QALBDataset(train_sents, train_cors, tokenizer)
    eval_dataset = QALBDataset(dev_sent, dev_cor, tokenizer)
    
    # 3. Training Hyperparameters
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results_augmented", # Checkpoints save here
        
        # Evaluation Strategy: Check loss every 500 steps
        eval_strategy="steps",          
        eval_steps=500,
        
        # Save Strategy: Save checkpoint every 500 steps
        save_strategy="steps",
        save_steps=500,
        
        # Optimizer & Learning Rate
        learning_rate=5e-5,            # Standard fine-tuning rate
        optim="adafactor",             # Memory-efficient optimizer (Critical for Mac M1/M2/M3)
        weight_decay=0.01,
        
        # Batch Size & Accumulation (Fit in memory)
        per_device_train_batch_size=1, # Very small batch size to avoid OOM
        gradient_accumulation_steps=8, # Simulate batch_size=8 by accumulating gradients
        per_device_eval_batch_size=1,  
        
        # Gradient Checkpointing (Saves memory by recomputing activations)
        gradient_checkpointing=True,   
        
        # Checkpoint Management
        save_total_limit=2,            # Keep only last 2 checkpoints
        num_train_epochs=30,           # Train longer (30 epochs) because data is complex
        predict_with_generate=True,    # Use generate() for validation metrics
        
        # Logging
        logging_dir='./logs',
        logging_steps=100,
        load_best_model_at_end=True,   # At end of training, revert to best checkpoint
        metric_for_best_model="loss",
        push_to_hub=False,
    )
    
    # Data Collator: Handles batching dynamically
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # 4. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 5. Check for Resuming
    # If the script crashed previously, resume from where we left off.
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint:
        print(f"Resuming training from checkpoint: {last_checkpoint}")
    else:
        print("Starting training from scratch")

    # 6. Start Training
    print("Starting training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # 7. Save Final Model
    trainer.save_model("./final_gec_model_arabart_augmented")
    print("Model saved to ./final_gec_model_arabart_augmented")

if __name__ == "__main__":
    train()     
