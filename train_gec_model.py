
import os
import re
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

# Paths to data
TRAIN_DIR = "/Users/emanismail/Documents/Diploma/NLP/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/train"
DEV_DIR = "/Users/emanismail/Documents/Diploma/NLP/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev"

class QALBDataset(Dataset):
    def __init__(self, sent_paths, cor_paths, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # Determine if input is single path (str) or list
        if isinstance(sent_paths, str):
            sent_paths = [sent_paths]
        if isinstance(cor_paths, str):
            cor_paths = [cor_paths]
            
        assert len(sent_paths) == len(cor_paths), "Mismatch in number of sent/cor files"

        for sent_p, cor_p in zip(sent_paths, cor_paths):
            print(f"Loading data from:\n  SENT: {sent_p}\n  COR:  {cor_p}")
            
            with open(sent_p, 'r', encoding='utf-8') as f_sent, \
                 open(cor_p, 'r', encoding='utf-8') as f_cor:
                
                sent_lines = f_sent.readlines()
                cor_lines = f_cor.readlines()
                
                assert len(sent_lines) == len(cor_lines), f"Mismatch in line counts for {sent_p}: {len(sent_lines)} vs {len(cor_lines)}"
                
                for sent_line, cor_line in zip(sent_lines, cor_lines):
                    # Process Source (.sent): Remove the ID (first token)
                    sent_parts = sent_line.strip().split(maxsplit=1)
                    if len(sent_parts) > 1:
                        source_text = sent_parts[1]
                    else:
                        source_text = "" 

                    # Process Target (.cor): Remove the 'S ' prefix if present
                    target_text = cor_line.strip()
                    if target_text.startswith("S "):
                        target_text = target_text[2:]
                    elif target_text == "S": 
                        target_text = ""
                    
                    self.samples.append((source_text, target_text))
                
        print(f"Total samples loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        source_text, target_text = self.samples[idx]
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            source_text, 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True
        )
        
        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_text, 
                max_length=self.max_length, 
                padding="max_length", 
                truncation=True
            )
            
        model_inputs["labels"] = labels["input_ids"]
        
        # Replace padding token id in labels with -100 so they are ignored by the loss
        model_inputs["labels"] = [
            (l if l != self.tokenizer.pad_token_id else -100) for l in model_inputs["labels"]
        ]
        
        return model_inputs

from transformers.trainer_utils import get_last_checkpoint

def train():
    model_name = "moussaKam/AraBART"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Create Datasets
    # 2015 Data
    train_sent_2015 = os.path.join(TRAIN_DIR, "QALB-2015-L2-Train.sent")
    train_cor_2015 = os.path.join(TRAIN_DIR, "QALB-2015-L2-Train.cor")
    
    # 2014 Data (Augmentation)
    # TRAIN_DIR is .../data/2015/train -> go up to data root
    data_root = os.path.dirname(os.path.dirname(TRAIN_DIR))
    train_sent_2014 = os.path.join(data_root, "2014/train/QALB-2014-L1-Train.sent")
    train_cor_2014 = os.path.join(data_root, "2014/train/QALB-2014-L1-Train.cor")
    
    # Combined Paths
    train_sents = [train_sent_2015, train_sent_2014]
    train_cors = [train_cor_2015, train_cor_2014]

    dev_sent = os.path.join(DEV_DIR, "QALB-2015-L2-Dev.sent")
    dev_cor = os.path.join(DEV_DIR, "QALB-2015-L2-Dev.cor")
    
    train_dataset = QALBDataset(train_sents, train_cors, tokenizer)
    eval_dataset = QALBDataset(dev_sent, dev_cor, tokenizer)
    
    # Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results_augmented", # New output dir for augmented run
        eval_strategy="steps",          
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        learning_rate=5e-5,            # Slightly higher LR for larger data
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=8, 
        per_device_eval_batch_size=1,  
        gradient_checkpointing=True,   
        optim="adafactor",             
        weight_decay=0.01,
        save_total_limit=2,            
        num_train_epochs=30,           
        predict_with_generate=True,
        logging_dir='./logs',
        logging_steps=100,
        load_best_model_at_end=True,   
        metric_for_best_model="loss",
        push_to_hub=False,
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Check for existing checkpoints to resume from
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint:
        print(f"Resuming training from checkpoint: {last_checkpoint}")
    else:
        print("Starting training from scratch")

    print("Starting training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # Save final model
    trainer.save_model("./final_gec_model_arabart_augmented")
    print("Model saved to ./final_gec_model_arabart_augmented")

if __name__ == "__main__":
    # verification...
    # ...
    train() 
