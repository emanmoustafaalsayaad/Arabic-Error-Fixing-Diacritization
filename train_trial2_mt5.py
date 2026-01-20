
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

# Paths to data (2015 Only for Trial 2)
TRAIN_DIR = "/Users/emanismail/Documents/Diploma/NLP/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/train"
DEV_DIR = "/Users/emanismail/Documents/Diploma/NLP/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev"

class QALBDataset(Dataset):
    def __init__(self, sent_path, cor_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        print(f"Loading data from:\n  SENT: {sent_path}\n  COR:  {cor_path}")
        
        with open(sent_path, 'r', encoding='utf-8') as f_sent, \
             open(cor_path, 'r', encoding='utf-8') as f_cor:
            
            sent_lines = f_sent.readlines()
            cor_lines = f_cor.readlines()
            
            assert len(sent_lines) == len(cor_lines), "Mismatch in line counts"
            
            for sent_line, cor_line in zip(sent_lines, cor_lines):
                sent_parts = sent_line.strip().split(maxsplit=1)
                source_text = sent_parts[1] if len(sent_parts) > 1 else ""
                target_text = cor_line.strip()
                if target_text.startswith("S "):
                    target_text = target_text[2:]
                
                self.samples.append((source_text, target_text))
            
        print(f"Total samples loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        source_text, target_text = self.samples[idx]
        model_inputs = self.tokenizer(source_text, max_length=self.max_length, padding="max_length", truncation=True)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(target_text, max_length=self.max_length, padding="max_length", truncation=True)
        model_inputs["labels"] = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels["input_ids"]]
        return model_inputs

def train():
    # TRIAL 2 MODEL: mt5-base
    model_name = "google/mt5-base"
    print(f"Loading model: {model_name}")
    
    # TRIAL 2 FAILURE: Model too large/multilingual -> Hallucinations on small data
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    train_sent = os.path.join(TRAIN_DIR, "QALB-2015-L2-Train.sent")
    train_cor = os.path.join(TRAIN_DIR, "QALB-2015-L2-Train.cor")
    dev_sent = os.path.join(DEV_DIR, "QALB-2015-L2-Dev.sent")
    dev_cor = os.path.join(DEV_DIR, "QALB-2015-L2-Dev.cor")
    
    train_dataset = QALBDataset(train_sent, train_cor, tokenizer)
    eval_dataset = QALBDataset(dev_sent, dev_cor, tokenizer)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results_mt5",
        evaluation_strategy="steps",          
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        learning_rate=5e-5,
        per_device_train_batch_size=2,  # mt5 is larger, smaller batch
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=2,  
        num_train_epochs=3,           
        predict_with_generate=True,
        logging_dir='./logs_mt5',
        logging_steps=50,
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
    
    trainer.train()
    trainer.save_model("./final_gec_model_mt5")

if __name__ == "__main__":
    train()
