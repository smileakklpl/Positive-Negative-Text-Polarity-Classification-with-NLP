import os
import pandas as pd
import numpy as np
import torch
import regex as re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# 1. 準備路徑
TRAIN_PATH = 'data/train_2022.csv'
TEST_PATH = 'data/test_no_answer_2022.csv'
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "result/version3_model"

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = '../data/train_2022.csv'
    TEST_PATH = '../data/test_no_answer_2022.csv'

def refined_clean_for_bert(text):
    """針對 BERT 的清洗：還原縮寫，保留情感標點"""
    if not isinstance(text, str): return ""
    text = text.lower()
    # 還原被拆開的縮寫
    text = re.sub(r"(\w+)\s+t\b", r"\1not", text) 
    text = re.sub(r"i\s+m\b", "i am", text)
    text = re.sub(r"it\s+s\b", "it is", text)
    text = re.sub(r"(\w+)\s+s\b", r"\1s", text)
    text = re.sub(r"(\w+)\s+re\b", r"\1 are", text)
    text = re.sub(r"(\w+)\s+ve\b", r"\1 have", text)
    text = re.sub(r"(\w+)\s+ll\b", r"\1 will", text)
    # 處理重複字母 (sooooo -> soo)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # 標記還原
    text = text.replace("num_num", "number")
    # 清理多餘空格，但保留標點
    text = re.sub(r"\s+", " ", text).strip()
    return text

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "f1": f1_score(labels, predictions, average="binary"),
        "accuracy": accuracy_score(labels, predictions)
    }

def main():
    print("Loading and cleaning data...")
    df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    df['CLEAN_TEXT'] = df['TEXT'].apply(refined_clean_for_bert)
    test_df['CLEAN_TEXT'] = test_df['TEXT'].apply(refined_clean_for_bert)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['CLEAN_TEXT'].tolist(), df['LABEL'].tolist(), test_size=0.15, random_state=42
    )

    print("Tokenizing data...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    def tokenize_function(texts):
        return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

    train_encodings = tokenize_function(train_texts)
    val_encodings = tokenize_function(val_texts)
    test_encodings = tokenize_function(test_df['CLEAN_TEXT'].tolist())

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels=None):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            if self.labels is not None:
                item['labels'] = torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.encodings['input_ids'])

    train_dataset = TextDataset(train_encodings, train_labels)
    val_dataset = TextDataset(val_encodings, val_labels)
    test_dataset = TextDataset(test_encodings)

    print("Initializing model...")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

    # 調整超參數以對抗過擬合
    args_dict = {
        "output_dir": OUTPUT_DIR,
        "num_train_epochs": 4,              # 增加一點 Epoch
        "per_device_train_batch_size": 16,  # 增加 Batch Size 提高穩定性
        "per_device_eval_batch_size": 16,
        "learning_rate": 2e-5,              # 降低學習率 (關鍵!)
        "warmup_ratio": 0.1,                # 使用比例型 Warmup
        "weight_decay": 0.05,               # 提高權重衰減 (正規化)
        "logging_dir": './logs',
        "logging_steps": 50,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "report_to": "none",
    }
    
    try:
        training_args = TrainingArguments(**args_dict, eval_strategy="epoch")
    except TypeError:
        training_args = TrainingArguments(**args_dict, evaluation_strategy="epoch")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting refined training...")
    trainer.train()

    print("Generating predictions...")
    test_preds = trainer.predict(test_dataset)
    predictions = np.argmax(test_preds.predictions, axis=-1)

    submission_path = 'result/submission_v3_refined.csv'
    pd.DataFrame({'row_id': test_df['row_id'], 'LABEL': predictions}).to_csv(submission_path, index=False)
    print(f"Successfully generated '{submission_path}'")

if __name__ == "__main__":
    main()
