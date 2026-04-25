import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

try:
    import regex as re
except ImportError:
    import re


TRAIN_PATH = "data/train_2022.csv"
TEST_PATH = "data/test_no_answer_2022.csv"
OUTPUT_DIR = "result/version5_model"
SUBMISSION_PATH = "result/submission_v5.csv"
MAX_LENGTH = 192


def parse_args():
    parser = argparse.ArgumentParser(description="Version 5: RoBERTa-large + Pseudo-labeling")
    parser.add_argument("--model_name", type=str, default="roberta-large", help="Hugging Face model id")
    parser.add_argument("--pseudo_label_path", type=str, default="result/submission_v4.csv", help="Path to v4 predictions for pseudo-labeling")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8, help="Smaller batch size for Large model")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Lower LR for Large model")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU training")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_for_transformer(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"(\w+)\s+t\b", r"\1not", text)
    text = re.sub(r"i\s+m\b", "i am", text)
    text = re.sub(r"it\s+s\b", "it is", text)
    text = re.sub(r"(\w+)\s+s\b", r"\1s", text)
    text = re.sub(r"(\w+)\s+re\b", r"\1 are", text)
    text = re.sub(r"(\w+)\s+ve\b", r"\1 have", text)
    text = re.sub(r"(\w+)\s+ll\b", r"\1 will", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = text.replace("num_num", "number")
    text = re.sub(r"\s+", " ", text).strip()
    return text


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights) if self.class_weights is not None else torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1": f1_score(labels, preds, average="binary"),
        "accuracy": accuracy_score(labels, preds),
    }


def build_training_args(output_dir, args, use_cpu=False):
    args_dict = {
        "output_dir": output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size * 2,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": 0.1,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "logging_steps": 50,
        "report_to": "none",
        "seed": args.seed,
        "fp16": not use_cpu, # 啟用 FP16 節省顯存並加速訓練
    }

    # 處理 Transformers 版本相容性
    try:
        return TrainingArguments(**args_dict, eval_strategy="epoch", use_cpu=use_cpu)
    except TypeError:
        try:
            return TrainingArguments(**args_dict, evaluation_strategy="epoch", use_cpu=use_cpu)
        except TypeError:
            return TrainingArguments(**args_dict, evaluation_strategy="epoch", no_cuda=use_cpu)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    use_cpu = args.force_cpu or not torch.cuda.is_available()
    print(f"Runtime device: {'cpu' if use_cpu else 'cuda'}")
    print(f"Using model: {args.model_name}")

    # 1. 讀取原始資料
    train_df = pd.read_csv(TRAIN_PATH if os.path.exists(TRAIN_PATH) else f"../{TRAIN_PATH}")
    test_df = pd.read_csv(TEST_PATH if os.path.exists(TEST_PATH) else f"../{TEST_PATH}")

    train_df["CLEAN_TEXT"] = train_df["TEXT"].apply(clean_for_transformer)
    test_df["CLEAN_TEXT"] = test_df["TEXT"].apply(clean_for_transformer)

    # 2. 讀取偽標籤 (Pseudo-labels)
    pseudo_df = None
    if os.path.exists(args.pseudo_label_path):
        print(f"Loaded pseudo-labels from {args.pseudo_label_path}")
        pseudo_labels = pd.read_csv(args.pseudo_label_path)
        pseudo_df = pd.merge(test_df, pseudo_labels, on="row_id")
    else:
        print(f"[Warning] Pseudo-label file {args.pseudo_label_path} not found. Training without it.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    test_encodings = tokenizer(test_df["CLEAN_TEXT"].tolist(), truncation=True, max_length=MAX_LENGTH)
    test_dataset = TextDataset(test_encodings)

    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    y = train_df["LABEL"].values

    fold_metrics = []
    fold_test_probs = []

    # 3. K-Fold 訓練
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(train_df, y), start=1):
        print(f"\n===== Fold {fold_idx}/{args.num_folds} =====")
        fold_dir = os.path.join(OUTPUT_DIR, f"fold_{fold_idx}")
        
        # 取得當前 Fold 的原始訓練與驗證集
        tr_texts = train_df.iloc[tr_idx]["CLEAN_TEXT"].tolist()
        tr_labels = y[tr_idx].tolist()
        va_texts = train_df.iloc[va_idx]["CLEAN_TEXT"].tolist()
        va_labels = y[va_idx].tolist()

        # 將 Pseudo-labels 加入到訓練集 (不加入驗證集，避免 Data Leakage)
        if pseudo_df is not None:
            tr_texts.extend(pseudo_df["CLEAN_TEXT"].tolist())
            tr_labels.extend(pseudo_df["LABEL"].tolist())

        # 計算當前混合訓練集的 Class Weights
        class_weights_arr = compute_class_weight(class_weight="balanced", classes=np.unique(tr_labels), y=tr_labels)
        class_weights_tensor = torch.tensor(class_weights_arr, dtype=torch.float).to("cuda" if not use_cpu else "cpu")

        tr_enc = tokenizer(tr_texts, truncation=True, max_length=MAX_LENGTH)
        va_enc = tokenizer(va_texts, truncation=True, max_length=MAX_LENGTH)

        trainer = WeightedTrainer(
            model=AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2, ignore_mismatched_sizes=True),
            args=build_training_args(fold_dir, args, use_cpu=use_cpu),
            train_dataset=TextDataset(tr_enc, tr_labels),
            eval_dataset=TextDataset(va_enc, va_labels),
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            class_weights=class_weights_tensor,
        )

        trainer.train()
        fold_metrics.append({"fold": fold_idx, **trainer.evaluate()})
        
        fold_test_probs.append(torch.softmax(torch.tensor(trainer.predict(test_dataset).predictions), dim=1).numpy())

    mean_test_probs = np.mean(np.stack(fold_test_probs, axis=0), axis=0)
    final_preds = np.argmax(mean_test_probs, axis=1)

    pd.DataFrame({"row_id": test_df["row_id"], "LABEL": final_preds}).to_csv(
        SUBMISSION_PATH, index=False
    )

    metrics_df = pd.DataFrame(fold_metrics)
    metrics_path = os.path.join(OUTPUT_DIR, "cv_metrics_v5.csv")
    metrics_df.to_csv(metrics_path, index=False)

    print("\n===== CV Summary =====")
    print(metrics_df[["fold", "eval_loss", "eval_f1", "eval_accuracy"]])
    print("-" * 30)
    if "eval_f1" in metrics_df.columns:
        print(f"Mean F1: {metrics_df['eval_f1'].mean():.4f}")
        print(f"Mean Acc: {metrics_df['eval_accuracy'].mean():.4f}")
    print(f"Saved submission to: {SUBMISSION_PATH}")
    print(f"Saved fold metrics to: {metrics_path}")

if __name__ == "__main__":
    main()