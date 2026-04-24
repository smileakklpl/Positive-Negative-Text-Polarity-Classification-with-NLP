import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
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
OUTPUT_DIR = "result/version4_model"
SUBMISSION_PATH = "result/submission_v4.csv"
MAX_LENGTH = 192


def parse_args():
    parser = argparse.ArgumentParser(description="Version 4: HF model + K-Fold ensemble")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Hugging Face model id")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
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


def is_cuda_usable():
    if not torch.cuda.is_available():
        return False
    try:
        test_tensor = torch.tensor([1.0], device="cuda")
        _ = (test_tensor * 2).cpu()
        return True
    except Exception as e:
        print(f"[Warning] CUDA is available but not usable. Fallback to CPU. Detail: {e}")
        return False


def build_training_args(output_dir, args, use_cpu=False):
    args_dict = {
        "output_dir": output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
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
    }

    if use_cpu:
        try:
            return TrainingArguments(**args_dict, eval_strategy="epoch", use_cpu=True)
        except TypeError:
            try:
                return TrainingArguments(**args_dict, evaluation_strategy="epoch", use_cpu=True)
            except TypeError:
                try:
                    return TrainingArguments(**args_dict, eval_strategy="epoch", no_cuda=True)
                except TypeError:
                    return TrainingArguments(**args_dict, evaluation_strategy="epoch", no_cuda=True)

    try:
        return TrainingArguments(**args_dict, eval_strategy="epoch")
    except TypeError:
        return TrainingArguments(**args_dict, evaluation_strategy="epoch")


def get_data_paths():
    train_path = TRAIN_PATH
    test_path = TEST_PATH
    if not os.path.exists(train_path):
        train_path = "../data/train_2022.csv"
        test_path = "../data/test_no_answer_2022.csv"
    return train_path, test_path


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    use_cpu = args.force_cpu or (not is_cuda_usable())
    print(f"Runtime device: {'cpu' if use_cpu else 'cuda'}")

    print(f"Using model: {args.model_name}")
    train_path, test_path = get_data_paths()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df["CLEAN_TEXT"] = train_df["TEXT"].apply(clean_for_transformer)
    test_df["CLEAN_TEXT"] = test_df["TEXT"].apply(clean_for_transformer)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    test_encodings = tokenizer(
        test_df["CLEAN_TEXT"].tolist(),
        truncation=True,
        max_length=MAX_LENGTH,
    )
    test_dataset = TextDataset(test_encodings)

    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    y = train_df["LABEL"].values

    fold_metrics = []
    fold_test_probs = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(train_df, y), start=1):
        print(f"\n===== Fold {fold_idx}/{args.num_folds} =====")
        fold_dir = os.path.join(OUTPUT_DIR, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        tr_texts = train_df.iloc[tr_idx]["CLEAN_TEXT"].tolist()
        va_texts = train_df.iloc[va_idx]["CLEAN_TEXT"].tolist()
        tr_labels = y[tr_idx]
        va_labels = y[va_idx]

        tr_enc = tokenizer(tr_texts, truncation=True, max_length=MAX_LENGTH)
        va_enc = tokenizer(va_texts, truncation=True, max_length=MAX_LENGTH)

        tr_ds = TextDataset(tr_enc, tr_labels)
        va_ds = TextDataset(va_enc, va_labels)

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )

        trainer = Trainer(
            model=model,
            args=build_training_args(fold_dir, args, use_cpu=use_cpu),
            train_dataset=tr_ds,
            eval_dataset=va_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        trainer.train()

        eval_result = trainer.evaluate()
        fold_metrics.append(
            {
                "fold": fold_idx,
                "eval_f1": eval_result.get("eval_f1", np.nan),
                "eval_accuracy": eval_result.get("eval_accuracy", np.nan),
                "eval_loss": eval_result.get("eval_loss", np.nan),
            }
        )
        print(
            f"Fold {fold_idx} -> F1: {fold_metrics[-1]['eval_f1']:.4f}, "
            f"Acc: {fold_metrics[-1]['eval_accuracy']:.4f}"
        )

        test_pred = trainer.predict(test_dataset)
        logits = torch.tensor(test_pred.predictions)
        probs = torch.softmax(logits, dim=1).numpy()
        fold_test_probs.append(probs)

    mean_test_probs = np.mean(np.stack(fold_test_probs, axis=0), axis=0)
    final_preds = np.argmax(mean_test_probs, axis=1)

    pd.DataFrame({"row_id": test_df["row_id"], "LABEL": final_preds}).to_csv(
        SUBMISSION_PATH, index=False
    )

    metrics_df = pd.DataFrame(fold_metrics)
    metrics_path = os.path.join(OUTPUT_DIR, "cv_metrics_v4.csv")
    metrics_df.to_csv(metrics_path, index=False)

    print("\n===== CV Summary =====")
    print(metrics_df)
    print("-" * 30)
    print(f"Mean F1: {metrics_df['eval_f1'].mean():.4f}")
    print(f"Mean Acc: {metrics_df['eval_accuracy'].mean():.4f}")
    print(f"Saved submission to: {SUBMISSION_PATH}")
    print(f"Saved fold metrics to: {metrics_path}")


if __name__ == "__main__":
    main()