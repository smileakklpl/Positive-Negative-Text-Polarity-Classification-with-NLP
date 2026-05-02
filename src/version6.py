import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

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
DEFAULT_OUTPUT_DIR = "result/version6_model"
DEFAULT_SUBMISSION_PATH = "result/submission_v6.csv"
MAX_LENGTH = 192


@dataclass
class PseudoSelectionResult:
    selected_indices: np.ndarray
    selected_probs: np.ndarray
    selected_labels: np.ndarray
    diagnostics: pd.DataFrame


def parse_args():
    parser = argparse.ArgumentParser(
        description="Version 6: Two-stage self-training with confidence-filtered soft pseudo labels"
    )
    parser.add_argument("--model_name", type=str, default="roberta-large", help="Hugging Face model id")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--epochs_stage1", type=int, default=2)
    parser.add_argument("--epochs_stage2", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_cpu", action="store_true")

    parser.add_argument("--confidence_threshold", type=float, default=0.92)
    parser.add_argument("--margin_threshold", type=float, default=0.35)
    parser.add_argument("--max_pseudo_per_class", type=int, default=3000)
    parser.add_argument("--pseudo_loss_weight", type=float, default=0.5)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--threshold_grid_step", type=float, default=0.01)

    parser.add_argument("--disable_class_weights", action="store_true")
    parser.add_argument("--disable_fp16", action="store_true",
                        help="Disable FP16 training (required for DeBERTa-v3)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--submission_path", type=str, default=DEFAULT_SUBMISSION_PATH)
    parser.add_argument(
        "--reuse_stage1",
        action="store_true",
        help="Skip Stage 1 training and reuse saved stage1 OOF/test probs from OUTPUT_DIR",
    )
    parser.add_argument(
        "--run_stage3",
        action="store_true",
        help="Run Stage 3: re-generate pseudo labels from Stage 2 predictions and train a third round",
    )
    parser.add_argument("--epochs_stage3", type=int, default=3)
    parser.add_argument(
        "--reuse_stage2",
        action="store_true",
        help="Skip Stage 2 training and reuse saved stage2 OOF/test probs from OUTPUT_DIR",
    )
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


def hard_to_distribution(labels: np.ndarray, label_smoothing: float) -> np.ndarray:
    eps = max(0.0, min(label_smoothing, 0.2))
    dist = np.zeros((len(labels), 2), dtype=np.float32)
    dist[np.arange(len(labels)), labels.astype(int)] = 1.0
    if eps > 0:
        dist = (1.0 - eps) * dist + eps / 2.0
    return dist


class TextDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        encodings,
        labels_dist: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ):
        self.encodings = encodings
        self.labels_dist = labels_dist
        self.sample_weights = sample_weights

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels_dist is not None:
            item["labels"] = torch.tensor(self.labels_dist[idx], dtype=torch.float)
        if self.sample_weights is not None:
            item["sample_weight"] = torch.tensor(float(self.sample_weights[idx]), dtype=torch.float)
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


class SoftLabelTrainer(Trainer):
    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)
        sample_weight = inputs.pop("sample_weight", None)
        outputs = model(**inputs)
        logits = outputs.logits

        if labels is None:
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        if labels.ndim == 1:
            labels = torch.nn.functional.one_hot(labels.long(), num_classes=logits.shape[-1]).float()

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        per_sample_loss = -(labels * log_probs).sum(dim=-1)

        if self.class_weights is not None:
            class_factor = (labels * self.class_weights.view(1, -1)).sum(dim=-1)
            per_sample_loss = per_sample_loss * class_factor

        if sample_weight is not None:
            per_sample_loss = per_sample_loss * sample_weight

        loss = per_sample_loss.mean()
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    if labels.ndim > 1:
        labels = np.argmax(labels, axis=-1)
    return {
        "f1": f1_score(labels, preds, average="binary"),
        "accuracy": accuracy_score(labels, preds),
    }


def build_training_args(output_dir, epochs, args, use_cpu=False):
    args_dict = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
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
        "fp16": (not use_cpu) and (not args.disable_fp16),
    }

    try:
        return TrainingArguments(**args_dict, eval_strategy="epoch", use_cpu=use_cpu)
    except TypeError:
        try:
            return TrainingArguments(**args_dict, evaluation_strategy="epoch", use_cpu=use_cpu)
        except TypeError:
            return TrainingArguments(**args_dict, evaluation_strategy="epoch", no_cuda=use_cpu)


def get_data_paths() -> Tuple[str, str]:
    train_path = TRAIN_PATH
    test_path = TEST_PATH
    if not os.path.exists(train_path):
        train_path = f"../{TRAIN_PATH}"
        test_path = f"../{TEST_PATH}"
    return train_path, test_path


def select_confident_pseudo_labels(
    probs: np.ndarray,
    confidence_threshold: float,
    margin_threshold: float,
    max_pseudo_per_class: int,
) -> PseudoSelectionResult:
    pred = np.argmax(probs, axis=1)
    conf = probs.max(axis=1)
    margin = np.abs(probs[:, 1] - probs[:, 0])

    candidates = np.where((conf >= confidence_threshold) & (margin >= margin_threshold))[0]

    by_class = {}
    for c in [0, 1]:
        cls_idx = candidates[pred[candidates] == c]
        cls_sorted = cls_idx[np.argsort(-conf[cls_idx])]
        by_class[c] = cls_sorted

    if len(by_class[0]) > 0 and len(by_class[1]) > 0:
        take_each = min(len(by_class[0]), len(by_class[1]))
        if max_pseudo_per_class > 0:
            take_each = min(take_each, max_pseudo_per_class)
        selected = np.concatenate([by_class[0][:take_each], by_class[1][:take_each]])
    else:
        global_sorted = candidates[np.argsort(-conf[candidates])]
        take_global = len(global_sorted) if max_pseudo_per_class <= 0 else min(len(global_sorted), max_pseudo_per_class)
        selected = global_sorted[:take_global]

    selected = selected[np.argsort(-conf[selected])]
    selected_probs = probs[selected]
    selected_labels = np.argmax(selected_probs, axis=1)

    diagnostics = pd.DataFrame(
        {
            "test_index": selected,
            "pseudo_label": selected_labels,
            "prob_0": selected_probs[:, 0],
            "prob_1": selected_probs[:, 1],
            "confidence": conf[selected],
            "margin": margin[selected],
        }
    )

    return PseudoSelectionResult(
        selected_indices=selected,
        selected_probs=selected_probs,
        selected_labels=selected_labels,
        diagnostics=diagnostics,
    )


def find_best_threshold(y_true: np.ndarray, y_prob_pos: np.ndarray, grid_step: float = 0.01) -> Tuple[float, float]:
    best_t = 0.5
    best_f1 = -1.0
    step = min(0.1, max(0.001, grid_step))
    thresholds = np.arange(0.05, 0.95 + 1e-8, step)

    for t in thresholds:
        pred = (y_prob_pos >= t).astype(int)
        score = f1_score(y_true, pred, average="binary")
        if score > best_f1:
            best_f1 = score
            best_t = float(t)

    return best_t, float(best_f1)


def train_kfold_stage(
    stage_name: str,
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    data_collator,
    skf: StratifiedKFold,
    y: np.ndarray,
    args,
    use_cpu: bool,
    epochs: int,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    pseudo_texts: Optional[List[str]] = None,
    pseudo_soft_labels: Optional[np.ndarray] = None,
):
    class_weights_tensor = None
    if not args.disable_class_weights:
        class_weights_arr = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
        class_weights_tensor = torch.tensor(class_weights_arr, dtype=torch.float).to("cuda" if not use_cpu else "cpu")
        print(f"[{stage_name}] Class weights: {class_weights_arr}")

    fold_metrics = []
    oof_probs = np.zeros((len(train_df), 2), dtype=np.float32)
    test_probs_all_folds = []

    test_enc = tokenizer(test_df["CLEAN_TEXT"].tolist(), truncation=True, max_length=MAX_LENGTH)
    test_ds = TextDataset(test_enc)

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(train_df, y), start=1):
        print(f"\n===== {stage_name} Fold {fold_idx}/{args.num_folds} =====")
        fold_dir = os.path.join(output_dir, stage_name, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        tr_texts = train_df.iloc[tr_idx]["CLEAN_TEXT"].tolist()
        tr_gold_labels = y[tr_idx]
        tr_labels_dist = hard_to_distribution(tr_gold_labels, args.label_smoothing)
        tr_sample_weights = np.ones(len(tr_texts), dtype=np.float32)

        if pseudo_texts is not None and pseudo_soft_labels is not None and len(pseudo_texts) > 0:
            tr_texts = tr_texts + pseudo_texts
            tr_labels_dist = np.vstack([tr_labels_dist, pseudo_soft_labels.astype(np.float32)])
            pseudo_weights = np.full(len(pseudo_texts), float(args.pseudo_loss_weight), dtype=np.float32)
            tr_sample_weights = np.concatenate([tr_sample_weights, pseudo_weights], axis=0)

        va_texts = train_df.iloc[va_idx]["CLEAN_TEXT"].tolist()
        va_labels = y[va_idx]
        va_labels_dist = hard_to_distribution(va_labels, 0.0)

        tr_enc = tokenizer(tr_texts, truncation=True, max_length=MAX_LENGTH)
        va_enc = tokenizer(va_texts, truncation=True, max_length=MAX_LENGTH)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )

        trainer = SoftLabelTrainer(
            model=model,
            args=build_training_args(fold_dir, epochs, args, use_cpu=use_cpu),
            train_dataset=TextDataset(tr_enc, tr_labels_dist, tr_sample_weights),
            eval_dataset=TextDataset(va_enc, va_labels_dist),
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            class_weights=class_weights_tensor,
        )

        trainer.train()
        eval_result = trainer.evaluate()

        va_pred = trainer.predict(TextDataset(va_enc, va_labels_dist)).predictions
        va_prob = torch.softmax(torch.tensor(va_pred), dim=1).numpy()
        oof_probs[va_idx] = va_prob

        test_pred = trainer.predict(test_ds).predictions
        test_prob = torch.softmax(torch.tensor(test_pred), dim=1).numpy()
        test_probs_all_folds.append(test_prob)

        fold_metrics.append(
            {
                "stage": stage_name,
                "fold": fold_idx,
                "eval_f1": eval_result.get("eval_f1", np.nan),
                "eval_accuracy": eval_result.get("eval_accuracy", np.nan),
                "eval_loss": eval_result.get("eval_loss", np.nan),
            }
        )

        print(
            f"{stage_name} fold {fold_idx} -> F1: {fold_metrics[-1]['eval_f1']:.4f}, "
            f"Acc: {fold_metrics[-1]['eval_accuracy']:.4f}"
        )

    mean_test_probs = np.mean(np.stack(test_probs_all_folds, axis=0), axis=0)
    return pd.DataFrame(fold_metrics), oof_probs, mean_test_probs


def main():
    args = parse_args()
    set_seed(args.seed)
    OUTPUT_DIR = args.output_dir
    SUBMISSION_PATH = args.submission_path
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    use_cpu = args.force_cpu or (not torch.cuda.is_available())
    print(f"Runtime device: {'cpu' if use_cpu else 'cuda'}")
    print(f"Using model: {args.model_name}")

    train_path, test_path = get_data_paths()
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df["CLEAN_TEXT"] = train_df["TEXT"].apply(clean_for_transformer)
    test_df["CLEAN_TEXT"] = test_df["TEXT"].apply(clean_for_transformer)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    y = train_df["LABEL"].values
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)

    stage1_oof_path = os.path.join(OUTPUT_DIR, "oof_stage1_probs_v6.csv")
    stage1_test_path = os.path.join(OUTPUT_DIR, "test_stage1_probs_v6.csv")
    stage1_metrics_path = os.path.join(OUTPUT_DIR, "cv_metrics_stage1_v6.csv")
    stage2_oof_path = os.path.join(OUTPUT_DIR, "oof_stage2_probs_v6.csv")
    stage2_test_path = os.path.join(OUTPUT_DIR, "test_stage2_probs_v6.csv")
    stage2_metrics_path = os.path.join(OUTPUT_DIR, "cv_metrics_stage2_v6.csv")

    if args.reuse_stage1 and os.path.exists(stage1_oof_path) and os.path.exists(stage1_test_path):
        print("\n[Stage 1] Reusing saved Stage 1 probs (--reuse_stage1 flag set)")
        s1_oof_df = pd.read_csv(stage1_oof_path)
        oof_probs_s1 = s1_oof_df[["prob_0", "prob_1"]].values.astype(np.float32)
        s1_test_df = pd.read_csv(stage1_test_path)
        test_probs_s1 = s1_test_df[["prob_0", "prob_1"]].values.astype(np.float32)
        stage1_metrics = pd.read_csv(stage1_metrics_path) if os.path.exists(stage1_metrics_path) else pd.DataFrame()
    else:
        print("\n[Stage 1] Train on gold labels only")
        stage1_metrics, oof_probs_s1, test_probs_s1 = train_kfold_stage(
            stage_name="stage1_gold_only",
            model_name=args.model_name,
            train_df=train_df,
            test_df=test_df,
            tokenizer=tokenizer,
            data_collator=data_collator,
            skf=skf,
            y=y,
            args=args,
            use_cpu=use_cpu,
            epochs=args.epochs_stage1,
            output_dir=OUTPUT_DIR,
        )
        s1_oof_df = pd.DataFrame({
            "row_id": train_df["row_id"],
            "label": y,
            "prob_0": oof_probs_s1[:, 0],
            "prob_1": oof_probs_s1[:, 1],
        })
        s1_oof_df.to_csv(stage1_oof_path, index=False)
        s1_test_df = pd.DataFrame({
            "row_id": test_df["row_id"],
            "prob_0": test_probs_s1[:, 0],
            "prob_1": test_probs_s1[:, 1],
        })
        s1_test_df.to_csv(stage1_test_path, index=False)
        stage1_metrics.to_csv(stage1_metrics_path, index=False)

    s1_threshold, s1_f1 = find_best_threshold(y, oof_probs_s1[:, 1], args.threshold_grid_step)
    print(f"[Stage 1] Best OOF threshold={s1_threshold:.3f}, F1={s1_f1:.4f}")

    pseudo_result = select_confident_pseudo_labels(
        probs=test_probs_s1,
        confidence_threshold=args.confidence_threshold,
        margin_threshold=args.margin_threshold,
        max_pseudo_per_class=args.max_pseudo_per_class,
    )

    pseudo_diag = pseudo_result.diagnostics.copy()
    pseudo_diag.insert(0, "row_id", test_df.iloc[pseudo_diag["test_index"]]["row_id"].values)

    pseudo_count = len(pseudo_result.selected_indices)
    print(f"[Pseudo] Selected {pseudo_count} high-confidence samples")
    if pseudo_count == 0:
        print("[Pseudo] No pseudo labels selected. Stage 2 will fallback to gold-only training.")

    pseudo_texts = test_df.iloc[pseudo_result.selected_indices]["CLEAN_TEXT"].tolist() if pseudo_count > 0 else None
    pseudo_soft_labels = pseudo_result.selected_probs if pseudo_count > 0 else None

    if args.reuse_stage2 and os.path.exists(stage2_oof_path) and os.path.exists(stage2_test_path):
        print("\n[Stage 2] Reusing saved Stage 2 probs (--reuse_stage2 flag set)")
        s2_oof_df = pd.read_csv(stage2_oof_path)
        oof_probs_s2 = s2_oof_df[["prob_0", "prob_1"]].values.astype(np.float32)
        s2_test_df = pd.read_csv(stage2_test_path)
        test_probs_s2 = s2_test_df[["prob_0", "prob_1"]].values.astype(np.float32)
        stage2_metrics = pd.read_csv(stage2_metrics_path) if os.path.exists(stage2_metrics_path) else pd.DataFrame()
    else:
        print("\n[Stage 2] Train on gold + confidence-filtered soft pseudo labels")
        stage2_metrics, oof_probs_s2, test_probs_s2 = train_kfold_stage(
            stage_name="stage2_gold_plus_pseudo",
            model_name=args.model_name,
            train_df=train_df,
            test_df=test_df,
            tokenizer=tokenizer,
            data_collator=data_collator,
            skf=skf,
            y=y,
            args=args,
            use_cpu=use_cpu,
            epochs=args.epochs_stage2,
            output_dir=OUTPUT_DIR,
            pseudo_texts=pseudo_texts,
            pseudo_soft_labels=pseudo_soft_labels,
        )
        s2_test_df = pd.DataFrame({
            "row_id": test_df["row_id"],
            "prob_0": test_probs_s2[:, 0],
            "prob_1": test_probs_s2[:, 1],
        })
        s2_test_df.to_csv(stage2_test_path, index=False)
        stage2_metrics.to_csv(stage2_metrics_path, index=False)

    s2_threshold, s2_f1 = find_best_threshold(y, oof_probs_s2[:, 1], args.threshold_grid_step)
    print(f"[Stage 2] Best OOF threshold={s2_threshold:.3f}, F1={s2_f1:.4f}")

    # Stage 3: re-generate pseudo labels from Stage 2 predictions
    if args.run_stage3:
        pseudo_result_s3 = select_confident_pseudo_labels(
            probs=test_probs_s2,
            confidence_threshold=args.confidence_threshold,
            margin_threshold=args.margin_threshold,
            max_pseudo_per_class=args.max_pseudo_per_class,
        )
        pseudo_count_s3 = len(pseudo_result_s3.selected_indices)
        print(f"\n[Stage 3 Pseudo] Selected {pseudo_count_s3} high-confidence samples from Stage 2 predictions")

        pseudo_texts_s3 = test_df.iloc[pseudo_result_s3.selected_indices]["CLEAN_TEXT"].tolist() if pseudo_count_s3 > 0 else None
        pseudo_soft_labels_s3 = pseudo_result_s3.selected_probs if pseudo_count_s3 > 0 else None

        print("\n[Stage 3] Train on gold + Stage-2-derived soft pseudo labels")
        stage3_metrics, oof_probs_s3, test_probs_s3 = train_kfold_stage(
            stage_name="stage3_gold_plus_pseudo_s2",
            model_name=args.model_name,
            train_df=train_df,
            test_df=test_df,
            tokenizer=tokenizer,
            data_collator=data_collator,
            skf=skf,
            y=y,
            args=args,
            use_cpu=use_cpu,
            epochs=args.epochs_stage3,
            output_dir=OUTPUT_DIR,
            pseudo_texts=pseudo_texts_s3,
            pseudo_soft_labels=pseudo_soft_labels_s3,
        )

        s3_threshold, s3_f1 = find_best_threshold(y, oof_probs_s3[:, 1], args.threshold_grid_step)
        print(f"[Stage 3] Best OOF threshold={s3_threshold:.3f}, F1={s3_f1:.4f}")

        # Save Stage 3 pseudo diagnostics
        pseudo_diag_s3 = pseudo_result_s3.diagnostics.copy()
        pseudo_diag_s3.insert(0, "row_id", test_df.iloc[pseudo_diag_s3["test_index"]]["row_id"].values)
        pseudo_diag_s3.to_csv(os.path.join(OUTPUT_DIR, "pseudo_selected_stage3_v6.csv"), index=False)

        # Save Stage 3 OOF probs
        oof_s3_df = pd.DataFrame({
            "row_id": train_df["row_id"],
            "label": y,
            "prob_0": oof_probs_s3[:, 0],
            "prob_1": oof_probs_s3[:, 1],
        })
        oof_s3_df.to_csv(os.path.join(OUTPUT_DIR, "oof_stage3_probs_v6.csv"), index=False)

        # Use Stage 3 for final output
        final_test_probs = test_probs_s3
        final_threshold = s3_threshold
        final_oof_probs = oof_probs_s3
        all_metrics = pd.concat([stage1_metrics, stage2_metrics, stage3_metrics], ignore_index=True)
    else:
        final_test_probs = test_probs_s2
        final_threshold = s2_threshold
        final_oof_probs = oof_probs_s2
        all_metrics = pd.concat([stage1_metrics, stage2_metrics], ignore_index=True)

    final_preds = (final_test_probs[:, 1] >= final_threshold).astype(int)
    submission_df = pd.DataFrame({"row_id": test_df["row_id"], "LABEL": final_preds})
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    metrics_path = os.path.join(OUTPUT_DIR, "cv_metrics_v6.csv")
    all_metrics.to_csv(metrics_path, index=False)

    oof_df = pd.DataFrame(
        {
            "row_id": train_df["row_id"],
            "label": y,
            "prob_0": final_oof_probs[:, 0],
            "prob_1": final_oof_probs[:, 1],
        }
    )
    oof_path = os.path.join(OUTPUT_DIR, "oof_stage2_probs_v6.csv")
    oof_df.to_csv(oof_path, index=False)

    test_prob_df = pd.DataFrame(
        {
            "row_id": test_df["row_id"],
            "prob_0": final_test_probs[:, 0],
            "prob_1": final_test_probs[:, 1],
            "pred_threshold": final_threshold,
        }
    )
    test_prob_path = os.path.join(OUTPUT_DIR, "test_probs_v6.csv")
    test_prob_df.to_csv(test_prob_path, index=False)

    pseudo_path = os.path.join(OUTPUT_DIR, "pseudo_selected_v6.csv")
    pseudo_diag.to_csv(pseudo_path, index=False)

    threshold_path = os.path.join(OUTPUT_DIR, "best_threshold_v6.txt")
    with open(threshold_path, "w", encoding="utf-8") as f:
        f.write(f"stage1_best_threshold={s1_threshold:.6f}\n")
        f.write(f"stage1_oof_f1={s1_f1:.6f}\n")
        f.write(f"stage2_best_threshold={s2_threshold:.6f}\n")
        f.write(f"stage2_oof_f1={s2_f1:.6f}\n")
        if args.run_stage3:
            f.write(f"stage3_best_threshold={s3_threshold:.6f}\n")
            f.write(f"stage3_oof_f1={s3_f1:.6f}\n")

    print("\n===== V6 Summary =====")
    print(all_metrics)
    print("-" * 40)
    print(f"Stage1 OOF Best F1: {s1_f1:.4f} @ threshold {s1_threshold:.3f}")
    print(f"Stage2 OOF Best F1: {s2_f1:.4f} @ threshold {s2_threshold:.3f}")
    if args.run_stage3:
        print(f"Stage3 OOF Best F1: {s3_f1:.4f} @ threshold {s3_threshold:.3f}")
    print(f"Saved submission to: {SUBMISSION_PATH}")
    print(f"Saved CV metrics to: {metrics_path}")
    print(f"Saved OOF probs to: {oof_path}")
    print(f"Saved test probs to: {test_prob_path}")
    print(f"Saved pseudo diagnostics to: {pseudo_path}")
    print(f"Saved threshold summary to: {threshold_path}")


if __name__ == "__main__":
    main()
