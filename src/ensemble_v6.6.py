"""
v6.6 Ensemble: RoBERTa-large (v6.1) + DeBERTa-v3-large (v6.5) + ELECTRA-large (v6.6)
Averages Stage2 test probabilities, tunes threshold on averaged OOF probabilities.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble v6.6: multi-model probability averaging")
    parser.add_argument(
        "--model_dirs",
        nargs="+",
        default=[
            "result/version6_model",
            "result/version65_model",
            "result/version66_electra_model",
        ],
        help="List of model output directories (each must contain oof_stage2_probs_v6.csv and test_stage2_probs_v6.csv)",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional per-model weights for weighted average (defaults to equal weights)",
    )
    parser.add_argument("--output_dir", type=str, default="result/version66_ensemble_model")
    parser.add_argument("--submission_path", type=str, default="result/submission_v66.csv")
    parser.add_argument("--test_path", type=str, default="data/test_no_answer_2022.csv")
    parser.add_argument("--threshold_grid_step", type=float, default=0.01)
    return parser.parse_args()


def load_probs(model_dir: str, filename: str) -> pd.DataFrame:
    path = os.path.join(model_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)


def search_threshold(oof_probs: np.ndarray, oof_labels: np.ndarray, step: float) -> tuple[float, float]:
    thresholds = np.arange(0.05, 0.95 + 1e-8, step)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds = (oof_probs >= t).astype(int)
        f1 = f1_score(oof_labels, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return round(best_t, 4), round(best_f1, 6)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    n_models = len(args.model_dirs)
    weights = args.weights
    if weights is None:
        weights = [1.0 / n_models] * n_models
    else:
        if len(weights) != n_models:
            raise ValueError(f"--weights length {len(weights)} != --model_dirs length {n_models}")
        total = sum(weights)
        weights = [w / total for w in weights]

    print(f"\n=== v6.6 Ensemble ===")
    print(f"Models ({n_models}):")
    for d, w in zip(args.model_dirs, weights):
        print(f"  {d}  (weight={w:.4f})")

    # ---- Load OOF probs ----
    oof_frames = [load_probs(d, "oof_stage2_probs_v6.csv") for d in args.model_dirs]

    # Align on row_id
    base_oof = oof_frames[0][["row_id", "label"]].copy()
    for i, df in enumerate(oof_frames):
        if not (df["row_id"].values == base_oof["row_id"].values).all():
            raise ValueError(f"OOF row_id mismatch between model 0 and model {i}")

    oof_avg = np.zeros(len(base_oof), dtype=np.float64)
    for df, w in zip(oof_frames, weights):
        oof_avg += w * df["prob_1"].values

    oof_labels = base_oof["label"].values
    best_t, best_f1 = search_threshold(oof_avg, oof_labels, args.threshold_grid_step)
    print(f"\nEnsemble OOF F1 = {best_f1:.6f}  @  threshold = {best_t}")

    oof_preds = (oof_avg >= best_t).astype(int)
    from sklearn.metrics import accuracy_score
    oof_acc = accuracy_score(oof_labels, oof_preds)
    print(f"Ensemble OOF Acc = {oof_acc:.6f}")

    # Save OOF ensemble probs
    oof_out = base_oof.copy()
    oof_out["prob_1_ensemble"] = oof_avg
    oof_out.to_csv(os.path.join(args.output_dir, "oof_ensemble_probs.csv"), index=False)

    # ---- Load test probs ----
    test_frames = [load_probs(d, "test_stage2_probs_v6.csv") for d in args.model_dirs]

    base_test = test_frames[0][["row_id"]].copy()
    for i, df in enumerate(test_frames):
        if not (df["row_id"].values == base_test["row_id"].values).all():
            raise ValueError(f"Test row_id mismatch between model 0 and model {i}")

    test_avg = np.zeros(len(base_test), dtype=np.float64)
    for df, w in zip(test_frames, weights):
        test_avg += w * df["prob_1"].values

    test_out = base_test.copy()
    test_out["prob_1_ensemble"] = test_avg
    test_out.to_csv(os.path.join(args.output_dir, "test_ensemble_probs.csv"), index=False)

    # ---- Generate submission ----
    test_df = pd.read_csv(args.test_path)
    preds = (test_avg >= best_t).astype(int)
    submission = pd.DataFrame({"row_id": test_df["row_id"], "label": preds})
    submission.to_csv(args.submission_path, index=False)

    pos_rate = preds.mean()
    print(f"\nSubmission saved: {args.submission_path}")
    print(f"  Positive rate: {pos_rate:.4f}  ({preds.sum()}/{len(preds)})")
    print(f"  Threshold used: {best_t}")

    # Save threshold / metrics
    with open(os.path.join(args.output_dir, "ensemble_metrics.txt"), "w") as f:
        f.write(f"models={[os.path.basename(d) for d in args.model_dirs]}\n")
        f.write(f"weights={weights}\n")
        f.write(f"best_threshold={best_t}\n")
        f.write(f"oof_f1={best_f1}\n")
        f.write(f"oof_acc={oof_acc:.6f}\n")
        f.write(f"positive_rate={pos_rate:.4f}\n")

    print("\nDone.")


if __name__ == "__main__":
    main()
