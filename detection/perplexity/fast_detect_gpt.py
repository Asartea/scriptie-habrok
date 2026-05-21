import argparse
import random
from enum import Enum
from os import environ
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from fast_detect_gpt.scripts.local_infer import FastDetectGPT
from models.models import HumanSample, Samples
from utils.utils import load_samples

CACHE_DIR = environ.get("HF_CACHE_DIR", None)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class ThresholdStrategy(str, Enum):
    PERCENTILE = "percentile"
    MEAN_STD = "mean_std"
    MAD = "mad"
    IQR = "iqr"


@torch.inference_mode()
def compute_threshold(strategy: ThresholdStrategy, scores: list[float]) -> float:
    np_scores = np.array(scores)

    if strategy == ThresholdStrategy.PERCENTILE:
        return float(np.percentile(np_scores, 95))

    if strategy == ThresholdStrategy.MEAN_STD:
        mu = np.mean(np_scores)
        sigma = np.std(np_scores)
        return float(mu + 2.0 * sigma)

    if strategy == ThresholdStrategy.MAD:
        med = np.median(np_scores)
        mad = np.median(np.abs(np_scores - med))
        return float(med + 3.0 * 1.4826 * mad)

    if strategy == ThresholdStrategy.IQR:
        q1 = np.percentile(np_scores, 25)
        q3 = np.percentile(np_scores, 75)
        iqr = q3 - q1
        return float(q3 + 1.5 * iqr)


def compute_scores_with_label(detector: FastDetectGPT, samples: Samples):
    scores: list[tuple[str, float]] = []
    for sample in tqdm(samples):
        crit, _ = detector.compute_crit(sample["code"])
        scores.append((sample["label"], float(crit)))
    return scores


@torch.inference_mode()
def classify_samples(
    scores: list[tuple[str, float]],
    threshold: float,
):

    results: list[dict[str, str | float]] = []

    for score in tqdm(scores):
        label, crit = score
        is_ai = crit > threshold

        results.append(
            {
                "score": crit,
                "pred_label": is_ai,
                "actual_label": label,
            }
        )
    return results


def split_calibration_samples(samples: list[HumanSample], n: int = 300):
    rng = random.Random(227)  # fixed seed for reproducibility
    shuffled = samples.copy()
    rng.shuffle(shuffled)

    calibration_samples = shuffled[:n]
    test_samples = shuffled[n:]
    return calibration_samples, test_samples


def evaluate(results):
    y_true = []
    y_pred = []
    y_score = []

    for r in results:
        if "score" not in r or r["score"] is None:
            continue

        y_true.append(1 if r["actual_label"] != "human" else 0)
        y_pred.append(1 if r["pred_label"] else 0)
        y_score.append(r["score"])

    print("\n=== EVALUATION ===")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    auc = roc_auc_score(y_true, y_score)

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "auroc": auc,
        "confusion_matrix": cm.tolist(),
    }


def results_output(
    model_name: str,
    dataset_name: str,
    threshold: float,
    metrics: dict[str, float | str | None],
):
    return f"""Model: {model_name}
Dataset: {dataset_name}
Threshold: {threshold:.6f}
=== METRICS ===
Accuracy : {metrics['accuracy']:.4f}
Precision: {metrics['precision']:.4f}
Recall   : {metrics['recall']:.4f}
F1       : {metrics['f1_score']:.4f}
AUROC    : {metrics['auroc']:.4f}
Confusion Matrix: {metrics['confusion_matrix']}
"""


def run(model_name: str):
    base_dir = Path("data")
    human_samples_path = (
        base_dir / "normal" / "human_samples.jsonl"
    )  # identical between normal and comp

    normal_machine_samples_path = base_dir / "normal" / "machine_samples.jsonl"
    comp_machine_samples_path = (
        base_dir / "competitive_programming" / "machine_samples.jsonl"
    )
    normal_machine_samples = load_samples(normal_machine_samples_path)
    comp_machine_samples = load_samples(comp_machine_samples_path)
    human_samples = load_samples(human_samples_path)

    calibration_samples, human_test_samples = split_calibration_samples(human_samples)

    detector = FastDetectGPT(
        scoring_model_name=model_name,
        sampling_model_name=model_name,
        cache_dir=CACHE_DIR,
        extra_distrib_params={},
    )

    calibration_scores = compute_scores_with_label(detector, calibration_samples)
    thresholds: list[tuple[ThresholdStrategy, float]] = []
    for strategy in ThresholdStrategy:
        threshold = compute_threshold(
            strategy, [score for _, score in calibration_scores]
        )
        print(f"Computed threshold using strategy {strategy}: {threshold:.6f}")
        thresholds.append((strategy, threshold))

    samples = {
        "Normal": normal_machine_samples + human_test_samples,
        "Competitive Programming": comp_machine_samples + human_test_samples,
    }
    for name, sample_set in samples.items():
        print(f"\n=== CLASSIFYING SAMPLE SET {name} ===")
        output_path = (
            Path("results")
            / f"perplexity_{model_name.replace('/', '_')}_{name.lower().replace(" ", "_")}.txt"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scores = compute_scores_with_label(detector, sample_set)
        for strategy, threshold in thresholds:
            output_path.open("a").write(
                f"Threshold strategy: {strategy}\nThreshold value: {threshold:.6f}\n\n"
            )
            results = classify_samples(scores, threshold)
            metrics = evaluate(results)

            output = results_output(
                model_name,
                name,
                threshold,
                metrics,
            )
            output_path.open("a").write(output + "\n\n")
            print(output)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Run FastDetectGPT on AoC samples")
    parser.add_argument("--model", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
