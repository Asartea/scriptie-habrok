import json
from pathlib import Path
from typing import TypedDict, Literal


class AblationRunFold(TypedDict):
    fold: int
    accuracy: float
    f1_score: float
    confusion_matrix: list[list[int]]


class AblationRunSummary(TypedDict):
    name: str
    group_mode: Literal["problem", "author_like", "strict"]
    accuracy_mean: float
    accuracy_std: float
    f1_mean: float
    f1_std: float
    confusion_matrix_sum: list[list[int]]


class AblationRun(TypedDict):
    ablation: str
    folds: list[AblationRunFold]
    summary: AblationRunSummary


def load_ablation_runs(log_path: Path) -> list[AblationRun]:
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    log = json.loads(log_path.read_text())
    return log.get("ablation_runs", [])


def analyse_run(run: AblationRun):
    name = run["ablation"]
    summary = run["summary"]
    acc_mean = summary["accuracy_mean"]
    acc_std = summary["accuracy_std"]
    f1_mean = summary["f1_mean"]
    f1_std = summary["f1_std"]
    confusion_matrix_sum = summary["confusion_matrix_sum"]
    return {
        "name": name,
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "f1_mean": f1_mean,
        "f1_std": f1_std,
        "confusion_matrix_sum": confusion_matrix_sum,
    }


def analyzer():
    sample_types = {
        "full": "samples",
        "normal": "normal_samples",
        "comp": "comp_samples",
    }
    sample_groups = {
        "problem": "problem",
        "author_like": "author",
        "composite": "strict",
    }

    for sample_type, sample_type_path in sample_types.items():
        for sample_group, sample_group_path in sample_groups.items():
            log_path = (
                Path("results")
                / f"{sample_type_path}_ablation_{sample_group_path}.json"
            )
            runs = load_ablation_runs(log_path)
            analyses = [analyse_run(run) for run in runs]
            print(f"Results for {sample_type} - {sample_group}:")
            for analysis in analyses:
                print(f"  Ablation: {analysis['name']}")
                print(
                    f"    Accuracy: {analysis['accuracy_mean']:.4f} ± {analysis['accuracy_std']:.4f}"
                )
                print(
                    f"    F1 Score: {analysis['f1_mean']:.4f} ± {analysis['f1_std']:.4f}"
                )
                print(f"    Confusion Matrix Sum: {analysis['confusion_matrix_sum']}")


def main():
    analyzer()


if __name__ == "__main__":
    main()
