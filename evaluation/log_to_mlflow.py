"""
Day 13 monitoring: log training/eval runs to MLflow.

Honest scope note: the segmentation and classifier models were trained on
Kaggle/Colab (Day 5-8) before MLflow was part of this project's toolchain --
there is no live run to "log" in the normal sense of instrumenting a
training loop as it runs. What this script does instead is a one-time
**backfill**: it takes the same already-measured, already-reviewed numbers
evaluation/run_all.py prints (parsed from docs/model_card.md,
evaluation/segmentation_report.md, evaluation/rag_eval_report.md,
evaluation/agent_eval_report.md) and logs them as MLflow runs, tagged
`run_type=backfill`, so there's a real queryable run history in an MLflow
tracking store going forward -- not a claim that these were tracked live
during the original training.

Any future retraining should call mlflow.log_metric(...) directly inside
the training loop instead of through this script.

Run: python evaluation/log_to_mlflow.py
Then: mlflow ui   (defaults to a local ./mlruns store)
"""

import re
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).parent))
from run_all import (  # noqa: E402
    agent_metrics,
    classifier_metrics,
    fairness_table,
    rag_metrics,
    segmentation_metrics,
)

EXPERIMENT_NAME = "skinscope-au"


def _to_float(value: str):
    """Best-effort parse of the report strings this script logs (plain
    decimals, "36.36%", or "16/20 (80.0%)") into a loggable float. Returns
    None (skip logging that metric) rather than raising, if the format
    changes -- an MLflow run missing one metric is far less confusing than
    a crashed backfill."""
    if value == "MISSING":
        return None
    m = re.search(r"\(([\d.]+)%\)", value)
    if m:
        return float(m.group(1)) / 100.0
    if value.endswith("%"):
        return float(value[:-1]) / 100.0
    try:
        return float(value)
    except ValueError:
        return None


def _safe_metric_name(name: str) -> str:
    """MLflow metric names may only contain alphanumerics, underscores,
    dashes, periods, spaces, colons, and slashes -- e.g. "~" (from
    "~95% specificity") is rejected and crashes log_metric. Replace
    anything else with "_" rather than assume the report text is already
    a valid metric name."""
    name = name.replace(" ", "_").replace("%", "pct").replace("@", "at").replace("~", "approx")
    return re.sub(r"[^A-Za-z0-9_\-.: /]", "_", name)


def _log_metrics(metrics: dict) -> None:
    for name, raw in metrics.items():
        val = _to_float(raw)
        if val is not None:
            mlflow.log_metric(_safe_metric_name(name), val)
        else:
            mlflow.set_tag(f"unparsed__{name}", raw)


def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="segmentation-deeplabv3plus-resnet34"):
        mlflow.set_tags({
            "run_type": "backfill",
            "component": "segmentation",
            "architecture": "DeepLabV3+ (ResNet-34 encoder, ImageNet-pretrained)",
            "dataset": "ISIC2018 Task 1",
            "source_report": "evaluation/segmentation_report.md",
        })
        mlflow.log_params({"input_size": "256x256", "loss": "Dice+BCE"})
        _log_metrics(segmentation_metrics())
    print("Logged segmentation backfill run.")

    with mlflow.start_run(run_name="classifier-efficientnet-b0"):
        mlflow.set_tags({
            "run_type": "backfill",
            "component": "classifier",
            "architecture": "EfficientNet-B0 (timm, ImageNet-pretrained)",
            "dataset": "HAM10000",
            "source_report": "docs/model_card.md",
        })
        mlflow.log_params({"input_size": "224x224", "loss": "BCEWithLogitsLoss (pos_weight=4.12)"})
        _log_metrics(classifier_metrics())

        for row in fairness_table():
            group = row["group"]
            sens = _to_float(row["sensitivity"])
            spec = _to_float(row["specificity"])
            if sens is not None:
                mlflow.log_metric(f"fairness_{group}_sensitivity", sens)
            if spec is not None:
                mlflow.log_metric(f"fairness_{group}_specificity", spec)
    print("Logged classifier backfill run (incl. fairness table).")

    with mlflow.start_run(run_name="rag-retriever-day10"):
        mlflow.set_tags({
            "run_type": "backfill",
            "component": "rag_retriever",
            "source_report": "evaluation/rag_eval_report.md",
        })
        _log_metrics(rag_metrics())
    print("Logged RAG retrieval backfill run.")

    with mlflow.start_run(run_name="agent-faithfulness-day11"):
        mlflow.set_tags({
            "run_type": "backfill",
            "component": "agent",
            "judge_model": "llama-3.1-8b-instant",
            "source_report": "evaluation/agent_eval_report.md",
        })
        _log_metrics(agent_metrics())
    print("Logged agent faithfulness backfill run.")

    print(f"\nAll runs logged under experiment '{EXPERIMENT_NAME}'. Run `mlflow ui` to view.")


if __name__ == "__main__":
    main()
