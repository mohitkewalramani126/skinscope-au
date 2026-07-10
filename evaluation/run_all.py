"""
Day 13: one-command evaluation summary.

Per the build plan: "Consolidate evaluation/ so one command prints all
metrics: seg IoU/Dice, classifier AUC/sensitivity/calibration, fairness
table, RAG precision/faithfulness."

Design choice: this script does NOT re-run training or re-derive any number
from raw data. The raw datasets (HAM10000, ISIC2018) live on Kaggle/Colab,
not in this repo -- only the ONNX exports and two local sample photos are
kept here (see docs/data_sources.md). Re-deriving everything on every run
would need those datasets, a GPU, and re-running the Day 10/11 RAG+agent
pipelines against a live Groq key and a rebuilt Chroma index, none of which
should be required just to *read* metrics that have already been measured
once, honestly, and written down.

Instead, this parses the headline numbers directly out of the existing,
already-reviewed report files (docs/model_card.md,
evaluation/segmentation_report.md, evaluation/rag_eval_report.md,
evaluation/agent_eval_report.md), so there is exactly one source of truth
per number -- this script only aggregates and prints it. If a report's
wording changes and a regex stops matching, this script says so explicitly
(MISSING) rather than silently printing a stale or wrong value.

Run: python evaluation/run_all.py
"""

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
DOCS = ROOT / "docs"
EVAL = ROOT / "evaluation"


def _read(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _find(pattern: str, text: str, group: int = 1, flags=re.IGNORECASE) -> str:
    m = re.search(pattern, text, flags)
    return m.group(group).strip() if m else "MISSING"


def segmentation_metrics() -> dict:
    text = _read(EVAL / "segmentation_report.md")
    return {
        "Test IoU": _find(r"\*\*Test \(held-out.*?\)\*\*\s*\|\s*\*\*([\d.]+)\*\*", text),
        "Test Dice": _find(r"\*\*Test \(held-out.*?\)\*\*\s*\|\s*\*\*[\d.]+\*\*\s*\|\s*\*\*([\d.]+)\*\*", text),
    }


def classifier_metrics() -> dict:
    text = _read(DOCS / "model_card.md")
    return {
        "Test AUC": _find(r"\*\*Test AUC\*\*\s*\|\s*\*\*([\d.]+)\*\*", text),
        "Sensitivity @ threshold 0.5": _find(r"Sensitivity @ threshold 0\.5\s*\|\s*([\d.]+)", text),
        "Specificity @ threshold 0.5": _find(r"Specificity @ threshold 0\.5\s*\|\s*([\d.]+)", text),
        "Sensitivity @ ~95% specificity": _find(
            r"\*\*Sensitivity @ ~95% specificity\*\*\s*\|\s*\*\*([\d.]+)\*\*", text
        ),
        "Calibration ECE (before temp scaling)": _find(r"Before temperature scaling\s*\|\s*([\d.]+)", text),
        "Calibration ECE (after temp scaling)": _find(r"After temperature scaling.*?\|\s*([\d.]+)", text),
    }


def fairness_table() -> list[dict]:
    text = _read(DOCS / "model_card.md")
    rows = re.findall(
        r"\|\s*(\w+_quartile)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*\**([\d.]+)\**\s*\|\s*([\d.]+)\s*\|",
        text,
    )
    return [
        {"group": g, "n": n, "n_malignant": nm, "sensitivity": sens, "specificity": spec}
        for g, n, nm, sens, spec in rows
    ]


def rag_metrics() -> dict:
    text = _read(EVAL / "rag_eval_report.md")
    return {
        "Mean precision@3": _find(r"Mean precision@3\s*\|\s*([\d.]+%)", text),
        "Citation coverage": _find(r"Citation coverage\s*\|\s*([\d.]+%)", text),
        "Mean faithfulness (Day 10, extractive)": _find(r"Mean faithfulness\s*\|\s*([\d.]+%)", text),
    }


def agent_metrics() -> dict:
    text = _read(EVAL / "agent_eval_report.md")
    return {
        "LLM-judge faithfulness (Day 11, generated)": _find(
            r"\*\*LLM-judge faithfulness:\s*([\d/]+\s*\([\d.]+%\))\*\*", text
        ),
        "Human-corrected faithfulness (Day 11)": _find(
            r"\*\*Human-corrected faithfulness:\s*([\d/]+\s*\([\d.]+%\))\*\*", text
        ),
    }


def _print_section(title: str, metrics: dict, source: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for k, v in metrics.items():
        flag = "  [!] not found in report" if v == "MISSING" else ""
        print(f"  {k}: {v}{flag}")
    print(f"  (source: {source})")


def main() -> None:
    print("=" * 60)
    print("SkinScope AU -- consolidated evaluation summary")
    print("All numbers below are read from already-reviewed reports,")
    print("not recomputed on every run. See each report for full methodology.")
    print("=" * 60)

    _print_section("Segmentation (DeepLabV3+, ResNet-34)", segmentation_metrics(), "evaluation/segmentation_report.md")
    _print_section("Classifier (EfficientNet-B0)", classifier_metrics(), "docs/model_card.md")

    print("\nFairness audit (classifier, by L*-lightness quartile)")
    print("-" * 54)
    rows = fairness_table()
    if not rows:
        print("  [!] not found in docs/model_card.md")
    else:
        print(f"  {'group':<22}{'n':>6}{'n_mal':>8}{'sensitivity':>13}{'specificity':>13}")
        for r in rows:
            print(f"  {r['group']:<22}{r['n']:>6}{r['n_malignant']:>8}{r['sensitivity']:>13}{r['specificity']:>13}")
    print("  (source: docs/model_card.md)")

    _print_section("RAG retrieval (Day 10, extractive)", rag_metrics(), "evaluation/rag_eval_report.md")
    _print_section("Agent faithfulness (Day 11, Groq-generated)", agent_metrics(), "evaluation/agent_eval_report.md")

    print("\n" + "=" * 60)
    print("The single most important number in this project: at the")
    print("~95%-specificity operating point, the classifier still misses")
    print("about HALF of actual malignant lesions. See docs/model_card.md.")
    print("=" * 60)


if __name__ == "__main__":
    main()
