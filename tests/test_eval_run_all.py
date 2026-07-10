"""
Guards evaluation/run_all.py's report-parsing regexes against silently
breaking if a report's wording changes -- MISSING would print in the
consolidated summary but wouldn't otherwise be caught by anything.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from run_all import (  # noqa: E402
    agent_metrics,
    classifier_metrics,
    fairness_table,
    rag_metrics,
    segmentation_metrics,
)


def _assert_no_missing(metrics: dict):
    missing = [k for k, v in metrics.items() if v == "MISSING"]
    assert not missing, f"run_all.py failed to parse: {missing}"


def test_segmentation_metrics_parse():
    _assert_no_missing(segmentation_metrics())


def test_classifier_metrics_parse():
    _assert_no_missing(classifier_metrics())


def test_fairness_table_parses_all_four_quartiles():
    rows = fairness_table()
    assert len(rows) == 4
    groups = {r["group"] for r in rows}
    assert groups == {
        "lightest_quartile",
        "light_medium_quartile",
        "dark_medium_quartile",
        "darkest_quartile",
    }


def test_rag_metrics_parse():
    _assert_no_missing(rag_metrics())


def test_agent_metrics_parse():
    _assert_no_missing(agent_metrics())
