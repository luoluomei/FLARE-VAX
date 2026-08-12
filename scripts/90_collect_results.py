#!/usr/bin/env python3
"""Combine the curated public metric tables into one long comparison CSV."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    frames: list[pd.DataFrame] = []

    ml = pd.read_csv(RESULTS / "ml" / "baseline_all_models.csv")
    ml = ml.rename(columns={"scenario": "feature_setting", "f1_positive": "f1"})
    ml["result_group"] = "ml"
    ml["version"] = ml["feature_setting"].map(
        {
            "with_other_vaccine_history": "V4",
            "without_other_vaccine_history": "V5",
        }
    )
    ml["method"] = ml["model"]
    ml["status"] = "complete"
    frames.append(ml)

    icl = pd.read_csv(RESULTS / "icl" / "benchmark_results_public.csv")
    icl["result_group"] = "icl"
    frames.append(icl)

    transfer = pd.read_csv(
        RESULTS / "transfer_baselines" / "benchmark_results_public.csv"
    )
    transfer["result_group"] = "transfer_baseline"
    frames.append(transfer)

    flare = pd.read_csv(RESULTS / "flare_vax" / "reported_metrics.csv")
    flare["result_group"] = "flare_vax"
    frames.append(flare)

    reward = pd.read_csv(RESULTS / "reward_memory" / "benchmark_results_public.csv")
    reward = reward.rename(columns={"variant": "version", "threshold": "selected_threshold"})
    reward["version"] = reward["version"].str.upper()
    reward["result_group"] = "reward_memory"
    reward["status"] = "complete"
    reward["source"] = "results/reward_memory/benchmark_results_public.csv"
    frames.append(reward)

    trbm = pd.read_csv(RESULTS / "trbm" / "full_results_public.csv")
    trbm = trbm.rename(columns={"variant": "version", "threshold": "selected_threshold"})
    trbm["version"] = trbm["version"].str.upper()
    trbm["result_group"] = "trbm"
    trbm["status"] = "complete"
    trbm["source"] = "results/trbm/full_results_public.csv"
    frames.append(trbm)

    columns = [
        "result_group",
        "version",
        "feature_setting",
        "model",
        "method",
        "test_n",
        "selected_threshold",
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "f1",
        "status",
        "note",
        "source",
    ]
    normalized: list[pd.DataFrame] = []
    for frame in frames:
        current = frame.copy()
        for column in columns:
            if column not in current.columns:
                current[column] = pd.NA
        normalized.append(current[columns])

    combined = pd.concat(normalized, ignore_index=True)
    output = RESULTS / "all_results_public.csv"
    combined.to_csv(output, index=False)
    print(f"Wrote {len(combined)} rows to {output}")


if __name__ == "__main__":
    main()
