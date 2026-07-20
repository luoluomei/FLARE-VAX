#!/usr/bin/env python3
"""Collect the archived summary JSON files into compact comparison CSVs."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUM = ROOT / "results" / "summaries"
OUT = ROOT / "results"


def load(name: str):
    return json.loads((SUM / name).read_text())


def metric_row(group, method, variant, n, metrics, notes=""):
    return {
        "comparison_group": group,
        "method": method,
        "variant": variant,
        "test_n": n,
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "specificity": metrics.get("specificity"),
        "f1": metrics.get("f1"),
        "roc_auc": metrics.get("roc_auc"),
        "average_precision": metrics.get("average_precision"),
        "brier": metrics.get("brier"),
        "log_loss": metrics.get("log_loss"),
        "notes": notes,
    }


def main():
    rows=[]
    v1=load("hbm2_v1_smoke_test.json")
    rows.append(metric_row("development", "HBM2 three-call", "v1 smoke test", 6, v1["test_metrics"], v1["note"]))

    v2=load("hbm2_async_1000.json")
    rows.append(metric_row("development", "HBM2 three-call", "V1-Async concurrent", 300, v2["test_metrics"], "Balanced 1,000-person sample; binary decisions."))

    v3=load("hbm2_reflection_1000.json")
    rows.append(metric_row("development", "HBM2 reflection", "calibrated threshold", 300, v3["test_metrics_calibrated"], "Natural class distribution; reflection memory and threshold calibration."))

    v4=load("hbm2_dual_memory_1000.json")
    rows.append(metric_row("development", "HBM2 pattern anchor", "dual memory selected threshold", 250, v4["metrics"]["test_final_selected"], "Pattern anchor plus prototype/reflection memory."))
    rows.append(metric_row("development", "HBM2 pattern anchor", "pattern-only selected threshold", 250, v4["metrics"]["test_pattern_only_selected"], "Ablation using the learned pattern anchor only."))

    hp=load("hbm5_prior_vaccine_5000.json")
    rows.append(metric_row("with_other_vaccine_history", "HBM5", "pattern only", 2000, hp["metrics"]["test_pattern_only_selected"], "Prior-vaccine behavior contributes to the vaccine-acceptance/benefit proxy."))
    rows.append(metric_row("with_other_vaccine_history", "HBM5", "LLM with reflective memory", 2000, hp["metrics"]["test_with_memory_selected"], "One residual decision call plus frozen reflective memory."))

    hn=load("hbm5_no_prior_vaccine_5000.json")
    for key,label in [
        ("test_hbm8_pattern_only","pattern only"),
        ("test_llm_without_memory","LLM without memory"),
        ("test_llm_with_memory","LLM with reflective memory"),
    ]:
        rows.append(metric_row("without_other_vaccine_history", "HBM5-NV", label, 2000, hn["metrics"][key], "All non-target vaccine variables are explicitly excluded."))

    reported = pd.DataFrame(rows)
    reported.to_csv(OUT/"reported_llm_hbm_metrics.csv",index=False)

    baseline = pd.DataFrame([
        ["without_other_vaccine_history",67,"logistic",0.6766,0.7433,0.6640],
        ["without_other_vaccine_history",67,"random_forest",0.6829,0.7507,0.6660],
        ["without_other_vaccine_history",67,"gradient_boosting",0.6867,0.7553,0.6691],
        ["without_other_vaccine_history",67,"svm",0.6786,0.7477,0.6636],
        ["without_other_vaccine_history",67,"mlp",0.6824,0.7506,0.6580],
        ["without_other_vaccine_history",67,"knn",0.6289,0.6708,0.6119],
        ["without_other_vaccine_history",67,"xgboost",0.6871,0.7570,0.6697],
        ["with_other_vaccine_history",75,"logistic",0.7604,0.8398,0.7484],
        ["with_other_vaccine_history",75,"random_forest",0.7617,0.8377,0.7471],
        ["with_other_vaccine_history",75,"gradient_boosting",0.7630,0.8444,0.7506],
        ["with_other_vaccine_history",75,"svm",0.7630,0.8356,0.7552],
        ["with_other_vaccine_history",75,"mlp",0.7587,0.8402,0.7509],
        ["with_other_vaccine_history",75,"knn",0.7102,0.7689,0.6981],
        ["with_other_vaccine_history",75,"xgboost",0.7647,0.8452,0.7519],
    ],columns=["scenario","n_features","model","accuracy","roc_auc","f1_positive"])
    baseline["source"]="user-supplied baseline console output"
    baseline.to_csv(OUT/"baseline_all_models.csv",index=False)

    # Main compact tables: two representative ML models plus HBM/LLM variants.
    for scenario in ["without_other_vaccine_history","with_other_vaccine_history"]:
        llm = reported[reported.comparison_group==scenario][["method","variant","test_n","accuracy","roc_auc","f1","notes"]].copy()
        llm["model_family"]="HBM/LLM"
        ml = baseline[(baseline.scenario==scenario)&baseline.model.isin(["logistic","xgboost"])].copy()
        ml = ml.rename(columns={"model":"variant","f1_positive":"f1"})
        ml["method"]="ML baseline"
        ml["test_n"]=pd.NA
        ml["notes"]="Reported baseline result; exact split metadata was not included in the supplied output."
        ml["model_family"]="ML"
        compact=pd.concat([
            llm[["model_family","method","variant","test_n","accuracy","roc_auc","f1","notes"]],
            ml[["model_family","method","variant","test_n","accuracy","roc_auc","f1","notes"]],
        ],ignore_index=True)
        compact.to_csv(OUT/f"benchmark_{scenario}.csv",index=False)

    print("Wrote comparison tables to", OUT)


if __name__ == "__main__":
    main()
