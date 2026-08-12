#!/usr/bin/env python3
"""
FLARE-VAX LLM In-Context Learning Benchmark — ASU Research Computing API

This standalone benchmark evaluates both V4 and V5 feature policies with:

1. zero_shot_direct
2. random_balanced_8shot_direct
3. similarity_selected_8shot_direct
4. representative_8shot_direct
5. random_balanced_8shot_generic_cot

Each method is evaluated with every requested LLM (intended defaults:
Llama 4 Scout 17B and an available Llama 3 70B model).

Experimental isolation
----------------------
- Memory/train split: demonstration pool and selection fitting only.
- Calibration split: select a classification threshold.
- Test split: final evaluation with frozen selection logic and threshold.
- All demonstration labels come only from the memory/train split.
- V4 allows the prior-vaccine variables used by the V4 pipeline.
- V5 explicitly excludes all non-target vaccine-history variables.
- No HBM proxy score, HBM pattern, pattern base rate, reflective memory, or
  FLARE correction rule is supplied to any benchmark prompt.

ASU endpoint default:
    https://openai.rc.asu.edu/v1

The exact model ID available to an account can vary. The script calls /v1/models,
resolves aliases when possible, and saves model_resolution.json.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import httpx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    pairwise_distances_argmin_min,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore

VERSION = "flare_vax_llm_icl_benchmark_asu_v1"
TARGET = "SHTFLU12M_A"
WEIGHT = "WTFA_A"
ID_COLUMNS = ["HHX", "SRVY_YR", "PSTRAT", "PPSU"]

METHODS = [
    "zero_shot_direct",
    "random_balanced_8shot_direct",
    "similarity_selected_8shot_direct",
    "representative_8shot_direct",
    "random_balanced_8shot_generic_cot",
]

DIRECT_METHODS = {
    "zero_shot_direct",
    "random_balanced_8shot_direct",
    "similarity_selected_8shot_direct",
    "representative_8shot_direct",
}
COT_METHOD = "random_balanced_8shot_generic_cot"


CHRONIC_VARS = [
    "HYPEV_A", "CHDEV_A", "ANGEV_A", "MIEV_A", "STREV_A", "ASTILL_A",
    "CANEV_A", "DIBEV_A", "COPDEV_A", "KIDWEAKEV_A", "LIVEREV_A",
]

BACKGROUND_COLUMNS = [
    "AGEP_A", "SEX_A", "PHSTAT_A", "ANYDIFF_A", "DISAB3_A", "HLTHCOND_A",
    "HISPALLP_A", "RACEALLP_A", "EDUCP_A", "RATCAT_A", "REGION",
    "BMICAT_A", "SMKCIGST_A",
]

BARRIER_COLUMNS = [
    "HICOV_A", "NOTCOV_A", "HINOTYR_A", "HINOTMYR_A", "RSNHICOST_A",
    "HISTOPCOST_A", "MEDDL12M_A", "MEDNG12M_A", "RXDL12M_A", "RXDG12M_A",
    "PAYWORRY_A", "PAYBLL12M_A", "PAYNOBLLNW_A", "TRANSPOR_A", "COMDIFF_A",
    "PRDEDUC1_A", "PRDEDUC2_A",
]

CONTACT_COLUMNS = [
    "LASTDR_A", "WELLNESS_A", "WELLVIS_A", "RETAILHC12MTC_A",
    "VIRAPP12M_A", "URGCC12MTC_A", "EMERG12MTC_A", "HOSPONGT_A",
]

NAVIGATION_COLUMNS = [
    "USUALPL_A", "USPLKIND_A", "ACCSSINT_A", "ACCSSHOM_A",
    "HITLOOK_A", "HITCOMM_A", "HITTEST_A",
]

V4_VACCINE_COLUMNS = [
    "SHTCVD191_A", "SHTCVD19NM2_A", "SHTPNUEV_A",
    "SHTSHINGL1_A", "SHINGRIX3_A", "SHTHEPA_A",
]

BASE_FEATURE_COLUMNS = sorted(set(
    BACKGROUND_COLUMNS
    + CHRONIC_VARS
    + BARRIER_COLUMNS
    + CONTACT_COLUMNS
    + NAVIGATION_COLUMNS
))

FEATURE_COLUMNS_BY_VARIANT = {
    "v4": sorted(set(BASE_FEATURE_COLUMNS + V4_VACCINE_COLUMNS)),
    "v5": sorted(set(BASE_FEATURE_COLUMNS)),
}

ALL_REQUIRED_COLUMNS = sorted(set(
    ID_COLUMNS + [TARGET, WEIGHT] + FEATURE_COLUMNS_BY_VARIANT["v4"]
))

# V5 explicitly excludes every other-vaccine-history variable.
FORBIDDEN_V5_VACCINE_COLUMNS = {
    "SHTCVD191_A", "SHTCVD19NM2_A", "SHTPNUEV_A", "SHTPNEUNB_A",
    "SHTSHINGL1_A", "SHINGRIX3_A", "SHTHEPA_A", "SHTFLUM_A", "SHTFLUY_A",
}

YES_NO_COLUMNS = set(
    CHRONIC_VARS
    + [
        "ANYDIFF_A", "DISAB3_A", "HLTHCOND_A",
        "HICOV_A", "NOTCOV_A", "HINOTYR_A", "RSNHICOST_A",
        "HISTOPCOST_A", "MEDDL12M_A", "MEDNG12M_A", "RXDL12M_A",
        "RXDG12M_A", "PAYBLL12M_A", "PAYNOBLLNW_A", "TRANSPOR_A",
        "PRDEDUC1_A", "PRDEDUC2_A", "WELLNESS_A", "VIRAPP12M_A",
        "HOSPONGT_A", "ACCSSINT_A", "ACCSSHOM_A", "HITLOOK_A",
        "HITCOMM_A", "HITTEST_A",
        "SHTCVD191_A", "SHTPNUEV_A", "SHTSHINGL1_A", "SHINGRIX3_A", "SHTHEPA_A",
    ]
)

FEATURE_NAMES = {
    "AGEP_A": "age_years",
    "SEX_A": "sex",
    "PHSTAT_A": "self_rated_health",
    "ANYDIFF_A": "functional_difficulty",
    "DISAB3_A": "disability",
    "HLTHCOND_A": "immune_or_health_vulnerability_indicator",
    "HISPALLP_A": "hispanic_group",
    "RACEALLP_A": "race_group",
    "EDUCP_A": "education",
    "RATCAT_A": "income_to_poverty_ratio_category",
    "REGION": "us_region",
    "BMICAT_A": "bmi_category",
    "SMKCIGST_A": "smoking_status",

    "HYPEV_A": "hypertension",
    "CHDEV_A": "coronary_heart_disease",
    "ANGEV_A": "angina",
    "MIEV_A": "heart_attack_history",
    "STREV_A": "stroke_history",
    "ASTILL_A": "current_asthma",
    "CANEV_A": "cancer_history",
    "DIBEV_A": "diabetes",
    "COPDEV_A": "copd",
    "KIDWEAKEV_A": "weak_kidneys",
    "LIVEREV_A": "liver_condition",

    "SHTCVD191_A": "covid_vaccinated",
    "SHTCVD19NM2_A": "covid_dose_category",
    "SHTPNUEV_A": "pneumonia_vaccinated",
    "SHTSHINGL1_A": "shingles_vaccinated",
    "SHINGRIX3_A": "shingrix_vaccinated",
    "SHTHEPA_A": "hepatitis_a_vaccinated",

    "HICOV_A": "currently_insured",
    "NOTCOV_A": "explicitly_not_covered",
    "HINOTYR_A": "uninsured_in_past_year",
    "HINOTMYR_A": "months_uninsured",
    "RSNHICOST_A": "coverage_unaffordable",
    "HISTOPCOST_A": "coverage_stopped_due_to_cost",
    "MEDDL12M_A": "delayed_medical_care_due_to_cost",
    "MEDNG12M_A": "needed_medical_care_not_received_due_to_cost",
    "RXDL12M_A": "delayed_prescription_due_to_cost",
    "RXDG12M_A": "needed_prescription_not_received_due_to_cost",
    "PAYWORRY_A": "medical_bill_worry_category",
    "PAYBLL12M_A": "medical_bill_problem",
    "PAYNOBLLNW_A": "unable_to_pay_medical_bills_now",
    "TRANSPOR_A": "transportation_barrier",
    "COMDIFF_A": "communication_difficulty",
    "PRDEDUC1_A": "deductible_plan_1",
    "PRDEDUC2_A": "deductible_plan_2",

    "LASTDR_A": "last_doctor_visit",
    "WELLNESS_A": "wellness_visit_indicator",
    "WELLVIS_A": "wellness_visit_recency",
    "RETAILHC12MTC_A": "retail_clinic_visit_category",
    "VIRAPP12M_A": "virtual_appointment",
    "URGCC12MTC_A": "urgent_care_visit_category",
    "EMERG12MTC_A": "emergency_visit_category",
    "HOSPONGT_A": "overnight_hospitalization",

    "USUALPL_A": "usual_care_place",
    "USPLKIND_A": "usual_care_setting",
    "ACCSSINT_A": "internet_access_anywhere",
    "ACCSSHOM_A": "internet_access_at_home",
    "HITLOOK_A": "looked_up_health_information_online",
    "HITCOMM_A": "communicated_with_doctor_online",
    "HITTEST_A": "viewed_test_results_online",
}

HEALTH_STATUS = {1: "excellent", 2: "very_good", 3: "good", 4: "fair", 5: "poor"}
SEX = {1: "male", 2: "female"}
REGION = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}
BMI = {1: "underweight", 2: "healthy_weight", 3: "overweight", 4: "obese"}
SMOKING = {1: "every_day", 2: "some_days", 3: "not_at_all", 4: "other_or_former_category"}
EDUCATION = {
    0: "never_attended_or_kindergarten",
    1: "grades_1_11",
    2: "12th_grade_no_diploma",
    3: "GED",
    4: "high_school_graduate",
    5: "some_college",
    6: "technical_associate",
    7: "academic_associate",
    8: "bachelor",
    9: "master",
    10: "professional_or_doctoral",
}
POVERTY_RATIO = {
    1: "0.00-0.49", 2: "0.50-0.74", 3: "0.75-0.99", 4: "1.00-1.24",
    5: "1.25-1.49", 6: "1.50-1.74", 7: "1.75-1.99", 8: "2.00-2.49",
    9: "2.50-2.99", 10: "3.00-3.49", 11: "3.50-3.99", 12: "4.00-4.49",
    13: "4.50-4.99", 14: "5.00_or_greater",
}
LAST_VISIT = {
    0: "never", 1: "within_past_year", 2: "1_to_under_2_years",
    3: "2_to_under_3_years", 4: "3_to_under_5_years",
    5: "5_to_under_10_years", 6: "10_or_more_years",
}
USUAL_PLACE = {1: "one_usual_place", 2: "no_usual_place", 3: "more_than_one_usual_place"}
USUAL_KIND = {
    1: "doctor_office_or_health_center",
    2: "urgent_care_or_retail_clinic",
    3: "hospital_emergency_room",
    4: "VA_facility",
    5: "other_place",
    6: "no_single_place_used_most_often",
}
COMM_DIFFICULTY = {1: "none", 2: "some", 3: "a_lot", 4: "cannot_do_at_all"}

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def numeric_value(value: Any, invalid: Iterable[int] = (7, 8, 9, 97, 98, 99)) -> Optional[float]:
    try:
        x = float(value)
        if not np.isfinite(x):
            return None
        if int(x) in set(invalid):
            return None
        return x
    except Exception:
        return None


def yes_no_value(value: Any) -> Optional[bool]:
    try:
        iv = int(float(value))
        if iv == 1:
            return True
        if iv == 2:
            return False
    except Exception:
        pass
    return None


def code_value(value: Any, mapping: Mapping[int, str], valid: Optional[Iterable[int]] = None) -> Optional[str]:
    try:
        iv = int(float(value))
        if valid is not None and iv not in set(valid):
            return None
        return mapping.get(iv, f"code_{iv}")
    except Exception:
        return None


def clean_feature_value(column: str, value: Any) -> Any:
    if column in YES_NO_COLUMNS:
        return yes_no_value(value)

    if column == "AGEP_A":
        x = numeric_value(value, invalid=(97, 98, 99))
        return None if x is None else int(x)
    if column == "SEX_A":
        return code_value(value, SEX, [1, 2])
    if column == "PHSTAT_A":
        return code_value(value, HEALTH_STATUS, [1, 2, 3, 4, 5])
    if column == "REGION":
        return code_value(value, REGION, [1, 2, 3, 4])
    if column == "BMICAT_A":
        return code_value(value, BMI, [1, 2, 3, 4])
    if column == "SMKCIGST_A":
        return code_value(value, SMOKING, [1, 2, 3, 4])
    if column == "EDUCP_A":
        return code_value(value, EDUCATION, range(0, 11))
    if column == "RATCAT_A":
        return code_value(value, POVERTY_RATIO, range(1, 15))
    if column in {"LASTDR_A", "WELLVIS_A"}:
        return code_value(value, LAST_VISIT, range(0, 7))
    if column == "USUALPL_A":
        return code_value(value, USUAL_PLACE, [1, 2, 3])
    if column == "USPLKIND_A":
        return code_value(value, USUAL_KIND, [1, 2, 3, 4, 5, 6])
    if column == "COMDIFF_A":
        return code_value(value, COMM_DIFFICULTY, [1, 2, 3, 4])

    # Sensitive categorical variables are kept as codes when explicitly enabled.
    if column in {"RACEALLP_A", "HISPALLP_A"}:
        x = numeric_value(value)
        return None if x is None else int(x)

    # Preserve interpretable category/count values without guessing labels that are
    # not explicitly defined in the V4/V5 source.
    x = numeric_value(value)
    if x is None:
        return None
    return int(x) if float(x).is_integer() else float(x)

def target_to_binary(value: Any) -> Optional[int]:
    y = yes_no_value(value)
    return None if y is None else int(y)


def config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def append_jsonl(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(obj), ensure_ascii=False, default=json_safe) + "\n")


def load_latest_jsonl(path: Path, expected_hash: str) -> Dict[int, Dict[str, Any]]:
    latest: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get("config_hash") == expected_hash and "data_idx" in obj:
                    latest[int(obj["data_idx"])] = obj
            except Exception:
                continue
    return latest


def usage_from_response(response: Any) -> Dict[str, int]:
    u = getattr(response, "usage", None)
    if u is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    inp = int(
        getattr(u, "prompt_tokens", None)
        or getattr(u, "input_tokens", 0)
        or 0
    )
    out = int(
        getattr(u, "completion_tokens", None)
        or getattr(u, "output_tokens", 0)
        or 0
    )
    total = int(getattr(u, "total_tokens", inp + out) or (inp + out))
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": total}


def sum_usage(entries: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for e in entries:
        u = e.get("usage") or {}
        for k in total:
            total[k] += int(u.get(k, 0) or 0)
    return total


class PredictionCallFailure(RuntimeError):
    """Final per-respondent prediction failure with structured diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        category: str,
        raw_response: str = "",
        finish_reason: str = "",
        usage: Optional[Mapping[str, int]] = None,
        attempt_count: int = 0,
        request_id: str = "",
    ) -> None:
        super().__init__(message)
        self.category = category
        self.raw_response = raw_response
        self.finish_reason = finish_reason
        self.usage = dict(usage or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        self.attempt_count = int(attempt_count)
        self.request_id = request_id


def _add_usage(total: Dict[str, int], usage: Mapping[str, Any]) -> None:
    for key in ["input_tokens", "output_tokens", "total_tokens"]:
        total[key] += int(usage.get(key, 0) or 0)


def _classify_api_exception(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "ratelimit" in name or "429" in msg or "rate limit" in msg:
        return "api_rate_limit"
    if "timeout" in name or "timed out" in msg:
        return "api_timeout"
    if "connection" in name or "connect" in name or "connection" in msg:
        return "api_connection"
    if "authentication" in name or "401" in msg:
        return "api_authentication"
    if "permission" in name or "403" in msg:
        return "api_permission"
    return "api_other"


def _json_candidates(text: str) -> List[Dict[str, Any]]:
    """Find valid JSON objects even when the model writes prose before/after them."""
    decoder = json.JSONDecoder()
    candidates: List[Dict[str, Any]] = []

    # Whole-output JSON first.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            candidates.append(obj)
    except Exception:
        pass

    # Scan from every opening brace. raw_decode stops at the end of one object,
    # so prose or another JSON object after it does not break parsing.
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
            if isinstance(obj, dict):
                candidates.append(obj)
        except Exception:
            continue
    return candidates


def extract_prediction_fields(raw_text: str) -> Tuple[Dict[str, Any], str]:
    """
    Robustly recover the requested prediction from the model output.

    Accepted without guessing:
    1. strict JSON;
    2. prose followed by a valid JSON object;
    3. fenced JSON;
    4. explicit `probability_yes: 72` / `prediction: YES` fields.

    A probability is always required. We never invent one from prose.
    """
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("empty model output")

    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()

    candidates = _json_candidates(cleaned)
    for obj in reversed(candidates):
        if "probability_yes" in obj:
            return obj, "json_object"

    # Conservative field fallback for JSON-like but technically invalid output.
    prob_match = re.search(
        r"(?i)[\"']?probability_yes[\"']?\s*[:=]\s*[\"']?(-?\d+(?:\.\d+)?)",
        cleaned,
    )
    pred_match = re.search(
        r"(?i)[\"']?prediction[\"']?\s*[:=]\s*[\"']?(YES|NO)[\"']?",
        cleaned,
    )
    if prob_match:
        obj: Dict[str, Any] = {"probability_yes": float(prob_match.group(1))}
        if pred_match:
            obj["prediction"] = pred_match.group(1).upper()
        return obj, "explicit_field_regex"

    raise ValueError("No parseable probability_yes field was found in the model output")



NUMERIC_SELECTION_COLUMNS = {
    "AGEP_A",
    "HINOTMYR_A",
    "SHTCVD19NM2_A",
    "RETAILHC12MTC_A",
    "URGCC12MTC_A",
    "EMERG12MTC_A",
}

FORBIDDEN_V5_VACCINE_COLUMNS = {
    "SHTCVD191_A", "SHTCVD19NM2_A", "SHTPNUEV_A", "SHTPNEUNB_A",
    "SHTSHINGL1_A", "SHINGRIX3_A", "SHTHEPA_A", "SHTFLUM_A", "SHTFLUY_A",
}


def feature_columns_for_variant(variant: str, include_sensitive: bool) -> List[str]:
    cols = list(FEATURE_COLUMNS_BY_VARIANT[variant])
    if not include_sensitive:
        cols = [c for c in cols if c not in {"RACEALLP_A", "HISPALLP_A"}]
    if variant == "v5":
        leaked = FORBIDDEN_V5_VACCINE_COLUMNS.intersection(cols)
        if leaked:
            raise AssertionError(f"V5 feature policy leaked vaccine history: {sorted(leaked)}")
    return cols


def build_raw_llm_profile(
    row: pd.Series,
    *,
    variant: str,
    feature_columns: Sequence[str],
) -> Dict[str, Any]:
    profile: Dict[str, Any] = {}
    for col in feature_columns:
        profile[FEATURE_NAMES.get(col, col)] = clean_feature_value(col, row.get(col))
    if variant == "v5":
        forbidden_names = {FEATURE_NAMES.get(c, c) for c in FORBIDDEN_V5_VACCINE_COLUMNS}
        overlap = forbidden_names.intersection(profile)
        if overlap:
            raise AssertionError(f"V5 prompt leaked vaccine-history variables: {sorted(overlap)}")
    return profile


DIRECT_SYSTEM_PROMPT = """You are a binary prediction model for an NHIS survey outcome.

TASK
Predict whether the target respondent received an influenza vaccination during the past 12 months.

BASELINE CONSTRAINTS
- Use only the supplied observed variables and, when present, labeled examples.
- Do not assume or invent missing values.
- Do not use an HBM framework, HBM proxy scores, meta-dimensions, patterns,
  pattern base rates, reflective memory, retrieval rules, or reflection.
- Do not provide analysis, reasoning, explanations, Markdown, or commentary.

OUTPUT CONTRACT
- Your first character must be { and your final character must be }.
- Return exactly one JSON object:
  {"probability_yes": 72.0, "prediction": "YES"}
- probability_yes must be a number from 0 to 100.
- prediction must be YES or NO.
- 72.0 is a formatting example only and is not a prior.
"""

COT_SYSTEM_PROMPT = """You are a binary prediction model for an NHIS survey outcome.

TASK
Predict whether the target respondent received an influenza vaccination during the past 12 months.

METHOD
Use generic step-by-step evidence comparison. This is NOT an HBM analysis.
1. Briefly summarize observed evidence that supports vaccination.
2. Briefly summarize observed evidence that supports non-vaccination.
3. Compare the target with the labeled examples without inventing facts.
4. Map the balance of evidence to a probability and binary prediction.

BASELINE CONSTRAINTS
- Use only the supplied observed variables and labeled examples.
- Do not assume missing values.
- Do not mention HBM, proxy scores, meta-dimensions, patterns, pattern base rates,
  reflective memory, retrieval rules, or FLARE memory.
- Keep each reasoning field concise and grounded in observed values.

OUTPUT CONTRACT
Return exactly one JSON object with this shape:
{
  "reasoning_trace": {
    "evidence_for_vaccination": "concise text",
    "evidence_against_vaccination": "concise text",
    "comparison_to_examples": "concise text",
    "decision_mapping": "concise text"
  },
  "probability_yes": 72.0,
  "prediction": "YES"
}
The first character must be { and the final character must be }.
"""

STRICT_RETRY_DIRECT = """
FORMAT RETRY: Return the JSON object immediately. Do not write analysis or prose.
The first character must be {.
"""

STRICT_RETRY_COT = """
FORMAT RETRY: Return exactly the required JSON object with reasoning_trace,
probability_yes, and prediction. Do not use Markdown or prose outside JSON.
"""


def build_prompt(
    method: str,
    demos: Sequence[Mapping[str, Any]],
    target_profile: Mapping[str, Any],
    *,
    strict_retry: bool = False,
) -> str:
    payload: Dict[str, Any] = {
        "outcome": "received_influenza_vaccination_in_past_12_months",
        "method": method,
        "target_respondent": {"features": target_profile, "label": "UNKNOWN"},
    }
    if method != "zero_shot_direct":
        payload["labeled_examples"] = [
            {
                "features": d["features"],
                "label": "YES" if int(d["label"]) == 1 else "NO",
            }
            for d in demos
        ]
    else:
        payload["labeled_examples"] = []
        payload["zero_shot_note"] = "No labeled examples are provided."

    if method == COT_METHOD:
        suffix = (
            '\nReturn only the required JSON object with four concise reasoning_trace fields, '
            'probability_yes, and prediction.'
        )
        if strict_retry:
            suffix += STRICT_RETRY_COT
    else:
        suffix = (
            '\nReturn only {"probability_yes": <0-100>, "prediction": "YES" or "NO"}. '
            'No reasoning or prose.'
        )
        if strict_retry:
            suffix += STRICT_RETRY_DIRECT
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + suffix


def validate_prediction(obj: Mapping[str, Any], method: str) -> Dict[str, Any]:
    if "probability_yes" not in obj:
        raise ValueError("missing probability_yes")
    p = float(obj["probability_yes"])
    if not np.isfinite(p):
        raise ValueError("probability_yes is not finite")
    p = float(np.clip(p, 0.0, 100.0))
    supplied = str(obj.get("prediction", "")).strip().upper()
    derived = "YES" if p >= 50 else "NO"

    reasoning_trace: Dict[str, str] = {}
    if method == COT_METHOD:
        raw_trace = obj.get("reasoning_trace")
        if not isinstance(raw_trace, Mapping):
            raise ValueError("generic CoT output is missing reasoning_trace")
        required = [
            "evidence_for_vaccination",
            "evidence_against_vaccination",
            "comparison_to_examples",
            "decision_mapping",
        ]
        for key in required:
            value = str(raw_trace.get(key, "")).strip()
            if not value:
                raise ValueError(f"generic CoT reasoning_trace is missing {key}")
            reasoning_trace[key] = value[:1200]

    return {
        "probability_yes": p,
        "raw_prediction": supplied if supplied in {"YES", "NO"} else "",
        "prediction_at_50": derived,
        "prediction_consistency_corrected": bool(supplied in {"YES", "NO"} and supplied != derived),
        "reasoning_trace": reasoning_trace,
    }


def _response_format_for_method(method: str) -> Dict[str, Any]:
    return {"type": "json_object"}


async def probe_json_mode(client: Any, model: str) -> Tuple[bool, str]:
    """Probe JSON mode once per model. Failure falls back to prompt-only JSON."""
    try:
        r = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return only JSON."},
                {"role": "user", "content": 'Return {"ok": true}.'},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=40,
        )
        text = str(r.choices[0].message.content or "") if getattr(r, "choices", None) else ""
        ok = bool(_json_candidates(text))
        return ok, text[:300]
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


async def call_prediction(
    client: Any,
    semaphore: asyncio.Semaphore,
    *,
    model: str,
    method: str,
    demos: Sequence[Mapping[str, Any]],
    target_profile: Mapping[str, Any],
    max_tokens_direct: int,
    max_tokens_cot: int,
    temperature: float,
    retries: int,
    use_json_mode: bool,
) -> Tuple[Dict[str, Any], str, Dict[str, int], str, Dict[str, Any]]:
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    last_raw = ""
    last_finish = ""
    last_request_id = ""
    last_category = "output_format"
    last_message = ""
    system_prompt = COT_SYSTEM_PROMPT if method == COT_METHOD else DIRECT_SYSTEM_PROMPT
    max_tokens = max_tokens_cot if method == COT_METHOD else max_tokens_direct

    for attempt in range(retries + 1):
        prompt = build_prompt(method, demos, target_profile, strict_retry=attempt > 0)
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if use_json_mode:
            kwargs["response_format"] = _response_format_for_method(method)

        try:
            async with semaphore:
                r = await client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_category = _classify_api_exception(exc)
            last_message = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                await asyncio.sleep(min(20.0, 1.5 * (2 ** attempt)))
                continue
            raise PredictionCallFailure(
                f"ASU call failed after {attempt + 1} attempts: {last_message}",
                category=last_category,
                usage=total_usage,
                attempt_count=attempt + 1,
            ) from exc

        if not getattr(r, "choices", None):
            last_category = "empty_response"
            last_message = "ASU Chat Completions returned no choices"
            if attempt < retries:
                continue
            raise PredictionCallFailure(
                last_message,
                category=last_category,
                usage=total_usage,
                attempt_count=attempt + 1,
            )

        usage = usage_from_response(r)
        _add_usage(total_usage, usage)
        choice = r.choices[0]
        raw_text = str(choice.message.content or "").strip()
        finish_reason = str(getattr(choice, "finish_reason", "") or "")
        request_id = str(getattr(r, "_request_id", "") or "")
        last_raw, last_finish, last_request_id = raw_text, finish_reason, request_id

        try:
            extracted, parse_method = extract_prediction_fields(raw_text)
            prediction = validate_prediction(extracted, method)
            return prediction, raw_text, total_usage, request_id, {
                "parse_method": parse_method,
                "finish_reason": finish_reason,
                "attempt_count": attempt + 1,
                "json_mode": bool(use_json_mode),
            }
        except Exception as exc:
            last_category = "output_truncated" if finish_reason.lower() == "length" else "output_format"
            last_message = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                await asyncio.sleep(min(5.0, 0.5 * (attempt + 1)))
                continue

    raise PredictionCallFailure(
        f"Prediction output failed after {retries + 1} attempts: {last_message}",
        category=last_category,
        raw_response=last_raw,
        finish_reason=last_finish,
        usage=total_usage,
        attempt_count=retries + 1,
        request_id=last_request_id,
    )


def safe_split(indices: np.ndarray, labels: pd.Series, left_size: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if left_size <= 0:
        return np.array([], dtype=int), indices.copy()
    if left_size >= len(indices):
        return indices.copy(), np.array([], dtype=int)
    s = labels.loc[indices]
    use_stratify = len(s.value_counts()) > 1 and s.value_counts().min() >= 2
    try:
        left, right = train_test_split(
            indices,
            train_size=left_size,
            random_state=seed,
            stratify=s if use_stratify else None,
        )
    except ValueError:
        left, right = train_test_split(indices, train_size=left_size, random_state=seed)
    return np.asarray(left, dtype=int), np.asarray(right, dtype=int)


def create_fallback_split(raw: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    valid = []
    for src in raw.index:
        y = target_to_binary(raw.loc[src, TARGET])
        if y is not None:
            valid.append({"source_row_index": int(src), "actual": int(y)})
    frame = pd.DataFrame(valid)
    ratios = np.array([args.memory_ratio, args.calibration_ratio, args.test_ratio], dtype=float)
    ratios = ratios / ratios.sum()
    n = len(frame)
    n_memory = max(1, int(round(n * ratios[0])))
    n_cal = max(1, int(round(n * ratios[1]))) if n >= 3 else 0
    if n_memory + n_cal >= n:
        n_cal = max(0, n - n_memory - 1)
    idx = np.arange(n, dtype=int)
    memory, rest = safe_split(idx, frame["actual"], n_memory, args.random_seed)
    calibration, test = safe_split(rest, frame["actual"], n_cal, args.random_seed + 1)
    split = np.full(n, "", dtype=object)
    split[memory] = "memory"
    split[calibration] = "calibration"
    split[test] = "test"
    frame["split"] = split
    return frame


def load_reference_split(path: Path, raw: pd.DataFrame) -> pd.DataFrame:
    ref = pd.read_csv(path)
    if {"source_index", "phase"}.issubset(ref.columns):
        idx_col, split_col = "source_index", "phase"
    elif {"source_row_index", "split"}.issubset(ref.columns):
        idx_col, split_col = "source_row_index", "split"
    else:
        raise ValueError(
            f"Unsupported reference split {path}. Expected source_index+phase "
            "or source_row_index+split."
        )
    out = ref[[idx_col, split_col]].copy()
    out.columns = ["source_row_index", "split"]
    out["source_row_index"] = pd.to_numeric(out["source_row_index"], errors="raise").astype(int)
    out["split"] = out["split"].astype(str).str.lower()
    out = out[out["split"].isin(["memory", "calibration", "test"])].copy()
    if out["source_row_index"].duplicated().any():
        raise ValueError(f"Duplicate source indices in {path}")
    if not set(out["source_row_index"]).issubset(set(raw.index)):
        raise ValueError(f"Reference split contains row indices not present in adult24.csv: {path}")
    out["actual"] = [target_to_binary(raw.loc[i, TARGET]) for i in out["source_row_index"]]
    out = out[out["actual"].isin([0, 1])].reset_index(drop=True)
    return out


def downsample_assignments(assignments: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size <= 0 or sample_size >= len(assignments):
        return assignments.copy()
    frac = sample_size / len(assignments)
    allocations = []
    used = 0
    groups = list(assignments.groupby(["split", "actual"], sort=True))
    for key, g in groups:
        n = max(1, int(round(len(g) * frac)))
        n = min(n, len(g))
        allocations.append([key, n, g])
        used += n
    while used > sample_size:
        changed = False
        for item in reversed(allocations):
            if item[1] > 1:
                item[1] -= 1; used -= 1; changed = True
                if used == sample_size: break
        if not changed: break
    while used < sample_size:
        changed = False
        for item in allocations:
            if item[1] < len(item[2]):
                item[1] += 1; used += 1; changed = True
                if used == sample_size: break
        if not changed: break
    pieces = [g.sample(n=n, random_state=seed + j) for j, (_, n, g) in enumerate(allocations)]
    return pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def prepare_assignments(
    variant: str,
    reference_path: Optional[Path],
    raw: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, str, bool]:
    if reference_path and reference_path.exists():
        assignments = load_reference_split(reference_path, raw)
        source = str(reference_path)
        exact = True
    else:
        if not args.allow_fallback_split:
            raise FileNotFoundError(
                f"{variant.upper()} reference split is missing. Provide the corresponding full-run "
                "split file or pass --allow-fallback-split."
            )
        assignments = create_fallback_split(raw, args)
        source = "fallback_outcome_stratified_40_20_40"
        exact = False

    assignments = downsample_assignments(assignments, args.sample_size, args.random_seed)
    split_order = pd.Categorical(
        assignments["split"], categories=["memory", "calibration", "test"], ordered=True
    )
    assignments = (
        assignments.assign(_order=split_order)
        .sort_values(["_order", "source_row_index"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    assignments["data_idx"] = np.arange(len(assignments), dtype=int)
    return assignments, source, exact


def make_demo(raw: pd.DataFrame, row: Mapping[str, Any], variant: str, feature_columns: Sequence[str]) -> Dict[str, Any]:
    src = int(row["source_row_index"])
    return {
        "source_row_index": src,
        "label": int(row["actual"]),
        "features": build_raw_llm_profile(raw.loc[src], variant=variant, feature_columns=feature_columns),
    }


def select_random_balanced(
    assignments: pd.DataFrame,
    raw: pd.DataFrame,
    variant: str,
    feature_columns: Sequence[str],
    seed: int,
    per_class: int = 4,
) -> List[Dict[str, Any]]:
    memory = assignments[assignments["split"] == "memory"]
    chosen_parts = []
    for y in [0, 1]:
        g = memory[memory["actual"] == y]
        if len(g) < per_class:
            raise ValueError(f"Not enough memory examples for class {y}: need {per_class}, found {len(g)}")
        chosen_parts.append(g.sample(n=per_class, random_state=seed + y))
    chosen = pd.concat(chosen_parts, ignore_index=True).sample(frac=1, random_state=seed + 77)
    return [make_demo(raw, r, variant, feature_columns) for _, r in chosen.iterrows()]


def build_selection_preprocessor(feature_columns: Sequence[str]) -> ColumnTransformer:
    numeric_cols = [c for c in feature_columns if c in NUMERIC_SELECTION_COLUMNS]
    categorical_cols = [c for c in feature_columns if c not in numeric_cols]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric_cols), ("categorical", categorical_pipe, categorical_cols)],
        remainder="drop",
    )


@dataclass
class SelectionArtifacts:
    random_demos: List[Dict[str, Any]]
    representative_demos: List[Dict[str, Any]]
    similarity_map: Dict[int, Dict[str, Any]]
    selection_metadata: Dict[str, Any]

    def demos_for(self, method: str, target_source: int, raw: pd.DataFrame, variant: str, feature_columns: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if method == "zero_shot_direct":
            return [], {"selection_method": "zero_shot", "demo_source_rows": []}
        if method in {"random_balanced_8shot_direct", COT_METHOD}:
            return self.random_demos, {
                "selection_method": "random_balanced_fixed",
                "demo_source_rows": [int(d["source_row_index"]) for d in self.random_demos],
            }
        if method == "representative_8shot_direct":
            return self.representative_demos, {
                "selection_method": "representative_kmeans_medoid_fixed",
                "demo_source_rows": [int(d["source_row_index"]) for d in self.representative_demos],
            }
        if method == "similarity_selected_8shot_direct":
            info = self.similarity_map[int(target_source)]
            ordered_sources: List[int] = []
            # Alternate YES and NO to reduce label-order bias.
            for y_src, n_src in zip(info["yes_sources"], info["no_sources"]):
                ordered_sources.extend([int(y_src), int(n_src)])
            lookup = self.selection_metadata["source_to_actual"]
            demos = [
                {
                    "source_row_index": src,
                    "label": int(lookup[str(src)]),
                    "features": build_raw_llm_profile(raw.loc[src], variant=variant, feature_columns=feature_columns),
                }
                for src in ordered_sources
            ]
            return demos, {
                "selection_method": "target_specific_cosine_similarity_balanced",
                "demo_source_rows": ordered_sources,
                "yes_similarities": info["yes_similarities"],
                "no_similarities": info["no_similarities"],
            }
        raise ValueError(f"Unknown method: {method}")


def _representative_rows(
    X: Any,
    assignment_positions: np.ndarray,
    source_rows: np.ndarray,
    *,
    n_representatives: int,
    seed: int,
) -> List[int]:
    if len(assignment_positions) < n_representatives:
        raise ValueError("Not enough rows for representative selection")
    Xc = X[assignment_positions]
    km = KMeans(n_clusters=n_representatives, random_state=seed, n_init=10)
    km.fit(Xc)
    nearest, _ = pairwise_distances_argmin_min(km.cluster_centers_, Xc, metric="euclidean")
    selected = []
    for j in nearest.tolist():
        src = int(source_rows[j])
        if src not in selected:
            selected.append(src)
    if len(selected) < n_representatives:
        # Deterministic fallback for a duplicate medoid.
        for src in source_rows.tolist():
            if int(src) not in selected:
                selected.append(int(src))
            if len(selected) == n_representatives:
                break
    return selected[:n_representatives]


def build_selection_artifacts(
    variant: str,
    assignments: pd.DataFrame,
    raw: pd.DataFrame,
    feature_columns: Sequence[str],
    seed: int,
    output_dir: Path,
    similarity_batch_size: int,
) -> SelectionArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    random_demos = select_random_balanced(assignments, raw, variant, feature_columns, seed)

    selected_raw = raw.loc[assignments["source_row_index"].tolist(), list(feature_columns)].reset_index(drop=True)
    preprocessor = build_selection_preprocessor(feature_columns)
    memory_positions = assignments.index[assignments["split"] == "memory"].to_numpy(dtype=int)
    preprocessor.fit(selected_raw.iloc[memory_positions])
    X = preprocessor.transform(selected_raw)
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    else:
        X = X.tocsr()

    representative_by_class: Dict[int, List[int]] = {}
    for y in [0, 1]:
        mask = (assignments["split"].eq("memory") & assignments["actual"].eq(y)).to_numpy()
        positions = np.where(mask)[0]
        sources = assignments.iloc[positions]["source_row_index"].to_numpy(dtype=int)
        representative_by_class[y] = _representative_rows(
            X, positions, sources, n_representatives=4, seed=seed + y
        )
    # Alternate YES and NO examples to avoid a fixed label-order block.
    representative_sources: List[int] = []
    for yes_src, no_src in zip(representative_by_class[1], representative_by_class[0]):
        representative_sources.extend([int(yes_src), int(no_src)])
    rep_rows = [
        {"source_row_index": src, "actual": int(assignments.loc[assignments["source_row_index"] == src, "actual"].iloc[0])}
        for src in representative_sources
    ]
    representative_demos = [make_demo(raw, r, variant, feature_columns) for r in rep_rows]

    target_positions = assignments.index[assignments["split"].isin(["calibration", "test"])].to_numpy(dtype=int)
    target_sources = assignments.iloc[target_positions]["source_row_index"].to_numpy(dtype=int)
    class_models: Dict[int, Tuple[NearestNeighbors, np.ndarray]] = {}
    for y in [0, 1]:
        positions = assignments.index[(assignments["split"] == "memory") & (assignments["actual"] == y)].to_numpy(dtype=int)
        if len(positions) < 4:
            raise ValueError(f"Need at least four memory examples for class {y}")
        nn = NearestNeighbors(n_neighbors=4, metric="cosine", algorithm="brute", n_jobs=-1)
        nn.fit(X[positions])
        class_models[y] = (nn, positions)

    similarity_map: Dict[int, Dict[str, Any]] = {}
    print(f"[{variant}] Precomputing target-specific similarity supports for {len(target_positions):,} targets...", flush=True)
    for start in range(0, len(target_positions), max(1, similarity_batch_size)):
        batch_pos = target_positions[start:start + similarity_batch_size]
        batch_sources = target_sources[start:start + similarity_batch_size]
        result_by_class = {}
        for y in [0, 1]:
            nn, memory_pos = class_models[y]
            distances, local_indices = nn.kneighbors(X[batch_pos], return_distance=True)
            source_matrix = assignments.iloc[memory_pos[local_indices.ravel()]]["source_row_index"].to_numpy().reshape(local_indices.shape)
            result_by_class[y] = (distances, source_matrix)
        for i, src in enumerate(batch_sources):
            no_d, no_s = result_by_class[0][0][i], result_by_class[0][1][i]
            yes_d, yes_s = result_by_class[1][0][i], result_by_class[1][1][i]
            similarity_map[int(src)] = {
                "no_sources": [int(x) for x in no_s],
                "yes_sources": [int(x) for x in yes_s],
                "no_similarities": [float(1.0 - x) for x in no_d],
                "yes_similarities": [float(1.0 - x) for x in yes_d],
            }
        done = min(start + similarity_batch_size, len(target_positions))
        print(f"[{variant}] similarity support map: {done:,}/{len(target_positions):,}", flush=True)

    source_to_actual = {
        str(int(r.source_row_index)): int(r.actual)
        for r in assignments.itertuples(index=False)
    }
    metadata = {
        "variant": variant,
        "random_demo_sources": [int(d["source_row_index"]) for d in random_demos],
        "random_demo_labels": [int(d["label"]) for d in random_demos],
        "representative_demo_sources": [int(d["source_row_index"]) for d in representative_demos],
        "representative_demo_labels": [int(d["label"]) for d in representative_demos],
        "selection_feature_columns": list(feature_columns),
        "source_to_actual": source_to_actual,
        "similarity_metric": "cosine_on_memory_fitted_imputed_scaled_onehot_features",
        "representative_method": "classwise_kmeans_k4_then_nearest_observed_row",
    }
    (output_dir / "selection_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "random_balanced_8shot.json").write_text(json.dumps(random_demos, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "representative_8shot.json").write_text(json.dumps(representative_demos, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for target, info in similarity_map.items():
        rows.append({
            "target_source_row_index": target,
            "yes_sources": json.dumps(info["yes_sources"]),
            "no_sources": json.dumps(info["no_sources"]),
            "yes_similarities": json.dumps(info["yes_similarities"]),
            "no_similarities": json.dumps(info["no_similarities"]),
        })
    pd.DataFrame(rows).to_csv(output_dir / "similarity_support_map.csv", index=False)
    return SelectionArtifacts(random_demos, representative_demos, similarity_map, metadata)


def binary_metrics(y: np.ndarray, p100: np.ndarray, threshold: float) -> Dict[str, Any]:
    if len(y) == 0:
        return {}
    p01 = np.clip(p100 / 100.0, 1e-6, 1 - 1e-6)
    pred = (p100 >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out: Dict[str, Any] = {
        "threshold": float(threshold), "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p01)),
        "log_loss": float(log_loss(y, p01, labels=[0, 1])),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, p01))
        out["average_precision"] = float(average_precision_score(y, p01))
    else:
        out["roc_auc"] = None; out["average_precision"] = None
    return out


def calibrate_threshold(entries: Sequence[Mapping[str, Any]], metric: str) -> Tuple[float, pd.DataFrame]:
    ok = [e for e in entries if e.get("status") == "ok"]
    if not ok:
        raise RuntimeError("No successful calibration predictions")
    y = np.array([int(e["actual"]) for e in ok], dtype=int)
    p = np.array([float(e["probability_yes"]) for e in ok], dtype=float)
    rows = [binary_metrics(y, p, float(t)) for t in np.arange(5, 96, 1)]
    table = pd.DataFrame(rows)
    best = table.sort_values([metric, "log_loss"], ascending=[False, True]).iloc[0]
    return float(best["threshold"]), table


def phase_diagnostics(entries: Sequence[Mapping[str, Any]], expected_n: int) -> Dict[str, Any]:
    latest_by_idx = {int(e["data_idx"]): e for e in entries if "data_idx" in e}
    values = list(latest_by_idx.values())
    ok = [e for e in values if e.get("status") == "ok"]
    errors = [e for e in values if e.get("status") != "ok"]
    return {
        "expected_n": int(expected_n),
        "logged_unique_n": len(values),
        "ok_n": len(ok),
        "error_n": len(errors),
        "missing_n": max(0, int(expected_n) - len(values)),
        "success_rate": len(ok) / expected_n if expected_n else 1.0,
        "failure_categories": dict(Counter(str(e.get("failure_category", "unknown")) for e in errors)),
        "finish_reasons": dict(Counter(str(e.get("finish_reason", "unknown")) for e in errors)),
        "parse_methods": dict(Counter(str(e.get("parse_method", "unknown")) for e in ok)),
        "attempt_counts": dict(Counter(str(e.get("attempt_count", "unknown")) for e in values)),
    }


def entries_dataframe(entries: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for e in entries:
        if e.get("status") != "ok":
            continue
        rows.append({
            "data_idx": e["data_idx"],
            "source_row_index": e["source_row_index"],
            "phase": e["phase"],
            "variant": e["variant"],
            "model": e["model"],
            "method": e["method"],
            "actual": e["actual"],
            "probability_yes": e["probability_yes"],
            "prediction_at_50": e["prediction_at_50"],
            "raw_prediction": e.get("raw_prediction", ""),
            "demo_source_rows": json.dumps(e.get("demo_source_rows", [])),
            "selection_method": e.get("selection_method", ""),
            "reasoning_trace": json.dumps(e.get("reasoning_trace", {}), ensure_ascii=False),
            "parse_method": e.get("parse_method", ""),
            "finish_reason": e.get("finish_reason", ""),
            "attempt_count": e.get("attempt_count", 0),
            "input_tokens": (e.get("usage") or {}).get("input_tokens", 0),
            "output_tokens": (e.get("usage") or {}).get("output_tokens", 0),
            "total_tokens": (e.get("usage") or {}).get("total_tokens", 0),
        })
    return pd.DataFrame(rows)


def resolve_key(cli: str) -> str:
    key = (cli or "").strip() or os.environ.get("ASU_LLM_API_KEY", "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set ASU_LLM_API_KEY, OPENAI_API_KEY, or pass --api-key")
    return key


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def parse_csv_arg(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def normalize_model_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


async def list_available_models(client: Any) -> List[str]:
    try:
        page = await client.models.list()
        return sorted({str(x.id) for x in page.data})
    except Exception as exc:
        print(f"WARNING: /v1/models could not be listed: {type(exc).__name__}: {exc}")
        return []


def resolve_model_request(requested: str, available: Sequence[str]) -> Tuple[str, str]:
    if not available:
        return requested, "unverified_requested_id"
    lower_map = {m.lower(): m for m in available}
    if requested.lower() in lower_map:
        return lower_map[requested.lower()], "exact"
    norm = normalize_model_id(requested)
    norm_matches = [m for m in available if normalize_model_id(m) == norm]
    if len(norm_matches) == 1:
        return norm_matches[0], "normalized_exact"

    r = requested.lower()
    candidates: List[str] = []
    if "llama4" in normalize_model_id(requested) or ("llama" in r and "17b" in r):
        candidates = [m for m in available if "llama" in m.lower() and ("17b" in m.lower() or "scout" in m.lower()) and "4" in m.lower()]
    elif "70b" in r and "llama" in r:
        candidates = [m for m in available if "llama" in m.lower() and "70b" in m.lower() and "4" not in m.lower()]
        # Prefer an ID explicitly containing llama3, then 3.3, then 3.1.
        candidates = sorted(candidates, key=lambda m: ("llama3" not in normalize_model_id(m), "3.3" not in m, "3.1" not in m, len(m)))
    if len(candidates) == 1:
        return candidates[0], "heuristic_unique"
    if candidates:
        return candidates[0], "heuristic_first_of_multiple:" + "|".join(candidates)
    return requested, "not_found_using_requested_id"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V4/V5 LLM in-context-learning benchmark over ASU Chat Completions")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--v4-reference-split", default="")
    p.add_argument("--v5-reference-split", default="")
    p.add_argument("--allow-fallback-split", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--api-key", default="")
    p.add_argument("--base-url", default="https://openai.rc.asu.edu/v1")
    p.add_argument("--models", default="llama4-scout-17b,llama3-70b")
    p.add_argument("--variants", default="v4,v5")
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--sample-size", type=int, default=0, help="0 uses the full matched split")
    p.add_argument("--memory-ratio", type=float, default=0.40)
    p.add_argument("--calibration-ratio", type=float, default=0.20)
    p.add_argument("--test-ratio", type=float, default=0.40)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--include-sensitive-context", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--threshold-metric", choices=["balanced_accuracy", "f1", "accuracy"], default="balanced_accuracy")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens-direct", type=int, default=320)
    p.add_argument("--max-tokens-cot", type=int, default=700)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--timeout", type=float, default=240.0)
    p.add_argument("--max-concurrent-requests", type=int, default=4)
    p.add_argument("--concurrent-samples", type=int, default=4)
    p.add_argument("--trust-env-proxy", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--json-mode", choices=["auto", "always", "never"], default="auto")
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--similarity-batch-size", type=int, default=1000)
    p.add_argument("--min-success-rate", type=float, default=0.995)
    p.add_argument("--continue-grid-on-error", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    return p.parse_args()


async def run_phase(
    *,
    phase: str,
    phase_df: pd.DataFrame,
    raw: pd.DataFrame,
    variant: str,
    feature_columns: Sequence[str],
    model: str,
    method: str,
    selector: SelectionArtifacts,
    args: argparse.Namespace,
    client: Any,
    semaphore: asyncio.Semaphore,
    log_path: Path,
    latest: Dict[int, Dict[str, Any]],
    run_hash: str,
    use_json_mode: bool,
) -> List[Dict[str, Any]]:
    rows = phase_df.reset_index(drop=True)
    jobs = []
    reused = 0
    for _, r in rows.iterrows():
        data_idx = int(r["data_idx"])
        if latest.get(data_idx, {}).get("status") == "ok":
            reused += 1
            continue
        jobs.append((data_idx, int(r["source_row_index"]), int(r["actual"])))

    print(f"\n[{variant}/{model}/{method}/{phase}] total={len(rows):,} reused_ok={reused:,} calls={len(jobs):,}")
    if not jobs:
        return [latest[int(r["data_idx"])] for _, r in rows.iterrows() if int(r["data_idx"]) in latest]

    start = time.time(); completed = 0; ok_new = 0; err_new = 0
    write_lock = asyncio.Lock()
    worker_sem = asyncio.Semaphore(max(1, args.concurrent_samples))

    async def one(job: Tuple[int, int, int]) -> Dict[str, Any]:
        data_idx, src, actual = job
        async with worker_sem:
            demos, selection_meta = selector.demos_for(method, src, raw, variant, feature_columns)
            target_profile = build_raw_llm_profile(raw.loc[src], variant=variant, feature_columns=feature_columns)
            try:
                pred, raw_text, usage, request_id, call_meta = await call_prediction(
                    client, semaphore,
                    model=model, method=method, demos=demos, target_profile=target_profile,
                    max_tokens_direct=args.max_tokens_direct,
                    max_tokens_cot=args.max_tokens_cot,
                    temperature=args.temperature,
                    retries=args.max_retries,
                    use_json_mode=use_json_mode,
                )
                entry = {
                    "config_hash": run_hash, "created_at": utc_now(), "status": "ok",
                    "phase": phase, "data_idx": data_idx, "source_row_index": src,
                    "actual": actual, "variant": variant, "model": model, "method": method,
                    **pred, **selection_meta, **call_meta,
                    "usage": usage, "request_id": request_id, "raw_response": raw_text,
                }
            except PredictionCallFailure as exc:
                entry = {
                    "config_hash": run_hash, "created_at": utc_now(), "status": "error",
                    "phase": phase, "data_idx": data_idx, "source_row_index": src,
                    "actual": actual, "variant": variant, "model": model, "method": method,
                    **selection_meta,
                    "error_type": type(exc).__name__, "failure_category": exc.category,
                    "error_message": str(exc), "attempt_count": exc.attempt_count,
                    "finish_reason": exc.finish_reason, "usage": exc.usage,
                    "request_id": exc.request_id, "last_raw_response": exc.raw_response,
                }
            except Exception as exc:
                entry = {
                    "config_hash": run_hash, "created_at": utc_now(), "status": "error",
                    "phase": phase, "data_idx": data_idx, "source_row_index": src,
                    "actual": actual, "variant": variant, "model": model, "method": method,
                    "error_type": type(exc).__name__, "failure_category": _classify_api_exception(exc),
                    "error_message": str(exc), "attempt_count": 0,
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                }
            async with write_lock:
                append_jsonl(log_path, entry)
                latest[data_idx] = entry
            return entry

    tasks = [asyncio.create_task(one(j)) for j in jobs]
    for fut in asyncio.as_completed(tasks):
        entry = await fut
        completed += 1
        if entry.get("status") == "ok": ok_new += 1
        else: err_new += 1
        if completed == 1 or completed % max(1, args.progress_every) == 0 or completed == len(jobs):
            elapsed = max(time.time() - start, 1e-9)
            rate = completed / elapsed
            eta_sec = (len(jobs) - completed) / rate if rate else math.inf
            eta = f"{eta_sec/60:.1f} min" if np.isfinite(eta_sec) and eta_sec < 3600 else f"{eta_sec/3600:.2f} h"
            print(
                f"[{variant}/{slugify(model)}/{method}/{phase}] {completed:,}/{len(jobs):,} "
                f"new complete | ok={ok_new:,} err={err_new:,} | {rate:.2f} calls/s | ETA={eta}",
                flush=True,
            )
    return [latest[int(r["data_idx"])] for _, r in rows.iterrows() if int(r["data_idx"]) in latest]


async def run_experiment(
    *,
    variant: str,
    model: str,
    method: str,
    assignments: pd.DataFrame,
    raw: pd.DataFrame,
    feature_columns: Sequence[str],
    selector: SelectionArtifacts,
    split_source: str,
    exact_split: bool,
    json_mode_supported: bool,
    args: argparse.Namespace,
    client: Any,
    semaphore: asyncio.Semaphore,
    output_dir: Path,
) -> Dict[str, Any]:
    exp_dir = output_dir / variant / slugify(model) / method
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)

    fixed_sources = []
    if method in {"random_balanced_8shot_direct", COT_METHOD}:
        fixed_sources = selector.selection_metadata["random_demo_sources"]
    elif method == "representative_8shot_direct":
        fixed_sources = selector.selection_metadata["representative_demo_sources"]

    stable_config = {
        "version": VERSION, "variant": variant, "model": model, "method": method,
        "split_source": split_source, "exact_matched_split": exact_split,
        "selected_source_rows_sha256": hashlib.sha256(json.dumps(assignments["source_row_index"].tolist()).encode()).hexdigest(),
        "feature_columns": list(feature_columns),
        "include_sensitive_context": args.include_sensitive_context,
        "random_seed": args.random_seed, "fixed_demo_sources": fixed_sources,
        "threshold_metric": args.threshold_metric, "temperature": args.temperature,
        "max_tokens_direct": args.max_tokens_direct, "max_tokens_cot": args.max_tokens_cot,
        "json_mode_supported": bool(json_mode_supported),
        "prompt_contract": "strict_json_v3_method_specific_format_retry",
    }
    run_hash = config_hash(stable_config)
    config_path = exp_dir / "run_config.json"
    if config_path.exists() and not args.overwrite:
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old.get("config_hash") != run_hash:
            raise RuntimeError(f"Configuration mismatch in existing output directory: {exp_dir}")
        created_at = old.get("created_at", utc_now())
    else:
        created_at = utc_now()
    run_config = {**stable_config, "config_hash": run_hash, "created_at": created_at}
    config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    cal_log = exp_dir / "logs" / "calibration.jsonl"
    test_log = exp_dir / "logs" / "test.jsonl"
    cal_latest = load_latest_jsonl(cal_log, run_hash)
    test_latest = load_latest_jsonl(test_log, run_hash)
    cal_frame = assignments[assignments["split"] == "calibration"]
    test_frame = assignments[assignments["split"] == "test"]

    cal_entries = await run_phase(
        phase="calibration", phase_df=cal_frame, raw=raw, variant=variant,
        feature_columns=feature_columns, model=model, method=method, selector=selector,
        args=args, client=client, semaphore=semaphore, log_path=cal_log,
        latest=cal_latest, run_hash=run_hash, use_json_mode=json_mode_supported,
    )
    cal_diag = phase_diagnostics(cal_entries, len(cal_frame))
    (exp_dir / "calibration_diagnostics.json").write_text(json.dumps(cal_diag, indent=2), encoding="utf-8")
    if cal_diag["success_rate"] < args.min_success_rate:
        raise RuntimeError(
            f"Calibration coverage {cal_diag['success_rate']:.2%} is below required {args.min_success_rate:.2%}."
        )
    selected_threshold, threshold_table = calibrate_threshold(cal_entries, args.threshold_metric)
    threshold_table.to_csv(exp_dir / "threshold_search.csv", index=False)

    test_entries = await run_phase(
        phase="test", phase_df=test_frame, raw=raw, variant=variant,
        feature_columns=feature_columns, model=model, method=method, selector=selector,
        args=args, client=client, semaphore=semaphore, log_path=test_log,
        latest=test_latest, run_hash=run_hash, use_json_mode=json_mode_supported,
    )
    test_diag = phase_diagnostics(test_entries, len(test_frame))
    (exp_dir / "test_diagnostics.json").write_text(json.dumps(test_diag, indent=2), encoding="utf-8")
    if test_diag["success_rate"] < args.min_success_rate:
        raise RuntimeError(
            f"Test coverage {test_diag['success_rate']:.2%} is below required {args.min_success_rate:.2%}."
        )

    cal_ok = [e for e in cal_entries if e.get("status") == "ok"]
    test_ok = [e for e in test_entries if e.get("status") == "ok"]
    entries_dataframe(cal_ok).to_csv(exp_dir / "calibration_predictions.csv", index=False)
    entries_dataframe(test_ok).to_csv(exp_dir / "test_predictions.csv", index=False)
    y_cal = np.array([int(e["actual"]) for e in cal_ok])
    p_cal = np.array([float(e["probability_yes"]) for e in cal_ok])
    y_test = np.array([int(e["actual"]) for e in test_ok])
    p_test = np.array([float(e["probability_yes"]) for e in test_ok])
    metrics = {
        "calibration_selected": binary_metrics(y_cal, p_cal, selected_threshold),
        "test_at_50": binary_metrics(y_test, p_test, 50.0),
        "test_selected": binary_metrics(y_test, p_test, selected_threshold),
    }
    usage = sum_usage(list(cal_latest.values()) + list(test_latest.values()))
    summary = {
        "experiment": "flare_vax_llm_icl_benchmark", "version": VERSION,
        "created_at": utc_now(), "variant": variant, "model": model, "method": method,
        "n_selected": len(assignments),
        "split_sizes": assignments["split"].value_counts().to_dict(),
        "selected_threshold": selected_threshold,
        "metrics": metrics, "usage_total": usage,
        "calibration_diagnostics": cal_diag, "test_diagnostics": test_diag,
        "run_config": run_config,
    }
    (exp_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=json_safe), encoding="utf-8")
    return summary


def flatten_result(summary: Mapping[str, Any]) -> Dict[str, Any]:
    m = summary["metrics"]["test_selected"]
    return {
        "variant": summary["variant"], "model": summary["model"], "method": summary["method"],
        "n_selected": summary["n_selected"], "test_n": m.get("n"),
        "selected_threshold": summary["selected_threshold"],
        "accuracy": m.get("accuracy"), "balanced_accuracy": m.get("balanced_accuracy"),
        "precision": m.get("precision"), "recall": m.get("recall"),
        "specificity": m.get("specificity"), "f1": m.get("f1"),
        "roc_auc": m.get("roc_auc"), "average_precision": m.get("average_precision"),
        "brier": m.get("brier"), "log_loss": m.get("log_loss"),
        "input_tokens": summary["usage_total"].get("input_tokens", 0),
        "output_tokens": summary["usage_total"].get("output_tokens", 0),
        "total_tokens": summary["usage_total"].get("total_tokens", 0),
        "calibration_success_rate": summary["calibration_diagnostics"].get("success_rate"),
        "test_success_rate": summary["test_diagnostics"].get("success_rate"),
    }


async def async_main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = parse_csv_arg(args.variants)
    methods = parse_csv_arg(args.methods)
    requested_models = parse_csv_arg(args.models)
    unknown_variants = set(variants) - {"v4", "v5"}
    unknown_methods = set(methods) - set(METHODS)
    if unknown_variants: raise ValueError(f"Unknown variants: {sorted(unknown_variants)}")
    if unknown_methods: raise ValueError(f"Unknown methods: {sorted(unknown_methods)}")

    header = pd.read_csv(input_path, nrows=0).columns
    missing = [c for c in ALL_REQUIRED_COLUMNS if c not in header]
    if missing:
        raise KeyError(f"adult24.csv is missing required V4/V5 columns: {missing}")
    raw = pd.read_csv(input_path, usecols=ALL_REQUIRED_COLUMNS, low_memory=False)

    ref_paths = {
        "v4": Path(args.v4_reference_split) if args.v4_reference_split else None,
        "v5": Path(args.v5_reference_split) if args.v5_reference_split else None,
    }
    variant_data: Dict[str, Dict[str, Any]] = {}
    planned_calls = 0
    for variant in variants:
        feature_columns = feature_columns_for_variant(variant, args.include_sensitive_context)
        assignments, split_source, exact = prepare_assignments(variant, ref_paths[variant], raw, args)
        variant_root = output_dir / variant
        variant_root.mkdir(parents=True, exist_ok=True)
        assignments.to_csv(variant_root / "split_assignments_used.csv", index=False)
        selector = build_selection_artifacts(
            variant, assignments, raw, feature_columns, args.random_seed,
            variant_root / "selection_artifacts", args.similarity_batch_size,
        )
        variant_data[variant] = {
            "feature_columns": feature_columns, "assignments": assignments,
            "split_source": split_source, "exact": exact, "selector": selector,
        }
        n_calls_per_method = int(assignments["split"].isin(["calibration", "test"]).sum())
        planned_calls += n_calls_per_method * len(methods) * len(requested_models)
        print(f"[{variant}] selected={len(assignments):,} split={assignments['split'].value_counts().to_dict()} features={len(feature_columns)}")

    plan = {
        "variants": variants, "requested_models": requested_models, "methods": methods,
        "planned_prediction_calls": int(planned_calls),
        "note": "JSON-mode probes add approximately one call per resolved model.",
    }
    (output_dir / "benchmark_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print("\nBENCHMARK PLAN")
    print(json.dumps(plan, indent=2))
    if args.plan_only:
        return

    if args.dry_run:
        for variant, data in variant_data.items():
            assignments = data["assignments"]
            target = assignments[assignments["split"] == "calibration"].iloc[0]
            src = int(target["source_row_index"])
            target_profile = build_raw_llm_profile(raw.loc[src], variant=variant, feature_columns=data["feature_columns"])
            for method in methods:
                demos, selection = data["selector"].demos_for(method, src, raw, variant, data["feature_columns"])
                prompt = build_prompt(method, demos, target_profile)
                prompt_file = output_dir / variant / "selection_artifacts" / f"dry_prompt_{method}.txt"
                prompt_file.write_text(prompt, encoding="utf-8")
                print(f"[{variant}/{method}] demos={selection.get('demo_source_rows', [])} prompt_chars={len(prompt):,} -> {prompt_file}")
        print("Dry run complete. No API calls were made.")
        return

    if AsyncOpenAI is None:
        raise RuntimeError("Install openai and httpx: pip install -U openai httpx")
    http_client = httpx.AsyncClient(
        trust_env=args.trust_env_proxy,
        timeout=httpx.Timeout(args.timeout, connect=min(30.0, args.timeout)),
    )
    client = AsyncOpenAI(
        api_key=resolve_key(args.api_key), base_url=args.base_url,
        timeout=args.timeout, max_retries=0, http_client=http_client,
    )
    semaphore = asyncio.Semaphore(max(1, args.max_concurrent_requests))

    available = await list_available_models(client)
    resolved_models = []
    resolution_rows = []
    for req in requested_models:
        resolved, how = resolve_model_request(req, available)
        resolved_models.append(resolved)
        resolution_rows.append({"requested": req, "resolved": resolved, "resolution": how})
    (output_dir / "model_resolution.json").write_text(
        json.dumps({"available_models": available, "resolution": resolution_rows}, indent=2), encoding="utf-8"
    )
    print("MODEL RESOLUTION")
    print(json.dumps(resolution_rows, indent=2))

    json_mode_by_model: Dict[str, bool] = {}
    probe_rows = []
    for model in resolved_models:
        if args.json_mode == "never":
            supported, detail = False, "disabled"
        elif args.json_mode == "always":
            supported, detail = True, "forced"
        else:
            supported, detail = await probe_json_mode(client, model)
        json_mode_by_model[model] = supported
        probe_rows.append({"model": model, "json_mode": supported, "detail": detail})
    (output_dir / "json_mode_probe.json").write_text(json.dumps(probe_rows, indent=2), encoding="utf-8")

    results = []
    failures = []
    for variant in variants:
        data = variant_data[variant]
        for model in resolved_models:
            for method in methods:
                try:
                    summary = await run_experiment(
                        variant=variant, model=model, method=method,
                        assignments=data["assignments"], raw=raw,
                        feature_columns=data["feature_columns"], selector=data["selector"],
                        split_source=data["split_source"], exact_split=data["exact"],
                        json_mode_supported=json_mode_by_model[model],
                        args=args, client=client, semaphore=semaphore, output_dir=output_dir,
                    )
                    results.append(flatten_result(summary))
                    pd.DataFrame(results).to_csv(output_dir / "benchmark_results.csv", index=False)
                except Exception as exc:
                    failure = {
                        "created_at": utc_now(), "variant": variant, "model": model,
                        "method": method, "error_type": type(exc).__name__, "error_message": str(exc),
                    }
                    failures.append(failure)
                    pd.DataFrame(failures).to_csv(output_dir / "benchmark_failures.csv", index=False)
                    print(f"EXPERIMENT FAILED: {failure}", flush=True)
                    if not args.continue_grid_on_error:
                        raise

    if results:
        result_df = pd.DataFrame(results).sort_values(["variant", "model", "method"])
        result_df.to_csv(output_dir / "benchmark_results.csv", index=False)
        metric_cols = ["accuracy", "balanced_accuracy", "f1", "roc_auc", "brier", "log_loss"]
        wide = result_df.pivot_table(index=["variant", "method"], columns="model", values=metric_cols)
        wide.to_csv(output_dir / "benchmark_results_wide.csv")
        print("\nFINAL BENCHMARK RESULTS")
        print(result_df.to_string(index=False))

    await client.close()
    await http_client.aclose()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
