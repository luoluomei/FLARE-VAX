#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLARE-VAX HBM2 — OpenAI API / Google Colab validation runner
=============================================================

Purpose
-------
Validate the revised three-call FLARE-VAX pipeline on the cleaned NHIS 2024
HBM2 dataset before moving the experiment to ASU Sol.

Pipeline
--------
CALL 1: HBM-aligned perceived-threat proxy assessment
        Inputs: age, self-rated health, chronic/risk conditions, functional
        limitations, BMI, and smoking status.
        Output: strict JSON with score 1-5, level, and one-sentence reason.

CALL 2: HBM-aligned perceived-barrier proxy assessment
        Inputs: insurance, cost, usual-care, transportation, language, and
        digital-access indicators.
        Output: strict JSON with score 1-5, level, and one-sentence reason.

CALL 3: Final flu-vaccination decision
        Inputs: CALL 1 + CALL 2 outputs, the PRECOMPUTED rule-based HBM2
        pattern, selected context variables, and similar TRAINING errors.
        Output: strict JSON containing only {"decision": "YES"|"NO"}.

Important evaluation design
---------------------------
* The HBM2 pattern is defined by the cleaned-data construction, not by the
  LLM's 1-5 scores:
    P0 = High threat / Low barrier
    P1 = High threat / High barrier
    P2 = Low threat / Low barrier
    P3 = Low threat / High barrier
* Similarity memory stores errors from the TRAIN phase only. It is frozen
  during TEST, preventing test-label leakage.
* The script does NOT globally replace 7/8/9 after loading the cleaned file,
  because some cleaned NHIS variables have valid category codes such as
  education=7. Missing-code handling belongs in clean_hbm2_nhis2024.py.
* Every completed sample is immediately appended to JSONL. Re-running the
  same command with the same output directory resumes from the first
  unfinished sample.

Colab setup
-----------
1) Install packages:
   !pip -q install -U openai pandas numpy scikit-learn tqdm

2) Mount Drive and expose the API key (recommended: Colab Secrets):
   from google.colab import drive, userdata
   import os
   drive.mount('/content/drive')
   os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')

3) Run a small validation:
   !python /content/drive/MyDrive/FLARE-VAX/run_flare_vax_hbm2_openai_colab.py \
       --data_path /content/drive/MyDrive/FLARE-VAX/Data/Processed/nhis2024_hbm2_clean.csv \
       --output_dir /content/drive/MyDrive/FLARE-VAX/results/gpt4o_mini_hbm2_val \
       --model gpt-4o-mini \
       --sample_size 20 \
       --checkpoint_every 1 \
       --pattern_prior_mode theory

4) Inspect one sample:
   import json
   p = '/content/drive/MyDrive/FLARE-VAX/results/gpt4o_mini_hbm2_val/logs/sample_log.jsonl'
   print(json.loads(open(p).readline()))

Security
--------
Prefer the OPENAI_API_KEY environment variable or Colab Secrets. Passing an
API key through --api_key is supported for convenience but may expose it in
notebook history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

try:
    from openai import OpenAI
except ImportError:  # --dry_run remains available without the SDK
    OpenAI = None  # type: ignore[assignment,misc]


VERSION = "hbm2_openai_colab_v1"
EXPERIMENT = "flare_vax_hbm2_three_call"

PATTERN_LABELS: Dict[int, str] = {
    0: "High threat / Low barrier",
    1: "High threat / High barrier",
    2: "Low threat / Low barrier",
    3: "Low threat / High barrier",
}

PATTERN_TENDENCIES: Dict[int, str] = {
    0: (
        "Most favorable HBM2 profile: stronger motivation and relatively little "
        "friction, so vaccination is more likely."
    ),
    1: (
        "Mixed HBM2 profile: stronger motivation is present, but barriers can block "
        "or delay vaccination."
    ),
    2: (
        "Mixed HBM2 profile: access is relatively favorable, but low perceived threat "
        "provides less motivation to vaccinate."
    ),
    3: (
        "Least favorable HBM2 profile: weaker motivation and stronger barriers, so "
        "vaccination is less likely."
    ),
}

THEORETICAL_PATTERN_ORDER = (
    "Directional HBM2 prior (not a deterministic rule): "
    "P0 High threat/Low barrier > P1 High threat/High barrier > "
    "P2 Low threat/Low barrier > P3 Low threat/High barrier."
)

# Selected from the prior all-clean variable-importance analysis, while keeping
# CALL 1 and CALL 2 theory-defined rather than importance-weighted.
CONTEXT_FEATURES: List[str] = [
    "retail_clinic_visits_12m",
    "last_doctor_visit",
    "used_internet_test_results",
    "used_internet_communicate_doctor",
    "education",
    "income_poverty_ratio",
    "hispanic",
    "time_since_wellness_visit",
    "last_visit_wellness_yes",
]

MEMORY_FEATURES: List[str] = [
    "age",
    "health_status",
    "chronic_or_risk_count",
    "hbm_threat_score",
    "hbm_barrier_count",
    "hbm_barrier_score",
    "hbm2_pattern",
    "uninsured_yes",
    "medicare_yes",
    "medicaid_yes",
    "private_insurance_yes",
    "usual_care_place",
    "retail_clinic_visits_12m",
    "last_doctor_visit",
    "used_internet_test_results_yes",
    "used_internet_communicate_doctor_yes",
    "education",
    "income_poverty_ratio",
    "time_since_wellness_visit",
]

THREAT_CONDITIONS: List[Tuple[str, str]] = [
    ("diabetes_yes", "diabetes"),
    ("copd_yes", "COPD"),
    ("cancer_ever_yes", "cancer history"),
    ("heart_disease_yes", "coronary heart disease"),
    ("angina_yes", "angina"),
    ("heart_attack_yes", "heart-attack history"),
    ("stroke_yes", "stroke history"),
    ("hypertension_yes", "hypertension"),
    ("asthma_ever_yes", "asthma history"),
    ("asthma_current_yes", "current asthma"),
    ("asthma_episode_12m_yes", "asthma episode in the past 12 months"),
    ("weak_kidneys_yes", "weak kidneys"),
    ("liver_condition_yes", "liver condition"),
    ("hepatitis_ever_yes", "hepatitis history"),
    ("disability_yes", "disability"),
    ("any_functional_difficulty_yes", "functional difficulty"),
]

BARRIER_INDICATORS: List[Tuple[str, str]] = [
    ("uninsured_yes", "currently uninsured"),
    ("uninsured_past_year_yes", "uninsured at some point in the past year"),
    ("no_insurance_cost_yes", "lacked insurance because of cost"),
    ("lost_coverage_cost_increase_yes", "lost coverage because cost increased"),
    ("delayed_care_cost_12m_yes", "delayed care because of cost"),
    (
        "needed_care_not_get_cost_12m_yes",
        "did not obtain needed care because of cost",
    ),
    (
        "problems_paying_medical_bills_12m_yes",
        "had problems paying medical bills",
    ),
    ("unable_pay_medical_bills_now_yes", "currently unable to pay medical bills"),
    ("has_deductible_plan1_yes", "has a deductible on insurance plan 1"),
    ("has_deductible_plan2_yes", "has a deductible on insurance plan 2"),
    ("no_usual_care", "has no usual place of care"),
    ("transportation_barrier_yes", "experienced a transportation barrier"),
    ("worry_medical_bills_any", "is worried about medical bills"),
    ("limited_language_at_doctor", "has limited language access at the doctor"),
    ("no_internet_access", "has no internet access"),
]


# ---------------------------------------------------------------------------
# JSON Schemas for strict OpenAI Structured Outputs
# ---------------------------------------------------------------------------
ASSESSMENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "level": {
            "type": "string",
            "enum": ["very_low", "low", "moderate", "high", "very_high"],
        },
        "reason": {"type": "string"},
    },
    "required": ["score", "level", "reason"],
    "additionalProperties": False,
}

DECISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["YES", "NO"]},
    },
    "required": ["decision"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
THREAT_SYSTEM = """You are a public-health behavioral analyst using the Health Belief Model (HBM).
This call assesses only an HBM-aligned proxy for perceived threat, combining perceived susceptibility and perceived severity.
Use only the supplied observable variables. Do not infer unobserved attitudes, do not predict vaccination, and do not discuss benefits, cues, or barriers.
Follow the requested score rubric. Return only the required structured output. The reason must be exactly one concise sentence of at most 35 words."""

THREAT_USER_TEMPLATE = """TASK
Assess this individual's HBM-aligned perceived-threat proxy for influenza.

DEFINITION
Perceived threat combines:
1. Susceptibility: how vulnerable the person appears to influenza.
2. Severity: how serious influenza consequences could be for this person.
The variables are observable risk proxies, not direct survey measures of private beliefs.

SCORING RUBRIC
1 = Very low: under 50, excellent/very good health, and no meaningful chronic, respiratory, cardiovascular, or functional risk indicators.
2 = Low: one limited or mild risk indicator, with otherwise favorable age and health.
3 = Moderate: age 50-64, fair/poor health, or several risk indicators, but not a concentrated high-risk profile.
4 = High: age 65+, or fair/poor health combined with a major chronic, cardiopulmonary, or functional risk.
5 = Very high: age 65+ together with poor health, multiple serious conditions, or substantial functional vulnerability.
Use the rubric as an anchor and evaluate the full profile holistically.

INPUT FORMAT
{profile}

OUTPUT FORMAT
Return only an object with score (1-5), level, and one concise sentence in reason."""

BARRIER_SYSTEM = """You are a public-health behavioral analyst using the Health Belief Model (HBM).
This call assesses only an HBM-aligned proxy for perceived barriers to receiving a flu vaccine.
Use only the supplied observable variables. Do not infer unobserved vaccine attitudes, do not predict vaccination, and do not use threat information.
Follow the requested score rubric. Return only the required structured output. The reason must be exactly one concise sentence of at most 35 words."""

BARRIER_USER_TEMPLATE = """TASK
Assess this individual's HBM-aligned perceived-barrier proxy for obtaining a flu vaccination.

DEFINITION
Perceived barriers are observable financial, insurance, healthcare-access, transportation, language, and digital-access obstacles that can make vaccination harder.

SCORING RUBRIC
1 = Very low: insured, stable access to usual care, and no observed cost, transportation, language, or digital-access barrier.
2 = Low: one mild concern, while healthcare access remains generally stable.
3 = Moderate: two or three barriers, or one clear cost/access obstacle that could interfere with vaccination.
4 = High: multiple clear financial, insurance, or structural barriers that materially complicate vaccination.
5 = Very high: severe combined barriers, such as being uninsured or lacking usual care together with major cost, transportation, language, or access problems.
Use the rubric as an anchor and evaluate the full barrier profile holistically.

INPUT FORMAT
{profile}

OUTPUT FORMAT
Return only an object with score (1-5), level, and one concise sentence in reason."""

DECISION_SYSTEM = """You predict whether an NHIS respondent received a flu vaccine in the past 12 months.
Use the supplied HBM assessments, rule-based HBM2 pattern, context variables, and any similar training errors.
Treat the pattern as a directional prior, not a deterministic rule. Do not invent facts.
Return only the required structured output with a single YES or NO decision and no explanation."""

DECISION_USER_TEMPLATE = """TASK
Predict whether this person received a flu vaccine in the past 12 months.

CALL 1 — THREAT ASSESSMENT
{threat_result}

CALL 2 — BARRIER ASSESSMENT
{barrier_result}

RULE-BASED HBM2 PATTERN
Pattern ID: P{pattern_id}
Pattern: {pattern_label}
Rule-based threat score: {rule_threat_score}/4
Rule-based barrier score: {rule_barrier_score}/3
Current-pattern tendency: {pattern_tendency}
{pattern_prior}

ADDITIONAL CONTEXT VARIABLES
These variables were retained because they had strong predictive signal outside the two HBM2 construct scores. Use them only to refine the decision.
{context_profile}

SIMILAR TRAINING ERRORS
{memory_text}

DECISION RULES
1. Integrate the two CALL assessments first.
2. Use the HBM2 pattern as a directional behavioral prior, not as an automatic label.
3. Use additional context to adjust the prediction when it provides clear evidence.
4. Use memory only when the prior cases are genuinely similar.
5. Output only the required YES/NO structured decision; provide no reasoning or extra fields."""


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return value is None


def safe_value(row: pd.Series, name: str, default: Any = np.nan) -> Any:
    return row[name] if name in row.index else default


def int_or_none(value: Any) -> Optional[int]:
    if is_missing(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def float_or_none(value: Any) -> Optional[float]:
    if is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_num(value: Any, digits: int = 2) -> str:
    number = float_or_none(value)
    if number is None:
        return "unknown"
    if float(number).is_integer():
        return str(int(number))
    return f"{number:.{digits}f}"


def yes_no_from_binary(value: Any) -> str:
    number = int_or_none(value)
    if number == 1:
        return "yes"
    if number == 0:
        return "no"
    return "unknown"


def raw_yes_no(row: pd.Series, base: str) -> str:
    binary_name = f"{base}_yes"
    if binary_name in row.index:
        return yes_no_from_binary(row[binary_name])
    value = int_or_none(safe_value(row, base))
    if value == 1:
        return "yes"
    if value == 2:
        return "no"
    return "unknown"


def label_code(value: Any, mapping: Dict[int, str], unknown: str = "unknown") -> str:
    code = int_or_none(value)
    if code is None:
        return unknown
    return mapping.get(code, f"code {code}")


HEALTH_STATUS = {
    1: "excellent",
    2: "very good",
    3: "good",
    4: "fair",
    5: "poor",
}
BMI_CATEGORY = {1: "underweight", 2: "healthy weight", 3: "overweight", 4: "obese"}
SMOKING_STATUS = {1: "uses smokeless tobacco every day", 2: "uses it some days", 3: "does not currently use it"}
INSURANCE_TYPE = {
    1: "private coverage",
    2: "Medicaid or other public coverage",
    3: "other coverage",
    4: "uninsured",
    5: "coverage type unknown",
}
USUAL_CARE_PLACE = {1: "one usual place", 2: "no usual place", 3: "more than one usual place"}
USUAL_CARE_TYPE = {
    1: "doctor's office or health center",
    2: "urgent care center or clinic in a drug/grocery store",
    3: "hospital emergency room",
    4: "VA Medical Center or VA outpatient clinic",
    5: "some other place",
    6: "does not go to one place most often",
}
LAST_VISIT = {
    0: "never",
    1: "within the past year",
    2: "1 to under 2 years ago",
    3: "2 to under 3 years ago",
    4: "3 to under 5 years ago",
    5: "5 to under 10 years ago",
    6: "10 or more years ago",
}
EDUCATION = {
    0: "never attended or kindergarten only",
    1: "grades 1-11",
    2: "12th grade without diploma",
    3: "GED or equivalent",
    4: "high school graduate",
    5: "some college without degree",
    6: "occupational/technical/vocational associate degree",
    7: "academic associate degree",
    8: "bachelor's degree",
    9: "master's degree",
    10: "professional-school or doctoral degree",
}
INCOME_POVERTY_RATIO = {
    1: "0.00-0.49",
    2: "0.50-0.74",
    3: "0.75-0.99",
    4: "1.00-1.24",
    5: "1.25-1.49",
    6: "1.50-1.74",
    7: "1.75-1.99",
    8: "2.00-2.49",
    9: "2.50-2.99",
    10: "3.00-3.49",
    11: "3.50-3.99",
    12: "4.00-4.49",
    13: "4.50-4.99",
    14: "5.00 or greater",
}
HISPANIC_GROUP = {
    1: "Hispanic",
    2: "non-Hispanic White only",
    3: "non-Hispanic Black/African American only",
    4: "non-Hispanic Asian only",
    5: "non-Hispanic American Indian/Alaska Native only",
    6: "non-Hispanic American Indian/Alaska Native and another group",
    7: "other single or multiple races",
}


def active_labels(row: pd.Series, variables: Sequence[Tuple[str, str]]) -> List[str]:
    labels: List[str] = []
    for column, label in variables:
        value = int_or_none(safe_value(row, column))
        if value == 1:
            labels.append(label)
    return labels


def build_threat_profile(row: pd.Series) -> str:
    age = int_or_none(safe_value(row, "age"))
    conditions = active_labels(row, THREAT_CONDITIONS)
    condition_text = ", ".join(conditions) if conditions else "none observed"
    return "\n".join(
        [
            f"Age: {age if age is not None else 'unknown'}",
            f"Age 50 or older: {yes_no_from_binary(safe_value(row, 'age_50plus'))}",
            f"Age 65 or older: {yes_no_from_binary(safe_value(row, 'age_65plus'))}",
            f"Self-rated health: {label_code(safe_value(row, 'health_status'), HEALTH_STATUS)}",
            f"Fair or poor health indicator: {yes_no_from_binary(safe_value(row, 'poor_fair_health'))}",
            f"Observed chronic/risk conditions: {condition_text}",
            f"Number of selected chronic/risk conditions: {fmt_num(safe_value(row, 'chronic_or_risk_count'))}",
            f"Any functional difficulty: {raw_yes_no(row, 'any_functional_difficulty')}",
            f"Disability: {raw_yes_no(row, 'disability')}",
            f"BMI category: {label_code(safe_value(row, 'bmi_category'), BMI_CATEGORY)}",
        ]
    )


def build_barrier_profile(row: pd.Series) -> str:
    observed = active_labels(row, BARRIER_INDICATORS)
    observed_text = "; ".join(observed) if observed else "none observed"
    months_uninsured = fmt_num(safe_value(row, "months_uninsured"))
    return "\n".join(
        [
            f"Has health insurance: {raw_yes_no(row, 'has_insurance')}",
            f"Currently uninsured: {raw_yes_no(row, 'uninsured')}",
            f"Insurance type: {label_code(safe_value(row, 'insurance_type'), INSURANCE_TYPE)}",
            f"Medicare: {raw_yes_no(row, 'medicare')}",
            f"Medicaid: {raw_yes_no(row, 'medicaid')}",
            f"Private insurance: {raw_yes_no(row, 'private_insurance')}",
            f"Uninsured during the past year: {raw_yes_no(row, 'uninsured_past_year')}",
            f"Months uninsured: {months_uninsured}",
            f"Usual place of care: {label_code(safe_value(row, 'usual_care_place'), USUAL_CARE_PLACE)}",
            f"Usual-care setting: {label_code(safe_value(row, 'usual_care_type'), USUAL_CARE_TYPE)}",
            f"Delayed care because of cost: {raw_yes_no(row, 'delayed_care_cost_12m')}",
            f"Did not obtain needed care because of cost: {raw_yes_no(row, 'needed_care_not_get_cost_12m')}",
            f"Problems paying medical bills: {raw_yes_no(row, 'problems_paying_medical_bills_12m')}",
            f"Unable to pay medical bills now: {raw_yes_no(row, 'unable_pay_medical_bills_now')}",
            f"Transportation barrier: {raw_yes_no(row, 'transportation_barrier')}",
            f"Limited language access at doctor: {yes_no_from_binary(safe_value(row, 'limited_language_at_doctor'))}",
            f"No internet access: {yes_no_from_binary(safe_value(row, 'no_internet_access'))}",
            f"Observed barrier indicators: {observed_text}",
            f"Number of observed barrier indicators: {fmt_num(safe_value(row, 'hbm_barrier_count'))}",
        ]
    )


def build_context_profile(row: pd.Series, include_sensitive_context: bool) -> str:
    lines = [
        f"Retail-clinic visits in past 12 months: {fmt_num(safe_value(row, 'retail_clinic_visits_12m'))} (5 means 5+)",
        f"Time since last doctor visit: {label_code(safe_value(row, 'last_doctor_visit'), LAST_VISIT)}",
        f"Used internet to view test results: {raw_yes_no(row, 'used_internet_test_results')}",
        f"Used internet to communicate with a doctor's office: {raw_yes_no(row, 'used_internet_communicate_doctor')}",
        f"Education: {label_code(safe_value(row, 'education'), EDUCATION)}",
        (
            "Family-income-to-poverty-ratio category: "
            f"{label_code(safe_value(row, 'income_poverty_ratio'), INCOME_POVERTY_RATIO)}"
        ),
        f"Was the last doctor visit a wellness visit: {raw_yes_no(row, 'last_visit_wellness')}",
        (
            "Time since an earlier wellness visit when applicable: "
            f"{label_code(safe_value(row, 'time_since_wellness_visit'), LAST_VISIT)}"
        ),
    ]
    if include_sensitive_context:
        lines.append(
            f"Race/Hispanic public-use group: {label_code(safe_value(row, 'hispanic'), HISPANIC_GROUP)}"
        )
    return "\n".join(lines)


def compact_context_summary(row: pd.Series) -> str:
    return (
        f"retail visits={fmt_num(safe_value(row, 'retail_clinic_visits_12m'))}; "
        f"last doctor={label_code(safe_value(row, 'last_doctor_visit'), LAST_VISIT)}; "
        f"online test results={raw_yes_no(row, 'used_internet_test_results')}; "
        f"online doctor communication={raw_yes_no(row, 'used_internet_communicate_doctor')}"
    )


def normalize_reason(reason: Any, max_chars: int = 260) -> str:
    text = " ".join(str(reason).strip().split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def validate_assessment(obj: Dict[str, Any], name: str) -> Dict[str, Any]:
    score = int(obj["score"])
    if score < 1 or score > 5:
        raise ValueError(f"{name} score outside 1-5: {score}")
    allowed = {"very_low", "low", "moderate", "high", "very_high"}
    level = str(obj["level"])
    if level not in allowed:
        raise ValueError(f"Invalid {name} level: {level}")
    return {"score": score, "level": level, "reason": normalize_reason(obj["reason"])}


def response_usage(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def add_usage(total: Dict[str, int], usage: Dict[str, int]) -> None:
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        total[key] = int(total.get(key, 0)) + int(usage.get(key, 0))


def structured_response_call(
    client: OpenAI,
    *,
    model: str,
    schema_name: str,
    schema: Dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    temperature: float,
) -> Tuple[Dict[str, Any], str, Dict[str, int], str]:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        store=False,
    )
    raw_text = str(response.output_text or "").strip()
    if not raw_text:
        raise RuntimeError(f"OpenAI returned empty output for {schema_name}")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Structured output was not parseable JSON for {schema_name}: {raw_text[:300]}"
        ) from exc
    request_id = str(getattr(response, "_request_id", "") or "")
    return parsed, raw_text, response_usage(response), request_id


# ---------------------------------------------------------------------------
# Similarity memory
# ---------------------------------------------------------------------------
@dataclass
class MemoryItem:
    data_idx: int
    text: str
    vector: np.ndarray


class SimilarityErrorMemory:
    """Cosine-similarity memory using normalized feature vectors."""

    def __init__(self, k: int = 3):
        self.k = max(0, int(k))
        self.items: List[MemoryItem] = []

    def store(self, data_idx: int, vector: np.ndarray, text: str) -> None:
        if self.k <= 0:
            return
        self.items.append(
            MemoryItem(data_idx=int(data_idx), text=str(text), vector=vector.astype(np.float32).copy())
        )

    def get(self, query_vector: np.ndarray) -> Tuple[str, List[int]]:
        if self.k <= 0 or not self.items:
            return "No similar training errors are available.", []
        matrix = np.vstack([item.vector for item in self.items])
        similarities = matrix @ query_vector.astype(np.float32)
        order = np.argsort(-similarities)[: min(self.k, len(self.items))]
        texts: List[str] = []
        ids: List[int] = []
        for rank, position in enumerate(order, start=1):
            item = self.items[int(position)]
            ids.append(item.data_idx)
            texts.append(
                f"Similar training error {rank} (cosine similarity={similarities[position]:.3f}):\n"
                f"{item.text}"
            )
        return "\n\n".join(texts), ids

    def size(self) -> int:
        return len(self.items)


def build_memory_record(entry: Dict[str, Any]) -> str:
    actual = "YES" if int(entry["actual"]) == 1 else "NO"
    predicted = str(entry["decision"])
    return (
        f"Pattern: P{entry['hbm2_pattern']} {entry['pattern_label']}. "
        f"LLM threat={entry['threat_result']['score']}/5; "
        f"LLM barriers={entry['barrier_result']['score']}/5. "
        f"Context: {entry['context_summary']}. "
        f"Previous prediction={predicted}; actual outcome={actual}. "
        "Use this only as evidence that a similar profile was previously misclassified."
    )


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def warn_about_possible_global_missing_code_damage(df: pd.DataFrame) -> None:
    """Warn when the cleaned file appears to have lost valid NHIS 7/8/9 categories.

    The supplied cleaner used a generic global replacement for 7/8/9. Those codes
    are missing for many yes/no variables, but they are valid categories for some
    recodes such as education and income-to-poverty ratio. This warning does not
    alter the data; regenerate the clean CSV with variable-specific missing rules
    before a final experiment.
    """
    warnings_found: List[str] = []
    if "education" in df.columns:
        values = set(pd.to_numeric(df["education"], errors="coerce").dropna().astype(int).unique())
        if 10 in values and not ({7, 8, 9} & values):
            warnings_found.append("education has code 10 but no valid codes 7/8/9")
    if "income_poverty_ratio" in df.columns:
        values = set(pd.to_numeric(df["income_poverty_ratio"], errors="coerce").dropna().astype(int).unique())
        if any(code >= 10 for code in values) and not ({7, 8, 9} & values):
            warnings_found.append("income_poverty_ratio has codes >=10 but no valid codes 7/8/9")
    if "hispanic" in df.columns:
        values = set(pd.to_numeric(df["hispanic"], errors="coerce").dropna().astype(int).unique())
        if values and 7 not in values:
            warnings_found.append("hispanic public-use group code 7 is absent")
    if warnings_found:
        print("WARNING: The cleaned CSV may reflect overly broad 7/8/9 missing-code replacement:")
        for warning in warnings_found:
            print(f"  - {warning}")
        print("  Quick API validation can proceed, but regenerate the clean data before final results.")


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            "The cleaned dataset is missing required columns: " + ", ".join(missing)
        )


def validate_or_construct_pattern(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    require_columns(out, ["vaccinated", "hbm_threat_score", "hbm_barrier_score"])
    constructed = np.select(
        [
            (out["hbm_threat_score"] >= 2) & (out["hbm_barrier_score"] <= 1),
            (out["hbm_threat_score"] >= 2) & (out["hbm_barrier_score"] >= 2),
            (out["hbm_threat_score"] < 2) & (out["hbm_barrier_score"] <= 1),
            (out["hbm_threat_score"] < 2) & (out["hbm_barrier_score"] >= 2),
        ],
        [0, 1, 2, 3],
        default=-1,
    ).astype(int)
    if "hbm2_pattern" not in out.columns:
        out["hbm2_pattern"] = constructed
    else:
        existing = pd.to_numeric(out["hbm2_pattern"], errors="coerce")
        mismatch = existing.notna() & (existing.astype("Int64") != pd.Series(constructed, index=out.index).astype("Int64"))
        n_mismatch = int(mismatch.sum())
        if n_mismatch:
            print(
                f"WARNING: {n_mismatch} rows have hbm2_pattern inconsistent with the "
                "current score rules. The script will use the reconstructed pattern."
            )
        out["hbm2_pattern"] = constructed
    return out


def balanced_sample(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size <= 0 or sample_size >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if sample_size < 4:
        raise ValueError("sample_size must be 0 or at least 4 for a train/test validation.")
    counts = df["vaccinated"].value_counts()
    if not {0, 1}.issubset(set(counts.index.astype(int))):
        raise ValueError("vaccinated must contain both 0 and 1 classes.")
    n0 = sample_size // 2
    n1 = sample_size - n0
    if counts.loc[0] < n0 or counts.loc[1] < n1:
        raise ValueError("Not enough rows in one target class for balanced sampling.")
    sampled = pd.concat(
        [
            df[df["vaccinated"] == 0].sample(n=n0, random_state=seed),
            df[df["vaccinated"] == 1].sample(n=n1, random_state=seed + 1),
        ],
        axis=0,
    )
    return sampled.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)


def ordinary_sample(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size <= 0 or sample_size >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sampled, _ = train_test_split(
        df,
        train_size=sample_size,
        random_state=seed,
        stratify=df["vaccinated"],
    )
    return sampled.reset_index(drop=True)


def split_train_test(
    df: pd.DataFrame, train_ratio: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.05 <= train_ratio <= 0.95:
        raise ValueError("train_ratio must be between 0.05 and 0.95.")
    train_df, test_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=seed,
        stratify=df["vaccinated"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare_memory_matrix(df_ordered: pd.DataFrame, train_size: int) -> Tuple[np.ndarray, List[str]]:
    features = [feature for feature in MEMORY_FEATURES if feature in df_ordered.columns]
    if not features:
        raise ValueError("No memory features are available in the cleaned dataset.")
    numeric = df_ordered[features].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_array = imputer.fit_transform(numeric.iloc[:train_size])
    train_scaled = scaler.fit_transform(train_array)
    all_array = imputer.transform(numeric)
    all_scaled = scaler.transform(all_array)
    norms = np.linalg.norm(all_scaled, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (all_scaled / norms).astype(np.float32), features


def calculate_train_pattern_rates(train_df: pd.DataFrame) -> Dict[int, float]:
    grouped = train_df.groupby("hbm2_pattern")["vaccinated"].mean()
    return {int(pattern): float(rate) for pattern, rate in grouped.items()}


def make_pattern_prior(
    pattern_id: int,
    mode: str,
    train_pattern_rates: Dict[int, float],
) -> str:
    if mode == "none":
        return "No explicit cross-pattern ordering is supplied."
    if mode == "theory":
        return THEORETICAL_PATTERN_ORDER
    rate = train_pattern_rates.get(pattern_id)
    if rate is None:
        return THEORETICAL_PATTERN_ORDER + " No training-only rate was available for this pattern."
    return (
        THEORETICAL_PATTERN_ORDER
        + f" Training-only vaccination rate for this current pattern: {rate:.1%}. "
        "This rate is a prior, not an automatic decision."
    )


# ---------------------------------------------------------------------------
# Logging, resume, and metrics
# ---------------------------------------------------------------------------
def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_latest_log_entries(path: Path) -> Dict[int, Dict[str, Any]]:
    latest: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                latest[int(entry["data_idx"])] = entry
            except Exception:
                continue
    return latest


def first_unfinished_index(latest: Dict[int, Dict[str, Any]], total: int) -> int:
    for idx in range(total):
        entry = latest.get(idx)
        if entry is None or entry.get("status") != "ok":
            return idx
    return total


def config_fingerprint(config: Dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def compute_metrics(entries: Sequence[Dict[str, Any]], phase: str) -> Dict[str, Any]:
    rows = [
        entry
        for entry in entries
        if entry.get("status") == "ok" and entry.get("phase") == phase
    ]
    if not rows:
        return {"n_valid": 0}
    y_true = np.array([int(entry["actual"]) for entry in rows], dtype=int)
    y_pred = np.array([int(entry["predicted"]) for entry in rows], dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]
    return {
        "n_valid": len(rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": safe_divide(tn, tn + fp),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "actual_yes_rate": float(y_true.mean()),
        "predicted_yes_rate": float(y_pred.mean()),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
    }


def compute_pattern_metrics(entries: Sequence[Dict[str, Any]], phase: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    phase_entries = [
        entry
        for entry in entries
        if entry.get("status") == "ok" and entry.get("phase") == phase
    ]
    for pattern_id, label in PATTERN_LABELS.items():
        subset = [entry for entry in phase_entries if int(entry["hbm2_pattern"]) == pattern_id]
        if not subset:
            continue
        y_true = np.array([int(entry["actual"]) for entry in subset], dtype=int)
        y_pred = np.array([int(entry["predicted"]) for entry in subset], dtype=int)
        rows.append(
            {
                "phase": phase,
                "pattern_id": pattern_id,
                "pattern_label": label,
                "n": len(subset),
                "actual_vaccination_rate": float(y_true.mean()),
                "predicted_yes_rate": float(y_pred.mean()),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def entries_to_predictions(entries: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for entry in sorted(entries, key=lambda item: int(item["data_idx"])):
        if entry.get("status") != "ok":
            continue
        rows.append(
            {
                "data_idx": entry["data_idx"],
                "source_row_id": entry["source_row_id"],
                "phase": entry["phase"],
                "actual": entry["actual"],
                "predicted": entry["predicted"],
                "decision": entry["decision"],
                "is_correct": entry["is_correct"],
                "hbm2_pattern": entry["hbm2_pattern"],
                "pattern_label": entry["pattern_label"],
                "rule_threat_score": entry["rule_threat_score"],
                "rule_barrier_score": entry["rule_barrier_score"],
                "llm_threat_score": entry["threat_result"]["score"],
                "llm_threat_level": entry["threat_result"]["level"],
                "llm_threat_reason": entry["threat_result"]["reason"],
                "llm_barrier_score": entry["barrier_result"]["score"],
                "llm_barrier_level": entry["barrier_result"]["level"],
                "llm_barrier_reason": entry["barrier_result"]["reason"],
                "memory_used": entry["memory_used"],
                "memory_size": entry["memory_size"],
                "sample_time_sec": entry["sample_time_sec"],
                "input_tokens": entry["usage_total"]["input_tokens"],
                "output_tokens": entry["usage_total"]["output_tokens"],
                "total_tokens": entry["usage_total"]["total_tokens"],
            }
        )
    return pd.DataFrame(rows)


def save_outputs(
    output_dir: Path,
    latest: Dict[int, Dict[str, Any]],
    run_config: Dict[str, Any],
) -> Dict[str, Any]:
    entries = [latest[idx] for idx in sorted(latest) if latest[idx].get("status") == "ok"]
    predictions = entries_to_predictions(entries)
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    pattern_frames = [
        compute_pattern_metrics(entries, "train"),
        compute_pattern_metrics(entries, "test"),
    ]
    pattern_metrics = pd.concat(
        [frame for frame in pattern_frames if not frame.empty], ignore_index=True
    ) if any(not frame.empty for frame in pattern_frames) else pd.DataFrame()
    pattern_metrics.to_csv(output_dir / "pattern_metrics.csv", index=False)

    usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for entry in entries:
        add_usage(usage_total, entry.get("usage_total", {}))

    status_counts: Dict[str, int] = {}
    for entry in latest.values():
        status = str(entry.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    summary = {
        "experiment": EXPERIMENT,
        "version": VERSION,
        "created_at": utc_now(),
        "run_config": run_config,
        "status_counts_latest_entries": status_counts,
        "train_metrics": compute_metrics(entries, "train"),
        "test_metrics": compute_metrics(entries, "test"),
        "usage_total": usage_total,
        "n_completed": len(entries),
        "files": {
            "sample_log": str(output_dir / "logs" / "sample_log.jsonl"),
            "predictions": str(output_dir / "predictions.csv"),
            "pattern_metrics": str(output_dir / "pattern_metrics.csv"),
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


# ---------------------------------------------------------------------------
# API key and CLI
# ---------------------------------------------------------------------------
def resolve_api_key(explicit_key: str) -> str:
    if explicit_key:
        return explicit_key.strip()
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    raise ValueError(
        "No API key found. Set OPENAI_API_KEY (recommended) or pass --api_key."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the revised three-call FLARE-VAX HBM2 pipeline through the official OpenAI API."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/content/drive/MyDrive/FLARE-VAX/Data/Processed/nhis2024_hbm2_clean.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/FLARE-VAX/results/gpt4o_mini_hbm2_val",
    )
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--sample_size", type=int, default=20, help="0 uses the full cleaned dataset")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--balanced_sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use equal vaccinated/unvaccinated counts in the validation sample.",
    )
    parser.add_argument("--memory_k", type=int, default=3)
    parser.add_argument(
        "--pattern_prior_mode",
        choices=["theory", "train_rate", "none"],
        default="theory",
        help=(
            "theory supplies only qualitative HBM ordering; train_rate also supplies "
            "the current pattern's rate calculated from the training split only."
        ),
    )
    parser.add_argument(
        "--include_sensitive_context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the NHIS race/Hispanic public-use group in CALL 3 context.",
    )
    parser.add_argument("--assessment_max_tokens", type=int, default=140)
    parser.add_argument("--decision_max_tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--checkpoint_every", type=int, default=1)
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue after an exhausted API/runtime error. Default stops to preserve sequential memory order.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing output directory before starting. Use carefully.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate data, split, profiles, and prompts without calling the API.",
    )
    parser.add_argument(
        "--print_samples",
        type=int,
        default=2,
        help="Print full prompt/output details for the first N newly processed samples.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()

    data_path = Path(args.data_path).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "logs" / "sample_log.jsonl"
    config_path = output_dir / "run_config.json"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\nMount Google Drive and check --data_path."
        )

    print("=" * 78)
    print("FLARE-VAX HBM2 — OpenAI API / Colab Validation")
    print("=" * 78)
    print(f"Version          : {VERSION}")
    print(f"Model            : {args.model}")
    print(f"Data             : {data_path}")
    print(f"Output           : {output_dir}")
    print(f"Sample size      : {args.sample_size}")
    print(f"Balanced sample  : {args.balanced_sample}")
    print(f"Train ratio      : {args.train_ratio}")
    print(f"Memory k         : {args.memory_k} (training errors only)")
    print(f"Pattern prior    : {args.pattern_prior_mode}")
    print(f"Dry run          : {args.dry_run}")

    df = pd.read_csv(data_path)
    warn_about_possible_global_missing_code_damage(df)
    df = validate_or_construct_pattern(df)
    require_columns(
        df,
        [
            "vaccinated",
            "age",
            "health_status",
            "hbm_threat_score",
            "hbm_barrier_score",
            "hbm2_pattern",
        ],
    )
    df["vaccinated"] = pd.to_numeric(df["vaccinated"], errors="coerce")
    df = df[df["vaccinated"].isin([0, 1])].copy()
    df = df.dropna(subset=["age", "health_status"]).copy()
    df["vaccinated"] = df["vaccinated"].astype(int)
    df["_source_row_id"] = df.index.astype(int)

    if args.balanced_sample:
        sampled = balanced_sample(df, args.sample_size, args.random_seed)
    else:
        sampled = ordinary_sample(df, args.sample_size, args.random_seed)

    train_df, test_df = split_train_test(sampled, args.train_ratio, args.random_seed)
    train_df["_phase"] = "train"
    test_df["_phase"] = "test"
    ordered = pd.concat([train_df, test_df], ignore_index=True)
    train_size = len(train_df)
    total_size = len(ordered)
    feature_matrix, memory_features = prepare_memory_matrix(ordered, train_size)
    train_pattern_rates = calculate_train_pattern_rates(train_df)

    print(f"Clean rows used  : {len(df):,}")
    print(f"Selected rows    : {total_size} (train={train_size}, test={len(test_df)})")
    print(f"Target rate      : {ordered['vaccinated'].mean():.1%}")
    print(f"Memory features  : {len(memory_features)}")
    print("Selected-sample pattern distribution:")
    for pattern_id, group in ordered.groupby("hbm2_pattern"):
        print(
            f"  P{int(pattern_id)} {PATTERN_LABELS[int(pattern_id)]:<29s} "
            f"n={len(group):4d} actual_vax={group['vaccinated'].mean():.1%}"
        )

    run_config: Dict[str, Any] = {
        "version": VERSION,
        "model": args.model,
        "data_path": str(data_path.resolve()),
        "data_size_bytes": data_path.stat().st_size,
        "sample_size": args.sample_size,
        "balanced_sample": args.balanced_sample,
        "train_ratio": args.train_ratio,
        "random_seed": args.random_seed,
        "memory_k": args.memory_k,
        "memory_features": memory_features,
        "pattern_prior_mode": args.pattern_prior_mode,
        "include_sensitive_context": args.include_sensitive_context,
        "assessment_max_tokens": args.assessment_max_tokens,
        "decision_max_tokens": args.decision_max_tokens,
        "temperature": args.temperature,
    }
    run_config["fingerprint"] = config_fingerprint(run_config)

    if config_path.exists() and log_path.exists() and not args.overwrite:
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config.get("fingerprint") != run_config["fingerprint"]:
            raise RuntimeError(
                "The output directory contains a different run configuration. "
                "Use a new --output_dir or pass --overwrite."
            )
    else:
        config_path.write_text(
            json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if args.dry_run:
        first = ordered.iloc[0]
        threat_profile = build_threat_profile(first)
        barrier_profile = build_barrier_profile(first)
        context_profile = build_context_profile(first, args.include_sensitive_context)
        print("\n--- DRY RUN: CALL 1 INPUT ---")
        print(THREAT_USER_TEMPLATE.format(profile=threat_profile))
        print("\n--- DRY RUN: CALL 2 INPUT ---")
        print(BARRIER_USER_TEMPLATE.format(profile=barrier_profile))
        print("\n--- DRY RUN: CALL 3 CONTEXT SKELETON ---")
        print(context_profile)
        print("\nDry run completed. No API calls were made.")
        return

    if OpenAI is None:
        raise SystemExit(
            "The OpenAI Python SDK is not installed. Run: pip install -U openai"
        )

    api_key = resolve_api_key(args.api_key)
    client = OpenAI(
        api_key=api_key,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    latest = load_latest_log_entries(log_path)
    start_idx = first_unfinished_index(latest, total_size)
    memory = SimilarityErrorMemory(k=args.memory_k)

    # Rebuild memory from completed TRAIN errors only, in chronological order.
    for idx in range(min(start_idx, train_size)):
        entry = latest.get(idx)
        if entry and entry.get("status") == "ok" and not bool(entry.get("is_correct")):
            memory.store(idx, feature_matrix[idx], build_memory_record(entry))

    print(f"Resume position   : {start_idx}/{total_size}")
    print(f"Rebuilt memory    : {memory.size()} training errors")

    newly_processed = 0
    run_start = time.time()

    for idx in tqdm(range(start_idx, total_size), desc="FLARE-VAX HBM2"):
        row = ordered.iloc[idx]
        phase = str(row["_phase"])
        actual = int(row["vaccinated"])
        pattern_id = int(row["hbm2_pattern"])
        pattern_label = PATTERN_LABELS[pattern_id]
        sample_start = time.time()

        threat_profile = build_threat_profile(row)
        barrier_profile = build_barrier_profile(row)
        context_profile = build_context_profile(row, args.include_sensitive_context)
        context_summary = compact_context_summary(row)

        threat_user = THREAT_USER_TEMPLATE.format(profile=threat_profile)
        barrier_user = BARRIER_USER_TEMPLATE.format(profile=barrier_profile)

        usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        try:
            threat_obj, threat_raw, threat_usage, threat_request_id = structured_response_call(
                client,
                model=args.model,
                schema_name="hbm_threat_assessment",
                schema=ASSESSMENT_SCHEMA,
                system_prompt=THREAT_SYSTEM,
                user_prompt=threat_user,
                max_output_tokens=args.assessment_max_tokens,
                temperature=args.temperature,
            )
            threat_result = validate_assessment(threat_obj, "threat")
            add_usage(usage_total, threat_usage)

            barrier_obj, barrier_raw, barrier_usage, barrier_request_id = structured_response_call(
                client,
                model=args.model,
                schema_name="hbm_barrier_assessment",
                schema=ASSESSMENT_SCHEMA,
                system_prompt=BARRIER_SYSTEM,
                user_prompt=barrier_user,
                max_output_tokens=args.assessment_max_tokens,
                temperature=args.temperature,
            )
            barrier_result = validate_assessment(barrier_obj, "barrier")
            add_usage(usage_total, barrier_usage)

            memory_text, memory_ids = memory.get(feature_matrix[idx])
            pattern_prior = make_pattern_prior(
                pattern_id, args.pattern_prior_mode, train_pattern_rates
            )
            decision_user = DECISION_USER_TEMPLATE.format(
                threat_result=json.dumps(threat_result, ensure_ascii=False),
                barrier_result=json.dumps(barrier_result, ensure_ascii=False),
                pattern_id=pattern_id,
                pattern_label=pattern_label,
                rule_threat_score=int(row["hbm_threat_score"]),
                rule_barrier_score=int(row["hbm_barrier_score"]),
                pattern_tendency=PATTERN_TENDENCIES[pattern_id],
                pattern_prior=pattern_prior,
                context_profile=context_profile,
                memory_text=memory_text,
            )

            decision_obj, decision_raw, decision_usage, decision_request_id = structured_response_call(
                client,
                model=args.model,
                schema_name="vaccination_decision",
                schema=DECISION_SCHEMA,
                system_prompt=DECISION_SYSTEM,
                user_prompt=decision_user,
                max_output_tokens=args.decision_max_tokens,
                temperature=args.temperature,
            )
            add_usage(usage_total, decision_usage)
            decision = str(decision_obj["decision"]).upper()
            if decision not in {"YES", "NO"}:
                raise ValueError(f"Invalid decision: {decision}")
            predicted = 1 if decision == "YES" else 0
            is_correct = bool(predicted == actual)

            entry: Dict[str, Any] = {
                "status": "ok",
                "experiment": EXPERIMENT,
                "version": VERSION,
                "timestamp": utc_now(),
                "data_idx": idx,
                "source_row_id": int(row["_source_row_id"]),
                "phase": phase,
                "model": args.model,
                "actual": actual,
                "predicted": predicted,
                "decision": decision,
                "is_correct": is_correct,
                "error_type": (
                    "correct"
                    if is_correct
                    else "false_positive"
                    if predicted == 1
                    else "false_negative"
                ),
                "hbm2_pattern": pattern_id,
                "pattern_label": pattern_label,
                "rule_threat_score": int(row["hbm_threat_score"]),
                "rule_barrier_score": int(row["hbm_barrier_score"]),
                "threat_profile": threat_profile,
                "barrier_profile": barrier_profile,
                "context_profile": context_profile,
                "context_summary": context_summary,
                "threat_result": threat_result,
                "barrier_result": barrier_result,
                "memory_used": bool(memory_ids),
                "memory_source_indices": memory_ids,
                "memory_size": memory.size(),
                "memory_frozen_in_test": True,
                "prompts": {
                    "threat_system": THREAT_SYSTEM,
                    "threat_user": threat_user,
                    "barrier_system": BARRIER_SYSTEM,
                    "barrier_user": barrier_user,
                    "decision_system": DECISION_SYSTEM,
                    "decision_user": decision_user,
                },
                "raw_outputs": {
                    "threat": threat_raw,
                    "barrier": barrier_raw,
                    "decision": decision_raw,
                },
                "request_ids": {
                    "threat": threat_request_id,
                    "barrier": barrier_request_id,
                    "decision": decision_request_id,
                },
                "usage": {
                    "threat": threat_usage,
                    "barrier": barrier_usage,
                    "decision": decision_usage,
                },
                "usage_total": usage_total,
                "sample_time_sec": round(time.time() - sample_start, 3),
            }
            append_jsonl(log_path, entry)
            latest[idx] = entry

            # Online error memory during train only. Test memory is frozen.
            if phase == "train" and not is_correct:
                memory.store(idx, feature_matrix[idx], build_memory_record(entry))

            newly_processed += 1
            if newly_processed <= args.print_samples:
                print("\n" + "=" * 78)
                print(
                    f"SAMPLE {idx} | phase={phase} | actual={actual} | "
                    f"decision={decision} | correct={is_correct}"
                )
                print(f"Pattern: P{pattern_id} {pattern_label}")
                print(f"Threat output: {json.dumps(threat_result, ensure_ascii=False)}")
                print(f"Barrier output: {json.dumps(barrier_result, ensure_ascii=False)}")
                print(f"Decision output: {decision_raw}")
                print(f"Memory examples used: {memory_ids}")
                print(f"Tokens: {usage_total}")
                print("=" * 78)

        except Exception as exc:
            error_entry = {
                "status": "error",
                "experiment": EXPERIMENT,
                "version": VERSION,
                "timestamp": utc_now(),
                "data_idx": idx,
                "source_row_id": int(row["_source_row_id"]),
                "phase": phase,
                "model": args.model,
                "actual": actual,
                "hbm2_pattern": pattern_id,
                "pattern_label": pattern_label,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(limit=8),
                "usage_total": usage_total,
                "sample_time_sec": round(time.time() - sample_start, 3),
            }
            append_jsonl(log_path, error_entry)
            latest[idx] = error_entry
            print(f"\nERROR at sample {idx}: {type(exc).__name__}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                print(
                    "Stopping at the failed sample to preserve sequential memory order. "
                    "Re-run the same command to retry and resume.",
                    file=sys.stderr,
                )
                break

        if args.checkpoint_every > 0 and newly_processed > 0 and newly_processed % args.checkpoint_every == 0:
            summary = save_outputs(output_dir, latest, run_config)
            elapsed = time.time() - run_start
            completed = summary["n_completed"]
            test_accuracy = summary.get("test_metrics", {}).get("accuracy")
            accuracy_text = "N/A" if test_accuracy is None else f"{test_accuracy:.3f}"
            print(
                f"\nCheckpoint: completed={completed}/{total_size}, "
                f"test_accuracy={accuracy_text}, elapsed={elapsed/60:.1f} min, "
                f"memory={memory.size()}"
            )

    latest = load_latest_log_entries(log_path)
    summary = save_outputs(output_dir, latest, run_config)

    print("\n" + "=" * 78)
    print("RUN SUMMARY")
    print("=" * 78)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSample log     : {log_path}")
    print(f"Predictions    : {output_dir / 'predictions.csv'}")
    print(f"Pattern metrics: {output_dir / 'pattern_metrics.csv'}")
    print(f"Summary        : {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
