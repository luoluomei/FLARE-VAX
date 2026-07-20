#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLARE-VAX HBM2 — OpenAI API / Colab reflection-memory + calibration runner
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
* Samples are processed in configurable concurrent batches. CALL 1 and CALL 2
  are issued concurrently, and multiple samples can be in flight at once. Each
  completed batch is appended to JSONL in data-index order. Re-running the same
  command with the same output directory resumes from the first unfinished sample.

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
   !python /content/drive/MyDrive/FLARE-VAX/run_flare_vax_hbm2_openai_colab_concurrent.py \
       --data_path /content/drive/MyDrive/FLARE-VAX/Data/Processed/nhis2024_hbm2_clean.csv \
       --output_dir /content/drive/MyDrive/FLARE-VAX/results/gpt4o_mini_hbm2_val \
       --model gpt-4o-mini \
       --sample_size 20 \
       --checkpoint_every 1 \
       --pattern_prior_mode theory \
       --concurrent_samples 4 \
       --max_concurrent_requests 8

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
import asyncio
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
    from openai import AsyncOpenAI
except ImportError:  # Allow --dry_run without the SDK installed.
    AsyncOpenAI = None  # type: ignore[assignment,misc]


VERSION = "hbm2_openai_colab_v3_reflection_calibrated"
EXPERIMENT = "flare_vax_hbm2_reflection_calibrated"

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


async def structured_response_call(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    *,
    model: str,
    schema_name: str,
    schema: Dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    temperature: float,
) -> Tuple[Dict[str, Any], str, Dict[str, int], str]:
    """Issue one asynchronous Responses API request under a concurrency limit."""
    async with semaphore:
        response = await client.responses.create(
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
        "--concurrent_samples",
        type=int,
        default=4,
        help=(
            "Number of respondents processed in one concurrent batch. "
            "Use 1 to preserve exact sample-by-sample memory updates while still "
            "running CALL 1 and CALL 2 concurrently."
        ),
    )
    parser.add_argument(
        "--max_concurrent_requests",
        type=int,
        default=8,
        help=(
            "Global cap on simultaneous OpenAI requests. A batch can generate up "
            "to two assessment requests per respondent, followed by decision requests."
        ),
    )
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
# V3: authoritative HBM pattern, staged concurrency, reflection memory,
#     and training-only probability calibration
# ---------------------------------------------------------------------------
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

VERSION = "hbm2_openai_colab_v3_reflection_calibrated"
EXPERIMENT = "flare_vax_hbm2_reflection_calibrated"


def authoritative_assessment_schema(expected_level: str) -> Dict[str, Any]:
    """Force CALL 1/2 to respect both the rule-based class and its valid score range.

    Low-class assessments may use scores 1-3; high-class assessments may use
    scores 3-5. Score 3 is intentionally shared as the boundary value.
    """
    if expected_level not in {"low", "high"}:
        raise ValueError(f"Unexpected authoritative level: {expected_level}")
    allowed_scores = [1, 2, 3] if expected_level == "low" else [3, 4, 5]
    return {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "enum": allowed_scores},
            "level": {"type": "string", "enum": [expected_level]},
            "reason": {"type": "string"},
        },
        "required": ["score", "level", "reason"],
        "additionalProperties": False,
    }


DECISION_SCHEMA_V3: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "probability_yes": {"type": "integer", "minimum": 0, "maximum": 100},
        "decision": {"type": "string", "enum": ["YES", "NO"]},
        "reason": {"type": "string"},
    },
    "required": ["probability_yes", "decision", "reason"],
    "additionalProperties": False,
}


REFLECTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "error_cause": {"type": "string"},
        "missed_or_overweighted_signal": {"type": "string"},
        "correction_rule": {"type": "string"},
        "applicable_pattern": {"type": "string", "enum": ["P0", "P1", "P2", "P3"]},
    },
    "required": [
        "error_cause",
        "missed_or_overweighted_signal",
        "correction_rule",
        "applicable_pattern",
    ],
    "additionalProperties": False,
}


THREAT_SYSTEM_V3 = """You are a public-health behavioral analyst using the Health Belief Model.
The cleaned dataset has already assigned an authoritative rule-based threat class (High or Low) from age, health status, and chronic/functional risk indicators.
Your task is only to explain and refine that fixed classification with a 1-5 score. Do not change the supplied High/Low class, do not predict vaccination, and do not discuss barriers, benefits, or cues.
Return only the required structured object. The reason must be exactly one concise sentence of at most 30 words."""


THREAT_USER_TEMPLATE_V3 = """TASK
Explain and refine the authoritative rule-based perceived-threat classification for influenza.

DEFINITION
Perceived threat combines susceptibility to influenza and severity of likely consequences. The supplied variables are observable risk proxies rather than direct measures of private beliefs.

AUTHORITATIVE RULE-BASED CLASSIFICATION
Rule threat score: {rule_score}/4
Rule threat level: {rule_level}
This High/Low classification is authoritative and must not be changed.

1-5 NUANCE RUBRIC
1 = very low risk evidence within the Low class.
2 = clearly low risk evidence within the Low class.
3 = borderline or mixed evidence near the High/Low boundary; retain the supplied authoritative class.
4 = clearly high risk evidence within the High class.
5 = very high or concentrated risk evidence within the High class.

INDIVIDUAL VARIABLES
{profile}

OUTPUT
Return only score, the supplied rule level, and one concise reason."""


BARRIER_SYSTEM_V3 = """You are a public-health behavioral analyst using the Health Belief Model.
The cleaned dataset has already assigned an authoritative rule-based barrier class (High or Low) from insurance, affordability, healthcare access, transportation, language, and digital-access indicators.
Your task is only to explain and refine that fixed classification with a 1-5 score. Do not change the supplied High/Low class, do not predict vaccination, and do not use threat information.
Return only the required structured object. The reason must be exactly one concise sentence of at most 30 words."""


BARRIER_USER_TEMPLATE_V3 = """TASK
Explain and refine the authoritative rule-based perceived-barrier classification for obtaining a flu vaccination.

DEFINITION
Perceived barriers are observable financial, insurance, healthcare-access, transportation, language, and digital obstacles that can make vaccination harder.

AUTHORITATIVE RULE-BASED CLASSIFICATION
Rule barrier score: {rule_score}/3
Rule barrier level: {rule_level}
This High/Low classification is authoritative and must not be changed.

1-5 NUANCE RUBRIC
1 = essentially no observable barrier within the Low class.
2 = limited or mild barriers within the Low class.
3 = borderline or mixed evidence near the High/Low boundary; retain the supplied authoritative class.
4 = multiple material barriers within the High class.
5 = severe, combined, or persistent barriers within the High class.

INDIVIDUAL VARIABLES
{profile}

OUTPUT
Return only score, the supplied rule level, and one concise reason."""


DECISION_SYSTEM_V3 = """You predict whether an NHIS respondent received a flu vaccine in the past 12 months.
The rule-based HBM2 High/Low threat and barrier pattern is the authoritative behavioral classification. CALL 1 and CALL 2 only explain and refine it; they must not redefine the pattern.
Use the pattern as a directional prior, then use additional context and genuinely similar reflected training errors to adjust the probability.
Return a calibrated probability from 0 to 100, a YES/NO decision consistent with a 50 percent internal cutoff, and exactly one concise reason of at most 30 words. Do not output anything outside the structured object."""


DECISION_USER_TEMPLATE_V3 = """TASK
Estimate the probability that this respondent received a flu vaccine in the past 12 months.

AUTHORITATIVE HBM2 CLASSIFICATION
Pattern ID: P{pattern_id}
Pattern: {pattern_label}
Rule threat score: {rule_threat_score}/4 ({rule_threat_level})
Rule barrier score: {rule_barrier_score}/3 ({rule_barrier_level})
Pattern tendency: {pattern_tendency}
{pattern_prior}

CALL 1 — THREAT EXPLANATION/NUANCE
{threat_result}

CALL 2 — BARRIER EXPLANATION/NUANCE
{barrier_result}

ADDITIONAL CONTEXT OUTSIDE THE TWO CONSTRUCT SCORES
{context_profile}

REFLECTION MEMORY FROM SIMILAR TRAINING ERRORS
{memory_text}

DECISION INSTRUCTIONS
1. Treat the rule-based pattern as authoritative.
2. Use CALL 1/2 only as nuance inside that pattern.
3. Adjust with context only when it supplies clear behavioral evidence.
4. Apply a retrieved correction rule only when the prior case is genuinely similar.
5. probability_yes must be an integer from 0 to 100.
6. decision must be YES when probability_yes >= 50 and NO when probability_yes < 50.
7. Give exactly one concise reason."""


REFLECTION_SYSTEM = """You are the self-reflection component of a FLARE-style behavioral prediction system.
The following is a TRAINING error, so the true outcome is available. Diagnose why the prediction was wrong and create a transferable correction for genuinely similar future respondents.
Do not merely repeat that the label differed. Identify the neglected or overweighted behavioral evidence and provide a specific correction rule.
Return only the required structured object. Each text field must be one concise sentence of at most 35 words."""


REFLECTION_USER_TEMPLATE = """TRAINING ERROR TO REFLECT ON
Actual vaccination outcome: {actual_label}
Predicted probability of YES: {probability_yes}%
Prediction at the internal 50% cutoff: {predicted_label}
Decision rationale: {decision_reason}

AUTHORITATIVE HBM2 PATTERN
P{pattern_id} {pattern_label}
Rule threat score: {rule_threat_score}/4 ({rule_threat_level})
Rule barrier score: {rule_barrier_score}/3 ({rule_barrier_level})

CALL 1 THREAT RESULT
{threat_result}

CALL 2 BARRIER RESULT
{barrier_result}

ADDITIONAL CONTEXT
{context_profile}

SIMILAR REFLECTIONS USED BEFORE THIS ERROR
{memory_text}

REFLECTION TASK
Explain the error cause, identify the most important missed or overweighted signal, and write a correction rule for similar future cases. The applicable pattern must be the current pattern."""


def validate_authoritative_assessment(
    obj: Dict[str, Any], name: str, expected_level: str
) -> Dict[str, Any]:
    """Validate CALL 1/2 and defensively repair a class-score mismatch.

    The JSON schema already constrains the valid score enum. This fallback keeps
    a long concurrent run from terminating if an older cached response or an
    unexpected provider response still violates the class-specific range.
    """
    raw_score = int(obj["score"])
    if not 1 <= raw_score <= 5:
        raise ValueError(f"{name} score outside 1-5: {raw_score}")
    level = str(obj["level"]).lower()
    if level != expected_level:
        raise ValueError(
            f"{name} returned level={level}, expected authoritative level={expected_level}"
        )

    score = raw_score
    consistency_corrected = False
    if expected_level == "low" and score > 3:
        score = 3
        consistency_corrected = True
    elif expected_level == "high" and score < 3:
        score = 3
        consistency_corrected = True

    return {
        "score": score,
        "level": level,
        "reason": normalize_reason(obj["reason"], max_chars=220),
        "raw_score": raw_score,
        "score_consistency_corrected": consistency_corrected,
    }


def validate_decision_v3(obj: Dict[str, Any]) -> Dict[str, Any]:
    probability = int(obj["probability_yes"])
    if not 0 <= probability <= 100:
        raise ValueError(f"probability_yes outside 0-100: {probability}")
    decision = str(obj["decision"]).upper()
    if decision not in {"YES", "NO"}:
        raise ValueError(f"Invalid decision: {decision}")
    expected = "YES" if probability >= 50 else "NO"
    consistency_corrected = decision != expected
    # Probability is the authoritative numeric output for downstream calibration.
    decision = expected
    return {
        "probability_yes": probability,
        "decision": decision,
        "reason": normalize_reason(obj["reason"], max_chars=220),
        "decision_consistency_corrected": consistency_corrected,
    }


def validate_reflection(obj: Dict[str, Any], pattern_id: int) -> Dict[str, Any]:
    expected_pattern = f"P{pattern_id}"
    applicable = str(obj["applicable_pattern"]).upper()
    if applicable != expected_pattern:
        raise ValueError(
            f"Reflection applicable_pattern={applicable}, expected {expected_pattern}"
        )
    return {
        "error_cause": normalize_reason(obj["error_cause"], max_chars=260),
        "missed_or_overweighted_signal": normalize_reason(
            obj["missed_or_overweighted_signal"], max_chars=260
        ),
        "correction_rule": normalize_reason(obj["correction_rule"], max_chars=300),
        "applicable_pattern": applicable,
    }


def rule_levels(row: pd.Series) -> Tuple[str, str]:
    threat_level = "high" if int(row["hbm_threat_score"]) >= 2 else "low"
    barrier_level = "high" if int(row["hbm_barrier_score"]) >= 2 else "low"
    return threat_level, barrier_level


# ---------------------------------------------------------------------------
# Reflection memory with same-pattern priority
# ---------------------------------------------------------------------------
@dataclass
class ReflectionMemoryItem:
    data_idx: int
    pattern_id: int
    vector: np.ndarray
    text: str
    reflection: Dict[str, Any]


class ReflectionSimilarityMemory:
    def __init__(self, k: int = 3, min_similarity: float = -1.0):
        self.k = max(0, int(k))
        self.min_similarity = float(min_similarity)
        self.items: List[ReflectionMemoryItem] = []

    def store(
        self,
        data_idx: int,
        pattern_id: int,
        vector: np.ndarray,
        text: str,
        reflection: Dict[str, Any],
    ) -> None:
        if self.k <= 0:
            return
        self.items.append(
            ReflectionMemoryItem(
                data_idx=int(data_idx),
                pattern_id=int(pattern_id),
                vector=vector.astype(np.float32).copy(),
                text=str(text),
                reflection=dict(reflection),
            )
        )

    def get(
        self, query_vector: np.ndarray, pattern_id: int
    ) -> Tuple[str, List[int], List[float]]:
        if self.k <= 0 or not self.items:
            return "No reflected training errors are available.", [], []
        query = query_vector.astype(np.float32)
        scored = [
            (float(item.vector @ query), pos, item)
            for pos, item in enumerate(self.items)
        ]
        # Same-pattern reflections are always considered first; remaining slots
        # can be filled globally when the same-pattern memory is sparse.
        same = sorted(
            [x for x in scored if x[2].pattern_id == int(pattern_id)],
            key=lambda x: -x[0],
        )
        other = sorted(
            [x for x in scored if x[2].pattern_id != int(pattern_id)],
            key=lambda x: -x[0],
        )
        selected: List[Tuple[float, int, ReflectionMemoryItem]] = []
        for candidate in same + other:
            if candidate[0] < self.min_similarity:
                continue
            selected.append(candidate)
            if len(selected) >= self.k:
                break
        if not selected:
            return "No sufficiently similar reflected training errors are available.", [], []
        blocks: List[str] = []
        ids: List[int] = []
        sims: List[float] = []
        for rank, (similarity, _, item) in enumerate(selected, start=1):
            ids.append(item.data_idx)
            sims.append(similarity)
            priority = "same pattern" if item.pattern_id == int(pattern_id) else "fallback pattern"
            blocks.append(
                f"Reflected training error {rank} ({priority}; cosine similarity={similarity:.3f}):\n"
                f"{item.text}"
            )
        return "\n\n".join(blocks), ids, sims

    def size(self) -> int:
        return len(self.items)


def build_reflection_memory_text(entry: Dict[str, Any]) -> str:
    reflection = entry["reflection"]
    return (
        f"Training pattern: P{entry['hbm2_pattern']} {entry['pattern_label']}. "
        f"Predicted YES probability={entry['probability_yes']}%; "
        f"prediction at 50%={entry['decision_at_50']}; actual={entry['actual_label']}. "
        f"Error cause: {reflection['error_cause']} "
        f"Missed/overweighted signal: {reflection['missed_or_overweighted_signal']} "
        f"Correction rule: {reflection['correction_rule']}"
    )


# ---------------------------------------------------------------------------
# Improved memory features: continuous scaling + categorical one-hot encoding
# ---------------------------------------------------------------------------
def prepare_memory_matrix_v3(
    df_ordered: pd.DataFrame, train_size: int
) -> Tuple[np.ndarray, List[str]]:
    continuous_candidates = [
        "age",
        "health_status",
        "chronic_or_risk_count",
        "hbm_threat_score",
        "hbm_barrier_count",
        "hbm_barrier_score",
        "retail_clinic_visits_12m",
    ]
    categorical_candidates = [
        "hbm2_pattern",
        "uninsured_yes",
        "medicare_yes",
        "medicaid_yes",
        "private_insurance_yes",
        "usual_care_place",
        "last_doctor_visit",
        "used_internet_test_results_yes",
        "used_internet_communicate_doctor_yes",
        "education",
        "income_poverty_ratio",
        "time_since_wellness_visit",
    ]
    continuous = [c for c in continuous_candidates if c in df_ordered.columns]
    categorical = [c for c in categorical_candidates if c in df_ordered.columns]
    blocks: List[np.ndarray] = []
    names: List[str] = []

    if continuous:
        numeric = df_ordered[continuous].apply(pd.to_numeric, errors="coerce")
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        train_num = imputer.fit_transform(numeric.iloc[:train_size])
        scaler.fit(train_num)
        all_num = scaler.transform(imputer.transform(numeric)).astype(np.float32)
        blocks.append(all_num)
        names.extend(continuous)

    if categorical:
        cat = df_ordered[categorical].copy()
        for column in categorical:
            cat[column] = cat[column].where(~cat[column].isna(), "MISSING").astype(str)
        encoded = pd.get_dummies(cat, columns=categorical, dummy_na=False, dtype=float)
        blocks.append(encoded.to_numpy(dtype=np.float32))
        names.extend(encoded.columns.astype(str).tolist())

    if not blocks:
        raise ValueError("No usable memory features were found.")
    matrix = np.concatenate(blocks, axis=1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32), names


# ---------------------------------------------------------------------------
# Joint-stratified split (target × pattern) when feasible
# ---------------------------------------------------------------------------
def split_train_test_v3(
    df: pd.DataFrame, train_ratio: float, seed: int, joint_stratify: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.05 <= train_ratio <= 0.95:
        raise ValueError("train_ratio must be between 0.05 and 0.95.")
    stratify = df["vaccinated"]
    if joint_stratify:
        joint = df["vaccinated"].astype(str) + "_P" + df["hbm2_pattern"].astype(str)
        if joint.value_counts().min() >= 2:
            stratify = joint
        else:
            print("WARNING: joint vaccinated×pattern stratification was not feasible; using target-only stratification.")
    train_df, test_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=seed,
        stratify=stratify,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Staged assessment precomputation
# ---------------------------------------------------------------------------
async def precompute_one_assessment(
    *,
    idx: int,
    row: pd.Series,
    args: argparse.Namespace,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    start = time.time()
    phase = str(row["_phase"])
    threat_level, barrier_level = rule_levels(row)
    threat_profile = build_threat_profile(row)
    barrier_profile = build_barrier_profile(row)
    threat_user = THREAT_USER_TEMPLATE_V3.format(
        rule_score=int(row["hbm_threat_score"]),
        rule_level=threat_level,
        profile=threat_profile,
    )
    barrier_user = BARRIER_USER_TEMPLATE_V3.format(
        rule_score=int(row["hbm_barrier_score"]),
        rule_level=barrier_level,
        profile=barrier_profile,
    )
    usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    try:
        threat_task = structured_response_call(
            client,
            semaphore,
            model=args.model,
            schema_name=f"authoritative_hbm_threat_{threat_level}",
            schema=authoritative_assessment_schema(threat_level),
            system_prompt=THREAT_SYSTEM_V3,
            user_prompt=threat_user,
            max_output_tokens=args.assessment_max_tokens,
            temperature=args.temperature,
        )
        barrier_task = structured_response_call(
            client,
            semaphore,
            model=args.model,
            schema_name=f"authoritative_hbm_barrier_{barrier_level}",
            schema=authoritative_assessment_schema(barrier_level),
            system_prompt=BARRIER_SYSTEM_V3,
            user_prompt=barrier_user,
            max_output_tokens=args.assessment_max_tokens,
            temperature=args.temperature,
        )
        threat_tuple, barrier_tuple = await asyncio.gather(threat_task, barrier_task)
        threat_obj, threat_raw, threat_usage, threat_request_id = threat_tuple
        barrier_obj, barrier_raw, barrier_usage, barrier_request_id = barrier_tuple
        threat_result = validate_authoritative_assessment(
            threat_obj, "threat", threat_level
        )
        barrier_result = validate_authoritative_assessment(
            barrier_obj, "barrier", barrier_level
        )
        add_usage(usage_total, threat_usage)
        add_usage(usage_total, barrier_usage)
        return {
            "status": "ok",
            "timestamp": utc_now(),
            "version": VERSION,
            "data_idx": idx,
            "source_row_id": int(row["_source_row_id"]),
            "phase": phase,
            "hbm2_pattern": int(row["hbm2_pattern"]),
            "rule_threat_score": int(row["hbm_threat_score"]),
            "rule_threat_level": threat_level,
            "rule_barrier_score": int(row["hbm_barrier_score"]),
            "rule_barrier_level": barrier_level,
            "threat_profile": threat_profile,
            "barrier_profile": barrier_profile,
            "threat_result": threat_result,
            "barrier_result": barrier_result,
            "prompts": {
                "threat_system": THREAT_SYSTEM_V3,
                "threat_user": threat_user,
                "barrier_system": BARRIER_SYSTEM_V3,
                "barrier_user": barrier_user,
            },
            "raw_outputs": {"threat": threat_raw, "barrier": barrier_raw},
            "request_ids": {
                "threat": threat_request_id,
                "barrier": barrier_request_id,
            },
            "usage": {"threat": threat_usage, "barrier": barrier_usage},
            "usage_total": usage_total,
            "elapsed_sec": round(time.time() - start, 3),
        }
    except Exception as exc:
        return {
            "status": "error",
            "timestamp": utc_now(),
            "version": VERSION,
            "data_idx": idx,
            "source_row_id": int(row["_source_row_id"]),
            "phase": phase,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(limit=8),
            "usage_total": usage_total,
            "elapsed_sec": round(time.time() - start, 3),
        }


async def precompute_assessments_for_indices(
    *,
    indices: Sequence[int],
    ordered: pd.DataFrame,
    args: argparse.Namespace,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    assessment_log: Path,
    assessment_latest: Dict[int, Dict[str, Any]],
    phase_name: str,
) -> None:
    missing = [
        idx
        for idx in indices
        if assessment_latest.get(idx, {}).get("status") != "ok"
    ]
    if not missing:
        print(f"{phase_name} CALL 1/2 assessments are already complete.")
        return
    print(f"Precomputing CALL 1/2 for {len(missing)} {phase_name} respondents...")
    batch_size = max(1, int(args.assessment_batch_size))
    progress = tqdm(total=len(missing), desc=f"{phase_name} CALL1/2")
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        results = await asyncio.gather(
            *[
                precompute_one_assessment(
                    idx=idx,
                    row=ordered.iloc[idx],
                    args=args,
                    client=client,
                    semaphore=semaphore,
                )
                for idx in batch
            ]
        )
        for entry in sorted(results, key=lambda x: int(x["data_idx"])):
            append_jsonl(assessment_log, entry)
            assessment_latest[int(entry["data_idx"])] = entry
            progress.update(1)
            if entry.get("status") != "ok" and not args.continue_on_error:
                progress.close()
                raise RuntimeError(
                    f"Assessment failed at data_idx={entry['data_idx']}: "
                    f"{entry.get('error_type')}: {entry.get('error_message')}"
                )
    progress.close()


# ---------------------------------------------------------------------------
# CALL 3 and reflection
# ---------------------------------------------------------------------------
def make_decision_prompt_v3(
    *,
    row: pd.Series,
    assessment: Dict[str, Any],
    args: argparse.Namespace,
    train_pattern_rates: Dict[int, float],
    memory_text: str,
) -> Tuple[str, str]:
    pattern_id = int(row["hbm2_pattern"])
    threat_level, barrier_level = rule_levels(row)
    context_profile = build_context_profile(row, args.include_sensitive_context)
    user_prompt = DECISION_USER_TEMPLATE_V3.format(
        pattern_id=pattern_id,
        pattern_label=PATTERN_LABELS[pattern_id],
        rule_threat_score=int(row["hbm_threat_score"]),
        rule_threat_level=threat_level,
        rule_barrier_score=int(row["hbm_barrier_score"]),
        rule_barrier_level=barrier_level,
        pattern_tendency=PATTERN_TENDENCIES[pattern_id],
        pattern_prior=make_pattern_prior(
            pattern_id, args.pattern_prior_mode, train_pattern_rates
        ),
        threat_result=json.dumps(assessment["threat_result"], ensure_ascii=False),
        barrier_result=json.dumps(assessment["barrier_result"], ensure_ascii=False),
        context_profile=context_profile,
        memory_text=memory_text,
    )
    return user_prompt, context_profile


async def call_decision_v3(
    *,
    row: pd.Series,
    assessment: Dict[str, Any],
    args: argparse.Namespace,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    train_pattern_rates: Dict[int, float],
    memory_text: str,
) -> Tuple[Dict[str, Any], str, Dict[str, int], str, str]:
    user_prompt, context_profile = make_decision_prompt_v3(
        row=row,
        assessment=assessment,
        args=args,
        train_pattern_rates=train_pattern_rates,
        memory_text=memory_text,
    )
    obj, raw, usage, request_id = await structured_response_call(
        client,
        semaphore,
        model=args.model,
        schema_name="vaccination_probability_decision",
        schema=DECISION_SCHEMA_V3,
        system_prompt=DECISION_SYSTEM_V3,
        user_prompt=user_prompt,
        max_output_tokens=args.decision_max_tokens,
        temperature=args.temperature,
    )
    return validate_decision_v3(obj), raw, usage, request_id, context_profile


async def call_reflection(
    *,
    row: pd.Series,
    assessment: Dict[str, Any],
    decision_result: Dict[str, Any],
    memory_text: str,
    context_profile: str,
    args: argparse.Namespace,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
) -> Tuple[Dict[str, Any], str, Dict[str, int], str, str]:
    pattern_id = int(row["hbm2_pattern"])
    threat_level, barrier_level = rule_levels(row)
    actual = int(row["vaccinated"])
    predicted = 1 if decision_result["probability_yes"] >= 50 else 0
    user_prompt = REFLECTION_USER_TEMPLATE.format(
        actual_label="YES" if actual == 1 else "NO",
        probability_yes=decision_result["probability_yes"],
        predicted_label="YES" if predicted == 1 else "NO",
        decision_reason=decision_result["reason"],
        pattern_id=pattern_id,
        pattern_label=PATTERN_LABELS[pattern_id],
        rule_threat_score=int(row["hbm_threat_score"]),
        rule_threat_level=threat_level,
        rule_barrier_score=int(row["hbm_barrier_score"]),
        rule_barrier_level=barrier_level,
        threat_result=json.dumps(assessment["threat_result"], ensure_ascii=False),
        barrier_result=json.dumps(assessment["barrier_result"], ensure_ascii=False),
        context_profile=context_profile,
        memory_text=memory_text,
    )
    reflection_model = args.reflection_model or args.model
    obj, raw, usage, request_id = await structured_response_call(
        client,
        semaphore,
        model=reflection_model,
        schema_name="training_error_reflection",
        schema=REFLECTION_SCHEMA,
        system_prompt=REFLECTION_SYSTEM,
        user_prompt=user_prompt,
        max_output_tokens=args.reflection_max_tokens,
        temperature=args.temperature,
    )
    return validate_reflection(obj, pattern_id), raw, usage, request_id, user_prompt


async def process_train_decision_sequential(
    *,
    idx: int,
    row: pd.Series,
    assessment: Dict[str, Any],
    args: argparse.Namespace,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    train_pattern_rates: Dict[int, float],
    memory: ReflectionSimilarityMemory,
    feature_matrix: np.ndarray,
) -> Dict[str, Any]:
    start = time.time()
    usage_total = dict(assessment.get("usage_total", {}))
    pattern_id = int(row["hbm2_pattern"])
    memory_text, memory_ids, memory_sims = memory.get(
        feature_matrix[idx], pattern_id
    )
    try:
        decision_result, decision_raw, decision_usage, decision_request_id, context_profile = await call_decision_v3(
            row=row,
            assessment=assessment,
            args=args,
            client=client,
            semaphore=semaphore,
            train_pattern_rates=train_pattern_rates,
            memory_text=memory_text,
        )
        add_usage(usage_total, decision_usage)
        actual = int(row["vaccinated"])
        predicted_50 = int(decision_result["probability_yes"] >= 50)
        is_correct_50 = bool(predicted_50 == actual)
        reflection: Optional[Dict[str, Any]] = None
        reflection_raw = ""
        reflection_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        reflection_request_id = ""
        reflection_user = ""
        if not is_correct_50 and args.memory_k > 0:
            (
                reflection,
                reflection_raw,
                reflection_usage,
                reflection_request_id,
                reflection_user,
            ) = await call_reflection(
                row=row,
                assessment=assessment,
                decision_result=decision_result,
                memory_text=memory_text,
                context_profile=context_profile,
                args=args,
                client=client,
                semaphore=semaphore,
            )
            add_usage(usage_total, reflection_usage)

        threat_level, barrier_level = rule_levels(row)
        entry = {
            "status": "ok",
            "experiment": EXPERIMENT,
            "version": VERSION,
            "timestamp": utc_now(),
            "data_idx": idx,
            "source_row_id": int(row["_source_row_id"]),
            "phase": "train",
            "model": args.model,
            "actual": actual,
            "actual_label": "YES" if actual == 1 else "NO",
            "probability_yes": int(decision_result["probability_yes"]),
            "decision_at_50": "YES" if predicted_50 == 1 else "NO",
            "predicted_at_50": predicted_50,
            "is_correct_at_50": is_correct_50,
            "decision_reason": decision_result["reason"],
            "decision_consistency_corrected": decision_result[
                "decision_consistency_corrected"
            ],
            "hbm2_pattern": pattern_id,
            "pattern_label": PATTERN_LABELS[pattern_id],
            "rule_threat_score": int(row["hbm_threat_score"]),
            "rule_threat_level": threat_level,
            "rule_barrier_score": int(row["hbm_barrier_score"]),
            "rule_barrier_level": barrier_level,
            "threat_result": assessment["threat_result"],
            "barrier_result": assessment["barrier_result"],
            "context_profile": context_profile,
            "context_summary": compact_context_summary(row),
            "memory_used": bool(memory_ids),
            "memory_source_indices": memory_ids,
            "memory_similarities": memory_sims,
            "memory_size_before": memory.size(),
            "memory_frozen_in_test": True,
            "reflection": reflection,
            "prompts": {
                "decision_system": DECISION_SYSTEM_V3,
                "decision_user": make_decision_prompt_v3(
                    row=row,
                    assessment=assessment,
                    args=args,
                    train_pattern_rates=train_pattern_rates,
                    memory_text=memory_text,
                )[0],
                "reflection_system": REFLECTION_SYSTEM if reflection else "",
                "reflection_user": reflection_user,
            },
            "raw_outputs": {
                "decision": decision_raw,
                "reflection": reflection_raw,
            },
            "request_ids": {
                "decision": decision_request_id,
                "reflection": reflection_request_id,
            },
            "usage": {
                "assessment": assessment.get("usage_total", {}),
                "decision": decision_usage,
                "reflection": reflection_usage,
            },
            "usage_total": usage_total,
            "sample_time_sec": round(time.time() - start, 3),
        }
        return entry
    except Exception as exc:
        return {
            "status": "error",
            "experiment": EXPERIMENT,
            "version": VERSION,
            "timestamp": utc_now(),
            "data_idx": idx,
            "source_row_id": int(row["_source_row_id"]),
            "phase": "train",
            "actual": int(row["vaccinated"]),
            "hbm2_pattern": pattern_id,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(limit=10),
            "usage_total": usage_total,
            "sample_time_sec": round(time.time() - start, 3),
        }


async def process_test_decision(
    *,
    idx: int,
    row: pd.Series,
    assessment: Dict[str, Any],
    args: argparse.Namespace,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    train_pattern_rates: Dict[int, float],
    frozen_memory: ReflectionSimilarityMemory,
    feature_matrix: np.ndarray,
    calibrated_threshold: int,
) -> Dict[str, Any]:
    start = time.time()
    usage_total = dict(assessment.get("usage_total", {}))
    pattern_id = int(row["hbm2_pattern"])
    memory_text, memory_ids, memory_sims = frozen_memory.get(
        feature_matrix[idx], pattern_id
    )
    try:
        decision_result, decision_raw, decision_usage, decision_request_id, context_profile = await call_decision_v3(
            row=row,
            assessment=assessment,
            args=args,
            client=client,
            semaphore=semaphore,
            train_pattern_rates=train_pattern_rates,
            memory_text=memory_text,
        )
        add_usage(usage_total, decision_usage)
        probability = int(decision_result["probability_yes"])
        predicted_50 = int(probability >= 50)
        predicted_calibrated = int(probability >= calibrated_threshold)
        actual = int(row["vaccinated"])
        threat_level, barrier_level = rule_levels(row)
        return {
            "status": "ok",
            "experiment": EXPERIMENT,
            "version": VERSION,
            "timestamp": utc_now(),
            "data_idx": idx,
            "source_row_id": int(row["_source_row_id"]),
            "phase": "test",
            "model": args.model,
            "actual": actual,
            "actual_label": "YES" if actual == 1 else "NO",
            "probability_yes": probability,
            "decision_at_50": "YES" if predicted_50 else "NO",
            "predicted_at_50": predicted_50,
            "predicted_calibrated": predicted_calibrated,
            "decision_calibrated": "YES" if predicted_calibrated else "NO",
            "calibrated_threshold": calibrated_threshold,
            "is_correct_at_50": bool(predicted_50 == actual),
            "is_correct_calibrated": bool(predicted_calibrated == actual),
            "decision_reason": decision_result["reason"],
            "decision_consistency_corrected": decision_result[
                "decision_consistency_corrected"
            ],
            "hbm2_pattern": pattern_id,
            "pattern_label": PATTERN_LABELS[pattern_id],
            "rule_threat_score": int(row["hbm_threat_score"]),
            "rule_threat_level": threat_level,
            "rule_barrier_score": int(row["hbm_barrier_score"]),
            "rule_barrier_level": barrier_level,
            "threat_result": assessment["threat_result"],
            "barrier_result": assessment["barrier_result"],
            "context_profile": context_profile,
            "context_summary": compact_context_summary(row),
            "memory_used": bool(memory_ids),
            "memory_source_indices": memory_ids,
            "memory_similarities": memory_sims,
            "memory_size": frozen_memory.size(),
            "memory_frozen_in_test": True,
            "reflection": None,
            "raw_outputs": {"decision": decision_raw},
            "request_ids": {"decision": decision_request_id},
            "usage": {
                "assessment": assessment.get("usage_total", {}),
                "decision": decision_usage,
            },
            "usage_total": usage_total,
            "sample_time_sec": round(time.time() - start, 3),
        }
    except Exception as exc:
        return {
            "status": "error",
            "experiment": EXPERIMENT,
            "version": VERSION,
            "timestamp": utc_now(),
            "data_idx": idx,
            "source_row_id": int(row["_source_row_id"]),
            "phase": "test",
            "actual": int(row["vaccinated"]),
            "hbm2_pattern": pattern_id,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(limit=10),
            "usage_total": usage_total,
            "sample_time_sec": round(time.time() - start, 3),
        }


# ---------------------------------------------------------------------------
# Threshold calibration and probability metrics
# ---------------------------------------------------------------------------
def binary_metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": safe_divide(tn, tn + fp),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "predicted_yes_rate": float(y_pred.mean()),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
    }


def calibrate_threshold(
    train_entries: Sequence[Dict[str, Any]], metric: str
) -> Tuple[int, pd.DataFrame, Dict[str, Any]]:
    rows = [e for e in train_entries if e.get("status") == "ok"]
    if not rows:
        raise ValueError("No completed training decisions are available for calibration.")
    y_true = np.array([int(e["actual"]) for e in rows], dtype=int)
    probabilities = np.array([int(e["probability_yes"]) for e in rows], dtype=float)
    records: List[Dict[str, Any]] = []
    for threshold in range(0, 101):
        y_pred = (probabilities >= threshold).astype(int)
        metrics = binary_metrics_from_arrays(y_true, y_pred)
        records.append({"threshold": threshold, **metrics})
    table = pd.DataFrame(records)
    if metric not in {"balanced_accuracy", "f1", "accuracy"}:
        raise ValueError(f"Unsupported threshold metric: {metric}")
    best_value = float(table[metric].max())
    candidates = table[np.isclose(table[metric], best_value)].copy()
    candidates["distance_from_50"] = (candidates["threshold"] - 50).abs()
    # Tie-break toward the conventional 50 threshold, then the smaller threshold
    # to avoid preserving an unexplained NO bias.
    chosen = candidates.sort_values(
        ["distance_from_50", "threshold"], ascending=[True, True]
    ).iloc[0]
    threshold = int(chosen["threshold"])
    calibration = {
        "threshold_metric": metric,
        "selected_threshold": threshold,
        "selected_metric_value": float(chosen[metric]),
        "train_metrics_at_50": binary_metrics_from_arrays(
            y_true, (probabilities >= 50).astype(int)
        ),
        "train_metrics_calibrated": binary_metrics_from_arrays(
            y_true, (probabilities >= threshold).astype(int)
        ),
    }
    return threshold, table, calibration


def probability_metrics(
    entries: Sequence[Dict[str, Any]], phase: str, threshold: int
) -> Dict[str, Any]:
    rows = [
        e
        for e in entries
        if e.get("status") == "ok" and e.get("phase") == phase
    ]
    if not rows:
        return {"n_valid": 0}
    y_true = np.array([int(e["actual"]) for e in rows], dtype=int)
    probabilities = np.array([float(e["probability_yes"]) / 100.0 for e in rows])
    y_pred = (probabilities * 100 >= threshold).astype(int)
    result = {
        "n_valid": len(rows),
        "actual_yes_rate": float(y_true.mean()),
        "threshold": int(threshold),
        **binary_metrics_from_arrays(y_true, y_pred),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, np.clip(probabilities, 1e-6, 1 - 1e-6), labels=[0, 1])),
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        result["average_precision"] = float(
            average_precision_score(y_true, probabilities)
        )
    else:
        result["roc_auc"] = None
        result["average_precision"] = None
    return result


def pattern_metrics_v3(
    entries: Sequence[Dict[str, Any]], phase: str, threshold: int
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    phase_entries = [
        e
        for e in entries
        if e.get("status") == "ok" and e.get("phase") == phase
    ]
    for pattern_id, label in PATTERN_LABELS.items():
        subset = [e for e in phase_entries if int(e["hbm2_pattern"]) == pattern_id]
        if not subset:
            continue
        y_true = np.array([int(e["actual"]) for e in subset], dtype=int)
        probs = np.array([int(e["probability_yes"]) for e in subset], dtype=float)
        y_pred = (probs >= threshold).astype(int)
        rows.append(
            {
                "phase": phase,
                "pattern_id": pattern_id,
                "pattern_label": label,
                "threshold": threshold,
                "n": len(subset),
                "actual_vaccination_rate": float(y_true.mean()),
                "mean_probability_yes": float(probs.mean() / 100.0),
                "predicted_yes_rate": float(y_pred.mean()),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def predictions_dataframe_v3(
    entries: Sequence[Dict[str, Any]], calibrated_threshold: int
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for entry in sorted(entries, key=lambda x: int(x["data_idx"])):
        if entry.get("status") != "ok":
            continue
        probability = int(entry["probability_yes"])
        pred_calibrated = int(probability >= calibrated_threshold)
        rows.append(
            {
                "data_idx": entry["data_idx"],
                "source_row_id": entry["source_row_id"],
                "phase": entry["phase"],
                "actual": entry["actual"],
                "probability_yes": probability,
                "predicted_at_50": int(probability >= 50),
                "predicted_calibrated": pred_calibrated,
                "calibrated_threshold": calibrated_threshold,
                "is_correct_at_50": int(int(probability >= 50) == int(entry["actual"])),
                "is_correct_calibrated": int(pred_calibrated == int(entry["actual"])),
                "decision_reason": entry["decision_reason"],
                "hbm2_pattern": entry["hbm2_pattern"],
                "pattern_label": entry["pattern_label"],
                "rule_threat_score": entry["rule_threat_score"],
                "rule_threat_level": entry["rule_threat_level"],
                "rule_barrier_score": entry["rule_barrier_score"],
                "rule_barrier_level": entry["rule_barrier_level"],
                "llm_threat_score": entry["threat_result"]["score"],
                "llm_threat_reason": entry["threat_result"]["reason"],
                "llm_barrier_score": entry["barrier_result"]["score"],
                "llm_barrier_reason": entry["barrier_result"]["reason"],
                "memory_used": entry["memory_used"],
                "memory_source_indices": json.dumps(entry["memory_source_indices"]),
                "reflection_created": bool(entry.get("reflection")),
                "input_tokens": entry.get("usage_total", {}).get("input_tokens", 0),
                "output_tokens": entry.get("usage_total", {}).get("output_tokens", 0),
                "total_tokens": entry.get("usage_total", {}).get("total_tokens", 0),
                "sample_time_sec": entry.get("sample_time_sec", 0),
            }
        )
    return pd.DataFrame(rows)


def save_v3_outputs(
    *,
    output_dir: Path,
    assessment_latest: Dict[int, Dict[str, Any]],
    decision_latest: Dict[int, Dict[str, Any]],
    run_config: Dict[str, Any],
    calibrated_threshold: int,
    calibration: Dict[str, Any],
) -> Dict[str, Any]:
    decisions = [
        decision_latest[idx]
        for idx in sorted(decision_latest)
        if decision_latest[idx].get("status") == "ok"
    ]
    predictions_dataframe_v3(decisions, calibrated_threshold).to_csv(
        output_dir / "predictions.csv", index=False
    )
    pattern_frames = [
        pattern_metrics_v3(decisions, "train", calibrated_threshold),
        pattern_metrics_v3(decisions, "test", calibrated_threshold),
    ]
    nonempty = [frame for frame in pattern_frames if not frame.empty]
    pattern_table = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()
    pattern_table.to_csv(output_dir / "pattern_metrics.csv", index=False)

    usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for entry in assessment_latest.values():
        if entry.get("status") == "ok":
            add_usage(usage_total, entry.get("usage_total", {}))
    # Decision logs already include assessment usage; subtract duplicate by adding
    # only decision/reflection components from each decision entry.
    for entry in decisions:
        usage = entry.get("usage", {})
        add_usage(usage_total, usage.get("decision", {}))
        add_usage(usage_total, usage.get("reflection", {}))

    summary = {
        "experiment": EXPERIMENT,
        "version": VERSION,
        "created_at": utc_now(),
        "run_config": run_config,
        "calibration": calibration,
        "train_metrics_at_50": probability_metrics(decisions, "train", 50),
        "train_metrics_calibrated": probability_metrics(
            decisions, "train", calibrated_threshold
        ),
        "test_metrics_at_50": probability_metrics(decisions, "test", 50),
        "test_metrics_calibrated": probability_metrics(
            decisions, "test", calibrated_threshold
        ),
        "n_assessments_completed": sum(
            1 for e in assessment_latest.values() if e.get("status") == "ok"
        ),
        "n_decisions_completed": len(decisions),
        "n_reflections": sum(1 for e in decisions if e.get("reflection")),
        "usage_total": usage_total,
        "files": {
            "assessment_log": str(output_dir / "logs" / "assessment_log.jsonl"),
            "sample_log": str(output_dir / "logs" / "sample_log.jsonl"),
            "reflection_memory": str(output_dir / "logs" / "reflection_memory.jsonl"),
            "predictions": str(output_dir / "predictions.csv"),
            "pattern_metrics": str(output_dir / "pattern_metrics.csv"),
            "threshold_search": str(output_dir / "threshold_search.csv"),
            "threshold_calibration": str(output_dir / "threshold_calibration.json"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


# ---------------------------------------------------------------------------
# V3 CLI
# ---------------------------------------------------------------------------
def build_parser_v3() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run FLARE-VAX with concurrent CALL1/2 precomputation, sequential "
            "training CALL3 + reflection memory, frozen-memory concurrent test CALL3, "
            "and training-only probability calibration."
        )
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/content/drive/MyDrive/Vaccination-Decision-Model/Data/nhis2024_hbm2_clean.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/Vaccination-Decision-Model/Results/gpt4o_reflection_calibrated",
    )
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument(
        "--reflection_model",
        type=str,
        default="",
        help="Optional separate model for reflection; defaults to --model.",
    )
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--balanced_sample",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--joint_stratify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stratify train/test by vaccinated × HBM2 pattern when feasible.",
    )
    parser.add_argument("--memory_k", type=int, default=3)
    parser.add_argument("--memory_min_similarity", type=float, default=-1.0)
    parser.add_argument(
        "--pattern_prior_mode",
        choices=["theory", "train_rate", "none"],
        default="theory",
    )
    parser.add_argument(
        "--include_sensitive_context",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--assessment_max_tokens", type=int, default=140)
    parser.add_argument("--decision_max_tokens", type=int, default=180)
    parser.add_argument("--reflection_max_tokens", type=int, default=240)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument(
        "--max_concurrent_requests",
        type=int,
        default=16,
        help="Global API request concurrency cap.",
    )
    parser.add_argument(
        "--assessment_batch_size",
        type=int,
        default=32,
        help="Respondents per CALL1/2 precomputation batch.",
    )
    parser.add_argument(
        "--test_concurrent_samples",
        type=int,
        default=8,
        help="Concurrent test CALL3 respondents after memory is frozen.",
    )
    parser.add_argument(
        "--threshold_metric",
        choices=["balanced_accuracy", "f1", "accuracy"],
        default="balanced_accuracy",
    )
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--print_samples", type=int, default=3)
    return parser


def print_entry_brief(entry: Dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print(
        f"SAMPLE {entry['data_idx']} | phase={entry['phase']} | actual={entry['actual_label']} | "
        f"p(YES)={entry['probability_yes']}% | at50={entry['decision_at_50']}"
    )
    print(f"Pattern: P{entry['hbm2_pattern']} {entry['pattern_label']}")
    print(f"Reason: {entry['decision_reason']}")
    print(f"Memory used: {entry['memory_source_indices']}")
    if entry.get("reflection"):
        print("Reflection: " + json.dumps(entry["reflection"], ensure_ascii=False))
    print("=" * 78)


# ---------------------------------------------------------------------------
# Main V3 experiment
# ---------------------------------------------------------------------------
async def async_main_v3() -> None:
    args = build_parser_v3().parse_args()
    if args.max_concurrent_requests < 1:
        raise ValueError("--max_concurrent_requests must be at least 1")
    if args.assessment_batch_size < 1:
        raise ValueError("--assessment_batch_size must be at least 1")
    if args.test_concurrent_samples < 1:
        raise ValueError("--test_concurrent_samples must be at least 1")

    data_path = Path(args.data_path).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    assessment_log = logs_dir / "assessment_log.jsonl"
    sample_log = logs_dir / "sample_log.jsonl"
    reflection_log = logs_dir / "reflection_memory.jsonl"
    config_path = output_dir / "run_config.json"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print("=" * 78)
    print("FLARE-VAX HBM2 — Reflection Memory + Probability Calibration")
    print("=" * 78)
    print(f"Version                  : {VERSION}")
    print(f"Model                    : {args.model}")
    print(f"Reflection model         : {args.reflection_model or args.model}")
    print(f"Data                     : {data_path}")
    print(f"Output                   : {output_dir}")
    print(f"Sample size              : {args.sample_size}")
    print(f"Train ratio              : {args.train_ratio}")
    print(f"Balanced sample          : {args.balanced_sample}")
    print(f"Joint stratify           : {args.joint_stratify}")
    print(f"Memory k                 : {args.memory_k}")
    print(f"Pattern prior            : {args.pattern_prior_mode}")
    print(f"Assessment batch size    : {args.assessment_batch_size}")
    print(f"Test concurrent samples  : {args.test_concurrent_samples}")
    print(f"Max concurrent requests  : {args.max_concurrent_requests}")
    print(f"Threshold metric         : {args.threshold_metric}")

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

    sampled = (
        balanced_sample(df, args.sample_size, args.random_seed)
        if args.balanced_sample
        else ordinary_sample(df, args.sample_size, args.random_seed)
    )
    train_df, test_df = split_train_test_v3(
        sampled, args.train_ratio, args.random_seed, args.joint_stratify
    )
    train_df["_phase"] = "train"
    test_df["_phase"] = "test"
    ordered = pd.concat([train_df, test_df], ignore_index=True)
    train_size = len(train_df)
    total_size = len(ordered)
    feature_matrix, memory_feature_names = prepare_memory_matrix_v3(
        ordered, train_size
    )
    train_pattern_rates = calculate_train_pattern_rates(train_df)

    print(f"Rows available           : {len(df):,}")
    print(f"Selected rows            : {total_size} (train={train_size}, test={len(test_df)})")
    print(f"Selected target rate     : {ordered['vaccinated'].mean():.1%}")
    print(f"Memory vector dimensions : {len(memory_feature_names)}")
    print("Pattern distribution:")
    for pattern_id, group in ordered.groupby("hbm2_pattern"):
        print(
            f"  P{int(pattern_id)} {PATTERN_LABELS[int(pattern_id)]:<29s} "
            f"n={len(group):4d} actual_vax={group['vaccinated'].mean():.1%}"
        )

    run_config: Dict[str, Any] = {
        "version": VERSION,
        "model": args.model,
        "reflection_model": args.reflection_model or args.model,
        "data_path": str(data_path.resolve()),
        "data_size_bytes": data_path.stat().st_size,
        "sample_size": args.sample_size,
        "balanced_sample": args.balanced_sample,
        "joint_stratify": args.joint_stratify,
        "train_ratio": args.train_ratio,
        "random_seed": args.random_seed,
        "memory_k": args.memory_k,
        "memory_min_similarity": args.memory_min_similarity,
        "memory_features": memory_feature_names,
        "pattern_prior_mode": args.pattern_prior_mode,
        "include_sensitive_context": args.include_sensitive_context,
        "assessment_max_tokens": args.assessment_max_tokens,
        "decision_max_tokens": args.decision_max_tokens,
        "reflection_max_tokens": args.reflection_max_tokens,
        "temperature": args.temperature,
        "assessment_batch_size": args.assessment_batch_size,
        "test_concurrent_samples": args.test_concurrent_samples,
        "max_concurrent_requests": args.max_concurrent_requests,
        "threshold_metric": args.threshold_metric,
        "pipeline": {
            "train_call12": "concurrent precomputation",
            "train_call3": "strictly sequential",
            "train_reflection": "immediate after each wrong 50-percent decision",
            "test_call12": "concurrent precomputation after train",
            "test_call3": "concurrent with frozen training memory",
            "test_memory_update": False,
        },
    }
    run_config["fingerprint"] = config_fingerprint(run_config)

    if config_path.exists() and not args.overwrite:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != run_config["fingerprint"]:
            raise RuntimeError(
                "Output directory contains a different run configuration. "
                "Use a new --output_dir or --overwrite."
            )
    else:
        config_path.write_text(
            json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if args.dry_run:
        row = ordered.iloc[0]
        threat_level, barrier_level = rule_levels(row)
        print("\n--- DRY RUN CALL 1 ---")
        print(
            THREAT_USER_TEMPLATE_V3.format(
                rule_score=int(row["hbm_threat_score"]),
                rule_level=threat_level,
                profile=build_threat_profile(row),
            )
        )
        print("\n--- DRY RUN CALL 2 ---")
        print(
            BARRIER_USER_TEMPLATE_V3.format(
                rule_score=int(row["hbm_barrier_score"]),
                rule_level=barrier_level,
                profile=build_barrier_profile(row),
            )
        )
        print("\nDry run completed; no API calls were made.")
        return

    if AsyncOpenAI is None:
        raise RuntimeError(
            "The OpenAI Python SDK is not installed. Run: pip install -U openai"
        )
    api_key = resolve_api_key(args.api_key)
    client = AsyncOpenAI(
        api_key=api_key,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    semaphore = asyncio.Semaphore(args.max_concurrent_requests)

    assessment_latest = load_latest_log_entries(assessment_log)
    decision_latest = load_latest_log_entries(sample_log)

    # ------------------------------------------------------------------
    # Phase 1: all TRAIN CALL 1/2 concurrently
    # ------------------------------------------------------------------
    await precompute_assessments_for_indices(
        indices=range(0, train_size),
        ordered=ordered,
        args=args,
        client=client,
        semaphore=semaphore,
        assessment_log=assessment_log,
        assessment_latest=assessment_latest,
        phase_name="TRAIN",
    )

    # ------------------------------------------------------------------
    # Phase 2: TRAIN CALL 3 strictly sequential; each error is reflected
    #          and added to memory before the next respondent.
    # ------------------------------------------------------------------
    memory = ReflectionSimilarityMemory(
        k=args.memory_k, min_similarity=args.memory_min_similarity
    )
    # Rebuild only completed reflected training errors in original order.
    for idx in range(train_size):
        entry = decision_latest.get(idx)
        if not entry or entry.get("status") != "ok":
            break
        if not bool(entry.get("is_correct_at_50")) and entry.get("reflection"):
            memory.store(
                idx,
                int(entry["hbm2_pattern"]),
                feature_matrix[idx],
                build_reflection_memory_text(entry),
                entry["reflection"],
            )

    first_train_missing = next(
        (
            idx
            for idx in range(train_size)
            if decision_latest.get(idx, {}).get("status") != "ok"
        ),
        train_size,
    )
    print(f"Sequential TRAIN CALL3 resumes at {first_train_missing}/{train_size}; memory={memory.size()}")
    newly_printed = 0
    for idx in tqdm(
        range(first_train_missing, train_size), desc="TRAIN CALL3 + reflection"
    ):
        assessment = assessment_latest.get(idx)
        if not assessment or assessment.get("status") != "ok":
            raise RuntimeError(f"Missing successful assessment for train idx={idx}")
        entry = await process_train_decision_sequential(
            idx=idx,
            row=ordered.iloc[idx],
            assessment=assessment,
            args=args,
            client=client,
            semaphore=semaphore,
            train_pattern_rates=train_pattern_rates,
            memory=memory,
            feature_matrix=feature_matrix,
        )
        append_jsonl(sample_log, entry)
        decision_latest[idx] = entry
        if entry.get("status") != "ok":
            if not args.continue_on_error:
                await client.close()
                raise RuntimeError(
                    f"Train decision failed at idx={idx}: {entry.get('error_message')}"
                )
            continue
        if entry.get("reflection"):
            memory.store(
                idx,
                int(entry["hbm2_pattern"]),
                feature_matrix[idx],
                build_reflection_memory_text(entry),
                entry["reflection"],
            )
            append_jsonl(
                reflection_log,
                {
                    "data_idx": idx,
                    "pattern_id": entry["hbm2_pattern"],
                    "pattern_label": entry["pattern_label"],
                    "probability_yes": entry["probability_yes"],
                    "decision_at_50": entry["decision_at_50"],
                    "actual_label": entry["actual_label"],
                    "reflection": entry["reflection"],
                    "memory_text": build_reflection_memory_text(entry),
                },
            )
        if newly_printed < args.print_samples:
            print_entry_brief(entry)
            newly_printed += 1
        if args.checkpoint_every > 0 and (idx + 1) % args.checkpoint_every == 0:
            print(
                f"TRAIN checkpoint: {idx+1}/{train_size}; reflected memory={memory.size()}"
            )

    # ------------------------------------------------------------------
    # Calibrate the probability cutoff on TRAIN only.
    # ------------------------------------------------------------------
    train_entries = [
        decision_latest[idx]
        for idx in range(train_size)
        if decision_latest.get(idx, {}).get("status") == "ok"
    ]
    calibrated_threshold, threshold_table, calibration = calibrate_threshold(
        train_entries, args.threshold_metric
    )
    threshold_table.to_csv(output_dir / "threshold_search.csv", index=False)
    (output_dir / "threshold_calibration.json").write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nTRAIN-ONLY THRESHOLD CALIBRATION")
    print(json.dumps(calibration, indent=2, ensure_ascii=False))

    # ------------------------------------------------------------------
    # Phase 3: all TEST CALL 1/2 concurrently after training is complete.
    # ------------------------------------------------------------------
    await precompute_assessments_for_indices(
        indices=range(train_size, total_size),
        ordered=ordered,
        args=args,
        client=client,
        semaphore=semaphore,
        assessment_log=assessment_log,
        assessment_latest=assessment_latest,
        phase_name="TEST",
    )

    # ------------------------------------------------------------------
    # Phase 4: freeze memory and run all TEST CALL 3 concurrently.
    # ------------------------------------------------------------------
    frozen_memory_size = memory.size()
    print(
        f"Frozen training reflection memory: {frozen_memory_size} items. "
        "TEST labels will never update memory."
    )
    test_indices = [
        idx
        for idx in range(train_size, total_size)
        if decision_latest.get(idx, {}).get("status") != "ok"
    ]
    batch_size = max(1, args.test_concurrent_samples)
    progress = tqdm(total=len(test_indices), desc="TEST CALL3 frozen memory")
    for start in range(0, len(test_indices), batch_size):
        batch = test_indices[start : start + batch_size]
        results = await asyncio.gather(
            *[
                process_test_decision(
                    idx=idx,
                    row=ordered.iloc[idx],
                    assessment=assessment_latest[idx],
                    args=args,
                    client=client,
                    semaphore=semaphore,
                    train_pattern_rates=train_pattern_rates,
                    frozen_memory=memory,
                    feature_matrix=feature_matrix,
                    calibrated_threshold=calibrated_threshold,
                )
                for idx in batch
            ]
        )
        for entry in sorted(results, key=lambda x: int(x["data_idx"])):
            append_jsonl(sample_log, entry)
            decision_latest[int(entry["data_idx"])] = entry
            progress.update(1)
            if entry.get("status") != "ok" and not args.continue_on_error:
                progress.close()
                await client.close()
                raise RuntimeError(
                    f"Test decision failed at idx={entry['data_idx']}: "
                    f"{entry.get('error_message')}"
                )
    progress.close()

    summary = save_v3_outputs(
        output_dir=output_dir,
        assessment_latest=assessment_latest,
        decision_latest=decision_latest,
        run_config=run_config,
        calibrated_threshold=calibrated_threshold,
        calibration=calibration,
    )
    await client.close()

    print("\n" + "=" * 78)
    print("RUN SUMMARY")
    print("=" * 78)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nAssessment log : {assessment_log}")
    print(f"Sample log     : {sample_log}")
    print(f"Reflection log : {reflection_log}")
    print(f"Predictions    : {output_dir / 'predictions.csv'}")
    print(f"Summary        : {output_dir / 'summary.json'}")


def main() -> None:
    asyncio.run(async_main_v3())


if __name__ == "__main__":
    main()
