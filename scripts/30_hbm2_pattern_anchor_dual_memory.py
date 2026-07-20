#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FLARE-VAX HBM2 V4: pattern anchor, balanced dual memory, and independent calibration.

Designed for Google Colab + the official OpenAI Responses API.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import shutil
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
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
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm.auto import tqdm

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore



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


def config_fingerprint(config: Dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def resolve_api_key(explicit_key: str) -> str:
    if explicit_key:
        return explicit_key.strip()
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    raise ValueError(
        "No API key found. Set OPENAI_API_KEY (recommended) or pass --api_key."
    )


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


# ===========================================================================
# V4: Pattern-anchored adjustment + dual memory + independent calibration
# ===========================================================================
# This version intentionally keeps the data/profile/OpenAI helper functions
# above, but replaces the V3 experiment entry point with a stricter design:
#   1) class-aware small-sample selection;
#   2) memory-build / calibration / test split;
#   3) all CALL 1/2 assessments precomputed concurrently;
#   4) sequential memory-build CALL 3;
#   5) pattern base rate as the probability anchor;
#   6) CALL 3 outputs only a bounded adjustment, not a free probability;
#   7) balanced prototype + high-confidence reflection memory;
#   8) calibration and test both use the same frozen final memory.

VERSION = "hbm2_openai_colab_v4_pattern_anchor_dual_memory"
EXPERIMENT = "flare_vax_hbm2_pattern_anchor_dual_memory"

ADJUSTMENT_SCHEMA_V4: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "adjustment": {"type": "integer", "minimum": -20, "maximum": 20},
        "reason": {"type": "string"},
    },
    "required": ["adjustment", "reason"],
    "additionalProperties": False,
}

REFLECTION_SCHEMA_V4: Dict[str, Any] = {
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

DECISION_SYSTEM_V4 = """You predict whether an NHIS respondent received a flu vaccine in the past 12 months.
The program supplies an authoritative HBM2 pattern and a training-only pattern base rate. The base rate is the probability anchor.
CALL 1 and CALL 2 only explain and refine the fixed rule-based pattern. Do not redefine the pattern.
Your task is not to invent a new absolute probability. Output only a bounded integer adjustment from -20 to +20 percentage points and one concise reason.
Prototype cases represent typical correctly predicted training cases. Reflections represent exceptional high-confidence training errors and must not be treated as the majority rule.
Do not infer facts that are missing. Return only the required structured object. The reason must be one sentence of at most 30 words."""

DECISION_USER_TEMPLATE_V4 = """TASK
Adjust the training-only HBM2 pattern base rate for this respondent.

AUTHORITATIVE HBM2 CLASSIFICATION
Pattern ID: P{pattern_id}
Pattern: {pattern_label}
Rule threat score: {rule_threat_score}/4 ({rule_threat_level})
Rule barrier score: {rule_barrier_score}/3 ({rule_barrier_level})
Theoretical tendency: {pattern_tendency}

TRAINING-ONLY PROBABILITY ANCHOR
Pattern base rate: {base_rate:.1f}%
Memory-build observations supporting this anchor: n={base_n}
This base rate is authoritative as the starting probability. Do not replace it with a free probability estimate.

CALL 1 — THREAT EXPLANATION/NUANCE
{threat_result}

CALL 2 — BARRIER EXPLANATION/NUANCE
{barrier_result}

ADDITIONAL CONTEXT OUTSIDE THE TWO CONSTRUCT SCORES
{context_profile}

BALANCED MEMORY EVIDENCE
{memory_text}

OUTPUT RULES
1. Return adjustment as an integer from -20 to +20 percentage points.
2. Positive values increase vaccination likelihood; negative values decrease it.
3. Use 0 when the individual context does not provide clear evidence beyond the pattern.
4. Keep the pattern base rate as the dominant anchor.
5. Treat prototypes as typical evidence and reflections only as exceptions for genuinely similar cases.
6. Give exactly one concise reason and output no final YES/NO label."""

REFLECTION_SYSTEM_V4 = """You are the self-reflection component of a FLARE-style behavioral prediction system.
The following memory-build case was a high-confidence prediction error. Diagnose a transferable error using only observed variables.
Do not overturn the HBM2 pattern base rate as a general rule. The correction must apply only to genuinely similar exceptional cases.
Return only the required structured object. Each text field must be one concise sentence of at most 35 words."""

REFLECTION_USER_TEMPLATE_V4 = """HIGH-CONFIDENCE MEMORY-BUILD ERROR
Actual vaccination outcome: {actual_label}
Pattern base rate: {base_rate:.1f}%
CALL 3 adjustment: {adjustment:+d} percentage points
Final predicted probability: {final_probability:.1f}%
Prediction at 50%: {predicted_label}
Decision reason: {decision_reason}

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

MEMORY USED BEFORE THIS ERROR
{memory_text}

REFLECTION TASK
Identify why this high-confidence prediction failed, the observed signal that was missed or overweighted, and a narrow correction rule for genuinely similar cases only."""


# ---------------------------------------------------------------------------
# Class-aware sampling and three-way split
# ---------------------------------------------------------------------------
def _allocate_integer_counts(
    total: int,
    weights: Dict[int, float],
    capacities: Dict[int, int],
) -> Dict[int, int]:
    """Allocate an exact total across groups using largest remainders."""
    keys = list(weights)
    if total < 0:
        raise ValueError("total must be nonnegative")
    if total > sum(capacities.get(k, 0) for k in keys):
        raise ValueError("Requested sample exceeds available capacity")
    weight_sum = sum(max(0.0, float(weights[k])) for k in keys)
    if weight_sum <= 0:
        weights = {k: 1.0 for k in keys}
        weight_sum = float(len(keys))
    exact = {k: total * max(0.0, float(weights[k])) / weight_sum for k in keys}
    out = {k: min(int(math.floor(exact[k])), capacities.get(k, 0)) for k in keys}
    remaining = total - sum(out.values())
    order = sorted(
        keys,
        key=lambda k: (exact[k] - math.floor(exact[k]), capacities.get(k, 0)),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for k in order:
            if out[k] < capacities.get(k, 0):
                out[k] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("Could not complete integer allocation")
    return out


def _sample_one_class_by_pattern(
    class_df: pd.DataFrame,
    n: int,
    seed: int,
    preserve_pattern: bool,
) -> pd.DataFrame:
    if n >= len(class_df):
        return class_df.sample(frac=1.0, random_state=seed)
    if not preserve_pattern:
        return class_df.sample(n=n, random_state=seed)
    counts = class_df["hbm2_pattern"].value_counts().sort_index()
    weights = {int(k): float(v) for k, v in counts.items()}
    capacities = {int(k): int(v) for k, v in counts.items()}
    allocation = _allocate_integer_counts(n, weights, capacities)
    parts: List[pd.DataFrame] = []
    for j, (pattern_id, take) in enumerate(sorted(allocation.items())):
        if take <= 0:
            continue
        group = class_df[class_df["hbm2_pattern"] == pattern_id]
        parts.append(group.sample(n=take, random_state=seed + 101 * (j + 1)))
    return pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed + 999)


def class_aware_sample_v4(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Randomly sample within target classes, optionally preserving pattern mix.

    Modes:
      proportional: preserve the full-data vaccinated rate;
      balanced:     50% vaccinated / 50% unvaccinated;
      custom:       use --positive_fraction.

    --class_fraction samples the same fraction independently from each target
    class and overrides --sample_size. This is useful for inexpensive pilots.
    """
    counts = df["vaccinated"].value_counts().to_dict()
    if not {0, 1}.issubset(set(int(k) for k in counts)):
        raise ValueError("vaccinated must contain both classes 0 and 1")

    if args.class_fraction > 0:
        if not 0 < args.class_fraction <= 1:
            raise ValueError("--class_fraction must be in (0, 1]")
        target_counts = {
            y: max(1, min(int(counts[y]), int(round(counts[y] * args.class_fraction))))
            for y in (0, 1)
        }
    else:
        total = len(df) if args.sample_size <= 0 else min(int(args.sample_size), len(df))
        if total < 12:
            raise ValueError("Use at least 12 sampled rows for a three-way split")
        if args.class_sampling == "balanced":
            n1 = total // 2
        elif args.class_sampling == "custom":
            if not 0.01 <= args.positive_fraction <= 0.99:
                raise ValueError("--positive_fraction must be between 0.01 and 0.99")
            n1 = int(round(total * args.positive_fraction))
        else:
            n1 = int(round(total * counts[1] / (counts[0] + counts[1])))
        n1 = max(1, min(n1, int(counts[1])))
        n0 = total - n1
        if n0 < 1:
            n0, n1 = 1, total - 1
        if n0 > counts[0]:
            shortage = n0 - counts[0]
            n0 = int(counts[0])
            n1 += shortage
        if n1 > counts[1]:
            shortage = n1 - counts[1]
            n1 = int(counts[1])
            n0 += shortage
        target_counts = {0: int(n0), 1: int(n1)}

    parts: List[pd.DataFrame] = []
    for y in (0, 1):
        class_df = df[df["vaccinated"] == y]
        parts.append(
            _sample_one_class_by_pattern(
                class_df,
                target_counts[y],
                args.random_seed + 10000 * y,
                args.preserve_pattern_within_class,
            )
        )
    sampled = pd.concat(parts, axis=0)
    return sampled.sample(frac=1.0, random_state=args.random_seed + 30000).reset_index(drop=True)


def _stratify_series_v4(df: pd.DataFrame, joint: bool) -> pd.Series:
    if joint:
        return df["vaccinated"].astype(str) + "_P" + df["hbm2_pattern"].astype(str)
    return df["vaccinated"].astype(str)


def _can_stratify(series: pd.Series, n_left: int, n_right: int) -> bool:
    counts = series.value_counts()
    n_classes = len(counts)
    return bool(
        n_classes >= 2
        and int(counts.min()) >= 2
        and n_left >= n_classes
        and n_right >= n_classes
    )


def _safe_split_exact(
    df: pd.DataFrame,
    left_size: int,
    seed: int,
    joint: bool,
    label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    right_size = len(df) - left_size
    if left_size < 1 or right_size < 1:
        raise ValueError(f"Invalid {label} split sizes: {left_size}/{right_size}")
    joint_series = _stratify_series_v4(df, joint)
    target_series = _stratify_series_v4(df, False)
    stratify: Optional[pd.Series] = None
    mode = "none"
    if _can_stratify(joint_series, left_size, right_size):
        stratify = joint_series
        mode = "vaccinated×pattern" if joint else "vaccinated"
    elif _can_stratify(target_series, left_size, right_size):
        stratify = target_series
        mode = "vaccinated fallback"
    else:
        print(f"WARNING: {label} split cannot be stratified safely; using seeded random split.")
    left, right = train_test_split(
        df,
        train_size=left_size,
        random_state=seed,
        stratify=stratify,
    )
    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True)
    left.attrs["stratification"] = mode
    return left, right


def split_memory_calibration_test_v4(
    sampled: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_ratio = args.memory_ratio + args.calibration_ratio + args.test_ratio
    if not math.isclose(total_ratio, 1.0, abs_tol=1e-8):
        raise ValueError("memory_ratio + calibration_ratio + test_ratio must equal 1")
    n = len(sampled)
    n_memory = max(2, int(round(n * args.memory_ratio)))
    n_calibration = max(2, int(round(n * args.calibration_ratio)))
    n_test = n - n_memory - n_calibration
    if n_test < 2:
        n_test = 2
        n_memory = n - n_calibration - n_test
    if min(n_memory, n_calibration, n_test) < 2:
        raise ValueError("Each split must contain at least two rows")

    memory_df, remainder = _safe_split_exact(
        sampled,
        n_memory,
        args.random_seed + 1,
        args.joint_stratify,
        "memory/remainder",
    )
    calibration_df, test_df = _safe_split_exact(
        remainder,
        n_calibration,
        args.random_seed + 2,
        args.joint_stratify,
        "calibration/test",
    )
    return memory_df, calibration_df, test_df


# ---------------------------------------------------------------------------
# Pattern base rates (memory-build labels only)
# ---------------------------------------------------------------------------
def fit_pattern_base_rates_v4(
    memory_df: pd.DataFrame,
    prior_strength: float,
) -> Dict[int, Dict[str, float]]:
    overall_rate = float(memory_df["vaccinated"].mean())
    rates: Dict[int, Dict[str, float]] = {}
    for pattern_id in PATTERN_LABELS:
        group = memory_df[memory_df["hbm2_pattern"] == pattern_id]
        n = int(len(group))
        yes = int(group["vaccinated"].sum())
        raw = float(yes / n) if n else overall_rate
        smoothed = float(
            (yes + prior_strength * overall_rate) / (n + prior_strength)
            if n + prior_strength > 0
            else overall_rate
        )
        rates[pattern_id] = {
            "n": n,
            "yes": yes,
            "raw_rate": raw,
            "smoothed_rate": smoothed,
        }
    rates[-1] = {
        "n": int(len(memory_df)),
        "yes": int(memory_df["vaccinated"].sum()),
        "raw_rate": overall_rate,
        "smoothed_rate": overall_rate,
    }
    return rates


def leave_one_out_pattern_anchor_v4(
    row: pd.Series,
    full_rates: Dict[int, Dict[str, float]],
    prior_strength: float,
) -> Tuple[float, int]:
    pattern_id = int(row["hbm2_pattern"])
    actual = int(row["vaccinated"])
    pstats = full_rates[pattern_id]
    overall = full_rates[-1]
    p_n = max(0, int(pstats["n"]) - 1)
    p_yes = max(0, int(pstats["yes"]) - actual)
    overall_n = max(1, int(overall["n"]) - 1)
    overall_yes = max(0, int(overall["yes"]) - actual)
    overall_rate = overall_yes / overall_n
    rate = (p_yes + prior_strength * overall_rate) / (p_n + prior_strength)
    return float(rate * 100.0), int(p_n)


def full_pattern_anchor_v4(
    row: pd.Series,
    full_rates: Dict[int, Dict[str, float]],
) -> Tuple[float, int]:
    stats = full_rates[int(row["hbm2_pattern"])]
    return float(stats["smoothed_rate"] * 100.0), int(stats["n"])


# ---------------------------------------------------------------------------
# Cleaner similarity matrix: same-pattern filtering is performed separately,
# so pattern IDs and duplicated pattern indicators are not included here.
# ---------------------------------------------------------------------------
MEMORY_NUMERIC_V4 = [
    "age",
    "health_status",
    "chronic_or_risk_count",
    "hbm_threat_score",
    "hbm_barrier_count",
    "hbm_barrier_score",
    "retail_clinic_visits_12m",
]

MEMORY_CATEGORICAL_V4 = [
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


def prepare_memory_matrix_v4(
    ordered: pd.DataFrame,
    memory_size: int,
) -> Tuple[np.ndarray, List[str]]:
    numeric_cols = [c for c in MEMORY_NUMERIC_V4 if c in ordered.columns]
    categorical_cols = [c for c in MEMORY_CATEGORICAL_V4 if c in ordered.columns]
    if not numeric_cols and not categorical_cols:
        raise ValueError("No memory features are available")

    transformers: List[Tuple[str, Any, List[str]]] = []
    if numeric_cols:
        numeric_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        categorical_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("cat", categorical_pipe, categorical_cols))
    transformer = ColumnTransformer(transformers, remainder="drop")
    fit_df = ordered.iloc[:memory_size]
    transformer.fit(fit_df)
    matrix = np.asarray(transformer.transform(ordered), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    try:
        names = [str(x) for x in transformer.get_feature_names_out()]
    except Exception:
        names = [f"feature_{i}" for i in range(matrix.shape[1])]
    return matrix.astype(np.float32), names


# ---------------------------------------------------------------------------
# Balanced dual memory
# ---------------------------------------------------------------------------
@dataclass
class DualMemoryItemV4:
    kind: str
    data_idx: int
    pattern_id: int
    actual: int
    vector: np.ndarray
    text: str
    confidence: float


class BalancedDualMemoryV4:
    def __init__(
        self,
        prototype_k: int,
        reflection_k: int,
        max_prototypes_per_bucket: int,
        max_reflections_per_bucket: int,
        min_similarity: float,
    ) -> None:
        self.prototype_k = max(0, int(prototype_k))
        self.reflection_k = max(0, int(reflection_k))
        self.max_prototypes_per_bucket = max(0, int(max_prototypes_per_bucket))
        self.max_reflections_per_bucket = max(0, int(max_reflections_per_bucket))
        self.min_similarity = float(min_similarity)
        self.prototypes: List[DualMemoryItemV4] = []
        self.reflections: List[DualMemoryItemV4] = []

    @staticmethod
    def _bucket(item: DualMemoryItemV4) -> Tuple[int, int]:
        return item.pattern_id, item.actual

    def _store_capped(
        self,
        container: List[DualMemoryItemV4],
        item: DualMemoryItemV4,
        cap: int,
    ) -> bool:
        if cap <= 0:
            return False
        bucket_indices = [
            i for i, old in enumerate(container) if self._bucket(old) == self._bucket(item)
        ]
        if len(bucket_indices) < cap:
            container.append(item)
            return True
        weakest_i = min(bucket_indices, key=lambda i: container[i].confidence)
        if item.confidence > container[weakest_i].confidence:
            container[weakest_i] = item
            return True
        return False

    def store_prototype(self, item: DualMemoryItemV4) -> bool:
        return self._store_capped(
            self.prototypes, item, self.max_prototypes_per_bucket
        )

    def store_reflection(self, item: DualMemoryItemV4) -> bool:
        return self._store_capped(
            self.reflections, item, self.max_reflections_per_bucket
        )

    def _similar_items(
        self,
        container: Sequence[DualMemoryItemV4],
        query: np.ndarray,
        pattern_id: int,
    ) -> List[Tuple[float, DualMemoryItemV4]]:
        candidates: List[Tuple[float, DualMemoryItemV4]] = []
        for item in container:
            if item.pattern_id != pattern_id:
                continue
            similarity = float(np.dot(query, item.vector))
            if similarity >= self.min_similarity:
                candidates.append((similarity, item))
        candidates.sort(key=lambda x: (x[0], x[1].confidence), reverse=True)
        return candidates

    def retrieve(
        self, query: np.ndarray, pattern_id: int
    ) -> Tuple[str, List[int], List[float], Dict[str, int]]:
        blocks: List[str] = []
        ids: List[int] = []
        sims: List[float] = []

        # Balance prototypes across actual outcomes: at most one YES and one NO
        # before adding any additional prototype.
        proto_candidates = self._similar_items(self.prototypes, query, pattern_id)
        selected_proto: List[Tuple[float, DualMemoryItemV4]] = []
        used_outcomes: set[int] = set()
        for sim, item in proto_candidates:
            if len(selected_proto) >= self.prototype_k:
                break
            if item.actual not in used_outcomes:
                selected_proto.append((sim, item))
                used_outcomes.add(item.actual)
        if len(selected_proto) < self.prototype_k:
            chosen_ids = {id(item) for _, item in selected_proto}
            for sim, item in proto_candidates:
                if len(selected_proto) >= self.prototype_k:
                    break
                if id(item) not in chosen_ids:
                    selected_proto.append((sim, item))
                    chosen_ids.add(id(item))

        reflection_candidates = self._similar_items(
            self.reflections, query, pattern_id
        )[: self.reflection_k]

        if selected_proto:
            blocks.append("TYPICAL CORRECT TRAINING PROTOTYPES")
            for rank, (sim, item) in enumerate(selected_proto, start=1):
                blocks.append(
                    f"Prototype {rank} (similarity={sim:.3f}):\n{item.text}"
                )
                ids.append(item.data_idx)
                sims.append(sim)
        if reflection_candidates:
            blocks.append("EXCEPTIONAL HIGH-CONFIDENCE ERROR REFLECTIONS")
            for rank, (sim, item) in enumerate(reflection_candidates, start=1):
                blocks.append(
                    f"Reflection {rank} (similarity={sim:.3f}):\n{item.text}"
                )
                ids.append(item.data_idx)
                sims.append(sim)
        if not blocks:
            blocks.append(
                "No sufficiently similar training memory is available. Use the pattern base rate and observed context only."
            )
        return (
            "\n\n".join(blocks),
            ids,
            sims,
            {
                "prototype_count": len(selected_proto),
                "reflection_count": len(reflection_candidates),
            },
        )

    def sizes(self) -> Dict[str, int]:
        return {
            "prototypes": len(self.prototypes),
            "reflections": len(self.reflections),
            "total": len(self.prototypes) + len(self.reflections),
        }


# ---------------------------------------------------------------------------
# CALL 3 adjustment and high-confidence reflection
# ---------------------------------------------------------------------------
def validate_adjustment_v4(obj: Dict[str, Any]) -> Dict[str, Any]:
    adjustment = int(obj["adjustment"])
    if not -20 <= adjustment <= 20:
        raise ValueError(f"Adjustment outside [-20,20]: {adjustment}")
    return {
        "adjustment": adjustment,
        "reason": normalize_reason(obj["reason"], max_chars=220),
    }


def make_decision_prompt_v4(
    *,
    row: pd.Series,
    assessment: Dict[str, Any],
    args: argparse.Namespace,
    base_rate: float,
    base_n: int,
    memory_text: str,
) -> Tuple[str, str]:
    pattern_id = int(row["hbm2_pattern"])
    threat_level, barrier_level = rule_levels(row)
    context_profile = build_context_profile(row, args.include_sensitive_context)
    prompt = DECISION_USER_TEMPLATE_V4.format(
        pattern_id=pattern_id,
        pattern_label=PATTERN_LABELS[pattern_id],
        rule_threat_score=int(row["hbm_threat_score"]),
        rule_threat_level=threat_level,
        rule_barrier_score=int(row["hbm_barrier_score"]),
        rule_barrier_level=barrier_level,
        pattern_tendency=PATTERN_TENDENCIES[pattern_id],
        base_rate=base_rate,
        base_n=base_n,
        threat_result=json.dumps(assessment["threat_result"], ensure_ascii=False),
        barrier_result=json.dumps(assessment["barrier_result"], ensure_ascii=False),
        context_profile=context_profile,
        memory_text=memory_text,
    )
    return prompt, context_profile


async def call_adjustment_v4(
    *,
    row: pd.Series,
    assessment: Dict[str, Any],
    args: argparse.Namespace,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    base_rate: float,
    base_n: int,
    memory_text: str,
) -> Tuple[Dict[str, Any], str, Dict[str, int], str, str, str]:
    user_prompt, context_profile = make_decision_prompt_v4(
        row=row,
        assessment=assessment,
        args=args,
        base_rate=base_rate,
        base_n=base_n,
        memory_text=memory_text,
    )
    obj, raw, usage, request_id = await structured_response_call(
        client,
        semaphore,
        model=args.model,
        schema_name="pattern_anchored_vaccination_adjustment",
        schema=ADJUSTMENT_SCHEMA_V4,
        system_prompt=DECISION_SYSTEM_V4,
        user_prompt=user_prompt,
        max_output_tokens=args.decision_max_tokens,
        temperature=args.temperature,
    )
    return (
        validate_adjustment_v4(obj),
        raw,
        usage,
        request_id,
        context_profile,
        user_prompt,
    )


async def call_reflection_v4(
    *,
    row: pd.Series,
    assessment: Dict[str, Any],
    adjustment_result: Dict[str, Any],
    base_rate: float,
    final_probability: float,
    memory_text: str,
    context_profile: str,
    args: argparse.Namespace,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
) -> Tuple[Dict[str, Any], str, Dict[str, int], str, str]:
    pattern_id = int(row["hbm2_pattern"])
    threat_level, barrier_level = rule_levels(row)
    predicted = int(final_probability >= 50.0)
    prompt = REFLECTION_USER_TEMPLATE_V4.format(
        actual_label="YES" if int(row["vaccinated"]) == 1 else "NO",
        base_rate=base_rate,
        adjustment=int(adjustment_result["adjustment"]),
        final_probability=final_probability,
        predicted_label="YES" if predicted else "NO",
        decision_reason=adjustment_result["reason"],
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
    obj, raw, usage, request_id = await structured_response_call(
        client,
        semaphore,
        model=args.reflection_model or args.model,
        schema_name="high_confidence_error_reflection",
        schema=REFLECTION_SCHEMA_V4,
        system_prompt=REFLECTION_SYSTEM_V4,
        user_prompt=prompt,
        max_output_tokens=args.reflection_max_tokens,
        temperature=args.temperature,
    )
    return validate_reflection(obj, pattern_id), raw, usage, request_id, prompt


def build_prototype_text_v4(entry: Dict[str, Any]) -> str:
    return (
        f"Actual outcome: {entry['actual_label']}. "
        f"Pattern: P{entry['hbm2_pattern']} {entry['pattern_label']}. "
        f"Base rate: {entry['base_probability']:.1f}%; adjustment: {entry['adjustment']:+d}; "
        f"final probability: {entry['probability_yes']:.1f}%. "
        f"Threat: {entry['threat_result']['reason']} "
        f"Barriers: {entry['barrier_result']['reason']} "
        f"Context: {entry['context_summary']}. "
        f"Reason: {entry['decision_reason']}"
    )


def build_reflection_text_v4(entry: Dict[str, Any]) -> str:
    reflection = entry["reflection"]
    return (
        f"Actual outcome: {entry['actual_label']}; erroneous probability: {entry['probability_yes']:.1f}%. "
        f"Pattern: P{entry['hbm2_pattern']} {entry['pattern_label']}. "
        f"Error cause: {reflection['error_cause']} "
        f"Missed/overweighted signal: {reflection['missed_or_overweighted_signal']} "
        f"Narrow correction: {reflection['correction_rule']}"
    )


def is_high_confidence_correct_v4(
    actual: int, probability: float, threshold: float
) -> bool:
    return bool(
        (actual == 1 and probability >= threshold)
        or (actual == 0 and probability <= 100.0 - threshold)
    )


def is_high_confidence_error_v4(
    actual: int, probability: float, threshold: float
) -> bool:
    return bool(
        (actual == 1 and probability <= 100.0 - threshold)
        or (actual == 0 and probability >= threshold)
    )


def prediction_confidence_v4(actual: int, probability: float) -> float:
    return float(probability if actual == 1 else 100.0 - probability)


async def process_memory_case_v4(
    *,
    idx: int,
    row: pd.Series,
    assessment: Dict[str, Any],
    args: argparse.Namespace,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    base_rates: Dict[int, Dict[str, float]],
    memory: BalancedDualMemoryV4,
    feature_matrix: np.ndarray,
) -> Dict[str, Any]:
    start = time.time()
    usage_total = dict(assessment.get("usage_total", {}))
    pattern_id = int(row["hbm2_pattern"])
    base_rate, base_n = leave_one_out_pattern_anchor_v4(
        row, base_rates, args.base_rate_prior_strength
    )
    memory_text, memory_ids, memory_sims, memory_mix = memory.retrieve(
        feature_matrix[idx], pattern_id
    )
    try:
        (
            adjustment_result,
            decision_raw,
            decision_usage,
            decision_request_id,
            context_profile,
            decision_user_prompt,
        ) = await call_adjustment_v4(
            row=row,
            assessment=assessment,
            args=args,
            client=client,
            semaphore=semaphore,
            base_rate=base_rate,
            base_n=base_n,
            memory_text=memory_text,
        )
        add_usage(usage_total, decision_usage)
        adjustment = int(adjustment_result["adjustment"])
        final_probability = float(np.clip(base_rate + adjustment, 0.0, 100.0))
        actual = int(row["vaccinated"])
        predicted = int(final_probability >= 50.0)
        correct = bool(predicted == actual)

        reflection: Optional[Dict[str, Any]] = None
        reflection_raw = ""
        reflection_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        reflection_request_id = ""
        reflection_prompt = ""
        high_confidence_error = is_high_confidence_error_v4(
            actual, final_probability, args.reflection_confidence_threshold
        )
        if high_confidence_error and args.reflection_k > 0:
            (
                reflection,
                reflection_raw,
                reflection_usage,
                reflection_request_id,
                reflection_prompt,
            ) = await call_reflection_v4(
                row=row,
                assessment=assessment,
                adjustment_result=adjustment_result,
                base_rate=base_rate,
                final_probability=final_probability,
                memory_text=memory_text,
                context_profile=context_profile,
                args=args,
                client=client,
                semaphore=semaphore,
            )
            add_usage(usage_total, reflection_usage)

        threat_level, barrier_level = rule_levels(row)
        entry: Dict[str, Any] = {
            "status": "ok",
            "experiment": EXPERIMENT,
            "version": VERSION,
            "timestamp": utc_now(),
            "data_idx": idx,
            "source_row_id": int(row["_source_row_id"]),
            "phase": "memory",
            "model": args.model,
            "actual": actual,
            "actual_label": "YES" if actual else "NO",
            "base_probability": round(base_rate, 3),
            "base_n": base_n,
            "adjustment": adjustment,
            "probability_yes": round(final_probability, 3),
            "predicted_at_50": predicted,
            "decision_at_50": "YES" if predicted else "NO",
            "is_correct_at_50": correct,
            "decision_reason": adjustment_result["reason"],
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
            "memory_mix": memory_mix,
            "memory_size_before": memory.sizes(),
            "high_confidence_correct": is_high_confidence_correct_v4(
                actual, final_probability, args.prototype_confidence_threshold
            ),
            "high_confidence_error": high_confidence_error,
            "reflection": reflection,
            "prototype_retained": False,
            "reflection_retained": False,
            "prompts": {
                "decision_system": DECISION_SYSTEM_V4,
                "decision_user": decision_user_prompt,
                "reflection_system": REFLECTION_SYSTEM_V4 if reflection else "",
                "reflection_user": reflection_prompt,
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
            "phase": "memory",
            "actual": int(row["vaccinated"]),
            "hbm2_pattern": pattern_id,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(limit=10),
            "usage_total": usage_total,
            "sample_time_sec": round(time.time() - start, 3),
        }


def apply_memory_updates_v4(
    entry: Dict[str, Any],
    memory: BalancedDualMemoryV4,
    vector: np.ndarray,
) -> Dict[str, Any]:
    if entry.get("status") != "ok":
        return entry
    actual = int(entry["actual"])
    if entry.get("high_confidence_correct"):
        retained = memory.store_prototype(
            DualMemoryItemV4(
                kind="prototype",
                data_idx=int(entry["data_idx"]),
                pattern_id=int(entry["hbm2_pattern"]),
                actual=actual,
                vector=vector,
                text=build_prototype_text_v4(entry),
                confidence=prediction_confidence_v4(
                    actual, float(entry["probability_yes"])
                ),
            )
        )
        entry["prototype_retained"] = bool(retained)
    if entry.get("reflection"):
        wrong_confidence = (
            100.0 - float(entry["probability_yes"])
            if actual == 1
            else float(entry["probability_yes"])
        )
        retained = memory.store_reflection(
            DualMemoryItemV4(
                kind="reflection",
                data_idx=int(entry["data_idx"]),
                pattern_id=int(entry["hbm2_pattern"]),
                actual=actual,
                vector=vector,
                text=build_reflection_text_v4(entry),
                confidence=float(wrong_confidence),
            )
        )
        entry["reflection_retained"] = bool(retained)
    entry["memory_size_after"] = memory.sizes()
    return entry


async def process_frozen_case_v4(
    *,
    idx: int,
    row: pd.Series,
    phase: str,
    assessment: Dict[str, Any],
    args: argparse.Namespace,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    base_rates: Dict[int, Dict[str, float]],
    frozen_memory: BalancedDualMemoryV4,
    feature_matrix: np.ndarray,
) -> Dict[str, Any]:
    start = time.time()
    usage_total = dict(assessment.get("usage_total", {}))
    pattern_id = int(row["hbm2_pattern"])
    base_rate, base_n = full_pattern_anchor_v4(row, base_rates)
    memory_text, memory_ids, memory_sims, memory_mix = frozen_memory.retrieve(
        feature_matrix[idx], pattern_id
    )
    try:
        (
            adjustment_result,
            decision_raw,
            decision_usage,
            decision_request_id,
            context_profile,
            decision_user_prompt,
        ) = await call_adjustment_v4(
            row=row,
            assessment=assessment,
            args=args,
            client=client,
            semaphore=semaphore,
            base_rate=base_rate,
            base_n=base_n,
            memory_text=memory_text,
        )
        add_usage(usage_total, decision_usage)
        adjustment = int(adjustment_result["adjustment"])
        final_probability = float(np.clip(base_rate + adjustment, 0.0, 100.0))
        actual = int(row["vaccinated"])
        predicted = int(final_probability >= 50.0)
        threat_level, barrier_level = rule_levels(row)
        return {
            "status": "ok",
            "experiment": EXPERIMENT,
            "version": VERSION,
            "timestamp": utc_now(),
            "data_idx": idx,
            "source_row_id": int(row["_source_row_id"]),
            "phase": phase,
            "model": args.model,
            "actual": actual,
            "actual_label": "YES" if actual else "NO",
            "base_probability": round(base_rate, 3),
            "base_n": base_n,
            "adjustment": adjustment,
            "probability_yes": round(final_probability, 3),
            "predicted_at_50": predicted,
            "decision_at_50": "YES" if predicted else "NO",
            "is_correct_at_50": bool(predicted == actual),
            "decision_reason": adjustment_result["reason"],
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
            "memory_mix": memory_mix,
            "memory_size": frozen_memory.sizes(),
            "memory_frozen": True,
            "reflection": None,
            "prompts": {
                "decision_system": DECISION_SYSTEM_V4,
                "decision_user": decision_user_prompt,
            },
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
            "phase": phase,
            "actual": int(row["vaccinated"]),
            "hbm2_pattern": pattern_id,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(limit=10),
            "usage_total": usage_total,
            "sample_time_sec": round(time.time() - start, 3),
        }


# ---------------------------------------------------------------------------
# Metrics and outputs
# ---------------------------------------------------------------------------
def _entries_ok_phase_v4(
    entries: Sequence[Dict[str, Any]], phase: str
) -> List[Dict[str, Any]]:
    return [
        e for e in entries if e.get("status") == "ok" and e.get("phase") == phase
    ]


def metrics_for_probability_field_v4(
    entries: Sequence[Dict[str, Any]],
    phase: str,
    probability_field: str,
    threshold: float,
) -> Dict[str, Any]:
    rows = _entries_ok_phase_v4(entries, phase)
    if not rows:
        return {"n_valid": 0}
    y_true = np.asarray([int(e["actual"]) for e in rows], dtype=int)
    probs_pct = np.asarray([float(e[probability_field]) for e in rows], dtype=float)
    probs = np.clip(probs_pct / 100.0, 1e-6, 1 - 1e-6)
    y_pred = (probs_pct >= threshold).astype(int)
    result: Dict[str, Any] = {
        "n_valid": len(rows),
        "actual_yes_rate": float(y_true.mean()),
        "mean_probability_yes": float(probs.mean()),
        "threshold": float(threshold),
        **binary_metrics_from_arrays(y_true, y_pred),
        "brier_score": float(brier_score_loss(y_true, probs)),
        "log_loss": float(log_loss(y_true, probs, labels=[0, 1])),
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probs))
        result["average_precision"] = float(average_precision_score(y_true, probs))
    else:
        result["roc_auc"] = None
        result["average_precision"] = None
    return result


def calibrate_threshold_v4(
    entries: Sequence[Dict[str, Any]],
    probability_field: str,
    metric: str,
) -> Tuple[int, pd.DataFrame, Dict[str, Any]]:
    rows = _entries_ok_phase_v4(entries, "calibration")
    if len(rows) < 4 or len({int(e["actual"]) for e in rows}) < 2:
        print("WARNING: Calibration split is too small or contains one class; using threshold 50.")
        return 50, pd.DataFrame(), {"selected_threshold": 50, "fallback": True}
    y_true = np.asarray([int(e["actual"]) for e in rows], dtype=int)
    probs = np.asarray([float(e[probability_field]) for e in rows], dtype=float)
    records: List[Dict[str, Any]] = []
    for threshold in range(0, 101):
        pred = (probs >= threshold).astype(int)
        records.append({"threshold": threshold, **binary_metrics_from_arrays(y_true, pred)})
    table = pd.DataFrame(records)
    best = float(table[metric].max())
    candidates = table[np.isclose(table[metric], best)].copy()
    candidates["distance_from_50"] = (candidates["threshold"] - 50).abs()
    chosen = candidates.sort_values(
        ["distance_from_50", "threshold"], ascending=[True, True]
    ).iloc[0]
    threshold = int(chosen["threshold"])
    return threshold, table, {
        "probability_field": probability_field,
        "threshold_metric": metric,
        "selected_threshold": threshold,
        "selected_metric_value": float(chosen[metric]),
        "calibration_metrics_at_50": binary_metrics_from_arrays(
            y_true, (probs >= 50).astype(int)
        ),
        "calibration_metrics_selected": binary_metrics_from_arrays(
            y_true, (probs >= threshold).astype(int)
        ),
        "fallback": False,
    }


def predictions_dataframe_v4(
    entries: Sequence[Dict[str, Any]],
    llm_threshold: int,
    base_threshold: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for e in sorted(entries, key=lambda x: int(x["data_idx"])):
        if e.get("status") != "ok":
            continue
        final_p = float(e["probability_yes"])
        base_p = float(e["base_probability"])
        rows.append(
            {
                "data_idx": int(e["data_idx"]),
                "source_row_id": int(e["source_row_id"]),
                "phase": e["phase"],
                "actual": int(e["actual"]),
                "hbm2_pattern": int(e["hbm2_pattern"]),
                "pattern_label": e["pattern_label"],
                "base_probability": base_p,
                "adjustment": int(e["adjustment"]),
                "probability_yes": final_p,
                "predicted_final_at_50": int(final_p >= 50),
                "predicted_final_calibrated": int(final_p >= llm_threshold),
                "llm_threshold": llm_threshold,
                "predicted_pattern_at_50": int(base_p >= 50),
                "predicted_pattern_calibrated": int(base_p >= base_threshold),
                "base_threshold": base_threshold,
                "decision_reason": e["decision_reason"],
                "rule_threat_score": int(e["rule_threat_score"]),
                "rule_threat_level": e["rule_threat_level"],
                "rule_barrier_score": int(e["rule_barrier_score"]),
                "rule_barrier_level": e["rule_barrier_level"],
                "llm_threat_score": int(e["threat_result"]["score"]),
                "llm_threat_reason": e["threat_result"]["reason"],
                "llm_barrier_score": int(e["barrier_result"]["score"]),
                "llm_barrier_reason": e["barrier_result"]["reason"],
                "memory_used": bool(e.get("memory_used")),
                "memory_source_indices": json.dumps(e.get("memory_source_indices", [])),
                "prototype_memory_count": int(e.get("memory_mix", {}).get("prototype_count", 0)),
                "reflection_memory_count": int(e.get("memory_mix", {}).get("reflection_count", 0)),
                "prototype_retained": bool(e.get("prototype_retained", False)),
                "reflection_retained": bool(e.get("reflection_retained", False)),
                "high_confidence_correct": bool(e.get("high_confidence_correct", False)),
                "high_confidence_error": bool(e.get("high_confidence_error", False)),
                "input_tokens": int(e.get("usage_total", {}).get("input_tokens", 0)),
                "output_tokens": int(e.get("usage_total", {}).get("output_tokens", 0)),
                "total_tokens": int(e.get("usage_total", {}).get("total_tokens", 0)),
                "sample_time_sec": float(e.get("sample_time_sec", 0)),
            }
        )
    return pd.DataFrame(rows)


def pattern_metrics_v4(
    entries: Sequence[Dict[str, Any]],
    phase: str,
    llm_threshold: int,
    base_threshold: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    phase_entries = _entries_ok_phase_v4(entries, phase)
    for pattern_id, label in PATTERN_LABELS.items():
        subset = [e for e in phase_entries if int(e["hbm2_pattern"]) == pattern_id]
        if not subset:
            continue
        y = np.asarray([int(e["actual"]) for e in subset], dtype=int)
        final_p = np.asarray([float(e["probability_yes"]) for e in subset])
        base_p = np.asarray([float(e["base_probability"]) for e in subset])
        rows.append(
            {
                "phase": phase,
                "pattern_id": pattern_id,
                "pattern_label": label,
                "n": len(subset),
                "actual_vaccination_rate": float(y.mean()),
                "mean_base_probability": float(base_p.mean() / 100.0),
                "mean_final_probability": float(final_p.mean() / 100.0),
                "mean_adjustment": float(np.mean([int(e["adjustment"]) for e in subset])),
                "final_accuracy_at_50": float(accuracy_score(y, final_p >= 50)),
                "final_accuracy_calibrated": float(accuracy_score(y, final_p >= llm_threshold)),
                "pattern_accuracy_at_50": float(accuracy_score(y, base_p >= 50)),
                "pattern_accuracy_calibrated": float(accuracy_score(y, base_p >= base_threshold)),
            }
        )
    return pd.DataFrame(rows)


def save_outputs_v4(
    *,
    output_dir: Path,
    assessment_latest: Dict[int, Dict[str, Any]],
    memory_latest: Dict[int, Dict[str, Any]],
    calibration_latest: Dict[int, Dict[str, Any]],
    test_latest: Dict[int, Dict[str, Any]],
    run_config: Dict[str, Any],
    base_rates: Dict[int, Dict[str, float]],
    llm_threshold: int,
    base_threshold: int,
    llm_calibration: Dict[str, Any],
    base_calibration: Dict[str, Any],
    llm_threshold_table: pd.DataFrame,
    base_threshold_table: pd.DataFrame,
    memory: BalancedDualMemoryV4,
) -> Dict[str, Any]:
    entries = [
        e
        for latest in (memory_latest, calibration_latest, test_latest)
        for e in latest.values()
        if e.get("status") == "ok"
    ]
    pred_df = predictions_dataframe_v4(entries, llm_threshold, base_threshold)
    pred_df.to_csv(output_dir / "predictions.csv", index=False)

    pattern_frames = [
        pattern_metrics_v4(entries, phase, llm_threshold, base_threshold)
        for phase in ("memory", "calibration", "test")
    ]
    pattern_df = pd.concat([x for x in pattern_frames if not x.empty], ignore_index=True)
    pattern_df.to_csv(output_dir / "pattern_metrics.csv", index=False)

    if not llm_threshold_table.empty:
        llm_threshold_table.to_csv(output_dir / "threshold_search_final.csv", index=False)
    if not base_threshold_table.empty:
        base_threshold_table.to_csv(output_dir / "threshold_search_pattern_only.csv", index=False)

    rate_rows = []
    for pattern_id in PATTERN_LABELS:
        stats = base_rates[pattern_id]
        rate_rows.append(
            {
                "pattern_id": pattern_id,
                "pattern_label": PATTERN_LABELS[pattern_id],
                **stats,
            }
        )
    pd.DataFrame(rate_rows).to_csv(output_dir / "pattern_base_rates.csv", index=False)

    usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for entry in assessment_latest.values():
        if entry.get("status") == "ok":
            add_usage(usage_total, entry.get("usage_total", {}))
    for entry in entries:
        # Decision entries include assessment usage, so subtracting would be
        # cumbersome. We report decision-stage totals separately below and the
        # full experiment total as assessment + non-assessment decision usage.
        decision_usage = entry.get("usage", {}).get("decision", {})
        reflection_usage = entry.get("usage", {}).get("reflection", {})
        add_usage(usage_total, decision_usage)
        add_usage(usage_total, reflection_usage)

    summary = {
        "experiment": EXPERIMENT,
        "version": VERSION,
        "created_at": utc_now(),
        "run_config": run_config,
        "pattern_base_rates": {str(k): v for k, v in base_rates.items() if k >= 0},
        "memory_final_sizes": memory.sizes(),
        "calibration": {
            "final_probability": llm_calibration,
            "pattern_only": base_calibration,
        },
        "metrics": {
            "memory_final_at_50": metrics_for_probability_field_v4(entries, "memory", "probability_yes", 50),
            "calibration_final_at_50": metrics_for_probability_field_v4(entries, "calibration", "probability_yes", 50),
            "calibration_final_selected": metrics_for_probability_field_v4(entries, "calibration", "probability_yes", llm_threshold),
            "test_final_at_50": metrics_for_probability_field_v4(entries, "test", "probability_yes", 50),
            "test_final_selected": metrics_for_probability_field_v4(entries, "test", "probability_yes", llm_threshold),
            "test_pattern_only_at_50": metrics_for_probability_field_v4(entries, "test", "base_probability", 50),
            "test_pattern_only_selected": metrics_for_probability_field_v4(entries, "test", "base_probability", base_threshold),
        },
        "usage_total": usage_total,
        "n_assessments_completed": sum(1 for e in assessment_latest.values() if e.get("status") == "ok"),
        "n_memory_decisions": sum(1 for e in memory_latest.values() if e.get("status") == "ok"),
        "n_calibration_decisions": sum(1 for e in calibration_latest.values() if e.get("status") == "ok"),
        "n_test_decisions": sum(1 for e in test_latest.values() if e.get("status") == "ok"),
        "files": {
            "assessment_log": str(output_dir / "logs" / "assessment_log.jsonl"),
            "memory_decision_log": str(output_dir / "logs" / "memory_decisions.jsonl"),
            "calibration_decision_log": str(output_dir / "logs" / "calibration_decisions.jsonl"),
            "test_decision_log": str(output_dir / "logs" / "test_decisions.jsonl"),
            "prototype_memory_log": str(output_dir / "logs" / "prototype_memory.jsonl"),
            "reflection_memory_log": str(output_dir / "logs" / "reflection_memory.jsonl"),
            "predictions": str(output_dir / "predictions.csv"),
            "pattern_metrics": str(output_dir / "pattern_metrics.csv"),
            "sampling_manifest": str(output_dir / "sampling_manifest.csv"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "threshold_calibration.json").write_text(
        json.dumps(
            {
                "final_probability": llm_calibration,
                "pattern_only": base_calibration,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser_v4() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run FLARE-VAX with pattern-anchored probabilities, balanced dual memory, "
            "and an independent calibration split."
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
        default="/content/drive/MyDrive/Vaccination-Decision-Model/Results/gpt4o_v4_pattern_anchor",
    )
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--reflection_model", type=str, default="")
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument(
        "--class_sampling",
        choices=["proportional", "balanced", "custom"],
        default="proportional",
        help="How the requested sample_size is allocated between vaccinated classes.",
    )
    parser.add_argument(
        "--positive_fraction",
        type=float,
        default=0.5,
        help="Vaccinated fraction used only when --class_sampling custom.",
    )
    parser.add_argument(
        "--class_fraction",
        type=float,
        default=0.0,
        help=(
            "Sample this fraction independently from each vaccinated class; "
            "when >0, it overrides --sample_size."
        ),
    )
    parser.add_argument(
        "--preserve_pattern_within_class",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preserve each class's HBM2 pattern mix during small-sample selection.",
    )
    parser.add_argument("--memory_ratio", type=float, default=0.50)
    parser.add_argument("--calibration_ratio", type=float, default=0.25)
    parser.add_argument("--test_ratio", type=float, default=0.25)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--joint_stratify",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--base_rate_prior_strength", type=float, default=10.0)
    parser.add_argument("--prototype_k", type=int, default=2)
    parser.add_argument("--reflection_k", type=int, default=1)
    parser.add_argument("--max_prototypes_per_bucket", type=int, default=5)
    parser.add_argument("--max_reflections_per_bucket", type=int, default=5)
    parser.add_argument("--memory_min_similarity", type=float, default=0.35)
    parser.add_argument("--prototype_confidence_threshold", type=float, default=65.0)
    parser.add_argument("--reflection_confidence_threshold", type=float, default=65.0)
    parser.add_argument(
        "--include_sensitive_context",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--assessment_max_tokens", type=int, default=140)
    parser.add_argument("--decision_max_tokens", type=int, default=100)
    parser.add_argument("--reflection_max_tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--max_concurrent_requests", type=int, default=20)
    parser.add_argument("--assessment_batch_size", type=int, default=32)
    parser.add_argument("--frozen_concurrent_samples", type=int, default=12)
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


def print_entry_brief_v4(entry: Dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print(
        f"SAMPLE {entry['data_idx']} | phase={entry['phase']} | actual={entry['actual_label']} | "
        f"base={entry['base_probability']:.1f}% | adj={entry['adjustment']:+d} | "
        f"final={entry['probability_yes']:.1f}%"
    )
    print(f"Pattern: P{entry['hbm2_pattern']} {entry['pattern_label']}")
    print(f"Reason: {entry['decision_reason']}")
    print(f"Memory mix: {entry.get('memory_mix', {})}; sources={entry.get('memory_source_indices', [])}")
    if entry.get("reflection"):
        print("Reflection: " + json.dumps(entry["reflection"], ensure_ascii=False))
    print("=" * 78)


# ---------------------------------------------------------------------------
# Main V4
# ---------------------------------------------------------------------------
async def async_main_v4() -> None:
    args = build_parser_v4().parse_args()
    if args.max_concurrent_requests < 1 or args.assessment_batch_size < 1:
        raise ValueError("Concurrency values must be at least 1")
    if args.frozen_concurrent_samples < 1:
        raise ValueError("--frozen_concurrent_samples must be at least 1")
    if not 50 <= args.prototype_confidence_threshold <= 100:
        raise ValueError("--prototype_confidence_threshold must be in [50,100]")
    if not 50 <= args.reflection_confidence_threshold <= 100:
        raise ValueError("--reflection_confidence_threshold must be in [50,100]")

    data_path = Path(args.data_path).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    assessment_log = logs_dir / "assessment_log.jsonl"
    memory_log = logs_dir / "memory_decisions.jsonl"
    calibration_log = logs_dir / "calibration_decisions.jsonl"
    test_log = logs_dir / "test_decisions.jsonl"
    prototype_log = logs_dir / "prototype_memory.jsonl"
    reflection_log = logs_dir / "reflection_memory.jsonl"
    config_path = output_dir / "run_config.json"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print("=" * 78)
    print("FLARE-VAX HBM2 — Pattern Anchor + Balanced Dual Memory")
    print("=" * 78)
    print(f"Version                    : {VERSION}")
    print(f"Model                      : {args.model}")
    print(f"Reflection model           : {args.reflection_model or args.model}")
    print(f"Data                       : {data_path}")
    print(f"Output                     : {output_dir}")
    print(f"Sample size                : {args.sample_size}")
    print(f"Class sampling             : {args.class_sampling}")
    print(f"Class fraction             : {args.class_fraction}")
    print(f"Preserve pattern/class     : {args.preserve_pattern_within_class}")
    print(
        f"Split ratios               : memory={args.memory_ratio}, "
        f"calibration={args.calibration_ratio}, test={args.test_ratio}"
    )
    print(f"Prototype/reflection k     : {args.prototype_k}/{args.reflection_k}")
    print(f"Confidence thresholds      : prototype={args.prototype_confidence_threshold}, reflection={args.reflection_confidence_threshold}")
    print(f"Assessment batch size      : {args.assessment_batch_size}")
    print(f"Frozen CALL3 concurrency   : {args.frozen_concurrent_samples}")

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

    sampled = class_aware_sample_v4(df, args)
    memory_df, calibration_df, test_df = split_memory_calibration_test_v4(sampled, args)
    memory_df["_phase"] = "memory"
    calibration_df["_phase"] = "calibration"
    test_df["_phase"] = "test"
    ordered = pd.concat([memory_df, calibration_df, test_df], ignore_index=True)
    memory_size = len(memory_df)
    calibration_start = memory_size
    test_start = memory_size + len(calibration_df)
    all_indices = list(range(len(ordered)))
    memory_indices = list(range(0, memory_size))
    calibration_indices = list(range(calibration_start, test_start))
    test_indices = list(range(test_start, len(ordered)))

    feature_matrix, memory_feature_names = prepare_memory_matrix_v4(
        ordered, memory_size
    )
    base_rates = fit_pattern_base_rates_v4(
        memory_df, args.base_rate_prior_strength
    )

    manifest_cols = [
        "_source_row_id",
        "_phase",
        "vaccinated",
        "hbm2_pattern",
        "hbm_threat_score",
        "hbm_barrier_score",
    ]
    manifest = ordered[manifest_cols].copy()
    manifest.insert(0, "data_idx", np.arange(len(ordered)))
    manifest.to_csv(output_dir / "sampling_manifest.csv", index=False)

    print(f"Rows available             : {len(df):,}")
    print(
        f"Selected rows              : {len(ordered)} "
        f"(memory={len(memory_df)}, calibration={len(calibration_df)}, test={len(test_df)})"
    )
    print(f"Selected target rate       : {ordered['vaccinated'].mean():.1%}")
    print(f"Memory vector dimensions   : {len(memory_feature_names)}")
    print("Selected class counts:")
    for y, n in ordered["vaccinated"].value_counts().sort_index().items():
        print(f"  class {int(y)}: n={int(n)}")
    print("Pattern distribution by phase:")
    for phase, phase_df in (("memory", memory_df), ("calibration", calibration_df), ("test", test_df)):
        print(f"  [{phase}]")
        for pattern_id in PATTERN_LABELS:
            group = phase_df[phase_df["hbm2_pattern"] == pattern_id]
            if len(group):
                print(
                    f"    P{pattern_id} {PATTERN_LABELS[pattern_id]:<29s} "
                    f"n={len(group):3d} actual_vax={group['vaccinated'].mean():.1%}"
                )
    print("Training-only pattern anchors (memory-build split):")
    for pattern_id in PATTERN_LABELS:
        stats = base_rates[pattern_id]
        print(
            f"  P{pattern_id}: n={int(stats['n'])}, raw={stats['raw_rate']:.1%}, "
            f"smoothed={stats['smoothed_rate']:.1%}"
        )

    run_config: Dict[str, Any] = {
        "version": VERSION,
        "model": args.model,
        "reflection_model": args.reflection_model or args.model,
        "data_path": str(data_path.resolve()),
        "data_size_bytes": data_path.stat().st_size,
        "sample_size": args.sample_size,
        "class_sampling": args.class_sampling,
        "positive_fraction": args.positive_fraction,
        "class_fraction": args.class_fraction,
        "preserve_pattern_within_class": args.preserve_pattern_within_class,
        "memory_ratio": args.memory_ratio,
        "calibration_ratio": args.calibration_ratio,
        "test_ratio": args.test_ratio,
        "random_seed": args.random_seed,
        "joint_stratify": args.joint_stratify,
        "base_rate_prior_strength": args.base_rate_prior_strength,
        "prototype_k": args.prototype_k,
        "reflection_k": args.reflection_k,
        "max_prototypes_per_bucket": args.max_prototypes_per_bucket,
        "max_reflections_per_bucket": args.max_reflections_per_bucket,
        "memory_min_similarity": args.memory_min_similarity,
        "prototype_confidence_threshold": args.prototype_confidence_threshold,
        "reflection_confidence_threshold": args.reflection_confidence_threshold,
        "memory_features": memory_feature_names,
        "include_sensitive_context": args.include_sensitive_context,
        "assessment_max_tokens": args.assessment_max_tokens,
        "decision_max_tokens": args.decision_max_tokens,
        "reflection_max_tokens": args.reflection_max_tokens,
        "temperature": args.temperature,
        "max_concurrent_requests": args.max_concurrent_requests,
        "assessment_batch_size": args.assessment_batch_size,
        "frozen_concurrent_samples": args.frozen_concurrent_samples,
        "threshold_metric": args.threshold_metric,
        "pipeline": {
            "call12": "all selected respondents precomputed concurrently",
            "memory_call3": "strictly sequential",
            "memory_update": "high-confidence prototypes and high-confidence error reflections only",
            "probability": "memory-build pattern base rate plus bounded [-20,+20] LLM adjustment",
            "calibration_call3": "concurrent with frozen final dual memory",
            "threshold_selection": "calibration split only",
            "test_call3": "concurrent with same frozen memory and frozen threshold",
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
        assessment_stub = {
            "threat_result": {"score": 4 if threat_level == "high" else 2, "level": threat_level, "reason": "Dry-run threat explanation."},
            "barrier_result": {"score": 4 if barrier_level == "high" else 2, "level": barrier_level, "reason": "Dry-run barrier explanation."},
        }
        if row["_phase"] == "memory":
            base_rate, base_n = leave_one_out_pattern_anchor_v4(
                row, base_rates, args.base_rate_prior_strength
            )
        else:
            base_rate, base_n = full_pattern_anchor_v4(row, base_rates)
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
        print("\n--- DRY RUN CALL 3 ---")
        prompt, _ = make_decision_prompt_v4(
            row=row,
            assessment=assessment_stub,
            args=args,
            base_rate=base_rate,
            base_n=base_n,
            memory_text="No memory in dry run.",
        )
        print(prompt)
        print("\nDry run completed; no API calls were made.")
        return

    if AsyncOpenAI is None:
        raise RuntimeError("Install the OpenAI SDK with: pip install -U openai")
    client = AsyncOpenAI(
        api_key=resolve_api_key(args.api_key),
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    semaphore = asyncio.Semaphore(args.max_concurrent_requests)

    assessment_latest = load_latest_log_entries(assessment_log)
    memory_latest = load_latest_log_entries(memory_log)
    calibration_latest = load_latest_log_entries(calibration_log)
    test_latest = load_latest_log_entries(test_log)

    # CALL 1/2 for every selected respondent, not phase by phase.
    await precompute_assessments_for_indices(
        indices=all_indices,
        ordered=ordered,
        args=args,
        client=client,
        semaphore=semaphore,
        assessment_log=assessment_log,
        assessment_latest=assessment_latest,
        phase_name="ALL",
    )

    # Reconstruct the final memory state from already completed sequential cases.
    dual_memory = BalancedDualMemoryV4(
        prototype_k=args.prototype_k,
        reflection_k=args.reflection_k,
        max_prototypes_per_bucket=args.max_prototypes_per_bucket,
        max_reflections_per_bucket=args.max_reflections_per_bucket,
        min_similarity=args.memory_min_similarity,
    )
    contiguous_memory_done = 0
    for idx in memory_indices:
        entry = memory_latest.get(idx)
        if not entry or entry.get("status") != "ok":
            break
        reconstructed = dict(entry)
        apply_memory_updates_v4(reconstructed, dual_memory, feature_matrix[idx])
        contiguous_memory_done += 1
    print(
        f"Sequential memory-build CALL3 resumes at {contiguous_memory_done}/{memory_size}; "
        f"memory={dual_memory.sizes()}"
    )

    progress = tqdm(
        total=memory_size,
        initial=contiguous_memory_done,
        desc="MEMORY CALL3 + selective reflection",
    )
    printed = 0
    for idx in memory_indices[contiguous_memory_done:]:
        assessment = assessment_latest.get(idx, {})
        if assessment.get("status") != "ok":
            if args.continue_on_error:
                progress.update(1)
                continue
            raise RuntimeError(f"Missing successful assessment for data_idx={idx}")
        entry = await process_memory_case_v4(
            idx=idx,
            row=ordered.iloc[idx],
            assessment=assessment,
            args=args,
            client=client,
            semaphore=semaphore,
            base_rates=base_rates,
            memory=dual_memory,
            feature_matrix=feature_matrix,
        )
        if entry.get("status") == "ok":
            entry = apply_memory_updates_v4(entry, dual_memory, feature_matrix[idx])
        append_jsonl(memory_log, entry)
        memory_latest[idx] = entry
        if entry.get("prototype_retained"):
            append_jsonl(
                prototype_log,
                {
                    "timestamp": utc_now(),
                    "data_idx": idx,
                    "pattern_id": entry["hbm2_pattern"],
                    "actual": entry["actual"],
                    "confidence": prediction_confidence_v4(entry["actual"], entry["probability_yes"]),
                    "text": build_prototype_text_v4(entry),
                },
            )
        if entry.get("reflection_retained"):
            append_jsonl(
                reflection_log,
                {
                    "timestamp": utc_now(),
                    "data_idx": idx,
                    "pattern_id": entry["hbm2_pattern"],
                    "actual": entry["actual"],
                    "confidence": (
                        100.0 - entry["probability_yes"]
                        if entry["actual"] == 1
                        else entry["probability_yes"]
                    ),
                    "reflection": entry["reflection"],
                    "text": build_reflection_text_v4(entry),
                },
            )
        if entry.get("status") != "ok" and not args.continue_on_error:
            progress.close()
            raise RuntimeError(
                f"Memory decision failed at data_idx={idx}: {entry.get('error_message')}"
            )
        if entry.get("status") == "ok" and printed < args.print_samples:
            print_entry_brief_v4(entry)
            printed += 1
        progress.update(1)
        if (idx + 1) % args.checkpoint_every == 0:
            print(
                f"Memory checkpoint: {idx + 1}/{memory_size}; "
                f"dual memory={dual_memory.sizes()}"
            )
    progress.close()

    print(f"Frozen final dual memory: {dual_memory.sizes()}")

    # Concurrent calibration using the final frozen memory.
    missing_cal = [
        idx
        for idx in calibration_indices
        if calibration_latest.get(idx, {}).get("status") != "ok"
    ]
    if missing_cal:
        progress = tqdm(total=len(missing_cal), desc="CALIBRATION CALL3 frozen memory")
        batch_size = max(1, args.frozen_concurrent_samples)
        for start in range(0, len(missing_cal), batch_size):
            batch = missing_cal[start : start + batch_size]
            results = await asyncio.gather(
                *[
                    process_frozen_case_v4(
                        idx=idx,
                        row=ordered.iloc[idx],
                        phase="calibration",
                        assessment=assessment_latest[idx],
                        args=args,
                        client=client,
                        semaphore=semaphore,
                        base_rates=base_rates,
                        frozen_memory=dual_memory,
                        feature_matrix=feature_matrix,
                    )
                    for idx in batch
                ]
            )
            for entry in sorted(results, key=lambda e: int(e["data_idx"])):
                append_jsonl(calibration_log, entry)
                calibration_latest[int(entry["data_idx"])] = entry
                progress.update(1)
                if entry.get("status") != "ok" and not args.continue_on_error:
                    progress.close()
                    raise RuntimeError(
                        f"Calibration decision failed at data_idx={entry['data_idx']}: {entry.get('error_message')}"
                    )
        progress.close()
    else:
        print("Calibration CALL3 is already complete.")

    all_decisions_for_calibration = list(calibration_latest.values())
    llm_threshold, llm_table, llm_calibration = calibrate_threshold_v4(
        all_decisions_for_calibration, "probability_yes", args.threshold_metric
    )
    base_threshold, base_table, base_calibration = calibrate_threshold_v4(
        all_decisions_for_calibration, "base_probability", args.threshold_metric
    )
    print("\nINDEPENDENT CALIBRATION-SPLIT THRESHOLDS")
    print(
        json.dumps(
            {
                "final_probability": llm_calibration,
                "pattern_only": base_calibration,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    # Concurrent test using the exact same frozen memory and frozen threshold.
    missing_test = [
        idx for idx in test_indices if test_latest.get(idx, {}).get("status") != "ok"
    ]
    if missing_test:
        progress = tqdm(total=len(missing_test), desc="TEST CALL3 frozen memory")
        batch_size = max(1, args.frozen_concurrent_samples)
        for start in range(0, len(missing_test), batch_size):
            batch = missing_test[start : start + batch_size]
            results = await asyncio.gather(
                *[
                    process_frozen_case_v4(
                        idx=idx,
                        row=ordered.iloc[idx],
                        phase="test",
                        assessment=assessment_latest[idx],
                        args=args,
                        client=client,
                        semaphore=semaphore,
                        base_rates=base_rates,
                        frozen_memory=dual_memory,
                        feature_matrix=feature_matrix,
                    )
                    for idx in batch
                ]
            )
            for entry in sorted(results, key=lambda e: int(e["data_idx"])):
                append_jsonl(test_log, entry)
                test_latest[int(entry["data_idx"])] = entry
                progress.update(1)
                if entry.get("status") != "ok" and not args.continue_on_error:
                    progress.close()
                    raise RuntimeError(
                        f"Test decision failed at data_idx={entry['data_idx']}: {entry.get('error_message')}"
                    )
        progress.close()
    else:
        print("Test CALL3 is already complete.")

    summary = save_outputs_v4(
        output_dir=output_dir,
        assessment_latest=assessment_latest,
        memory_latest=memory_latest,
        calibration_latest=calibration_latest,
        test_latest=test_latest,
        run_config=run_config,
        base_rates=base_rates,
        llm_threshold=llm_threshold,
        base_threshold=base_threshold,
        llm_calibration=llm_calibration,
        base_calibration=base_calibration,
        llm_threshold_table=llm_table,
        base_threshold_table=base_table,
        memory=dual_memory,
    )
    await client.close()

    print("\n" + "=" * 78)
    print("RUN SUMMARY")
    print("=" * 78)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSampling manifest : {output_dir / 'sampling_manifest.csv'}")
    print(f"Predictions       : {output_dir / 'predictions.csv'}")
    print(f"Pattern metrics   : {output_dir / 'pattern_metrics.csv'}")
    print(f"Summary           : {output_dir / 'summary.json'}")


def main() -> None:
    asyncio.run(async_main_v4())


if __name__ == "__main__":
    main()
