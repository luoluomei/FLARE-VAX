#!/usr/bin/env python3
"""
FLARE-VAX theory-scaffolded LLM baselines via the ASU Research Computing API.

Implemented experiments
-----------------------
1. HBM-CoPB on V4 with every requested LLM.
2. HBM-CoPB on V5 with every requested LLM.
3. HBM-PB&J on V4 with every requested LLM.

HBM-CoPB
~~~~~~~~
A single prediction workflow. Eight fixed, balanced memory-split respondents are
converted into profile -> HBM-structured reasoning -> label demonstrations. A
new respondent is then predicted through the same five-stage theory scaffold.
The model receives no deterministic HBM scores, meta-dimensions, HBM8 pattern,
pattern base rate, retrieved rule, reflection, or training-error feedback.

HBM-PB&J
~~~~~~~~
A two-call workflow, implemented only for V4 because prior non-target vaccine
behaviours provide the closest analogue to PB&J seed judgements.

Call 1: raw profile -> label-blind HBM-scaffolded health persona.
Call 2: raw profile + generated persona + eight fixed persona demonstrations
        -> influenza-vaccination probability and label.

Call 1 never receives the influenza-vaccination target. All labels used in
few-shot demonstrations come only from the memory split.

Default ASU endpoint:
    https://openai.rc.asu.edu/v1

The script is resume-safe. Successful JSONL entries are reused, failed or
missing entries are retried, provided the experimental configuration is
unchanged and --overwrite is not used.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import httpx
import numpy as np
import pandas as pd
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

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore


VERSION = "flare_vax_hbm_copb_pbj_asu_v1"
PROMPT_VERSION = "hbm_copb_pbj_grounded_compact_v1"
TARGET = "SHTFLU12M_A"
WEIGHT = "WTFA_A"
ID_COLUMNS = ["HHX", "SRVY_YR", "PSTRAT", "PPSU"]

EXPERIMENTS = ["hbm_copb_v4", "hbm_copb_v5", "hbm_pbj_v4"]

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
    BACKGROUND_COLUMNS + CHRONIC_VARS + BARRIER_COLUMNS + CONTACT_COLUMNS + NAVIGATION_COLUMNS
))
FEATURE_COLUMNS_BY_VARIANT = {
    "v4": sorted(set(BASE_FEATURE_COLUMNS + V4_VACCINE_COLUMNS)),
    "v5": sorted(set(BASE_FEATURE_COLUMNS)),
}
ALL_REQUIRED_COLUMNS = sorted(set(ID_COLUMNS + [TARGET, WEIGHT] + FEATURE_COLUMNS_BY_VARIANT["v4"]))

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
        "HITCOMM_A", "HITTEST_A", "SHTCVD191_A", "SHTPNUEV_A",
        "SHTSHINGL1_A", "SHINGRIX3_A", "SHTHEPA_A",
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
    0: "never_attended_or_kindergarten", 1: "grades_1_11",
    2: "12th_grade_no_diploma", 3: "GED", 4: "high_school_graduate",
    5: "some_college", 6: "technical_associate", 7: "academic_associate",
    8: "bachelor", 9: "master", 10: "professional_or_doctoral",
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
    1: "doctor_office_or_health_center", 2: "urgent_care_or_retail_clinic",
    3: "hospital_emergency_room", 4: "VA_facility", 5: "other_place",
    6: "no_single_place_used_most_often",
}
COMM_DIFFICULTY = {1: "none", 2: "some", 3: "a_lot", 4: "cannot_do_at_all"}

LEVELS = {"low", "moderate", "high", "mixed", "uncertain"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.integer):
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
        if not np.isfinite(x) or int(x) in set(invalid):
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
    if column in {"RACEALLP_A", "HISPALLP_A"}:
        x = numeric_value(value)
        return None if x is None else int(x)
    x = numeric_value(value)
    if x is None:
        return None
    return int(x) if float(x).is_integer() else float(x)


def target_to_binary(value: Any) -> Optional[int]:
    value_bool = yes_no_value(value)
    return None if value_bool is None else int(value_bool)


def feature_columns_for_variant(variant: str, include_sensitive: bool) -> List[str]:
    columns = list(FEATURE_COLUMNS_BY_VARIANT[variant])
    if not include_sensitive:
        columns = [c for c in columns if c not in {"RACEALLP_A", "HISPALLP_A"}]
    if variant == "v5":
        leaked = FORBIDDEN_V5_VACCINE_COLUMNS.intersection(columns)
        if leaked:
            raise AssertionError(f"V5 feature policy leaked vaccine variables: {sorted(leaked)}")
    return columns


def build_profile(row: pd.Series, variant: str, feature_columns: Sequence[str]) -> Dict[str, Any]:
    profile = {FEATURE_NAMES.get(c, c): clean_feature_value(c, row.get(c)) for c in feature_columns}
    if variant == "v5":
        forbidden_names = {FEATURE_NAMES.get(c, c) for c in FORBIDDEN_V5_VACCINE_COLUMNS}
        overlap = forbidden_names.intersection(profile)
        if overlap:
            raise AssertionError(f"V5 prompt leaked vaccine variables: {sorted(overlap)}")
    return profile


def compact_values(row: pd.Series, variant: str, feature_columns: Sequence[str]) -> List[Any]:
    profile = build_profile(row, variant, feature_columns)
    order = [FEATURE_NAMES.get(c, c) for c in feature_columns]
    return [profile[name] for name in order]


def config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def append_jsonl(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(obj), ensure_ascii=False, default=json_safe) + "\n")


def load_latest_jsonl(
    path: Path,
    expected_hash: str,
    *,
    key_field: str,
) -> Dict[int, Dict[str, Any]]:
    latest: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                obj = json.loads(line)
                if obj.get("config_hash") == expected_hash and key_field in obj:
                    latest[int(obj[key_field])] = obj
            except Exception:
                continue
    return latest


def usage_from_response(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens = int(
        getattr(usage, "prompt_tokens", None)
        or getattr(usage, "input_tokens", 0)
        or 0
    )
    output_tokens = int(
        getattr(usage, "completion_tokens", None)
        or getattr(usage, "output_tokens", 0)
        or 0
    )
    total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def sum_usage(entries: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for entry in entries:
        usage = entry.get("usage") or {}
        for key in total:
            total[key] += int(usage.get(key, 0) or 0)
    return total


def _add_usage(total: Dict[str, int], usage: Mapping[str, Any]) -> None:
    for key in total:
        total[key] += int(usage.get(key, 0) or 0)


def _classify_api_exception(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if "ratelimit" in name or "429" in message or "rate limit" in message:
        return "api_rate_limit"
    if "timeout" in name or "timed out" in message:
        return "api_timeout"
    if "connection" in name or "connect" in name or "connection" in message:
        return "api_connection"
    if "authentication" in name or "401" in message:
        return "api_authentication"
    if "permission" in name or "403" in message:
        return "api_permission"
    return "api_other"


class StructuredCallFailure(RuntimeError):
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


def _json_candidates(text: str) -> List[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    candidates: List[Dict[str, Any]] = []
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            candidates.append(obj)
    except Exception:
        pass
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
            if isinstance(obj, dict):
                candidates.append(obj)
        except Exception:
            continue
    return candidates


def extract_json_object(raw_text: str) -> Dict[str, Any]:
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("empty model output")
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()
    candidates = _json_candidates(cleaned)
    if not candidates:
        raise ValueError("no valid JSON object found")
    return max(candidates, key=lambda obj: len(json.dumps(obj, ensure_ascii=False)))


def _limited_string(value: Any, field: str, limit: int = 1200) -> str:
    text = " ".join(str(value or "").strip().split())
    if not text:
        raise ValueError(f"missing text field: {field}")
    return text[:limit]


def _optional_string(
    value: Any,
    *,
    default: str = "No additional uncertainty was identified from the supplied observations.",
    limit: int = 500,
) -> str:
    """Normalize an optional text field without rejecting an otherwise valid persona.

    Some models correctly return an empty string when no additional uncertainty is
    identifiable. Treat that as an explicit neutral statement rather than failing
    the entire structured response.
    """
    text = " ".join(str(value or "").strip().split())
    if not text:
        text = default
    return text[:limit]


def _limited_list(value: Any, field: str, max_items: int = 6, item_limit: int = 300) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return [" ".join(str(x).strip().split())[:item_limit] for x in value[:max_items] if str(x).strip()]


def _dimension(obj: Any, field: str) -> Dict[str, Any]:
    if not isinstance(obj, Mapping):
        raise ValueError(f"{field} must be an object")
    level = str(obj.get("level", "")).strip().lower()
    if level not in LEVELS:
        raise ValueError(f"{field}.level must be one of {sorted(LEVELS)}")
    return {
        "level": level,
        "evidence": _limited_list(obj.get("evidence"), f"{field}.evidence", max_items=4),
        "interpretation": _limited_string(obj.get("interpretation"), f"{field}.interpretation", 700),
    }


def validate_copb_reasoning(obj: Mapping[str, Any], variant: str, *, require_prediction: bool) -> Dict[str, Any]:
    stage2 = "vaccine_acceptance_or_benefit" if variant == "v4" else "preventive_engagement"
    out: Dict[str, Any] = {
        "observed_threat": _dimension(obj.get("observed_threat"), "observed_threat"),
        stage2: _dimension(obj.get(stage2), stage2),
        "structural_barriers": _dimension(obj.get("structural_barriers"), "structural_barriers"),
        "healthcare_cues": _dimension(obj.get("healthcare_cues"), "healthcare_cues"),
        "navigation_self_efficacy": _dimension(obj.get("navigation_self_efficacy"), "navigation_self_efficacy"),
    }
    integration = obj.get("theory_guided_integration")
    if not isinstance(integration, Mapping):
        raise ValueError("theory_guided_integration must be an object")
    out["theory_guided_integration"] = {
        "factors_increasing_probability": _limited_list(
            integration.get("factors_increasing_probability"),
            "theory_guided_integration.factors_increasing_probability",
            max_items=5,
        ),
        "factors_decreasing_probability": _limited_list(
            integration.get("factors_decreasing_probability"),
            "theory_guided_integration.factors_decreasing_probability",
            max_items=5,
        ),
        "uncertainties": _limited_list(
            integration.get("uncertainties"),
            "theory_guided_integration.uncertainties",
            max_items=4,
        ),
        "summary": _limited_string(integration.get("summary"), "theory_guided_integration.summary", 900),
    }
    if require_prediction:
        prediction = validate_prediction(obj)
        out.update(prediction)
    else:
        label = str(obj.get("observed_label", "")).strip().upper()
        if label not in {"YES", "NO"}:
            raise ValueError("observed_label must be YES or NO")
        out["observed_label"] = label
    return out


def validate_persona(obj: Mapping[str, Any]) -> Dict[str, Any]:
    required = [
        "observed_health_vulnerability",
        "preventive_orientation",
        "access_and_barrier_profile",
        "healthcare_engagement_style",
        "navigation_capacity",
        "integrated_health_persona",
    ]

    # Work on a shallow mutable copy. This preserves strict validation while
    # repairing only a narrow, observed formatting error: a required top-level
    # object may be placed one level too deep inside another required object.
    candidate: Dict[str, Any] = dict(obj)

    # Some models wrap the requested persona under one obvious container.
    for wrapper in ("persona", "health_persona", "hbm_persona", "output"):
        wrapped = candidate.get(wrapper)
        if isinstance(wrapped, Mapping) and sum(
            isinstance(wrapped.get(field), Mapping) for field in required
        ) >= 4:
            candidate = dict(wrapped)
            break

    # Promote a uniquely nested required object one level upward. This repairs
    # outputs such as navigation_capacity.integrated_health_persona while still
    # rejecting ambiguous or missing structures.
    for missing_field in required:
        if isinstance(candidate.get(missing_field), Mapping):
            continue
        nested_matches = []
        for parent_field in required:
            parent = candidate.get(parent_field)
            if isinstance(parent, Mapping) and isinstance(parent.get(missing_field), Mapping):
                nested_matches.append(parent.get(missing_field))
        if len(nested_matches) == 1:
            candidate[missing_field] = nested_matches[0]

    # Detect input-envelope echo explicitly so the retry path can switch away
    # from response_format=json_object and issue a stronger formatting request.
    echoed_envelope_keys = {
        "raw_respondent_profile",
        "influenza_target_is_withheld",
        "required_output_schema",
    }.intersection(candidate)
    if echoed_envelope_keys and not any(
        isinstance(candidate.get(field), Mapping) for field in required
    ):
        raise ValueError(
            "model echoed the input envelope instead of returning the persona object"
        )

    for field in required:
        if not isinstance(candidate.get(field), Mapping):
            raise ValueError(f"{field} must be an object")

    obj = candidate
    output = {
        "observed_health_vulnerability": {
            "summary": _limited_string(obj["observed_health_vulnerability"].get("summary"), "observed_health_vulnerability.summary", 700),
            "evidence": _limited_list(obj["observed_health_vulnerability"].get("evidence"), "observed_health_vulnerability.evidence", 5),
            "uncertainty": _optional_string(
                obj["observed_health_vulnerability"].get("uncertainty"),
                limit=500,
            ),
        },
        "preventive_orientation": {
            "summary": _limited_string(obj["preventive_orientation"].get("summary"), "preventive_orientation.summary", 700),
            "evidence": _limited_list(obj["preventive_orientation"].get("evidence"), "preventive_orientation.evidence", 5),
            "counter_evidence": _limited_list(obj["preventive_orientation"].get("counter_evidence"), "preventive_orientation.counter_evidence", 5),
            "uncertainty": _optional_string(
                obj["preventive_orientation"].get("uncertainty"),
                limit=500,
            ),
        },
        "access_and_barrier_profile": {
            "summary": _limited_string(obj["access_and_barrier_profile"].get("summary"), "access_and_barrier_profile.summary", 700),
            "facilitators": _limited_list(obj["access_and_barrier_profile"].get("facilitators"), "access_and_barrier_profile.facilitators", 5),
            "constraints": _limited_list(obj["access_and_barrier_profile"].get("constraints"), "access_and_barrier_profile.constraints", 5),
        },
        "healthcare_engagement_style": {
            "summary": _limited_string(obj["healthcare_engagement_style"].get("summary"), "healthcare_engagement_style.summary", 700),
            "preventive_contacts": _limited_list(obj["healthcare_engagement_style"].get("preventive_contacts"), "healthcare_engagement_style.preventive_contacts", 5),
            "reactive_contacts": _limited_list(obj["healthcare_engagement_style"].get("reactive_contacts"), "healthcare_engagement_style.reactive_contacts", 5),
            "uncertainty": _optional_string(
                obj["healthcare_engagement_style"].get("uncertainty"),
                limit=500,
            ),
        },
        "navigation_capacity": {
            "summary": _limited_string(obj["navigation_capacity"].get("summary"), "navigation_capacity.summary", 700),
            "supporting_evidence": _limited_list(obj["navigation_capacity"].get("supporting_evidence"), "navigation_capacity.supporting_evidence", 5),
            "limitations": _limited_list(obj["navigation_capacity"].get("limitations"), "navigation_capacity.limitations", 5),
        },
        "integrated_health_persona": {
            "persona_statement": _limited_string(obj["integrated_health_persona"].get("persona_statement"), "integrated_health_persona.persona_statement", 1200),
            "stable_tendencies_supported_by_evidence": _limited_list(
                obj["integrated_health_persona"].get("stable_tendencies_supported_by_evidence"),
                "integrated_health_persona.stable_tendencies_supported_by_evidence",
                6,
            ),
            "important_contradictions": _limited_list(
                obj["integrated_health_persona"].get("important_contradictions"),
                "integrated_health_persona.important_contradictions",
                6,
            ),
            "do_not_infer": _limited_list(obj["integrated_health_persona"].get("do_not_infer"), "integrated_health_persona.do_not_infer", 6),
        },
    }
    raw_text = json.dumps(output, ensure_ascii=False).lower()
    forbidden_patterns = [r"\bshtflu12m_a\b", r"\bhbm8\b", r"pattern base rate", r"reflective memory"]
    if any(re.search(pattern, raw_text) for pattern in forbidden_patterns):
        raise ValueError("persona contains forbidden target/method information")
    return output


def validate_prediction(obj: Mapping[str, Any]) -> Dict[str, Any]:
    if "probability_yes" not in obj:
        raise ValueError("missing probability_yes")
    probability = float(obj["probability_yes"])
    if not np.isfinite(probability):
        raise ValueError("probability_yes is not finite")
    probability = float(np.clip(probability, 0.0, 100.0))
    raw_prediction = str(obj.get("prediction", "")).strip().upper()
    derived = "YES" if probability >= 50 else "NO"
    return {
        "probability_yes": probability,
        "raw_prediction": raw_prediction if raw_prediction in {"YES", "NO"} else "",
        "prediction_at_50": derived,
        "prediction_consistency_corrected": bool(raw_prediction in {"YES", "NO"} and raw_prediction != derived),
    }


def validate_pbj_prediction(obj: Mapping[str, Any]) -> Dict[str, Any]:
    application = obj.get("persona_application")
    if not isinstance(application, Mapping):
        raise ValueError("persona_application must be an object")
    output = {
        "persona_application": {
            "relevant_persona_evidence": _limited_list(
                application.get("relevant_persona_evidence"),
                "persona_application.relevant_persona_evidence",
                5,
            ),
            "contradictory_current_evidence": _limited_list(
                application.get("contradictory_current_evidence"),
                "persona_application.contradictory_current_evidence",
                5,
            ),
            "summary": _limited_string(application.get("summary"), "persona_application.summary", 800),
        }
    }
    output.update(validate_prediction(obj))
    return output


COPB_DEMO_SYSTEM = """You are authoring one evidence-grounded HBM-CoPB few-shot demonstration.

You receive a raw NHIS respondent profile and its observed training label for influenza vaccination.
The label is supplied only because this is a training demonstration. Do not use the label to invent
unobserved beliefs or force every dimension to agree with it. Preserve contradictory evidence.

Structure the evidence through five HBM-inspired observed dimensions:
1. objective health vulnerability relevant to threat;
2. {stage2_description};
3. structural barriers;
4. healthcare-contact opportunities as cues;
5. observed healthcare navigation capacity as a self-efficacy proxy.

Constraints:
- These are observed proxies, not direct psychometric beliefs.
- Missing/unknown/not applicable is not the same as no, refusal, or absence.
- Never invent physician recommendations, reminders, intentions, trust, or private attitudes.
- Do not calculate or mention deterministic HBM scores, Motivation/Capability/Activation,
  HBM8 patterns, base rates, retrieval, reflection, memory, or error feedback.
- Keep each interpretation concise and cite at most four supplied observations.
- Return only JSON.
"""

COPB_PREDICTION_SYSTEM = """You are implementing the HBM-CoPB theory-structured reasoning baseline.

Predict whether the target NHIS respondent received an influenza vaccination in the past 12 months.
Follow the same five-stage HBM-inspired observed reasoning structure shown in the eight demonstrations.

Constraints:
- Use only supplied raw observations and demonstration content.
- These dimensions are observed proxies, not direct measurements of private beliefs.
- Missing/unknown/not applicable is not no or refusal.
- Never invent physician recommendations, reminders, intentions, trust, or private attitudes.
- Do not use or mention deterministic HBM scores, Motivation/Capability/Activation,
  HBM8 patterns, pattern base rates, retrieval, reflective memory, correction rules,
  training errors, or the target's true label.
- Return only JSON with the five dimensions, theory_guided_integration,
  probability_yes, and prediction.
"""

PBJ_PERSONA_SYSTEM = """You are implementing Call 1 of an HBM-PB&J-style baseline.

Construct an evidence-grounded health persona from the respondent's observed V4 profile,
including demographics, health/access context, healthcare behaviour, navigation behaviour,
and prior non-target vaccination history.

This call is label-blind:
- You are NOT given the respondent's influenza-vaccination outcome.
- Do not predict or discuss whether the respondent received an influenza vaccine.
- Explain observed prior preventive behaviour through an HBM-inspired scaffold.

Constraints:
- Prior non-target vaccine non-receipt may reflect eligibility, opportunity, or missingness;
  do not automatically interpret it as refusal.
- Missing/unknown/not applicable is not no, refusal, or absence.
- Do not invent physician recommendations, reminders, intentions, trust, values, or private beliefs.
- Do not mention deterministic HBM scores, HBM8 patterns, pattern base rates, retrieval,
  reflective memory, correction rules, training errors, or SHTFLU12M_A.
- Return only JSON using the required persona schema.
"""

PBJ_PREDICTION_SYSTEM = """You are implementing Call 2 of an HBM-PB&J-style baseline.

Predict whether the target respondent received an influenza vaccination in the past 12 months.
Use the supplied raw profile, the label-blind HBM-rationalized health persona, and eight fixed
memory-split persona demonstrations.

The persona explains observed prior behaviour; do not redo a full HBM-CoPB five-stage analysis.
Do not use or mention HBM8 patterns, pattern base rates, retrieval, reflective memory,
correction rules, training errors, or the target's true label.
Return only JSON with persona_application, probability_yes, and prediction.
"""

FORMAT_RETRY = """
FORMAT RETRY:

The previous response did not satisfy the required output contract.

Do NOT repeat or return the input envelope. In particular, do not return these input keys:
- variant
- raw_respondent_profile
- influenza_target_is_withheld
- required_output_schema
- feature_order
- persona_demonstrations
- target_respondent

Create a NEW answer object that follows the requested output schema.
Keep every required object at the top level shown in the schema.
Return valid JSON only, with no Markdown and no prose outside the JSON object.
"""


def copb_stage2_description(variant: str) -> str:
    if variant == "v4":
        return (
            "observed vaccine acceptance/benefit-consistent evidence using prior non-target "
            "vaccination behaviour, while considering eligibility and opportunity uncertainty"
        )
    return (
        "observed preventive engagement using wellness, health-information, clinician-communication, "
        "result-review, and healthcare-use behaviour; do not call this a direct perceived-benefit scale"
    )


def copb_demo_schema(variant: str) -> Dict[str, Any]:
    stage2 = "vaccine_acceptance_or_benefit" if variant == "v4" else "preventive_engagement"
    dimension = {"level": "low|moderate|high|mixed|uncertain", "evidence": ["observed feature=value"], "interpretation": "concise evidence-grounded interpretation"}
    return {
        "observed_threat": dimension,
        stage2: dimension,
        "structural_barriers": dimension,
        "healthcare_cues": dimension,
        "navigation_self_efficacy": dimension,
        "theory_guided_integration": {
            "factors_increasing_probability": ["observed evidence"],
            "factors_decreasing_probability": ["observed evidence"],
            "uncertainties": ["uncertainty"],
            "summary": "concise integration",
        },
        "observed_label": "YES|NO",
    }


def copb_prediction_schema(variant: str) -> Dict[str, Any]:
    schema = copb_demo_schema(variant).copy()
    schema.pop("observed_label", None)
    schema["probability_yes"] = "number from 0 to 100"
    schema["prediction"] = "YES|NO"
    return schema


def persona_schema() -> Dict[str, Any]:
    return {
        "observed_health_vulnerability": {"summary": "text", "evidence": ["feature=value"], "uncertainty": "text"},
        "preventive_orientation": {"summary": "text", "evidence": ["observed prior behaviour"], "counter_evidence": ["contradiction"], "uncertainty": "text"},
        "access_and_barrier_profile": {"summary": "text", "facilitators": ["observed facilitator"], "constraints": ["observed constraint"]},
        "healthcare_engagement_style": {"summary": "text", "preventive_contacts": ["observed contact"], "reactive_contacts": ["observed contact"], "uncertainty": "text"},
        "navigation_capacity": {"summary": "text", "supporting_evidence": ["observed evidence"], "limitations": ["observed limitation"]},
        "integrated_health_persona": {
            "persona_statement": "concise evidence-grounded persona",
            "stable_tendencies_supported_by_evidence": ["supported tendency"],
            "important_contradictions": ["contradiction"],
            "do_not_infer": ["unsupported inference to avoid"],
        },
    }


def build_copb_demo_prompt(profile: Mapping[str, Any], label: int, variant: str, strict_retry: bool) -> str:
    payload = {
        "variant": variant,
        "raw_respondent_profile": profile,
        "observed_training_label": "YES" if label == 1 else "NO",
        "required_output_schema": copb_demo_schema(variant),
    }
    suffix = FORMAT_RETRY if strict_retry else ""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + suffix


def build_copb_prediction_prompt(
    feature_order: Sequence[str],
    demos: Sequence[Mapping[str, Any]],
    target_values: Sequence[Any],
    variant: str,
    strict_retry: bool,
) -> str:
    payload = {
        "variant": variant,
        "feature_order": list(feature_order),
        "theory_structured_demonstrations": [
            {
                "source_row_index": int(d["source_row_index"]),
                "values": d["values"],
                "hbm_structured_reasoning": d["reasoning"],
                "label": "YES" if int(d["label"]) == 1 else "NO",
            }
            for d in demos
        ],
        "target_respondent": {"values": list(target_values), "label": "UNKNOWN"},
        "required_output_schema": copb_prediction_schema(variant),
    }
    suffix = FORMAT_RETRY if strict_retry else ""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + suffix


def build_persona_prompt(profile: Mapping[str, Any], strict_retry: bool) -> str:
    payload = {
        "variant": "v4",
        "raw_respondent_profile": profile,
        "influenza_target_is_withheld": True,
        "required_output_schema": persona_schema(),
    }
    suffix = FORMAT_RETRY if strict_retry else ""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + suffix


def build_pbj_prediction_prompt(
    feature_order: Sequence[str],
    demos: Sequence[Mapping[str, Any]],
    target_values: Sequence[Any],
    target_persona: Mapping[str, Any],
    strict_retry: bool,
) -> str:
    payload = {
        "variant": "v4",
        "feature_order": list(feature_order),
        "persona_demonstrations": [
            {
                "source_row_index": int(d["source_row_index"]),
                "values": d["values"],
                "hbm_rationalized_persona": d["persona"],
                "label": "YES" if int(d["label"]) == 1 else "NO",
            }
            for d in demos
        ],
        "target_respondent": {
            "values": list(target_values),
            "hbm_rationalized_persona": target_persona,
            "label": "UNKNOWN",
        },
        "required_output_schema": {
            "persona_application": {
                "relevant_persona_evidence": ["relevant persona evidence"],
                "contradictory_current_evidence": ["contradictory evidence"],
                "summary": "concise explanation of how the persona informs prediction",
            },
            "probability_yes": "number from 0 to 100",
            "prediction": "YES|NO",
        },
    }
    suffix = FORMAT_RETRY if strict_retry else ""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + suffix


async def probe_json_mode(client: Any, model: str) -> Tuple[bool, str]:
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": 'Return {"ok": true}.'},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=40,
        )
        text = str(response.choices[0].message.content or "") if getattr(response, "choices", None) else ""
        return bool(_json_candidates(text)), text[:300]
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


async def call_structured_json(
    client: Any,
    semaphore: asyncio.Semaphore,
    *,
    model: str,
    system_prompt: str,
    prompt_builder: Callable[[bool], str],
    validator: Callable[[Mapping[str, Any]], Dict[str, Any]],
    max_tokens: int,
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

    # ASU-hosted models can occasionally echo a JSON input envelope when
    # response_format=json_object is enabled for a deeply nested schema.
    # Start with the configured mode, then fall back to prompt-only JSON after
    # the first structured-format failure.
    disable_json_mode_after_format_error = False

    for attempt in range(retries + 1):
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_builder(attempt > 0)},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        use_json_this_attempt = (
            bool(use_json_mode) and not disable_json_mode_after_format_error
        )
        if use_json_this_attempt:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            async with semaphore:
                response = await client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_category = _classify_api_exception(exc)
            last_message = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                await asyncio.sleep(min(20.0, 1.5 * (2 ** attempt)))
                continue
            raise StructuredCallFailure(
                f"ASU call failed after {attempt + 1} attempts: {last_message}",
                category=last_category,
                usage=total_usage,
                attempt_count=attempt + 1,
            ) from exc

        if not getattr(response, "choices", None):
            last_category = "empty_response"
            last_message = "ASU Chat Completions returned no choices"
            if attempt < retries:
                continue
            raise StructuredCallFailure(
                last_message,
                category=last_category,
                usage=total_usage,
                attempt_count=attempt + 1,
            )

        usage = usage_from_response(response)
        _add_usage(total_usage, usage)
        choice = response.choices[0]
        raw_text = str(choice.message.content or "").strip()
        finish_reason = str(getattr(choice, "finish_reason", "") or "")
        request_id = str(getattr(response, "_request_id", "") or "")
        last_raw, last_finish, last_request_id = raw_text, finish_reason, request_id

        try:
            parsed = extract_json_object(raw_text)
            validated = validator(parsed)
            return validated, raw_text, total_usage, request_id, {
                "parse_method": "json_object",
                "finish_reason": finish_reason,
                "attempt_count": attempt + 1,
                "json_mode": bool(use_json_this_attempt),
                "json_mode_fallback_used": bool(
                    use_json_mode and not use_json_this_attempt
                ),
            }
        except Exception as exc:
            last_category = "output_truncated" if finish_reason.lower() == "length" else "output_format"
            last_message = f"{type(exc).__name__}: {exc}"

            # Preserve JSON mode for true truncation; for schema/format errors,
            # retry without response_format so the model is less likely to echo
            # the JSON input envelope. The local parser and validator still
            # enforce valid JSON and the exact schema.
            if last_category == "output_format":
                disable_json_mode_after_format_error = True

            if attempt < retries:
                await asyncio.sleep(min(5.0, 0.5 * (attempt + 1)))
                continue

    raise StructuredCallFailure(
        f"Structured output failed after {retries + 1} attempts: {last_message}",
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
    strata = labels.loc[indices]
    use_stratify = len(strata.value_counts()) > 1 and strata.value_counts().min() >= 2
    try:
        left, right = train_test_split(
            indices,
            train_size=left_size,
            random_state=seed,
            stratify=strata if use_stratify else None,
        )
    except ValueError:
        left, right = train_test_split(indices, train_size=left_size, random_state=seed)
    return np.asarray(left, dtype=int), np.asarray(right, dtype=int)


def create_fallback_split(raw: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    valid = []
    for source in raw.index:
        actual = target_to_binary(raw.loc[source, TARGET])
        if actual is not None:
            valid.append({"source_row_index": int(source), "actual": int(actual)})
    frame = pd.DataFrame(valid)
    ratios = np.array([args.memory_ratio, args.calibration_ratio, args.test_ratio], dtype=float)
    ratios /= ratios.sum()
    n = len(frame)
    n_memory = max(1, int(round(n * ratios[0])))
    n_calibration = max(1, int(round(n * ratios[1]))) if n >= 3 else 0
    if n_memory + n_calibration >= n:
        n_calibration = max(0, n - n_memory - 1)
    indices = np.arange(n, dtype=int)
    memory, remaining = safe_split(indices, frame["actual"], n_memory, args.random_seed)
    calibration, test = safe_split(remaining, frame["actual"], n_calibration, args.random_seed + 1)
    split = np.full(n, "", dtype=object)
    split[memory] = "memory"
    split[calibration] = "calibration"
    split[test] = "test"
    frame["split"] = split
    return frame


def load_reference_split(path: Path, raw: pd.DataFrame) -> pd.DataFrame:
    reference = pd.read_csv(path)
    if {"source_index", "phase"}.issubset(reference.columns):
        index_column, split_column = "source_index", "phase"
    elif {"source_row_index", "split"}.issubset(reference.columns):
        index_column, split_column = "source_row_index", "split"
    else:
        raise ValueError(
            f"Unsupported reference split {path}. Expected source_index+phase or source_row_index+split."
        )
    output = reference[[index_column, split_column]].copy()
    output.columns = ["source_row_index", "split"]
    output["source_row_index"] = pd.to_numeric(output["source_row_index"], errors="raise").astype(int)
    output["split"] = output["split"].astype(str).str.lower()
    output = output[output["split"].isin(["memory", "calibration", "test"])].copy()
    if output["source_row_index"].duplicated().any():
        raise ValueError(f"Duplicate source indices in {path}")
    if not set(output["source_row_index"]).issubset(set(raw.index)):
        raise ValueError(f"Reference split contains source indices absent from adult24.csv: {path}")
    output["actual"] = [target_to_binary(raw.loc[i, TARGET]) for i in output["source_row_index"]]
    return output[output["actual"].isin([0, 1])].reset_index(drop=True)


def downsample_assignments(assignments: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size <= 0 or sample_size >= len(assignments):
        return assignments.copy()
    fraction = sample_size / len(assignments)
    allocations: List[List[Any]] = []
    used = 0
    for key, group in assignments.groupby(["split", "actual"], sort=True):
        n = min(len(group), max(1, int(round(len(group) * fraction))))
        allocations.append([key, n, group])
        used += n
    while used > sample_size:
        changed = False
        for item in reversed(allocations):
            if item[1] > 1:
                item[1] -= 1
                used -= 1
                changed = True
                if used == sample_size:
                    break
        if not changed:
            break
    while used < sample_size:
        changed = False
        for item in allocations:
            if item[1] < len(item[2]):
                item[1] += 1
                used += 1
                changed = True
                if used == sample_size:
                    break
        if not changed:
            break
    pieces = [group.sample(n=n, random_state=seed + j) for j, (_, n, group) in enumerate(allocations)]
    return pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def prepare_assignments(
    variant: str,
    reference_path: Optional[Path],
    raw: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, bool]:
    """Prepare target assignments and the full support-selection pool.

    ``assignments`` may be downsampled for smoke tests via ``--sample-size``.
    ``support_pool`` always preserves the complete reference split so that a
    fixed 8-shot support JSON remains valid even when the target evaluation
    rows are downsampled. Support respondents are demonstrations, not target
    rows, so they do not need to be included in the smoke-test target sample.
    """
    if reference_path and reference_path.exists():
        full_assignments = load_reference_split(reference_path, raw)
        source = str(reference_path)
        exact = True
    else:
        if not args.allow_fallback_split:
            raise FileNotFoundError(
                f"{variant.upper()} reference split is missing. Provide it or pass --allow-fallback-split."
            )
        full_assignments = create_fallback_split(raw, args)
        source = "fallback_outcome_stratified_40_20_40"
        exact = False

    # Use a downsampled copy only for calibration/test targets in dry/smoke runs.
    assignments = downsample_assignments(
        full_assignments, args.sample_size, args.random_seed
    )

    def _ordered(frame: pd.DataFrame, *, add_data_idx: bool) -> pd.DataFrame:
        split_order = pd.Categorical(
            frame["split"],
            categories=["memory", "calibration", "test"],
            ordered=True,
        )
        ordered = (
            frame.assign(_order=split_order)
            .sort_values(["_order", "source_row_index"])
            .drop(columns="_order")
            .reset_index(drop=True)
        )
        if add_data_idx:
            ordered["data_idx"] = np.arange(len(ordered), dtype=int)
        return ordered

    assignments = _ordered(assignments, add_data_idx=True)
    support_pool = _ordered(full_assignments, add_data_idx=False)
    return assignments, support_pool, source, exact


def select_random_balanced_support(
    assignments: pd.DataFrame,
    raw: pd.DataFrame,
    variant: str,
    feature_columns: Sequence[str],
    seed: int,
    support_json: Optional[Path],
) -> List[Dict[str, Any]]:
    memory = assignments[assignments["split"] == "memory"].copy()
    memory_lookup = {
        int(row.source_row_index): int(row.actual)
        for row in memory.itertuples(index=False)
    }

    if support_json and support_json.exists():
        loaded = json.loads(support_json.read_text(encoding="utf-8"))
        if not isinstance(loaded, list) or len(loaded) != 8:
            raise ValueError(f"Support JSON must contain exactly eight examples: {support_json}")
        selected = []
        for item in loaded:
            source = int(item["source_row_index"])
            if source not in memory_lookup:
                raise ValueError(f"Support source {source} is not in the memory split")
            label = int(item.get("label", memory_lookup[source]))
            if label != memory_lookup[source]:
                raise ValueError(f"Support label mismatch for source {source}")
            selected.append({"source_row_index": source, "label": label})
        counts = Counter(item["label"] for item in selected)
        if counts != Counter({0: 4, 1: 4}):
            raise ValueError(f"Support JSON must be 4 YES + 4 NO, found {dict(counts)}")
    else:
        parts = []
        for label in [0, 1]:
            group = memory[memory["actual"] == label]
            if len(group) < 4:
                raise ValueError(f"Need four memory examples for label {label}, found {len(group)}")
            parts.append(group.sample(n=4, random_state=seed + label))
        chosen = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed + 77)
        selected = [
            {"source_row_index": int(row.source_row_index), "label": int(row.actual)}
            for row in chosen.itertuples(index=False)
        ]

    feature_order = [FEATURE_NAMES.get(c, c) for c in feature_columns]
    output = []
    for item in selected:
        source = int(item["source_row_index"])
        output.append({
            "source_row_index": source,
            "label": int(item["label"]),
            "features": build_profile(raw.loc[source], variant, feature_columns),
            "values": compact_values(raw.loc[source], variant, feature_columns),
            "feature_order": feature_order,
        })
    return output


def binary_metrics(y: np.ndarray, probabilities_100: np.ndarray, threshold: float) -> Dict[str, Any]:
    if len(y) == 0:
        return {}
    probabilities_01 = np.clip(probabilities_100 / 100.0, 1e-6, 1 - 1e-6)
    predicted = (probabilities_100 >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    output: Dict[str, Any] = {
        "threshold": float(threshold),
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "brier": float(brier_score_loss(y, probabilities_01)),
        "log_loss": float(log_loss(y, probabilities_01, labels=[0, 1])),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
    if len(np.unique(y)) > 1:
        output["roc_auc"] = float(roc_auc_score(y, probabilities_01))
        output["average_precision"] = float(average_precision_score(y, probabilities_01))
    else:
        output["roc_auc"] = None
        output["average_precision"] = None
    return output


def calibrate_threshold(entries: Sequence[Mapping[str, Any]], metric: str) -> Tuple[float, pd.DataFrame]:
    successful = [entry for entry in entries if entry.get("status") == "ok"]
    if not successful:
        raise RuntimeError("No successful calibration predictions")
    y = np.array([int(entry["actual"]) for entry in successful], dtype=int)
    probabilities = np.array([float(entry["probability_yes"]) for entry in successful], dtype=float)
    rows = [binary_metrics(y, probabilities, float(threshold)) for threshold in np.arange(5, 96, 1)]
    table = pd.DataFrame(rows)
    best = table.sort_values([metric, "log_loss"], ascending=[False, True]).iloc[0]
    return float(best["threshold"]), table


def phase_diagnostics(entries: Sequence[Mapping[str, Any]], expected_n: int) -> Dict[str, Any]:
    unique = {int(entry["data_idx"]): entry for entry in entries if "data_idx" in entry}
    successful = [entry for entry in unique.values() if entry.get("status") == "ok"]
    errors = [entry for entry in unique.values() if entry.get("status") == "error"]
    return {
        "expected_n": int(expected_n),
        "logged_unique_n": int(len(unique)),
        "ok_n": int(len(successful)),
        "error_n": int(len(errors)),
        "missing_n": int(max(0, expected_n - len(unique))),
        "success_rate": float(len(successful) / expected_n) if expected_n else 1.0,
        "failure_categories": dict(Counter(entry.get("failure_category", "unknown") for entry in errors)),
        "finish_reasons": dict(Counter(entry.get("finish_reason", "") for entry in errors)),
        "attempt_counts": dict(Counter(str(entry.get("attempt_count", 0)) for entry in unique.values())),
    }


def persona_diagnostics(entries: Sequence[Mapping[str, Any]], expected_n: int) -> Dict[str, Any]:
    unique = {int(entry["data_idx"]): entry for entry in entries if "data_idx" in entry}
    successful = [entry for entry in unique.values() if entry.get("status") == "ok"]
    errors = [entry for entry in unique.values() if entry.get("status") == "error"]
    return {
        "expected_n": int(expected_n),
        "logged_unique_n": int(len(unique)),
        "ok_n": int(len(successful)),
        "error_n": int(len(errors)),
        "missing_n": int(max(0, expected_n - len(unique))),
        "success_rate": float(len(successful) / expected_n) if expected_n else 1.0,
        "failure_categories": dict(Counter(entry.get("failure_category", "unknown") for entry in errors)),
    }


def prediction_dataframe(entries: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for entry in entries:
        if entry.get("status") != "ok":
            continue
        rows.append({
            "data_idx": entry["data_idx"],
            "source_row_index": entry["source_row_index"],
            "phase": entry["phase"],
            "actual": entry["actual"],
            "probability_yes": entry["probability_yes"],
            "prediction_at_50": entry["prediction_at_50"],
            "raw_prediction": entry.get("raw_prediction", ""),
            "attempt_count": entry.get("attempt_count", 0),
            "finish_reason": entry.get("finish_reason", ""),
            "input_tokens": (entry.get("usage") or {}).get("input_tokens", 0),
            "output_tokens": (entry.get("usage") or {}).get("output_tokens", 0),
            "total_tokens": (entry.get("usage") or {}).get("total_tokens", 0),
            "structured_output": json.dumps(entry.get("structured_output", {}), ensure_ascii=False),
        })
    return pd.DataFrame(rows)


def persona_dataframe(entries: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for entry in entries:
        if entry.get("status") != "ok":
            continue
        rows.append({
            "data_idx": entry["data_idx"],
            "source_row_index": entry["source_row_index"],
            "phase": entry["phase"],
            "persona": json.dumps(entry.get("persona", {}), ensure_ascii=False),
            "attempt_count": entry.get("attempt_count", 0),
            "input_tokens": (entry.get("usage") or {}).get("input_tokens", 0),
            "output_tokens": (entry.get("usage") or {}).get("output_tokens", 0),
            "total_tokens": (entry.get("usage") or {}).get("total_tokens", 0),
        })
    return pd.DataFrame(rows)


def enforce_coverage(name: str, diagnostics: Mapping[str, Any], minimum: float) -> None:
    rate = float(diagnostics.get("success_rate", 0.0))
    if rate < minimum:
        raise RuntimeError(f"{name} coverage {rate:.2%} is below required {minimum:.2%}.")


async def run_bounded_jobs(
    *,
    jobs: Sequence[Any],
    key_fn: Callable[[Any], int],
    latest: Dict[int, Dict[str, Any]],
    run_one: Callable[[Any], Awaitable[Dict[str, Any]]],
    log_path: Path,
    label: str,
    workers: int,
    progress_every: int,
) -> Dict[int, Dict[str, Any]]:
    pending = [job for job in jobs if latest.get(key_fn(job), {}).get("status") != "ok"]
    reused = len(jobs) - len(pending)
    print(f"\n[{label}] total={len(jobs):,} reused_ok={reused:,} calls={len(pending):,}", flush=True)
    if not pending:
        return latest

    queue: asyncio.Queue[Any] = asyncio.Queue()
    for job in pending:
        queue.put_nowait(job)
    worker_count = min(max(1, workers), len(pending))
    for _ in range(worker_count):
        queue.put_nowait(None)

    write_lock = asyncio.Lock()
    state_lock = asyncio.Lock()
    start = time.time()
    state = {"completed": 0, "ok": 0, "error": 0}

    async def worker() -> None:
        while True:
            job = await queue.get()
            try:
                if job is None:
                    return
                entry = await run_one(job)
                key = key_fn(job)
                async with write_lock:
                    append_jsonl(log_path, entry)
                    latest[key] = entry
                async with state_lock:
                    state["completed"] += 1
                    if entry.get("status") == "ok":
                        state["ok"] += 1
                    else:
                        state["error"] += 1
                    completed = state["completed"]
                    if completed == 1 or completed % max(1, progress_every) == 0 or completed == len(pending):
                        elapsed = max(time.time() - start, 1e-9)
                        rate = completed / elapsed
                        eta_seconds = (len(pending) - completed) / rate if rate else math.inf
                        eta = (
                            f"{eta_seconds / 60:.1f} min"
                            if np.isfinite(eta_seconds) and eta_seconds < 3600
                            else f"{eta_seconds / 3600:.2f} h"
                        )
                        print(
                            f"[{label}] {completed:,}/{len(pending):,} new complete | "
                            f"ok={state['ok']:,} err={state['error']:,} | "
                            f"{rate:.2f} calls/s | ETA={eta}",
                            flush=True,
                        )
            finally:
                queue.task_done()

    tasks = [asyncio.create_task(worker()) for _ in range(worker_count)]
    await queue.join()
    await asyncio.gather(*tasks)
    return latest


def make_error_entry(
    *,
    run_hash: str,
    base: Mapping[str, Any],
    exc: Exception,
) -> Dict[str, Any]:
    if isinstance(exc, StructuredCallFailure):
        return {
            **base,
            "config_hash": run_hash,
            "created_at": utc_now(),
            "status": "error",
            "error_type": type(exc).__name__,
            "failure_category": exc.category,
            "error_message": str(exc),
            "attempt_count": exc.attempt_count,
            "finish_reason": exc.finish_reason,
            "usage": exc.usage,
            "request_id": exc.request_id,
            "last_raw_response": exc.raw_response,
        }
    return {
        **base,
        "config_hash": run_hash,
        "created_at": utc_now(),
        "status": "error",
        "error_type": type(exc).__name__,
        "failure_category": _classify_api_exception(exc),
        "error_message": str(exc),
        "attempt_count": 0,
        "finish_reason": "",
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }


def experiment_directory(output_dir: Path, variant: str, model: str, method: str) -> Path:
    return output_dir / variant / slugify(model) / method


def stable_experiment_config(
    *,
    variant: str,
    model: str,
    method: str,
    assignments: pd.DataFrame,
    split_source: str,
    exact_split: bool,
    feature_columns: Sequence[str],
    support: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    json_mode_supported: bool,
) -> Dict[str, Any]:
    return {
        "version": VERSION,
        "prompt_version": PROMPT_VERSION,
        "variant": variant,
        "model": model,
        "method": method,
        "split_source": split_source,
        "exact_matched_split": exact_split,
        "selected_source_rows_sha256": hashlib.sha256(
            json.dumps(assignments["source_row_index"].tolist()).encode()
        ).hexdigest(),
        "feature_columns": list(feature_columns),
        "support_source_rows": [int(item["source_row_index"]) for item in support],
        "support_labels": [int(item["label"]) for item in support],
        "random_seed": args.random_seed,
        "threshold_metric": args.threshold_metric,
        "temperature": args.temperature,
        "max_tokens_copb_demo": args.max_tokens_copb_demo,
        "max_tokens_copb_prediction": args.max_tokens_copb_prediction,
        "max_tokens_persona": args.max_tokens_persona,
        "max_tokens_pbj_prediction": args.max_tokens_pbj_prediction,
        "json_mode_supported": bool(json_mode_supported),
        "include_sensitive_context": args.include_sensitive_context,
    }


def write_or_validate_config(exp_dir: Path, stable_config: Mapping[str, Any], overwrite: bool) -> Tuple[str, Dict[str, Any]]:
    run_hash = config_hash(stable_config)
    config_path = exp_dir / "run_config.json"
    if config_path.exists() and not overwrite:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("config_hash") != run_hash:
            raise RuntimeError(f"Configuration mismatch in existing output directory: {exp_dir}")
        created_at = previous.get("created_at", utc_now())
    else:
        created_at = utc_now()
    run_config = {**stable_config, "config_hash": run_hash, "created_at": created_at}
    config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_hash, run_config


async def ensure_copb_support_reasoning(
    *,
    support: Sequence[Mapping[str, Any]],
    variant: str,
    model: str,
    run_hash: str,
    exp_dir: Path,
    args: argparse.Namespace,
    client: Any,
    semaphore: asyncio.Semaphore,
    use_json_mode: bool,
) -> List[Dict[str, Any]]:
    log_path = exp_dir / "logs" / "support_reasoning.jsonl"
    latest = load_latest_jsonl(log_path, run_hash, key_field="source_row_index")

    async def run_one(item: Mapping[str, Any]) -> Dict[str, Any]:
        source = int(item["source_row_index"])
        label = int(item["label"])
        base = {"source_row_index": source, "actual": label, "variant": variant, "model": model, "phase": "support_reasoning"}
        try:
            system = COPB_DEMO_SYSTEM.format(stage2_description=copb_stage2_description(variant))
            validated, raw_text, usage, request_id, meta = await call_structured_json(
                client,
                semaphore,
                model=model,
                system_prompt=system,
                prompt_builder=lambda retry: build_copb_demo_prompt(item["features"], label, variant, retry),
                validator=lambda obj: validate_copb_reasoning(obj, variant, require_prediction=False),
                max_tokens=args.max_tokens_copb_demo,
                temperature=args.temperature,
                retries=args.max_retries,
                use_json_mode=use_json_mode,
            )
            return {
                **base,
                "config_hash": run_hash,
                "created_at": utc_now(),
                "status": "ok",
                "structured_reasoning": validated,
                **meta,
                "usage": usage,
                "request_id": request_id,
                "raw_response": raw_text,
            }
        except Exception as exc:
            return make_error_entry(run_hash=run_hash, base=base, exc=exc)

    await run_bounded_jobs(
        jobs=list(support),
        key_fn=lambda item: int(item["source_row_index"]),
        latest=latest,
        run_one=run_one,
        log_path=log_path,
        label=f"{variant}/{slugify(model)}/hbm_copb/support_reasoning",
        workers=args.concurrent_samples,
        progress_every=1,
    )
    entries = [latest[int(item["source_row_index"])] for item in support if int(item["source_row_index"]) in latest]
    successful = [entry for entry in entries if entry.get("status") == "ok"]
    if len(successful) != len(support):
        raise RuntimeError(f"HBM-CoPB requires all 8 support rationales; successful={len(successful)}/8")
    output = []
    lookup = {int(entry["source_row_index"]): entry for entry in successful}
    for item in support:
        entry = lookup[int(item["source_row_index"])]
        output.append({
            "source_row_index": int(item["source_row_index"]),
            "label": int(item["label"]),
            "values": item["values"],
            "reasoning": entry["structured_reasoning"],
        })
    (exp_dir / "support_reasoning_demonstrations.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


async def run_copb_prediction_phase(
    *,
    phase: str,
    phase_df: pd.DataFrame,
    raw: pd.DataFrame,
    variant: str,
    feature_columns: Sequence[str],
    demos: Sequence[Mapping[str, Any]],
    model: str,
    run_hash: str,
    exp_dir: Path,
    args: argparse.Namespace,
    client: Any,
    semaphore: asyncio.Semaphore,
    use_json_mode: bool,
) -> List[Dict[str, Any]]:
    log_path = exp_dir / "logs" / f"{phase}_predictions.jsonl"
    latest = load_latest_jsonl(log_path, run_hash, key_field="data_idx")
    feature_order = [FEATURE_NAMES.get(c, c) for c in feature_columns]
    jobs = [row._asdict() for row in phase_df.itertuples(index=False)]

    async def run_one(job: Mapping[str, Any]) -> Dict[str, Any]:
        data_idx = int(job["data_idx"])
        source = int(job["source_row_index"])
        actual = int(job["actual"])
        base = {
            "data_idx": data_idx,
            "source_row_index": source,
            "actual": actual,
            "variant": variant,
            "model": model,
            "method": "hbm_copb",
            "phase": phase,
        }
        try:
            target_values = compact_values(raw.loc[source], variant, feature_columns)
            validated, raw_text, usage, request_id, meta = await call_structured_json(
                client,
                semaphore,
                model=model,
                system_prompt=COPB_PREDICTION_SYSTEM,
                prompt_builder=lambda retry: build_copb_prediction_prompt(
                    feature_order, demos, target_values, variant, retry
                ),
                validator=lambda obj: validate_copb_reasoning(obj, variant, require_prediction=True),
                max_tokens=args.max_tokens_copb_prediction,
                temperature=args.temperature,
                retries=args.max_retries,
                use_json_mode=use_json_mode,
            )
            prediction_fields = {
                key: validated[key]
                for key in ["probability_yes", "raw_prediction", "prediction_at_50", "prediction_consistency_corrected"]
            }
            structured_output = {
                key: value
                for key, value in validated.items()
                if key not in prediction_fields
            }
            return {
                **base,
                "config_hash": run_hash,
                "created_at": utc_now(),
                "status": "ok",
                **prediction_fields,
                "structured_output": structured_output,
                **meta,
                "usage": usage,
                "request_id": request_id,
                "raw_response": raw_text,
            }
        except Exception as exc:
            return make_error_entry(run_hash=run_hash, base=base, exc=exc)

    await run_bounded_jobs(
        jobs=jobs,
        key_fn=lambda job: int(job["data_idx"]),
        latest=latest,
        run_one=run_one,
        log_path=log_path,
        label=f"{variant}/{slugify(model)}/hbm_copb/{phase}",
        workers=args.concurrent_samples,
        progress_every=args.progress_every,
    )
    return [latest[int(job["data_idx"])] for job in jobs if int(job["data_idx"]) in latest]


async def run_copb_experiment(
    *,
    variant: str,
    model: str,
    assignments: pd.DataFrame,
    raw: pd.DataFrame,
    feature_columns: Sequence[str],
    support: Sequence[Mapping[str, Any]],
    split_source: str,
    exact_split: bool,
    json_mode_supported: bool,
    args: argparse.Namespace,
    client: Any,
    semaphore: asyncio.Semaphore,
    output_dir: Path,
) -> Dict[str, Any]:
    exp_dir = experiment_directory(output_dir, variant, model, "hbm_copb")
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "support_set.json").write_text(json.dumps(list(support), ensure_ascii=False, indent=2), encoding="utf-8")

    stable = stable_experiment_config(
        variant=variant,
        model=model,
        method="hbm_copb",
        assignments=assignments,
        split_source=split_source,
        exact_split=exact_split,
        feature_columns=feature_columns,
        support=support,
        args=args,
        json_mode_supported=json_mode_supported,
    )
    run_hash, run_config = write_or_validate_config(exp_dir, stable, args.overwrite)
    demos = await ensure_copb_support_reasoning(
        support=support,
        variant=variant,
        model=model,
        run_hash=run_hash,
        exp_dir=exp_dir,
        args=args,
        client=client,
        semaphore=semaphore,
        use_json_mode=json_mode_supported,
    )

    calibration_df = assignments[assignments["split"] == "calibration"].copy()
    test_df = assignments[assignments["split"] == "test"].copy()
    calibration_entries = await run_copb_prediction_phase(
        phase="calibration", phase_df=calibration_df, raw=raw, variant=variant,
        feature_columns=feature_columns, demos=demos, model=model, run_hash=run_hash,
        exp_dir=exp_dir, args=args, client=client, semaphore=semaphore,
        use_json_mode=json_mode_supported,
    )
    calibration_diag = phase_diagnostics(calibration_entries, len(calibration_df))
    (exp_dir / "calibration_diagnostics.json").write_text(
        json.dumps(calibration_diag, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    enforce_coverage("Calibration", calibration_diag, args.min_success_rate)
    threshold, threshold_table = calibrate_threshold(calibration_entries, args.threshold_metric)
    threshold_table.to_csv(exp_dir / "threshold_search.csv", index=False)

    test_entries = await run_copb_prediction_phase(
        phase="test", phase_df=test_df, raw=raw, variant=variant,
        feature_columns=feature_columns, demos=demos, model=model, run_hash=run_hash,
        exp_dir=exp_dir, args=args, client=client, semaphore=semaphore,
        use_json_mode=json_mode_supported,
    )
    test_diag = phase_diagnostics(test_entries, len(test_df))
    (exp_dir / "test_diagnostics.json").write_text(
        json.dumps(test_diag, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    enforce_coverage("Test", test_diag, args.min_success_rate)

    calibration_ok = [entry for entry in calibration_entries if entry.get("status") == "ok"]
    test_ok = [entry for entry in test_entries if entry.get("status") == "ok"]
    prediction_dataframe(calibration_ok).to_csv(exp_dir / "calibration_predictions.csv", index=False)
    prediction_dataframe(test_ok).to_csv(exp_dir / "test_predictions.csv", index=False)

    y_cal = np.array([int(entry["actual"]) for entry in calibration_ok])
    p_cal = np.array([float(entry["probability_yes"]) for entry in calibration_ok])
    y_test = np.array([int(entry["actual"]) for entry in test_ok])
    p_test = np.array([float(entry["probability_yes"]) for entry in test_ok])
    metrics = {
        "calibration_selected": binary_metrics(y_cal, p_cal, threshold),
        "test_at_50": binary_metrics(y_test, p_test, 50.0),
        "test_selected": binary_metrics(y_test, p_test, threshold),
    }
    support_log = load_latest_jsonl(exp_dir / "logs" / "support_reasoning.jsonl", run_hash, key_field="source_row_index")
    usage = sum_usage(list(support_log.values()) + calibration_entries + test_entries)
    summary = {
        "experiment": "hbm_copb",
        "version": VERSION,
        "created_at": utc_now(),
        "variant": variant,
        "model": model,
        "method": "hbm_copb",
        "n_selected": len(assignments),
        "split_sizes": assignments["split"].value_counts().to_dict(),
        "selected_threshold": threshold,
        "metrics": metrics,
        "usage_total": usage,
        "calibration_diagnostics": calibration_diag,
        "test_diagnostics": test_diag,
        "run_config": run_config,
        "design_note": "Theory-structured reasoning only; no deterministic HBM scores, pattern prior, or reflective memory.",
    }
    (exp_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_safe), encoding="utf-8"
    )
    return summary


async def ensure_pbj_support_personas(
    *,
    support: Sequence[Mapping[str, Any]],
    model: str,
    run_hash: str,
    exp_dir: Path,
    args: argparse.Namespace,
    client: Any,
    semaphore: asyncio.Semaphore,
    use_json_mode: bool,
) -> List[Dict[str, Any]]:
    log_path = exp_dir / "logs" / "support_personas.jsonl"
    latest = load_latest_jsonl(log_path, run_hash, key_field="source_row_index")

    async def run_one(item: Mapping[str, Any]) -> Dict[str, Any]:
        source = int(item["source_row_index"])
        base = {"source_row_index": source, "variant": "v4", "model": model, "phase": "support_persona"}
        try:
            persona, raw_text, usage, request_id, meta = await call_structured_json(
                client,
                semaphore,
                model=model,
                system_prompt=PBJ_PERSONA_SYSTEM,
                prompt_builder=lambda retry: build_persona_prompt(item["features"], retry),
                validator=validate_persona,
                max_tokens=args.max_tokens_persona,
                temperature=args.temperature,
                retries=args.max_retries,
                use_json_mode=use_json_mode,
            )
            return {
                **base,
                "config_hash": run_hash,
                "created_at": utc_now(),
                "status": "ok",
                "persona": persona,
                **meta,
                "usage": usage,
                "request_id": request_id,
                "raw_response": raw_text,
            }
        except Exception as exc:
            return make_error_entry(run_hash=run_hash, base=base, exc=exc)

    await run_bounded_jobs(
        jobs=list(support),
        key_fn=lambda item: int(item["source_row_index"]),
        latest=latest,
        run_one=run_one,
        log_path=log_path,
        label=f"v4/{slugify(model)}/hbm_pbj/support_personas",
        workers=args.concurrent_samples,
        progress_every=1,
    )
    entries = [latest[int(item["source_row_index"])] for item in support if int(item["source_row_index"]) in latest]
    successful = [entry for entry in entries if entry.get("status") == "ok"]
    if len(successful) != len(support):
        raise RuntimeError(f"HBM-PB&J requires all 8 support personas; successful={len(successful)}/8")
    lookup = {int(entry["source_row_index"]): entry for entry in successful}
    demos = []
    for item in support:
        source = int(item["source_row_index"])
        demos.append({
            "source_row_index": source,
            "label": int(item["label"]),
            "values": item["values"],
            "persona": lookup[source]["persona"],
        })
    (exp_dir / "support_persona_demonstrations.json").write_text(
        json.dumps(demos, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return demos


async def run_pbj_persona_phase(
    *,
    phase: str,
    phase_df: pd.DataFrame,
    raw: pd.DataFrame,
    feature_columns: Sequence[str],
    model: str,
    run_hash: str,
    exp_dir: Path,
    args: argparse.Namespace,
    client: Any,
    semaphore: asyncio.Semaphore,
    use_json_mode: bool,
) -> List[Dict[str, Any]]:
    log_path = exp_dir / "logs" / f"{phase}_personas.jsonl"
    latest = load_latest_jsonl(log_path, run_hash, key_field="data_idx")
    jobs = [row._asdict() for row in phase_df.itertuples(index=False)]

    async def run_one(job: Mapping[str, Any]) -> Dict[str, Any]:
        data_idx = int(job["data_idx"])
        source = int(job["source_row_index"])
        base = {"data_idx": data_idx, "source_row_index": source, "variant": "v4", "model": model, "phase": phase}
        try:
            profile = build_profile(raw.loc[source], "v4", feature_columns)
            persona, raw_text, usage, request_id, meta = await call_structured_json(
                client,
                semaphore,
                model=model,
                system_prompt=PBJ_PERSONA_SYSTEM,
                prompt_builder=lambda retry: build_persona_prompt(profile, retry),
                validator=validate_persona,
                max_tokens=args.max_tokens_persona,
                temperature=args.temperature,
                retries=args.max_retries,
                use_json_mode=use_json_mode,
            )
            return {
                **base,
                "config_hash": run_hash,
                "created_at": utc_now(),
                "status": "ok",
                "persona": persona,
                **meta,
                "usage": usage,
                "request_id": request_id,
                "raw_response": raw_text,
            }
        except Exception as exc:
            return make_error_entry(run_hash=run_hash, base=base, exc=exc)

    await run_bounded_jobs(
        jobs=jobs,
        key_fn=lambda job: int(job["data_idx"]),
        latest=latest,
        run_one=run_one,
        log_path=log_path,
        label=f"v4/{slugify(model)}/hbm_pbj/{phase}_personas",
        workers=args.concurrent_samples,
        progress_every=args.progress_every,
    )
    return [latest[int(job["data_idx"])] for job in jobs if int(job["data_idx"]) in latest]


async def run_pbj_prediction_phase(
    *,
    phase: str,
    phase_df: pd.DataFrame,
    persona_entries: Sequence[Mapping[str, Any]],
    raw: pd.DataFrame,
    feature_columns: Sequence[str],
    demos: Sequence[Mapping[str, Any]],
    model: str,
    run_hash: str,
    exp_dir: Path,
    args: argparse.Namespace,
    client: Any,
    semaphore: asyncio.Semaphore,
    use_json_mode: bool,
) -> List[Dict[str, Any]]:
    log_path = exp_dir / "logs" / f"{phase}_predictions.jsonl"
    latest = load_latest_jsonl(log_path, run_hash, key_field="data_idx")
    persona_lookup = {
        int(entry["data_idx"]): entry["persona"]
        for entry in persona_entries
        if entry.get("status") == "ok"
    }
    jobs = [row._asdict() for row in phase_df.itertuples(index=False) if int(row.data_idx) in persona_lookup]
    feature_order = [FEATURE_NAMES.get(c, c) for c in feature_columns]

    async def run_one(job: Mapping[str, Any]) -> Dict[str, Any]:
        data_idx = int(job["data_idx"])
        source = int(job["source_row_index"])
        actual = int(job["actual"])
        base = {
            "data_idx": data_idx,
            "source_row_index": source,
            "actual": actual,
            "variant": "v4",
            "model": model,
            "method": "hbm_pbj",
            "phase": phase,
        }
        try:
            target_values = compact_values(raw.loc[source], "v4", feature_columns)
            validated, raw_text, usage, request_id, meta = await call_structured_json(
                client,
                semaphore,
                model=model,
                system_prompt=PBJ_PREDICTION_SYSTEM,
                prompt_builder=lambda retry: build_pbj_prediction_prompt(
                    feature_order, demos, target_values, persona_lookup[data_idx], retry
                ),
                validator=validate_pbj_prediction,
                max_tokens=args.max_tokens_pbj_prediction,
                temperature=args.temperature,
                retries=args.max_retries,
                use_json_mode=use_json_mode,
            )
            prediction_fields = {
                key: validated[key]
                for key in ["probability_yes", "raw_prediction", "prediction_at_50", "prediction_consistency_corrected"]
            }
            return {
                **base,
                "config_hash": run_hash,
                "created_at": utc_now(),
                "status": "ok",
                **prediction_fields,
                "structured_output": {"persona_application": validated["persona_application"]},
                **meta,
                "usage": usage,
                "request_id": request_id,
                "raw_response": raw_text,
            }
        except Exception as exc:
            return make_error_entry(run_hash=run_hash, base=base, exc=exc)

    await run_bounded_jobs(
        jobs=jobs,
        key_fn=lambda job: int(job["data_idx"]),
        latest=latest,
        run_one=run_one,
        log_path=log_path,
        label=f"v4/{slugify(model)}/hbm_pbj/{phase}_predictions",
        workers=args.concurrent_samples,
        progress_every=args.progress_every,
    )
    all_rows = [row._asdict() for row in phase_df.itertuples(index=False)]
    return [latest[int(job["data_idx"])] for job in all_rows if int(job["data_idx"]) in latest]


async def run_pbj_experiment(
    *,
    model: str,
    assignments: pd.DataFrame,
    raw: pd.DataFrame,
    feature_columns: Sequence[str],
    support: Sequence[Mapping[str, Any]],
    split_source: str,
    exact_split: bool,
    json_mode_supported: bool,
    args: argparse.Namespace,
    client: Any,
    semaphore: asyncio.Semaphore,
    output_dir: Path,
) -> Dict[str, Any]:
    exp_dir = experiment_directory(output_dir, "v4", model, "hbm_pbj")
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "support_set.json").write_text(json.dumps(list(support), ensure_ascii=False, indent=2), encoding="utf-8")

    stable = stable_experiment_config(
        variant="v4", model=model, method="hbm_pbj", assignments=assignments,
        split_source=split_source, exact_split=exact_split, feature_columns=feature_columns,
        support=support, args=args, json_mode_supported=json_mode_supported,
    )
    run_hash, run_config = write_or_validate_config(exp_dir, stable, args.overwrite)
    demos = await ensure_pbj_support_personas(
        support=support, model=model, run_hash=run_hash, exp_dir=exp_dir, args=args,
        client=client, semaphore=semaphore, use_json_mode=json_mode_supported,
    )

    calibration_df = assignments[assignments["split"] == "calibration"].copy()
    test_df = assignments[assignments["split"] == "test"].copy()

    calibration_personas = await run_pbj_persona_phase(
        phase="calibration", phase_df=calibration_df, raw=raw, feature_columns=feature_columns,
        model=model, run_hash=run_hash, exp_dir=exp_dir, args=args, client=client,
        semaphore=semaphore, use_json_mode=json_mode_supported,
    )
    calibration_persona_diag = persona_diagnostics(calibration_personas, len(calibration_df))
    (exp_dir / "calibration_persona_diagnostics.json").write_text(
        json.dumps(calibration_persona_diag, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    enforce_coverage("Calibration persona", calibration_persona_diag, args.min_success_rate)
    persona_dataframe(calibration_personas).to_csv(exp_dir / "calibration_personas.csv", index=False)

    calibration_predictions = await run_pbj_prediction_phase(
        phase="calibration", phase_df=calibration_df, persona_entries=calibration_personas,
        raw=raw, feature_columns=feature_columns, demos=demos, model=model, run_hash=run_hash,
        exp_dir=exp_dir, args=args, client=client, semaphore=semaphore,
        use_json_mode=json_mode_supported,
    )
    calibration_prediction_diag = phase_diagnostics(calibration_predictions, len(calibration_df))
    (exp_dir / "calibration_prediction_diagnostics.json").write_text(
        json.dumps(calibration_prediction_diag, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    enforce_coverage("Calibration prediction", calibration_prediction_diag, args.min_success_rate)
    threshold, threshold_table = calibrate_threshold(calibration_predictions, args.threshold_metric)
    threshold_table.to_csv(exp_dir / "threshold_search.csv", index=False)

    test_personas = await run_pbj_persona_phase(
        phase="test", phase_df=test_df, raw=raw, feature_columns=feature_columns,
        model=model, run_hash=run_hash, exp_dir=exp_dir, args=args, client=client,
        semaphore=semaphore, use_json_mode=json_mode_supported,
    )
    test_persona_diag = persona_diagnostics(test_personas, len(test_df))
    (exp_dir / "test_persona_diagnostics.json").write_text(
        json.dumps(test_persona_diag, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    enforce_coverage("Test persona", test_persona_diag, args.min_success_rate)
    persona_dataframe(test_personas).to_csv(exp_dir / "test_personas.csv", index=False)

    test_predictions = await run_pbj_prediction_phase(
        phase="test", phase_df=test_df, persona_entries=test_personas, raw=raw,
        feature_columns=feature_columns, demos=demos, model=model, run_hash=run_hash,
        exp_dir=exp_dir, args=args, client=client, semaphore=semaphore,
        use_json_mode=json_mode_supported,
    )
    test_prediction_diag = phase_diagnostics(test_predictions, len(test_df))
    (exp_dir / "test_prediction_diagnostics.json").write_text(
        json.dumps(test_prediction_diag, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    enforce_coverage("Test prediction", test_prediction_diag, args.min_success_rate)

    calibration_ok = [entry for entry in calibration_predictions if entry.get("status") == "ok"]
    test_ok = [entry for entry in test_predictions if entry.get("status") == "ok"]
    prediction_dataframe(calibration_ok).to_csv(exp_dir / "calibration_predictions.csv", index=False)
    prediction_dataframe(test_ok).to_csv(exp_dir / "test_predictions.csv", index=False)

    y_cal = np.array([int(entry["actual"]) for entry in calibration_ok])
    p_cal = np.array([float(entry["probability_yes"]) for entry in calibration_ok])
    y_test = np.array([int(entry["actual"]) for entry in test_ok])
    p_test = np.array([float(entry["probability_yes"]) for entry in test_ok])
    metrics = {
        "calibration_selected": binary_metrics(y_cal, p_cal, threshold),
        "test_at_50": binary_metrics(y_test, p_test, 50.0),
        "test_selected": binary_metrics(y_test, p_test, threshold),
    }

    support_persona_log = load_latest_jsonl(exp_dir / "logs" / "support_personas.jsonl", run_hash, key_field="source_row_index")
    usage = sum_usage(
        list(support_persona_log.values())
        + calibration_personas
        + calibration_predictions
        + test_personas
        + test_predictions
    )
    summary = {
        "experiment": "hbm_pbj",
        "version": VERSION,
        "created_at": utc_now(),
        "variant": "v4",
        "model": model,
        "method": "hbm_pbj",
        "n_selected": len(assignments),
        "split_sizes": assignments["split"].value_counts().to_dict(),
        "selected_threshold": threshold,
        "metrics": metrics,
        "usage_total": usage,
        "calibration_persona_diagnostics": calibration_persona_diag,
        "calibration_prediction_diagnostics": calibration_prediction_diag,
        "test_persona_diagnostics": test_persona_diag,
        "test_prediction_diagnostics": test_prediction_diag,
        "run_config": run_config,
        "design_note": "V4-only PB&J-style adaptation: label-blind HBM persona construction followed by persona-conditioned prediction.",
    }
    (exp_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_safe), encoding="utf-8"
    )
    return summary


def flatten_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = summary["metrics"]["test_selected"]
    return {
        "variant": summary["variant"],
        "model": summary["model"],
        "method": summary["method"],
        "n_selected": summary["n_selected"],
        "test_n": metrics.get("n"),
        "selected_threshold": summary["selected_threshold"],
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
        "input_tokens": summary["usage_total"].get("input_tokens", 0),
        "output_tokens": summary["usage_total"].get("output_tokens", 0),
        "total_tokens": summary["usage_total"].get("total_tokens", 0),
    }


def resolve_key(explicit: str) -> str:
    key = (
        (explicit or "").strip()
        or os.environ.get("ASU_LLM_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    if not key:
        raise RuntimeError("Set ASU_LLM_API_KEY, OPENAI_API_KEY, or pass --api-key")
    return key


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "model"


def parse_csv_arg(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def normalize_model_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


async def list_available_models(client: Any) -> List[str]:
    try:
        response = await client.models.list()
        return sorted(str(item.id) for item in response.data)
    except Exception as exc:
        print(f"WARNING: could not list ASU models: {type(exc).__name__}: {exc}", flush=True)
        return []


def resolve_model_request(requested: str, available: Sequence[str]) -> Tuple[str, str]:
    if requested in available:
        return requested, "exact"
    normalized = normalize_model_id(requested)
    exact_normalized = [model for model in available if normalize_model_id(model) == normalized]
    if len(exact_normalized) == 1:
        return exact_normalized[0], "normalized_exact"
    lowered = requested.lower()
    candidates = []
    for model in available:
        candidate = model.lower()
        if "llama4" in lowered and "llama4" in candidate and ("scout" in candidate or "17b" in candidate):
            candidates.append(model)
        elif "llama3" in lowered and "70b" in lowered and "llama" in candidate and "70b" in candidate:
            candidates.append(model)
    if candidates:
        return candidates[0], "heuristic_first_of_multiple:" + "|".join(candidates)
    return requested, "not_found_using_requested_id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HBM-CoPB V4/V5 and HBM-PB&J V4 baselines over ASU Chat Completions")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--v4-reference-split", default="")
    parser.add_argument("--v5-reference-split", default="")
    parser.add_argument("--v4-support-json", default="", help="Optional prior random_balanced_8shot.json to reuse exact V4 support rows")
    parser.add_argument("--v5-support-json", default="", help="Optional prior random_balanced_8shot.json to reuse exact V5 support rows")
    parser.add_argument("--allow-fallback-split", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="https://openai.rc.asu.edu/v1")
    parser.add_argument("--models", default="llama4-scout-17b,llama3-70b")
    parser.add_argument("--experiments", default=",".join(EXPERIMENTS))
    parser.add_argument("--sample-size", type=int, default=0, help="0 uses the full matched split")
    parser.add_argument("--memory-ratio", type=float, default=0.40)
    parser.add_argument("--calibration-ratio", type=float, default=0.20)
    parser.add_argument("--test-ratio", type=float, default=0.40)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--include-sensitive-context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--threshold-metric", choices=["balanced_accuracy", "f1", "accuracy"], default="balanced_accuracy")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens-copb-demo", type=int, default=950)
    parser.add_argument("--max-tokens-copb-prediction", type=int, default=1100)
    parser.add_argument("--max-tokens-persona", type=int, default=1000)
    parser.add_argument("--max-tokens-pbj-prediction", type=int, default=500)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-concurrent-requests", type=int, default=1)
    parser.add_argument("--concurrent-samples", type=int, default=1)
    parser.add_argument("--trust-env-proxy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--json-mode", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--min-success-rate", type=float, default=0.995)
    parser.add_argument("--continue-grid-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = parse_csv_arg(args.experiments)
    unknown = set(experiments) - set(EXPERIMENTS)
    if unknown:
        raise ValueError(f"Unknown experiments: {sorted(unknown)}")
    requested_models = parse_csv_arg(args.models)

    header = pd.read_csv(input_path, nrows=0).columns
    missing = [column for column in ALL_REQUIRED_COLUMNS if column not in header]
    if missing:
        raise KeyError(f"adult24.csv is missing required columns: {missing}")
    raw = pd.read_csv(input_path, usecols=ALL_REQUIRED_COLUMNS, low_memory=False)

    need_v4 = any(exp.endswith("_v4") for exp in experiments)
    need_v5 = "hbm_copb_v5" in experiments
    reference_paths = {
        "v4": Path(args.v4_reference_split) if args.v4_reference_split else None,
        "v5": Path(args.v5_reference_split) if args.v5_reference_split else None,
    }
    support_paths = {
        "v4": Path(args.v4_support_json) if args.v4_support_json else None,
        "v5": Path(args.v5_support_json) if args.v5_support_json else None,
    }

    variant_data: Dict[str, Dict[str, Any]] = {}
    for variant, needed in [("v4", need_v4), ("v5", need_v5)]:
        if not needed:
            continue
        feature_columns = feature_columns_for_variant(variant, args.include_sensitive_context)
        assignments, support_pool, split_source, exact = prepare_assignments(
            variant, reference_paths[variant], raw, args
        )
        support = select_random_balanced_support(
            support_pool,
            raw,
            variant,
            feature_columns,
            args.random_seed,
            support_paths[variant],
        )
        variant_root = output_dir / variant
        variant_root.mkdir(parents=True, exist_ok=True)
        assignments.to_csv(variant_root / "split_assignments_used.csv", index=False)
        (variant_root / "support_set_used.json").write_text(
            json.dumps(support, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        variant_data[variant] = {
            "feature_columns": feature_columns,
            "assignments": assignments,
            "split_source": split_source,
            "exact": exact,
            "support": support,
        }
        print(
            f"[{variant}] selected={len(assignments):,} "
            f"split={assignments['split'].value_counts().to_dict()} "
            f"features={len(feature_columns)} support={[x['source_row_index'] for x in support]}",
            flush=True,
        )

    plan_rows = []
    for model in requested_models:
        if "hbm_copb_v4" in experiments:
            n = int(variant_data["v4"]["assignments"]["split"].isin(["calibration", "test"]).sum())
            plan_rows.append({"experiment": "hbm_copb_v4", "model": model, "planned_calls": n + 8})
        if "hbm_copb_v5" in experiments:
            n = int(variant_data["v5"]["assignments"]["split"].isin(["calibration", "test"]).sum())
            plan_rows.append({"experiment": "hbm_copb_v5", "model": model, "planned_calls": n + 8})
        if "hbm_pbj_v4" in experiments:
            n = int(variant_data["v4"]["assignments"]["split"].isin(["calibration", "test"]).sum())
            plan_rows.append({"experiment": "hbm_pbj_v4", "model": model, "planned_calls": 2 * n + 8})
    plan = {
        "experiments": experiments,
        "requested_models": requested_models,
        "runs": plan_rows,
        "planned_calls_total": int(sum(row["planned_calls"] for row in plan_rows)),
        "note": "JSON-mode probes add approximately one request per resolved model.",
    }
    (output_dir / "benchmark_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nBENCHMARK PLAN")
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.plan_only:
        return

    if args.dry_run:
        for variant, data in variant_data.items():
            support = data["support"]
            feature_order = [FEATURE_NAMES.get(c, c) for c in data["feature_columns"]]
            target_row = data["assignments"][data["assignments"]["split"] == "calibration"].iloc[0]
            source = int(target_row["source_row_index"])
            profile = build_profile(raw.loc[source], variant, data["feature_columns"])
            (output_dir / variant / "dry_copb_demo_prompt.txt").write_text(
                build_copb_demo_prompt(support[0]["features"], support[0]["label"], variant, False),
                encoding="utf-8",
            )
            placeholder_reasoning = validate_copb_reasoning(
                {
                    **{
                        "observed_threat": {"level": "moderate", "evidence": ["example"], "interpretation": "example"},
                        ("vaccine_acceptance_or_benefit" if variant == "v4" else "preventive_engagement"): {"level": "moderate", "evidence": ["example"], "interpretation": "example"},
                        "structural_barriers": {"level": "moderate", "evidence": ["example"], "interpretation": "example"},
                        "healthcare_cues": {"level": "moderate", "evidence": ["example"], "interpretation": "example"},
                        "navigation_self_efficacy": {"level": "moderate", "evidence": ["example"], "interpretation": "example"},
                    },
                    "theory_guided_integration": {
                        "factors_increasing_probability": ["example"],
                        "factors_decreasing_probability": ["example"],
                        "uncertainties": ["example"],
                        "summary": "example",
                    },
                    "observed_label": "YES",
                },
                variant,
                require_prediction=False,
            )
            placeholder_demos = [
                {"source_row_index": d["source_row_index"], "label": d["label"], "values": d["values"], "reasoning": placeholder_reasoning}
                for d in support
            ]
            (output_dir / variant / "dry_copb_prediction_prompt.txt").write_text(
                build_copb_prediction_prompt(
                    feature_order,
                    placeholder_demos,
                    compact_values(raw.loc[source], variant, data["feature_columns"]),
                    variant,
                    False,
                ),
                encoding="utf-8",
            )
            if variant == "v4":
                (output_dir / variant / "dry_pbj_persona_prompt.txt").write_text(
                    build_persona_prompt(profile, False), encoding="utf-8"
                )
                placeholder_persona = validate_persona({
                    "observed_health_vulnerability": {"summary": "example", "evidence": ["example"], "uncertainty": "example"},
                    "preventive_orientation": {"summary": "example", "evidence": ["example"], "counter_evidence": ["example"], "uncertainty": "example"},
                    "access_and_barrier_profile": {"summary": "example", "facilitators": ["example"], "constraints": ["example"]},
                    "healthcare_engagement_style": {"summary": "example", "preventive_contacts": ["example"], "reactive_contacts": ["example"], "uncertainty": "example"},
                    "navigation_capacity": {"summary": "example", "supporting_evidence": ["example"], "limitations": ["example"]},
                    "integrated_health_persona": {"persona_statement": "example", "stable_tendencies_supported_by_evidence": ["example"], "important_contradictions": ["example"], "do_not_infer": ["example"]},
                })
                placeholder_pbj_demos = [
                    {"source_row_index": d["source_row_index"], "label": d["label"], "values": d["values"], "persona": placeholder_persona}
                    for d in support
                ]
                (output_dir / variant / "dry_pbj_prediction_prompt.txt").write_text(
                    build_pbj_prediction_prompt(
                        feature_order,
                        placeholder_pbj_demos,
                        compact_values(raw.loc[source], "v4", data["feature_columns"]),
                        placeholder_persona,
                        False,
                    ),
                    encoding="utf-8",
                )
        print("Dry run complete. No API calls were made.")
        return

    if AsyncOpenAI is None:
        raise RuntimeError("Install openai and httpx: pip install -U openai httpx")
    http_client = httpx.AsyncClient(
        trust_env=args.trust_env_proxy,
        timeout=httpx.Timeout(args.timeout, connect=min(30.0, args.timeout)),
    )
    client = AsyncOpenAI(
        api_key=resolve_key(args.api_key),
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=0,
        http_client=http_client,
    )
    semaphore = asyncio.Semaphore(max(1, args.max_concurrent_requests))

    available_models = await list_available_models(client)
    resolved_models = []
    resolution = []
    for requested in requested_models:
        resolved, how = resolve_model_request(requested, available_models)
        resolved_models.append(resolved)
        resolution.append({"requested": requested, "resolved": resolved, "resolution": how})
    (output_dir / "model_resolution.json").write_text(
        json.dumps({"available_models": available_models, "resolution": resolution}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("MODEL RESOLUTION")
    print(json.dumps(resolution, ensure_ascii=False, indent=2))

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
    (output_dir / "json_mode_probe.json").write_text(json.dumps(probe_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for model in resolved_models:
        run_specs: List[Tuple[str, str]] = []
        if "hbm_copb_v4" in experiments:
            run_specs.append(("hbm_copb", "v4"))
        if "hbm_copb_v5" in experiments:
            run_specs.append(("hbm_copb", "v5"))
        if "hbm_pbj_v4" in experiments:
            run_specs.append(("hbm_pbj", "v4"))

        for method, variant in run_specs:
            data = variant_data[variant]
            try:
                if method == "hbm_copb":
                    summary = await run_copb_experiment(
                        variant=variant,
                        model=model,
                        assignments=data["assignments"],
                        raw=raw,
                        feature_columns=data["feature_columns"],
                        support=data["support"],
                        split_source=data["split_source"],
                        exact_split=data["exact"],
                        json_mode_supported=json_mode_by_model[model],
                        args=args,
                        client=client,
                        semaphore=semaphore,
                        output_dir=output_dir,
                    )
                else:
                    summary = await run_pbj_experiment(
                        model=model,
                        assignments=data["assignments"],
                        raw=raw,
                        feature_columns=data["feature_columns"],
                        support=data["support"],
                        split_source=data["split_source"],
                        exact_split=data["exact"],
                        json_mode_supported=json_mode_by_model[model],
                        args=args,
                        client=client,
                        semaphore=semaphore,
                        output_dir=output_dir,
                    )
                results.append(flatten_summary(summary))
                pd.DataFrame(results).to_csv(output_dir / "benchmark_results.csv", index=False)
            except Exception as exc:
                failure = {
                    "created_at": utc_now(),
                    "variant": variant,
                    "model": model,
                    "method": method,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                failures.append(failure)
                pd.DataFrame(failures).to_csv(output_dir / "benchmark_failures.csv", index=False)
                print(f"EXPERIMENT FAILED: {failure}", flush=True)
                if not args.continue_grid_on_error:
                    raise

    if results:
        result_df = pd.DataFrame(results).sort_values(["variant", "model", "method"])
        result_df.to_csv(output_dir / "benchmark_results.csv", index=False)
        print("\nFINAL RESULTS")
        print(result_df.to_string(index=False))

    await client.close()
    await http_client.aclose()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
