#!/usr/bin/env python3
"""
FLARE-VAX V4: SILIC-inspired LLM-guided inverse contextual reward inference.

This script adapts the structure of SILIC to cross-sectional NHIS vaccine data
without claiming sequential IRL. Four observed non-target vaccination decisions
are treated as contextual binary choices. A hierarchical latent reward model is
fit on the V4 memory split, respondent-specific 5-dimensional preventive reward
vectors are inferred without using the influenza target, and an ASU-hosted LLM
performs a concise HBM-inspired CCR prediction of influenza vaccination.

Implemented reward modes
------------------------
1. prior_gradient
   Global hierarchical prior -> numerical MAP/gradient inference.
2. llm_init_gradient
   ASU LLM initialization -> numerical MAP/gradient inference.
3. prior_gradient_llm_update
   Global prior -> numerical inference -> one ASU LLM update -> refinement.
4. llm_init_gradient_llm_update
   ASU LLM initialization -> numerical inference -> one ASU LLM update -> refinement.

Important methodological boundary
---------------------------------
This is an inverse contextual choice / contextual-bandit-style adaptation. NHIS
has no ordered state transitions, so this script does NOT call the procedure a
sequential MDP or Maximum-Entropy IRL. The frozen LLM is never fine-tuned. The
optimized reward vectors are external respondent-level latent parameters.

Default ASU endpoint:
    https://openai.rc.asu.edu/v1

Resume behavior
---------------
Successful JSONL entries are reused. Failed or missing entries are retried when
configuration is unchanged and --overwrite is not supplied.
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
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import httpx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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


VERSION = "flare_vax_silic_v4_asu_v1"
PROMPT_VERSION = "silic_inverse_contextual_reward_hbm_ccr_compact_v1"
UPDATE_PROMPT_VERSION = "silic_inverse_contextual_reward_update_signed_v2"
FINAL_PROMPT_VERSION = "silic_inverse_contextual_reward_hbm_ccr_reward_only_v2"
TARGET = "SHTFLU12M_A"
WEIGHT = "WTFA_A"
ID_COLUMNS = ["HHX", "SRVY_YR", "PSTRAT", "PPSU"]

DIMENSION_NAMES = [
    "threat_responsiveness",
    "preventive_acceptance",
    "barrier_sensitivity",
    "cue_responsiveness",
    "navigation_capacity",
]
DIMENSION_SIGNS = np.array([1.0, 1.0, -1.0, 1.0, 1.0], dtype=np.float32)

REWARD_MODES = [
    "prior_gradient",
    "llm_init_gradient",
    "prior_gradient_llm_update",
    "llm_init_gradient_llm_update",
]

NON_TARGET_VACCINES = [
    "covid",
    "pneumonia",
    "shingles",
    "hepatitis_a",
]
VACCINE_ACTION_COLUMNS = {
    "covid": "SHTCVD191_A",
    "pneumonia": "SHTPNUEV_A",
    "shingles": "SHTSHINGL1_A",
    "hepatitis_a": "SHTHEPA_A",
}
VACCINE_OBSERVATION_WEIGHTS = {
    "covid": 1.0,
    "pneumonia": 1.0,
    "shingles": 1.0,
    "hepatitis_a": 0.5,
}

# Positive anchor loadings. The barrier dimension receives a negative sign in
# the utility through DIMENSION_SIGNS, so a higher barrier_sensitivity reduces
# the propensity to receive a vaccine.
ANCHOR_LOADINGS = np.array(
    [
        [0.65, 1.00, 0.55, 0.80, 0.55],  # COVID
        [1.00, 0.70, 0.50, 0.95, 0.45],  # Pneumonia
        [0.85, 0.80, 0.60, 0.80, 0.60],  # Shingles
        [0.50, 0.70, 0.75, 0.50, 0.70],  # Hepatitis A
    ],
    dtype=np.float32,
)

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
BASE_FEATURE_COLUMNS = sorted(
    set(BACKGROUND_COLUMNS + CHRONIC_VARS + BARRIER_COLUMNS + CONTACT_COLUMNS + NAVIGATION_COLUMNS)
)
V4_FEATURE_COLUMNS = sorted(set(BASE_FEATURE_COLUMNS + V4_VACCINE_COLUMNS))
ALL_REQUIRED_COLUMNS = sorted(set(ID_COLUMNS + [TARGET, WEIGHT] + V4_FEATURE_COLUMNS))

YES_NO_COLUMNS = set(
    CHRONIC_VARS
    + [
        "ANYDIFF_A", "DISAB3_A", "HLTHCOND_A", "HICOV_A", "NOTCOV_A",
        "HINOTYR_A", "RSNHICOST_A", "HISTOPCOST_A", "MEDDL12M_A",
        "MEDNG12M_A", "RXDL12M_A", "RXDG12M_A", "PAYBLL12M_A",
        "PAYNOBLLNW_A", "TRANSPOR_A", "PRDEDUC1_A", "PRDEDUC2_A",
        "WELLNESS_A", "VIRAPP12M_A", "HOSPONGT_A", "ACCSSINT_A",
        "ACCSSHOM_A", "HITLOOK_A", "HITCOMM_A", "HITTEST_A",
        "SHTCVD191_A", "SHTPNUEV_A", "SHTSHINGL1_A", "SHINGRIX3_A",
        "SHTHEPA_A",
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
REGION_MAP = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}
BMI = {1: "underweight", 2: "healthy_weight", 3: "overweight", 4: "obese"}
SMOKING = {1: "every_day", 2: "some_days", 3: "not_at_all", 4: "other_or_former_category"}
EDUCATION = {
    0: "never_attended_or_kindergarten", 1: "grades_1_11", 2: "12th_grade_no_diploma",
    3: "GED", 4: "high_school_graduate", 5: "some_college", 6: "technical_associate",
    7: "academic_associate", 8: "bachelor", 9: "master", 10: "professional_or_doctoral",
}
POVERTY_RATIO = {
    1: "0.00-0.49", 2: "0.50-0.74", 3: "0.75-0.99", 4: "1.00-1.24",
    5: "1.25-1.49", 6: "1.50-1.74", 7: "1.75-1.99", 8: "2.00-2.49",
    9: "2.50-2.99", 10: "3.00-3.49", 11: "3.50-3.99", 12: "4.00-4.49",
    13: "4.50-4.99", 14: "5.00_or_greater",
}
LAST_VISIT = {
    0: "never", 1: "within_past_year", 2: "1_to_under_2_years", 3: "2_to_under_3_years",
    4: "3_to_under_5_years", 5: "5_to_under_10_years", 6: "10_or_more_years",
}
USUAL_PLACE = {1: "one_usual_place", 2: "no_usual_place", 3: "more_than_one_usual_place"}
USUAL_KIND = {
    1: "doctor_office_or_health_center", 2: "urgent_care_or_retail_clinic",
    3: "hospital_emergency_room", 4: "VA_facility", 5: "other_place",
    6: "no_single_place_used_most_often",
}
COMM_DIFFICULTY = {1: "none", 2: "some", 3: "a_lot", 4: "cannot_do_at_all"}


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

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


def config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def configs_match_except(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    ignored_fields: Iterable[str],
) -> bool:
    ignored = set(ignored_fields)
    left_clean = {k: v for k, v in left.items() if k not in ignored}
    right_clean = {k: v for k, v in right.items() if k not in ignored}
    return left_clean == right_clean


def archive_config(path: Path, suffix: str) -> Path:
    backup = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def append_jsonl(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(obj), ensure_ascii=False, default=json_safe) + "\n")


def load_latest_jsonl(path: Path, expected_hash: str, *, key_field: str) -> Dict[int, Dict[str, Any]]:
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
        return code_value(value, REGION_MAP, [1, 2, 3, 4])
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


def build_profile(row: pd.Series, include_sensitive: bool) -> Dict[str, Any]:
    """Full V4 profile used for reward initialization.

    Non-target vaccination variables are intentionally retained here because
    SILIC-style initialization is allowed to observe the expert behavior it is
    trying to reconstruct.
    """
    columns = list(V4_FEATURE_COLUMNS)
    if not include_sensitive:
        columns = [c for c in columns if c not in {"RACEALLP_A", "HISPALLP_A"}]
    return {FEATURE_NAMES.get(c, c): clean_feature_value(c, row.get(c)) for c in columns}


def build_context_only_profile(row: pd.Series, include_sensitive: bool) -> Dict[str, Any]:
    """Non-vaccine context used by the final CCR prediction.

    The optimized reward vector must carry the information extracted from
    non-target vaccine behavior. Supplying the raw vaccine actions again in the
    final prediction would let the LLM bypass the reward inference mechanism.
    """
    columns = list(BASE_FEATURE_COLUMNS)
    if not include_sensitive:
        columns = [c for c in columns if c not in {"RACEALLP_A", "HISPALLP_A"}]
    return {FEATURE_NAMES.get(c, c): clean_feature_value(c, row.get(c)) for c in columns}


def safe_mean(values: Iterable[Optional[float]], default: float = 0.5) -> float:
    usable = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(usable)) if usable else float(default)


def yn01(row: pd.Series, column: str, *, invert: bool = False) -> Optional[float]:
    value = yes_no_value(row.get(column))
    if value is None:
        return None
    out = 1.0 if value else 0.0
    return 1.0 - out if invert else out


def code01(value: Any, mapping: Mapping[int, float]) -> Optional[float]:
    try:
        iv = int(float(value))
        return mapping.get(iv)
    except Exception:
        return None


def proxy_scores(row: pd.Series) -> Dict[str, float]:
    """Construct five observed, non-target-vaccine proxy scores in [-1, 1].

    These are priors for the latent reward model, not measured psychological
    constructs. No non-target vaccination outcome is used in these proxies.
    """
    age = numeric_value(row.get("AGEP_A"), invalid=(97, 98, 99))
    age_risk = None if age is None else float(np.clip((age - 18.0) / 67.0, 0.0, 1.0))
    ph = code01(row.get("PHSTAT_A"), {1: 0.0, 2: 0.15, 3: 0.35, 4: 0.70, 5: 1.0})
    chronic_values = [yn01(row, c) for c in CHRONIC_VARS]
    chronic_count = sum(v == 1.0 for v in chronic_values if v is not None)
    chronic_score = float(np.clip(chronic_count / 5.0, 0.0, 1.0))
    bmi = code01(row.get("BMICAT_A"), {1: 0.25, 2: 0.0, 3: 0.35, 4: 0.70})
    smoking = code01(row.get("SMKCIGST_A"), {1: 1.0, 2: 0.75, 3: 0.0, 4: 0.35})
    threat01 = safe_mean(
        [
            age_risk,
            ph,
            chronic_score,
            yn01(row, "ANYDIFF_A"),
            yn01(row, "DISAB3_A"),
            yn01(row, "HLTHCOND_A"),
            bmi,
            smoking,
        ]
    )

    last_dr = code01(row.get("LASTDR_A"), {0: 0.0, 1: 1.0, 2: 0.65, 3: 0.45, 4: 0.25, 5: 0.10, 6: 0.0})
    wellness_recency = code01(row.get("WELLVIS_A"), {0: 0.0, 1: 1.0, 2: 0.65, 3: 0.45, 4: 0.25, 5: 0.10, 6: 0.0})
    usual_place = code01(row.get("USUALPL_A"), {1: 1.0, 2: 0.0, 3: 0.45})
    preventive01 = safe_mean(
        [
            yn01(row, "WELLNESS_A"),
            wellness_recency,
            last_dr,
            usual_place,
            yn01(row, "HITLOOK_A"),
            yn01(row, "HITCOMM_A"),
            yn01(row, "HITTEST_A"),
            yn01(row, "VIRAPP12M_A"),
        ]
    )

    months_uninsured = numeric_value(row.get("HINOTMYR_A"), invalid=(97, 98, 99))
    months_uninsured01 = None if months_uninsured is None else float(np.clip(months_uninsured / 12.0, 0.0, 1.0))
    bill_worry = code01(row.get("PAYWORRY_A"), {1: 0.0, 2: 0.25, 3: 0.50, 4: 0.75, 5: 1.0})
    communication = code01(row.get("COMDIFF_A"), {1: 0.0, 2: 0.35, 3: 0.70, 4: 1.0})
    barrier01 = safe_mean(
        [
            yn01(row, "HICOV_A", invert=True),
            yn01(row, "NOTCOV_A"),
            yn01(row, "HINOTYR_A"),
            months_uninsured01,
            yn01(row, "RSNHICOST_A"),
            yn01(row, "HISTOPCOST_A"),
            yn01(row, "MEDDL12M_A"),
            yn01(row, "MEDNG12M_A"),
            yn01(row, "RXDL12M_A"),
            yn01(row, "RXDG12M_A"),
            bill_worry,
            yn01(row, "PAYBLL12M_A"),
            yn01(row, "PAYNOBLLNW_A"),
            yn01(row, "TRANSPOR_A"),
            communication,
        ]
    )

    retail = code01(row.get("RETAILHC12MTC_A"), {0: 0.0, 1: 0.35, 2: 0.65, 3: 1.0})
    urgent = code01(row.get("URGCC12MTC_A"), {0: 0.0, 1: 0.35, 2: 0.65, 3: 1.0})
    emergency = code01(row.get("EMERG12MTC_A"), {0: 0.0, 1: 0.35, 2: 0.65, 3: 1.0})
    cue01 = safe_mean(
        [
            last_dr,
            yn01(row, "WELLNESS_A"),
            wellness_recency,
            retail,
            yn01(row, "VIRAPP12M_A"),
            urgent,
            emergency,
            yn01(row, "HOSPONGT_A"),
            usual_place,
        ]
    )

    usual_kind = code01(row.get("USPLKIND_A"), {1: 1.0, 2: 0.55, 3: 0.20, 4: 0.80, 5: 0.45, 6: 0.15})
    navigation01 = safe_mean(
        [
            usual_place,
            usual_kind,
            yn01(row, "ACCSSINT_A"),
            yn01(row, "ACCSSHOM_A"),
            yn01(row, "HITLOOK_A"),
            yn01(row, "HITCOMM_A"),
            yn01(row, "HITTEST_A"),
            None if communication is None else 1.0 - communication,
        ]
    )

    values01 = [threat01, preventive01, barrier01, cue01, navigation01]
    values11 = [float(np.clip(2.0 * value - 1.0, -1.0, 1.0)) for value in values01]
    return dict(zip(DIMENSION_NAMES, values11))


def chronic_risk_present(row: pd.Series) -> bool:
    if any(yes_no_value(row.get(c)) is True for c in CHRONIC_VARS):
        return True
    return any(yes_no_value(row.get(c)) is True for c in ["ANYDIFF_A", "DISAB3_A", "HLTHCOND_A"])


def vaccine_actions(row: pd.Series, eligibility_policy: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return action labels, observation weights, and eligibility notes.

    A weight of zero excludes an action from likelihood/mismatch optimization.
    """
    age = numeric_value(row.get("AGEP_A"), invalid=(97, 98, 99))
    labels: List[float] = []
    weights: List[float] = []
    notes: List[str] = []

    for vaccine in NON_TARGET_VACCINES:
        raw = yes_no_value(row.get(VACCINE_ACTION_COLUMNS[vaccine]))
        if raw is None:
            labels.append(0.0)
            weights.append(0.0)
            notes.append(f"{vaccine}: missing_or_unknown")
            continue

        eligible = True
        uncertain = False
        if vaccine == "pneumonia":
            if age is None:
                eligible = False
                uncertain = True
            else:
                eligible = age >= 65 or chronic_risk_present(row)
        elif vaccine == "shingles":
            if age is None:
                eligible = False
                uncertain = True
            else:
                eligible = age >= 50
        elif vaccine == "hepatitis_a":
            uncertain = True

        base_weight = float(VACCINE_OBSERVATION_WEIGHTS[vaccine])
        if eligibility_policy == "strict":
            obs_weight = base_weight if eligible else 0.0
        else:
            obs_weight = base_weight if eligible else 0.25 * base_weight

        labels.append(1.0 if raw else 0.0)
        weights.append(obs_weight)
        if obs_weight == 0:
            notes.append(f"{vaccine}: excluded_likely_inapplicable")
        elif uncertain:
            notes.append(f"{vaccine}: included_with_eligibility_uncertainty")
        else:
            notes.append(f"{vaccine}: included")

    return np.asarray(labels, dtype=np.float32), np.asarray(weights, dtype=np.float32), notes


# ---------------------------------------------------------------------------
# Split handling
# ---------------------------------------------------------------------------

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


def create_fallback_split(raw: pd.DataFrame, seed: int) -> pd.DataFrame:
    valid = []
    for source in raw.index:
        actual = target_to_binary(raw.loc[source, TARGET])
        if actual is not None:
            valid.append({"source_row_index": int(source), "actual": int(actual)})
    frame = pd.DataFrame(valid)
    n = len(frame)
    n_memory = int(round(n * 0.40))
    n_calibration = int(round(n * 0.20))
    indices = np.arange(n, dtype=int)
    memory, remaining = safe_split(indices, frame["actual"], n_memory, seed)
    calibration, test = safe_split(remaining, frame["actual"], n_calibration, seed + 1)
    split = np.full(n, "", dtype=object)
    split[memory] = "memory"
    split[calibration] = "calibration"
    split[test] = "test"
    frame["split"] = split
    frame["survey_weight"] = [
        float(raw.loc[int(source), WEIGHT]) if pd.notna(raw.loc[int(source), WEIGHT]) else 1.0
        for source in frame["source_row_index"]
    ]
    return frame


def load_reference_split(path: Path, raw: pd.DataFrame) -> pd.DataFrame:
    reference = pd.read_csv(path)
    if {"source_index", "phase"}.issubset(reference.columns):
        index_column, split_column = "source_index", "phase"
    elif {"source_row_index", "split"}.issubset(reference.columns):
        index_column, split_column = "source_row_index", "split"
    else:
        raise ValueError(
            f"Unsupported V4 reference split {path}. Expected source_index+phase or source_row_index+split."
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
    output["survey_weight"] = [json_safe(raw.loc[i, WEIGHT]) for i in output["source_row_index"]]
    return output[output["actual"].isin([0, 1])].reset_index(drop=True)


def downsample_targets(assignments: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size <= 0 or sample_size >= len(assignments):
        return assignments.copy()
    target = assignments[assignments["split"].isin(["calibration", "test"])].copy()
    if sample_size >= len(target):
        return assignments.copy()
    fraction = sample_size / len(target)
    pieces: List[pd.DataFrame] = []
    allocated = 0
    groups: List[Tuple[Tuple[str, int], int, pd.DataFrame]] = []
    for key, group in target.groupby(["split", "actual"], sort=True):
        n = min(len(group), max(1, int(round(len(group) * fraction))))
        groups.append((key, n, group))
        allocated += n
    while allocated > sample_size:
        changed = False
        for k in range(len(groups) - 1, -1, -1):
            key, n, group = groups[k]
            if n > 1:
                groups[k] = (key, n - 1, group)
                allocated -= 1
                changed = True
                if allocated == sample_size:
                    break
        if not changed:
            break
    while allocated < sample_size:
        changed = False
        for k in range(len(groups)):
            key, n, group = groups[k]
            if n < len(group):
                groups[k] = (key, n + 1, group)
                allocated += 1
                changed = True
                if allocated == sample_size:
                    break
        if not changed:
            break
    for j, (_, n, group) in enumerate(groups):
        pieces.append(group.sample(n=n, random_state=seed + j))
    sampled_targets = pd.concat(pieces, ignore_index=True)
    memory = assignments[assignments["split"] == "memory"].copy()
    return pd.concat([memory, sampled_targets], ignore_index=True)


def ordered_assignments(frame: pd.DataFrame) -> pd.DataFrame:
    split_order = pd.Categorical(
        frame["split"], categories=["memory", "calibration", "test"], ordered=True
    )
    output = (
        frame.assign(_order=split_order)
        .sort_values(["_order", "source_row_index"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    output["data_idx"] = np.arange(len(output), dtype=int)
    return output


# ---------------------------------------------------------------------------
# Data matrices
# ---------------------------------------------------------------------------
@dataclass
class ContextualDataset:
    sources: np.ndarray
    x: np.ndarray
    actions: np.ndarray
    action_weights: np.ndarray
    eligibility_notes: List[List[str]]
    actual_flu: np.ndarray
    survey_weights: np.ndarray


def build_contextual_dataset(assignments: pd.DataFrame, raw: pd.DataFrame, eligibility_policy: str) -> ContextualDataset:
    sources: List[int] = []
    x_rows: List[List[float]] = []
    action_rows: List[np.ndarray] = []
    weight_rows: List[np.ndarray] = []
    notes: List[List[str]] = []
    actuals: List[int] = []
    survey_weights: List[float] = []
    for row in assignments.itertuples(index=False):
        source = int(row.source_row_index)
        raw_row = raw.loc[source]
        proxies = proxy_scores(raw_row)
        action, action_weight, action_notes = vaccine_actions(raw_row, eligibility_policy)
        sources.append(source)
        x_rows.append([proxies[name] for name in DIMENSION_NAMES])
        action_rows.append(action)
        weight_rows.append(action_weight)
        notes.append(action_notes)
        actuals.append(int(row.actual))
        survey_weights.append(float(row.survey_weight) if pd.notna(row.survey_weight) else 1.0)
    return ContextualDataset(
        sources=np.asarray(sources, dtype=int),
        x=np.asarray(x_rows, dtype=np.float32),
        actions=np.vstack(action_rows).astype(np.float32),
        action_weights=np.vstack(weight_rows).astype(np.float32),
        eligibility_notes=notes,
        actual_flu=np.asarray(actuals, dtype=int),
        survey_weights=np.asarray(survey_weights, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Hierarchical contextual reward model
# ---------------------------------------------------------------------------

def inverse_softplus(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-4, None)
    return np.log(np.expm1(x))


class HierarchicalRewardModel(nn.Module):
    def __init__(self, n_people: int, n_dim: int = 5, n_vaccines: int = 4) -> None:
        super().__init__()
        self.prior_map = nn.Parameter(torch.zeros(n_dim, n_dim + 1))
        with torch.no_grad():
            self.prior_map[:, 1:] = torch.eye(n_dim)
        self.beta = nn.Parameter(torch.zeros(n_vaccines))
        self.loading_raw = nn.Parameter(torch.tensor(inverse_softplus(ANCHOR_LOADINGS), dtype=torch.float32))
        self.person_residual = nn.Parameter(torch.zeros(n_people, n_dim))
        self.register_buffer("signs", torch.tensor(DIMENSION_SIGNS, dtype=torch.float32))

    def prior_raw(self, x: torch.Tensor) -> torch.Tensor:
        x_aug = torch.cat([torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device), x], dim=1)
        return x_aug @ self.prior_map.T

    def theta(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.tanh(self.prior_raw(x) + self.person_residual)

    def positive_loadings(self) -> torch.Tensor:
        return F.softplus(self.loading_raw)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        theta_effective = self.theta(x) * self.signs
        return self.beta + theta_effective @ self.positive_loadings().T


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def fit_or_load_global_model(
    output_dir: Path,
    memory: ContextualDataset,
    args: argparse.Namespace,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    model_dir = output_dir / "shared" / "global_contextual_reward_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / "model.pt"
    config_path = model_dir / "run_config.json"
    history_path = model_dir / "training_history.csv"

    cfg = {
        "version": VERSION,
        "model_component": "hierarchical_inverse_contextual_choice",
        "memory_sources_sha256": hashlib.sha256(memory.sources.tobytes()).hexdigest(),
        "dimension_names": DIMENSION_NAMES,
        "dimension_signs": DIMENSION_SIGNS.tolist(),
        "vaccine_names": NON_TARGET_VACCINES,
        "anchor_loadings": ANCHOR_LOADINGS.tolist(),
        "eligibility_policy": args.eligibility_policy,
        "global_epochs": args.global_epochs,
        "global_lr": args.global_lr,
        "lambda_person": args.lambda_person,
        "lambda_global": args.lambda_global,
        "lambda_loading_anchor": args.lambda_loading_anchor,
        "learn_vaccine_loadings": args.learn_vaccine_loadings,
        "random_seed": args.random_seed,
    }
    cfg_hash = config_hash(cfg)
    cfg["config_hash"] = cfg_hash

    if config_path.exists() and checkpoint.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("config_hash") != cfg_hash:
            raise RuntimeError(
                f"Configuration mismatch in existing global model directory: {model_dir}. "
                "Use a new output directory or --overwrite."
            )
        state = torch.load(checkpoint, map_location="cpu")
        arrays = {
            "prior_map": state["prior_map"].cpu().numpy(),
            "beta": state["beta"].cpu().numpy(),
            "loadings": F.softplus(state["loading_raw"]).cpu().numpy(),
        }
        return arrays, existing

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    device = choose_device(args.device)
    model = HierarchicalRewardModel(len(memory.sources)).to(device)
    x = torch.tensor(memory.x, dtype=torch.float32, device=device)
    y = torch.tensor(memory.actions, dtype=torch.float32, device=device)
    obs_w = torch.tensor(memory.action_weights, dtype=torch.float32, device=device)

    # Initialize vaccine intercepts from masked memory prevalences.
    with torch.no_grad():
        for j in range(len(NON_TARGET_VACCINES)):
            mask = obs_w[:, j] > 0
            if bool(mask.any()):
                prevalence = float((y[mask, j] * obs_w[mask, j]).sum() / obs_w[mask, j].sum())
                prevalence = float(np.clip(prevalence, 0.02, 0.98))
                model.beta[j] = math.log(prevalence / (1.0 - prevalence))
        if not args.learn_vaccine_loadings:
            model.loading_raw.requires_grad_(False)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.global_lr)
    anchor = torch.tensor(ANCHOR_LOADINGS, dtype=torch.float32, device=device)
    history: List[Dict[str, Any]] = []
    best_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    stale = 0

    for epoch in range(1, args.global_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model.logits(x)
        element = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        likelihood = (element * obs_w).sum() / obs_w.sum().clamp_min(1.0)
        reg_person = args.lambda_person * model.person_residual.pow(2).mean()
        reg_global = args.lambda_global * (
            model.prior_map.pow(2).mean() + model.beta.pow(2).mean()
        )
        loadings = model.positive_loadings()
        reg_anchor = args.lambda_loading_anchor * (loadings - anchor).pow(2).mean()
        loss = likelihood + reg_person + reg_global + reg_anchor
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        value = float(loss.detach().cpu())
        history.append(
            {
                "epoch": epoch,
                "loss": value,
                "likelihood": float(likelihood.detach().cpu()),
                "person_regularization": float(reg_person.detach().cpu()),
                "global_regularization": float(reg_global.detach().cpu()),
                "loading_anchor_regularization": float(reg_anchor.detach().cpu()),
            }
        )
        if value + args.global_min_delta < best_loss:
            best_loss = value
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if args.global_progress_every > 0 and (epoch == 1 or epoch % args.global_progress_every == 0):
            print(f"[global_model] epoch={epoch} loss={value:.6f} best={best_loss:.6f}")
        if stale >= args.global_patience:
            print(f"[global_model] early stopping at epoch {epoch}")
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(best_state, checkpoint)
    pd.DataFrame(history).to_csv(history_path, index=False)
    config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    arrays = {
        "prior_map": best_state["prior_map"].numpy(),
        "beta": best_state["beta"].numpy(),
        "loadings": F.softplus(best_state["loading_raw"]).numpy(),
    }
    pd.DataFrame(arrays["loadings"], index=NON_TARGET_VACCINES, columns=DIMENSION_NAMES).to_csv(
        model_dir / "learned_vaccine_loadings.csv"
    )
    pd.DataFrame(arrays["prior_map"], index=DIMENSION_NAMES, columns=["intercept"] + DIMENSION_NAMES).to_csv(
        model_dir / "learned_prior_map.csv"
    )
    return arrays, cfg


def prior_raw_from_x(x: np.ndarray, prior_map: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([[1.0], x.astype(np.float32)])
    return prior_map @ x_aug


def theta_from_raw(raw_latent: np.ndarray) -> np.ndarray:
    return 2.0 * np.tanh(raw_latent)


def raw_from_theta(theta: np.ndarray) -> np.ndarray:
    scaled = np.clip(theta / 2.0, -0.999, 0.999)
    return np.arctanh(scaled)


def policy_probabilities(theta: np.ndarray, beta: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    effective = theta * DIMENSION_SIGNS
    logits = beta + loadings @ effective
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))


def infer_individual_reward(
    x: np.ndarray,
    actions: np.ndarray,
    action_weights: np.ndarray,
    global_arrays: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    initial_theta: Optional[np.ndarray] = None,
    update_direction: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    prior_map = np.asarray(global_arrays["prior_map"], dtype=np.float32)
    beta = np.asarray(global_arrays["beta"], dtype=np.float32)
    loadings = np.asarray(global_arrays["loadings"], dtype=np.float32)
    mu_raw = prior_raw_from_x(x, prior_map)
    if initial_theta is None:
        u0 = np.zeros(len(DIMENSION_NAMES), dtype=np.float32)
    else:
        u0 = raw_from_theta(np.asarray(initial_theta, dtype=np.float32)) - mu_raw

    device = choose_device(args.device)
    u = torch.tensor(u0, dtype=torch.float32, device=device, requires_grad=True)
    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.float32, device=device)
    weights_t = torch.tensor(action_weights, dtype=torch.float32, device=device)
    mu_t = torch.tensor(mu_raw, dtype=torch.float32, device=device)
    beta_t = torch.tensor(beta, dtype=torch.float32, device=device)
    load_t = torch.tensor(loadings, dtype=torch.float32, device=device)
    signs_t = torch.tensor(DIMENSION_SIGNS, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([u], lr=args.individual_lr)

    def run_steps(steps: int) -> None:
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            theta_t = 2.0 * torch.tanh(mu_t + u)
            logits = beta_t + load_t @ (theta_t * signs_t)
            element = F.binary_cross_entropy_with_logits(logits, actions_t, reduction="none")
            likelihood = (element * weights_t).sum() / weights_t.sum().clamp_min(1.0)
            loss = likelihood + args.individual_lambda * u.pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([u], 5.0)
            optimizer.step()

    run_steps(args.individual_steps)
    with torch.no_grad():
        theta_pre = (2.0 * torch.tanh(mu_t + u)).cpu().numpy()

    if update_direction is not None:
        adjusted = np.clip(
            theta_pre + args.llm_update_scale * np.asarray(update_direction, dtype=np.float32),
            -1.99,
            1.99,
        )
        with torch.no_grad():
            u.copy_(torch.tensor(raw_from_theta(adjusted) - mu_raw, dtype=torch.float32, device=device))
        optimizer = torch.optim.Adam([u], lr=args.individual_lr)
        run_steps(args.post_update_steps)

    with torch.no_grad():
        theta_final = (2.0 * torch.tanh(mu_t + u)).cpu().numpy()
    probabilities = policy_probabilities(theta_final, beta, loadings)
    mismatches = actions - probabilities
    return {
        "theta": theta_final.astype(float).tolist(),
        "theta_by_name": {name: float(theta_final[k]) for k, name in enumerate(DIMENSION_NAMES)},
        "prior_theta": theta_from_raw(mu_raw).astype(float).tolist(),
        "predicted_non_target_probabilities": {
            vaccine: float(probabilities[j]) for j, vaccine in enumerate(NON_TARGET_VACCINES)
        },
        "mismatches": {vaccine: float(mismatches[j]) for j, vaccine in enumerate(NON_TARGET_VACCINES)},
        "n_observed_actions": int(np.sum(action_weights > 0)),
    }


def infer_reward_batch(
    x: np.ndarray,
    actions: np.ndarray,
    action_weights: np.ndarray,
    global_arrays: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    initial_theta: Optional[np.ndarray] = None,
    update_directions: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Vectorized respondent-level MAP inference for a whole phase."""
    if len(x) == 0:
        return []
    prior_map = np.asarray(global_arrays["prior_map"], dtype=np.float32)
    beta = np.asarray(global_arrays["beta"], dtype=np.float32)
    loadings = np.asarray(global_arrays["loadings"], dtype=np.float32)
    x_aug = np.concatenate([np.ones((len(x), 1), dtype=np.float32), x.astype(np.float32)], axis=1)
    mu_raw = x_aug @ prior_map.T
    if initial_theta is None:
        u0 = np.zeros_like(mu_raw, dtype=np.float32)
    else:
        u0 = raw_from_theta(np.asarray(initial_theta, dtype=np.float32)) - mu_raw

    device = choose_device(args.device)
    u = torch.tensor(u0, dtype=torch.float32, device=device, requires_grad=True)
    actions_t = torch.tensor(actions, dtype=torch.float32, device=device)
    weights_t = torch.tensor(action_weights, dtype=torch.float32, device=device)
    mu_t = torch.tensor(mu_raw, dtype=torch.float32, device=device)
    beta_t = torch.tensor(beta, dtype=torch.float32, device=device)
    load_t = torch.tensor(loadings, dtype=torch.float32, device=device)
    signs_t = torch.tensor(DIMENSION_SIGNS, dtype=torch.float32, device=device)

    def optimize(steps: int) -> None:
        optimizer = torch.optim.Adam([u], lr=args.individual_lr)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            theta_t = 2.0 * torch.tanh(mu_t + u)
            logits = beta_t.unsqueeze(0) + (theta_t * signs_t.unsqueeze(0)) @ load_t.T
            element = F.binary_cross_entropy_with_logits(logits, actions_t, reduction="none")
            per_person_den = weights_t.sum(dim=1).clamp_min(1.0)
            per_person_likelihood = (element * weights_t).sum(dim=1) / per_person_den
            per_person_reg = args.individual_lambda * u.pow(2).mean(dim=1)
            loss = (per_person_likelihood + per_person_reg).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([u], 5.0)
            optimizer.step()

    optimize(args.individual_steps)
    if update_directions is not None:
        with torch.no_grad():
            theta_pre = 2.0 * torch.tanh(mu_t + u)
            update_t = torch.tensor(update_directions, dtype=torch.float32, device=device)
            adjusted = torch.clamp(theta_pre + args.llm_update_scale * update_t, -1.99, 1.99)
            adjusted_raw = torch.atanh(torch.clamp(adjusted / 2.0, -0.999, 0.999))
            u.copy_(adjusted_raw - mu_t)
        optimize(args.post_update_steps)

    with torch.no_grad():
        theta = (2.0 * torch.tanh(mu_t + u)).cpu().numpy()
        logits = beta_t.unsqueeze(0) + (torch.tensor(theta, device=device) * signs_t.unsqueeze(0)) @ load_t.T
        probabilities = torch.sigmoid(logits).cpu().numpy()
    prior_theta = 2.0 * np.tanh(mu_raw)
    mismatches = actions - probabilities
    output: List[Dict[str, Any]] = []
    for i in range(len(x)):
        output.append({
            "theta": theta[i].astype(float).tolist(),
            "theta_by_name": {name: float(theta[i, k]) for k, name in enumerate(DIMENSION_NAMES)},
            "prior_theta": prior_theta[i].astype(float).tolist(),
            "predicted_non_target_probabilities": {
                vaccine: float(probabilities[i, j]) for j, vaccine in enumerate(NON_TARGET_VACCINES)
            },
            "mismatches": {
                vaccine: float(mismatches[i, j]) for j, vaccine in enumerate(NON_TARGET_VACCINES)
            },
            "n_observed_actions": int(np.sum(action_weights[i] > 0)),
        })
    return output


# ---------------------------------------------------------------------------
# LLM structured calls
# ---------------------------------------------------------------------------

def usage_from_response(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens = int(
        getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0) or 0
    )
    output_tokens = int(
        getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0) or 0
    )
    total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or 0)
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}


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


def _limited_string(value: Any, field: str, limit: int = 800) -> str:
    text = " ".join(str(value or "").strip().split())
    if not text:
        raise ValueError(f"missing text field: {field}")
    return text[:limit]


def _vector_source(obj: Mapping[str, Any], preferred_key: str) -> Any:
    """Return a mapping or a five-element vector from common model formats."""
    if preferred_key in obj:
        return obj[preferred_key]
    for key in ("theta", "reward_vector", "vector", "update_direction", "direction"):
        if key in obj:
            return obj[key]
    return obj


def _named_numeric_vector(source: Any, *, integer: bool) -> Dict[str, Any]:
    if isinstance(source, Mapping):
        values = [source.get(name) for name in DIMENSION_NAMES]
    elif isinstance(source, (list, tuple)) and len(source) == len(DIMENSION_NAMES):
        values = list(source)
    else:
        raise ValueError(
            f"expected an object with {DIMENSION_NAMES} or a five-element array"
        )

    output: Dict[str, Any] = {}
    for name, raw_value in zip(DIMENSION_NAMES, values):
        if raw_value is None:
            raise ValueError(f"missing vector field: {name}")
        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError(f"non-finite vector field: {name}")
        if integer:
            rounded = int(round(value))
            if rounded not in {-1, 0, 1}:
                raise ValueError(f"update field {name} must be -1, 0, or 1")
            output[name] = rounded
        else:
            output[name] = float(np.clip(value, -2.0, 2.0))
    return output


def validate_llm_initialization(obj: Mapping[str, Any]) -> Dict[str, Any]:
    source = _vector_source(obj, "theta")
    return {"theta": _named_numeric_vector(source, integer=False)}


def validate_llm_update(obj: Mapping[str, Any]) -> Dict[str, Any]:
    source = _vector_source(obj, "update_direction")
    return {"update_direction": _named_numeric_vector(source, integer=True)}


def migrate_compatible_initializations(
    log_path: Path,
    *,
    new_hash: str,
    legacy_hashes: Iterable[str],
    items: Sequence[Mapping[str, Any]],
) -> int:
    """Copy compatible successful v1 initialization rows to the shared v2 hash.

    The initialization prompt and output semantics are unchanged. This migration
    only fixes the old mode-dependent cache key, so matching source rows can be
    safely reused without another ASU call.
    """
    if not log_path.exists():
        return 0
    accepted = set(legacy_hashes)
    expected = {
        int(item["data_idx"]): int(item["source_row_index"])
        for item in items
    }
    existing_new = load_latest_jsonl(log_path, new_hash, key_field="data_idx")
    compatible: Dict[int, Dict[str, Any]] = {}
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                obj = json.loads(line)
                data_idx = int(obj.get("data_idx"))
                if (
                    obj.get("config_hash") not in accepted
                    or obj.get("status") != "ok"
                    or data_idx not in expected
                    or int(obj.get("source_row_index")) != expected[data_idx]
                ):
                    continue
                theta_source = (
                    obj.get("theta_by_name")
                    if isinstance(obj.get("theta_by_name"), Mapping)
                    else obj.get("theta")
                )
                validated = _named_numeric_vector(theta_source, integer=False)
                migrated = dict(obj)
                migrated["config_hash"] = new_hash
                migrated["theta_by_name"] = validated
                migrated["theta"] = [validated[name] for name in DIMENSION_NAMES]
                migrated["migrated_from_config_hash"] = obj.get("config_hash")
                migrated["cache_migration_note"] = (
                    "Reused identical v1 LLM initialization after fixing the "
                    "mode-dependent shared-cache hash."
                )
                compatible[data_idx] = migrated
            except Exception:
                continue

    migrated_count = 0
    for data_idx, obj in compatible.items():
        if data_idx not in existing_new:
            append_jsonl(log_path, obj)
            migrated_count += 1
    return migrated_count


def _signed_level(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "negative": "negative",
        "decreasing": "negative",
        "inverse": "negative",
        "low": "negative",
        "near_zero": "near_zero",
        "neutral": "near_zero",
        "zero": "near_zero",
        "moderate": "near_zero",
        "positive": "positive",
        "increasing": "positive",
        "high": "positive",
        "mixed": "uncertain",
        "uncertain": "uncertain",
        "unknown": "uncertain",
    }
    if normalized not in aliases:
        raise ValueError(
            f"{field} interpretation must be negative, near_zero, positive, or uncertain"
        )
    return aliases[normalized]


def _probability_0_100(value: Any) -> float:
    if value is None:
        raise ValueError("missing probability_yes")
    text = str(value).strip().replace("%", "")
    probability = float(text)
    if not np.isfinite(probability):
        raise ValueError("probability_yes is not finite")
    # Accept a clearly fractional probability while preserving an explicit 1%.
    if 0.0 < probability < 1.0:
        probability *= 100.0
    return float(np.clip(probability, 0.0, 100.0))


def validate_final_prediction(obj: Mapping[str, Any]) -> Dict[str, Any]:
    interpretation = obj.get("latent_reward_interpretation")
    if not isinstance(interpretation, Mapping):
        raise ValueError("latent_reward_interpretation must be an object")

    levels = {
        name: _signed_level(interpretation.get(name), name)
        for name in DIMENSION_NAMES
    }
    probability = _probability_0_100(obj.get("probability_yes"))
    raw_prediction = str(obj.get("prediction", "")).strip().upper()
    prediction = "YES" if probability >= 50 else "NO"
    return {
        "latent_reward_interpretation": levels,
        "integration": _limited_string(obj.get("integration"), "integration", 600),
        "probability_yes": probability,
        "raw_prediction": raw_prediction if raw_prediction in {"YES", "NO"} else "",
        "prediction_at_50": prediction,
        "prediction_consistency_corrected": bool(
            raw_prediction in {"YES", "NO"} and raw_prediction != prediction
        ),
    }


LLM_INIT_SYSTEM = """You initialize a five-dimensional respondent-level preventive reward vector for an inverse contextual choice model.

The vector is external to the language model and is NOT an LLM reward or model parameter. It should summarize latent tendencies that can help reconstruct the respondent's observed non-target vaccination choices.

Dimensions, each restricted to [-2, 2]:
- threat_responsiveness: how strongly observed health vulnerability is associated with preventive action;
- preventive_acceptance: general preventive/vaccine acceptance tendency supported by observed behavior;
- barrier_sensitivity: how strongly structural barriers suppress preventive action; positive means more barrier-sensitive;
- cue_responsiveness: responsiveness to healthcare-contact opportunities;
- navigation_capacity: ability to access and navigate healthcare resources.

Rules:
- Use only the supplied V4 profile and non-target vaccination actions.
- Never infer or mention influenza vaccination.
- Missing, inapplicable, or eligibility-uncertain observations are not refusals.
- These values are latent behavioral parameters, not measured psychological states.
- Return JSON only with a top-level object named theta containing exactly the five numeric fields.
"""

LLM_UPDATE_SYSTEM = """You provide one SILIC-style heuristic update direction for a five-dimensional external preventive reward vector.

You receive the current reward vector and mismatches defined as:
    mismatch = observed action - predicted probability.
A positive mismatch means the learner under-predicted an observed vaccination.
A negative mismatch means the learner over-predicted vaccination.

Utility semantics:
- Increasing threat_responsiveness tends to increase vaccination utility when health-threat context is present.
- Increasing preventive_acceptance tends to increase vaccination utility.
- Increasing barrier_sensitivity tends to DECREASE vaccination utility because barriers become more suppressive.
- Increasing cue_responsiveness tends to increase vaccination utility when healthcare-contact cues are present.
- Increasing navigation_capacity tends to increase vaccination utility when navigation resources are present.

Choose the smallest grounded change that could reduce the largest eligible mismatches. Do not move every dimension in the same direction unless the supplied mismatches clearly require it.

Rules:
- Each direction must be exactly -1, 0, or 1.
- Respect eligibility uncertainty and do not treat missing or likely inapplicable vaccination as refusal.
- Never use or discuss influenza vaccination.
- Do not rewrite the current vector; return only an update direction.
- Return JSON only with a top-level object named update_direction containing exactly the five fields.
"""

FINAL_CCR_SYSTEM = """You implement a concise HBM-inspired Cognitive Chain Reasoning prediction for influenza vaccination.

You receive:
1. an optimized five-dimensional latent preventive reward vector inferred from non-target vaccination choices;
2. the respondent's NON-VACCINE V4 context only.

The raw non-target vaccination actions are intentionally withheld at this stage. The optimized reward vector must be used as the behavioral summary; do not attempt to reconstruct or guess the original actions.

Reward-vector semantics, each on approximately [-2, 2]:
- threat_responsiveness: positive values indicate stronger preventive response to observed health vulnerability; negative values indicate weaker or inverse response.
- preventive_acceptance: positive values indicate stronger general preventive/vaccine propensity.
- barrier_sensitivity: positive values mean structural barriers suppress preventive action more strongly; negative values mean less suppression by barriers.
- cue_responsiveness: positive values indicate stronger response to healthcare-contact opportunities.
- navigation_capacity: positive values indicate stronger capacity-related support for preventive action.
Values near zero indicate weak or uncertain behavioral evidence.

First interpret the signed reward dimensions. Then combine them with current non-vaccine health vulnerability, access/barriers, healthcare contacts, and navigation context to estimate influenza vaccination.

Constraints:
- The latent vector is an external model parameter, not a measured psychological belief or causal effect.
- Do not use HBM8 patterns, pattern base rates, retrieval, reflective memory, correction rules, or training-error feedback.
- Do not invent physician recommendations, reminders, intentions, trust, or private attitudes.
- Missing/unknown/not applicable is not refusal.
- Return compact JSON only.
"""

FORMAT_RETRY = """
FORMAT RETRY. Do not repeat the input envelope. Return a new JSON object that follows the requested schema exactly. No Markdown and no prose outside JSON.
"""


def build_init_prompt(
    profile: Mapping[str, Any],
    actions: Mapping[str, Optional[int]],
    notes: Sequence[str],
    strict_retry: bool,
) -> str:
    payload = {
        "raw_v4_profile": profile,
        "observed_non_target_vaccination_actions": actions,
        "eligibility_notes": list(notes),
        "required_output_schema": {
            "theta": {name: "number in [-2,2]" for name in DIMENSION_NAMES}
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + (FORMAT_RETRY if strict_retry else "")


def build_update_prompt(
    theta: Mapping[str, float],
    observed: Mapping[str, Optional[int]],
    probabilities: Mapping[str, float],
    mismatches: Mapping[str, float],
    notes: Sequence[str],
    proxy: Mapping[str, float],
    strict_retry: bool,
) -> str:
    ranked = sorted(
        [
            {
                "vaccine": vaccine,
                "observed": observed.get(vaccine),
                "predicted_probability": probabilities.get(vaccine),
                "mismatch": mismatches.get(vaccine),
                "eligibility_note": notes[j],
            }
            for j, vaccine in enumerate(NON_TARGET_VACCINES)
            if observed.get(vaccine) is not None and "excluded" not in notes[j]
        ],
        key=lambda item: abs(float(item["mismatch"] or 0.0)),
        reverse=True,
    )
    payload = {
        "current_theta": theta,
        "observed_proxy_prior": proxy,
        "ranked_non_target_mismatches": ranked,
        "required_output_schema": {
            "update_direction": {name: "-1|0|1" for name in DIMENSION_NAMES}
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + (FORMAT_RETRY if strict_retry else "")


def build_final_prompt(
    context_profile: Mapping[str, Any],
    theta: Mapping[str, float],
    strict_retry: bool,
) -> str:
    payload = {
        "optimized_latent_reward_vector": theta,
        "non_vaccine_v4_context": context_profile,
        "influenza_target": "UNKNOWN_AND_WITHHELD",
        "required_output_schema": {
            "latent_reward_interpretation": {
                name: "negative|near_zero|positive|uncertain"
                for name in DIMENSION_NAMES
            },
            "integration": "one concise evidence-grounded paragraph",
            "probability_yes": "number from 0 to 100; use respondent-specific evidence rather than a default value",
            "prediction": "YES|NO",
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + (
        FORMAT_RETRY if strict_retry else ""
    )


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
        use_json_this_attempt = use_json_mode and not disable_json_mode_after_format_error
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
        for key in total_usage:
            total_usage[key] += int(usage.get(key, 0) or 0)
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
            }
        except Exception as exc:
            last_category = "output_truncated" if finish_reason.lower() == "length" else "output_format"
            last_message = f"{type(exc).__name__}: {exc}"
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


async def run_async_phase(
    *,
    label: str,
    items: Sequence[Mapping[str, Any]],
    log_path: Path,
    cfg_hash: str,
    key_field: str,
    workers: int,
    progress_every: int,
    handler: Callable[[Mapping[str, Any]], Awaitable[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    latest = load_latest_jsonl(log_path, cfg_hash, key_field=key_field)
    pending = [item for item in items if latest.get(int(item[key_field]), {}).get("status") != "ok"]
    reused = len(items) - len(pending)
    print(f"\n[{label}] total={len(items):,} reused_ok={reused:,} calls={len(pending):,}")
    if not pending:
        return [latest[int(item[key_field])] for item in items]

    queue: asyncio.Queue[Optional[Mapping[str, Any]]] = asyncio.Queue()
    for item in pending:
        queue.put_nowait(item)
    for _ in range(workers):
        queue.put_nowait(None)

    completed = 0
    ok_count = 0
    err_count = 0
    started = time.time()
    lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal completed, ok_count, err_count
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                result = await handler(item)
                append_jsonl(log_path, result)
                async with lock:
                    completed += 1
                    if result.get("status") == "ok":
                        ok_count += 1
                    else:
                        err_count += 1
                    if completed == 1 or completed % progress_every == 0 or completed == len(pending):
                        elapsed = max(time.time() - started, 1e-6)
                        rate = completed / elapsed
                        eta = (len(pending) - completed) / max(rate, 1e-9)
                        eta_text = f"{eta / 60:.1f} min" if eta < 7200 else f"{eta / 3600:.2f} h"
                        print(
                            f"[{label}] {completed:,}/{len(pending):,} new complete | "
                            f"ok={ok_count:,} err={err_count:,} | {rate:.2f} calls/s | ETA={eta_text}"
                        )
            finally:
                queue.task_done()

    tasks = [asyncio.create_task(worker()) for _ in range(max(1, workers))]
    await queue.join()
    await asyncio.gather(*tasks)
    latest = load_latest_jsonl(log_path, cfg_hash, key_field=key_field)
    return [latest[int(item[key_field])] for item in items if int(item[key_field]) in latest]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def binary_metrics(
    y: np.ndarray,
    probabilities_100: np.ndarray,
    threshold: float,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    if len(y) == 0:
        return {}
    p01 = np.clip(probabilities_100 / 100.0, 1e-6, 1 - 1e-6)
    pred = (probabilities_100 >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1], sample_weight=sample_weight)
    tn, fp, fn, tp = cm.ravel()
    out: Dict[str, Any] = {
        "threshold": float(threshold),
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred, sample_weight=sample_weight)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred, sample_weight=sample_weight)),
        "precision": float(precision_score(y, pred, zero_division=0, sample_weight=sample_weight)),
        "recall": float(recall_score(y, pred, zero_division=0, sample_weight=sample_weight)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "f1": float(f1_score(y, pred, zero_division=0, sample_weight=sample_weight)),
        "brier": float(brier_score_loss(y, p01, sample_weight=sample_weight)),
        "log_loss": float(log_loss(y, p01, labels=[0, 1], sample_weight=sample_weight)),
        "TN": float(tn), "FP": float(fp), "FN": float(fn), "TP": float(tp),
    }
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, p01, sample_weight=sample_weight))
        out["average_precision"] = float(average_precision_score(y, p01, sample_weight=sample_weight))
    else:
        out["roc_auc"] = None
        out["average_precision"] = None
    return out


def calibrate_threshold(entries: Sequence[Mapping[str, Any]], metric: str) -> Tuple[float, pd.DataFrame]:
    successful = [entry for entry in entries if entry.get("status") == "ok"]
    if not successful:
        raise RuntimeError("No successful calibration predictions")
    y = np.asarray([int(entry["actual"]) for entry in successful], dtype=int)
    p = np.asarray([float(entry["probability_yes"]) for entry in successful], dtype=float)
    w = np.asarray([float(entry.get("survey_weight", 1.0) or 1.0) for entry in successful], dtype=float)
    rows: List[Dict[str, Any]] = []
    for threshold in np.arange(5, 96, 1):
        unweighted = binary_metrics(y, p, float(threshold))
        weighted = binary_metrics(y, p, float(threshold), sample_weight=w)
        row = dict(unweighted)
        row.update({f"weighted_{key}": value for key, value in weighted.items() if key not in {"threshold", "n"}})
        rows.append(row)
    table = pd.DataFrame(rows)
    if metric not in table.columns:
        raise ValueError(f"Unknown threshold metric {metric}; available={sorted(table.columns)}")
    best = table.sort_values([metric, "log_loss"], ascending=[False, True]).iloc[0]
    return float(best["threshold"]), table


def phase_diagnostics(entries: Sequence[Mapping[str, Any]], expected_n: int, key_field: str = "data_idx") -> Dict[str, Any]:
    unique = {int(entry[key_field]): entry for entry in entries if key_field in entry}
    successful = [entry for entry in unique.values() if entry.get("status") == "ok"]
    errors = [entry for entry in unique.values() if entry.get("status") == "error"]
    return {
        "expected_n": int(expected_n),
        "observed_latest_n": int(len(unique)),
        "successful_n": int(len(successful)),
        "error_n": int(len(errors)),
        "success_rate": float(len(successful) / expected_n) if expected_n else 1.0,
        "failure_categories": dict(Counter(entry.get("failure_category", "unknown") for entry in errors)),
    }


def enforce_coverage(name: str, diagnostics: Mapping[str, Any], minimum: float) -> None:
    rate = float(diagnostics.get("success_rate", 0.0))
    if rate < minimum:
        raise RuntimeError(f"{name} coverage {rate:.2%} is below required {minimum:.2%}.")


def sum_usage(entries: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for entry in entries:
        usage = entry.get("usage") or {}
        for key in total:
            total[key] += int(usage.get(key, 0) or 0)
    return total


# ---------------------------------------------------------------------------
# Model resolution and client
# ---------------------------------------------------------------------------

def fuzzy_model_match(requested: str, available: Sequence[str]) -> str:
    if requested in available:
        return requested
    lower = [(model, model.lower()) for model in available]
    req = requested.lower()
    if "llama4" in req or "llama-4" in req:
        candidates = [m for m, l in lower if "llama" in l and "4" in l and ("scout" in l or "17b" in l)]
    elif "llama3" in req or "llama-3" in req:
        candidates = [m for m, l in lower if "llama" in l and "3" in l and "70b" in l]
    else:
        candidates = [m for m, l in lower if req in l]
    return candidates[0] if candidates else requested


async def resolve_models(client: Any, requested_models: Sequence[str]) -> Tuple[Dict[str, str], List[str]]:
    try:
        response = await client.models.list()
        available = sorted({str(item.id) for item in response.data})
    except Exception as exc:
        print(f"Warning: could not list ASU models: {type(exc).__name__}: {exc}")
        available = []
    resolved = {requested: fuzzy_model_match(requested, available) for requested in requested_models}
    return resolved, available


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------

def action_summary(actions: np.ndarray, weights: np.ndarray) -> Dict[str, Optional[int]]:
    return {
        vaccine: (int(actions[j]) if weights[j] > 0 else None)
        for j, vaccine in enumerate(NON_TARGET_VACCINES)
    }


def reward_vector_config(
    args: argparse.Namespace,
    split_source: str,
    global_cfg_hash: str,
    mode: str,
    model: str,
    json_mode_supported: bool,
) -> Dict[str, Any]:
    cfg = {
        "version": VERSION,
        "prompt_version": (
            UPDATE_PROMPT_VERSION
            if mode in {"prior_gradient_llm_update", "llm_init_gradient_llm_update"}
            else PROMPT_VERSION
        ),
        "component": "respondent_reward_inference",
        "mode": mode,
        "model": model,
        "split_source": split_source,
        "global_model_config_hash": global_cfg_hash,
        "eligibility_policy": args.eligibility_policy,
        "individual_steps": args.individual_steps,
        "individual_lr": args.individual_lr,
        "individual_lambda": args.individual_lambda,
        "llm_update_scale": args.llm_update_scale,
        "post_update_steps": args.post_update_steps,
        "temperature": args.temperature,
        "max_tokens_init": args.max_tokens_init,
        "max_tokens_update": args.max_tokens_update,
        "json_mode_supported": json_mode_supported,
        "include_sensitive_context": args.include_sensitive_context,
        "random_seed": args.random_seed,
    }
    cfg["config_hash"] = config_hash(cfg)
    return cfg


def prediction_config(
    args: argparse.Namespace,
    split_source: str,
    reward_cfg_hash: str,
    mode: str,
    model: str,
    json_mode_supported: bool,
) -> Dict[str, Any]:
    cfg = {
        "version": VERSION,
        "prompt_version": FINAL_PROMPT_VERSION,
        "component": "final_hbm_ccr_prediction",
        "mode": mode,
        "model": model,
        "split_source": split_source,
        "reward_config_hash": reward_cfg_hash,
        "temperature": args.temperature,
        "max_tokens_prediction": args.max_tokens_prediction,
        "json_mode_supported": json_mode_supported,
        "include_sensitive_context": args.include_sensitive_context,
        "threshold_metric": args.threshold_metric,
        "random_seed": args.random_seed,
    }
    cfg["config_hash"] = config_hash(cfg)
    return cfg


async def run_reward_mode(
    *,
    mode: str,
    model: str,
    client: Any,
    semaphore: asyncio.Semaphore,
    raw: pd.DataFrame,
    phase_assignments: pd.DataFrame,
    phase_data: ContextualDataset,
    global_arrays: Mapping[str, np.ndarray],
    global_cfg: Mapping[str, Any],
    split_source: str,
    experiment_dir: Path,
    args: argparse.Namespace,
    json_mode_supported: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    phase = str(phase_assignments.iloc[0]["split"])
    reward_cfg = reward_vector_config(
        args, split_source, str(global_cfg["config_hash"]), mode, model, json_mode_supported
    )
    reward_cfg_path = experiment_dir / "reward_config.json"
    if reward_cfg_path.exists():
        existing = json.loads(reward_cfg_path.read_text(encoding="utf-8"))
        if existing.get("config_hash") != reward_cfg["config_hash"]:
            prompt_only_upgrade = (
                mode in {"prior_gradient_llm_update", "llm_init_gradient_llm_update"}
                and existing.get("prompt_version") == PROMPT_VERSION
                and reward_cfg.get("prompt_version") == UPDATE_PROMPT_VERSION
                and configs_match_except(
                    existing,
                    reward_cfg,
                    {"config_hash", "prompt_version"},
                )
            )
            if prompt_only_upgrade:
                backup = archive_config(reward_cfg_path, "legacy_update_v1")
                reward_cfg_path.write_text(
                    json.dumps(reward_cfg, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(
                    f"Upgraded LLM-update prompt in {experiment_dir}. "
                    f"Old config archived at {backup.name}; global model and compatible "
                    "LLM initializations remain reusable."
                )
            else:
                raise RuntimeError(
                    f"Configuration mismatch in existing reward directory: {experiment_dir}. "
                    "Use a new output directory or --overwrite."
                )
    else:
        reward_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        reward_cfg_path.write_text(json.dumps(reward_cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    source_to_pos = {int(source): pos for pos, source in enumerate(phase_data.sources)}
    items: List[Dict[str, Any]] = []
    for row in phase_assignments.itertuples(index=False):
        source = int(row.source_row_index)
        items.append(
            {
                "data_idx": int(row.data_idx),
                "source_row_index": source,
                "actual": int(row.actual),
                "survey_weight": float(row.survey_weight) if pd.notna(row.survey_weight) else 1.0,
                "pos": source_to_pos[source],
            }
        )

    requires_init = mode in {"llm_init_gradient", "llm_init_gradient_llm_update"}
    requires_update = mode in {"prior_gradient_llm_update", "llm_init_gradient_llm_update"}

    init_entries: Dict[int, Dict[str, Any]] = {}
    if requires_init:
        init_log = experiment_dir.parent / "shared_llm_initializations" / f"{phase}.jsonl"
        init_cfg = {
            "version": VERSION,
            "prompt_version": PROMPT_VERSION,
            "component": "llm_initialization",
            "mode": "shared",
            "model": model,
            "split_source": split_source,
            "eligibility_policy": args.eligibility_policy,
            "temperature": args.temperature,
            "max_tokens_init": args.max_tokens_init,
            "json_mode_supported": json_mode_supported,
            "include_sensitive_context": args.include_sensitive_context,
            "random_seed": args.random_seed,
        }
        init_cfg["config_hash"] = config_hash(init_cfg)

        # v1 accidentally included the parent reward-mode hash in the shared
        # initialization hash, so the two LLM-initialized modes repeated calls.
        # Migrate matching successful rows to the corrected shared hash.
        legacy_init_hashes: List[str] = []
        for legacy_mode in ("llm_init_gradient", "llm_init_gradient_llm_update"):
            legacy_reward_cfg = reward_vector_config(
                args,
                split_source,
                str(global_cfg["config_hash"]),
                legacy_mode,
                model,
                json_mode_supported,
            )
            # Recreate the pre-fix v1 prompt version for the update mode.
            legacy_reward_cfg["prompt_version"] = PROMPT_VERSION
            legacy_reward_cfg["config_hash"] = config_hash(
                {k: v for k, v in legacy_reward_cfg.items() if k != "config_hash"}
            )
            legacy_init_cfg = dict(legacy_reward_cfg)
            legacy_init_cfg["component"] = "llm_initialization"
            legacy_init_cfg["mode"] = "shared"
            legacy_init_cfg["config_hash"] = config_hash(legacy_init_cfg)
            legacy_init_hashes.append(str(legacy_init_cfg["config_hash"]))

        migrated = migrate_compatible_initializations(
            init_log,
            new_hash=str(init_cfg["config_hash"]),
            legacy_hashes=legacy_init_hashes,
            items=items,
        )
        if migrated:
            print(
                f"Migrated {migrated:,} compatible LLM initialization rows "
                "to the corrected shared cache key."
            )

        async def init_handler(item: Mapping[str, Any]) -> Dict[str, Any]:
            source = int(item["source_row_index"])
            pos = int(item["pos"])
            actions_map = action_summary(phase_data.actions[pos], phase_data.action_weights[pos])
            notes = phase_data.eligibility_notes[pos]
            try:
                validated, raw_text, usage, request_id, meta = await call_structured_json(
                    client,
                    semaphore,
                    model=model,
                    system_prompt=LLM_INIT_SYSTEM,
                    prompt_builder=lambda strict: build_init_prompt(
                        build_profile(raw.loc[source], args.include_sensitive_context),
                        actions_map,
                        notes,
                        strict,
                    ),
                    validator=validate_llm_initialization,
                    max_tokens=args.max_tokens_init,
                    temperature=args.temperature,
                    retries=args.max_retries,
                    use_json_mode=json_mode_supported,
                )
                theta = [float(validated["theta"][name]) for name in DIMENSION_NAMES]
                return {
                    "created_at": utc_now(),
                    "config_hash": init_cfg["config_hash"],
                    "data_idx": int(item["data_idx"]),
                    "source_row_index": source,
                    "status": "ok",
                    "theta": theta,
                    "theta_by_name": validated["theta"],
                    "usage": usage,
                    "request_id": request_id,
                    "raw_response": raw_text[:1500],
                    **meta,
                }
            except StructuredCallFailure as exc:
                return {
                    "created_at": utc_now(),
                    "config_hash": init_cfg["config_hash"],
                    "data_idx": int(item["data_idx"]),
                    "source_row_index": source,
                    "status": "error",
                    "failure_category": exc.category,
                    "error_message": str(exc),
                    "raw_response": exc.raw_response,
                    "finish_reason": exc.finish_reason,
                    "attempt_count": exc.attempt_count,
                    "usage": exc.usage,
                }

        init_list = await run_async_phase(
            label=f"{model}/{mode}/{phase}/llm_initialization",
            items=items,
            log_path=init_log,
            cfg_hash=init_cfg["config_hash"],
            key_field="data_idx",
            workers=args.concurrent_samples,
            progress_every=args.progress_every,
            handler=init_handler,
        )
        init_diag = phase_diagnostics(init_list, len(items))
        enforce_coverage("LLM initialization", init_diag, args.min_success_rate)
        init_entries = {int(entry["data_idx"]): entry for entry in init_list if entry.get("status") == "ok"}

    reward_log = experiment_dir / "logs" / f"{phase}_reward_vectors.jsonl"
    existing_rewards = load_latest_jsonl(reward_log, reward_cfg["config_hash"], key_field="data_idx")
    pending = [item for item in items if existing_rewards.get(int(item["data_idx"]), {}).get("status") != "ok"]
    print(
        f"\n[{model}/{mode}/{phase}/reward_inference] total={len(items):,} "
        f"reused_ok={len(items)-len(pending):,} calls_or_compute={len(pending):,}"
    )
    if not pending:
        return [existing_rewards[int(item["data_idx"])] for item in items], reward_cfg

    # Vectorized numerical inference before any optional LLM update.
    pending_pos = np.asarray([int(item["pos"]) for item in pending], dtype=int)
    initial_theta_matrix: Optional[np.ndarray] = None
    if requires_init:
        initial_theta_matrix = np.vstack(
            [np.asarray(init_entries[int(item["data_idx"])]["theta"], dtype=np.float32) for item in pending]
        )
    pre_results = infer_reward_batch(
        phase_data.x[pending_pos],
        phase_data.actions[pending_pos],
        phase_data.action_weights[pending_pos],
        global_arrays,
        args,
        initial_theta=initial_theta_matrix,
    )

    update_entries: Dict[int, Dict[str, Any]] = {}
    if requires_update:
        update_log = experiment_dir / "logs" / f"{phase}_llm_updates.jsonl"
        update_cfg = dict(reward_cfg)
        update_cfg["component"] = "llm_reward_update"
        update_cfg["config_hash"] = config_hash(update_cfg)
        update_items: List[Dict[str, Any]] = []
        for item, pre in zip(pending, pre_results):
            payload = dict(item)
            payload["pre_result"] = pre
            update_items.append(payload)

        async def update_handler(item: Mapping[str, Any]) -> Dict[str, Any]:
            source = int(item["source_row_index"])
            pos = int(item["pos"])
            pre = item["pre_result"]
            if int(pre["n_observed_actions"]) == 0:
                return {
                    "created_at": utc_now(),
                    "config_hash": update_cfg["config_hash"],
                    "data_idx": int(item["data_idx"]),
                    "source_row_index": source,
                    "status": "ok",
                    "update_direction": {name: 0 for name in DIMENSION_NAMES},
                    "direction": [0, 0, 0, 0, 0],
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "note": "No eligible observed non-target actions; no LLM update requested.",
                }
            observed_map = action_summary(phase_data.actions[pos], phase_data.action_weights[pos])
            proxy = dict(zip(DIMENSION_NAMES, phase_data.x[pos].astype(float).tolist()))
            notes = phase_data.eligibility_notes[pos]
            try:
                validated, raw_text, usage, request_id, meta = await call_structured_json(
                    client,
                    semaphore,
                    model=model,
                    system_prompt=LLM_UPDATE_SYSTEM,
                    prompt_builder=lambda strict: build_update_prompt(
                        pre["theta_by_name"],
                        observed_map,
                        pre["predicted_non_target_probabilities"],
                        pre["mismatches"],
                        notes,
                        proxy,
                        strict,
                    ),
                    validator=validate_llm_update,
                    max_tokens=args.max_tokens_update,
                    temperature=args.temperature,
                    retries=args.max_retries,
                    use_json_mode=json_mode_supported,
                )
                direction = [validated["update_direction"][name] for name in DIMENSION_NAMES]
                return {
                    "created_at": utc_now(),
                    "config_hash": update_cfg["config_hash"],
                    "data_idx": int(item["data_idx"]),
                    "source_row_index": source,
                    "status": "ok",
                    "update_direction": validated["update_direction"],
                    "direction": direction,
                    "usage": usage,
                    "request_id": request_id,
                    "raw_response": raw_text[:1500],
                    **meta,
                }
            except StructuredCallFailure as exc:
                return {
                    "created_at": utc_now(),
                    "config_hash": update_cfg["config_hash"],
                    "data_idx": int(item["data_idx"]),
                    "source_row_index": source,
                    "status": "error",
                    "failure_category": exc.category,
                    "error_message": str(exc),
                    "raw_response": exc.raw_response,
                    "finish_reason": exc.finish_reason,
                    "attempt_count": exc.attempt_count,
                    "usage": exc.usage,
                }

        update_list = await run_async_phase(
            label=f"{model}/{mode}/{phase}/llm_reward_update",
            items=update_items,
            log_path=update_log,
            cfg_hash=update_cfg["config_hash"],
            key_field="data_idx",
            workers=args.concurrent_samples,
            progress_every=args.progress_every,
            handler=update_handler,
        )
        update_diag = phase_diagnostics(update_list, len(update_items))
        enforce_coverage("LLM reward update", update_diag, args.min_success_rate)
        update_entries = {int(entry["data_idx"]): entry for entry in update_list if entry.get("status") == "ok"}
        update_matrix = np.vstack(
            [np.asarray(update_entries[int(item["data_idx"])]["direction"], dtype=np.float32) for item in pending]
        )
        final_results = infer_reward_batch(
            phase_data.x[pending_pos],
            phase_data.actions[pending_pos],
            phase_data.action_weights[pending_pos],
            global_arrays,
            args,
            initial_theta=initial_theta_matrix,
            update_directions=update_matrix,
        )
    else:
        final_results = pre_results

    for item, result in zip(pending, final_results):
        update_entry = update_entries.get(int(item["data_idx"]))
        record = {
            "created_at": utc_now(),
            "config_hash": reward_cfg["config_hash"],
            "data_idx": int(item["data_idx"]),
            "source_row_index": int(item["source_row_index"]),
            "actual": int(item["actual"]),
            "survey_weight": float(item["survey_weight"]),
            "status": "ok",
            "mode": mode,
            "theta": result["theta"],
            "theta_by_name": result["theta_by_name"],
            "prior_theta": result["prior_theta"],
            "predicted_non_target_probabilities": result["predicted_non_target_probabilities"],
            "mismatches": result["mismatches"],
            "n_observed_actions": result["n_observed_actions"],
            "eligibility_notes": phase_data.eligibility_notes[int(item["pos"])],
            "usage": (update_entry or {}).get(
                "usage", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            ),
        }
        if update_entry is not None:
            record["llm_update_direction"] = update_entry.get("update_direction")
            record["update_request_id"] = update_entry.get("request_id", "")
        append_jsonl(reward_log, record)

    latest = load_latest_jsonl(reward_log, reward_cfg["config_hash"], key_field="data_idx")
    rewards = [latest[int(item["data_idx"])] for item in items if int(item["data_idx"]) in latest]
    reward_diag = phase_diagnostics(rewards, len(items))
    enforce_coverage("Reward inference", reward_diag, args.min_success_rate)
    return rewards, reward_cfg


async def run_final_predictions(
    *,
    mode: str,
    model: str,
    client: Any,
    semaphore: asyncio.Semaphore,
    raw: pd.DataFrame,
    phase_assignments: pd.DataFrame,
    phase_data: ContextualDataset,
    reward_entries: Sequence[Mapping[str, Any]],
    reward_cfg: Mapping[str, Any],
    split_source: str,
    experiment_dir: Path,
    args: argparse.Namespace,
    json_mode_supported: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pred_cfg = prediction_config(
        args,
        split_source,
        str(reward_cfg["config_hash"]),
        mode,
        model,
        json_mode_supported,
    )
    pred_cfg_path = experiment_dir / "prediction_config.json"
    if pred_cfg_path.exists():
        existing = json.loads(pred_cfg_path.read_text(encoding="utf-8"))
        if existing.get("config_hash") != pred_cfg["config_hash"]:
            final_prompt_only_upgrade = (
                existing.get("prompt_version") == PROMPT_VERSION
                and pred_cfg.get("prompt_version") == FINAL_PROMPT_VERSION
                and configs_match_except(
                    existing,
                    pred_cfg,
                    {"config_hash", "prompt_version", "reward_config_hash"},
                )
            )
            if final_prompt_only_upgrade:
                backup = archive_config(pred_cfg_path, "legacy_direct_actions_v1")
                pred_cfg_path.write_text(
                    json.dumps(pred_cfg, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(
                    f"Upgraded final CCR prompt in {experiment_dir}. "
                    f"Old config archived at {backup.name}; reward vectors are retained, "
                    "while old final predictions are intentionally not reused."
                )
            else:
                raise RuntimeError(
                    f"Configuration mismatch in existing prediction directory: {experiment_dir}. "
                    "Use a new output directory or --overwrite."
                )
    else:
        pred_cfg_path.write_text(json.dumps(pred_cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    reward_lookup = {
        int(entry["data_idx"]): entry for entry in reward_entries if entry.get("status") == "ok"
    }
    source_to_pos = {int(source): pos for pos, source in enumerate(phase_data.sources)}
    items: List[Dict[str, Any]] = []
    for row in phase_assignments.itertuples(index=False):
        items.append(
            {
                "data_idx": int(row.data_idx),
                "source_row_index": int(row.source_row_index),
                "actual": int(row.actual),
                "survey_weight": float(row.survey_weight) if pd.notna(row.survey_weight) else 1.0,
                "pos": source_to_pos[int(row.source_row_index)],
            }
        )

    pred_log = experiment_dir / "logs" / f"{phase_assignments.iloc[0]['split']}_predictions.jsonl"

    async def handler(item: Mapping[str, Any]) -> Dict[str, Any]:
        data_idx = int(item["data_idx"])
        source = int(item["source_row_index"])
        reward = reward_lookup.get(data_idx)
        if reward is None:
            return {
                "created_at": utc_now(),
                "config_hash": pred_cfg["config_hash"],
                "data_idx": data_idx,
                "source_row_index": source,
                "status": "error",
                "failure_category": "missing_reward_vector",
                "error_message": "Reward vector is unavailable",
            }
        try:
            validated, raw_text, usage, request_id, meta = await call_structured_json(
                client,
                semaphore,
                model=model,
                system_prompt=FINAL_CCR_SYSTEM,
                prompt_builder=lambda strict: build_final_prompt(
                    build_context_only_profile(
                        raw.loc[source],
                        args.include_sensitive_context,
                    ),
                    reward["theta_by_name"],
                    strict,
                ),
                validator=validate_final_prediction,
                max_tokens=args.max_tokens_prediction,
                temperature=args.temperature,
                retries=args.max_retries,
                use_json_mode=json_mode_supported,
            )
            return {
                "created_at": utc_now(),
                "config_hash": pred_cfg["config_hash"],
                "data_idx": data_idx,
                "source_row_index": source,
                "actual": int(item["actual"]),
                "survey_weight": float(item["survey_weight"]),
                "status": "ok",
                "mode": mode,
                "theta": reward["theta"],
                "theta_by_name": reward["theta_by_name"],
                "probability_yes": validated["probability_yes"],
                "prediction_at_50": validated["prediction_at_50"],
                "raw_prediction": validated["raw_prediction"],
                "latent_reward_interpretation": validated["latent_reward_interpretation"],
                "integration": validated["integration"],
                "usage": usage,
                "request_id": request_id,
                "raw_response": raw_text[:3000],
                **meta,
            }
        except StructuredCallFailure as exc:
            return {
                "created_at": utc_now(),
                "config_hash": pred_cfg["config_hash"],
                "data_idx": data_idx,
                "source_row_index": source,
                "actual": int(item["actual"]),
                "survey_weight": float(item["survey_weight"]),
                "status": "error",
                "failure_category": exc.category,
                "error_message": str(exc),
                "raw_response": exc.raw_response,
                "finish_reason": exc.finish_reason,
                "attempt_count": exc.attempt_count,
                "usage": exc.usage,
            }

    predictions = await run_async_phase(
        label=f"{model}/{mode}/{phase_assignments.iloc[0]['split']}/hbm_ccr_prediction",
        items=items,
        log_path=pred_log,
        cfg_hash=pred_cfg["config_hash"],
        key_field="data_idx",
        workers=args.concurrent_samples,
        progress_every=args.progress_every,
        handler=handler,
    )
    pred_diag = phase_diagnostics(predictions, len(items))
    enforce_coverage("HBM-CCR prediction", pred_diag, args.min_success_rate)
    return predictions, pred_cfg


def entries_to_frame(entries: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for entry in entries:
        if entry.get("status") != "ok":
            continue
        row = {
            "data_idx": entry.get("data_idx"),
            "source_row_index": entry.get("source_row_index"),
            "actual": entry.get("actual"),
            "survey_weight": entry.get("survey_weight"),
            "probability_yes": entry.get("probability_yes"),
            "prediction_at_50": entry.get("prediction_at_50"),
            "raw_prediction": entry.get("raw_prediction"),
            "integration": entry.get("integration"),
        }
        theta = entry.get("theta_by_name") or {}
        for name in DIMENSION_NAMES:
            row[f"theta_{name}"] = theta.get(name)
        interpretation = entry.get("latent_reward_interpretation") or {}
        for name in DIMENSION_NAMES:
            row[f"interpretation_{name}"] = interpretation.get(name)
        rows.append(row)
    return pd.DataFrame(rows)


async def async_main(args: argparse.Namespace) -> None:
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input_csv, low_memory=False)
    raw.index = np.arange(len(raw), dtype=int)
    missing = [c for c in ALL_REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"adult24.csv is missing required columns: {missing}")

    if args.v4_reference_split and args.v4_reference_split.exists():
        full_assignments = load_reference_split(args.v4_reference_split, raw)
        split_source = str(args.v4_reference_split)
        exact_split = True
    elif args.allow_fallback_split:
        full_assignments = create_fallback_split(raw, args.random_seed)
        split_source = "fallback_outcome_stratified_40_20_40"
        exact_split = False
    else:
        raise FileNotFoundError("Provide --v4-reference-split or pass --allow-fallback-split.")

    full_assignments = ordered_assignments(full_assignments)
    selected_assignments = ordered_assignments(
        downsample_targets(full_assignments.drop(columns=["data_idx"]), args.sample_size, args.random_seed)
    )
    memory_assignments = full_assignments[full_assignments["split"] == "memory"].copy()
    calibration_assignments = selected_assignments[selected_assignments["split"] == "calibration"].copy()
    test_assignments = selected_assignments[selected_assignments["split"] == "test"].copy()

    print(
        f"[v4] full_selected={len(full_assignments):,} "
        f"split={dict(full_assignments['split'].value_counts())} "
        f"evaluation_sample={len(calibration_assignments) + len(test_assignments):,}"
    )

    memory_data = build_contextual_dataset(memory_assignments, raw, args.eligibility_policy)
    calibration_data = build_contextual_dataset(calibration_assignments, raw, args.eligibility_policy)
    test_data = build_contextual_dataset(test_assignments, raw, args.eligibility_policy)

    action_summary_rows = []
    for j, vaccine in enumerate(NON_TARGET_VACCINES):
        mask = memory_data.action_weights[:, j] > 0
        prevalence = float(memory_data.actions[mask, j].mean()) if np.any(mask) else None
        action_summary_rows.append(
            {
                "vaccine": vaccine,
                "observed_memory_n": int(mask.sum()),
                "memory_prevalence": prevalence,
                "observation_weight": VACCINE_OBSERVATION_WEIGHTS[vaccine],
            }
        )
    pd.DataFrame(action_summary_rows).to_csv(args.output_dir / "memory_non_target_action_summary.csv", index=False)

    plan = {
        "version": VERSION,
        "created_at": utc_now(),
        "input_csv": str(args.input_csv),
        "split_source": split_source,
        "exact_reference_split": exact_split,
        "models_requested": args.models,
        "reward_modes": args.reward_modes,
        "memory_n": len(memory_assignments),
        "calibration_n": len(calibration_assignments),
        "test_n": len(test_assignments),
        "eligibility_policy": args.eligibility_policy,
        "estimated_calls_by_mode_per_model": {
            mode: {
                "llm_initialization_needed": mode in {"llm_init_gradient", "llm_init_gradient_llm_update"},
                "llm_update_calls": (
                    len(calibration_assignments) + len(test_assignments)
                    if mode in {"prior_gradient_llm_update", "llm_init_gradient_llm_update"}
                    else 0
                ),
                "final_hbm_ccr_prediction_calls": len(calibration_assignments) + len(test_assignments),
            }
            for mode in args.reward_modes
        },
        "estimated_unique_calls_per_model": {
            "shared_llm_initializations": (
                len(calibration_assignments) + len(test_assignments)
                if any(mode in {"llm_init_gradient", "llm_init_gradient_llm_update"} for mode in args.reward_modes)
                else 0
            ),
            "llm_updates": (
                len(calibration_assignments) + len(test_assignments)
            ) * sum(mode in {"prior_gradient_llm_update", "llm_init_gradient_llm_update"} for mode in args.reward_modes),
            "final_hbm_ccr_predictions": (
                len(calibration_assignments) + len(test_assignments)
            ) * len(args.reward_modes),
        },
    }
    unique_calls = plan["estimated_unique_calls_per_model"]
    unique_calls["total"] = int(sum(unique_calls.values()))
    (args.output_dir / "benchmark_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if args.plan_only:
        return

    global_arrays, global_cfg = fit_or_load_global_model(args.output_dir, memory_data, args)
    if args.fit_global_only:
        print("Global contextual reward model completed; stopping because --fit-global-only was used.")
        return

    if args.dry_run:
        source = int(calibration_assignments.iloc[0]["source_row_index"] if len(calibration_assignments) else test_assignments.iloc[0]["source_row_index"])
        row = raw.loc[source]
        proxy = proxy_scores(row)
        labels, weights, notes = vaccine_actions(row, args.eligibility_policy)
        prior_reward = infer_individual_reward(
            np.asarray([proxy[name] for name in DIMENSION_NAMES], dtype=np.float32),
            labels,
            weights,
            global_arrays,
            args,
        )
        dry_dir = args.output_dir / "dry_run_prompts"
        dry_dir.mkdir(parents=True, exist_ok=True)
        (dry_dir / "llm_initialization_prompt.txt").write_text(
            LLM_INIT_SYSTEM + "\n\n" + build_init_prompt(
                build_profile(row, args.include_sensitive_context), action_summary(labels, weights), notes, False
            ),
            encoding="utf-8",
        )
        (dry_dir / "llm_update_prompt.txt").write_text(
            LLM_UPDATE_SYSTEM + "\n\n" + build_update_prompt(
                prior_reward["theta_by_name"], action_summary(labels, weights),
                prior_reward["predicted_non_target_probabilities"], prior_reward["mismatches"], notes,
                proxy, False,
            ),
            encoding="utf-8",
        )
        (dry_dir / "final_hbm_ccr_prompt.txt").write_text(
            FINAL_CCR_SYSTEM + "\n\n" + build_final_prompt(
                build_context_only_profile(row, args.include_sensitive_context),
                prior_reward["theta_by_name"],
                False,
            ),
            encoding="utf-8",
        )
        print(f"Dry-run prompts saved to {dry_dir}")
        return

    if AsyncOpenAI is None:
        raise ImportError("A recent openai package is required for API experiments. Run: pip install -U openai httpx")

    api_key = os.environ.get("ASU_LLM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set ASU_LLM_API_KEY in the environment before running API experiments.")

    timeout = httpx.Timeout(
        connect=min(args.timeout, 60.0),
        read=args.timeout,
        write=min(args.timeout, 60.0),
        pool=min(args.timeout, 60.0),
    )
    http_client = httpx.AsyncClient(timeout=timeout, trust_env=False, http2=False)
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=api_key,
        http_client=http_client,
        max_retries=0,
    )
    semaphore = asyncio.Semaphore(max(1, args.max_concurrent_requests))

    try:
        resolved_models, available_models = await resolve_models(client, args.models)
        (args.output_dir / "model_resolution.json").write_text(
            json.dumps(
                {
                    "requested_to_resolved": resolved_models,
                    "available_models": available_models,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        json_support: Dict[str, bool] = {}
        probe_records: Dict[str, Any] = {}
        for requested, resolved in resolved_models.items():
            if args.json_mode == "always":
                supported, detail = True, "forced_always"
            elif args.json_mode == "never":
                supported, detail = False, "forced_never"
            else:
                supported, detail = await probe_json_mode(client, resolved)
            json_support[resolved] = supported
            probe_records[resolved] = {"supported": supported, "detail": detail}
        (args.output_dir / "json_mode_probe.json").write_text(
            json.dumps(probe_records, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        result_rows: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []

        for requested, resolved_model in resolved_models.items():
            for mode in args.reward_modes:
                experiment_dir = args.output_dir / resolved_model / mode
                experiment_dir.mkdir(parents=True, exist_ok=True)
                try:
                    cal_rewards, reward_cfg = await run_reward_mode(
                        mode=mode,
                        model=resolved_model,
                        client=client,
                        semaphore=semaphore,
                        raw=raw,
                        phase_assignments=calibration_assignments,
                        phase_data=calibration_data,
                        global_arrays=global_arrays,
                        global_cfg=global_cfg,
                        split_source=split_source,
                        experiment_dir=experiment_dir,
                        args=args,
                        json_mode_supported=json_support[resolved_model],
                    )
                    test_rewards, _ = await run_reward_mode(
                        mode=mode,
                        model=resolved_model,
                        client=client,
                        semaphore=semaphore,
                        raw=raw,
                        phase_assignments=test_assignments,
                        phase_data=test_data,
                        global_arrays=global_arrays,
                        global_cfg=global_cfg,
                        split_source=split_source,
                        experiment_dir=experiment_dir,
                        args=args,
                        json_mode_supported=json_support[resolved_model],
                    )
                    cal_predictions, pred_cfg = await run_final_predictions(
                        mode=mode,
                        model=resolved_model,
                        client=client,
                        semaphore=semaphore,
                        raw=raw,
                        phase_assignments=calibration_assignments,
                        phase_data=calibration_data,
                        reward_entries=cal_rewards,
                        reward_cfg=reward_cfg,
                        split_source=split_source,
                        experiment_dir=experiment_dir,
                        args=args,
                        json_mode_supported=json_support[resolved_model],
                    )
                    test_predictions, _ = await run_final_predictions(
                        mode=mode,
                        model=resolved_model,
                        client=client,
                        semaphore=semaphore,
                        raw=raw,
                        phase_assignments=test_assignments,
                        phase_data=test_data,
                        reward_entries=test_rewards,
                        reward_cfg=reward_cfg,
                        split_source=split_source,
                        experiment_dir=experiment_dir,
                        args=args,
                        json_mode_supported=json_support[resolved_model],
                    )

                    threshold, threshold_table = calibrate_threshold(cal_predictions, args.threshold_metric)
                    threshold_table.to_csv(experiment_dir / "threshold_search.csv", index=False)
                    cal_frame = entries_to_frame(cal_predictions)
                    test_frame = entries_to_frame(test_predictions)
                    cal_frame.to_csv(experiment_dir / "calibration_predictions.csv", index=False)
                    test_frame.to_csv(experiment_dir / "test_predictions.csv", index=False)

                    y = test_frame["actual"].to_numpy(dtype=int)
                    p = test_frame["probability_yes"].to_numpy(dtype=float)
                    w = test_frame["survey_weight"].to_numpy(dtype=float)
                    metrics_selected = binary_metrics(y, p, threshold)
                    metrics_50 = binary_metrics(y, p, 50.0)
                    metrics_weighted = binary_metrics(y, p, threshold, sample_weight=w)
                    cal_diag = phase_diagnostics(cal_predictions, len(calibration_assignments))
                    test_diag = phase_diagnostics(test_predictions, len(test_assignments))
                    usage = sum_usage(list(cal_rewards) + list(test_rewards) + list(cal_predictions) + list(test_predictions))
                    summary = {
                        "created_at": utc_now(),
                        "version": VERSION,
                        "variant": "v4",
                        "requested_model": requested,
                        "model": resolved_model,
                        "method": "silic_inverse_contextual_reward_hbm_ccr",
                        "reward_mode": mode,
                        "split_source": split_source,
                        "selected_threshold": threshold,
                        "threshold_metric": args.threshold_metric,
                        "metrics": {
                            "test_selected": metrics_selected,
                            "test_at_50": metrics_50,
                            "test_selected_survey_weighted": metrics_weighted,
                        },
                        "calibration_diagnostics": cal_diag,
                        "test_diagnostics": test_diag,
                        "usage_total": usage,
                        "reward_config_hash": reward_cfg["config_hash"],
                        "prediction_config_hash": pred_cfg["config_hash"],
                    }
                    (experiment_dir / "summary.json").write_text(
                        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
                    )
                    result_rows.append(
                        {
                            "variant": "v4",
                            "model": resolved_model,
                            "reward_mode": mode,
                            "selected_threshold": threshold,
                            **metrics_selected,
                            **{f"survey_weighted_{k}": v for k, v in metrics_weighted.items() if k not in {"threshold", "n"}},
                            "calibration_success_rate": cal_diag["success_rate"],
                            "test_success_rate": test_diag["success_rate"],
                            "input_tokens": usage["input_tokens"],
                            "output_tokens": usage["output_tokens"],
                            "total_tokens": usage["total_tokens"],
                        }
                    )
                except Exception as exc:
                    failure = {
                        "created_at": utc_now(),
                        "variant": "v4",
                        "model": resolved_model,
                        "reward_mode": mode,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                    failures.append(failure)
                    print(f"EXPERIMENT FAILED: {failure}")
                    if not args.continue_grid_on_error:
                        raise

        pd.DataFrame(result_rows).to_csv(args.output_dir / "benchmark_results.csv", index=False)
        if failures:
            pd.DataFrame(failures).to_csv(args.output_dir / "benchmark_failures.csv", index=False)
        elif (args.output_dir / "benchmark_failures.csv").exists():
            (args.output_dir / "benchmark_failures.csv").unlink()
        print("\nCompleted result rows:", len(result_rows))
        if result_rows:
            print(pd.DataFrame(result_rows).to_string(index=False))
    finally:
        await client.close()
        await http_client.aclose()


def parse_csv_list(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--v4-reference-split", type=Path)
    parser.add_argument("--allow-fallback-split", action="store_true")
    parser.add_argument("--models", type=parse_csv_list, default=["llama4-scout-17b", "llama3-groq-70b-tool-use"])
    parser.add_argument("--reward-modes", type=parse_csv_list, default=REWARD_MODES)
    parser.add_argument("--base-url", default="https://openai.rc.asu.edu/v1")
    parser.add_argument("--sample-size", type=int, default=0, help="Calibration+test target sample; 0 means full.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--eligibility-policy", choices=["strict", "broad"], default="strict")
    parser.add_argument("--include-sensitive-context", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--global-epochs", type=int, default=300)
    parser.add_argument("--global-lr", type=float, default=0.02)
    parser.add_argument("--global-patience", type=int, default=35)
    parser.add_argument("--global-min-delta", type=float, default=1e-5)
    parser.add_argument("--global-progress-every", type=int, default=25)
    parser.add_argument("--lambda-person", type=float, default=0.25)
    parser.add_argument("--lambda-global", type=float, default=0.002)
    parser.add_argument("--lambda-loading-anchor", type=float, default=0.20)
    parser.add_argument("--learn-vaccine-loadings", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--individual-steps", type=int, default=80)
    parser.add_argument("--individual-lr", type=float, default=0.05)
    parser.add_argument("--individual-lambda", type=float, default=0.60)
    parser.add_argument("--llm-update-scale", type=float, default=0.20)
    parser.add_argument("--post-update-steps", type=int, default=30)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens-init", type=int, default=180)
    parser.add_argument("--max-tokens-update", type=int, default=140)
    parser.add_argument("--max-tokens-prediction", type=int, default=320)
    parser.add_argument("--json-mode", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--max-concurrent-requests", type=int, default=1)
    parser.add_argument("--concurrent-samples", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--min-success-rate", type=float, default=0.995)
    parser.add_argument("--threshold-metric", default="balanced_accuracy")

    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--fit-global-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-grid-on-error", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    unknown = sorted(set(args.reward_modes) - set(REWARD_MODES))
    if unknown:
        raise ValueError(f"Unknown reward modes: {unknown}; allowed={REWARD_MODES}")
    if not args.models:
        raise ValueError("At least one model is required")
    if args.max_concurrent_requests < 1 or args.concurrent_samples < 1:
        raise ValueError("Concurrency values must be >= 1")
    if not (0 < args.min_success_rate <= 1):
        raise ValueError("min_success_rate must be in (0,1]")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
