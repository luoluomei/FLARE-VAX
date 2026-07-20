#!/usr/bin/env python3
"""
FLARE-VAX-NV: theory-guided influenza-vaccination prediction without any
other-vaccine-history predictors.

Core design
-----------
1. Construct five deterministic observed proxies from NHIS 2024:
   - observed_threat_proxy
   - preventive_engagement_proxy
   - structural_barrier_proxy
   - healthcare_cue_proxy
   - navigation_self_efficacy_proxy
2. Form three meta-dimensions and eight HBM-style patterns:
   Motivation = mean(Threat, Preventive Engagement)
   Capability = mean(5 - Barriers, Self-Efficacy)
   Activation = Healthcare Cues
3. Run a FLARE-inspired staged reasoning call. The model returns a concise,
   structured reasoning trace rather than unconstrained free-form chain of thought.
4. Reflect on selected high-confidence errors, distill structured memories that
   preserve the failed reasoning stage, corrected reasoning path, applicability,
   contradictions, and retrieval keys.
5. Freeze memory, calibrate the classification threshold on a held-out split,
   and evaluate once on test data.

IMPORTANT
---------
- The target is SHTFLU12M_A.
- All other vaccine variables are excluded from scoring, profiles, retrieval,
  prompts, and similarity representations:
    SHTCVD191_A, SHTCVD19NM2_A, SHTPNUEV_A, SHTPNEUNB_A,
    SHTSHINGL1_A, SHINGRIX3_A, SHTHEPA_A, SHTFLUM_A, SHTFLUY_A.
- These constructs are observed behavioral proxies, not direct psychometric HBM
  measurements. "Preventive engagement" replaces the earlier prior-vaccine-based
  benefit proxy and should not be described as a direct perceived-benefit scale.

The script is self-contained and supports --dry-run for prompt inspection.
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

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

try:
    from openai import AsyncOpenAI
except ImportError:  # dry-run still works
    AsyncOpenAI = None  # type: ignore

VERSION = "flare_vax_no_prior_vax_cot_memory_v1"
TARGET = "SHTFLU12M_A"
WEIGHT = "WTFA_A"
ID_COLUMNS = ["HHX", "SRVY_YR", "PSTRAT", "PPSU"]

# Every vaccine-related variable except the target is explicitly forbidden.
EXCLUDED_VACCINE_COLUMNS = {
    "SHTCVD191_A", "SHTCVD19NM2_A", "SHTPNUEV_A", "SHTPNEUNB_A",
    "SHTSHINGL1_A", "SHINGRIX3_A", "SHTHEPA_A", "SHTFLUM_A", "SHTFLUY_A",
}

CHRONIC_VARS = [
    "HYPEV_A", "CHDEV_A", "ANGEV_A", "MIEV_A", "STREV_A", "ASTILL_A",
    "CANEV_A", "DIBEV_A", "COPDEV_A", "KIDWEAKEV_A", "LIVEREV_A",
]

BACKGROUND_COLUMNS = [
    "AGEP_A", "SEX_A", "HISPALLP_A", "RACEALLP_A", "EDUCP_A", "RATCAT_A",
    "REGION", "PHSTAT_A", "BMICAT_A", "SMKCIGST_A",
]

BARRIER_COLUMNS = [
    "HICOV_A", "NOTCOV_A", "HINOTYR_A", "HINOTMYR_A", "RSNHICOST_A",
    "HISTOPCOST_A", "MEDDL12M_A", "MEDNG12M_A", "RXDL12M_A", "RXDG12M_A",
    "PAYWORRY_A", "PAYBLL12M_A", "PAYNOBLLNW_A", "TRANSPOR_A", "COMDIFF_A",
    "PRDEDUC1_A", "PRDEDUC2_A",
]

CONTACT_COLUMNS = [
    "LASTDR_A", "WELLNESS_A", "WELLVIS_A", "RETAILHC12MTC_A", "VIRAPP12M_A",
    "URGCC12MTC_A", "EMERG12MTC_A", "HOSPONGT_A",
]

NAVIGATION_COLUMNS = [
    "USUALPL_A", "USPLKIND_A", "ACCSSINT_A", "ACCSSHOM_A",
    "HITLOOK_A", "HITCOMM_A", "HITTEST_A",
]

REQUIRED_COLUMNS = sorted(set(
    ID_COLUMNS
    + [TARGET, WEIGHT, "ANYDIFF_A", "DISAB3_A", "HLTHCOND_A"]
    + BACKGROUND_COLUMNS + BARRIER_COLUMNS + CONTACT_COLUMNS + NAVIGATION_COLUMNS
    + CHRONIC_VARS
))


DEFAULT_CONSTRUCT_CONFIG: Dict[str, Any] = {
    "threat_weights": {
        "threat_age_component": 0.25,
        "threat_health_component": 0.20,
        "threat_chronic_component": 0.35,
        "threat_function_component": 0.10,
        "threat_immune_component": 0.10,
    },
    "engagement_weights": {
        "engagement_wellness_component": 0.45,
        "engagement_information_component": 0.20,
        "engagement_doctor_communication_component": 0.20,
        "engagement_result_review_component": 0.15,
    },
    "barrier_weights": {
        "barrier_insurance_component": 0.25,
        "barrier_cost_care_component": 0.35,
        "barrier_financial_component": 0.20,
        "barrier_transport_component": 0.10,
        "barrier_communication_component": 0.10,
    },
    "cue_weights": {
        "cue_doctor_component": 0.45,
        "cue_retail_virtual_component": 0.25,
        "cue_acute_component": 0.10,
        "cue_usual_care_opportunity_component": 0.20,
    },
    "self_efficacy_weights": {
        "selfeff_usual_care_component": 0.35,
        "selfeff_care_setting_component": 0.15,
        "selfeff_internet_component": 0.20,
        "selfeff_virtual_component": 0.10,
        "selfeff_communication_component": 0.20,
    },
    "motivation_weights": {"observed_threat_proxy": 0.50, "preventive_engagement_proxy": 0.50},
    "capability_weights": {"reversed_structural_barrier": 0.50, "navigation_self_efficacy_proxy": 0.50},
}


def load_construct_config(path: str = "") -> Dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONSTRUCT_CONFIG))
    if not path:
        return config
    user = json.loads(Path(path).read_text(encoding="utf-8"))
    for section, values in user.items():
        if section not in config or not isinstance(values, Mapping):
            raise ValueError(f"Unknown or invalid construct-config section: {section}")
        config[section].update({str(k): float(v) for k, v in values.items()})
    return config


def _weighted_score_from_existing(df: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    missing = [c for c in weights if c not in df.columns]
    if missing:
        raise KeyError(f"Construct config references missing components: {missing}")
    vals = df[list(weights)].apply(pd.to_numeric, errors="coerce")
    w = pd.Series(weights, dtype=float)
    denom = vals.notna().mul(w, axis=1).sum(axis=1)
    return vals.mul(w, axis=1).sum(axis=1, skipna=True) / denom.replace(0, np.nan)

def apply_construct_config(scores: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    out = scores.copy()
    out["observed_threat_proxy"] = (5 * _weighted_score_from_existing(out, config["threat_weights"])).clip(0, 5)
    out["preventive_engagement_proxy"] = (5 * _weighted_score_from_existing(out, config["engagement_weights"])).clip(0, 5)
    out["structural_barrier_proxy"] = (5 * _weighted_score_from_existing(out, config["barrier_weights"])).clip(0, 5)
    out["healthcare_cue_proxy"] = (5 * _weighted_score_from_existing(out, config["cue_weights"])).clip(0, 5)
    out["navigation_self_efficacy_proxy"] = (5 * _weighted_score_from_existing(out, config["self_efficacy_weights"])).clip(0, 5)
    mvals = pd.DataFrame({
        "observed_threat_proxy": out["observed_threat_proxy"],
        "preventive_engagement_proxy": out["preventive_engagement_proxy"],
    })
    mw = pd.Series(config["motivation_weights"], dtype=float)
    out["motivation_score"] = mvals.mul(mw, axis=1).sum(axis=1, skipna=True) / mvals.notna().mul(mw, axis=1).sum(axis=1).replace(0, np.nan)
    cvals = pd.DataFrame({
        "reversed_structural_barrier": 5 - out["structural_barrier_proxy"],
        "navigation_self_efficacy_proxy": out["navigation_self_efficacy_proxy"],
    })
    cw = pd.Series(config["capability_weights"], dtype=float)
    out["capability_score"] = cvals.mul(cw, axis=1).sum(axis=1, skipna=True) / cvals.notna().mul(cw, axis=1).sum(axis=1).replace(0, np.nan)
    out["activation_score"] = out["healthcare_cue_proxy"]
    return out

PATTERN_THEORY_ORDER = {
    "High Motivation / High Capability / Strong Cue": 1,
    "High Motivation / High Capability / Weak Cue": 2,
    "High Motivation / Low Capability / Strong Cue": 3,
    "High Motivation / Low Capability / Weak Cue": 4,
    "Low Motivation / High Capability / Strong Cue": 5,
    "Low Motivation / High Capability / Weak Cue": 6,
    "Low Motivation / Low Capability / Strong Cue": 7,
    "Low Motivation / Low Capability / Weak Cue": 8,
}

PATTERN_TENDENCY = {
    "High Motivation / High Capability / Strong Cue": "very high theory-based tendency",
    "High Motivation / High Capability / Weak Cue": "high tendency with weaker activation",
    "High Motivation / Low Capability / Strong Cue": "motivated and activated but constrained",
    "High Motivation / Low Capability / Weak Cue": "mixed tendency; motivation may not become action",
    "Low Motivation / High Capability / Strong Cue": "cue-activated but weakly motivated tendency",
    "Low Motivation / High Capability / Weak Cue": "generally low tendency",
    "Low Motivation / Low Capability / Strong Cue": "low tendency despite contact opportunities",
    "Low Motivation / Low Capability / Weak Cue": "very low theory-based tendency",
}

INVALID_DEFAULT = {7, 8, 9, 97, 98, 99}

HEALTH_STATUS_LABELS = {1: "excellent", 2: "very good", 3: "good", 4: "fair", 5: "poor"}
SEX_LABELS = {1: "male", 2: "female"}
REGION_LABELS = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}
BMI_LABELS = {1: "underweight", 2: "healthy weight", 3: "overweight", 4: "obese"}
SMOKING_LABELS = {1: "every day", 2: "some days", 3: "not at all", 4: "other/former category"}
LAST_VISIT_LABELS = {
    0: "never", 1: "within past year", 2: "1 to under 2 years", 3: "2 to under 3 years",
    4: "3 to under 5 years", 5: "5 to under 10 years", 6: "10 or more years",
}
USUAL_PLACE_LABELS = {1: "one usual place", 2: "no usual place", 3: "more than one usual place"}
USUAL_KIND_LABELS = {
    1: "doctor office/health center", 2: "urgent care or retail clinic", 3: "hospital ER",
    4: "VA facility", 5: "other place", 6: "no single place used most often",
}
COMM_DIFFICULTY_LABELS = {1: "none", 2: "some", 3: "a lot", 4: "cannot do at all"}
EDUCATION_LABELS = {
    0: "never attended/kindergarten", 1: "grades 1-11", 2: "12th grade without diploma",
    3: "GED", 4: "high school graduate", 5: "some college", 6: "technical associate",
    7: "academic associate", 8: "bachelor", 9: "master", 10: "professional/doctoral",
}
POVERTY_RATIO_LABELS = {i: str(i) for i in range(1, 15)}
CHRONIC_LABELS = {
    "HYPEV_A": "hypertension", "CHDEV_A": "coronary heart disease", "ANGEV_A": "angina",
    "MIEV_A": "heart attack", "STREV_A": "stroke", "ASTILL_A": "current asthma",
    "CANEV_A": "cancer history", "DIBEV_A": "diabetes", "COPDEV_A": "COPD",
    "KIDWEAKEV_A": "weak kidneys", "LIVEREV_A": "liver condition",
}

# Features available to the baseline script. These contain no other-vaccine variables.
ML_NUMERIC_FEATURES = ["AGEP_A", "HINOTMYR_A", "RETAILHC12MTC_A", "URGCC12MTC_A", "EMERG12MTC_A"]
ML_BINARY_FEATURES = sorted(set(
    CHRONIC_VARS
    + ["ANYDIFF_A", "DISAB3_A", "HLTHCOND_A"]
    + [
        "HICOV_A", "NOTCOV_A", "HINOTYR_A", "RSNHICOST_A", "HISTOPCOST_A",
        "MEDDL12M_A", "MEDNG12M_A", "RXDL12M_A", "RXDG12M_A", "PAYBLL12M_A",
        "PAYNOBLLNW_A", "TRANSPOR_A", "PRDEDUC1_A", "PRDEDUC2_A", "WELLNESS_A",
        "VIRAPP12M_A", "HOSPONGT_A", "ACCSSINT_A", "ACCSSHOM_A", "HITLOOK_A",
        "HITCOMM_A", "HITTEST_A",
    ]
))
ML_CATEGORICAL_FEATURES = [
    "SEX_A", "EDUCP_A", "RATCAT_A", "REGION", "PHSTAT_A", "BMICAT_A", "SMKCIGST_A",
    "PAYWORRY_A", "COMDIFF_A", "LASTDR_A", "WELLVIS_A", "USUALPL_A", "USPLKIND_A",
]
ML_FEATURES = ML_NUMERIC_FEATURES + ML_BINARY_FEATURES + ML_CATEGORICAL_FEATURES


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def valid_numeric(series: pd.Series, valid: Optional[Iterable[float]] = None,
                  invalid: Iterable[float] = INVALID_DEFAULT) -> pd.Series:
    x = numeric(series)
    x = x.mask(x.isin(list(invalid)))
    if valid is not None:
        x = x.where(x.isin(list(valid)))
    return x


def yes_no(series: pd.Series) -> pd.Series:
    return valid_numeric(series, valid=[1, 2]).map({1: 1.0, 2: 0.0})


def weighted_available_average(components: Mapping[str, pd.Series],
                               weights: Mapping[str, float]) -> Tuple[pd.Series, pd.Series, pd.Series]:
    values = pd.DataFrame(components)
    w = pd.Series(weights, dtype=float)
    numerator = values.mul(w, axis=1).sum(axis=1, skipna=True)
    denominator = values.notna().mul(w, axis=1).sum(axis=1)
    score = numerator / denominator.replace(0, np.nan)
    observed_n = values.notna().sum(axis=1)
    coverage = values.notna().mul(w, axis=1).sum(axis=1) / float(w.sum())
    return score, observed_n, coverage


def evidence_strength(observed_n: pd.Series, strong_min: int, moderate_min: int) -> pd.Series:
    return pd.Series(np.select(
        [observed_n >= strong_min, observed_n >= moderate_min, observed_n > 0],
        ["strong", "moderate", "weak"], default="none"), index=observed_n.index)


def code_label(value: Any, mapping: Mapping[int, str]) -> Optional[str]:
    try:
        if pd.isna(value):
            return None
        iv = int(float(value))
        return mapping.get(iv, f"code {iv}")
    except Exception:
        return None


def yn(value: Any) -> Optional[bool]:
    try:
        iv = int(float(value))
        if iv == 1:
            return True
        if iv == 2:
            return False
    except Exception:
        pass
    return None


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def clean_text(value: Any, max_chars: int = 500) -> str:
    text = " ".join(str(value or "").strip().split())
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


def active_chronic_conditions(raw_row: pd.Series) -> List[str]:
    return [label for var, label in CHRONIC_LABELS.items() if yn(raw_row.get(var)) is True]


def build_scores(raw: pd.DataFrame) -> pd.DataFrame:
    """Build no-prior-vaccine HBM proxies. No excluded vaccine column is read here."""
    forbidden_present = EXCLUDED_VACCINE_COLUMNS.intersection(raw.columns)
    # Presence in the raw DataFrame is allowed, but no line below references those columns.
    _ = forbidden_present

    out = pd.DataFrame(index=raw.index)
    out["vaccinated"] = yes_no(raw[TARGET])
    out["survey_weight"] = numeric(raw[WEIGHT])
    for c in ID_COLUMNS + BACKGROUND_COLUMNS:
        if c in raw:
            out[c] = raw[c]

    age = valid_numeric(raw["AGEP_A"], invalid=[97, 98, 99])

    # 1) Observed threat (0-5)
    age_component = pd.Series(np.nan, index=raw.index, dtype=float)
    age_component[age < 50] = 0.0
    age_component[(age >= 50) & (age < 65)] = 0.45
    age_component[(age >= 65) & (age < 75)] = 0.75
    age_component[age >= 75] = 1.0
    health_component = valid_numeric(raw["PHSTAT_A"], valid=[1, 2, 3, 4, 5]).map(
        {1: 0.0, 2: 0.15, 3: 0.35, 4: 0.75, 5: 1.0})
    chronic_matrix = pd.DataFrame({v: yes_no(raw[v]) for v in CHRONIC_VARS})
    chronic_count = chronic_matrix.sum(axis=1, min_count=1)
    chronic_component = pd.Series(np.nan, index=raw.index, dtype=float)
    chronic_component[chronic_count == 0] = 0.0
    chronic_component[chronic_count == 1] = 0.35
    chronic_component[chronic_count == 2] = 0.60
    chronic_component[chronic_count == 3] = 0.80
    chronic_component[chronic_count >= 4] = 1.0
    functional_matrix = pd.DataFrame({
        "functioning_difficulty": yes_no(raw["ANYDIFF_A"]),
        "disability": yes_no(raw["DISAB3_A"]),
    })
    function_component = functional_matrix.max(axis=1, skipna=True)
    function_component[functional_matrix.notna().sum(axis=1) == 0] = np.nan
    immune_component = yes_no(raw["HLTHCOND_A"])
    threat_raw, threat_n, threat_cov = weighted_available_average(
        {
            "age_risk": age_component,
            "self_rated_health": health_component,
            "chronic_burden": chronic_component,
            "functional_vulnerability": function_component,
            "immune_vulnerability": immune_component,
        },
        {"age_risk": .25, "self_rated_health": .20, "chronic_burden": .35,
         "functional_vulnerability": .10, "immune_vulnerability": .10},
    )
    out["observed_threat_proxy"] = (5 * threat_raw).clip(0, 5)
    out["threat_evidence_strength"] = evidence_strength(threat_n, 4, 3)
    out["threat_evidence_coverage"] = threat_cov
    out["threat_age_component"] = age_component
    out["threat_health_component"] = health_component
    out["threat_chronic_component"] = chronic_component
    out["threat_function_component"] = function_component
    out["threat_immune_component"] = immune_component
    out["chronic_or_risk_count"] = chronic_count

    # 2) Preventive engagement (0-5), replacing prior-vaccine benefit proxy.
    #    It measures observed proactive health-management behavior, not perceived benefit.
    wellness_recency = valid_numeric(raw["WELLVIS_A"], valid=[0, 1, 2, 3, 4, 5, 6]).map(
        {0: 0.0, 1: 1.0, 2: .65, 3: .40, 4: .20, 5: .05, 6: 0.0})
    wellness_indicator = yes_no(raw["WELLNESS_A"])
    preventive_contact = wellness_recency.fillna(wellness_indicator)
    information_seeking = yes_no(raw["HITLOOK_A"])
    online_doctor_communication = yes_no(raw["HITCOMM_A"])
    online_result_review = yes_no(raw["HITTEST_A"])
    engagement_raw, engagement_n, engagement_cov = weighted_available_average(
        {
            "preventive_wellness_engagement": preventive_contact,
            "health_information_seeking": information_seeking,
            "online_doctor_communication": online_doctor_communication,
            "online_result_review": online_result_review,
        },
        {"preventive_wellness_engagement": .45, "health_information_seeking": .20,
         "online_doctor_communication": .20, "online_result_review": .15},
    )
    out["preventive_engagement_proxy"] = (5 * engagement_raw).clip(0, 5)
    out["engagement_evidence_strength"] = evidence_strength(engagement_n, 4, 2)
    out["engagement_evidence_coverage"] = engagement_cov
    out["engagement_wellness_component"] = preventive_contact
    out["engagement_information_component"] = information_seeking
    out["engagement_doctor_communication_component"] = online_doctor_communication
    out["engagement_result_review_component"] = online_result_review

    # 3) Structural barriers (0-5; higher = more barriers)
    current_uninsured_matrix = pd.DataFrame({
        "hicov_no": 1 - yes_no(raw["HICOV_A"]),
        "notcov_yes": yes_no(raw["NOTCOV_A"]),
    })
    current_uninsured = current_uninsured_matrix.max(axis=1, skipna=True)
    current_uninsured[current_uninsured_matrix.notna().sum(axis=1) == 0] = np.nan
    months_uninsured = valid_numeric(raw["HINOTMYR_A"], invalid=[97, 98, 99]).clip(0, 12) / 12.0
    insurance_matrix = pd.DataFrame({
        "current_uninsured": current_uninsured,
        "past_year_uninsured": yes_no(raw["HINOTYR_A"]) * .70,
        "months_uninsured": months_uninsured,
        "coverage_not_affordable": yes_no(raw["RSNHICOST_A"]),
        "coverage_stopped_cost": yes_no(raw["HISTOPCOST_A"]) * .50,
    })
    insurance_component = insurance_matrix.max(axis=1, skipna=True)
    insurance_component[insurance_matrix.notna().sum(axis=1) == 0] = np.nan
    unmet_matrix = pd.DataFrame({
        "needed_care_not_received": yes_no(raw["MEDNG12M_A"]),
        "delayed_medical_care": yes_no(raw["MEDDL12M_A"]) * .80,
        "needed_rx_not_received": yes_no(raw["RXDG12M_A"]) * .80,
        "delayed_rx": yes_no(raw["RXDL12M_A"]) * .60,
    })
    unmet_component = unmet_matrix.max(axis=1, skipna=True)
    unmet_component[unmet_matrix.notna().sum(axis=1) == 0] = np.nan
    pay_worry = valid_numeric(raw["PAYWORRY_A"], valid=[1, 2, 3]).map({1: .70, 2: .35, 3: 0.0})
    financial_matrix = pd.DataFrame({
        "unable_pay_bills": yes_no(raw["PAYNOBLLNW_A"]),
        "problems_paying_bills": yes_no(raw["PAYBLL12M_A"]) * .70,
        "medical_bill_worry": pay_worry,
        "deductible_plan1": yes_no(raw["PRDEDUC1_A"]) * .20,
        "deductible_plan2": yes_no(raw["PRDEDUC2_A"]) * .20,
    })
    financial_component = financial_matrix.max(axis=1, skipna=True)
    financial_component[financial_matrix.notna().sum(axis=1) == 0] = np.nan
    transportation_component = yes_no(raw["TRANSPOR_A"])
    communication_constraint = valid_numeric(raw["COMDIFF_A"], valid=[1, 2, 3, 4]).map(
        {1: 0.0, 2: .33, 3: .67, 4: 1.0})
    barrier_raw, barrier_n, barrier_cov = weighted_available_average(
        {
            "insurance_instability": insurance_component,
            "cost_related_unmet_care": unmet_component,
            "medical_financial_stress": financial_component,
            "transportation_constraint": transportation_component,
            "communication_constraint": communication_constraint,
        },
        {"insurance_instability": .25, "cost_related_unmet_care": .35,
         "medical_financial_stress": .20, "transportation_constraint": .10,
         "communication_constraint": .10},
    )
    out["structural_barrier_proxy"] = (5 * barrier_raw).clip(0, 5)
    out["barrier_evidence_strength"] = evidence_strength(barrier_n, 4, 3)
    out["barrier_evidence_coverage"] = barrier_cov
    out["barrier_insurance_component"] = insurance_component
    out["barrier_cost_care_component"] = unmet_component
    out["barrier_financial_component"] = financial_component
    out["barrier_transport_component"] = transportation_component
    out["barrier_communication_component"] = communication_constraint

    # 4) Healthcare cue opportunity (0-5). Wellness behavior is deliberately not
    #    included here, because it now belongs to preventive engagement.
    doctor_recency = valid_numeric(raw["LASTDR_A"], valid=[0, 1, 2, 3, 4, 5, 6]).map(
        {0: 0.0, 1: 1.0, 2: .50, 3: .25, 4: .10, 5: 0.0, 6: 0.0})
    retail_count = valid_numeric(raw["RETAILHC12MTC_A"], valid=[0, 1, 2, 3, 4, 5])
    retail_component = retail_count.map({0: 0.0, 1: .50, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0})
    virtual_component = yes_no(raw["VIRAPP12M_A"]) * .60
    retail_virtual = pd.concat([retail_component, virtual_component], axis=1).max(axis=1, skipna=True)
    retail_virtual[pd.concat([retail_component, virtual_component], axis=1).notna().sum(axis=1) == 0] = np.nan
    urgent = valid_numeric(raw["URGCC12MTC_A"], valid=[0, 1, 2, 3, 4, 5])
    emergency = valid_numeric(raw["EMERG12MTC_A"], valid=[0, 1, 2, 3, 4])
    acute_matrix = pd.DataFrame({
        "urgent": (urgent > 0).where(urgent.notna()).astype(float),
        "emergency": (emergency > 0).where(emergency.notna()).astype(float),
        "hospital": yes_no(raw["HOSPONGT_A"]),
    })
    acute_component = acute_matrix.max(axis=1, skipna=True) * .45
    acute_component[acute_matrix.notna().sum(axis=1) == 0] = np.nan
    usual_opportunity = valid_numeric(raw["USUALPL_A"], valid=[1, 2, 3]).map({1: 1.0, 2: 0.0, 3: .60})
    cue_raw, cue_n, cue_cov = weighted_available_average(
        {
            "recent_doctor_contact": doctor_recency,
            "retail_or_virtual_contact": retail_virtual,
            "acute_care_contact": acute_component,
            "ongoing_care_opportunity": usual_opportunity,
        },
        {"recent_doctor_contact": .45, "retail_or_virtual_contact": .25,
         "acute_care_contact": .10, "ongoing_care_opportunity": .20},
    )
    out["healthcare_cue_proxy"] = (5 * cue_raw).clip(0, 5)
    out["cue_evidence_strength"] = evidence_strength(cue_n, 4, 3)
    out["cue_evidence_coverage"] = cue_cov
    out["cue_doctor_component"] = doctor_recency
    out["cue_retail_virtual_component"] = retail_virtual
    out["cue_acute_component"] = acute_component
    out["cue_usual_care_opportunity_component"] = usual_opportunity

    # 5) Navigation self-efficacy/capacity (0-5). Actual digital engagement was
    #    moved to preventive engagement; here we retain access/capacity only.
    usual_care = usual_opportunity
    care_setting = valid_numeric(raw["USPLKIND_A"], valid=[1, 2, 3, 4, 5, 6]).map(
        {1: 1.0, 2: .70, 3: .20, 4: 1.0, 5: .50, 6: .30})
    internet_matrix = pd.DataFrame({
        "internet_anywhere": yes_no(raw["ACCSSINT_A"]),
        "internet_home": yes_no(raw["ACCSSHOM_A"]),
    })
    internet_access = internet_matrix.mean(axis=1, skipna=True)
    internet_access[internet_matrix.notna().sum(axis=1) == 0] = np.nan
    virtual_experience = yes_no(raw["VIRAPP12M_A"])
    communication_capacity = valid_numeric(raw["COMDIFF_A"], valid=[1, 2, 3, 4]).map(
        {1: 1.0, 2: .67, 3: .33, 4: 0.0})
    efficacy_raw, efficacy_n, efficacy_cov = weighted_available_average(
        {
            "usual_care_access": usual_care,
            "stable_care_setting": care_setting,
            "internet_access": internet_access,
            "virtual_care_experience": virtual_experience,
            "communication_capacity": communication_capacity,
        },
        {"usual_care_access": .35, "stable_care_setting": .15, "internet_access": .20,
         "virtual_care_experience": .10, "communication_capacity": .20},
    )
    out["navigation_self_efficacy_proxy"] = (5 * efficacy_raw).clip(0, 5)
    out["self_efficacy_evidence_strength"] = evidence_strength(efficacy_n, 4, 3)
    out["self_efficacy_evidence_coverage"] = efficacy_cov
    out["selfeff_usual_care_component"] = usual_care
    out["selfeff_care_setting_component"] = care_setting
    out["selfeff_internet_component"] = internet_access
    out["selfeff_virtual_component"] = virtual_experience
    out["selfeff_communication_component"] = communication_capacity

    # Meta dimensions and preliminary pattern scores.
    out["motivation_score"] = out[["observed_threat_proxy", "preventive_engagement_proxy"]].mean(axis=1, skipna=True)
    out["capability_score"] = pd.concat(
        [5 - out["structural_barrier_proxy"], out["navigation_self_efficacy_proxy"]], axis=1
    ).mean(axis=1, skipna=True)
    out["activation_score"] = out["healthcare_cue_proxy"]
    return out


def clean_ml_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Create a fair raw-feature matrix with all vaccine-history variables excluded."""
    x = pd.DataFrame(index=raw.index)
    x["AGEP_A"] = valid_numeric(raw["AGEP_A"], invalid=[97, 98, 99])
    x["HINOTMYR_A"] = valid_numeric(raw["HINOTMYR_A"], invalid=[97, 98, 99])
    x["RETAILHC12MTC_A"] = valid_numeric(raw["RETAILHC12MTC_A"], valid=[0, 1, 2, 3, 4, 5])
    x["URGCC12MTC_A"] = valid_numeric(raw["URGCC12MTC_A"], valid=[0, 1, 2, 3, 4, 5])
    x["EMERG12MTC_A"] = valid_numeric(raw["EMERG12MTC_A"], valid=[0, 1, 2, 3, 4])
    for c in ML_BINARY_FEATURES:
        x[c] = yes_no(raw[c])
    valid_maps = {
        "SEX_A": [1, 2], "EDUCP_A": list(range(0, 11)), "RATCAT_A": list(range(1, 15)),
        "REGION": [1, 2, 3, 4], "PHSTAT_A": [1, 2, 3, 4, 5], "BMICAT_A": [1, 2, 3, 4],
        "SMKCIGST_A": [1, 2, 3, 4], "PAYWORRY_A": [1, 2, 3], "COMDIFF_A": [1, 2, 3, 4],
        "LASTDR_A": [0, 1, 2, 3, 4, 5, 6], "WELLVIS_A": [0, 1, 2, 3, 4, 5, 6],
        "USUALPL_A": [1, 2, 3], "USPLKIND_A": [1, 2, 3, 4, 5, 6],
    }
    for c in ML_CATEGORICAL_FEATURES:
        x[c] = valid_numeric(raw[c], valid=valid_maps[c], invalid=[])
    assert not EXCLUDED_VACCINE_COLUMNS.intersection(x.columns)
    return x


def assign_patterns(df: pd.DataFrame, thresholds: Mapping[str, float]) -> pd.DataFrame:
    out = df.copy()
    m = np.where(out["motivation_score"] >= thresholds["motivation"], "High Motivation", "Low Motivation")
    c = np.where(out["capability_score"] >= thresholds["capability"], "High Capability", "Low Capability")
    a = np.where(out["activation_score"] >= thresholds["activation"], "Strong Cue", "Weak Cue")
    out["motivation_level"] = m
    out["capability_level"] = c
    out["activation_level"] = a
    out["hbm8_pattern"] = pd.Series(m, index=out.index) + " / " + c + " / " + a
    out["hbm8_theory_order"] = out["hbm8_pattern"].map(PATTERN_THEORY_ORDER)
    return out


def preliminary_patterns(scores: pd.DataFrame) -> pd.Series:
    temp = assign_patterns(scores, {"motivation": 2.5, "capability": 2.5, "activation": 2.5})
    return temp["hbm8_pattern"]


def allocate_counts(total: int, proportions: Mapping[Any, float]) -> Dict[Any, int]:
    raw = {k: total * float(v) for k, v in proportions.items()}
    floor = {k: int(math.floor(v)) for k, v in raw.items()}
    remainder = total - sum(floor.values())
    for k in sorted(raw, key=lambda z: raw[z] - floor[z], reverse=True)[:remainder]:
        floor[k] += 1
    return floor


def sample_by_class_pattern(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.sample_size <= 0 or args.sample_size >= len(df):
        return df.sample(frac=1, random_state=args.random_seed).copy()
    class_rates = df["vaccinated"].value_counts(normalize=True).to_dict()
    if args.class_sampling == "balanced":
        class_props = {0.0: .5, 1.0: .5}
    elif args.class_sampling == "custom":
        p = float(np.clip(args.positive_fraction, .01, .99))
        class_props = {0.0: 1-p, 1.0: p}
    else:
        class_props = {0.0: float(class_rates.get(0.0, 0)), 1.0: float(class_rates.get(1.0, 0))}
    class_counts = allocate_counts(args.sample_size, class_props)
    pieces: List[pd.DataFrame] = []
    for y in [0.0, 1.0]:
        g = df[df["vaccinated"] == y]
        n = min(class_counts.get(y, 0), len(g))
        if n <= 0:
            continue
        if args.preserve_pattern_within_class:
            pcounts = allocate_counts(n, g["preliminary_pattern"].value_counts(normalize=True).to_dict())
            chosen = []
            for j, (pattern, nn) in enumerate(pcounts.items()):
                pg = g[g["preliminary_pattern"] == pattern]
                if nn > 0 and len(pg):
                    chosen.append(pg.sample(n=min(nn, len(pg)), random_state=args.random_seed + 100*int(y) + j))
            part = pd.concat(chosen) if chosen else g.iloc[:0]
            if len(part) < n:
                pool = g.drop(index=part.index)
                part = pd.concat([part, pool.sample(n=min(n-len(part), len(pool)), random_state=args.random_seed+700+int(y))])
            pieces.append(part.iloc[:n])
        else:
            pieces.append(g.sample(n=n, random_state=args.random_seed + int(y)))
    sampled = pd.concat(pieces)
    if len(sampled) < args.sample_size:
        pool = df.drop(index=sampled.index)
        sampled = pd.concat([sampled, pool.sample(n=min(args.sample_size-len(sampled), len(pool)), random_state=args.random_seed+999)])
    return sampled.sample(frac=1, random_state=args.random_seed).copy()


def safe_split(indices: np.ndarray, strata: pd.Series, left_size: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if left_size <= 0:
        return np.array([], dtype=int), indices.copy()
    if left_size >= len(indices):
        return indices.copy(), np.array([], dtype=int)
    s = strata.loc[indices]
    use = s.value_counts().min() >= 2 and left_size >= s.nunique() and len(indices)-left_size >= s.nunique()
    try:
        left, right = train_test_split(indices, train_size=left_size, random_state=seed,
                                       stratify=s if use else None)
    except ValueError:
        left, right = train_test_split(indices, train_size=left_size, random_state=seed, stratify=None)
    return np.array(left, dtype=int), np.array(right, dtype=int)


def make_splits(df: pd.DataFrame, args: argparse.Namespace) -> Dict[str, List[int]]:
    ratios = np.array([args.memory_ratio, args.calibration_ratio, args.test_ratio], dtype=float)
    if np.any(ratios < 0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("memory/calibration/test ratios must be nonnegative and sum to 1")
    n = len(df)
    counts = allocate_counts(n, {"memory": ratios[0], "calibration": ratios[1], "test": ratios[2]})
    idx = np.arange(n)
    strata = df["vaccinated"].astype(str) + "|" + df["preliminary_pattern"].astype(str)
    memory, rest = safe_split(idx, strata, counts["memory"], args.random_seed + 1)
    calibration, test = safe_split(rest, strata, counts["calibration"], args.random_seed + 2)
    return {"memory": memory.tolist(), "calibration": calibration.tolist(), "test": test.tolist()}


def fit_thresholds(memory_df: pd.DataFrame, args: argparse.Namespace) -> Dict[str, float]:
    if args.pattern_threshold_mode == "fixed":
        return {"motivation": args.fixed_motivation_threshold,
                "capability": args.fixed_capability_threshold,
                "activation": args.fixed_activation_threshold}
    return {"motivation": float(memory_df["motivation_score"].median()),
            "capability": float(memory_df["capability_score"].median()),
            "activation": float(memory_df["activation_score"].median())}


def fit_pattern_base_rates(memory_df: pd.DataFrame, prior_strength: float) -> Dict[str, Dict[str, float]]:
    overall = float(memory_df["vaccinated"].mean())
    out: Dict[str, Dict[str, float]] = {}
    for p in PATTERN_THEORY_ORDER:
        g = memory_df[memory_df["hbm8_pattern"] == p]
        n = len(g); yes = float(g["vaccinated"].sum())
        raw = yes/n if n else overall
        smoothed = (yes + prior_strength*overall)/(n+prior_strength) if n+prior_strength > 0 else overall
        out[p] = {"n": int(n), "yes": yes, "raw_rate": raw, "smoothed_rate": smoothed}
    out["__overall__"] = {"n": len(memory_df), "yes": float(memory_df["vaccinated"].sum()),
                          "raw_rate": overall, "smoothed_rate": overall}
    return out


def row_pattern_anchor(row: pd.Series, base_rates: Mapping[str, Mapping[str, float]],
                       memory_df: pd.DataFrame, prior_strength: float, leave_one_out: bool) -> Tuple[float, int]:
    p = str(row["hbm8_pattern"]); info = base_rates.get(p, base_rates["__overall__"])
    if not leave_one_out:
        return 100*float(info["smoothed_rate"]), int(info["n"])
    n = int(info["n"]); yes = float(info["yes"]); actual = int(row["vaccinated"])
    if n <= 1 or len(memory_df) <= 1:
        return 100*float(base_rates["__overall__"]["raw_rate"]), max(n-1, 0)
    overall_loo = (float(memory_df["vaccinated"].sum()) - actual)/(len(memory_df)-1)
    smoothed = (yes-actual + prior_strength*overall_loo)/(n-1+prior_strength)
    return 100*float(smoothed), n-1


def build_observed_profile(raw_row: pd.Series, s: pd.Series, include_sensitive: bool) -> Dict[str, Any]:
    age = json_safe(valid_numeric(pd.Series([raw_row.get("AGEP_A")]), invalid=[97,98,99]).iloc[0])
    profile: Dict[str, Any] = {
        "constructs": {
            "observed_threat": {"score": json_safe(s.get("observed_threat_proxy")), "strength": s.get("threat_evidence_strength"),
                "components": {"age_risk": json_safe(s.get("threat_age_component")), "self_rated_health": json_safe(s.get("threat_health_component")),
                    "chronic_burden": json_safe(s.get("threat_chronic_component")), "functional_vulnerability": json_safe(s.get("threat_function_component")),
                    "immune_vulnerability": json_safe(s.get("threat_immune_component"))}},
            "preventive_engagement": {"score": json_safe(s.get("preventive_engagement_proxy")), "strength": s.get("engagement_evidence_strength"),
                "components": {"wellness_engagement": json_safe(s.get("engagement_wellness_component")),
                    "health_information_seeking": json_safe(s.get("engagement_information_component")),
                    "online_doctor_communication": json_safe(s.get("engagement_doctor_communication_component")),
                    "online_result_review": json_safe(s.get("engagement_result_review_component"))}},
            "structural_barriers": {"score": json_safe(s.get("structural_barrier_proxy")), "strength": s.get("barrier_evidence_strength"),
                "components": {"insurance": json_safe(s.get("barrier_insurance_component")), "cost_unmet": json_safe(s.get("barrier_cost_care_component")),
                    "financial": json_safe(s.get("barrier_financial_component")), "transport": json_safe(s.get("barrier_transport_component")),
                    "communication": json_safe(s.get("barrier_communication_component"))}},
            "healthcare_cues": {"score": json_safe(s.get("healthcare_cue_proxy")), "strength": s.get("cue_evidence_strength"),
                "components": {"doctor_contact": json_safe(s.get("cue_doctor_component")), "retail_virtual": json_safe(s.get("cue_retail_virtual_component")),
                    "acute_contact": json_safe(s.get("cue_acute_component")), "usual_care_opportunity": json_safe(s.get("cue_usual_care_opportunity_component"))}},
            "navigation_self_efficacy": {"score": json_safe(s.get("navigation_self_efficacy_proxy")), "strength": s.get("self_efficacy_evidence_strength"),
                "components": {"usual_care": json_safe(s.get("selfeff_usual_care_component")), "care_setting": json_safe(s.get("selfeff_care_setting_component")),
                    "internet_access": json_safe(s.get("selfeff_internet_component")), "virtual_experience": json_safe(s.get("selfeff_virtual_component")),
                    "communication_capacity": json_safe(s.get("selfeff_communication_component"))}},
        },
        "meta_dimensions": {"motivation": json_safe(s.get("motivation_score")), "capability": json_safe(s.get("capability_score")),
                            "activation": json_safe(s.get("activation_score"))},
        "hbm8_pattern": s.get("hbm8_pattern"),
        "health_risk_context": {"age": age, "self_rated_health": code_label(raw_row.get("PHSTAT_A"), HEALTH_STATUS_LABELS),
            "chronic_conditions": active_chronic_conditions(raw_row), "functional_difficulty": yn(raw_row.get("ANYDIFF_A")),
            "disability": yn(raw_row.get("DISAB3_A")), "bmi": code_label(raw_row.get("BMICAT_A"), BMI_LABELS),
            "smoking": code_label(raw_row.get("SMKCIGST_A"), SMOKING_LABELS)},
        "access_and_cost": {"insured": yn(raw_row.get("HICOV_A")), "uninsured_past_year": yn(raw_row.get("HINOTYR_A")),
            "coverage_unaffordable": yn(raw_row.get("RSNHICOST_A")), "delayed_care_cost": yn(raw_row.get("MEDDL12M_A")),
            "needed_care_not_received_cost": yn(raw_row.get("MEDNG12M_A")), "medical_bill_problem": yn(raw_row.get("PAYBLL12M_A")),
            "transportation_barrier": yn(raw_row.get("TRANSPOR_A")), "communication_difficulty": code_label(raw_row.get("COMDIFF_A"), COMM_DIFFICULTY_LABELS)},
        "healthcare_contact": {"last_doctor_visit": code_label(raw_row.get("LASTDR_A"), LAST_VISIT_LABELS),
            "wellness_visit": code_label(raw_row.get("WELLVIS_A"), LAST_VISIT_LABELS), "retail_visit_category": json_safe(raw_row.get("RETAILHC12MTC_A")),
            "virtual_appointment": yn(raw_row.get("VIRAPP12M_A")), "urgent_visit_category": json_safe(raw_row.get("URGCC12MTC_A")),
            "emergency_visit_category": json_safe(raw_row.get("EMERG12MTC_A")), "hospitalized": yn(raw_row.get("HOSPONGT_A"))},
        "navigation_and_engagement": {"usual_place": code_label(raw_row.get("USUALPL_A"), USUAL_PLACE_LABELS),
            "usual_setting": code_label(raw_row.get("USPLKIND_A"), USUAL_KIND_LABELS), "internet": yn(raw_row.get("ACCSSINT_A")),
            "looked_up_health_information": yn(raw_row.get("HITLOOK_A")), "communicated_with_doctor_online": yn(raw_row.get("HITCOMM_A")),
            "viewed_test_results_online": yn(raw_row.get("HITTEST_A"))},
        "background": {"sex": code_label(raw_row.get("SEX_A"), SEX_LABELS), "education": code_label(raw_row.get("EDUCP_A"), EDUCATION_LABELS),
            "income_to_poverty_category": code_label(raw_row.get("RATCAT_A"), POVERTY_RATIO_LABELS), "region": code_label(raw_row.get("REGION"), REGION_LABELS)},
        "explicit_exclusions": "No other-vaccine-history variables are included.",
    }
    if include_sensitive:
        profile["background"]["race_code"] = json_safe(raw_row.get("RACEALLP_A"))
        profile["background"]["hispanic_group_code"] = json_safe(raw_row.get("HISPALLP_A"))
    return profile


# Structured observable reasoning trace; not an unrestricted hidden chain of thought.
DECISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "construct_interpretation": {
            "type": "object",
            "properties": {k: {"type": "string", "enum": ["low", "moderate", "high"]} for k in
                           ["threat", "preventive_engagement", "barriers", "self_efficacy", "cues"]},
            "required": ["threat", "preventive_engagement", "barriers", "self_efficacy", "cues"],
            "additionalProperties": False,
        },
        "reasoning_trace": {
            "type": "object",
            "properties": {
                "stage_1_evidence_synthesis": {"type": "string"},
                "stage_2_pattern_interpretation": {"type": "string"},
                "stage_3_residual_context": {"type": "string"},
                "stage_4_decision_mapping": {"type": "string"},
            },
            "required": ["stage_1_evidence_synthesis", "stage_2_pattern_interpretation", "stage_3_residual_context", "stage_4_decision_mapping"],
            "additionalProperties": False,
        },
        "residual_adjustment": {"type": "integer", "minimum": -25, "maximum": 25},
        "probability_yes": {"type": "number", "minimum": 0, "maximum": 100},
        "deviation_direction": {"type": "string", "enum": ["lower_than_pattern", "similar_to_pattern", "higher_than_pattern"]},
        "dominant_observed_factors": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 6},
        "memory_application": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["construct_interpretation", "reasoning_trace", "residual_adjustment", "probability_yes",
                 "deviation_direction", "dominant_observed_factors", "memory_application", "confidence"],
    "additionalProperties": False,
}

REFLECTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "correction_direction": {"type": "string", "enum": ["increase", "decrease"]},
        "error_type": {"type": "string", "enum": ["underprediction", "overprediction"]},
        "failure_stage": {"type": "string", "enum": ["construct_inference", "pattern_interpretation", "context_integration", "decision_mapping", "memory_misapplication"]},
        "incorrect_assumption": {"type": "string"},
        "corrected_reasoning_path": {
            "type": "object",
            "properties": {
                "evidence_reinterpretation": {"type": "string"},
                "pattern_exception": {"type": "string"},
                "decision_correction": {"type": "string"},
            },
            "required": ["evidence_reinterpretation", "pattern_exception", "decision_correction"],
            "additionalProperties": False,
        },
        "supporting_variables": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 7},
        "correction_rule": {"type": "string"},
        "applicability_conditions": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 6},
        "contradiction_conditions": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5},
        "retrieval_keys": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 8},
        "reflection_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "estimated_memory_value": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["correction_direction", "error_type", "failure_stage", "incorrect_assumption", "corrected_reasoning_path",
                 "supporting_variables", "correction_rule", "applicability_conditions", "contradiction_conditions",
                 "retrieval_keys", "reflection_confidence", "estimated_memory_value"],
    "additionalProperties": False,
}

DECISION_SYSTEM = """You are a public-health behavioral analyst using a theory-guided HBM-style framework.
The input contains five OBSERVED PROXIES, three meta-dimensions, an eight-pattern training prior, observed context, and sometimes reflective memories.

Theory path:
1. Threat and preventive engagement jointly inform motivation.
2. Low structural barriers and navigation self-efficacy jointly inform capability.
3. Healthcare contact opportunities inform activation.
4. Motivation, capability, and activation jointly shape influenza-vaccination behavior.

Important constraints:
- Preventive engagement is an observed proxy based on wellness and health-management behavior; it is not a direct perceived-benefit scale.
- No other-vaccine-history information is available. Never infer or mention COVID, pneumonia, shingles, hepatitis, or any other vaccine history.
- Produce only a concise structured reasoning trace grounded in supplied variables. Do not invent private beliefs, physician recommendations, intentions, or reminders.
- The HBM8 base rate is the anchor. Use a residual adjustment only for observed within-pattern exceptions.
- Apply a memory only if applicability conditions match and contradiction conditions do not match.
- The returned probability must equal the base probability plus the residual adjustment, clipped to 0-100.
Return only the required JSON object."""

REFLECTION_SYSTEM = """You are building a FLARE-inspired reflective memory from a high-confidence prediction error.
Diagnose which observable reasoning stage failed, then rewrite a concise corrected path from evidence to pattern exception to decision.

Constraints:
- Ground every statement in supplied observed variables.
- Never invent beliefs, recommendations, intentions, or other-vaccine history.
- Do not merely restate the HBM8 pattern.
- A useful memory must include applicability conditions, contradiction conditions, and retrieval keys.
- The required correction direction and error type must be followed exactly.
Return only the required JSON object."""


def config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def append_jsonl(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(obj), ensure_ascii=False, default=json_safe) + "\n")


def load_latest_jsonl(path: Path, expected_hash: str) -> Dict[int, Dict[str, Any]]:
    latest: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as f:
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
    i = int(getattr(u, "input_tokens", 0) or 0); o = int(getattr(u, "output_tokens", 0) or 0)
    return {"input_tokens": i, "output_tokens": o, "total_tokens": int(getattr(u, "total_tokens", i+o) or i+o)}


async def call_structured(client: Any, semaphore: asyncio.Semaphore, *, model: str, name: str,
                          schema: Dict[str, Any], system: str, user: str, max_tokens: int,
                          temperature: float, retries: int) -> Tuple[Dict[str, Any], str, Dict[str, int], str]:
    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            async with semaphore:
                r = await client.responses.create(
                    model=model,
                    input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    text={"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema}},
                    max_output_tokens=max_tokens, temperature=temperature, store=False,
                )
            text = str(getattr(r, "output_text", "") or "").strip()
            if not text:
                raise RuntimeError("empty output")
            return json.loads(text), text, usage_from_response(r), str(getattr(r, "_request_id", "") or "")
        except Exception as exc:
            last = exc
            if attempt >= retries:
                break
            await asyncio.sleep(min(20, 1.5 * 2**attempt))
    raise RuntimeError(f"Structured call failed: {last}") from last


def validate_decision(obj: Mapping[str, Any], base: float) -> Dict[str, Any]:
    adj = int(np.clip(int(obj["residual_adjustment"]), -25, 25))
    p = float(np.clip(base + adj, 0, 100))
    expected = "higher_than_pattern" if adj > 2 else "lower_than_pattern" if adj < -2 else "similar_to_pattern"
    trace = {k: clean_text(v, 360) for k, v in obj["reasoning_trace"].items()}
    return {
        "construct_interpretation": dict(obj["construct_interpretation"]),
        "reasoning_trace": trace,
        "residual_adjustment": adj,
        "raw_probability_yes": float(obj["probability_yes"]),
        "probability_yes": p,
        "probability_consistency_corrected": abs(float(obj["probability_yes"]) - p) > 1,
        "deviation_direction": expected,
        "raw_deviation_direction": str(obj["deviation_direction"]),
        "dominant_observed_factors": [clean_text(x, 150) for x in obj["dominant_observed_factors"]],
        "memory_application": [clean_text(x, 180) for x in obj["memory_application"]],
        "confidence": float(np.clip(float(obj["confidence"]), 0, 1)),
    }


def expected_reflection(actual: int, p: float) -> Tuple[str, str]:
    if actual == 1 and p < 50:
        return "increase", "underprediction"
    if actual == 0 and p >= 50:
        return "decrease", "overprediction"
    raise ValueError("not an error")


def validate_reflection(obj: Mapping[str, Any], direction: str, error_type: str) -> Dict[str, Any]:
    if obj["correction_direction"] != direction or obj["error_type"] != error_type:
        raise ValueError("reflection direction/error type mismatch")
    supporting = [clean_text(x, 160) for x in obj["supporting_variables"] if str(x).strip()]
    if len(supporting) < 2:
        raise ValueError("at least two supporting variables required")
    return {
        "correction_direction": direction,
        "error_type": error_type,
        "failure_stage": str(obj["failure_stage"]),
        "incorrect_assumption": clean_text(obj["incorrect_assumption"], 300),
        "corrected_reasoning_path": {k: clean_text(v, 360) for k, v in obj["corrected_reasoning_path"].items()},
        "supporting_variables": supporting[:7],
        "correction_rule": clean_text(obj["correction_rule"], 420),
        "applicability_conditions": [clean_text(x, 180) for x in obj["applicability_conditions"][:6]],
        "contradiction_conditions": [clean_text(x, 180) for x in obj["contradiction_conditions"][:5]],
        "retrieval_keys": [clean_text(x, 100) for x in obj["retrieval_keys"][:8]],
        "reflection_confidence": float(np.clip(float(obj["reflection_confidence"]), 0, 1)),
        "estimated_memory_value": float(np.clip(float(obj["estimated_memory_value"]), 0, 1)),
    }


def build_decision_prompt(profile: Mapping[str, Any], base: float, pattern_n: int,
                          memories: Sequence[Mapping[str, Any]]) -> str:
    memory_payload = []
    for m in memories:
        memory_payload.append({
            "similarity": round(float(m.get("similarity", 0)), 3), "source_pattern": m.get("pattern"),
            "direction": m.get("correction_direction"), "failure_stage": m.get("failure_stage"),
            "correction_rule": m.get("correction_rule"), "corrected_reasoning_path": m.get("corrected_reasoning_path"),
            "applicability_conditions": m.get("applicability_conditions"), "contradiction_conditions": m.get("contradiction_conditions"),
            "retrieval_keys": m.get("retrieval_keys"), "memory_value": round(float(m.get("memory_value", 0)), 3),
        })
    p = str(profile.get("hbm8_pattern"))
    payload = {
        "pattern_prior": {"pattern": p, "general_tendency": PATTERN_TENDENCY.get(p),
                          "base_probability_yes": round(base, 2), "memory_build_pattern_n": int(pattern_n)},
        "respondent_profile": profile,
        "retrieved_reflective_memories": memory_payload,
    }
    return (
        "Predict influenza vaccination in the past 12 months. Follow the four observable reasoning stages in the output schema.\n"
        "Start from the pattern prior and make only a supported residual adjustment in [-25,25].\n"
        "Do not infer any other-vaccine history.\n\nINPUT JSON\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def build_reflection_prompt(profile: Mapping[str, Any], decision: Mapping[str, Any], actual: int,
                            base: float, direction: str, error_type: str) -> str:
    payload = {
        "required_correction_direction": direction, "required_error_type": error_type,
        "actual_outcome": "YES" if actual else "NO", "pattern_base_probability": round(base, 2),
        "initial_decision": {
            "construct_interpretation": decision["construct_interpretation"],
            "reasoning_trace": decision["reasoning_trace"], "residual_adjustment": decision["residual_adjustment"],
            "probability_yes": decision["probability_yes"], "dominant_observed_factors": decision["dominant_observed_factors"],
        },
        "respondent_profile": profile,
    }
    return (
        "Diagnose this high-confidence no-memory prediction error. Identify the failed reasoning stage and create a reusable corrected path.\n"
        "Do not use or infer any other-vaccine history.\n\nINPUT JSON\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


HBM_SIM_COLUMNS = [
    "observed_threat_proxy", "preventive_engagement_proxy", "structural_barrier_proxy",
    "healthcare_cue_proxy", "navigation_self_efficacy_proxy", "motivation_score",
    "capability_score", "activation_score",
]
RAW_SIM_NUMERIC = [
    "chronic_or_risk_count", "threat_age_component", "threat_health_component", "threat_chronic_component",
    "threat_function_component", "threat_immune_component", "engagement_wellness_component",
    "engagement_information_component", "engagement_doctor_communication_component", "engagement_result_review_component",
    "barrier_insurance_component", "barrier_cost_care_component", "barrier_financial_component",
    "barrier_transport_component", "barrier_communication_component", "cue_doctor_component",
    "cue_retail_virtual_component", "cue_acute_component", "cue_usual_care_opportunity_component",
    "selfeff_usual_care_component", "selfeff_care_setting_component", "selfeff_internet_component",
    "selfeff_virtual_component", "selfeff_communication_component",
]
RAW_SIM_CATEGORICAL = ["SEX_A", "EDUCP_A", "RATCAT_A", "REGION", "BMICAT_A", "SMKCIGST_A"]


@dataclass
class SimilaritySpace:
    hbm_matrix: np.ndarray
    raw_matrix: np.ndarray
    patterns: List[str]

    def similarity(self, q: int, s: int) -> float:
        def cos(a: np.ndarray, b: np.ndarray) -> float:
            d = float(np.linalg.norm(a) * np.linalg.norm(b))
            return 0.0 if d <= 0 else float(np.dot(a, b)/d)
        hbm = (cos(self.hbm_matrix[q], self.hbm_matrix[s]) + 1)/2
        raw = (cos(self.raw_matrix[q], self.raw_matrix[s]) + 1)/2
        same = 1.0 if self.patterns[q] == self.patterns[s] else 0.0
        return float(np.clip(.50*hbm + .35*raw + .15*same, 0, 1))


def dense(x: Any) -> np.ndarray:
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def fit_similarity_space(df: pd.DataFrame, memory_idx: Sequence[int]) -> SimilaritySpace:
    hbm = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    raw = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), RAW_SIM_NUMERIC),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), RAW_SIM_CATEGORICAL),
    ])
    train = df.iloc[list(memory_idx)]
    hbm.fit(train[HBM_SIM_COLUMNS]); raw.fit(train[RAW_SIM_NUMERIC + RAW_SIM_CATEGORICAL])
    return SimilaritySpace(dense(hbm.transform(df[HBM_SIM_COLUMNS])).astype(float),
                           dense(raw.transform(df[RAW_SIM_NUMERIC + RAW_SIM_CATEGORICAL])).astype(float),
                           df["hbm8_pattern"].astype(str).tolist())


@dataclass
class MemoryStore:
    items: List[Dict[str, Any]] = field(default_factory=list)
    similarity_space: Optional[SimilaritySpace] = None
    min_similarity: float = .45
    top_k: int = 3
    same_pattern_only: bool = False
    direction_diversity: bool = True

    def retrieve(self, query_idx: int) -> List[Dict[str, Any]]:
        if not self.items or self.similarity_space is None:
            return []
        qp = self.similarity_space.patterns[query_idx]
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for item in self.items:
            if self.same_pattern_only and str(item.get("pattern")) != qp:
                continue
            sim = self.similarity_space.similarity(query_idx, int(item["source_data_idx"]))
            if sim >= self.min_similarity:
                e = dict(item); e["similarity"] = sim
                score = .70*sim + .30*float(item.get("memory_value", 0))
                scored.append((score, e))
        scored.sort(key=lambda z: z[0], reverse=True)
        if not self.direction_diversity:
            return [x[1] for x in scored[:self.top_k]]
        selected: List[Dict[str, Any]] = []
        used: Set[str] = set()
        for _, e in scored:
            d = str(e.get("correction_direction"))
            if d not in used:
                selected.append(e); used.add(d)
            if len(selected) >= self.top_k:
                return selected
        for _, e in scored:
            if e not in selected:
                selected.append(e)
            if len(selected) >= self.top_k:
                break
        return selected


def tokenize_memory(item: Mapping[str, Any]) -> Set[str]:
    text = " ".join([
        str(item.get("failure_stage", "")), str(item.get("incorrect_assumption", "")),
        str(item.get("correction_rule", "")), " ".join(item.get("supporting_variables", []) or []),
        " ".join(item.get("applicability_conditions", []) or []), " ".join(item.get("retrieval_keys", []) or []),
    ]).lower()
    return set(re.findall(r"[a-z0-9_]+", text))


def jaccard(a: Set[str], b: Set[str]) -> float:
    return 1.0 if not a and not b else len(a & b)/max(1, len(a | b))


def error_confidence(actual: int, p: float) -> float:
    return float(np.clip((50-p)/50 if actual == 1 else (p-50)/50, 0, 1))


def select_reflection_candidates(entries: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    cands = []
    for e in entries:
        if e.get("status") != "ok":
            continue
        actual = int(e["actual"]); p = float(e["probability_yes"])
        if not ((actual == 1 and p <= args.reflection_low_cutoff) or (actual == 0 and p >= args.reflection_high_cutoff)):
            continue
        d, t = expected_reflection(actual, p)
        c = dict(e); c["required_direction"] = d; c["required_error_type"] = t; c["error_confidence"] = error_confidence(actual, p)
        cands.append(c)
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for c in cands:
        buckets.setdefault((str(c["pattern"]), str(c["required_direction"])), []).append(c)
    selected = []
    for b in buckets.values():
        b.sort(key=lambda x: (float(x["error_confidence"]), float(x.get("confidence", 0))), reverse=True)
        selected.extend(b[:args.max_reflection_calls_per_bucket])
    selected.sort(key=lambda x: float(x["error_confidence"]), reverse=True)
    return selected[:args.max_reflection_candidates]


def distill_memory(reflections: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    prelim = []
    for e in reflections:
        if e.get("status") != "ok":
            continue
        r = e["reflection"]
        if r["reflection_confidence"] < args.min_reflection_confidence or r["estimated_memory_value"] < args.min_estimated_memory_value:
            continue
        specificity = min(1.0, (len(r["supporting_variables"]) + len(r["applicability_conditions"])) / 8)
        trace_quality = min(1.0, len(" ".join(r["corrected_reasoning_path"].values())) / 450)
        value = (.30*float(e["error_confidence"]) + .20*specificity + .15*trace_quality
                 + .175*r["reflection_confidence"] + .175*r["estimated_memory_value"])
        item = {
            "source_data_idx": int(e["data_idx"]), "pattern": e["pattern"], "actual": int(e["actual"]),
            "initial_probability_yes": float(e["probability_yes"]), "error_confidence": float(e["error_confidence"]),
            "source_reasoning_trace": e.get("reasoning_trace"), **r, "specificity": specificity,
            "trace_quality": trace_quality, "base_memory_value": value,
        }
        item["tokens"] = tokenize_memory(item); prelim.append(item)
    prelim.sort(key=lambda x: float(x["base_memory_value"]), reverse=True)
    selected = []; counts: Dict[Tuple[str, str], int] = {}
    for item in prelim:
        peers = [x for x in selected if x["correction_direction"] == item["correction_direction"]]
        max_sim = max((jaccard(item["tokens"], x["tokens"]) for x in peers), default=0)
        novelty = 1-max_sim
        if novelty < args.min_memory_novelty:
            continue
        key = (str(item["pattern"]), str(item["correction_direction"]))
        if counts.get(key, 0) >= args.max_memories_per_bucket:
            continue
        item["novelty"] = novelty; item["memory_value"] = .85*item["base_memory_value"] + .15*novelty
        if item["memory_value"] < args.min_memory_value:
            continue
        selected.append(item); counts[key] = counts.get(key, 0)+1
        if len(selected) >= args.max_memory_items:
            break
    return [{k:v for k,v in x.items() if k != "tokens"} for x in selected]


async def run_decision_case(data_idx: int, phase: str, score_row: pd.Series, raw_row: pd.Series,
                            base: float, base_n: int, memories: Sequence[Mapping[str, Any]],
                            args: argparse.Namespace, client: Any, sem: asyncio.Semaphore,
                            run_hash: str) -> Dict[str, Any]:
    profile = build_observed_profile(raw_row, score_row, args.include_sensitive_context)
    prompt = build_decision_prompt(profile, base, base_n, memories)
    started = time.time()
    try:
        obj, raw_text, usage, req = await call_structured(client, sem, model=args.model,
            name="flare_vax_no_prior_vax_decision", schema=DECISION_SCHEMA, system=DECISION_SYSTEM,
            user=prompt, max_tokens=args.decision_max_tokens, temperature=args.temperature, retries=args.max_retries)
        d = validate_decision(obj, base)
        return {"timestamp": utc_now(), "config_hash": run_hash, "status": "ok", "phase": phase,
                "data_idx": data_idx, "actual": int(score_row["vaccinated"]), "pattern": str(score_row["hbm8_pattern"]),
                "pattern_theory_order": int(score_row["hbm8_theory_order"]), "base_probability": base,
                "base_pattern_n": base_n, **d, "memory_count": len(memories),
                "memory_source_indices": [int(m["source_data_idx"]) for m in memories],
                "observed_profile": profile, "raw_response": raw_text, "usage": usage, "request_id": req,
                "elapsed_seconds": time.time()-started}
    except Exception as exc:
        return {"timestamp": utc_now(), "config_hash": run_hash, "status": "error", "phase": phase,
                "data_idx": data_idx, "actual": int(score_row["vaccinated"]), "pattern": str(score_row["hbm8_pattern"]),
                "base_probability": base, "error_type": type(exc).__name__, "error_message": str(exc),
                "elapsed_seconds": time.time()-started}


async def run_reflection_case(entry: Mapping[str, Any], args: argparse.Namespace, client: Any,
                              sem: asyncio.Semaphore, run_hash: str) -> Dict[str, Any]:
    d = str(entry["required_direction"]); t = str(entry["required_error_type"])
    prompt = build_reflection_prompt(entry["observed_profile"], entry, int(entry["actual"]), float(entry["base_probability"]), d, t)
    started = time.time()
    try:
        obj, raw_text, usage, req = await call_structured(client, sem, model=args.reflection_model or args.model,
            name="flare_vax_no_prior_vax_reflection", schema=REFLECTION_SCHEMA, system=REFLECTION_SYSTEM,
            user=prompt, max_tokens=args.reflection_max_tokens, temperature=args.temperature, retries=args.max_retries)
        r = validate_reflection(obj, d, t)
        return {"timestamp": utc_now(), "config_hash": run_hash, "status": "ok", "phase": "reflection",
                "data_idx": int(entry["data_idx"]), "actual": int(entry["actual"]), "pattern": entry["pattern"],
                "probability_yes": float(entry["probability_yes"]), "error_confidence": float(entry["error_confidence"]),
                "reflection": r, "raw_response": raw_text, "usage": usage, "request_id": req,
                "elapsed_seconds": time.time()-started}
    except Exception as exc:
        return {"timestamp": utc_now(), "config_hash": run_hash, "status": "error", "phase": "reflection",
                "data_idx": int(entry["data_idx"]), "actual": int(entry["actual"]), "pattern": entry["pattern"],
                "error_confidence": float(entry["error_confidence"]), "error_type": type(exc).__name__,
                "error_message": str(exc), "elapsed_seconds": time.time()-started}


async def run_decision_batch(indices: Sequence[int], phase: str, scores: pd.DataFrame, raw: pd.DataFrame,
                             args: argparse.Namespace, client: Any, sem: asyncio.Semaphore, log: Path,
                             latest: Dict[int, Dict[str, Any]], base_rates: Mapping[str, Mapping[str, float]],
                             memory_df: pd.DataFrame, memory_store: Optional[MemoryStore], leave_one_out: bool,
                             run_hash: str) -> None:
    todo = [i for i in indices if i not in latest]
    gate = asyncio.Semaphore(max(1, args.concurrent_samples))
    async def one(i: int) -> Dict[str, Any]:
        async with gate:
            row = scores.iloc[i]; base, n = row_pattern_anchor(row, base_rates, memory_df, args.base_rate_prior_strength, leave_one_out)
            memories = [] if memory_store is None else memory_store.retrieve(i)
            return await run_decision_case(i, phase, row, raw.iloc[i], base, n, memories, args, client, sem, run_hash)
    tasks = [asyncio.create_task(one(i)) for i in todo]
    for fut in asyncio.as_completed(tasks):
        e = await fut; latest[int(e["data_idx"])] = e; append_jsonl(log, e)
        if e.get("status") != "ok" and not args.continue_on_error:
            for t in tasks: t.cancel()
            raise RuntimeError(e.get("error_message"))
    print(f"{phase}: {len(indices)-len(todo)} reused, {len(todo)} new, {sum(v.get('status')=='ok' for v in latest.values())} ok")


async def run_reflection_batch(candidates: Sequence[Mapping[str, Any]], args: argparse.Namespace,
                               client: Any, sem: asyncio.Semaphore, log: Path,
                               latest: Dict[int, Dict[str, Any]], run_hash: str) -> None:
    todo = [c for c in candidates if int(c["data_idx"]) not in latest]
    gate = asyncio.Semaphore(max(1, args.concurrent_samples))
    async def one(c: Mapping[str, Any]) -> Dict[str, Any]:
        async with gate:
            return await run_reflection_case(c, args, client, sem, run_hash)
    tasks = [asyncio.create_task(one(c)) for c in todo]
    for fut in asyncio.as_completed(tasks):
        e = await fut; latest[int(e["data_idx"])] = e; append_jsonl(log, e)
        if e.get("status") != "ok" and not args.continue_on_error:
            for t in tasks: t.cancel()
            raise RuntimeError(e.get("error_message"))


def binary_metrics(y: np.ndarray, p_percent: np.ndarray, threshold: float) -> Dict[str, Any]:
    p = np.clip(p_percent/100, 1e-7, 1-1e-7); pred = (p_percent >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    out = {"n": len(y), "threshold": threshold, "accuracy": accuracy_score(y,pred),
           "balanced_accuracy": balanced_accuracy_score(y,pred), "precision": precision_score(y,pred,zero_division=0),
           "recall": recall_score(y,pred,zero_division=0), "specificity": tn/max(1,tn+fp),
           "f1": f1_score(y,pred,zero_division=0), "brier": brier_score_loss(y,p), "log_loss": log_loss(y,p,labels=[0,1]),
           "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    out["roc_auc"] = roc_auc_score(y,p) if len(np.unique(y)) > 1 else None
    out["average_precision"] = average_precision_score(y,p) if len(np.unique(y)) > 1 else None
    return out


def calibrate_threshold(entries: Sequence[Mapping[str, Any]], key: str, metric: str) -> Tuple[float, pd.DataFrame]:
    ok = [e for e in entries if e.get("status") == "ok"]
    y = np.array([int(e["actual"]) for e in ok]); p = np.array([float(e[key]) for e in ok])
    rows = []
    for t in np.arange(5, 96, 1):
        m = binary_metrics(y,p,float(t)); rows.append(m)
    tab = pd.DataFrame(rows); best = tab.sort_values([metric,"log_loss"], ascending=[False,True]).iloc[0]
    return float(best["threshold"]), tab


def entries_df(entries: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for e in entries:
        if e.get("status") != "ok": continue
        rows.append({k:v for k,v in e.items() if k not in {"observed_profile","raw_response"}})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FLARE-inspired HBM8 reflective-memory model without other-vaccine-history predictors")
    p.add_argument("--input-csv", "--input_csv", dest="input_csv", required=True)
    p.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    p.add_argument("--api-key", default="")
    p.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    p.add_argument("--reflection-model", default="")
    p.add_argument("--sample-size", type=int, default=1000, help="0 uses all valid target rows")
    p.add_argument("--class-sampling", choices=["proportional","balanced","custom"], default="proportional")
    p.add_argument("--positive-fraction", type=float, default=.5)
    p.add_argument("--preserve-pattern-within-class", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--memory-ratio", type=float, default=.40)
    p.add_argument("--calibration-ratio", type=float, default=.20)
    p.add_argument("--test-ratio", type=float, default=.40)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--pattern-threshold-mode", choices=["median","fixed"], default="median")
    p.add_argument("--fixed-motivation-threshold", type=float, default=2.5)
    p.add_argument("--fixed-capability-threshold", type=float, default=2.5)
    p.add_argument("--fixed-activation-threshold", type=float, default=2.5)
    p.add_argument("--base-rate-prior-strength", type=float, default=10)
    p.add_argument("--construct-config-json", default="", help="Optional JSON overriding proxy and meta-dimension weights")
    p.add_argument("--reflection-low-cutoff", type=float, default=30)
    p.add_argument("--reflection-high-cutoff", type=float, default=70)
    p.add_argument("--max-reflection-candidates", type=int, default=64)
    p.add_argument("--max-reflection-calls-per-bucket", type=int, default=6)
    p.add_argument("--min-reflection-confidence", type=float, default=.65)
    p.add_argument("--min-estimated-memory-value", type=float, default=.60)
    p.add_argument("--min-memory-novelty", type=float, default=.25)
    p.add_argument("--min-memory-value", type=float, default=.55)
    p.add_argument("--max-memories-per-bucket", type=int, default=2)
    p.add_argument("--max-memory-items", type=int, default=32)
    p.add_argument("--memory-top-k", type=int, default=3)
    p.add_argument("--memory-min-similarity", type=float, default=.45)
    p.add_argument("--memory-same-pattern-only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--memory-direction-diversity", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-sensitive-context", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--decision-max-tokens", type=int, default=650)
    p.add_argument("--reflection-max-tokens", type=int, default=750)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--max-concurrent-requests", type=int, default=20)
    p.add_argument("--concurrent-samples", type=int, default=12)
    p.add_argument("--threshold-metric", choices=["balanced_accuracy","f1","accuracy"], default="balanced_accuracy")
    p.add_argument("--run-test-without-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--print-samples", type=int, default=2)
    return p.parse_args()


def resolve_key(cli: str) -> str:
    key = cli or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("Provide --api-key or set OPENAI_API_KEY")
    return key


async def async_main() -> None:
    args = parse_args(); output = Path(args.output_dir)
    if args.overwrite and output.exists(): shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True); (output/"logs").mkdir(exist_ok=True)

    raw_header = pd.read_csv(args.input_csv, nrows=0).columns
    missing = [c for c in REQUIRED_COLUMNS if c not in raw_header]
    if missing: raise KeyError(f"Missing required columns: {missing}")
    raw = pd.read_csv(args.input_csv, usecols=REQUIRED_COLUMNS, low_memory=False)
    construct_config = load_construct_config(args.construct_config_json)
    scores = apply_construct_config(build_scores(raw), construct_config)
    valid = scores["vaccinated"].isin([0,1]) & scores[["motivation_score","capability_score","activation_score"]].notna().all(axis=1)
    raw = raw.loc[valid].copy(); scores = scores.loc[valid].copy()
    scores["preliminary_pattern"] = preliminary_patterns(scores)
    sampling_frame = scores[["vaccinated","preliminary_pattern"]].copy()
    sampling_frame["source_index"] = sampling_frame.index
    sampled = sample_by_class_pattern(sampling_frame, args)
    selected_index = sampled["source_index"].tolist()
    raw = raw.loc[selected_index].reset_index(drop=True); scores = scores.loc[selected_index].reset_index(drop=True)
    scores["preliminary_pattern"] = preliminary_patterns(scores)
    splits = make_splits(scores, args)
    thresholds = fit_thresholds(scores.iloc[splits["memory"]], args)
    scores = assign_patterns(scores, thresholds)
    memory_df = scores.iloc[splits["memory"]].copy()
    base_rates = fit_pattern_base_rates(memory_df, args.base_rate_prior_strength)
    similarity = fit_similarity_space(scores, splits["memory"])

    run_config = {k:v for k,v in vars(args).items() if k not in {"api_key","overwrite"}}
    run_config.update({"version": VERSION, "thresholds": thresholds, "selected_source_indices_sha256": hashlib.sha256(json.dumps(selected_index).encode()).hexdigest(),
                       "excluded_vaccine_columns": sorted(EXCLUDED_VACCINE_COLUMNS), "construct_config": construct_config})
    run_hash = config_hash(run_config)
    (output/"run_config.json").write_text(json.dumps(run_config,ensure_ascii=False,indent=2),encoding="utf-8")
    scores.assign(source_row_index=selected_index).to_csv(output/"selected_profiles_no_prior_vax.csv", index=False)
    pd.DataFrame([{"data_idx":i,"source_row_index":selected_index[i],"split":name} for name,idxs in splits.items() for i in idxs]).to_csv(output/"split_assignments.csv",index=False)
    pd.DataFrame([{"pattern":p,**v} for p,v in base_rates.items()]).to_csv(output/"pattern_base_rates.csv",index=False)

    if args.dry_run:
        for i in range(min(args.print_samples, len(scores))):
            base,n = row_pattern_anchor(scores.iloc[i],base_rates,memory_df,args.base_rate_prior_strength,False)
            profile = build_observed_profile(raw.iloc[i],scores.iloc[i],args.include_sensitive_context)
            print("\n--- DECISION PROMPT ---\n", build_decision_prompt(profile,base,n,[]))
            fake = {"construct_interpretation":{"threat":"moderate","preventive_engagement":"moderate","barriers":"moderate","self_efficacy":"moderate","cues":"moderate"},
                    "reasoning_trace":{"stage_1_evidence_synthesis":"example","stage_2_pattern_interpretation":"example","stage_3_residual_context":"example","stage_4_decision_mapping":"example"},
                    "residual_adjustment":-10,"probability_yes":base-10,"dominant_observed_factors":["example1","example2"]}
            d,t = ("increase","underprediction") if int(scores.iloc[i]["vaccinated"])==1 else ("decrease","overprediction")
            print("\n--- REFLECTION PROMPT ---\n", build_reflection_prompt(profile,fake,int(scores.iloc[i]["vaccinated"]),base,d,t))
        print("\nDry run complete. No API calls were made.")
        return

    if AsyncOpenAI is None: raise RuntimeError("pip install -U openai")
    client = AsyncOpenAI(api_key=resolve_key(args.api_key), timeout=args.timeout, max_retries=args.max_retries)
    sem = asyncio.Semaphore(max(1,args.max_concurrent_requests))
    logs = {name: output/"logs"/name for name in ["train_no_memory.jsonl","reflections.jsonl","calibration_with_memory.jsonl","test_with_memory.jsonl","test_without_memory.jsonl"]}
    latest = {k: load_latest_jsonl(v,run_hash) for k,v in logs.items()}

    await run_decision_batch(splits["memory"],"train_no_memory",scores,raw,args,client,sem,logs["train_no_memory.jsonl"],latest["train_no_memory.jsonl"],base_rates,memory_df,None,True,run_hash)
    train_entries = [latest["train_no_memory.jsonl"][i] for i in splits["memory"] if i in latest["train_no_memory.jsonl"]]
    candidates = select_reflection_candidates(train_entries,args)
    pd.DataFrame([{k:v for k,v in c.items() if k not in {"observed_profile","raw_response"}} for c in candidates]).to_csv(output/"reflection_candidates.csv",index=False)
    await run_reflection_batch(candidates,args,client,sem,logs["reflections.jsonl"],latest["reflections.jsonl"],run_hash)
    reflection_entries = [latest["reflections.jsonl"][int(c["data_idx"])] for c in candidates if int(c["data_idx"]) in latest["reflections.jsonl"]]
    memories = distill_memory(reflection_entries,args)
    store = MemoryStore(memories, similarity, args.memory_min_similarity,args.memory_top_k,args.memory_same_pattern_only,args.memory_direction_diversity)
    with (output/"memory_final.jsonl").open("w",encoding="utf-8") as f:
        for m in memories: f.write(json.dumps(m,ensure_ascii=False)+"\n")
    pd.DataFrame(memories).to_csv(output/"memory_final.csv",index=False)

    await run_decision_batch(splits["calibration"],"calibration_with_memory",scores,raw,args,client,sem,logs["calibration_with_memory.jsonl"],latest["calibration_with_memory.jsonl"],base_rates,memory_df,store,False,run_hash)
    cal = [latest["calibration_with_memory.jsonl"][i] for i in splits["calibration"] if i in latest["calibration_with_memory.jsonl"]]
    llm_t,llm_table = calibrate_threshold(cal,"probability_yes",args.threshold_metric)
    base_t,base_table = calibrate_threshold(cal,"base_probability",args.threshold_metric)
    llm_table.to_csv(output/"threshold_search_llm.csv",index=False); base_table.to_csv(output/"threshold_search_pattern.csv",index=False)

    await run_decision_batch(splits["test"],"test_with_memory",scores,raw,args,client,sem,logs["test_with_memory.jsonl"],latest["test_with_memory.jsonl"],base_rates,memory_df,store,False,run_hash)
    if args.run_test_without_memory:
        await run_decision_batch(splits["test"],"test_without_memory",scores,raw,args,client,sem,logs["test_without_memory.jsonl"],latest["test_without_memory.jsonl"],base_rates,memory_df,None,False,run_hash)
    test = [latest["test_with_memory.jsonl"][i] for i in splits["test"] if latest["test_with_memory.jsonl"].get(i,{}).get("status")=="ok"]
    y=np.array([e["actual"] for e in test]); p=np.array([e["probability_yes"] for e in test]); pb=np.array([e["base_probability"] for e in test])
    metrics={"test_hbm8_pattern_only":binary_metrics(y,pb,base_t),"test_llm_with_memory":binary_metrics(y,p,llm_t)}
    if args.run_test_without_memory:
        nm=[latest["test_without_memory.jsonl"][i] for i in splits["test"] if latest["test_without_memory.jsonl"].get(i,{}).get("status")=="ok"]
        metrics["test_llm_without_memory"]=binary_metrics(np.array([e["actual"] for e in nm]),np.array([e["probability_yes"] for e in nm]),llm_t)
        entries_df(nm).to_csv(output/"test_predictions_without_memory.csv",index=False)
    entries_df(train_entries).to_csv(output/"train_predictions_no_memory.csv",index=False)
    entries_df(cal).to_csv(output/"calibration_predictions_with_memory.csv",index=False)
    entries_df(test).to_csv(output/"test_predictions_with_memory.csv",index=False)
    summary={"version":VERSION,"created_at":utc_now(),"config_hash":run_hash,"n_selected":len(scores),
             "split_sizes":{k:len(v) for k,v in splits.items()},"thresholds":thresholds,"final_memory_items":len(memories),
             "selected_llm_threshold":llm_t,"selected_pattern_threshold":base_t,"metrics":metrics,
             "excluded_vaccine_columns":sorted(EXCLUDED_VACCINE_COLUMNS)}
    (output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=json_safe),encoding="utf-8")
    await client.close(); print(json.dumps(summary,ensure_ascii=False,indent=2))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
