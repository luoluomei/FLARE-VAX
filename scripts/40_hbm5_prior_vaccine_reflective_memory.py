#!/usr/bin/env python3
"""
FLARE-VAX HBM5: construct deterministic HBM proxy scores from raw NHIS 2024,
form three meta-dimensions and eight patterns, run an LLM residual decision call,
distill high-value reflective memory from high-confidence training errors, and
evaluate a frozen-memory test pass.

The deterministic HBM layer creates reproducible proxies:

1. observed_threat_proxy
2. vaccine_acceptance_benefit_proxy
3. structural_barrier_proxy
4. healthcare_cue_proxy
5. navigation_self_efficacy_proxy

Meta-dimensions:
    Motivation = mean(Threat, Benefits)
    Capability = mean(5 - Barriers, Self-Efficacy)
    Activation = Cues

Important interpretation:
These are theory-guided observed proxies, not direct psychometric measurements of
perceived HBM beliefs. The 2024 NHIS does not directly measure every HBM construct.

Example:
    python build_hbm5_profiles_from_nhis2024.py \
        --input_csv /content/drive/MyDrive/Vaccination-Decision-Model/Data/adult24.csv \
        --output_dir /content/drive/MyDrive/Vaccination-Decision-Model/Results/hbm5_raw_analysis
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

VERSION = "hbm5_proxy_v1_nhis2024"
TARGET = "SHTFLU12M_A"
WEIGHT = "WTFA_A"
ID_COLUMNS = ["HHX", "SRVY_YR", "PSTRAT", "PPSU"]

CHRONIC_VARS = [
    "HYPEV_A", "CHDEV_A", "ANGEV_A", "MIEV_A", "STREV_A", "ASTILL_A",
    "CANEV_A", "DIBEV_A", "COPDEV_A", "KIDWEAKEV_A", "LIVEREV_A",
]

REQUIRED_COLUMNS = sorted(set(
    ID_COLUMNS
    + [
        TARGET, WEIGHT, "AGEP_A", "SEX_A", "PHSTAT_A", "ANYDIFF_A", "DISAB3_A",
        "HLTHCOND_A", "HISPALLP_A", "RACEALLP_A", "EDUCP_A", "RATCAT_A",
        "REGION", "BMICAT_A", "SMKCIGST_A",
        # Prior vaccine behavior / benefit proxy
        "SHTCVD191_A", "SHTCVD19NM2_A", "SHTPNUEV_A", "SHTSHINGL1_A",
        "SHINGRIX3_A", "SHTHEPA_A",
        # Structural barriers
        "HICOV_A", "NOTCOV_A", "HINOTYR_A", "HINOTMYR_A", "RSNHICOST_A",
        "HISTOPCOST_A", "MEDDL12M_A", "MEDNG12M_A", "RXDL12M_A",
        "RXDG12M_A", "PAYWORRY_A", "PAYBLL12M_A", "PAYNOBLLNW_A",
        "TRANSPOR_A", "COMDIFF_A", "PRDEDUC1_A", "PRDEDUC2_A",
        # Cues to action
        "LASTDR_A", "WELLNESS_A", "WELLVIS_A", "RETAILHC12MTC_A",
        "VIRAPP12M_A", "URGCC12MTC_A", "EMERG12MTC_A", "HOSPONGT_A",
        # Navigation self-efficacy
        "USUALPL_A", "USPLKIND_A", "ACCSSINT_A", "ACCSSHOM_A",
        "HITLOOK_A", "HITCOMM_A", "HITTEST_A",
    ]
    + CHRONIC_VARS
))

BACKGROUND_COLUMNS = [
    "AGEP_A", "SEX_A", "HISPALLP_A", "RACEALLP_A", "EDUCP_A", "RATCAT_A",
    "REGION", "PHSTAT_A", "BMICAT_A", "SMKCIGST_A",
]

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

INVALID_DEFAULT = {7, 8, 9, 97, 98, 99}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Construct HBM5 proxy scores and analyze eight meta-patterns from raw NHIS 2024."
    )
    p.add_argument("--input_csv", required=True, help="Path to raw adult24.csv")
    p.add_argument("--output_dir", required=True, help="Directory for output CSV/JSON/TXT files")
    p.add_argument(
        "--primary_pattern_mode", choices=["median", "fixed"], default="median",
        help="Which pattern definition is treated as primary in the report. Both are always produced.",
    )
    p.add_argument("--fixed_motivation_threshold", type=float, default=2.5)
    p.add_argument("--fixed_capability_threshold", type=float, default=2.5)
    p.add_argument("--fixed_activation_threshold", type=float, default=2.5)
    p.add_argument(
        "--sample_size", type=int, default=0,
        help="Optional target-valid sample size for quick testing. 0 uses every target-valid row.",
    )
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument(
        "--sample_mode", choices=["proportional", "balanced"], default="proportional",
        help="Applied only when --sample_size > 0.",
    )
    p.add_argument(
        "--include_profile_json", action=argparse.BooleanOptionalAction, default=True,
        help="Include an LLM-ready structured JSON profile column in the row-level output.",
    )
    return p.parse_args()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def valid_numeric(
    series: pd.Series,
    valid: Optional[Iterable[float]] = None,
    invalid: Iterable[float] = INVALID_DEFAULT,
) -> pd.Series:
    x = numeric(series)
    x = x.mask(x.isin(list(invalid)))
    if valid is not None:
        x = x.where(x.isin(list(valid)))
    return x


def yes_no(series: pd.Series) -> pd.Series:
    return valid_numeric(series, valid=[1, 2]).map({1: 1.0, 2: 0.0})


def weighted_available_average(
    components: Mapping[str, pd.Series], weights: Mapping[str, float]
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    values = pd.DataFrame(components)
    weight_vec = pd.Series(weights, dtype=float)
    numerator = values.mul(weight_vec, axis=1).sum(axis=1, skipna=True)
    denominator = values.notna().mul(weight_vec, axis=1).sum(axis=1)
    score = numerator / denominator.replace(0, np.nan)
    observed_n = values.notna().sum(axis=1)
    coverage = values.notna().mul(weight_vec, axis=1).sum(axis=1) / float(weight_vec.sum())
    return score, observed_n, coverage


def evidence_strength(observed_n: pd.Series, strong_min: int, moderate_min: int) -> pd.Series:
    return pd.Series(
        np.select(
            [observed_n >= strong_min, observed_n >= moderate_min, observed_n > 0],
            ["strong", "moderate", "weak"],
            default="none",
        ),
        index=observed_n.index,
    )


def label_binary(series: pd.Series, yes_label: str, no_label: str) -> pd.Series:
    return series.map({1.0: yes_label, 0.0: no_label}).fillna("unknown")


def weighted_rate(group: pd.DataFrame, target_col: str, weight_col: str) -> float:
    mask = (
        group[target_col].notna()
        & group[weight_col].notna()
        & np.isfinite(group[weight_col])
        & (group[weight_col] > 0)
    )
    if not mask.any():
        return float("nan")
    return float(np.average(group.loc[mask, target_col], weights=group.loc[mask, weight_col]))


def weighted_mean(group: pd.DataFrame, value_col: str, weight_col: str) -> float:
    mask = (
        group[value_col].notna()
        & group[weight_col].notna()
        & np.isfinite(group[weight_col])
        & (group[weight_col] > 0)
    )
    if not mask.any():
        return float("nan")
    return float(np.average(group.loc[mask, value_col], weights=group.loc[mask, weight_col]))


def maybe_sample(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.sample_size <= 0 or args.sample_size >= len(df):
        return df.copy()

    rng_seed = args.sample_seed
    if args.sample_mode == "balanced":
        each = args.sample_size // 2
        pieces = []
        for y in [0.0, 1.0]:
            g = df[df["vaccinated"] == y]
            n = min(each, len(g))
            pieces.append(g.sample(n=n, random_state=rng_seed + int(y)))
        sampled = pd.concat(pieces, ignore_index=False)
        remaining = args.sample_size - len(sampled)
        if remaining > 0:
            pool = df.drop(index=sampled.index)
            sampled = pd.concat(
                [sampled, pool.sample(n=min(remaining, len(pool)), random_state=rng_seed + 17)]
            )
    else:
        # Stratified proportional allocation by target class.
        counts = df["vaccinated"].value_counts(normalize=True)
        allocations = {k: int(round(v * args.sample_size)) for k, v in counts.items()}
        delta = args.sample_size - sum(allocations.values())
        if delta != 0:
            largest = counts.idxmax()
            allocations[largest] += delta
        pieces = []
        for y, n in allocations.items():
            g = df[df["vaccinated"] == y]
            pieces.append(g.sample(n=min(n, len(g)), random_state=rng_seed + int(y)))
        sampled = pd.concat(pieces, ignore_index=False)

    return sampled.sample(frac=1, random_state=rng_seed).copy()


def build_scores(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)

    # ------------------------------------------------------------------
    # Outcome and basic context
    # ------------------------------------------------------------------
    out["vaccinated"] = yes_no(raw[TARGET])
    out["survey_weight"] = numeric(raw[WEIGHT])
    for c in ID_COLUMNS:
        if c in raw:
            out[c] = raw[c]
    for c in BACKGROUND_COLUMNS:
        out[c] = raw[c]

    age = valid_numeric(raw["AGEP_A"], invalid=[97, 98, 99])

    # ------------------------------------------------------------------
    # 1) Observed Threat Proxy (0-5)
    # ------------------------------------------------------------------
    age_component = pd.Series(np.nan, index=raw.index, dtype=float)
    age_component[age < 50] = 0.0
    age_component[(age >= 50) & (age < 65)] = 0.45
    age_component[(age >= 65) & (age < 75)] = 0.75
    age_component[age >= 75] = 1.0

    health_component = valid_numeric(raw["PHSTAT_A"], valid=[1, 2, 3, 4, 5]).map(
        {1: 0.0, 2: 0.15, 3: 0.35, 4: 0.75, 5: 1.0}
    )

    chronic_matrix = pd.DataFrame({v: yes_no(raw[v]) for v in CHRONIC_VARS})
    chronic_count = chronic_matrix.sum(axis=1, min_count=1)
    chronic_observed = chronic_matrix.notna().sum(axis=1)
    chronic_component = pd.Series(np.nan, index=raw.index, dtype=float)
    chronic_component[chronic_count == 0] = 0.0
    chronic_component[chronic_count == 1] = 0.35
    chronic_component[chronic_count == 2] = 0.60
    chronic_component[chronic_count == 3] = 0.80
    chronic_component[chronic_count >= 4] = 1.0

    function_matrix = pd.DataFrame(
        {"functioning_difficulty": yes_no(raw["ANYDIFF_A"]), "disability": yes_no(raw["DISAB3_A"])}
    )
    function_component = function_matrix.max(axis=1, skipna=True)
    function_component[function_matrix.notna().sum(axis=1) == 0] = np.nan
    immune_component = yes_no(raw["HLTHCOND_A"])

    threat_raw, threat_n, threat_coverage = weighted_available_average(
        {
            "age_risk": age_component,
            "self_rated_health": health_component,
            "chronic_burden": chronic_component,
            "functional_vulnerability": function_component,
            "immune_vulnerability": immune_component,
        },
        {
            "age_risk": 0.25,
            "self_rated_health": 0.20,
            "chronic_burden": 0.35,
            "functional_vulnerability": 0.10,
            "immune_vulnerability": 0.10,
        },
    )
    out["observed_threat_proxy"] = (5 * threat_raw).clip(0, 5)
    out["threat_evidence_strength"] = evidence_strength(threat_n, strong_min=4, moderate_min=3)
    out["threat_evidence_coverage"] = threat_coverage
    out["threat_age_component"] = age_component
    out["threat_health_component"] = health_component
    out["threat_chronic_component"] = chronic_component
    out["threat_function_component"] = function_component
    out["threat_immune_component"] = immune_component
    out["chronic_or_risk_count"] = chronic_count
    out["chronic_variables_observed"] = chronic_observed

    # ------------------------------------------------------------------
    # 2) Vaccine-Acceptance / Benefit Proxy (0-5)
    # This is NOT direct perceived benefit. It uses prior vaccine behavior
    # as an observed proxy for vaccine acceptance and preventive belief.
    # ------------------------------------------------------------------
    covid_yes = yes_no(raw["SHTCVD191_A"])
    covid_doses = valid_numeric(raw["SHTCVD19NM2_A"], valid=[1, 2, 3])
    covid_component = pd.Series(np.nan, index=raw.index, dtype=float)
    covid_component[covid_yes == 0] = 0.0
    covid_component[(covid_yes == 1) & (covid_doses == 1)] = 0.70
    covid_component[(covid_yes == 1) & (covid_doses == 2)] = 0.85
    covid_component[(covid_yes == 1) & (covid_doses == 3)] = 1.00
    covid_component[(covid_yes == 1) & covid_doses.isna()] = 0.70

    pneumonia_yes = yes_no(raw["SHTPNUEV_A"])
    immune_yes = immune_component == 1
    pneumonia_eligible_proxy = (age >= 65) | (chronic_count >= 1) | immune_yes
    pneumonia_component = pneumonia_yes.where(pneumonia_eligible_proxy)

    shingles_yes = yes_no(raw["SHTSHINGL1_A"])
    shingrix_yes = yes_no(raw["SHINGRIX3_A"])
    shingles_component = shingles_yes.where(age >= 50)
    shingles_component[(age >= 50) & (shingrix_yes == 1)] = 1.0

    benefit_raw, benefit_n, benefit_coverage = weighted_available_average(
        {
            "covid_vaccine_acceptance": covid_component,
            "pneumonia_vaccine_acceptance": pneumonia_component,
            "shingles_vaccine_acceptance": shingles_component,
        },
        {
            "covid_vaccine_acceptance": 0.50,
            "pneumonia_vaccine_acceptance": 0.25,
            "shingles_vaccine_acceptance": 0.25,
        },
    )
    hepatitis_a_yes = yes_no(raw["SHTHEPA_A"])
    # Hepatitis A is not universally indicated. A positive response is weak
    # additional evidence; a negative response is not treated as a penalty.
    hepatitis_bonus = (hepatitis_a_yes == 1).astype(float) * 0.25
    out["vaccine_acceptance_benefit_proxy"] = (5 * benefit_raw + hepatitis_bonus).clip(0, 5)
    out["benefit_evidence_strength"] = evidence_strength(benefit_n, strong_min=3, moderate_min=2)
    out["benefit_evidence_coverage"] = benefit_coverage
    out["benefit_covid_component"] = covid_component
    out["benefit_pneumonia_component"] = pneumonia_component
    out["benefit_shingles_component"] = shingles_component
    out["benefit_hepatitis_a_positive"] = hepatitis_a_yes
    out["benefit_pneumonia_eligibility_proxy"] = pneumonia_eligible_proxy.astype(float)

    # ------------------------------------------------------------------
    # 3) Structural Barrier Proxy (0-5; higher means more barriers)
    # ------------------------------------------------------------------
    current_uninsured_matrix = pd.DataFrame(
        {
            "hicov_no": 1 - yes_no(raw["HICOV_A"]),
            "notcov_yes": yes_no(raw["NOTCOV_A"]),
        }
    )
    current_uninsured = current_uninsured_matrix.max(axis=1, skipna=True)
    current_uninsured[current_uninsured_matrix.notna().sum(axis=1) == 0] = np.nan

    past_uninsured = yes_no(raw["HINOTYR_A"])
    months_uninsured = valid_numeric(raw["HINOTMYR_A"], invalid=[97, 98, 99]).clip(0, 12) / 12.0
    insurance_matrix = pd.DataFrame(
        {
            "current_uninsured": current_uninsured,
            "past_year_uninsured": past_uninsured * 0.70,
            "months_uninsured": months_uninsured,
            "coverage_not_affordable": yes_no(raw["RSNHICOST_A"]),
            "coverage_stopped_cost_increase": yes_no(raw["HISTOPCOST_A"]) * 0.50,
        }
    )
    insurance_component = insurance_matrix.max(axis=1, skipna=True)
    insurance_component[insurance_matrix.notna().sum(axis=1) == 0] = np.nan

    cost_care_matrix = pd.DataFrame(
        {
            "needed_care_not_received": yes_no(raw["MEDNG12M_A"]),
            "delayed_medical_care": yes_no(raw["MEDDL12M_A"]) * 0.80,
            "needed_rx_not_received": yes_no(raw["RXDG12M_A"]) * 0.80,
            "delayed_rx": yes_no(raw["RXDL12M_A"]) * 0.60,
        }
    )
    cost_care_component = cost_care_matrix.max(axis=1, skipna=True)
    cost_care_component[cost_care_matrix.notna().sum(axis=1) == 0] = np.nan

    pay_worry = valid_numeric(raw["PAYWORRY_A"], valid=[1, 2, 3]).map(
        {1: 0.70, 2: 0.35, 3: 0.0}
    )
    financial_matrix = pd.DataFrame(
        {
            "unable_pay_bills": yes_no(raw["PAYNOBLLNW_A"]),
            "problems_paying_bills": yes_no(raw["PAYBLL12M_A"]) * 0.70,
            "medical_bill_worry": pay_worry,
            "deductible_plan1": yes_no(raw["PRDEDUC1_A"]) * 0.20,
            "deductible_plan2": yes_no(raw["PRDEDUC2_A"]) * 0.20,
        }
    )
    financial_component = financial_matrix.max(axis=1, skipna=True)
    financial_component[financial_matrix.notna().sum(axis=1) == 0] = np.nan

    transportation_component = yes_no(raw["TRANSPOR_A"])
    communication_component = valid_numeric(raw["COMDIFF_A"], valid=[1, 2, 3, 4]).map(
        {1: 0.0, 2: 0.33, 3: 0.67, 4: 1.0}
    )

    barrier_raw, barrier_n, barrier_coverage = weighted_available_average(
        {
            "insurance_instability": insurance_component,
            "cost_related_unmet_care": cost_care_component,
            "medical_financial_stress": financial_component,
            "transportation_constraint": transportation_component,
            "communication_constraint": communication_component,
        },
        {
            "insurance_instability": 0.25,
            "cost_related_unmet_care": 0.35,
            "medical_financial_stress": 0.20,
            "transportation_constraint": 0.10,
            "communication_constraint": 0.10,
        },
    )
    out["structural_barrier_proxy"] = (5 * barrier_raw).clip(0, 5)
    out["barrier_evidence_strength"] = evidence_strength(barrier_n, strong_min=4, moderate_min=3)
    out["barrier_evidence_coverage"] = barrier_coverage
    out["barrier_insurance_component"] = insurance_component
    out["barrier_cost_care_component"] = cost_care_component
    out["barrier_financial_component"] = financial_component
    out["barrier_transport_component"] = transportation_component
    out["barrier_communication_component"] = communication_component

    # ------------------------------------------------------------------
    # 4) Healthcare Cue Proxy (0-5)
    # Measures opportunity for a cue, not direct physician recommendation.
    # ------------------------------------------------------------------
    doctor_recency = valid_numeric(raw["LASTDR_A"], valid=[0, 1, 2, 3, 4, 5, 6]).map(
        {0: 0.0, 1: 1.0, 2: 0.50, 3: 0.25, 4: 0.10, 5: 0.0, 6: 0.0}
    )
    wellness_recency = valid_numeric(raw["WELLVIS_A"], valid=[0, 1, 2, 3, 4, 5, 6]).map(
        {0: 0.0, 1: 1.0, 2: 0.50, 3: 0.25, 4: 0.10, 5: 0.0, 6: 0.0}
    )
    wellness_recency = wellness_recency.fillna(yes_no(raw["WELLNESS_A"]))

    retail_count = valid_numeric(raw["RETAILHC12MTC_A"], valid=[0, 1, 2, 3, 4, 5])
    retail_component = retail_count.map({0: 0.0, 1: 0.50, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0})
    virtual_component = yes_no(raw["VIRAPP12M_A"]) * 0.60
    retail_virtual_matrix = pd.DataFrame(
        {"retail_clinic": retail_component, "virtual_visit": virtual_component}
    )
    retail_virtual_component = retail_virtual_matrix.max(axis=1, skipna=True)
    retail_virtual_component[retail_virtual_matrix.notna().sum(axis=1) == 0] = np.nan

    urgent_count = valid_numeric(raw["URGCC12MTC_A"], valid=[0, 1, 2, 3, 4, 5])
    emergency_count = valid_numeric(raw["EMERG12MTC_A"], valid=[0, 1, 2, 3, 4])
    acute_matrix = pd.DataFrame(
        {
            "urgent_care": (urgent_count > 0).where(urgent_count.notna()).astype(float),
            "emergency_room": (emergency_count > 0).where(emergency_count.notna()).astype(float),
            "overnight_hospital": yes_no(raw["HOSPONGT_A"]),
        }
    )
    acute_component = acute_matrix.max(axis=1, skipna=True) * 0.50
    acute_component[acute_matrix.notna().sum(axis=1) == 0] = np.nan

    cue_raw, cue_n, cue_coverage = weighted_available_average(
        {
            "recent_doctor_contact": doctor_recency,
            "preventive_wellness_contact": wellness_recency,
            "retail_or_virtual_contact": retail_virtual_component,
            "acute_care_contact": acute_component,
        },
        {
            "recent_doctor_contact": 0.35,
            "preventive_wellness_contact": 0.35,
            "retail_or_virtual_contact": 0.15,
            "acute_care_contact": 0.15,
        },
    )
    out["healthcare_cue_proxy"] = (5 * cue_raw).clip(0, 5)
    out["cue_evidence_strength"] = evidence_strength(cue_n, strong_min=4, moderate_min=3)
    out["cue_evidence_coverage"] = cue_coverage
    out["cue_doctor_component"] = doctor_recency
    out["cue_wellness_component"] = wellness_recency
    out["cue_retail_virtual_component"] = retail_virtual_component
    out["cue_acute_component"] = acute_component

    # ------------------------------------------------------------------
    # 5) Navigation Self-Efficacy Proxy (0-5)
    # Measures observed healthcare-navigation capacity, not direct confidence.
    # ------------------------------------------------------------------
    usual_care_component = valid_numeric(raw["USUALPL_A"], valid=[1, 2, 3]).map(
        {1: 1.0, 2: 0.0, 3: 0.60}
    )
    care_setting_component = valid_numeric(raw["USPLKIND_A"], valid=[1, 2, 3, 4, 5, 6]).map(
        {1: 1.0, 2: 0.70, 3: 0.20, 4: 1.0, 5: 0.50, 6: 0.30}
    )

    internet_matrix = pd.DataFrame(
        {"internet_anywhere": yes_no(raw["ACCSSINT_A"]), "internet_home": yes_no(raw["ACCSSHOM_A"])}
    )
    internet_component = internet_matrix.mean(axis=1, skipna=True)
    internet_component[internet_matrix.notna().sum(axis=1) == 0] = np.nan

    digital_matrix = pd.DataFrame(
        {
            "looked_up_health_information": yes_no(raw["HITLOOK_A"]),
            "communicated_with_doctor": yes_no(raw["HITCOMM_A"]),
            "viewed_test_results": yes_no(raw["HITTEST_A"]),
        }
    )
    no_internet = yes_no(raw["ACCSSINT_A"]) == 0
    digital_matrix = digital_matrix.mask(no_internet.to_numpy()[:, None] & digital_matrix.isna(), 0.0)
    digital_component = digital_matrix.mean(axis=1, skipna=True)
    digital_component[digital_matrix.notna().sum(axis=1) == 0] = np.nan

    virtual_navigation_component = yes_no(raw["VIRAPP12M_A"])
    communication_capacity = valid_numeric(raw["COMDIFF_A"], valid=[1, 2, 3, 4]).map(
        {1: 1.0, 2: 0.67, 3: 0.33, 4: 0.0}
    )

    efficacy_raw, efficacy_n, efficacy_coverage = weighted_available_average(
        {
            "usual_care_access": usual_care_component,
            "stable_care_setting": care_setting_component,
            "internet_access": internet_component,
            "digital_health_navigation": digital_component,
            "virtual_care_experience": virtual_navigation_component,
            "communication_capacity": communication_capacity,
        },
        {
            "usual_care_access": 0.30,
            "stable_care_setting": 0.10,
            "internet_access": 0.15,
            "digital_health_navigation": 0.25,
            "virtual_care_experience": 0.10,
            "communication_capacity": 0.10,
        },
    )
    out["navigation_self_efficacy_proxy"] = (5 * efficacy_raw).clip(0, 5)
    out["self_efficacy_evidence_strength"] = evidence_strength(
        efficacy_n, strong_min=5, moderate_min=3
    )
    out["self_efficacy_evidence_coverage"] = efficacy_coverage
    out["selfeff_usual_care_component"] = usual_care_component
    out["selfeff_care_setting_component"] = care_setting_component
    out["selfeff_internet_component"] = internet_component
    out["selfeff_digital_component"] = digital_component
    out["selfeff_virtual_component"] = virtual_navigation_component
    out["selfeff_communication_component"] = communication_capacity

    # ------------------------------------------------------------------
    # Three meta-dimensions
    # ------------------------------------------------------------------
    out["motivation_score"] = out[
        ["observed_threat_proxy", "vaccine_acceptance_benefit_proxy"]
    ].mean(axis=1, skipna=True)
    out["capability_score"] = pd.concat(
        [5 - out["structural_barrier_proxy"], out["navigation_self_efficacy_proxy"]],
        axis=1,
    ).mean(axis=1, skipna=True)
    out["activation_score"] = out["healthcare_cue_proxy"]

    # Short evidence strings for later LLM-ready profiles.
    out["threat_evidence"] = (
        "age=" + age.round(0).astype("Int64").astype(str)
        + "; health_component=" + health_component.round(2).astype(str)
        + "; chronic_count=" + chronic_count.round(0).astype("Int64").astype(str)
        + "; functional=" + function_component.round(2).astype(str)
        + "; immune=" + immune_component.round(2).astype(str)
    )
    out["benefit_evidence"] = (
        "covid=" + covid_component.round(2).astype(str)
        + "; pneumonia=" + pneumonia_component.round(2).astype(str)
        + "; shingles=" + shingles_component.round(2).astype(str)
        + "; hepA_positive=" + hepatitis_a_yes.round(0).astype("Int64").astype(str)
    )
    out["barrier_evidence"] = (
        "insurance=" + insurance_component.round(2).astype(str)
        + "; cost_unmet=" + cost_care_component.round(2).astype(str)
        + "; financial=" + financial_component.round(2).astype(str)
        + "; transport=" + transportation_component.round(2).astype(str)
        + "; communication=" + communication_component.round(2).astype(str)
    )
    out["cue_evidence"] = (
        "doctor=" + doctor_recency.round(2).astype(str)
        + "; wellness=" + wellness_recency.round(2).astype(str)
        + "; retail_virtual=" + retail_virtual_component.round(2).astype(str)
        + "; acute=" + acute_component.round(2).astype(str)
    )
    out["self_efficacy_evidence"] = (
        "usual_care=" + usual_care_component.round(2).astype(str)
        + "; setting=" + care_setting_component.round(2).astype(str)
        + "; internet=" + internet_component.round(2).astype(str)
        + "; digital=" + digital_component.round(2).astype(str)
        + "; virtual=" + virtual_navigation_component.round(2).astype(str)
        + "; communication=" + communication_capacity.round(2).astype(str)
    )

    return out


def assign_patterns(
    df: pd.DataFrame,
    motivation_threshold: float,
    capability_threshold: float,
    activation_threshold: float,
    suffix: str,
) -> pd.DataFrame:
    result = df.copy()
    m = np.where(
        result["motivation_score"] >= motivation_threshold,
        "High Motivation",
        "Low Motivation",
    )
    c = np.where(
        result["capability_score"] >= capability_threshold,
        "High Capability",
        "Low Capability",
    )
    a = np.where(
        result["activation_score"] >= activation_threshold,
        "Strong Cue",
        "Weak Cue",
    )
    result[f"motivation_level_{suffix}"] = m
    result[f"capability_level_{suffix}"] = c
    result[f"activation_level_{suffix}"] = a
    result[f"hbm8_pattern_{suffix}"] = pd.Series(m, index=result.index) + " / " + c + " / " + a
    result[f"hbm8_theory_order_{suffix}"] = result[f"hbm8_pattern_{suffix}"].map(PATTERN_THEORY_ORDER)
    return result


def pattern_summary(df: pd.DataFrame, pattern_col: str, weight_col: str) -> pd.DataFrame:
    rows: List[dict] = []
    for pattern, g in df.groupby(pattern_col, dropna=False):
        y = g["vaccinated"]
        rows.append(
            {
                "pattern": pattern,
                "theory_order": PATTERN_THEORY_ORDER.get(str(pattern), np.nan),
                "n": int(len(g)),
                "actual_yes": int((y == 1).sum()),
                "actual_no": int((y == 0).sum()),
                "unweighted_vaccination_rate": float(y.mean()),
                "weighted_vaccination_rate": weighted_rate(g, "vaccinated", weight_col),
                "mean_threat": float(g["observed_threat_proxy"].mean()),
                "mean_benefits": float(g["vaccine_acceptance_benefit_proxy"].mean()),
                "mean_barriers": float(g["structural_barrier_proxy"].mean()),
                "mean_cues": float(g["healthcare_cue_proxy"].mean()),
                "mean_self_efficacy": float(g["navigation_self_efficacy_proxy"].mean()),
                "mean_motivation": float(g["motivation_score"].mean()),
                "mean_capability": float(g["capability_score"].mean()),
                "mean_activation": float(g["activation_score"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["empirical_rank_unweighted"] = out["unweighted_vaccination_rate"].rank(
            ascending=False, method="min"
        ).astype("Int64")
        out["empirical_rank_weighted"] = out["weighted_vaccination_rate"].rank(
            ascending=False, method="min"
        ).astype("Int64")
        out = out.sort_values(["theory_order", "pattern"], na_position="last")
    return out


def score_trend_summary(df: pd.DataFrame, score_columns: Mapping[str, str]) -> pd.DataFrame:
    labels = ["Very low (0-1)", "Low (1-2)", "Moderate (2-3)", "High (3-4)", "Very high (4-5)"]
    bins = [-np.inf, 1, 2, 3, 4, np.inf]
    rows: List[dict] = []
    for construct, col in score_columns.items():
        work = df[[col, "vaccinated", "survey_weight"]].copy()
        work["score_band"] = pd.cut(work[col], bins=bins, labels=labels, right=False)
        for band, g in work.groupby("score_band", observed=True):
            rows.append(
                {
                    "construct": construct,
                    "score_column": col,
                    "score_band": str(band),
                    "n": int(len(g)),
                    "mean_score": float(g[col].mean()),
                    "unweighted_vaccination_rate": float(g["vaccinated"].mean()),
                    "weighted_vaccination_rate": weighted_rate(g, "vaccinated", "survey_weight"),
                }
            )
    return pd.DataFrame(rows)


def construct_summary(df: pd.DataFrame, score_columns: Mapping[str, str]) -> pd.DataFrame:
    rows = []
    for construct, col in score_columns.items():
        s = df[col]
        rows.append(
            {
                "construct": construct,
                "score_column": col,
                "n_nonmissing": int(s.notna().sum()),
                "missing_rate": float(s.isna().mean()),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "p25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "p75": float(s.quantile(0.75)),
                "max": float(s.max()),
                "pearson_with_vaccination": float(df[[col, "vaccinated"]].corr().iloc[0, 1]),
                "spearman_with_vaccination": float(df[[col, "vaccinated"]].corr(method="spearman").iloc[0, 1]),
                "weighted_mean": weighted_mean(df, col, "survey_weight"),
            }
        )
    return pd.DataFrame(rows)


def build_profile_json(row: pd.Series) -> str:
    payload = {
        "hbm5_proxies": {
            "observed_threat": {
                "score": round(float(row["observed_threat_proxy"]), 3) if pd.notna(row["observed_threat_proxy"]) else None,
                "evidence_strength": row["threat_evidence_strength"],
                "evidence": row["threat_evidence"],
            },
            "vaccine_acceptance_benefits": {
                "score": round(float(row["vaccine_acceptance_benefit_proxy"]), 3) if pd.notna(row["vaccine_acceptance_benefit_proxy"]) else None,
                "evidence_strength": row["benefit_evidence_strength"],
                "evidence": row["benefit_evidence"],
            },
            "structural_barriers": {
                "score": round(float(row["structural_barrier_proxy"]), 3) if pd.notna(row["structural_barrier_proxy"]) else None,
                "evidence_strength": row["barrier_evidence_strength"],
                "evidence": row["barrier_evidence"],
            },
            "healthcare_cues": {
                "score": round(float(row["healthcare_cue_proxy"]), 3) if pd.notna(row["healthcare_cue_proxy"]) else None,
                "evidence_strength": row["cue_evidence_strength"],
                "evidence": row["cue_evidence"],
            },
            "navigation_self_efficacy": {
                "score": round(float(row["navigation_self_efficacy_proxy"]), 3) if pd.notna(row["navigation_self_efficacy_proxy"]) else None,
                "evidence_strength": row["self_efficacy_evidence_strength"],
                "evidence": row["self_efficacy_evidence"],
            },
        },
        "meta_dimensions": {
            "motivation": round(float(row["motivation_score"]), 3) if pd.notna(row["motivation_score"]) else None,
            "capability": round(float(row["capability_score"]), 3) if pd.notna(row["capability_score"]) else None,
            "activation": round(float(row["activation_score"]), 3) if pd.notna(row["activation_score"]) else None,
        },
        "empirical_pattern_median": row.get("hbm8_pattern_median"),
        "background": {
            "age": int(row["AGEP_A"]) if pd.notna(row.get("AGEP_A")) else None,
            "sex_code": int(row["SEX_A"]) if pd.notna(row.get("SEX_A")) else None,
            "education_code": int(row["EDUCP_A"]) if pd.notna(row.get("EDUCP_A")) else None,
            "income_to_poverty_code": int(row["RATCAT_A"]) if pd.notna(row.get("RATCAT_A")) else None,
            "region_code": int(row["REGION"]) if pd.notna(row.get("REGION")) else None,
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def safe_json_number(x: float) -> Optional[float]:
    return None if pd.isna(x) or not np.isfinite(x) else float(x)


# =============================================================================
# HBM5 + LLM reflective-memory prediction pipeline
# =============================================================================

import asyncio
import hashlib
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Set

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
except ImportError:  # pragma: no cover - dry-run can work without the SDK
    AsyncOpenAI = None  # type: ignore


PIPELINE_VERSION = "hbm5_openai_v1_offline_reflective_memory"

PATTERN_TENDENCY = {
    "High Motivation / High Capability / Strong Cue": "very high general vaccination tendency",
    "High Motivation / High Capability / Weak Cue": "high general vaccination tendency",
    "High Motivation / Low Capability / Strong Cue": "moderately high but constrained tendency",
    "High Motivation / Low Capability / Weak Cue": "mixed-to-moderate tendency",
    "Low Motivation / High Capability / Strong Cue": "moderate cue-activated tendency",
    "Low Motivation / High Capability / Weak Cue": "generally low tendency",
    "Low Motivation / Low Capability / Strong Cue": "low tendency despite a cue opportunity",
    "Low Motivation / Low Capability / Weak Cue": "very low general vaccination tendency",
}

HEALTH_STATUS_LABELS = {
    1: "excellent", 2: "very good", 3: "good", 4: "fair", 5: "poor"
}
SEX_LABELS = {1: "male", 2: "female"}
REGION_LABELS = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}
BMI_LABELS = {1: "underweight", 2: "healthy weight", 3: "overweight", 4: "obese"}
SMOKING_LABELS = {
    1: "every day", 2: "some days", 3: "not at all", 4: "unknown/former category"
}
EDUCATION_LABELS = {
    0: "never attended/kindergarten only",
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
POVERTY_RATIO_LABELS = {
    1: "0.00-0.49", 2: "0.50-0.74", 3: "0.75-0.99", 4: "1.00-1.24",
    5: "1.25-1.49", 6: "1.50-1.74", 7: "1.75-1.99", 8: "2.00-2.49",
    9: "2.50-2.99", 10: "3.00-3.49", 11: "3.50-3.99", 12: "4.00-4.49",
    13: "4.50-4.99", 14: "5.00 or greater",
}
RACE_LABELS = {
    1: "White only", 2: "Black/African American only", 3: "Asian only",
    4: "American Indian/Alaska Native only", 5: "AIAN and another race",
    6: "other single/multiple race"
}
HISPANIC_LABELS = {
    1: "Hispanic", 2: "non-Hispanic White", 3: "non-Hispanic Black",
    4: "non-Hispanic Asian", 5: "non-Hispanic AIAN",
    6: "non-Hispanic AIAN and another race", 7: "other/multiple race"
}
LAST_VISIT_LABELS = {
    0: "never", 1: "within past year", 2: "1 to under 2 years",
    3: "2 to under 3 years", 4: "3 to under 5 years",
    5: "5 to under 10 years", 6: "10 or more years",
}
USUAL_PLACE_LABELS = {1: "one usual place", 2: "no usual place", 3: "more than one usual place"}
USUAL_KIND_LABELS = {
    1: "doctor office/health center",
    2: "urgent care or retail clinic",
    3: "hospital emergency room",
    4: "VA facility",
    5: "other place",
    6: "no single place used most often",
}
COMM_DIFFICULTY_LABELS = {1: "none", 2: "some", 3: "a lot", 4: "cannot do at all"}

CHRONIC_LABELS = {
    "HYPEV_A": "hypertension",
    "CHDEV_A": "coronary heart disease",
    "ANGEV_A": "angina",
    "MIEV_A": "heart attack",
    "STREV_A": "stroke",
    "ASTILL_A": "current asthma",
    "CANEV_A": "cancer history",
    "DIBEV_A": "diabetes",
    "COPDEV_A": "COPD",
    "KIDWEAKEV_A": "weak kidneys",
    "LIVEREV_A": "liver condition",
}

DECISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "residual_adjustment": {"type": "integer", "minimum": -30, "maximum": 30},
        "probability_yes": {"type": "number", "minimum": 0, "maximum": 100},
        "deviation_direction": {
            "type": "string",
            "enum": ["lower_than_pattern", "similar_to_pattern", "higher_than_pattern"],
        },
        "dominant_hbm_factors": {
            "type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4,
        },
        "residual_observed_factors": {
            "type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 4,
        },
        "reason": {"type": "string"},
    },
    "required": [
        "residual_adjustment", "probability_yes", "deviation_direction",
        "dominant_hbm_factors", "residual_observed_factors", "reason",
    ],
    "additionalProperties": False,
}

REFLECTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "correction_direction": {"type": "string", "enum": ["increase", "decrease"]},
        "error_type": {"type": "string", "enum": ["underprediction", "overprediction"]},
        "residual_factor": {"type": "string"},
        "supporting_variables": {
            "type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 6,
        },
        "correction_rule": {"type": "string"},
        "applicability_conditions": {
            "type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5,
        },
        "non_generalization_warning": {"type": "string"},
        "reflection_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "estimated_memory_value": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "correction_direction", "error_type", "residual_factor", "supporting_variables",
        "correction_rule", "applicability_conditions", "non_generalization_warning",
        "reflection_confidence", "estimated_memory_value",
    ],
    "additionalProperties": False,
}

DECISION_SYSTEM = """You are a public-health behavioral analyst using the Health Belief Model (HBM).
You receive five theory-guided observed proxy scores, three meta-dimensions, an eight-pattern behavioral prior, complete observed evidence, and sometimes a few reflective memories.

Theory:
- Threat and prior vaccine acceptance/benefit contribute to motivation.
- Low barriers and high navigation self-efficacy contribute to capability.
- Healthcare contact opportunities contribute to activation.

Important constraints:
1. These are observed proxies, not direct psychometric measurements of private beliefs.
2. The HBM8 pattern and its training-only base rate are a prior, not a deterministic label.
3. Use the complete observed profile to identify residual factors that may move this respondent above or below the pattern tendency.
4. Retrieved memories are exception rules. Apply one only when its stated applicability conditions match the current profile.
5. Do not use the true vaccination label; it is not provided.
6. Return only the required structured JSON object.
"""

REFLECTION_SYSTEM = """You are distilling a small, high-value reflective memory from a high-confidence prediction error.
The goal is not to restate the HBM pattern. Identify a concrete observed residual factor that explains why this respondent deviated from the prediction and may generalize to genuinely similar respondents.

Rules:
1. Ground every rule in specific observed variables from the supplied profile.
2. Do not invent attitudes, physician recommendations, reminders, or intentions that were not observed.
3. Distinguish a reusable residual rule from random or unobserved heterogeneity.
4. State applicability conditions and a non-generalization warning.
5. The required correction direction is supplied and must be followed exactly.
6. Return only the required structured JSON object.
"""


def pipeline_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any, max_chars: int = 420) -> str:
    text = " ".join(str(value or "").strip().split())
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def code_label(value: Any, mapping: Mapping[int, str]) -> Optional[str]:
    try:
        if pd.isna(value):
            return None
        return mapping.get(int(float(value)), f"code {int(float(value))}")
    except Exception:
        return None


def yn(value: Any) -> Optional[bool]:
    try:
        if pd.isna(value):
            return None
        iv = int(float(value))
        if iv == 1:
            return True
        if iv == 2:
            return False
    except Exception:
        pass
    return None


def valid_num_scalar(value: Any, invalid: Set[int] | None = None) -> Optional[float]:
    invalid = invalid or {7, 8, 9, 97, 98, 99}
    try:
        x = float(value)
        if not np.isfinite(x) or int(x) in invalid:
            return None
        return x
    except Exception:
        return None


def active_chronic_conditions(raw_row: pd.Series) -> List[str]:
    out: List[str] = []
    for var, label in CHRONIC_LABELS.items():
        if yn(raw_row.get(var)) is True:
            out.append(label)
    return out


def build_full_observed_profile(
    raw_row: pd.Series,
    score_row: pd.Series,
    *,
    include_sensitive_context: bool,
) -> Dict[str, Any]:
    age = valid_num_scalar(raw_row.get("AGEP_A"), {97, 98, 99})
    profile: Dict[str, Any] = {
        "hbm5_proxies": {
            "observed_threat": {
                "score": json_safe(score_row.get("observed_threat_proxy")),
                "evidence_strength": json_safe(score_row.get("threat_evidence_strength")),
                "evidence_coverage": json_safe(score_row.get("threat_evidence_coverage")),
                "components": {
                    "age_risk": json_safe(score_row.get("threat_age_component")),
                    "self_rated_health": json_safe(score_row.get("threat_health_component")),
                    "chronic_burden": json_safe(score_row.get("threat_chronic_component")),
                    "functional_vulnerability": json_safe(score_row.get("threat_function_component")),
                    "immune_vulnerability": json_safe(score_row.get("threat_immune_component")),
                },
            },
            "vaccine_acceptance_benefits": {
                "score": json_safe(score_row.get("vaccine_acceptance_benefit_proxy")),
                "evidence_strength": json_safe(score_row.get("benefit_evidence_strength")),
                "evidence_coverage": json_safe(score_row.get("benefit_evidence_coverage")),
                "components": {
                    "covid_vaccine_acceptance": json_safe(score_row.get("benefit_covid_component")),
                    "pneumonia_vaccine_acceptance": json_safe(score_row.get("benefit_pneumonia_component")),
                    "shingles_vaccine_acceptance": json_safe(score_row.get("benefit_shingles_component")),
                    "hepatitis_a_positive": json_safe(score_row.get("benefit_hepatitis_a_positive")),
                },
            },
            "structural_barriers": {
                "score": json_safe(score_row.get("structural_barrier_proxy")),
                "evidence_strength": json_safe(score_row.get("barrier_evidence_strength")),
                "evidence_coverage": json_safe(score_row.get("barrier_evidence_coverage")),
                "components": {
                    "insurance_instability": json_safe(score_row.get("barrier_insurance_component")),
                    "cost_related_unmet_care": json_safe(score_row.get("barrier_cost_care_component")),
                    "medical_financial_stress": json_safe(score_row.get("barrier_financial_component")),
                    "transportation_constraint": json_safe(score_row.get("barrier_transport_component")),
                    "communication_constraint": json_safe(score_row.get("barrier_communication_component")),
                },
            },
            "healthcare_cues": {
                "score": json_safe(score_row.get("healthcare_cue_proxy")),
                "evidence_strength": json_safe(score_row.get("cue_evidence_strength")),
                "evidence_coverage": json_safe(score_row.get("cue_evidence_coverage")),
                "components": {
                    "doctor_contact": json_safe(score_row.get("cue_doctor_component")),
                    "wellness_contact": json_safe(score_row.get("cue_wellness_component")),
                    "retail_or_virtual_contact": json_safe(score_row.get("cue_retail_virtual_component")),
                    "acute_care_contact": json_safe(score_row.get("cue_acute_component")),
                },
            },
            "navigation_self_efficacy": {
                "score": json_safe(score_row.get("navigation_self_efficacy_proxy")),
                "evidence_strength": json_safe(score_row.get("self_efficacy_evidence_strength")),
                "evidence_coverage": json_safe(score_row.get("self_efficacy_evidence_coverage")),
                "components": {
                    "usual_care_access": json_safe(score_row.get("selfeff_usual_care_component")),
                    "stable_care_setting": json_safe(score_row.get("selfeff_care_setting_component")),
                    "internet_access": json_safe(score_row.get("selfeff_internet_component")),
                    "digital_health_navigation": json_safe(score_row.get("selfeff_digital_component")),
                    "virtual_care_experience": json_safe(score_row.get("selfeff_virtual_component")),
                    "communication_capacity": json_safe(score_row.get("selfeff_communication_component")),
                },
            },
        },
        "meta_dimensions": {
            "motivation": json_safe(score_row.get("motivation_score")),
            "capability": json_safe(score_row.get("capability_score")),
            "activation": json_safe(score_row.get("activation_score")),
        },
        "hbm8_pattern": json_safe(score_row.get("hbm8_pattern")),
        "health_risk": {
            "age": None if age is None else int(age),
            "self_rated_health": code_label(raw_row.get("PHSTAT_A"), HEALTH_STATUS_LABELS),
            "chronic_conditions": active_chronic_conditions(raw_row),
            "chronic_condition_count": json_safe(score_row.get("chronic_or_risk_count")),
            "functional_difficulty": yn(raw_row.get("ANYDIFF_A")),
            "disability": yn(raw_row.get("DISAB3_A")),
            "immune_vulnerability": yn(raw_row.get("HLTHCOND_A")),
            "bmi_category": code_label(raw_row.get("BMICAT_A"), BMI_LABELS),
            "smoking_status": code_label(raw_row.get("SMKCIGST_A"), SMOKING_LABELS),
        },
        "prior_vaccine_behavior": {
            "covid_vaccinated": yn(raw_row.get("SHTCVD191_A")),
            "covid_dose_category": valid_num_scalar(raw_row.get("SHTCVD19NM2_A")),
            "pneumonia_vaccinated": yn(raw_row.get("SHTPNUEV_A")),
            "pneumonia_eligibility_proxy": json_safe(score_row.get("benefit_pneumonia_eligibility_proxy")),
            "shingles_vaccinated": yn(raw_row.get("SHTSHINGL1_A")),
            "shingrix_vaccinated": yn(raw_row.get("SHINGRIX3_A")),
            "hepatitis_a_vaccinated": yn(raw_row.get("SHTHEPA_A")),
        },
        "access_and_cost": {
            "currently_insured": yn(raw_row.get("HICOV_A")),
            "explicitly_not_covered": yn(raw_row.get("NOTCOV_A")),
            "uninsured_in_past_year": yn(raw_row.get("HINOTYR_A")),
            "months_uninsured": valid_num_scalar(raw_row.get("HINOTMYR_A"), {97, 98, 99}),
            "coverage_unaffordable": yn(raw_row.get("RSNHICOST_A")),
            "coverage_stopped_due_to_cost": yn(raw_row.get("HISTOPCOST_A")),
            "delayed_medical_care_due_to_cost": yn(raw_row.get("MEDDL12M_A")),
            "needed_medical_care_not_received_due_to_cost": yn(raw_row.get("MEDNG12M_A")),
            "delayed_prescription_due_to_cost": yn(raw_row.get("RXDL12M_A")),
            "needed_prescription_not_received_due_to_cost": yn(raw_row.get("RXDG12M_A")),
            "medical_bill_problem": yn(raw_row.get("PAYBLL12M_A")),
            "unable_to_pay_medical_bills_now": yn(raw_row.get("PAYNOBLLNW_A")),
            "transportation_barrier": yn(raw_row.get("TRANSPOR_A")),
            "communication_difficulty": code_label(raw_row.get("COMDIFF_A"), COMM_DIFFICULTY_LABELS),
            "deductible_plan1": yn(raw_row.get("PRDEDUC1_A")),
            "deductible_plan2": yn(raw_row.get("PRDEDUC2_A")),
        },
        "healthcare_contact": {
            "last_doctor_visit": code_label(raw_row.get("LASTDR_A"), LAST_VISIT_LABELS),
            "wellness_visit_recency": code_label(raw_row.get("WELLVIS_A"), LAST_VISIT_LABELS),
            "wellness_visit_indicator": yn(raw_row.get("WELLNESS_A")),
            "retail_clinic_visit_category": valid_num_scalar(raw_row.get("RETAILHC12MTC_A")),
            "virtual_appointment": yn(raw_row.get("VIRAPP12M_A")),
            "urgent_care_visit_category": valid_num_scalar(raw_row.get("URGCC12MTC_A")),
            "emergency_visit_category": valid_num_scalar(raw_row.get("EMERG12MTC_A")),
            "overnight_hospitalization": yn(raw_row.get("HOSPONGT_A")),
        },
        "navigation_capacity": {
            "usual_care_place": code_label(raw_row.get("USUALPL_A"), USUAL_PLACE_LABELS),
            "usual_care_setting": code_label(raw_row.get("USPLKIND_A"), USUAL_KIND_LABELS),
            "internet_access": yn(raw_row.get("ACCSSINT_A")),
            "internet_at_home": yn(raw_row.get("ACCSSHOM_A")),
            "looked_up_health_information": yn(raw_row.get("HITLOOK_A")),
            "communicated_with_doctor_online": yn(raw_row.get("HITCOMM_A")),
            "viewed_test_results_online": yn(raw_row.get("HITTEST_A")),
            "virtual_care_experience": yn(raw_row.get("VIRAPP12M_A")),
        },
        "background": {
            "sex": code_label(raw_row.get("SEX_A"), SEX_LABELS),
            "education": code_label(raw_row.get("EDUCP_A"), EDUCATION_LABELS),
            "income_to_poverty_ratio": code_label(raw_row.get("RATCAT_A"), POVERTY_RATIO_LABELS),
            "region": code_label(raw_row.get("REGION"), REGION_LABELS),
        },
    }
    if include_sensitive_context:
        profile["background"].update(
            {
                "race": code_label(raw_row.get("RACEALLP_A"), RACE_LABELS),
                "hispanic_group": code_label(raw_row.get("HISPALLP_A"), HISPANIC_LABELS),
            }
        )
    return profile


def config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def append_jsonl(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(obj), ensure_ascii=False, default=json_safe) + "\n")


def load_latest_jsonl(path: Path, key: str = "data_idx") -> Dict[int, Dict[str, Any]]:
    latest: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if key in obj:
                    latest[int(obj[key])] = obj
            except Exception:
                continue
    return latest


def usage_from_response(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    inp = int(getattr(usage, "input_tokens", 0) or 0)
    out = int(getattr(usage, "output_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", inp + out) or 0)
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": total}


def sum_usage(entries: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for e in entries:
        u = e.get("usage", {}) if isinstance(e, Mapping) else {}
        for k in total:
            total[k] += int(u.get(k, 0) or 0)
    return total


async def call_structured_json(
    client: Any,
    semaphore: asyncio.Semaphore,
    *,
    model: str,
    schema_name: str,
    schema: Dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    temperature: float,
    retries: int,
) -> Tuple[Dict[str, Any], str, Dict[str, int], str]:
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
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
            raw_text = str(getattr(response, "output_text", "") or "").strip()
            if not raw_text:
                raise RuntimeError("empty structured output")
            parsed = json.loads(raw_text)
            return (
                parsed,
                raw_text,
                usage_from_response(response),
                str(getattr(response, "_request_id", "") or ""),
            )
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            await asyncio.sleep(min(20.0, 1.5 * (2 ** attempt)))
    raise RuntimeError(f"OpenAI structured call failed after retries: {last_error}") from last_error


def validate_decision(obj: Mapping[str, Any], base_probability: float) -> Dict[str, Any]:
    adjustment = int(obj["residual_adjustment"])
    adjustment = max(-30, min(30, adjustment))
    computed = float(np.clip(base_probability + adjustment, 0.0, 100.0))
    raw_probability = float(obj["probability_yes"])
    direction = str(obj["deviation_direction"])
    expected_direction = (
        "higher_than_pattern" if adjustment > 2 else
        "lower_than_pattern" if adjustment < -2 else
        "similar_to_pattern"
    )
    return {
        "residual_adjustment": adjustment,
        "raw_probability_yes": raw_probability,
        "probability_yes": computed,
        "probability_consistency_corrected": abs(raw_probability - computed) > 1.0,
        "deviation_direction": expected_direction,
        "raw_deviation_direction": direction,
        "direction_consistency_corrected": direction != expected_direction,
        "dominant_hbm_factors": [clean_text(x, 120) for x in obj["dominant_hbm_factors"]],
        "residual_observed_factors": [clean_text(x, 140) for x in obj["residual_observed_factors"]],
        "reason": clean_text(obj["reason"], 420),
    }


def expected_reflection_direction(actual: int, probability_yes: float) -> Tuple[str, str]:
    if actual == 1 and probability_yes < 50:
        return "increase", "underprediction"
    if actual == 0 and probability_yes >= 50:
        return "decrease", "overprediction"
    raise ValueError("Reflection requested for a non-error")


def validate_reflection(
    obj: Mapping[str, Any],
    *,
    expected_direction: str,
    expected_error_type: str,
) -> Dict[str, Any]:
    if str(obj["correction_direction"]) != expected_direction:
        raise ValueError(
            f"Reflection direction {obj['correction_direction']} conflicts with required {expected_direction}"
        )
    if str(obj["error_type"]) != expected_error_type:
        raise ValueError(
            f"Reflection error type {obj['error_type']} conflicts with required {expected_error_type}"
        )
    supporting = [clean_text(x, 180) for x in obj["supporting_variables"] if str(x).strip()]
    conditions = [clean_text(x, 180) for x in obj["applicability_conditions"] if str(x).strip()]
    if len(supporting) < 2:
        raise ValueError("Reflection needs at least two concrete supporting variables")
    return {
        "correction_direction": expected_direction,
        "error_type": expected_error_type,
        "residual_factor": clean_text(obj["residual_factor"], 220),
        "supporting_variables": supporting[:6],
        "correction_rule": clean_text(obj["correction_rule"], 420),
        "applicability_conditions": conditions[:5],
        "non_generalization_warning": clean_text(obj["non_generalization_warning"], 320),
        "reflection_confidence": float(np.clip(float(obj["reflection_confidence"]), 0, 1)),
        "estimated_memory_value": float(np.clip(float(obj["estimated_memory_value"]), 0, 1)),
    }


def build_decision_prompt(
    *,
    observed_profile: Mapping[str, Any],
    pattern_base_probability: float,
    pattern_n: int,
    retrieved_memories: Sequence[Mapping[str, Any]],
) -> str:
    pattern = str(observed_profile.get("hbm8_pattern"))
    memory_payload = []
    for m in retrieved_memories:
        memory_payload.append(
            {
                "similarity": round(float(m.get("similarity", 0.0)), 3),
                "source_pattern": m.get("pattern"),
                "correction_direction": m.get("correction_direction"),
                "residual_factor": m.get("residual_factor"),
                "supporting_variables": m.get("supporting_variables"),
                "correction_rule": m.get("correction_rule"),
                "applicability_conditions": m.get("applicability_conditions"),
                "non_generalization_warning": m.get("non_generalization_warning"),
            }
        )
    payload = {
        "training_only_pattern_prior": {
            "pattern": pattern,
            "general_tendency": PATTERN_TENDENCY.get(pattern, "mixed general tendency"),
            "base_probability_yes": round(float(pattern_base_probability), 2),
            "memory_build_pattern_n": int(pattern_n),
        },
        "respondent_observed_profile": observed_profile,
        "retrieved_reflective_memories": memory_payload,
    }
    return (
        "Predict whether this respondent received an influenza vaccination during the past 12 months.\n"
        "Start from the training-only HBM8 pattern base probability. Apply a residual adjustment between -30 and +30 percentage points.\n"
        "Use memories only when their applicability conditions match; do not treat them as labels.\n"
        "The returned probability_yes must equal base_probability_yes + residual_adjustment, clipped to 0-100.\n\n"
        "INPUT JSON\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def build_reflection_prompt(
    *,
    observed_profile: Mapping[str, Any],
    decision: Mapping[str, Any],
    actual: int,
    pattern_base_probability: float,
    expected_direction: str,
    expected_error_type: str,
) -> str:
    payload = {
        "required_correction_direction": expected_direction,
        "required_error_type": expected_error_type,
        "actual_outcome": "YES" if actual == 1 else "NO",
        "training_only_pattern_base_probability": round(float(pattern_base_probability), 2),
        "initial_decision_without_memory": {
            "residual_adjustment": decision["residual_adjustment"],
            "probability_yes": round(float(decision["probability_yes"]), 2),
            "dominant_hbm_factors": decision["dominant_hbm_factors"],
            "residual_observed_factors": decision["residual_observed_factors"],
            "reason": decision["reason"],
        },
        "respondent_observed_profile": observed_profile,
    }
    return (
        "This is a high-confidence error from the no-memory training pass.\n"
        "Identify one concrete, reusable residual factor supported by observed variables that could correct predictions for genuinely similar respondents.\n"
        "Do not merely restate the HBM pattern and do not invent unobserved attitudes or recommendations.\n\n"
        "INPUT JSON\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def allocate_counts(total: int, proportions: Mapping[Any, float]) -> Dict[Any, int]:
    raw = {k: total * float(v) for k, v in proportions.items()}
    floor = {k: int(math.floor(v)) for k, v in raw.items()}
    remainder = total - sum(floor.values())
    order = sorted(raw, key=lambda k: raw[k] - floor[k], reverse=True)
    for k in order[:remainder]:
        floor[k] += 1
    return floor


def sample_by_class_pattern(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.sample_size <= 0 or args.sample_size >= len(df):
        return df.sample(frac=1, random_state=args.random_seed).copy()

    class_rates = df["vaccinated"].value_counts(normalize=True).to_dict()
    if args.class_sampling == "balanced":
        class_props = {0.0: 0.5, 1.0: 0.5}
    elif args.class_sampling == "custom":
        p = float(np.clip(args.positive_fraction, 0.01, 0.99))
        class_props = {0.0: 1.0 - p, 1.0: p}
    else:
        class_props = {0.0: float(class_rates.get(0.0, 0.0)), 1.0: float(class_rates.get(1.0, 0.0))}

    class_counts = allocate_counts(args.sample_size, class_props)
    pieces: List[pd.DataFrame] = []
    for y in [0.0, 1.0]:
        group = df[df["vaccinated"] == y]
        n_class = min(int(class_counts.get(y, 0)), len(group))
        if n_class <= 0:
            continue
        if args.preserve_pattern_within_class:
            pattern_props = group["preliminary_pattern"].value_counts(normalize=True).to_dict()
            pattern_counts = allocate_counts(n_class, pattern_props)
            chosen_parts: List[pd.DataFrame] = []
            for j, (pattern, n) in enumerate(pattern_counts.items()):
                pg = group[group["preliminary_pattern"] == pattern]
                if n > 0 and len(pg) > 0:
                    chosen_parts.append(pg.sample(n=min(n, len(pg)), random_state=args.random_seed + int(y) * 100 + j))
            chosen = pd.concat(chosen_parts, axis=0) if chosen_parts else group.iloc[0:0]
            if len(chosen) < n_class:
                remainder = group.drop(index=chosen.index)
                chosen = pd.concat(
                    [chosen, remainder.sample(n=min(n_class - len(chosen), len(remainder)), random_state=args.random_seed + 700 + int(y))]
                )
            pieces.append(chosen.iloc[:n_class])
        else:
            pieces.append(group.sample(n=n_class, random_state=args.random_seed + int(y)))

    sampled = pd.concat(pieces, axis=0)
    if len(sampled) < args.sample_size:
        remaining = df.drop(index=sampled.index)
        sampled = pd.concat(
            [sampled, remaining.sample(n=min(args.sample_size - len(sampled), len(remaining)), random_state=args.random_seed + 999)]
        )
    return sampled.sample(frac=1, random_state=args.random_seed).copy()


def safe_split_indices(
    indices: np.ndarray,
    strata: pd.Series,
    left_size: int,
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if left_size <= 0:
        return np.array([], dtype=int), indices.copy()
    if left_size >= len(indices):
        return indices.copy(), np.array([], dtype=int)
    strat = strata.loc[indices]
    use_strat = strat.value_counts().min() >= 2 and left_size >= strat.nunique() and (len(indices) - left_size) >= strat.nunique()
    try:
        left, right = train_test_split(
            indices,
            train_size=left_size,
            random_state=random_seed,
            stratify=strat if use_strat else None,
        )
    except ValueError:
        left, right = train_test_split(indices, train_size=left_size, random_state=random_seed, stratify=None)
    return np.sort(left), np.sort(right)


def split_three_way(df: pd.DataFrame, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    ratios = np.array([args.memory_ratio, args.calibration_ratio, args.test_ratio], dtype=float)
    if np.any(ratios < 0) or ratios.sum() <= 0:
        raise ValueError("Split ratios must be nonnegative and sum to a positive value")
    ratios = ratios / ratios.sum()
    n = len(df)
    n_memory = max(1, int(round(n * ratios[0])))
    n_cal = int(round(n * ratios[1]))
    if n >= 3 and ratios[1] > 0:
        n_cal = max(1, n_cal)
    n_test = n - n_memory - n_cal
    if n_test < 1:
        n_test = 1
        if n_cal > 1:
            n_cal -= 1
        else:
            n_memory -= 1
    all_idx = np.arange(n, dtype=int)
    strata = df["vaccinated"].astype(int).astype(str) + "|" + df["preliminary_pattern"].astype(str)
    memory_idx, remainder = safe_split_indices(all_idx, strata, n_memory, args.random_seed)
    if n_cal > 0:
        cal_idx, test_idx = safe_split_indices(remainder, strata, n_cal, args.random_seed + 1)
    else:
        cal_idx, test_idx = np.array([], dtype=int), remainder
    return {"memory": memory_idx, "calibration": cal_idx, "test": test_idx}


def fit_pattern_thresholds(memory_df: pd.DataFrame, args: argparse.Namespace) -> Dict[str, float]:
    if args.pattern_threshold_mode == "fixed":
        return {
            "motivation": args.fixed_motivation_threshold,
            "capability": args.fixed_capability_threshold,
            "activation": args.fixed_activation_threshold,
        }
    return {
        "motivation": float(memory_df["motivation_score"].median()),
        "capability": float(memory_df["capability_score"].median()),
        "activation": float(memory_df["activation_score"].median()),
    }


def assign_final_patterns(df: pd.DataFrame, thresholds: Mapping[str, float]) -> pd.DataFrame:
    out = assign_patterns(
        df,
        thresholds["motivation"],
        thresholds["capability"],
        thresholds["activation"],
        suffix="pipeline",
    )
    out["hbm8_pattern"] = out["hbm8_pattern_pipeline"]
    out["hbm8_theory_order"] = out["hbm8_theory_order_pipeline"]
    return out


def fit_pattern_base_rates(memory_df: pd.DataFrame, prior_strength: float) -> Dict[str, Dict[str, float]]:
    overall = float(memory_df["vaccinated"].mean())
    result: Dict[str, Dict[str, float]] = {}
    for pattern in PATTERN_THEORY_ORDER:
        g = memory_df[memory_df["hbm8_pattern"] == pattern]
        n = int(len(g))
        yes = float(g["vaccinated"].sum())
        raw = yes / n if n else overall
        smoothed = (yes + prior_strength * overall) / (n + prior_strength) if (n + prior_strength) > 0 else overall
        result[pattern] = {
            "n": n, "yes": yes, "raw_rate": raw, "smoothed_rate": smoothed,
        }
    result["__overall__"] = {"n": int(len(memory_df)), "yes": float(memory_df["vaccinated"].sum()), "raw_rate": overall, "smoothed_rate": overall}
    return result


def row_pattern_anchor(
    row: pd.Series,
    base_rates: Mapping[str, Mapping[str, float]],
    *,
    leave_one_out: bool,
    memory_df: pd.DataFrame,
    prior_strength: float,
) -> Tuple[float, int]:
    pattern = str(row["hbm8_pattern"])
    info = base_rates.get(pattern, base_rates["__overall__"])
    if not leave_one_out:
        return 100.0 * float(info["smoothed_rate"]), int(info["n"])
    n = int(info["n"])
    yes = float(info["yes"])
    actual = int(row["vaccinated"])
    overall_n = len(memory_df)
    overall_yes = float(memory_df["vaccinated"].sum())
    if n <= 1 or overall_n <= 1:
        return 100.0 * float(base_rates["__overall__"]["raw_rate"]), max(0, n - 1)
    pattern_yes_loo = yes - actual
    pattern_n_loo = n - 1
    overall_rate_loo = (overall_yes - actual) / (overall_n - 1)
    smoothed = (pattern_yes_loo + prior_strength * overall_rate_loo) / (pattern_n_loo + prior_strength)
    return 100.0 * float(smoothed), pattern_n_loo


HBM_SIM_COLUMNS = [
    "observed_threat_proxy", "vaccine_acceptance_benefit_proxy", "structural_barrier_proxy",
    "healthcare_cue_proxy", "navigation_self_efficacy_proxy",
    "motivation_score", "capability_score", "activation_score",
]
RAW_SIM_NUMERIC = [
    "chronic_or_risk_count",
    "threat_age_component", "threat_health_component", "threat_chronic_component",
    "threat_function_component", "threat_immune_component",
    "benefit_covid_component", "benefit_pneumonia_component", "benefit_shingles_component",
    "barrier_insurance_component", "barrier_cost_care_component", "barrier_financial_component",
    "barrier_transport_component", "barrier_communication_component",
    "cue_doctor_component", "cue_wellness_component", "cue_retail_virtual_component", "cue_acute_component",
    "selfeff_usual_care_component", "selfeff_care_setting_component", "selfeff_internet_component",
    "selfeff_digital_component", "selfeff_virtual_component", "selfeff_communication_component",
]
RAW_SIM_CATEGORICAL = ["SEX_A", "EDUCP_A", "RATCAT_A", "REGION", "BMICAT_A", "SMKCIGST_A"]


@dataclass
class SimilaritySpace:
    hbm_matrix: np.ndarray
    raw_matrix: np.ndarray
    patterns: List[str]

    def similarity(self, query_idx: int, source_idx: int) -> float:
        def cos(a: np.ndarray, b: np.ndarray) -> float:
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom <= 0:
                return 0.0
            return float(np.dot(a, b) / denom)
        hbm = (cos(self.hbm_matrix[query_idx], self.hbm_matrix[source_idx]) + 1.0) / 2.0
        raw = (cos(self.raw_matrix[query_idx], self.raw_matrix[source_idx]) + 1.0) / 2.0
        pattern_bonus = 1.0 if self.patterns[query_idx] == self.patterns[source_idx] else 0.0
        return float(np.clip(0.50 * hbm + 0.40 * raw + 0.10 * pattern_bonus, 0.0, 1.0))


def make_dense(x: Any) -> np.ndarray:
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def fit_similarity_space(df: pd.DataFrame, memory_idx: Sequence[int]) -> SimilaritySpace:
    hbm_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    raw_transform = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), RAW_SIM_NUMERIC),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), RAW_SIM_CATEGORICAL),
        ],
        remainder="drop",
    )
    train = df.iloc[list(memory_idx)]
    hbm_pipe.fit(train[HBM_SIM_COLUMNS])
    raw_transform.fit(train[RAW_SIM_NUMERIC + RAW_SIM_CATEGORICAL])
    hbm_matrix = make_dense(hbm_pipe.transform(df[HBM_SIM_COLUMNS])).astype(float)
    raw_matrix = make_dense(raw_transform.transform(df[RAW_SIM_NUMERIC + RAW_SIM_CATEGORICAL])).astype(float)
    return SimilaritySpace(hbm_matrix=hbm_matrix, raw_matrix=raw_matrix, patterns=df["hbm8_pattern"].astype(str).tolist())


@dataclass
class MemoryStore:
    items: List[Dict[str, Any]] = field(default_factory=list)
    similarity_space: Optional[SimilaritySpace] = None
    min_similarity: float = 0.45
    top_k: int = 2

    def retrieve(self, query_idx: int) -> List[Dict[str, Any]]:
        if not self.items or self.similarity_space is None:
            return []
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for item in self.items:
            sim = self.similarity_space.similarity(query_idx, int(item["source_data_idx"]))
            if sim >= self.min_similarity:
                enriched = dict(item)
                enriched["similarity"] = sim
                scored.append((sim, enriched))
        scored.sort(key=lambda t: (t[0], float(t[1].get("memory_value", 0.0))), reverse=True)
        return [x[1] for x in scored[: self.top_k]]


def tokenize_rule(item: Mapping[str, Any]) -> Set[str]:
    text = " ".join(
        [
            str(item.get("residual_factor", "")),
            str(item.get("correction_rule", "")),
            " ".join(item.get("supporting_variables", []) or []),
            " ".join(item.get("applicability_conditions", []) or []),
        ]
    ).lower()
    return set(re.findall(r"[a-z0-9_]+", text))


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def error_confidence(actual: int, probability_yes: float) -> float:
    if actual == 1:
        return float(np.clip((50.0 - probability_yes) / 50.0, 0.0, 1.0))
    return float(np.clip((probability_yes - 50.0) / 50.0, 0.0, 1.0))


def select_reflection_candidates(
    train_entries: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for e in train_entries:
        if e.get("status") != "ok":
            continue
        actual = int(e["actual"])
        p = float(e["probability_yes"])
        is_candidate = (actual == 1 and p <= args.reflection_low_cutoff) or (actual == 0 and p >= args.reflection_high_cutoff)
        if not is_candidate:
            continue
        direction, error_type = expected_reflection_direction(actual, p)
        c = dict(e)
        c["required_direction"] = direction
        c["required_error_type"] = error_type
        c["error_confidence"] = error_confidence(actual, p)
        candidates.append(c)

    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for c in candidates:
        key = (str(c["pattern"]), str(c["required_direction"]))
        buckets.setdefault(key, []).append(c)
    selected: List[Dict[str, Any]] = []
    for key, bucket in buckets.items():
        bucket.sort(key=lambda x: float(x["error_confidence"]), reverse=True)
        selected.extend(bucket[: args.max_reflection_calls_per_bucket])
    selected.sort(key=lambda x: float(x["error_confidence"]), reverse=True)
    return selected[: args.max_reflection_candidates]


def distill_memory(
    reflection_entries: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    prelim: List[Dict[str, Any]] = []
    for e in reflection_entries:
        if e.get("status") != "ok":
            continue
        r = e["reflection"]
        specificity = min(1.0, len(r.get("supporting_variables", [])) / 4.0)
        base_value = (
            0.35 * float(e.get("error_confidence", 0.0))
            + 0.25 * specificity
            + 0.20 * float(r.get("reflection_confidence", 0.0))
            + 0.20 * float(r.get("estimated_memory_value", 0.0))
        )
        if float(r.get("reflection_confidence", 0.0)) < args.min_reflection_confidence:
            continue
        if float(r.get("estimated_memory_value", 0.0)) < args.min_estimated_memory_value:
            continue
        item = {
            "source_data_idx": int(e["data_idx"]),
            "pattern": e["pattern"],
            "actual": int(e["actual"]),
            "initial_probability_yes": float(e["probability_yes"]),
            "error_confidence": float(e.get("error_confidence", 0.0)),
            **r,
            "specificity": specificity,
            "base_memory_value": base_value,
        }
        item["tokens"] = tokenize_rule(item)
        prelim.append(item)

    prelim.sort(key=lambda x: float(x["base_memory_value"]), reverse=True)
    selected: List[Dict[str, Any]] = []
    bucket_counts: Dict[Tuple[str, str], int] = {}
    for item in prelim:
        same_direction = [x for x in selected if x["correction_direction"] == item["correction_direction"]]
        max_sim = max((jaccard(item["tokens"], x["tokens"]) for x in same_direction), default=0.0)
        novelty = 1.0 - max_sim
        if novelty < args.min_memory_novelty:
            continue
        key = (str(item["pattern"]), str(item["correction_direction"]))
        if bucket_counts.get(key, 0) >= args.max_memories_per_bucket:
            continue
        memory_value = 0.85 * float(item["base_memory_value"]) + 0.15 * novelty
        if memory_value < args.min_memory_value:
            continue
        item["novelty"] = novelty
        item["memory_value"] = memory_value
        # Keep the internal token set while memory candidates are still being
        # compared. Removing it here makes the next candidate fail when it
        # tries to access x["tokens"] from an already selected item.
        selected.append(item)
        bucket_counts[key] = bucket_counts.get(key, 0) + 1
        if len(selected) >= args.max_memory_items:
            break

    # Token sets are only an internal de-duplication aid. Strip them after the
    # full selection pass so the returned memories remain JSON serializable.
    return [
        {k: v for k, v in item.items() if k != "tokens"}
        for item in selected
    ]


async def run_decision_case(
    *,
    data_idx: int,
    phase: str,
    score_row: pd.Series,
    raw_row: pd.Series,
    base_probability: float,
    base_n: int,
    memories: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    client: Any,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    profile = build_full_observed_profile(raw_row, score_row, include_sensitive_context=args.include_sensitive_context)
    prompt = build_decision_prompt(
        observed_profile=profile,
        pattern_base_probability=base_probability,
        pattern_n=base_n,
        retrieved_memories=memories,
    )
    started = time.time()
    try:
        obj, raw_text, usage, request_id = await call_structured_json(
            client,
            semaphore,
            model=args.model,
            schema_name="hbm5_residual_vaccination_decision",
            schema=DECISION_SCHEMA,
            system_prompt=DECISION_SYSTEM,
            user_prompt=prompt,
            max_output_tokens=args.decision_max_tokens,
            temperature=args.temperature,
            retries=args.max_retries,
        )
        decision = validate_decision(obj, base_probability)
        return {
            "timestamp": pipeline_utc_now(),
            "status": "ok",
            "phase": phase,
            "data_idx": data_idx,
            "actual": int(score_row["vaccinated"]),
            "actual_label": "YES" if int(score_row["vaccinated"]) == 1 else "NO",
            "pattern": str(score_row["hbm8_pattern"]),
            "pattern_theory_order": int(score_row["hbm8_theory_order"]),
            "base_probability": float(base_probability),
            "base_pattern_n": int(base_n),
            **decision,
            "memory_count": len(memories),
            "memory_source_indices": [int(m["source_data_idx"]) for m in memories],
            "memory_similarities": [float(m.get("similarity", 0.0)) for m in memories],
            "observed_profile": profile,
            "raw_response": raw_text,
            "usage": usage,
            "request_id": request_id,
            "elapsed_seconds": time.time() - started,
        }
    except Exception as exc:
        return {
            "timestamp": pipeline_utc_now(), "status": "error", "phase": phase,
            "data_idx": data_idx, "actual": int(score_row["vaccinated"]),
            "pattern": str(score_row["hbm8_pattern"]),
            "base_probability": float(base_probability), "base_pattern_n": int(base_n),
            "error_type": type(exc).__name__, "error_message": str(exc),
            "elapsed_seconds": time.time() - started,
        }


async def run_reflection_case(
    *,
    train_entry: Mapping[str, Any],
    args: argparse.Namespace,
    client: Any,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    expected_direction = str(train_entry["required_direction"])
    expected_error_type = str(train_entry["required_error_type"])
    prompt = build_reflection_prompt(
        observed_profile=train_entry["observed_profile"],
        decision=train_entry,
        actual=int(train_entry["actual"]),
        pattern_base_probability=float(train_entry["base_probability"]),
        expected_direction=expected_direction,
        expected_error_type=expected_error_type,
    )
    started = time.time()
    try:
        obj, raw_text, usage, request_id = await call_structured_json(
            client,
            semaphore,
            model=args.reflection_model or args.model,
            schema_name="hbm5_high_value_reflection",
            schema=REFLECTION_SCHEMA,
            system_prompt=REFLECTION_SYSTEM,
            user_prompt=prompt,
            max_output_tokens=args.reflection_max_tokens,
            temperature=args.temperature,
            retries=args.max_retries,
        )
        reflection = validate_reflection(
            obj,
            expected_direction=expected_direction,
            expected_error_type=expected_error_type,
        )
        return {
            "timestamp": pipeline_utc_now(), "status": "ok",
            "data_idx": int(train_entry["data_idx"]),
            "pattern": train_entry["pattern"],
            "actual": int(train_entry["actual"]),
            "probability_yes": float(train_entry["probability_yes"]),
            "error_confidence": float(train_entry["error_confidence"]),
            "reflection": reflection,
            "raw_response": raw_text, "usage": usage, "request_id": request_id,
            "elapsed_seconds": time.time() - started,
        }
    except Exception as exc:
        return {
            "timestamp": pipeline_utc_now(), "status": "error",
            "data_idx": int(train_entry["data_idx"]), "pattern": train_entry["pattern"],
            "actual": int(train_entry["actual"]), "probability_yes": float(train_entry["probability_yes"]),
            "error_confidence": float(train_entry["error_confidence"]),
            "error_type": type(exc).__name__, "error_message": str(exc),
            "elapsed_seconds": time.time() - started,
        }


async def run_decision_batch(
    *,
    indices: Sequence[int],
    phase: str,
    scores: pd.DataFrame,
    raw_selected: pd.DataFrame,
    args: argparse.Namespace,
    client: Any,
    semaphore: asyncio.Semaphore,
    log_path: Path,
    latest: Dict[int, Dict[str, Any]],
    base_rates: Mapping[str, Mapping[str, float]],
    memory_df: pd.DataFrame,
    memory_store: Optional[MemoryStore],
    leave_one_out: bool,
) -> None:
    missing = [int(i) for i in indices if latest.get(int(i), {}).get("status") != "ok"]
    if not missing:
        print(f"{phase}: already complete")
        return
    pbar = tqdm(total=len(missing), desc=phase)
    batch_size = max(1, args.concurrent_samples)
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        tasks = []
        for idx in batch:
            row = scores.iloc[idx]
            base, base_n = row_pattern_anchor(
                row, base_rates, leave_one_out=leave_one_out,
                memory_df=memory_df, prior_strength=args.base_rate_prior_strength,
            )
            memories = memory_store.retrieve(idx) if memory_store is not None else []
            tasks.append(
                run_decision_case(
                    data_idx=idx, phase=phase, score_row=row, raw_row=raw_selected.iloc[idx],
                    base_probability=base, base_n=base_n, memories=memories,
                    args=args, client=client, semaphore=semaphore,
                )
            )
        results = await asyncio.gather(*tasks)
        for entry in sorted(results, key=lambda x: int(x["data_idx"])):
            append_jsonl(log_path, entry)
            latest[int(entry["data_idx"])] = entry
            pbar.update(1)
            if entry.get("status") != "ok" and not args.continue_on_error:
                pbar.close()
                raise RuntimeError(f"{phase} failed at {entry['data_idx']}: {entry.get('error_message')}")
    pbar.close()


async def run_reflection_batch(
    *,
    candidates: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    client: Any,
    semaphore: asyncio.Semaphore,
    log_path: Path,
    latest: Dict[int, Dict[str, Any]],
) -> None:
    missing = [c for c in candidates if latest.get(int(c["data_idx"]), {}).get("status") != "ok"]
    if not missing:
        print("REFLECTION: already complete or no candidates")
        return
    pbar = tqdm(total=len(missing), desc="TRAIN CALL2 reflection")
    batch_size = max(1, args.concurrent_samples)
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        results = await asyncio.gather(
            *[run_reflection_case(train_entry=c, args=args, client=client, semaphore=semaphore) for c in batch]
        )
        for entry in sorted(results, key=lambda x: int(x["data_idx"])):
            append_jsonl(log_path, entry)
            latest[int(entry["data_idx"])] = entry
            pbar.update(1)
            if entry.get("status") != "ok" and not args.continue_on_error:
                pbar.close()
                raise RuntimeError(f"Reflection failed at {entry['data_idx']}: {entry.get('error_message')}")
    pbar.close()


def binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> Dict[str, Any]:
    valid = np.isfinite(y_true) & np.isfinite(probability)
    y = y_true[valid].astype(int)
    p = probability[valid].astype(float)
    if len(y) == 0:
        return {"n_valid": 0, "threshold": threshold}
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    result: Dict[str, Any] = {
        "n_valid": int(len(y)),
        "actual_yes_rate": float(y.mean()),
        "mean_probability_yes": float(p.mean() / 100.0),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "predicted_yes_rate": float(pred.mean()),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }
    prob01 = np.clip(p / 100.0, 1e-6, 1 - 1e-6)
    result["brier_score"] = float(brier_score_loss(y, prob01))
    result["log_loss"] = float(log_loss(y, prob01, labels=[0, 1]))
    result["roc_auc"] = float(roc_auc_score(y, prob01)) if len(np.unique(y)) == 2 else float("nan")
    result["average_precision"] = float(average_precision_score(y, prob01)) if len(np.unique(y)) == 2 else float("nan")
    return result


def calibrate_threshold(
    entries: Sequence[Mapping[str, Any]],
    probability_field: str,
    metric: str,
) -> Tuple[float, pd.DataFrame, Dict[str, Any]]:
    rows = [e for e in entries if e.get("status") == "ok"]
    if not rows:
        return 50.0, pd.DataFrame(), {"fallback": True, "selected_threshold": 50.0, "reason": "no calibration rows"}
    y = np.array([float(e["actual"]) for e in rows])
    p = np.array([float(e[probability_field]) for e in rows])
    if len(rows) < 10 or len(np.unique(y)) < 2:
        return 50.0, pd.DataFrame(), {"fallback": True, "selected_threshold": 50.0, "reason": "calibration split too small or one class"}
    table_rows = []
    for t in range(1, 100):
        m = binary_metrics(y, p, t)
        table_rows.append({"threshold": t, **m})
    table = pd.DataFrame(table_rows)
    metric_col = {"balanced_accuracy": "balanced_accuracy", "f1": "f1", "accuracy": "accuracy"}[metric]
    best_value = table[metric_col].max()
    candidates = table[np.isclose(table[metric_col], best_value)].copy()
    candidates["distance_from_50"] = (candidates["threshold"] - 50).abs()
    selected = candidates.sort_values(["distance_from_50", "threshold"]).iloc[0]
    threshold = float(selected["threshold"])
    return threshold, table, {
        "fallback": False,
        "probability_field": probability_field,
        "threshold_metric": metric,
        "selected_threshold": threshold,
        "selected_metric_value": float(selected[metric_col]),
        "metrics_at_50": binary_metrics(y, p, 50.0),
        "metrics_selected": binary_metrics(y, p, threshold),
    }


def entries_dataframe(entries: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for e in entries:
        if e.get("status") != "ok":
            continue
        rows.append(
            {
                "data_idx": e["data_idx"], "phase": e["phase"], "actual": e["actual"],
                "pattern": e["pattern"], "base_probability": e["base_probability"],
                "residual_adjustment": e["residual_adjustment"], "probability_yes": e["probability_yes"],
                "deviation_direction": e["deviation_direction"], "memory_count": e.get("memory_count", 0),
                "memory_source_indices": json.dumps(e.get("memory_source_indices", [])),
                "dominant_hbm_factors": json.dumps(e.get("dominant_hbm_factors", []), ensure_ascii=False),
                "residual_observed_factors": json.dumps(e.get("residual_observed_factors", []), ensure_ascii=False),
                "reason": e.get("reason", ""),
            }
        )
    return pd.DataFrame(rows)


def parse_args_pipeline() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HBM5 proxy construction + two-pass offline reflective-memory influenza vaccination prediction."
    )
    p.add_argument("--input-csv", "--input_csv", dest="input_csv", required=True)
    p.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    p.add_argument("--api-key", "--api_key", dest="api_key", default="")
    p.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    p.add_argument("--reflection-model", "--reflection_model", dest="reflection_model", default="")
    p.add_argument("--sample-size", "--sample_size", dest="sample_size", type=int, default=100)
    p.add_argument("--class-sampling", "--class_sampling", dest="class_sampling", choices=["proportional", "balanced", "custom"], default="proportional")
    p.add_argument("--positive-fraction", "--positive_fraction", dest="positive_fraction", type=float, default=0.5)
    p.add_argument("--preserve-pattern-within-class", "--preserve_pattern_within_class", dest="preserve_pattern_within_class", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--memory-ratio", "--memory_ratio", dest="memory_ratio", type=float, default=0.40)
    p.add_argument("--calibration-ratio", "--calibration_ratio", dest="calibration_ratio", type=float, default=0.20)
    p.add_argument("--test-ratio", "--test_ratio", dest="test_ratio", type=float, default=0.40)
    p.add_argument("--random-seed", "--random_seed", dest="random_seed", type=int, default=42)
    p.add_argument("--pattern-threshold-mode", "--pattern_threshold_mode", dest="pattern_threshold_mode", choices=["median", "fixed"], default="median")
    p.add_argument("--fixed-motivation-threshold", "--fixed_motivation_threshold", dest="fixed_motivation_threshold", type=float, default=2.5)
    p.add_argument("--fixed-capability-threshold", "--fixed_capability_threshold", dest="fixed_capability_threshold", type=float, default=2.5)
    p.add_argument("--fixed-activation-threshold", "--fixed_activation_threshold", dest="fixed_activation_threshold", type=float, default=2.5)
    p.add_argument("--base-rate-prior-strength", "--base_rate_prior_strength", dest="base_rate_prior_strength", type=float, default=10.0)
    p.add_argument("--reflection-low-cutoff", "--reflection_low_cutoff", dest="reflection_low_cutoff", type=float, default=30.0)
    p.add_argument("--reflection-high-cutoff", "--reflection_high_cutoff", dest="reflection_high_cutoff", type=float, default=70.0)
    p.add_argument("--max-reflection-candidates", "--max_reflection_candidates", dest="max_reflection_candidates", type=int, default=64)
    p.add_argument("--max-reflection-calls-per-bucket", "--max_reflection_calls_per_bucket", dest="max_reflection_calls_per_bucket", type=int, default=6)
    p.add_argument("--min-reflection-confidence", "--min_reflection_confidence", dest="min_reflection_confidence", type=float, default=0.65)
    p.add_argument("--min-estimated-memory-value", "--min_estimated_memory_value", dest="min_estimated_memory_value", type=float, default=0.60)
    p.add_argument("--min-memory-novelty", "--min_memory_novelty", dest="min_memory_novelty", type=float, default=0.25)
    p.add_argument("--min-memory-value", "--min_memory_value", dest="min_memory_value", type=float, default=0.55)
    p.add_argument("--max-memories-per-bucket", "--max_memories_per_bucket", dest="max_memories_per_bucket", type=int, default=2)
    p.add_argument("--max-memory-items", "--max_memory_items", dest="max_memory_items", type=int, default=32)
    p.add_argument("--memory-top-k", "--memory_top_k", dest="memory_top_k", type=int, default=2)
    p.add_argument("--memory-min-similarity", "--memory_min_similarity", dest="memory_min_similarity", type=float, default=0.45)
    p.add_argument("--include-sensitive-context", "--include_sensitive_context", dest="include_sensitive_context", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--decision-max-tokens", "--decision_max_tokens", dest="decision_max_tokens", type=int, default=260)
    p.add_argument("--reflection-max-tokens", "--reflection_max_tokens", dest="reflection_max_tokens", type=int, default=420)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--max-retries", "--max_retries", dest="max_retries", type=int, default=4)
    p.add_argument("--max-concurrent-requests", "--max_concurrent_requests", dest="max_concurrent_requests", type=int, default=20)
    p.add_argument("--concurrent-samples", "--concurrent_samples", dest="concurrent_samples", type=int, default=12)
    p.add_argument("--threshold-metric", "--threshold_metric", dest="threshold_metric", choices=["balanced_accuracy", "f1", "accuracy"], default="balanced_accuracy")
    p.add_argument("--run-test-without-memory", "--run_test_without_memory", dest="run_test_without_memory", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--continue-on-error", "--continue_on_error", dest="continue_on_error", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    p.add_argument("--print-samples", "--print_samples", dest="print_samples", type=int, default=3)
    return p.parse_args()


def resolve_key(explicit: str) -> str:
    key = explicit.strip() or os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY or pass --api-key")
    return key


def phase_distribution(df: pd.DataFrame, indices: Sequence[int]) -> List[Dict[str, Any]]:
    rows = []
    phase_df = df.iloc[list(indices)]
    for pattern, g in phase_df.groupby("hbm8_pattern"):
        rows.append({"pattern": pattern, "n": int(len(g)), "actual_vax_rate": float(g["vaccinated"].mean())})
    return sorted(rows, key=lambda x: PATTERN_THEORY_ORDER.get(x["pattern"], 99))


async def async_pipeline_main() -> None:
    args = parse_args_pipeline()
    input_path = Path(args.input_csv).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_log = logs_dir / "train_call1_no_memory.jsonl"
    reflection_log = logs_dir / "train_call2_reflections.jsonl"
    calibration_log = logs_dir / "calibration_call1_with_memory.jsonl"
    test_log = logs_dir / "test_call1_with_memory.jsonl"
    test_nomemory_log = logs_dir / "test_call1_without_memory.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(input_path)
    header = pd.read_csv(input_path, nrows=0)
    missing = [c for c in REQUIRED_COLUMNS if c not in header.columns]
    if missing:
        raise ValueError("Raw NHIS file is missing required columns: " + ", ".join(missing))

    print("=" * 88)
    print("FLARE-VAX HBM5 — Offline Reflective Memory")
    print("=" * 88)
    print(f"Version                  : {PIPELINE_VERSION}")
    print(f"Input                    : {input_path}")
    print(f"Output                   : {output_dir}")
    print(f"Model                    : {args.model}")
    print(f"Reflection model         : {args.reflection_model or args.model}")
    print(f"Sample size              : {args.sample_size}")
    print(f"Split                    : memory={args.memory_ratio}, calibration={args.calibration_ratio}, test={args.test_ratio}")
    print(f"Test without memory      : {args.run_test_without_memory}")

    raw = pd.read_csv(input_path, usecols=REQUIRED_COLUMNS, low_memory=False)
    profiles = build_scores(raw)
    profiles = profiles[profiles["vaccinated"].notna()].copy()
    valid_target_rows = len(profiles)
    profiles["source_index"] = profiles.index.astype(int)

    # Preliminary fixed pattern is used only for class-aware sampling and split stratification.
    preliminary = assign_patterns(profiles, 2.5, 2.5, 2.5, suffix="pre")
    profiles["preliminary_pattern"] = preliminary["hbm8_pattern_pre"]
    profiles = sample_by_class_pattern(profiles, args)
    profiles = profiles.reset_index(drop=True)

    splits = split_three_way(profiles, args)
    preliminary_phase = np.full(len(profiles), "", dtype=object)
    for phase, idxs in splits.items():
        preliminary_phase[idxs] = phase
    profiles["phase"] = preliminary_phase

    thresholds = fit_pattern_thresholds(profiles.iloc[splits["memory"]], args)
    profiles = assign_final_patterns(profiles, thresholds)
    # Reorder by phase for readable data_idx while preserving the split membership.
    phase_order = pd.Categorical(profiles["phase"], categories=["memory", "calibration", "test"], ordered=True)
    profiles = profiles.assign(_phase_order=phase_order).sort_values(["_phase_order", "source_index"]).drop(columns="_phase_order").reset_index(drop=True)
    profiles["data_idx"] = np.arange(len(profiles), dtype=int)
    new_splits = {
        phase: profiles.index[profiles["phase"] == phase].to_numpy(dtype=int)
        for phase in ["memory", "calibration", "test"]
    }
    splits = new_splits

    raw_selected = raw.loc[profiles["source_index"].astype(int).tolist()].copy().reset_index(drop=True)
    memory_df = profiles.iloc[splits["memory"]].copy()
    base_rates = fit_pattern_base_rates(memory_df, args.base_rate_prior_strength)
    similarity_space = fit_similarity_space(profiles, splits["memory"])

    print(f"Rows with valid target    : {valid_target_rows:,}")
    print(f"Selected rows             : {len(profiles):,}")
    print(f"Selected target rate      : {profiles['vaccinated'].mean():.1%}")
    print(f"Pattern thresholds        : {json.dumps(thresholds)}")
    for phase in ["memory", "calibration", "test"]:
        print(f"[{phase}] n={len(splits[phase])}")
        for x in phase_distribution(profiles, splits[phase]):
            print(f"  {x['pattern']:<55} n={x['n']:>4} actual_vax={x['actual_vax_rate']:.1%}")
    print("Training-only pattern base rates:")
    for pattern in PATTERN_THEORY_ORDER:
        info = base_rates[pattern]
        print(f"  {pattern:<55} n={int(info['n']):>4} raw={info['raw_rate']:.1%} smoothed={info['smoothed_rate']:.1%}")

    run_config = {
        "version": PIPELINE_VERSION,
        "created_at": pipeline_utc_now(),
        "input_csv": str(input_path),
        "model": args.model,
        "reflection_model": args.reflection_model or args.model,
        "sample_size": len(profiles),
        "class_sampling": args.class_sampling,
        "split_ratios": {"memory": args.memory_ratio, "calibration": args.calibration_ratio, "test": args.test_ratio},
        "pattern_thresholds": thresholds,
        "base_rate_prior_strength": args.base_rate_prior_strength,
        "reflection_cutoffs": {"low": args.reflection_low_cutoff, "high": args.reflection_high_cutoff},
        "memory": {
            "top_k": args.memory_top_k,
            "min_similarity": args.memory_min_similarity,
            "max_items": args.max_memory_items,
            "max_per_pattern_direction": args.max_memories_per_bucket,
        },
        "test_without_memory": args.run_test_without_memory,
    }
    run_config["fingerprint"] = config_hash(run_config)
    (output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    profiles.to_csv(output_dir / "hbm5_selected_profiles.csv", index=False)
    pd.DataFrame(
        [{"pattern": p, **v} for p, v in base_rates.items() if p != "__overall__"]
    ).to_csv(output_dir / "pattern_base_rates.csv", index=False)

    # Dry-run prints real prompts without making API calls.
    if args.dry_run:
        idx = int(splits["memory"][0])
        row = profiles.iloc[idx]
        base, base_n = row_pattern_anchor(row, base_rates, leave_one_out=True, memory_df=memory_df, prior_strength=args.base_rate_prior_strength)
        profile = build_full_observed_profile(raw_selected.iloc[idx], row, include_sensitive_context=args.include_sensitive_context)
        print("\n--- DRY RUN CALL 1 PROMPT ---\n")
        print(build_decision_prompt(observed_profile=profile, pattern_base_probability=base, pattern_n=base_n, retrieved_memories=[]))
        hypothetical = {
            "residual_adjustment": -20, "probability_yes": max(0.0, base - 20),
            "dominant_hbm_factors": ["example factor"],
            "residual_observed_factors": ["example residual"],
            "reason": "Hypothetical error for prompt inspection.",
        }
        direction = "increase" if int(row["vaccinated"]) == 1 else "decrease"
        err_type = "underprediction" if direction == "increase" else "overprediction"
        print("\n--- DRY RUN CALL 2 PROMPT ---\n")
        print(build_reflection_prompt(observed_profile=profile, decision=hypothetical, actual=int(row["vaccinated"]), pattern_base_probability=base, expected_direction=direction, expected_error_type=err_type))
        print("\nDry run complete; no API calls were made.")
        return

    if AsyncOpenAI is None:
        raise RuntimeError("Install the official OpenAI Python SDK: pip install -U openai")
    client = AsyncOpenAI(api_key=resolve_key(args.api_key), timeout=args.timeout, max_retries=args.max_retries)
    semaphore = asyncio.Semaphore(max(1, args.max_concurrent_requests))

    train_latest = load_latest_jsonl(train_log)
    reflection_latest = load_latest_jsonl(reflection_log)
    calibration_latest = load_latest_jsonl(calibration_log)
    test_latest = load_latest_jsonl(test_log)
    test_nomemory_latest = load_latest_jsonl(test_nomemory_log)

    # Pass 1: all memory-build respondents receive Call 1 without memory.
    await run_decision_batch(
        indices=splits["memory"], phase="train_no_memory", scores=profiles, raw_selected=raw_selected,
        args=args, client=client, semaphore=semaphore, log_path=train_log, latest=train_latest,
        base_rates=base_rates, memory_df=memory_df, memory_store=None, leave_one_out=True,
    )

    train_entries = [train_latest[i] for i in splits["memory"] if i in train_latest]
    candidates = select_reflection_candidates(train_entries, args)
    pd.DataFrame(
        [{k: v for k, v in c.items() if k not in {"observed_profile", "raw_response"}} for c in candidates]
    ).to_csv(output_dir / "reflection_candidates.csv", index=False)
    print(f"High-confidence reflection candidates: {len(candidates)}")

    # Pass 2: Call 2 only on selected high-confidence training errors.
    await run_reflection_batch(
        candidates=candidates, args=args, client=client, semaphore=semaphore,
        log_path=reflection_log, latest=reflection_latest,
    )
    reflection_entries = [reflection_latest[int(c["data_idx"])] for c in candidates if int(c["data_idx"]) in reflection_latest]
    memory_items = distill_memory(reflection_entries, args)
    memory_store = MemoryStore(
        items=memory_items,
        similarity_space=similarity_space,
        min_similarity=args.memory_min_similarity,
        top_k=args.memory_top_k,
    )
    with (output_dir / "memory_final.jsonl").open("w", encoding="utf-8") as f:
        for item in memory_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    pd.DataFrame(memory_items).to_csv(output_dir / "memory_final.csv", index=False)
    print(f"Frozen high-value memory: {len(memory_items)} items")

    # Calibration and test use the same frozen memory. Test defaults to memory-only.
    await run_decision_batch(
        indices=splits["calibration"], phase="calibration_with_memory", scores=profiles, raw_selected=raw_selected,
        args=args, client=client, semaphore=semaphore, log_path=calibration_log, latest=calibration_latest,
        base_rates=base_rates, memory_df=memory_df, memory_store=memory_store, leave_one_out=False,
    )
    calibration_entries = [calibration_latest[i] for i in splits["calibration"] if i in calibration_latest]
    llm_threshold, threshold_table, calibration_summary = calibrate_threshold(
        calibration_entries, "probability_yes", args.threshold_metric
    )
    base_threshold, base_threshold_table, base_calibration_summary = calibrate_threshold(
        calibration_entries, "base_probability", args.threshold_metric
    )
    threshold_table.to_csv(output_dir / "threshold_search_llm.csv", index=False)
    base_threshold_table.to_csv(output_dir / "threshold_search_pattern_only.csv", index=False)
    (output_dir / "threshold_calibration.json").write_text(
        json.dumps({"llm_with_memory": calibration_summary, "pattern_only": base_calibration_summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    await run_decision_batch(
        indices=splits["test"], phase="test_with_memory", scores=profiles, raw_selected=raw_selected,
        args=args, client=client, semaphore=semaphore, log_path=test_log, latest=test_latest,
        base_rates=base_rates, memory_df=memory_df, memory_store=memory_store, leave_one_out=False,
    )

    if args.run_test_without_memory:
        await run_decision_batch(
            indices=splits["test"], phase="test_without_memory", scores=profiles, raw_selected=raw_selected,
            args=args, client=client, semaphore=semaphore, log_path=test_nomemory_log, latest=test_nomemory_latest,
            base_rates=base_rates, memory_df=memory_df, memory_store=None, leave_one_out=False,
        )

    test_entries = [test_latest[i] for i in splits["test"] if i in test_latest]
    train_ok = [e for e in train_entries if e.get("status") == "ok"]
    cal_ok = [e for e in calibration_entries if e.get("status") == "ok"]
    test_ok = [e for e in test_entries if e.get("status") == "ok"]

    train_df = entries_dataframe(train_ok)
    cal_df = entries_dataframe(cal_ok)
    test_df = entries_dataframe(test_ok)
    train_df.to_csv(output_dir / "train_predictions_no_memory.csv", index=False)
    cal_df.to_csv(output_dir / "calibration_predictions_with_memory.csv", index=False)
    test_df.to_csv(output_dir / "test_predictions_with_memory.csv", index=False)

    y_test = np.array([float(e["actual"]) for e in test_ok])
    p_test = np.array([float(e["probability_yes"]) for e in test_ok])
    p_base = np.array([float(e["base_probability"]) for e in test_ok])
    metrics = {
        "train_no_memory_at_50": binary_metrics(
            np.array([float(e["actual"]) for e in train_ok]),
            np.array([float(e["probability_yes"]) for e in train_ok]),
            50.0,
        ),
        "calibration_with_memory_selected": binary_metrics(
            np.array([float(e["actual"]) for e in cal_ok]),
            np.array([float(e["probability_yes"]) for e in cal_ok]),
            llm_threshold,
        ) if cal_ok else {},
        "test_with_memory_at_50": binary_metrics(y_test, p_test, 50.0),
        "test_with_memory_selected": binary_metrics(y_test, p_test, llm_threshold),
        "test_pattern_only_selected": binary_metrics(y_test, p_base, base_threshold),
    }
    if args.run_test_without_memory:
        nm_entries = [test_nomemory_latest[i] for i in splits["test"] if test_nomemory_latest.get(i, {}).get("status") == "ok"]
        entries_dataframe(nm_entries).to_csv(output_dir / "test_predictions_without_memory.csv", index=False)
        metrics["test_without_memory_selected"] = binary_metrics(
            np.array([float(e["actual"]) for e in nm_entries]),
            np.array([float(e["probability_yes"]) for e in nm_entries]),
            llm_threshold,
        )

    # Pattern-specific test metrics.
    pattern_rows = []
    for pattern in PATTERN_THEORY_ORDER:
        subset = [e for e in test_ok if e["pattern"] == pattern]
        if not subset:
            continue
        pattern_rows.append(
            {
                "pattern": pattern,
                **binary_metrics(
                    np.array([float(e["actual"]) for e in subset]),
                    np.array([float(e["probability_yes"]) for e in subset]),
                    llm_threshold,
                ),
            }
        )
    pd.DataFrame(pattern_rows).to_csv(output_dir / "test_pattern_metrics.csv", index=False)

    all_usage_entries: List[Mapping[str, Any]] = []
    all_usage_entries.extend(train_latest.values())
    all_usage_entries.extend(reflection_latest.values())
    all_usage_entries.extend(calibration_latest.values())
    all_usage_entries.extend(test_latest.values())
    if args.run_test_without_memory:
        all_usage_entries.extend(test_nomemory_latest.values())

    summary = {
        "experiment": "flare_vax_hbm5_offline_reflective_memory",
        "version": PIPELINE_VERSION,
        "created_at": pipeline_utc_now(),
        "run_config": run_config,
        "n_selected": len(profiles),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "pattern_base_rates": base_rates,
        "reflection_candidates": len(candidates),
        "final_memory_items": len(memory_items),
        "selected_llm_threshold": llm_threshold,
        "selected_pattern_threshold": base_threshold,
        "calibration": {"llm_with_memory": calibration_summary, "pattern_only": base_calibration_summary},
        "metrics": metrics,
        "usage_total": sum_usage(all_usage_entries),
        "important_design_note": (
            "The default test pass uses frozen memory only. Enable --run-test-without-memory for a direct same-test memory ablation."
        ),
        "files": {
            "profiles": str(output_dir / "hbm5_selected_profiles.csv"),
            "pattern_base_rates": str(output_dir / "pattern_base_rates.csv"),
            "memory": str(output_dir / "memory_final.jsonl"),
            "train_predictions": str(output_dir / "train_predictions_no_memory.csv"),
            "calibration_predictions": str(output_dir / "calibration_predictions_with_memory.csv"),
            "test_predictions": str(output_dir / "test_predictions_with_memory.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=json_safe), encoding="utf-8")
    await client.close()

    print("\n" + "=" * 88)
    print("RUN SUMMARY")
    print("=" * 88)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_safe))
    print(f"\nSummary: {output_dir / 'summary.json'}")


def main() -> None:
    asyncio.run(async_pipeline_main())


if __name__ == "__main__":
    main()
