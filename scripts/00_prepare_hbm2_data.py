#!/usr/bin/env python3
"""Create the cleaned HBM2 table used by the three-call FLARE-VAX runners.

The script intentionally performs variable-specific missing-code handling. It does
not globally convert 7/8/9 to missing because some NHIS variables legitimately use
those values (for example education, income-to-poverty ratio, and HISPALLP_A).

The resulting constructs are observable theory-guided proxies, not direct
psychometric measurements of Health Belief Model beliefs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TARGET = "SHTFLU12M_A"

RAW_COLUMNS = [
    TARGET, "WTFA_A", "HHX", "SRVY_YR", "PSTRAT", "PPSU",
    "AGEP_A", "PHSTAT_A", "BMICAT_A", "SMKCIGST_A",
    "HYPEV_A", "CHDEV_A", "ANGEV_A", "MIEV_A", "STREV_A",
    "ASEV_A", "ASTILL_A", "ASAT12M_A", "CANEV_A", "DIBEV_A",
    "COPDEV_A", "KIDWEAKEV_A", "LIVEREV_A", "HEPEV_A",
    "DISAB3_A", "ANYDIFF_A",
    "HICOV_A", "NOTCOV_A", "COVER_A", "MEDICARE_A", "MEDICAID_A",
    "PRIVATE_A", "HINOTYR_A", "HINOTMYR_A", "RSNHICOST_A",
    "HISTOPCOST_A", "MEDDL12M_A", "MEDNG12M_A", "PAYBLL12M_A",
    "PAYNOBLLNW_A", "PAYWORRY_A", "PRDEDUC1_A", "PRDEDUC2_A",
    "USUALPL_A", "USPLKIND_A", "TRANSPOR_A", "LANGDOC_A",
    "ACCSSINT_A", "RETAILHC12MTC_A", "LASTDR_A", "HITTEST_A",
    "HITCOMM_A", "EDUCP_A", "RATCAT_A", "HISPALLP_A",
    "WELLVIS_A", "WELLNESS_A",
]

DISEASE_MAP = {
    "hypertension": "HYPEV_A",
    "heart_disease": "CHDEV_A",
    "angina": "ANGEV_A",
    "heart_attack": "MIEV_A",
    "stroke": "STREV_A",
    "asthma_ever": "ASEV_A",
    "asthma_current": "ASTILL_A",
    "asthma_episode_12m": "ASAT12M_A",
    "cancer_ever": "CANEV_A",
    "diabetes": "DIBEV_A",
    "copd": "COPDEV_A",
    "weak_kidneys": "KIDWEAKEV_A",
    "liver_condition": "LIVEREV_A",
    "hepatitis_ever": "HEPEV_A",
    "disability": "DISAB3_A",
    "any_functional_difficulty": "ANYDIFF_A",
}


def yes_no(s: pd.Series) -> pd.Series:
    """Standard NHIS yes/no item: 1 -> 1, 2 -> 0, else missing."""
    x = pd.to_numeric(s, errors="coerce")
    return x.map({1: 1.0, 2: 0.0})


def coverage_recode(s: pd.Series) -> pd.Series:
    """NHIS insurance recodes: 1/2 = yes, 3 = no, 7/8/9 = missing."""
    x = pd.to_numeric(s, errors="coerce")
    return x.map({1: 1.0, 2: 1.0, 3: 0.0})


def valid_numeric(s: pd.Series, *, valid: Iterable[int] | None = None,
                  invalid: Iterable[int] = ()) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if valid is not None:
        x = x.where(x.isin(list(valid)))
    if invalid:
        x = x.mask(x.isin(list(invalid)))
    return x


def safe_yes_no(raw: pd.DataFrame, name: str) -> pd.Series:
    return yes_no(raw[name]) if name in raw else pd.Series(np.nan, index=raw.index)


def build_clean(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    out["vaccinated"] = yes_no(raw[TARGET])
    for c in ["WTFA_A", "HHX", "SRVY_YR", "PSTRAT", "PPSU"]:
        if c in raw:
            out[c.lower()] = raw[c]

    out["age"] = valid_numeric(raw["AGEP_A"], invalid=[97, 98, 99])
    out["age_50plus"] = (out["age"] >= 50).where(out["age"].notna()).astype(float)
    out["age_65plus"] = (out["age"] >= 65).where(out["age"].notna()).astype(float)
    out["health_status"] = valid_numeric(raw["PHSTAT_A"], valid=[1, 2, 3, 4, 5])
    out["poor_fair_health"] = out["health_status"].isin([4, 5]).where(out["health_status"].notna()).astype(float)
    out["bmi_category"] = valid_numeric(raw["BMICAT_A"], valid=[1, 2, 3, 4])
    out["smoking_status"] = valid_numeric(raw["SMKCIGST_A"], valid=[1, 2, 3])

    risk_cols = []
    for clean_name, raw_name in DISEASE_MAP.items():
        out[f"{clean_name}_yes"] = safe_yes_no(raw, raw_name)
        risk_cols.append(f"{clean_name}_yes")
    out["chronic_or_risk_count"] = out[risk_cols].sum(axis=1, min_count=1)
    out["any_chronic_or_risk_condition"] = (out["chronic_or_risk_count"] > 0).where(
        out["chronic_or_risk_count"].notna()).astype(float)

    # Four-point deterministic threat score:
    # age 50-64 = 1 point; age 65+ = 2 points; fair/poor health = 1;
    # any chronic/functional risk = 1. Threshold >=2 defines High Threat.
    age_points = out["age_50plus"].fillna(0) + out["age_65plus"].fillna(0)
    out["hbm_threat_score"] = (
        age_points
        + out["poor_fair_health"].fillna(0)
        + out["any_chronic_or_risk_condition"].fillna(0)
    ).clip(0, 4).astype(int)
    out["hbm_threat_level"] = np.where(out["hbm_threat_score"] >= 2, "high", "low")

    # Insurance/access variables.
    out["has_insurance_yes"] = safe_yes_no(raw, "HICOV_A")
    out["uninsured_yes"] = safe_yes_no(raw, "NOTCOV_A")
    # Fall back to inverse current coverage when NOTCOV_A is unavailable/missing.
    out["uninsured_yes"] = out["uninsured_yes"].fillna(1 - out["has_insurance_yes"])
    out["insurance_type"] = valid_numeric(raw["COVER_A"], valid=[1, 2, 3, 4, 5])
    # These three NCHS recodes use 1/2 = yes and 3 = no, unlike standard
    # binary NHIS items where 1 = yes and 2 = no.
    out["medicare_yes"] = coverage_recode(raw["MEDICARE_A"])
    out["medicaid_yes"] = coverage_recode(raw["MEDICAID_A"])
    out["private_insurance_yes"] = coverage_recode(raw["PRIVATE_A"])
    out["uninsured_past_year_yes"] = safe_yes_no(raw, "HINOTYR_A")
    out["months_uninsured"] = valid_numeric(raw["HINOTMYR_A"], invalid=[97, 98, 99]).clip(0, 12)
    out["no_insurance_cost_yes"] = safe_yes_no(raw, "RSNHICOST_A")
    out["lost_coverage_cost_increase_yes"] = safe_yes_no(raw, "HISTOPCOST_A")
    out["delayed_care_cost_12m_yes"] = safe_yes_no(raw, "MEDDL12M_A")
    out["needed_care_not_get_cost_12m_yes"] = safe_yes_no(raw, "MEDNG12M_A")
    out["problems_paying_medical_bills_12m_yes"] = safe_yes_no(raw, "PAYBLL12M_A")
    out["unable_pay_medical_bills_now_yes"] = safe_yes_no(raw, "PAYNOBLLNW_A")
    out["has_deductible_plan1_yes"] = safe_yes_no(raw, "PRDEDUC1_A")
    out["has_deductible_plan2_yes"] = safe_yes_no(raw, "PRDEDUC2_A")
    out["usual_care_place"] = valid_numeric(raw["USUALPL_A"], valid=[1, 2, 3])
    out["usual_care_type"] = valid_numeric(raw["USPLKIND_A"], valid=[1, 2, 3, 4, 5, 6])
    out["no_usual_care"] = (out["usual_care_place"] == 2).where(out["usual_care_place"].notna()).astype(float)
    out["multiple_or_no_usual_care"] = out["usual_care_place"].isin([2, 3]).where(
        out["usual_care_place"].notna()).astype(float)
    out["transportation_barrier_yes"] = safe_yes_no(raw, "TRANSPOR_A")
    # LANGDOC_A: 1 = English, 2 = another language in this public-use item.
    lang = valid_numeric(raw["LANGDOC_A"], valid=[1, 2])
    out["limited_language_at_doctor"] = (lang == 2).where(lang.notna()).astype(float)
    internet = safe_yes_no(raw, "ACCSSINT_A")
    out["no_internet_access"] = (1 - internet).where(internet.notna())
    worry = valid_numeric(raw["PAYWORRY_A"], valid=[1, 2, 3])
    out["worry_medical_bills_any"] = worry.isin([1, 2]).where(worry.notna()).astype(float)

    barrier_cols = [
        "uninsured_yes", "uninsured_past_year_yes", "no_insurance_cost_yes",
        "lost_coverage_cost_increase_yes", "delayed_care_cost_12m_yes",
        "needed_care_not_get_cost_12m_yes", "problems_paying_medical_bills_12m_yes",
        "unable_pay_medical_bills_now_yes", "no_usual_care",
        "transportation_barrier_yes", "worry_medical_bills_any",
        "limited_language_at_doctor", "no_internet_access",
    ]
    out["hbm_barrier_count"] = out[barrier_cols].sum(axis=1, min_count=1)
    # 0=no observed barrier, 1=one, 2=two/three, 3=four or more.
    count = out["hbm_barrier_count"]
    out["hbm_barrier_score"] = pd.cut(
        count, bins=[-np.inf, 0, 1, 3, np.inf], labels=[0, 1, 2, 3]
    ).astype("float").fillna(0).astype(int)
    out["hbm_barrier_level"] = np.where(out["hbm_barrier_score"] >= 2, "high", "low")

    high_t = out["hbm_threat_score"] >= 2
    high_b = out["hbm_barrier_score"] >= 2
    out["hbm2_pattern"] = np.select(
        [high_t & ~high_b, high_t & high_b, ~high_t & ~high_b, ~high_t & high_b],
        [0, 1, 2, 3], default=np.nan,
    )

    # Context used by CALL 3 and similarity memory.
    out["retail_clinic_visits_12m"] = valid_numeric(raw["RETAILHC12MTC_A"], valid=[0, 1, 2, 3, 4, 5])
    out["last_doctor_visit"] = valid_numeric(raw["LASTDR_A"], valid=[0, 1, 2, 3, 4, 5, 6])
    out["used_internet_test_results_yes"] = safe_yes_no(raw, "HITTEST_A")
    out["used_internet_communicate_doctor_yes"] = safe_yes_no(raw, "HITCOMM_A")
    out["education"] = valid_numeric(raw["EDUCP_A"])  # do not globally remove 7/8/9
    out["income_poverty_ratio"] = valid_numeric(raw["RATCAT_A"], valid=range(1, 15))
    out["hispanic"] = valid_numeric(raw["HISPALLP_A"], valid=range(1, 8))
    out["time_since_wellness_visit"] = valid_numeric(raw["WELLVIS_A"], valid=[0, 1, 2, 3, 4, 5, 6])
    out["last_visit_wellness_yes"] = safe_yes_no(raw, "WELLNESS_A")

    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--metadata_json", default="")
    args = p.parse_args()

    header = pd.read_csv(args.input_csv, nrows=0).columns
    missing = [c for c in RAW_COLUMNS if c not in header]
    if missing:
        raise KeyError(f"Missing required NHIS columns: {missing}")
    raw = pd.read_csv(args.input_csv, usecols=RAW_COLUMNS, low_memory=False)
    clean = build_clean(raw)
    clean = clean[clean["vaccinated"].isin([0, 1])].copy()
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.output_csv, index=False)

    metadata = {
        "input_rows": int(len(raw)),
        "output_rows_with_valid_target": int(len(clean)),
        "output_columns": int(clean.shape[1]),
        "target_rate": float(clean["vaccinated"].mean()),
        "pattern_counts": clean["hbm2_pattern"].value_counts(dropna=False).sort_index().to_dict(),
        "important_note": "Observable theory-guided proxies; not direct HBM psychometric scales.",
    }
    meta_path = args.metadata_json or str(Path(args.output_csv).with_suffix(".metadata.json"))
    Path(meta_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
