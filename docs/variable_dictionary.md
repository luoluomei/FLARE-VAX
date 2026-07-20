# Variable and Construct Dictionary

## Outcome

| Repository field | NHIS variable | Meaning |
|---|---|---|
| `vaccinated` | `SHTFLU12M_A` | Flu vaccination during the past 12 months |

## HBM2 threat inputs

| Clean field | NHIS source |
|---|---|
| `age` | `AGEP_A` |
| `health_status` | `PHSTAT_A` |
| `bmi_category` | `BMICAT_A` |
| `hypertension_yes` | `HYPEV_A` |
| `heart_disease_yes` | `CHDEV_A` |
| `angina_yes` | `ANGEV_A` |
| `heart_attack_yes` | `MIEV_A` |
| `stroke_yes` | `STREV_A` |
| `asthma_ever_yes` | `ASEV_A` |
| `asthma_current_yes` | `ASTILL_A` |
| `asthma_episode_12m_yes` | `ASAT12M_A` |
| `cancer_ever_yes` | `CANEV_A` |
| `diabetes_yes` | `DIBEV_A` |
| `copd_yes` | `COPDEV_A` |
| `weak_kidneys_yes` | `KIDWEAKEV_A` |
| `liver_condition_yes` | `LIVEREV_A` |
| `hepatitis_ever_yes` | `HEPEV_A` |
| `disability_yes` | `DISAB3_A` |
| `any_functional_difficulty_yes` | `ANYDIFF_A` |

## HBM2 barrier inputs

| Clean field | NHIS source |
|---|---|
| Current coverage / uninsured status | `HICOV_A`, `NOTCOV_A` |
| Coverage type | `COVER_A` |
| Medicare / Medicaid / private | `MEDICARE_A`, `MEDICAID_A`, `PRIVATE_A` |
| Past-year uninsurance | `HINOTYR_A`, `HINOTMYR_A` |
| Insurance unaffordable | `RSNHICOST_A` |
| Coverage stopped because of cost | `HISTOPCOST_A` |
| Delayed or forgone medical care | `MEDDL12M_A`, `MEDNG12M_A` |
| Medical-bill problems | `PAYBLL12M_A`, `PAYNOBLLNW_A`, `PAYWORRY_A` |
| Deductible | `PRDEDUC1_A`, `PRDEDUC2_A` |
| Usual source of care | `USUALPL_A`, `USPLKIND_A` |
| Transportation barrier | `TRANSPOR_A` |
| Language at doctor | `LANGDOC_A` |
| Internet access | `ACCSSINT_A` |

## HBM5 prior-vaccine acceptance proxy

This proxy is allowed only in the “with other vaccine history” setting.

| Component | NHIS source |
|---|---|
| COVID-19 vaccine acceptance | `SHTCVD191_A`, `SHTCVD19NM2_A` |
| Pneumonia vaccine acceptance | `SHTPNUEV_A` |
| Shingles vaccine acceptance | `SHTSHINGL1_A`, `SHINGRIX3_A` |
| Weak positive hepatitis A evidence | `SHTHEPA_A` |

The construct should be called an **observed vaccine-acceptance/benefit proxy**, not a direct perceived-benefit measure.

## Preventive engagement proxy in the no-prior version

| Component | NHIS source |
|---|---|
| Preventive wellness engagement | `WELLNESS_A`, `WELLVIS_A` |
| Health-information seeking | `HITLOOK_A` |
| Online doctor communication | `HITCOMM_A` |
| Online review of test results | `HITTEST_A` |

## Healthcare cues

| Component | NHIS source |
|---|---|
| Doctor recency | `LASTDR_A` |
| Wellness contact | `WELLNESS_A`, `WELLVIS_A` |
| Retail or virtual care | `RETAILHC12MTC_A`, `VIRAPP12M_A` |
| Acute-care contact | `URGCC12MTC_A`, `EMERG12MTC_A`, `HOSPONGT_A` |

This is an opportunity-for-cue proxy. It does not directly observe a physician recommendation or reminder.

## Navigation self-efficacy

| Component | NHIS source |
|---|---|
| Usual source of care | `USUALPL_A` |
| Stable care setting | `USPLKIND_A` |
| Internet access | `ACCSSINT_A`, `ACCSSHOM_A` |
| Digital health navigation | `HITLOOK_A`, `HITCOMM_A`, `HITTEST_A` |
| Virtual-care experience | `VIRAPP12M_A` |
| Communication capacity | `COMDIFF_A` |

This is an observed navigation-capacity proxy, not a direct confidence scale.
