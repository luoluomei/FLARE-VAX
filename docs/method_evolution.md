# Method Evolution and Relation to FLARE

## 1. What is transferred from the original FLARE framework?

The original FLARE pipeline contains five broad ideas:

1. Behavioral theory determines which latent perceptions matter.
2. Survey variables are selected or organized around those perceptions.
3. Multiple reasoning patterns represent heterogeneous decision logic.
4. An LLM turns perceptions, context, and a reasoning template into a decision.
5. Training errors are reflected on and retrieved from memory for later cases.

The vaccination experiments preserve this overall logic but adapt it to NHIS 2024 and the Health Belief Model (HBM).

## 2. What is not an exact replication?

The first HBM2 versions should be described as a **structural transfer**, not a direct replication:

- FLARE models wildfire **threat assessment** and **risk perception**.
- HBM2 models observable proxies for influenza **threat** and **barriers**.
- FLARE learns a reasoning-pattern classifier from repeated LLM trials.
- HBM2 constructs patterns deterministically from cleaned scores.
- FLARE calibrates perception generation using a knowledge base of perception examples.
- HBM2 asks separate LLM calls to score threat and barriers but anchors the pattern with deterministic rules.

The later HBM5 versions are more theoretically complete for vaccination, but are also further from the exact original FLARE implementation.

## 3. Detailed version-by-version workflow

## V1: HBM2 three-call direct transfer

### Python preprocessing

1. Recode the flu-vaccination target.
2. Construct a 0–4 threat score.
3. Construct a 0–3 barrier score.
4. Assign one of four HBM2 patterns.
5. Build a standardized feature matrix for cosine-similarity memory.

### LLM workflow

- **CALL 1:** observable threat profile -> score 1–5, level, one-sentence reason.
- **CALL 2:** observable barrier profile -> score 1–5, level, one-sentence reason.
- **CALL 3:** CALL 1 + CALL 2 + deterministic pattern + context + retrieved errors -> `YES`/`NO`.

### Learning mechanism

When a training prediction is wrong, the full error case is inserted into memory. Later samples retrieve the nearest errors. No separate reflection call is used.

## V1-Async: engineering-only acceleration

The concurrent script preserves the same prompts, variables, patterns, and memory logic. CALL 1 and CALL 2 are issued in parallel, and multiple respondents are processed concurrently.

Because memory updates happen after each batch, respondents within one batch share the same pre-batch memory snapshot. This is a small implementation difference from fully sequential online memory.

## V2: HBM2 reflection and calibration

### Added components

- CALL 3 returns a probability rather than only a class.
- Incorrect training cases can trigger a reflection call.
- Reflections are stored as structured correction rules.
- A held-out calibration set selects the decision threshold.
- Test inference uses frozen memory and a frozen threshold.

### Why this matters

V2 attempts to separate two questions:

1. Can the LLM rank respondents by vaccination likelihood?
2. What threshold converts that probability into a class under the observed class distribution?

## V3: Pattern anchor and dual memory

### Pattern anchor

For each HBM2 pattern, estimate a smoothed training-only base rate. This base rate becomes the starting probability for each respondent.

### LLM role

The LLM receives the base probability and can only make a bounded residual adjustment. This makes the incremental contribution of the LLM explicit.

### Dual memory

- **Prototype memory:** high-confidence correct examples.
- **Reflection memory:** high-confidence errors and their correction rules.

The retrieval step mixes the two sources so the model receives both typical cases and exceptions.

## V4: HBM5 with prior-vaccine history

### Deterministic proxies

1. Observed threat proxy.
2. Vaccine-acceptance/benefit proxy.
3. Structural-barrier proxy.
4. Healthcare-cue proxy.
5. Navigation-self-efficacy proxy.

### Meta-dimensions

- Motivation = mean(threat, vaccine acceptance/benefit proxy)
- Capability = mean(reversed barriers, self-efficacy)
- Activation = cues

### Pattern space

Three binary dimensions produce eight patterns. Pattern thresholds are fit on the memory-build split.

### LLM role

The LLM no longer creates the five scores. Python computes them deterministically. The LLM receives:

- all five proxy scores and supporting components,
- the three meta-dimensions,
- the eight-pattern label,
- a training-only pattern base rate,
- detailed observed variables,
- retrieved reflective rules.

It returns a residual adjustment and final vaccination probability.

## V5: HBM5 without other-vaccine history

### Main change

The vaccine-acceptance/benefit proxy is removed because it directly uses other vaccination outcomes. It is replaced by preventive engagement.

### Explicit exclusion

The following variables are forbidden from scoring, prompts, similarity features, and memory retrieval:

- `SHTCVD191_A`
- `SHTCVD19NM2_A`
- `SHTPNUEV_A`
- `SHTPNEUNB_A`
- `SHTSHINGL1_A`
- `SHINGRIX3_A`
- `SHTHEPA_A`
- `SHTFLUM_A`
- `SHTFLUY_A`

### Additional evaluation

The same test split is evaluated with memory and without memory, making the contribution of reflection memory directly observable.

## 4. Method comparison

| Dimension | V1 | V2 | V3 | V4 | V5 |
|---|---|---|---|---|---|
| Behavioral constructs | 2 | 2 | 2 | 5 | 5 |
| Patterns | 4 | 4 | 4 | 8 | 8 |
| Construct scores | LLM + rules | LLM + rules | LLM + rules | deterministic | deterministic |
| Main decision calls per respondent | 3 | 3 | 3 | 1 | 1 |
| Reflection | no | selected errors | selected errors | selected errors | selected errors |
| Pattern base rate | qualitative only | qualitative only | explicit anchor | explicit anchor | explicit anchor |
| Memory | raw errors | reflections | prototypes + reflections | distilled reflections | distilled reflections |
| Calibration set | no | yes | yes | yes | yes |
| Other vaccine history | no | no | no | yes | no |

## 5. What should be considered the primary innovation?

The strongest conceptual innovations over a direct FLARE transfer are:

1. **Residual decision formulation:** the LLM adjusts a transparent pattern probability instead of generating a probability from scratch.
2. **Theory expansion:** five observed proxies and eight patterns represent a richer vaccination decision profile.
3. **Feature-policy comparison:** the project explicitly distinguishes prediction with and without other-vaccine history.
4. **Memory ablation:** the no-prior version evaluates identical test cases with and without memory.

The results also identify a scientifically useful negative finding: reflective memory did not improve held-out prediction in the current implementation.
