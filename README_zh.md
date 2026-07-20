# FLARE-VAX：基于 NHIS 2024 的理论引导型流感疫苗接种行为预测

[English README](README.md)

这个仓库系统整理了将 **FLARE** 从山火疏散决策预测迁移到流感疫苗接种行为预测的一系列实验。使用的数据是 **2024 National Health Interview Survey（NHIS）Sample Adult**，预测目标为：

```text
SHTFLU12M_A：受访者在过去 12 个月内是否接种流感疫苗
```

这个仓库的重点不是宣称“版本越复杂，效果一定越好”，而是完整记录方法如何从一个接近原始 FLARE 的三次 LLM 调用架构，逐步发展为：

- 结构化的错误反思；
- 概率输出与 threshold calibration；
- pattern base-rate anchor；
- prototype 与 reflection 双记忆；
- 五个可复现的 HBM-inspired observable proxies；
- 三个 meta-dimensions 与八类 behavioral patterns；
- 完全排除其他疫苗历史的严格预测版本；
- pattern-only、LLM without memory 与 LLM with memory 的同测试集消融比较。

> **重要解释。** 本项目中的 HBM 构念都是根据 NHIS 可观测变量构建的 theory-guided proxies，并不是通过心理量表直接测量得到的 perceived susceptibility、perceived benefits、perceived barriers、cues to action 或 self-efficacy。

---

## 1. 研究目标

这个项目关注的不只是“LLM 能不能预测一个人是否接种流感疫苗”，而是：

> **行为理论应该如何进入一个 LLM 决策预测系统，以及 LLM 应该承担多大的决策责任？**

不同版本逐步测试了六种 LLM 角色：

1. **构念评估器：**分别从调查变量中评估 Threat 和 Barrier。
2. **直接分类器：**综合两个构念、pattern、背景变量和 memory，直接输出 YES/NO。
3. **概率估计器：**输出接种概率而不是只输出二元标签。
4. **错误反思器：**将训练错误转化为可复用的 correction rule。
5. **Residual adjuster：**从 pattern 的训练接种率出发，只进行有限的个体修正。
6. **结构化推理器：**明确经过 evidence synthesis、pattern interpretation、residual context 和 decision mapping 四个阶段。

项目最终形成两个主要预测场景：

- **允许使用其他疫苗接种历史：**COVID-19、肺炎、带状疱疹、Shingrix 和甲肝疫苗变量可以进入行为画像。
- **不允许使用其他疫苗接种历史：**除流感疫苗目标变量外，所有疫苗变量都不得进入评分、prompt、similarity representation、memory retrieval 或 reflection。

---

## 2. 仓库结构

```text
scripts/
  00_prepare_hbm2_data.py
  01_ml_baselines.py
  10_hbm2_three_call.py
  11_hbm2_three_call_async.py
  20_hbm2_reflection_calibration.py
  30_hbm2_pattern_anchor_dual_memory.py
  40_hbm5_prior_vaccine_reflective_memory.py
  50_hbm5_no_prior_vaccine_reflective_memory.py
  90_collect_results.py

configs/
  experiment_catalog.json
  baseline_feature_manifest.example.json

docs/
  method_evolution.md
  variable_dictionary.md
  evaluation_notes.md
  replication_commands.md

results/
  run_index.csv
  reported_llm_hbm_metrics.csv
  baseline_all_models.csv
  benchmark_without_other_vaccine_history.csv
  benchmark_with_other_vaccine_history.csv
  raw_logs/
  summaries/
```

每个实验脚本中都保留了完整 system prompt、user prompt template 和 JSON schema。README 重点解释 prompt 的组成、调用顺序以及每个模块承担的功能。

---

## 3. 方法演进总览

| 版本 | 行为表征 | LLM 的主要角色 | 每个样本的常规 LLM 调用 | 条件性 Reflection Call | 预测 Anchor | Memory 类型 | 是否使用其他疫苗历史 |
|---|---|---|---:|---|---|---|---|
| V1 | Threat + Barrier；4 类 HBM2 pattern | 评估两个构念并直接分类 | 3 | 无单独 reflection | 定性 HBM2 pattern | 原始错误案例 | 否 |
| V1-Async | 与 V1 相同 | 与 V1 相同 | 3，但并发执行 | 无 | 与 V1 相同 | 按 batch 更新的错误案例 | 否 |
| V2 | authoritative Threat + Barrier；4 类 pattern | 细化构念、输出概率并解释 | 3 | 错误训练样本增加 1 次 | 定性 pattern prior 或训练 pattern rate | LLM 生成的修正规则 | 否 |
| V3 | authoritative HBM2 + pattern 接种率 | 只输出有限的 residual adjustment | 3 | 高置信错误增加 1 次 | 数值化 pattern base rate | 正确 prototype + 错误 reflection | 否 |
| V4 | 5 个确定性 proxy；3 个 meta-dimension；8 类 pattern | 一次综合 residual probability decision | 1 | 部分高置信错误 | HBM8 pattern base rate | 离线筛选的 exception rules | 是 |
| V5 | 5 个确定性 proxy；3 个 meta-dimension；8 类 pattern | 四阶段结构化 residual decision | 1 | 部分高置信错误 | HBM8 pattern base rate | 定位失败阶段的 reflection rules | 完全排除 |

最重要的变化是：

> 随着 Python 中的理论表征越来越完整，LLM 的调用次数反而减少，LLM 的自由度也逐步被限制。

V1–V3 对一个常规样本通常调用 3 次 LLM；V4–V5 的五个 proxy 全部由 Python 计算，因此一个样本通常只需要 1 次综合 Decision Call。

---

# 4. 每个方法的详细实现

## 4.1 V1 — HBM2 三次调用的直接迁移版本

**代码：**`scripts/10_hbm2_three_call.py`  
**Version string：**`hbm2_openai_colab_v1`

### 4.1.1 方法目的

V1 是与原始 FLARE 结构最接近的简化迁移版本。它用两个 HBM-inspired 构念表征流感疫苗接种行为：

- **Threat：**个体对流感的可观测易感性与潜在严重性。
- **Barrier：**个体在保险、费用、就医、交通、语言和数字访问方面的可观测阻碍。

两个构念被组合为四类 rule-based HBM2 pattern：

| Pattern | Threat | Barrier | 理论倾向 |
|---|---|---|---|
| P0 | High | Low | 最有利于接种 |
| P1 | High | High | 有风险动机，但障碍可能阻止接种 |
| P2 | Low | Low | 接种较容易，但风险动机较弱 |
| P3 | Low | High | 最不利于接种 |

这里的 pattern 是 Python 在 LLM 调用前计算的，CALL 1 和 CALL 2 输出的 1–5 分并不用于重新定义 pattern。

### 4.1.2 第一步：构建确定性的 HBM2 Profile

`00_prepare_hbm2_data.py` 将原始 `adult24.csv` 转换为可供 LLM 使用的清洗数据。

Threat 侧主要使用：

- 年龄及 50+/65+ 指标；
- 自评健康；
- 糖尿病、COPD、癌症、心脏病、心绞痛、心梗、中风、高血压、哮喘、肾脏和肝脏疾病；
- disability 与 functional difficulty；
- BMI 和吸烟相关背景。

Barrier 侧主要使用：

- 当前及过去一年的保险状态；
- 因费用失去保险；
- 因费用延迟或放弃医疗；
- 医疗账单压力；
- 是否有固定就医地点；
- 交通障碍；
- 语言和沟通困难；
- 网络与数字访问条件。

Python 生成：

```text
hbm_threat_score
hbm_threat_level
hbm_barrier_score
hbm_barrier_level
hbm2_pattern
```

输入 LLM 的不是未经解释的 NHIS 数字代码，而是整理后的可读 profile。

### 4.1.3 CALL 1：Threat Assessment

#### LLM 在这一 Call 中的角色

LLM 只负责评估 Threat proxy，不允许：

- 直接预测是否接种；
- 使用 barrier 信息；
- 讨论 benefit、cue 或其他 HBM 构念；
- 推断没有在数据中观察到的私人态度。

#### 输入内容

输入是一个格式化的 Threat Profile，包含：

```text
Age
Age 50+
Age 65+
Self-rated health
Fair/poor health indicator
Observed chronic/risk conditions
Count of chronic/risk conditions
Functional difficulty
Disability
BMI category
Smoking-related context
```

#### System Prompt 的结构

System prompt 指定：

1. 模型身份是 public-health behavioral analyst；
2. 该 Call 只评估 susceptibility + severity；
3. 只能使用提供的可观测变量；
4. 不能预测接种结果；
5. 必须按照 1–5 rubric；
6. reason 必须为一句简短解释。

#### User Prompt 的结构

```text
TASK
  评估这个人的 influenza perceived-threat proxy。

DEFINITION
  Threat = susceptibility + severity。
  这些是 observable proxies，不是直接测量的私人信念。

SCORING RUBRIC
  1 = Very low
  2 = Low
  3 = Moderate
  4 = High
  5 = Very high

INPUT FORMAT
  个人 Threat Profile

OUTPUT FORMAT
  只返回结构化 JSON
```

#### 输出格式

```json
{
  "score": 4,
  "level": "high",
  "reason": "The respondent has concentrated age and chronic-risk evidence."
}
```

`level` 只能是：

```text
very_low
low
moderate
high
very_high
```

### 4.1.4 CALL 2：Barrier Assessment

#### LLM 在这一 Call 中的角色

LLM 只负责 Barrier proxy，不允许：

- 使用 Threat 信息；
- 推断未观察到的疫苗态度；
- 直接预测是否接种。

#### 输入内容

Barrier Profile 包含：

```text
Insurance status
Insurance instability
Cost-related delayed or forgone care
Medical-bill stress
Usual place of care
Transportation barrier
Language or communication difficulty
Internet and digital access
```

#### System Prompt 的结构

System prompt 要求模型：

1. 只关注财务、保险、医疗可及性、交通、语言和数字障碍；
2. 不能使用 Threat；
3. 不能预测 vaccination；
4. 按照 1–5 rubric；
5. 返回一句 reason。

#### User Prompt 的结构

```text
TASK
  评估获得流感疫苗的 perceived-barrier proxy。

DEFINITION
  保险、费用、就医、交通、语言和数字障碍。

SCORING RUBRIC
  1 = Very low
  ...
  5 = Very high

INPUT FORMAT
  Barrier Profile

OUTPUT FORMAT
  只返回结构化 JSON
```

#### 输出格式

```json
{
  "score": 3,
  "level": "moderate",
  "reason": "The respondent has cost concerns and no stable usual-care location."
}
```

### 4.1.5 CALL 3 之前如何检索 Memory

在训练阶段，如果某个样本预测错误，程序会直接将以下信息组合为一条 memory record：

- rule-based HBM2 pattern；
- CALL 1 Threat 输出；
- CALL 2 Barrier 输出；
- 选定的背景变量；
- 模型原来的预测；
- 正确的训练标签。

同时，程序会为样本构建标准化的数值特征向量。对于下一个样本，使用 cosine similarity 检索最相似的 `k` 个训练错误。

这一过程没有额外调用 LLM 来总结错误，因此 V1 的 memory 本质是：

```text
raw error-example memory
```

而不是 correction-rule memory。

### 4.1.6 CALL 3：最终 YES/NO Decision

#### 输入结构

CALL 3 的 user prompt 由六个部分组成：

```text
1. CALL 1 — THREAT ASSESSMENT
2. CALL 2 — BARRIER ASSESSMENT
3. RULE-BASED HBM2 PATTERN
4. OPTIONAL PATTERN PRIOR
5. ADDITIONAL CONTEXT VARIABLES
6. SIMILAR TRAINING ERRORS
```

其中 pattern prior 有三种模式：

- `theory`：只提供 P0–P3 的定性理论排序；
- `train_rate`：额外提供训练集中当前 pattern 的接种率；
- `none`：不提供显式 prior。

#### Decision Prompt 的约束

Prompt 要求 LLM：

1. 首先整合 Threat 与 Barrier 两个 Call；
2. 将 pattern 作为 directional prior，而不是自动标签；
3. 只有在额外背景变量提供明确证据时才调整；
4. 只有 memory case 与当前样本真正相似时才使用；
5. 不输出额外解释。

#### 输出格式

```json
{
  "decision": "YES"
}
```

或：

```json
{
  "decision": "NO"
}
```

### 4.1.7 训练和测试顺序

```text
TRAIN SAMPLE
  → CALL 1 Threat
  → CALL 2 Barrier
  → retrieve existing training errors
  → CALL 3 YES/NO
  → compare with true training label
  → if wrong, save raw error example

END OF TRAINING
  → freeze memory

TEST SAMPLE
  → CALL 1
  → CALL 2
  → retrieve frozen training errors
  → CALL 3
  → test label never updates memory
```

### 4.1.8 V1 中 LLM 实际承担了什么

V1 给予 LLM 的自由度最大。它需要：

- 评估两个理论构念；
- 决定如何整合构念；
- 判断是否遵循 pattern；
- 解释额外背景变量；
- 判断历史错误是否适用；
- 直接输出最终标签。

优点是与原始 FLARE 的多阶段推理结构接近；缺点是预测错误时，很难确定究竟是构念判断、pattern 解释、memory 使用还是最终映射出了问题。

---

## 4.2 V1-Async — 三次调用的并发实现

**代码：**`scripts/11_hbm2_three_call_async.py`  
**Version string：**`hbm2_openai_colab_v2_concurrent`

V1-Async 的 prompts、schemas、变量、pattern 和决策逻辑与 V1 相同。它不是新的理论模型，而是工程层面的加速版本。

### 4.2.1 并发执行顺序

对每个 batch：

1. 同一个样本的 CALL 1 与 CALL 2 并发运行；
2. 多个样本也可以同时运行；
3. 两个 assessment 都完成后，执行 CALL 3；
4. 按 data index 顺序写入日志；
5. 一个 batch 完成后，才将该 batch 中的训练错误加入 memory。

### 4.2.2 与顺序 V1 的实际差异

```text
顺序 V1：
case 1 出错
→ 立即加入 memory
→ case 2 可以检索 case 1

V1-Async：
一个 batch 开始时固定 memory snapshot M
→ batch 内所有 case 都使用 M
→ batch 完成后再统一加入新错误
```

因此 V1-Async 在 prompt 和理论上与 V1 一致，但 online memory update 的粒度从“每个样本”变成了“每个 batch”。

---

## 4.3 V2 — Authoritative HBM2 + Reflection Memory + Threshold Calibration

**代码：**`scripts/20_hbm2_reflection_calibration.py`  
**Version string：**`hbm2_openai_colab_v3_reflection_calibrated`

### 4.3.1 相比 V1 的主要变化

V2 增加了四个关键设计：

1. Python 计算的 High/Low Threat 与 Barrier 成为 authoritative class；
2. CALL 3 不再只输出 YES/NO，而是输出 probability + decision + reason；
3. 错误训练样本额外触发 Reflection Call；
4. Python 根据训练预测选择最终分类 threshold。

> **实现修正说明：**当前保留的 V2 脚本是在训练预测上选择 threshold，并没有使用独立 calibration split。真正的 memory/calibration/test 三分割从 V3 开始。

### 4.3.2 CALL 1 和 CALL 2：只能细化，不能重分类

Python 首先提供：

```text
Rule threat level = High or Low
Rule barrier level = High or Low
```

LLM 不允许改变这个 High/Low class，只能在该类别内部给出更细的 1–5 分数。

输出 schema 根据 authoritative class 动态限制：

```text
Low class  → score 只能是 1、2、3
High class → score 只能是 3、4、5
```

`3` 是边界值，但 `level` 必须与 Python 给出的 class 完全一致。

#### Prompt 结构

```text
TASK
DEFINITION
AUTHORITATIVE RULE-BASED CLASSIFICATION
  rule score
  rule level
  明确说明 High/Low 不可改变
1–5 NUANCE RUBRIC
INDIVIDUAL VARIABLES
OUTPUT
```

所以 V2 的 CALL 1/2 已经不再负责“发现 High/Low”，而是负责：

```text
解释固定分类 + 在分类内部提供 nuance
```

### 4.3.3 CALL 3：输出接种概率

CALL 3 的输入包括：

```text
1. authoritative HBM2 pattern
2. rule threat/barrier scores
3. optional pattern prior
4. CALL 1 Threat nuance
5. CALL 2 Barrier nuance
6. additional context
7. retrieved reflection memories
```

输出：

```json
{
  "probability_yes": 68,
  "decision": "YES",
  "reason": "High threat and stable access outweigh the remaining cost concern."
}
```

内部要求：

```text
probability_yes >= 50 → decision 必须为 YES
probability_yes <  50 → decision 必须为 NO
```

后续最终分类时，可以使用 Python 选择的 calibrated threshold，而不是固定 50。

### 4.3.4 错误样本的 Reflection Call

如果训练样本在 50% 内部 cutoff 下预测错误，Reflection Call 会获得：

```text
TRAINING ERROR
  true outcome
  predicted probability
  predicted label
  original reason

AUTHORITATIVE HBM2 PATTERN
  pattern ID
  threat/barrier rule scores

CALL 1 RESULT
CALL 2 RESULT
ADDITIONAL CONTEXT
MEMORY USED BEFORE THIS ERROR
```

Reflection system prompt 要求模型不能简单说“预测和标签不同”，而要识别：

- 哪个因素被忽略；
- 哪个因素被过度加权；
- 对未来相似样本可以怎样修正。

输出：

```json
{
  "error_cause": "The model over-weighted financial barriers.",
  "missed_or_overweighted_signal": "Recent repeated healthcare contact was under-weighted.",
  "correction_rule": "For similar P1 respondents, increase probability when recent healthcare contact is strong despite moderate cost concern.",
  "applicable_pattern": "P1"
}
```

### 4.3.5 V2 的 Memory 如何更新

V2 采用 online sequential reflection：

```text
TRAIN CASE
  → CALL 1/2
  → retrieve existing reflection rules
  → CALL 3 probability
  → if wrong at 50%, REFLECTION CALL
  → immediately store new correction rule
```

因此早期训练错误生成的 reflection rule，可以影响后面的训练样本。

训练结束后：

- memory 冻结；
- test label 不能更新 memory。

### 4.3.6 Threshold Calibration

Python 在一系列候选 threshold 中搜索最优值，可以优化：

- balanced accuracy；
- F1；
- accuracy。

这个过程不调用 LLM。

### 4.3.7 V2 想验证什么

V2 的研究问题是：

> 将错误样本直接存储，是否不如让 LLM 将错误总结成结构化 correction rule？

同时，概率输出使模型可以计算 AUC、average precision 和 log loss，而不仅是 accuracy/F1。

---

## 4.4 V3 — Pattern Anchor + Bounded Residual + Dual Memory

**代码：**`scripts/30_hbm2_pattern_anchor_dual_memory.py`  
**Version string：**`hbm2_openai_colab_v4_pattern_anchor_dual_memory`

### 4.4.1 相比 V2 的主要变化

V3 将预测拆为：

```text
Final Probability
  = HBM2 Pattern Base Rate
  + LLM Residual Adjustment
```

这意味着 pattern 不再只是定性倾向，而是提供一个数值化概率起点。

V3 还首次采用真正独立的：

```text
memory-build split
calibration split
test split
```

### 4.4.2 第一步：三分割数据

默认比例为：

```text
memory-build = 50%
calibration  = 25%
test         = 25%
```

分割时可以同时保持 vaccination class 与 HBM2 pattern 的组成。

只有 memory-build split 可以用于：

- 计算 pattern base rate；
- 建立 prototype memory；
- 建立 reflection memory。

### 4.4.3 第二步：计算 Pattern Base Rate

Python 根据 memory-build split，为 P0–P3 计算经过 smoothing 的接种率，例如：

```text
P0 High Threat / Low Barrier  → 62.6%
P1 High Threat / High Barrier → 48.3%
P2 Low Threat / Low Barrier   → 38.8%
P3 Low Threat / High Barrier  → 31.2%
```

对于 memory-build 样本，程序使用 leave-one-out anchor，避免当前样本自己的真实标签直接进入自己的 base rate。

对于 calibration/test 样本，使用完整 memory-build split 计算出的 pattern rate。

### 4.4.4 CALL 1 和 CALL 2

CALL 1/2 延续 V2 的 authoritative design：

- Python 给出的 High/Low class 不能改变；
- LLM 只输出 1–5 nuance score 和一句解释；
- 不允许预测 vaccination。

所有选中样本的 CALL 1/2 可以提前并发计算。

### 4.4.5 Dual Memory 的两个组成部分

#### Prototype Memory

Prototype 来自：

- memory-build split；
- 预测正确；
- 预测置信度较高；
- 能代表该 pattern 的典型情况。

Prototype 不需要额外的 Reflection Call。它告诉最终 Decision Call：

```text
这种样本是该 pattern 中的正常、典型情况
```

#### Reflection Memory

Reflection 来自：

- memory-build split；
- 预测错误；
- 错误置信度较高；
- 值得总结为窄范围 exception rule。

它需要额外的 Reflection Call。

#### Retrieval 结构

例如默认可以检索：

```text
prototype_k = 2
reflection_k = 1
```

Prompt 会明确说明：

- prototypes 是典型证据；
- reflections 是异常规则；
- reflection 不能被当成多数规律；
- 只有高度相似时才能覆盖 pattern tendency。

### 4.4.6 CALL 3：只能输出有限 Adjustment

CALL 3 prompt 包含：

```text
AUTHORITATIVE HBM2 CLASSIFICATION
  pattern
  threat/barrier rule score and level
  theoretical tendency

TRAINING-ONLY PROBABILITY ANCHOR
  pattern base rate
  number of supporting memory-build cases

CALL 1 — THREAT EXPLANATION/NUANCE
CALL 2 — BARRIER EXPLANATION/NUANCE
ADDITIONAL CONTEXT
BALANCED MEMORY EVIDENCE
OUTPUT RULES
```

System prompt 明确规定：

- pattern base rate 是 authoritative starting probability；
- LLM 不能自由创造一个新的绝对概率；
- 只允许输出一个有限 adjustment。

输出：

```json
{
  "adjustment": 10,
  "reason": "Recent healthcare engagement supports a modest upward adjustment within P0."
}
```

允许范围：

```text
-20 到 +20 percentage points
```

最终概率由 Python 计算：

```text
probability_yes
  = clip(pattern_base_rate + adjustment, 0, 100)
```

所以 V3 的 LLM 不再是完整 classifier，而是：

```text
pattern-conditioned residual model
```

### 4.4.7 Reflection Call 的输入输出

如果一个高置信 memory-build prediction 出错，Reflection Call 会看到：

```text
actual outcome
pattern base rate
CALL 3 adjustment
final predicted probability
predicted label
original reason
HBM2 pattern
CALL 1 result
CALL 2 result
additional context
memory used before the error
```

输出：

```json
{
  "error_cause": "The model treated the pattern tendency as too deterministic.",
  "missed_or_overweighted_signal": "Recent healthcare contact was under-weighted.",
  "correction_rule": "Apply an upward exception only to genuinely similar P3 respondents with strong healthcare engagement.",
  "applicable_pattern": "P3"
}
```

### 4.4.8 完整的 Memory/Calibration/Test 顺序

```text
MEMORY-BUILD
  → precompute CALL 1/2
  → sequential CALL 3
  → save high-confidence correct prototypes
  → reflect on selected high-confidence errors
  → freeze dual memory

CALIBRATION
  → CALL 3 with frozen dual memory
  → select threshold
  → no memory update

TEST
  → CALL 3 with the same frozen memory
  → use frozen threshold
  → no memory update
```

### 4.4.9 V3 的核心消融

V3 可以在同一个 test set 上比较：

```text
Pattern-only probability
vs.
Pattern base probability + LLM residual adjustment
```

因此它第一次能够直接回答：

> LLM 是否在 HBM2 pattern 接种率之外提供了额外预测信息？

---

## 4.5 V4 — HBM5：允许使用其他疫苗历史

**代码：**`scripts/40_hbm5_prior_vaccine_reflective_memory.py`  
**Version string：**`hbm5_openai_v1_offline_reflective_memory`

### 4.5.1 从 HBM2 到 HBM5 的变化

V4 不再让 LLM 分别评估 Threat 和 Barrier。Python 直接从原始 NHIS 变量构造五个可复现的 observable proxies。

| Proxy | 理论功能 | 主要可观测证据 |
|---|---|---|
| `observed_threat_proxy` | 对疾病风险与后果严重性的可观测表征 | 年龄、自评健康、慢性病负担、功能和免疫脆弱性 |
| `vaccine_acceptance_benefit_proxy` | 已表现出的预防性疫苗接受行为 | COVID-19、肺炎、带状疱疹、Shingrix、甲肝疫苗历史 |
| `structural_barrier_proxy` | 从意愿转化为实际接种的结构阻碍 | 保险不稳定、费用导致的未满足医疗、账单压力、交通、沟通困难 |
| `healthcare_cue_proxy` | 可能触发预防行为的医疗接触机会 | 最近医生或 wellness contact、retail/virtual care、urgent/emergency/hospital contact |
| `navigation_self_efficacy_proxy` | 使用和导航医疗与信息系统的能力 | usual source of care、医疗场所、网络访问、数字健康工具、沟通能力 |

这些 proxy 仍然不是直接 HBM psychometric measurements。

### 4.5.2 第一步：Python 确定性构建五个 Proxy

每个 proxy 都由 NHIS 变量 recode、加权和组合得到，不调用 LLM。

随后构建三个 meta-dimensions：

```text
Motivation
  = mean(Observed Threat,
         Vaccine Acceptance/Benefit Proxy)

Capability
  = mean(5 - Structural Barriers,
         Navigation Self-Efficacy)

Activation
  = Healthcare Cues
```

### 4.5.3 第二步：构建八类 Pattern

对三个 meta-dimension 分别进行 High/Low 划分，得到：

```text
High Motivation / High Capability / Strong Cue
High Motivation / High Capability / Weak Cue
High Motivation / Low Capability  / Strong Cue
High Motivation / Low Capability  / Weak Cue
Low Motivation  / High Capability / Strong Cue
Low Motivation  / High Capability / Weak Cue
Low Motivation  / Low Capability  / Strong Cue
Low Motivation  / Low Capability  / Weak Cue
```

阈值可以：

- 从 memory-build split 的分布中拟合；
- 使用固定阈值。

Python 再根据 memory-build split 计算每个 pattern 的平滑接种率。

### 4.5.4 数据分割

默认比例：

```text
memory-build = 40%
calibration  = 20%
test         = 40%
```

Memory-build split 用于：

- 拟合 High/Low thresholds；
- 拟合 HBM8 pattern base rates；
- 拟合 similarity space；
- 建立 reflection memory。

### 4.5.5 常规 Decision Call：每个样本只调用一次 LLM

V4 不存在五个 proxy 对应五次 LLM 调用。五个 proxy 已由 Python 完成。

每个常规样本只进行一次综合 Decision Call。

#### 输入 Payload

Prompt 包含：

```text
pattern prior
  HBM8 pattern
  training-only base probability
  number of supporting memory-build cases

respondent observed profile
  five proxy scores
  three meta-dimensions
  underlying observed evidence
  selected background variables

retrieved reflective memories
  similarity
  source pattern
  correction direction
  residual factor
  supporting variables
  correction rule
  applicability conditions
  non-generalization warning
```

#### System Prompt 的理论路径

System prompt 告诉模型：

1. Threat 和 prior-vaccine acceptance/benefit 共同形成 Motivation；
2. 低 Barrier 和高 Self-Efficacy 共同形成 Capability；
3. Healthcare contact 形成 Activation；
4. HBM8 pattern base rate 是 prior，不是 deterministic label；
5. LLM 只需要寻找 pattern 内的 residual observed factors；
6. memory 是 exception rule，只有符合 applicability conditions 时才能使用；
7. true label 不会提供给 Decision Call。

#### 输出 Schema

```json
{
  "residual_adjustment": 8,
  "probability_yes": 74.2,
  "deviation_direction": "higher_than_pattern",
  "dominant_hbm_factors": [
    "high vaccine acceptance",
    "strong healthcare cues"
  ],
  "residual_observed_factors": [
    "recent wellness contact"
  ],
  "reason": "The observed profile supports a modest increase above the pattern base rate."
}
```

Adjustment 范围：

```text
-30 到 +30 percentage points
```

Python 会验证：

```text
probability_yes
≈ pattern_base_probability + residual_adjustment
```

### 4.5.6 Offline Reflective Memory 如何产生

V4 不再在训练过程中边预测边更新 memory，而是采用 offline memory construction。

#### Stage A：No-memory Decision Pass

Memory-build split 中的每个样本先执行一次不带 memory 的 Decision Call。

这些样本的预测互不影响。

#### Stage B：选择 Reflection Candidates

Python 从 no-memory predictions 中选择：

- 高置信 underprediction；
- 高置信 overprediction；
- 不同 HBM8 pattern/error bucket 中有代表性的错误。

同时限制：

- 最大候选数量；
- 每个 bucket 的最大 Reflection Call 数量。

#### Stage C：Reflection Call

Reflection prompt 输入：

```text
required correction direction
required error type
actual training outcome
pattern base probability
initial no-memory decision
respondent observed profile
```

输出：

```json
{
  "correction_direction": "increase",
  "error_type": "underprediction",
  "residual_factor": "recent repeated preventive healthcare contact",
  "supporting_variables": ["WELLNESS_A", "LASTDR_A"],
  "correction_rule": "Increase probability for genuinely similar respondents with strong recent preventive contact.",
  "applicability_conditions": ["same pattern", "recent preventive contact"],
  "non_generalization_warning": "Do not apply when contact is absent or only emergency-driven.",
  "reflection_confidence": 0.82,
  "estimated_memory_value": 0.76
}
```

#### Stage D：筛选并冻结 Memory

Python 根据以下条件过滤 reflection：

- reflection confidence；
- estimated memory value；
- novelty；
- pattern/error bucket 上限；
- memory 总量上限。

只有最终保留的高价值规则才可以进入 calibration/test prompt。

### 4.5.7 Calibration 与 Test

```text
CALIBRATION
  → retrieve frozen reflection memory
  → one Decision Call per respondent
  → select LLM threshold
  → separately select pattern-only threshold

TEST
  → retrieve the same frozen memory
  → one Decision Call per respondent
  → apply frozen threshold
```

代码也支持额外运行 test without memory，但已报告的 V4 运行没有开启该消融。

### 4.5.8 V4 中 LLM 的真正作用

在 LLM 调用之前，Python 已经提供：

- 五个 proxy；
- 三个 meta-dimension；
- 八类 pattern；
- pattern base probability。

所以 LLM 的唯一预测功能是：

```text
识别当前样本为什么会高于或低于所在 pattern 的平均接种率
```

这使得 LLM 的增量价值可以直接与 pattern-only 比较。

---

## 4.6 V5 — HBM5-NV：完全排除其他疫苗历史

**代码：**`scripts/50_hbm5_no_prior_vaccine_reflective_memory.py`  
**Version string：**`flare_vax_no_prior_vax_cot_memory_v1`

### 4.6.1 方法目的

V5 移除最强但可能具有 shortcut 性质的特征组：其他疫苗接种历史。

以下变量被明确禁止进入：

- proxy scoring；
- prompt；
- similarity vector；
- memory retrieval；
- reflection content。

```text
SHTCVD191_A
SHTCVD19NM2_A
SHTPNUEV_A
SHTPNEUNB_A
SHTSHINGL1_A
SHINGRIX3_A
SHTHEPA_A
SHTFLUM_A
SHTFLUY_A
```

### 4.6.2 用 Preventive Engagement 替换 Vaccine Acceptance

V4 的 `vaccine_acceptance_benefit_proxy` 被替换为：

```text
preventive_engagement_proxy
```

它来自非疫苗行为：

- wellness visit；
- health-information seeking；
- 在线联系医生；
- 在线查看检查结果；
- 其他健康管理参与行为。

它不能被描述为直接的 perceived-benefit scale。

### 4.6.3 Python 构建的完整表征

```text
observed_threat_proxy
preventive_engagement_proxy
structural_barrier_proxy
healthcare_cue_proxy
navigation_self_efficacy_proxy
```

然后：

```text
Motivation
  = mean(Threat, Preventive Engagement)

Capability
  = mean(5 - Barriers, Navigation Self-Efficacy)

Activation
  = Healthcare Cues
```

再形成与 V4 相同的八类 pattern。

### 4.6.4 Decision Prompt 的输入结构

Prompt 使用紧凑 JSON payload，包含：

```text
pattern_prior
  pattern
  general tendency
  base probability
  memory-build pattern count

respondent_profile
  five proxies
  three meta-dimensions
  HBM8 pattern
  raw observed evidence
  allowed background variables

retrieved_reflective_memories
  similarity
  correction direction
  failure stage
  correction rule
  corrected reasoning path
  applicability conditions
  contradiction conditions
  retrieval keys
  memory value
```

System prompt 明确禁止模型推断或提及任何 COVID、肺炎、带状疱疹、Shingrix、甲肝或其他疫苗历史。

### 4.6.5 Decision Call 的四阶段结构化输出

V5 要求 LLM 输出简短、可观察、可记录的 reasoning trace，而不是不受限制的 hidden chain-of-thought。

输出：

```json
{
  "construct_interpretation": {
    "threat": "high",
    "preventive_engagement": "moderate",
    "barriers": "low",
    "self_efficacy": "high",
    "cues": "moderate"
  },
  "reasoning_trace": {
    "stage_1_evidence_synthesis": "Summarize the observed construct evidence.",
    "stage_2_pattern_interpretation": "Explain the HBM8 pattern tendency.",
    "stage_3_residual_context": "Identify evidence not captured by the pattern average.",
    "stage_4_decision_mapping": "Map the anchor and residual evidence to probability."
  },
  "residual_adjustment": -10,
  "probability_yes": 48.0,
  "deviation_direction": "lower_than_pattern",
  "dominant_observed_factors": [
    "weak preventive engagement",
    "limited recent healthcare cues"
  ],
  "memory_application": [],
  "confidence": 0.73
}
```

Adjustment 范围：

```text
-25 到 +25 percentage points
```

并且必须满足：

```text
probability_yes
  = clip(pattern base probability + residual adjustment, 0, 100)
```

### 4.6.6 Stage-Specific Reflection Call

对于被选中的高置信错误，Reflection Call 需要判断失败发生在哪个阶段：

```text
construct_inference
pattern_interpretation
context_integration
decision_mapping
memory_misapplication
```

输出：

```json
{
  "correction_direction": "increase",
  "error_type": "underprediction",
  "failure_stage": "context_integration",
  "incorrect_assumption": "The model treated weak activation as decisive despite strong preventive engagement.",
  "corrected_reasoning_path": {
    "evidence_reinterpretation": "Preventive engagement provides independent positive evidence.",
    "pattern_exception": "This profile can exceed the pattern average despite weaker cues.",
    "decision_correction": "Apply a moderate upward residual adjustment."
  },
  "supporting_variables": ["WELLNESS_A", "HITCOMM_A"],
  "correction_rule": "Increase predictions for similar profiles when sustained preventive engagement offsets weak cues.",
  "applicability_conditions": ["high preventive engagement", "low structural barriers"],
  "contradiction_conditions": ["no recent health-management behavior"],
  "retrieval_keys": ["preventive engagement", "weak cue exception"],
  "reflection_confidence": 0.81,
  "estimated_memory_value": 0.77
}
```

这类 memory 不只保存“应该提高还是降低概率”，还保存：

- 哪个推理阶段失败；
- 正确的 reasoning path；
- 适用条件；
- 冲突条件；
- 检索关键词。

### 4.6.7 5,000 样本实验的完整调用顺序

```text
MEMORY-BUILD SPLIT：2,000
  → 2,000 次 Decision Call without memory
  → 选择高置信错误
  → 对候选执行 Reflection Calls
  → 筛选并保留 6 条 final memory

CALIBRATION SPLIT：1,000
  → 1,000 次 Decision Call with frozen memory
  → 选择 LLM threshold = 52
  → 选择 pattern-only threshold = 42

TEST SPLIT：2,000
  → 2,000 次 Decision Call with frozen memory
  → 2,000 次 Decision Call without memory
  → pattern-only 不调用 LLM
```

因此 V5 在完全相同的 test cases 上比较：

1. HBM8 pattern only；
2. LLM without memory；
3. LLM with frozen reflective memory。

---

# 5. 六个版本的详细横向对比

## 5.1 行为表征与 LLM 责任

| 维度 | V1 | V1-Async | V2 | V3 | V4 | V5 |
|---|---|---|---|---|---|---|
| 主要构念 | Threat、Barrier | 相同 | Threat、Barrier | Threat、Barrier | 5 个 proxy | 5 个 proxy |
| 构念来源 | LLM 1–5 评分 + rule score | 相同 | Python authoritative class + LLM nuance | Python authoritative class + LLM nuance | Python 确定性计算 | Python 确定性计算 |
| Pattern 数量 | 4 | 4 | 4 | 4 | 8 | 8 |
| Pattern 功能 | 定性方向 | 定性方向 | authoritative class + directional prior | 数值化概率 anchor | 数值化概率 anchor | 数值化概率 anchor |
| LLM 的最终角色 | 完整二元分类器 | 完整二元分类器 | 自由概率估计器 | 有界 residual adjuster | 综合 residual adjuster | 四阶段结构化 residual adjuster |
| 最终 LLM 输出 | YES/NO | YES/NO | probability + YES/NO + reason | adjustment + reason | adjustment + probability + factors | structured trace + adjustment + probability |
| 可解释性 | 较低 | 较低 | 中等 | 较高 | 较高 | 最高 |

## 5.2 Final Decision Prompt 的结构差异

| 版本 | Final Decision Prompt 包含什么 | 最关键的 System Constraint |
|---|---|---|
| V1 | Threat result、Barrier result、HBM2 pattern、optional prior、context、相似原始错误 | pattern 只是方向；只输出 YES/NO |
| V1-Async | 与 V1 完全相同 | 与 V1 相同 |
| V2 | authoritative pattern、CALL1/2 nuance、context、reflection rules | 输出 probability，并与内部 50% 决策一致 |
| V3 | authoritative pattern、numeric base rate、CALL1/2 nuance、context、prototype、reflection | 只能输出 `[-20,+20]` adjustment |
| V4 | 5 proxies、3 dimensions、HBM8 pattern/base rate、原始证据、reflection | 只寻找 residual factors；adjustment `[-30,+30]` |
| V5 | 无其他疫苗历史的完整 profile、stage-aware memories | 按四阶段推理；adjustment `[-25,+25]`；禁止推断其他疫苗 |

## 5.3 Memory 机制对比

| 版本 | Memory 中保存什么 | 如何生成 | 什么时候更新 | 检索内容 |
|---|---|---|---|---|
| V1 | 错误训练案例 | 程序直接组合，无 reflection call | 每个顺序训练错误后立即更新 | 最相似的 raw errors |
| V1-Async | 错误训练案例 | 与 V1 相同 | 每个 batch 后更新 | pre-batch memory 中的 raw errors |
| V2 | 结构化 correction rule | 错误样本增加一次 Reflection Call | 顺序训练过程中立即更新 | 相似 reflection rules |
| V3 Prototype | 高置信正确案例 | 无额外 LLM Call | memory-build 阶段 | 典型正确案例 |
| V3 Reflection | 高置信错误案例 | 增加 Reflection Call | memory-build 阶段 | 窄范围 exception rules |
| V4 | 高价值 residual rule | no-memory pass → candidate selection → reflection → filtering | 一次性离线建立后冻结 | 相似 exception rules |
| V5 | 定位失败阶段的 correction rule | 与 V4 相同，但增加 applicability/contradiction/retrieval keys | 一次性建立后冻结 | stage-aware exception rules |

## 5.4 数据分割、Calibration 与 Leakage Control

| 版本 | 数据结构 | Threshold 来源 | Test 是否更新 Memory | 主要 Leakage Control |
|---|---|---|---|---|
| V1 | train/test | 固定二元输出 | 否 | test labels 不进入 memory |
| V1-Async | train/test | 固定二元输出 | 否 | test labels 不进入 memory |
| V2 | train/test | training predictions 上选择 | 否 | frozen training reflection memory |
| V3 | memory/calibration/test | 独立 calibration split | 否 | pattern rate、memory、threshold 在 test 前全部冻结 |
| V4 | memory/calibration/test | 独立 calibration split | 否 | offline memory + frozen pattern/LLM thresholds |
| V5 | memory/calibration/test | 独立 calibration split | 否 | 显式排除变量 + 同 test memory ablation |

## 5.5 LLM 调用成本对比

设：

```text
N  = 总样本数
R  = 实际执行的 Reflection Calls 数量
Nt = test 样本数
```

| 版本 | 近似 LLM 调用数 |
|---|---:|
| V1 | `3N` |
| V1-Async | `3N`，但并发执行 |
| V2 | `3N + R` |
| V3 | `3N + R` |
| V4 | `N + R`；如果额外跑 no-memory test，再加 `Nt` |
| V5 当前实验 | `N + Nt + R`，因为 test 同时运行 with memory 与 without memory |

需要特别强调：

> V4 和 V5 的五个 proxy 不对应五次 LLM 调用，它们全部由 Python 计算。

## 5.6 LLM 自由度如何逐步下降

| 模块 | V1 | V2 | V3 | V4/V5 |
|---|---|---|---|---|
| 构念 High/Low | LLM 可以独立评分 | Python class authoritative | Python class authoritative | 完全由 Python 计算 |
| Pattern | 定性输入 | authoritative classification | empirical probability anchor | empirical probability anchor |
| 绝对概率 | 不输出 | LLM 自由输出 | anchor + 有界 adjustment | anchor + 有界 adjustment |
| 错误学习 | raw examples | LLM correction rules | prototype + exception rule | 离线筛选 exception rules |
| LLM 整体自由度 | 最高 | 降低 | 明显降低 | 只保留 narrow residual role |

这一演进链的核心方法问题是：

```text
LLM 应该做完整的人类决策预测，
还是应该让行为理论决定主要结构，
让 LLM 只负责 pattern 内的个体偏离？
```

---

# 6. 代码与结果的对应关系

| 版本 | 脚本 | 样本 / 测试集 | 结果文件 | 主要结果 |
|---|---|---:|---|---|
| V1 | `scripts/10_hbm2_three_call.py` | 20 / 6 | `results/summaries/hbm2_v1_smoke_test.json` | Acc 0.6667；F1 0.5000；仅 smoke test |
| V1-Async | `scripts/11_hbm2_three_call_async.py` | 1000 / 300 | `results/raw_logs/hbm2_async_1000.txt` | Acc 0.6100；F1 0.5551 |
| V2 | `scripts/20_hbm2_reflection_calibration.py` | 1000 / 300 | `results/raw_logs/hbm2_reflection_1000.txt` | Acc 0.5133；AUC 0.5409；F1 0.5756 |
| V3 | `scripts/30_hbm2_pattern_anchor_dual_memory.py` | 1000 / 250 | `results/raw_logs/hbm2_dual_memory_1000.txt` | LLM Acc 0.6240/AUC 0.6633；pattern-only Acc 0.6160/AUC 0.6512 |
| V4 | `scripts/40_hbm5_prior_vaccine_reflective_memory.py` | 5000 / 2000 | `results/raw_logs/hbm5_prior_vaccine_5000.txt` | pattern-only Acc 0.7260/AUC 0.7658；LLM-memory Acc 0.7195/AUC 0.7332 |
| V5 | `scripts/50_hbm5_no_prior_vaccine_reflective_memory.py` | 5000 / 2000 | `results/summaries/hbm5_no_prior_vaccine_5000.json` | pattern-only Acc 0.6255/AUC 0.6812；LLM-no-memory Acc 0.6135/AUC 0.6584；LLM-memory Acc 0.5705/AUC 0.5808 |

机器可读索引：

```text
results/run_index.csv
results/reported_llm_hbm_metrics.csv
```

---

# 7. 最终结果对比

## 7.1 不允许使用其他疫苗接种历史

| 方法 | Accuracy | Balanced Accuracy | ROC-AUC | F1 |
|---|---:|---:|---:|---:|
| HBM5-NV Pattern Only | 0.6255 | 0.6326 | 0.6812 | 0.6597 |
| HBM5-NV LLM Without Memory | 0.6135 | 0.6162 | 0.6584 | 0.6201 |
| HBM5-NV LLM With Reflective Memory | 0.5705 | 0.5653 | 0.5808 | 0.5077 |
| Logistic Regression，67 variables | 0.6766 | — | 0.7433 | 0.6640 |
| Random Forest，67 variables | 0.6829 | — | 0.7507 | 0.6660 |
| Gradient Boosting，67 variables | 0.6867 | — | 0.7553 | 0.6691 |
| XGBoost，67 variables | **0.6871** | — | **0.7570** | **0.6697** |

当前结果说明：

- 八类 deterministic pattern 优于两个 LLM 版本；
- LLM without memory 优于 LLM with memory；
- 最终仅保留的 6 条 memory rule 没有在测试集上稳定泛化；
- 传统 ML 仍然是最强预测 benchmark。

## 7.2 允许使用其他疫苗接种历史

| 方法 | Accuracy | Balanced Accuracy | ROC-AUC | F1 |
|---|---:|---:|---:|---:|
| HBM5 Pattern Only | 0.7260 | 0.7266 | 0.7658 | 0.7169 |
| HBM5 LLM With Reflective Memory | 0.7195 | 0.7177 | 0.7332 | 0.6976 |
| Logistic Regression，75 variables | 0.7604 | — | 0.8398 | 0.7484 |
| Random Forest，75 variables | 0.7617 | — | 0.8377 | 0.7471 |
| Gradient Boosting，75 variables | 0.7630 | — | 0.8444 | 0.7506 |
| XGBoost，75 variables | **0.7647** | — | **0.8452** | 0.7519 |
| SVM，75 variables | 0.7630 | — | 0.8356 | **0.7552** |

当前结果说明：

- 其他疫苗历史包含非常强的 flu-vaccination predictive signal；
- HBM5 pattern 本身已经吸收了大量这一信号；
- residual LLM layer 没有超过 pattern-only；
- ML 的整体 discrimination 仍然最高。

---

# 8. 当前实验能够与不能够说明什么

## 当前结果支持

1. 确定性的理论层可以形成接种率排序明显的行为 pattern。
2. 其他疫苗历史是流感疫苗预测中的强特征组。
3. Pattern-only 可以优于 LLM residual layer。
4. 当 reflection rules 稀疏、噪声较大或泛化范围过宽时，memory 可能降低效果。
5. 将 LLM 限制为 residual role 后，可以直接测试其增量价值。

## 当前结果尚不能支持

1. 不能因为后一个版本更复杂，就认为它一定更好。
2. HBM2 的开发版本使用了不同的 sample size、class sampling 和 test size，不能将指标变化直接解释为版本升级的因果效果。
3. 当前结果尚未证明 LLM reflection memory 能稳定学习可迁移的 behavioral rules。
4. 这些 observable proxies 不能被当成已经验证的 HBM 心理量表。
5. ML baseline 目前保存的是汇总结果，所有方法并没有完整共享同一组保存下来的 split assignments。

最终 paper-quality comparison 应该：

- 固定一个共同 split；
- 所有模型使用相同 respondents；
- 保存预测概率；
- 使用 bootstrap confidence intervals；
- 进行 paired significance tests。

---

# 9. 运行方式

## 9.1 安装环境

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

设置 API key：

```bash
export OPENAI_API_KEY="..."
```

Windows PowerShell：

```powershell
$env:OPENAI_API_KEY="..."
```

不要将 API key 上传到 GitHub。

## 9.2 数据位置

将 NHIS 2024 Sample Adult CSV 放置为：

```text
data/adult24.csv
```

仓库不重新分发原始 NHIS 数据。

## 9.3 构建 HBM2 Clean Data

```bash
python scripts/00_prepare_hbm2_data.py \
  --input_csv data/adult24.csv \
  --output_csv data/nhis2024_hbm2_clean.csv
```

## 9.4 不调用 API，只检查 Prompt

V1：

```bash
python scripts/10_hbm2_three_call.py \
  --data_path data/nhis2024_hbm2_clean.csv \
  --output_dir results/runtime/hbm2_v1 \
  --sample_size 20 \
  --dry_run
```

V3：

```bash
python scripts/30_hbm2_pattern_anchor_dual_memory.py \
  --data_path data/nhis2024_hbm2_clean.csv \
  --output_dir results/runtime/hbm2_v3 \
  --sample_size 100 \
  --dry_run
```

V5：

```bash
python scripts/50_hbm5_no_prior_vaccine_reflective_memory.py \
  --input-csv data/adult24.csv \
  --output-dir results/runtime/hbm5_no_prior \
  --sample-size 100 \
  --dry-run
```

`--dry_run` 或 `--dry-run` 会打印代表性 prompt 并验证数据流程，不发送 OpenAI API 请求。

## 9.5 运行 ML Baselines

```bash
python scripts/01_ml_baselines.py \
  --input_csv data/nhis2024_hbm2_clean.csv \
  --output_dir results/runtime/ml_baselines
```

更多命令见：

```text
configs/baseline_feature_manifest.example.json
docs/replication_commands.md
```

---

# 10. 整体方法链总结

```text
V1
可观测变量
→ LLM 评估 Threat
→ LLM 评估 Barrier
→ LLM 直接输出 YES/NO
→ 原始错误案例 memory

V2
Python 固定 Threat/Barrier High-Low
→ LLM 做 within-class nuance
→ LLM 输出 probability
→ LLM 将错误总结为 correction rule

V3
Python 固定 HBM2 pattern
→ pattern training base rate
→ LLM 只输出有界 residual adjustment
→ prototype + reflection dual memory

V4
Python 构建 5 proxies
→ 3 meta-dimensions
→ 8 patterns + pattern base rate
→ 一次 LLM residual decision
→ offline reflective memory

V5
保留 V4 的结构
→ 完全排除其他疫苗 predictor
→ 用 preventive engagement 替代 vaccine acceptance
→ 四阶段结构化 decision trace
→ 同测试集比较 pattern / no-memory / memory
```

因此，这个仓库最终测试的不是简单的“LLM vs ML”，而是一个更具体的方法问题：

> 是让 LLM 从调查数据端到端推断完整的人类决策，还是先由确定性的行为理论定义主要结构，再让 LLM 只解释和修正个体对 pattern 平均规律的偏离？

---

## Reference

Chen, R., Wang, C., Sun, Y., Zhao, X., and Xu, S. (2025). *From Perceptions to Decisions: Wildfire Evacuation Decision Prediction with Behavioral Theory-informed LLMs*. Proceedings of ACL 2025.
