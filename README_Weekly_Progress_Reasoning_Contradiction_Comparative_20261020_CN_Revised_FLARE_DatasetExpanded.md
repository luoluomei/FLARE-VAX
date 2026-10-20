# 本周进展 — Reasoning Mechanism、Feature Contradiction 与 Comparative Reasoning

**项目：** NHIS 2024 Individual Influenza Vaccination Prediction  
**实验时间：** 2026-10-15 — 2026-10-18；整理与复核：2026-10-20  
**数据：** NHIS 2024，Train20 = 6,414，Test80 = 25,656，62 个输入 features；目标为过去 12 个月是否接种流感疫苗（`SHTFLU12M_A`）。  
**结果来源：** 本周 SOL 实验压缩包 `weekly_experiment_results_20261015_20261018.zip`；上周 `README_Weekly_Progress_Multitask_LocalPredictability_CN_20261013_v4(1).md`；10 月 18 日 meeting recording。

> **本周主线：** 对照实验回答此前的 reasoning improvement 究竟来自什么；深入检查“相似 profile、相反 label”时 observable features 到底能提供多少信息；最后在**没有继续训练 Student 的情况下**，直接让 ASU Qwen3.8 Teacher 比较历史案例，检验 explicit comparative reasoning 是否有额外 predictive value。
>
> **数据复核说明：** Experiment 2 的压缩包内旧 `EVALUATION_REPORT.md` 仍只包含 Smoke Test 的 **8 个完整案例**，与实际 API checkpoint 不同步。本 README **直接重新读取** `success.jsonl`（成功 1,799/1,800）、`teacher_inputs.jsonl` 与独立保存的 `EVAL_ONLY_target_labels.csv`，重新在 **599 个完整 A/B/C 配对案例**上计算下列所有 Experiment 2 数值。不要引用旧 8-case 结果来代表 600-case 实验。

---

# 1. 与上周 Meeting 的核心问题如何对应？

上周汇报后，讨论的关键已经不再是简单增加 reasoning 长度或更换 HBM prompt，而是三个层次的问题。

| 上周老师关注的问题 | 本周对应实验 | 需要回答的研究问题 |
|---|---|---|
| **Q1.** 之前 DSBS 的 improvement 究竟来自重新分配 label/rationale loss，还是分开两个 task？训练 reasoning 和测试时真的生成 reasoning 有什么区别？ | **Experiment A — Balanced Post-thinking & mechanism diagnostics** | 在相同 90%/10% loss 权重下，单序列 `LABEL → RATIONALE` 是否也能取得 DSBS 的效果？是否能发现可信的 reasoning failure pattern？ |
| **Q2.** 当前错误中到底哪些有可学习的 signal？那些相似但接种结果不同的人，究竟在哪些 feature 上 contradictory？ | **Experiment B — Correctability + Feature Contradiction Audit** | 结合 Train20 最近邻的 label distribution、feature-level contrast 和 paired examples，重新理解错误来源，而不是一概归因于 HBM reasoning。 |
| **Q3.** 如果把多个相似但 outcome 相反的历史案例一起交给 LLM，它能否识别有效差异、学会何时证据不足？ | **Experiment C — Teacher Multi-case Comparative Reasoning (Experiment 2)** | 对同一个 **frozen ASU Qwen3.8-27B Teacher** 比较 Target-only / Balanced Few-shot / Explicit Comparative Reasoning，并分析四种 target 来源对应的效果。 |

同时，老师还提出两个后续方向：**uncertainty / additional classification head**，以及使用 LLM 进行 **persona / latent-feature augmentation**。本周将它们作为 ongoing work 单独记录，**不报告尚未产生的预测结果**。

---

# 2. 共同基准：此前我们已经知道什么？

所有 Student SFT 相关实验使用相同 Train20/Test80 split，Student 基础模型为 **Qwen3.5-9B + LoRA**。Teacher rationale 的来源为 **Qwen3.8-27B，HBM-Structured / No Thinking**。本周后面的 Experiment 2 则是直接运行 **ASU API Qwen3.8-27B Teacher**，不是微调 Student，因此两部分不能混为一种训练方法。

| Setting | Student training / inference | Test80 Accuracy | Balanced Acc. | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| Label-only | `profile → label` | 0.7623 | 0.7619 | 0.7503 | 0.8447 | 0.8223 |
| Old Reasoning SFT | `LABEL → RATIONALE`，普通 token CE | 0.7443 | 0.7434 | 0.7292 | 0.8231 | 0.7991 |
| **DSBS / two-task α=0.90** | 分离 label、rationale 两个 task；推理时只输出 label | **0.7670** | **0.7681** | 0.7627 | 0.8462 | 0.8236 |
| **Balanced Post-thinking α=0.90（本周新增）** | 单序列 `LABEL → RATIONALE`，显式 90%/10% loss | **0.7619** | 0.7648 | **0.7656** | **0.8468** | **0.8248** |

**此前的关键不确定性：** DSBS 比 Label-only 高约 **0.46 percentage points**，但单次运行的 gain 小，且原始 Old Reasoning SFT 可能因为 rationale 长度远高于 label，使普通 token-average CE 把绝大多数训练预算花在生成解释上。因此，需要在相同 loss weighting 下补一个 *不拆任务* 的 control，才能进一步讨论 task separation 的作用。

---

# 3. Experiment A — Balanced Post-thinking：隔离 Loss Balancing 与 Task Separation

## 3.1 为什么设计这一实验？

上周分析发现，Old Reasoning SFT 在一个 sequence 内训练：

```text
Respondent profile
      ↓
<LABEL>vaccinated / not_vaccinated</LABEL>
<RATIONALE> ... long HBM rationale ... </RATIONALE>
```

因为 rationale 通常有数百 token，直接做全序列 token-average CE 会导致 label prediction 的相对损失贡献很低。DSBS 则做两个独立 task：

```text
Task 1: profile → label
Task 2: profile → HBM rationale
Shared Qwen3.5-9B / LoRA parameters

L = α × L_label + (1−α) × L_rationale,  α = 0.90
```

但 DSBS 同时改变了**loss 权重、task organization 和 inference protocol**。为了尽量隔离权重的影响，本周完整训练了 **Balanced Post-thinking**：

```text
Training:  profile → LABEL → RATIONALE
Inference: profile → LABEL → RATIONALE
Loss:      0.90 × meanCE(label tokens) + 0.10 × meanCE(rationale tokens)
```

这个方法依然只有**一个 joint autoregressive sequence**，不把 label/rationale 拆成两个 independent tasks；但它使用和 DSBS 一样的显式 90%/10% 权重。为了公平，它从相同 Qwen3.5-9B base 重新训练独立的 LoRA adapter，而不是从已训练 DSBS adapter 继续微调。

另外，Balanced Post 的分类指标使用**实际生成文本中解析到的 label**，连续 score 指标则来自 label-first 位置的两种候选 label likelihood；它们不是完全相同的 prediction representation，也不能把两标签归一化分数理解成严格校准的个人接种概率。

## 3.2 正式 Test80 结果：Balancing 能恢复 Old SFT 的下降，但不足以超过 Label-only

| Pairwise comparison | Accuracy difference | 含义 |
|---|---:|---|
| Balanced Post − Old Reasoning SFT | **+1.76 pp** | 当单序列 reasoning 的 label loss 被明确加权后，原先很大的性能下降基本被恢复。 |
| Balanced Post − Label-only | **−0.04 pp** | 虽然大幅高于原始 reasoning SFT，但**没有**超越 label-only。 |
| DSBS α=0.90 − Balanced Post | **+0.50 pp** | 相同 90%/10% weighting 下，two-task formulation 在 Accuracy 上仍更好。 |
| DSBS α=0.90 − Label-only | **+0.46 pp** | 上周报告的小幅正向差异保持。 |

本周最直接的回答是：

> **Loss balancing 确实是解释原始 Reasoning SFT performance drop 的重要线索；但仅仅修复 loss balancing 不足以产生 DSBS 的那一小部分 Accuracy gain。**

这说明 **task separation / training formulation** 可能提供了独立于 loss balancing 的额外收益。接下来仍需统一 inference protocol 并做 repeated-seed comparison，明确这 0.50 pp 的差异能否稳定复现。

## 3.3 为什么 Balanced Post 的 AUC/F1 不差，但 Accuracy 没提高？

Balanced Post 的 ROC-AUC **0.8468**、F1 **0.7656** 都略高于 DSBS，而 Accuracy 却只有 **0.7619**。原因线索是其 prediction tendency 发生明显偏移：

| Model | Predicted vaccinated rate | Recall (vaccinated) | Specificity (not vaccinated) |
|---|---:|---:|---:|
| True Test80 label distribution | **47.42%** | — | — |
| Label-only | 47.79% | 75.33% | 77.04% |
| Balanced Post | **54.17%** | **82.01%** | **70.94%** |
| DSBS α=0.90 | 50.79% | 78.98% | 74.63% |

Balanced Post 更容易预测 `vaccinated`：这有利于 recall，但同时带来更多 false positives。它的排序指标（AUC）较好，并不代表固定 binary decision protocol 下的 Accuracy 一定更好。后续若比较 score threshold / calibration，必须统一方法、保持 test labels 不参与 threshold tuning。

---

# 4. Experiment B — Error Correctability & Feature Contradiction Audit

## 4.1 回应老师：5,979 个错误是不是都因为当前数据模糊？

老师上周特别追问：如果确实有一部分 case 在相似 historical profiles 下有比较一致的方向，却仍被 student 判错，这部分有没有可以纠正的 signal？如果不是，remaining errors 是否更多来自 observable features 未充分解释的 individual behavior？

本周首先完成了 `alpha090_error_correctability_v1`，然后进行了更加细粒度的 `feature_contradiction_audit_v1`。共同使用：

```text
Test80 target profile
    ↓
62 coded features → categorical one-hot representation
    ↓
在 Train20 = 6,414 中查找 nearest neighbors（主分析 k=20）
    ↓
local label distribution / distance / feature-level differences
    ↓
结合 Test80 true label 做事后 error taxonomy
```

**方法边界：** 检索邻居时只用 Train20 的 labels；Test80 true label 仅用于事后的 `true-label support` 与 error analysis。所谓“能否纠正”的筛选不能在真实未知标签的预测阶段直接使用。

## 4.2 Correctability：从“预测错误”进一步区分局部证据类型

我们对 **DSBS α=0.90 在 Test80 的 5,979 个错误**，重新使用 Train20 的 20-nearest neighbors 进行互斥分桶。核心区分有两个：**邻域是否有明显多数方向（local purity）**，以及**这个多数方向是否支持当前 respondent 的实际 outcome（true-label support）**。

| Category | N | 占 DSBS errors | 具体定义与典型情况 |
|---|---:|---:|---|
| **Intermediate mixed / low-support** | **2,825** | **47.25%** | 其余三类之外的非稀疏错误：邻域通常**有一定多数方向**，但这个方向不一定支持该 respondent 的实际 label。例如，附近 15 人未接种、5 人接种，而该 respondent 实际接种，模型却预测未接种。该组的平均 local purity 为 **74.42%**，平均 true-label support 只有 **29.08%**。它更接近 *local-label atypical / conflicting support*，而不只是简单的五五开。 |
| **Likely ambiguous** | **2,507** | **41.93%** | 位于正常 Train20 覆盖范围内，但 **20NN local purity ≤60%**：邻居标签大约是 10:10、11:9 或 12:8。模型所看到的相似案例本身没有明确多数方向；该组平均 local purity 为 **55.80%**。 |
| **Sparse / weak support** | **620** | **10.37%** | 20NN 平均 cosine distance 超过 Train20 leave-one-out 参考分布的第 90 百分位数（**0.2274**）。其主要问题是**没有足够接近的历史样本**，因此即使某些邻居同意，也不能轻易当作稳定局部规律。该组平均距离为 **0.2583**。 |
| **Potentially correctable** | **27** | **0.45%** | 非 sparse、且 **至少 16/20（80%）邻居支持 respondent 的真实 label**，模型却预测相反方向。也就是已有相对一致的同方向 Train20 evidence，却未体现在最终预测中。这是严格定义下最值得检查的候选集合。 |
| **Total** | **5,979** | **100%** | 四类互斥，覆盖全部 DSBS errors。 |

**这里最关键的区别是前两类。** `Likely ambiguous` 指邻居自己意见分裂，接近 10:10；而 `Intermediate mixed / low-support` 很多时候是**邻居已经有明显多数，但 respondent 恰好处在与多数相反的 outcome**。前者更像局部证据不足，后者则需要检查为什么同一组相似特征可以对应相反的实际行为。

相比之下，`Potentially correctable` 的 27 个案例具有最直接的“当前数据可能已提供足够 signal”特征；但进一步检查发现其中 **26 个仍位于 decision boundary 附近，仅 1 个属于较明确的 non-boundary review**。这说明即使在最严格筛选下，当前更值得研究的仍是 *local evidence 与 model decision 为什么没有对齐*，而不是简单要求模型写更长的 HBM rationale。

分桶是基于本周 `alpha090_error_correctability_v1` 的保守互斥规则；与上周六分类（912/1,629/2,503/282/32/621）的划分方式不同。本周后续的 feature audit 主要针对上述“相似特征、相反行为”继续展开。

## 4.3 Full Feature Contradiction Audit：从“邻域有多少反例”进入“到底哪些 feature 不一致”

### 4.3.1 审计范围与局部对照构造

完整结果来自 Job `65966675`，已覆盖：

| 统计对象 | 本周输出规模 |
|---|---:|
| 全部 Test80 的 neighborhood summaries | **25,656** |
| 逐 target 深入 contrast / neighbor edges 的重点样本 | **7,271** |
| Train20 中相似但 label 相反的最近案例对 | **2,000 对** |
| 每个详细 target 比较的 Train20 邻居 | **20 个** |
| Input feature | **62 个** |

本实验除 `true-label support`、`local purity`、`sparse distance` 外，还保存了：

- 20 个 Train20 近邻在每个 feature 上相对 target 的 raw-code difference。
- vaccinated 与 not-vaccinated 邻居分别在每个 feature 的取值分布及 target match rate。
- 各 feature 的 **Total Variation Distance (TVD)**，以及对应 feature-family contrast。
- k=5/10/20/50 的 neighborhood sensitivity。
- 可以人工检查的 opposite-label matched pair 和 review case pack。

这直接回答了老师要求的 **“look at the feature distribution and see where the contradictions are”**，但 TVD 仍只是描述性差异，并不能告诉我们哪个 feature 因果影响 vaccination。


### 4.3.2 哪些 features 能帮助区分“其他特征相似，但 outcome 不同”的 respondents？

老师上周真正追问的是：**相似 respondents 之间究竟有哪些有用的差异？能否找到某些变量，把 vaccinated 和 not-vaccinated 区分开？** 本周的 audit 提供了两种互补证据：

1. **Local distribution contrast：** 在每个 Test80 target 的 20 个 Train20 邻居中，把 vaccinated / not-vaccinated 两组拆开，计算某一 feature 的取值分布差异（TVD）。TVD 越高，说明**这两组历史邻居在该变量上越不一样**。
2. **Target-level directional match：** 再看当前 target 的某个 feature value，究竟更常出现在 vaccinated neighbors 还是 not-vaccinated neighbors 中。这个方向是否和 target 的实际 outcome 一致，能帮助我们判断：**某个差异对区分当前 individual 是否真的有用，还是只是在两组邻居之间制造了差异。**

先看最难的 `Both wrong` 组（5,406 个 Test80 cases）的 mean TVD：

| Feature | Family | Mean TVD | 对“个体区分能力”的初步判断 |
|---|---|---:|---|
| `AGEP_A` | Demographics | **0.838** | 表面差异最大，但精确年龄是多取值变量；年龄差 1 岁与差几十岁在 one-hot 下都会被当成“不一致”，**高 TVD 不意味着最有用**。 |
| `RATCAT_A` | Socioeconomic | 0.452 | 局部分布差异明显，但类别多，尚不能判断其对单个 target 的预测方向。 |
| `EDUCP_A` | Socioeconomic | 0.417 | 同样具有类别丰富带来的差异；方向是否稳定需单独检查。 |
| `SHTCVD19NM2_A` | Vaccination history | **0.345** | **最值得细看的一项行为相关信号**：正负邻居之间存在明显编码差异，而且 target 与两组邻居的匹配率差距较大。 |
| `REGION` | Demographics | 0.313 | 有地区差异，但单个 case 上的方向较弱；不能把它当作某种心理倾向。 |
| `PHSTAT_A` | Self-reported health | 0.277 | 可能区分某些局部案例，但总体仍需要与疫苗史、医疗接触变量联合判断。 |
| `BMICAT_A` | Self-reported health | 0.271 | 具有一定编码分布差异，单独判断力有限。 |
| `SHTSHINGL1_A` | Vaccination history | 0.247 | 能构成部分 vaccine-history contrast，但并没有稳定地指向每个 target 的真实 flu label。 |
| `SHTPNUEV_A` | Vaccination history | 0.192 | 类似地，可提供历史预防行为线索，但不能单独解决相反 outcome。 |

仅看 TVD 还不足以回答“能不能区分”。因此我又用本周原始 `03_target_feature_label_contrasts.csv`，进一步检查了 **5,084 个 Both-wrong targets**：这些 target 的 20 个邻居中 vaccinated / not-vaccinated **双方都至少有 3 人**，可以进行较有意义的组间匹配率比较。

这里的 *Target match gap* 定义为：

```text
P(target 的 feature 编码与 vaccinated neighbors 相同)
−
P(target 的 feature 编码与 not-vaccinated neighbors 相同)
```

如果更匹配的邻居组与 target 的实际 label 相同，称为“指向真实 outcome”；反之则称为“指向相反 outcome”。重新统计得到：

| Feature | Mean absolute match gap | 更匹配的邻居指向真实 label | 更匹配的邻居指向相反 label | 如何理解 |
|---|---:|---:|---:|---|
| `AGEP_A` | 0.094 | 34.5% | 40.1% | 虽然 TVD 极高，但 target 自身的 exact-age match 对正负标签的分辨很弱；另有 25.4% 两边匹配率相同。 |
| **`SHTCVD19NM2_A`** | **0.272** | **30.8%** | **65.3%** | 两类邻居的匹配差距较大，确实存在明显的**局部方向性信号**；但在这些被模型共同判错的案例中，这个方向**更多与实际 flu outcome 相反**。 |
| `SHTSHINGL1_A` | 0.201 | 46.5% | 46.6% | 两个方向几乎各占一半，难以作为可普遍复用的个人判断规则。 |
| `SHTPNUEV_A` | 0.180 | 37.3% | 42.4% | 略偏向相反方向，且有较多两边匹配率相同的 case。 |
| `PHSTAT_A` | 0.176 | 49.4% | 43.7% | 有一定局部区分倾向，但远远达不到稳定区分。 |
| `REGION` | 0.194 | 50.2% | 45.1% | 仅有很小的方向差别。 |

**这里真正有意思的是 `SHTCVD19NM2_A`。** 它不是没有信号：在很多局部 neighborhood 中，它确实把 vaccinated / not-vaccinated 历史案例区分开了。但在 `Both wrong` 案例中，target 的编码往往反而更像**实际 outcome 相反的历史人群**。换句话说，某个 feature 对历史标签有预测关联，**不代表它能解释所有符合该模式的 individual**。这与我们之前发现的“接种过其他疫苗，但这次没有接种流感疫苗；或反过来”的行为冲突是一致的。

进一步查看 **2,000 对距离最近、但 flu label 相反的 Train20 pairs**，情况更加直观：

| Feature | 在相反标签配对中取值不同 | 两人取值仍然相同 |
|---|---:|---:|
| `AGEP_A` | 89.85% | 10.15% |
| `SHTCVD19NM2_A` | 26.65% | **73.35%** |
| `SHTSHINGL1_A` | 20.40% | **79.60%** |
| `SHTPNUEV_A` | 12.30% | **87.70%** |
| `PHSTAT_A` | 30.20% | 69.80% |

这说明，**其他疫苗接种历史是值得研究的区别性 evidence，但它本身仍然无法把最相似的正负 flu outcomes 完全分开**。在这些特意选出的相反标签最近邻配对中，有 73%–88% 的 pairs 连上述各项 vaccination-history 编码都一样。相反，年龄几乎总是不同，却未必有稳定、可转移的标签方向。

**综合结论：** 我们已经看到两件可以同时成立的事：

- **存在局部 outcome-associated signals：** 尤其是 `SHTCVD19NM2_A` 和部分 preventive-vaccination history 特征，可以在某些 neighborhood 中明显区分两组历史标签；
- **但尚未找到单个稳定的“矛盾解决变量”：** 即使上述行为史一致，仍然存在大量相反的 flu outcomes；健康状态、地区、教育等差异也没有形成跨案例一致的判别规则。

因此后续最值得做的不是让 LLM 再罗列这些高 TVD 变量，而是比较：**当 vaccination history 与 healthcare access、health status 等 signals 指向不同方向时，哪些组合和相对权重能从 Train20 学到可重复的预测规律？** 这也正是下一节 Teacher comparative reasoning 要检验的问题。

*注：以上 target-direction statistics 是针对已判错案例的事后诊断，并不等于这些变量在完整 Test80 上的预测准确率；变量具体编码的行为含义仍按 NHIS codebook 核对。*

### 4.3.3 一个最直观的 contradictory historical pair

`07_nearest_opposite_label_train_pairs.csv` 的最相近配对之一：

| 字段 | Train20 Case A | Train20 Case B |
|---|---|---|
| Respondent ID | `H043019` | `H015648` |
| Flu vaccination label | **1** | **0** |
| `REGION` | `3` | `1` |
| `TRANSPOR_A` | `1` | `2` |
| 其余 60 个 model input features | 相同 | 相同 |

这两名 respondents 在 62 个输入特征中只有 **2 个不同**（cosine distance ≈ 0.0323），但 outcome 相反。此时要求 LLM 为每一个真实 label 强行写出一套完全确定的 HBM explanation，很容易产生 **post-hoc rationalization**。

在没有独立验证前，我们不应该声称是 REGION 或 TRANSPOR_A “导致”了差异；这个例子真正说明的是：**当前 62-feature representation 并非总能唯一确定个体接种行为**。

### 4.3.4 对“可能可修”样本的更严格核查

从全量 `01_all_targets_neighborhood_summary.csv` 重新计算（**DSBS 错误，且非 sparse**）：

| 20-NN true-label support threshold | Candidate N |
|---|---:|
| ≥13/20（65%） | **304** |
| ≥14/20（70%） | **151** |
| ≥16/20（80%） | **27** |

对于 151 个满足 ≥14/20 的候选：

- 只有 **116** 个在 k=10 时仍满足 ≥70%。
- 只有 **27** 个在 k=50 时仍满足 ≥70%。
- 仅 **23** 个在 k=10 和 k=50 **同时**满足 ≥70%。

这进一步说明“20NN 支持真实 label”只是初步 screen，绝非稳定的 theoretical correctability guarantee。这里的 `151 / 27` 与上周 `144 / 32` 或另一分桶的 `27` 存在 threshold、neighbor tie、筛选规则口径差异，不能把它们合并成同一版本的固定数量。

## 4.4 本实验阶段性结论

**(1)** 一部分错误确实有明显 data ambiguity / sparse support；并非所有错误都对应 LLM 推理失败。  
**(2)** 目前“局部较容易学习却被错判”的 strict candidates 很少，且不少仍位于 decision boundary。  
**(3)** `SHTCVD19NM2_A` 等 vaccination-history variables 可以形成明显的局部标签对比，但在很多 Both-wrong cases 中，target 反而更接近真实标签相反的邻居。**有统计关联的变量不必然能区分每一个相似个体**。下一步应让 LLM 同时面对多个历史案例的反例与对照，而不是只为一个已知标签生成解释。  
**(4)** 如果想对 local feature contradictions 给出强结论，需要进一步采用可靠 codebook、stable neighborhood definitions、distance weighting / ML feature attribution，不能只按照单变量 TVD 排序。

---

# 5. Experiment C — Experiment 2：ASU Qwen3.8 Teacher Multi-case Comparative Reasoning

## 5.1 为什么这次先用 Teacher，不直接训练新的 Student？

这正是上周老师关于 **“把几个相似、但行为不一样的人一起给 LLM，让它比较差别”** 的建议。

此前 HBM Teacher rationale 存在一个根本问题：它常常在**先知道训练案例真实 vaccination label** 后，为这个已知 label 生成 explanation。对于相似却 outcome 相反的案例，这种生成过程可能对同样的 features 给出相反方向的权重，变成 *post-hoc reverse explanation*。

因此，本周 **Experiment 2 没有进行 Qwen3.5-9B Student SFT**，而是直接调用 **ASU API `qwen38-27b` Teacher（no thinking，temperature=0）**，对于每个目标 respondent：

```text
Target profile（62 features；真实流感疫苗 label 对 teacher 隐藏）
    ↓
在相同 Teacher 上比较三种提示条件 A / B / C
    ↓
直接看 Teacher 的 target prediction
    ↓
检验 historical examples 和 explicit comparison 是否提供增量信息
    ↓
如果 Teacher 都无法稳定学到有用的差异，则不急着把这一类 rationale 大量蒸馏给 Student
```

**这样做的意义是验证 reasoning signal 的“来源端”是否有效。** 若 Teacher 仅是在已知答案以后生成一套看起来合理的解释，那么直接拿来监督 Student 很难识别 causal learning value。现在遮蔽 target label、使用相同 Teacher 和相同 demonstrations 对照，能更加直接检验 comparative reasoning 在**预测未知 case**时是否有价值。

需要注意：Teacher 是 *frozen/API inference*，而 76%–77% 的 Student 为 *训练过的 Qwen3.5-9B*。两者绝对 Accuracy **不能**直接归因于架构谁强谁弱。Experiment 2 的因果对照主要是 **B vs C**，而不是 Teacher vs Student。

## 5.2 三种 Teacher 条件：区分“历史案例有用”与“显式比较有用”

三个条件使用**同一个 frozen ASU API Qwen3.8-27B Teacher**，不进行 Student fine-tuning。每个 target 都必须在**不知道自身真实 flu label** 的情况下预测 `vaccinated=1` 或 `not_vaccinated=0`。差别仅在于 Teacher 能看到哪些历史案例、是否被要求进行比较。

| Condition | Teacher 实际看到的输入 | Teacher 的实际输出（JSON fields） | 它回答的问题 |
|---|---|---|---|
| **A — Target-only** | 只有当前 respondent 的 **62-feature profile**；不提供其他人的 label。 | `prediction: 0/1`，直接给最终 flu vaccination binary decision。 | 不提供历史案例时，Teacher 直接预测能做到什么程度？ |
| **B — Balanced Few-shot Direct** | 当前 target + 从 Train20 选出的 **8 个相似 labeled examples**：4 vaccinated、4 not vaccinated；每个例子展示与 target 不同的特征编码。 | `prediction: 0/1`。要求利用示例直接分类，**不要求生成比较过程**。 | 只是让 Teacher 看见相似人的已知 outcomes，是否比 A 更准确？ |
| **C — Balanced Comparative Reasoning** | 与 B **完全同一组、同一顺序的 8 个例子**，但要求明确比较哪些 feature patterns 支持或反对接种。 | `prediction: 0/1`（最终**仍然必须二选一**）；`supporting_feature_patterns`（支持当前比较判断的特征模式）；`conflicting_feature_patterns`（相反的证据和反例模式）；`comparative_summary`（综合比较）；`insufficient_evidence: true/false`（模型认为证据是否不足）。 | 在历史案例信息相同的条件下，**显式 comparison 能否比 B 的直接预测多带来一些可用信号？** |

例如，B 只需要回答 `{"prediction": 1}`；C 则需要在比较历史案例后同时给出上述 explanation fields 和 `prediction`。**C 不是先判断“不确定”就停止作答，而是在表达证据不足时仍然需要给出二分类预测。** 因此三组统一用 binary Accuracy 评价；`insufficient_evidence` 只用于额外分析 Teacher 何时认为证据冲突。

两个主要对照分别是：

- **B − A：** historical demonstrations 本身是否提供增量信息；
- **C − B（最关键）：** 在示例完全一致的情况下，explicit comparative reasoning 是否有额外价值。

这样就避免把“给了更多 labeled examples”误认为“reasoning 做得更好”。本实验的 Teacher prompt 只包含 Train20 历史案例真实 labels，**不包含目标 Test80 的真实 label**。

## 5.3 四种 target 来源：每组都对应一个不同的 reasoning 假设

我们不是随机抽取 600 人，而是从 Test80 中选出 **4 组 × 150 个**，分别测试不同的局部可预测性结构。筛选依据来自 Train20 的 20NN labels 与之前 DSBS Student 的 prediction margin；**旧 DSBS 仅用于筛样本，不参加本次 A/B/C Teacher 预测**。

| Stratum | N planned | 更明确的筛选规则 | 为什么要单独研究这类人？ |
|---|---:|---|---|
| **G1 — Near-tie Boundary** | 150 | 最近 20 个 Train20 中，vaccinated 数为 **8–12 人**（另一标签为 12–8 人）；旧 DSBS 两标签 score 离 0.5 的 margin **≤0.02**。 | **标签本身高度混杂，模型也犹豫。** 比较案例若有效，应帮助识别真正有用的区别；若无稳定特征，额外 reasoning 也可能只是增加猜测。 |
| **G2 — Directional Feature Conflict** | 150 | 最近 20 人中 vaccinated 数为 **5–7 或 13–15**（已有 65%–75% 左右的多数方向）；DSBS margin **≤0.03**；同时在选定的行为相关特征中，至少有一项 target exact-code match 更偏向 vaccinated neighbors（match gap **≥+0.35**），又至少有另一项更偏向 not-vaccinated neighbors（**≤−0.35**）。 | **最符合本研究原始假设。** 局部有一定总体方向，但不同特征给出相反证据。这里最需要的不是更多例子，而是判断哪个 evidence 更重要、怎样整合。优先检验 C 是否比 B 好。 |
| **G3 — High Neighborhood Consensus** | 150 | 最近 20 人的标签为 **16:4 或 4:16**；训练邻域已有 80% 的多数方向。 | **有较强的历史共识。** 观察 Teacher 能否利用历史中的稳定方向，以及强制比较正反证据是否反而稀释这种信息。 |
| **G4 — Moderate Purity / High Margin** | 150 | 20NN vaccinated 数为 **6–14**，意味着邻域仍有少数反例；但旧 DSBS margin **≥0.05**，比 G1/G2 明显远离原预测边界。 | **原 Student 相对自信，但历史邻居并非全都一致。** 比较是否帮助发现被忽略的例外，还是在没有充分理由时推翻已有预测；重点分析 C 相比 B 的 harmful flips。 |

G1 与 G2 都是 low-margin cases，但困难不同：**G1 是邻域整体很混；G2 是邻域已有方向，却存在特征层面的明显冲突。** G3 与 G4 则构成另一个对照：**G3 的历史 outcome 更一致；G4 的原 Student 更自信，但邻域仍然混合。** 因此四个 group 不是随机分桶，而是直接对应四种关于 comparative reasoning 的研究假设。

抽样时还排除了 Train20 support 过稀疏的目标，并尽量平衡四组中的**旧 DSBS 预测正负类别**，避免某个 group 只是单一预测类。整个选样过程不利用 target true label。

**重要的 design detail：** B/C 最后都只收到最近邻中各 **4 个 vaccinated + 4 个 not-vaccinated** 的示例，而不是原来的全部 20 个。这样两组输入可控，但原来 16:4、14:6 等多数比例在 prompt 中被人为平衡。尤其对于 G3，Teacher 并没有直接看到“16 人接种、4 人未接种”这个局部 prevalence；这是解释其结果时需要考虑的实验处理因素。

## 5.4 统一的评价方式

600 个 targets 分别在 A、B、C 下预测，每个 target 有三个 binary predictions 才进入 paired comparison。目前 **599 个 target 三种条件全部完成**；剩余 1 个 C 的结果因格式解析失败而未能纳入。因此后续整体比较与分组比较均使用**完全相同的 599 人**：G1/G2/G3 各 150 人，G4 为 149 人。

评价报告三类结果：**各组 A/B/C Accuracy**、**B−A 与 C−B 的 paired HELP/HURT**，以及 C 的 `insufficient_evidence` 和预测正类比例。这样我们可以同时判断模型是否提高准确率、在哪类 target 上提高、是否系统性偏向某个 outcome。

## 5.5 Teacher 三种方法的整体结果（599 paired targets；本周重新计算）

| Evaluation method | N | Correct | Accuracy | Predicted vaccinated rate |
|---|---:|---:|---:|---:|
| **A — Target-only Teacher** | 599 | 337 | **56.26%** | 70.62% |
| **B — Balanced Few-shot Teacher** | 599 | 371 | **61.94%** | 63.94% |
| **C — Comparative Reasoning Teacher** | 599 | 323 | **53.92%** | **15.53%** |
| Label-only Student（仅作背景参照） | 599 | 439 | 73.29% | 47.58% |
| DSBS α=0.90 Student（仅作背景参照） | 599 | 439 | 73.29% | 49.92% |

599 个 targets 的 true vaccinated rate 为 **49.25%**。Teacher A/B/C 皆没有 Student 的训练优势，所以真正应比较的是**同一个 Teacher、同一批 cases 的三个 inference conditions**。

成对比较：

| Paired difference | HELP（后者改对） | HURT（后者改错） | Net correct | Δ Accuracy | 95% case-bootstrap CI | Paired exact p-value |
|---|---:|---:|---:|---:|---:|---:|
| **B − A** | 63 | 29 | +34 | **+5.68 pp** | **[+2.67, +8.85] pp** | ≈0.00051 |
| **C − B** | 135 | 183 | −48 | **−8.01 pp** | **[−13.86, −2.17] pp** | ≈0.0083 |

统计区间来自相同 599 人上的 paired bootstrap（5,000 resamples），p-value 基于 paired discordant cases；本节结论对应刻意挑选的四类 diagnostic samples。

**初步结论：** 提供相似历史案例（B）比 target-only（A）表现更好；但在相同 8 个 historical cases 上再强制 LLM 展开 comparative reasoning（C），并没有改善预测，反而出现 **8.01 percentage points** 的下降。这个负结果应当直接报告，不能替换成“reasoning 在某些案例可能更好”的泛泛推测。

## 5.6 核心分析：Comparative Reasoning 对哪些 case 有帮助，在哪些 case 上反而变差？

600-case 设计的重点不是 overall Accuracy，而是 **C 是否真的能解决我们为它挑选的 feature-conflict cases，以及有没有损害其他本来较容易判断的 respondents**。

### 5.6.1 四类案例的 A / B / C 结果

| Case group | Paired N | A Accuracy | B Accuracy | C Accuracy | B − A | **C − B** |
|---|---:|---:|---:|---:|---:|---:|
| **G1 Near-tie Boundary** | 150 | 53.33% | **56.00%** | 47.33% | +2.67 pp | **−8.67 pp** |
| **G2 Directional Feature Conflict** | 150 | 51.33% | **58.00%** | 54.00% | +6.67 pp | **−4.00 pp** |
| **G3 High Neighborhood Consensus** | 150 | 60.67% | **64.67%** | 58.67% | +4.00 pp | **−6.00 pp** |
| **G4 Moderate Purity / High Margin** | 149 | 59.73% | **69.13%** | 55.70% | **+9.40 pp** | **−13.42 pp** |
| **All paired** | **599** | 56.26% | **61.94%** | 53.92% | **+5.68 pp** | **−8.01 pp** |

把每个 target 从 B 换成 C 后，究竟是“改对了”还是“改错了”？

| Group | B wrong → C correct（HELP） | B correct → C wrong（HURT） | Net correct |
|---|---:|---:|---:|
| G1 | 26 | 39 | −13 |
| **G2** | **38** | **44** | **−6** |
| G3 | 34 | 43 | −9 |
| G4 | 37 | **57** | **−20** |
| **Total** | **135** | **183** | **−48** |

四组都呈现 **B 优于 A、C 低于 B**，但下降幅度不同。进一步分析发现，除了 `reasoning` 是否能够正确整合 evidence，**C 的输出类别偏向也是一个非常明显的变化**：

| Group | Test target 实际 vaccinated rate | B 预测 vaccinated rate | **C 预测 vaccinated rate** | C 标记 `insufficient_evidence` |
|---|---:|---:|---:|---:|
| G1 | 50.67% | 53.33% | **16.67%** | 80.00% |
| **G2** | 50.67% | 66.00% | **15.33%** | 73.33% |
| G3 | 47.33% | 62.67% | **16.67%** | 77.33% |
| **G4** | 48.32% | **73.83%** | **13.42%** | 79.19% |

也就是说，C 无论遇到哪一种局部结构，都很少预测 vaccinated。它不只是选择了不同证据，而是把 decision tendency 整体转向了 `not vaccinated`。因此需要把“某类 reasoning 有无价值”和“prompt 触发的类别倾向变化”分开看。

### 5.6.2 G2 — 最值得期待 Comparative Reasoning 的 Feature-conflict Cases

**G2 是这个实验最重要的 hypothesis test。** 这些 target 不同于纯粹的 near-tie：在 20NN 中已经有 13–15 对 5–7 的局部多数趋势，同时至少有两个行为相关 features 分别更贴近正负历史人群。我们原本希望 Teacher 经过比较后，可以识别**哪部分局部 evidence 更可信**，从而比简单 few-shot prediction 更好。

结果是：

```text
G2 (n = 150)
A: 51.33%
B: 58.00%   (+6.67 pp vs A)
C: 54.00%   (−4.00 pp vs B)

B → C: 38 HELP / 44 HURT
```

这个结果有两层含义。

**第一，提供相似 historical examples 是有帮助的。** B 比 A 多预测正确了 10 人，说明这些局部对照案例确实可能提供 Teacher 单看 target profile 时缺少的信息。

**第二，显式比较没有进一步把这些信息变成更准确的 decision。** C 虽然相对 B 改对了 38 人，但同时改错 44 人。更具体地说，**有 42 名实际 vaccinated 的 target，被 C 从 B 原本正确的 vaccinated 改成了 not vaccinated**；另一方面，有 37 名实际 not-vaccinated 的 target，因为这种转向而被 C 改对。也就是说，C 在 G2 的收益与损失主要来自同一种**朝 not-vaccinated 移动的预测偏向**，而非已经观察到一套稳定、精确的 feature-conflict resolution rule。

因此，目前我们不能说 G2 完全没有可学的 signal。相反，**B 的提升说明局部案例信息值得保留；需要重新设计的是 C 如何利用这些信息**：它应当学习有条件地比较 feature evidence，而不是把冲突普遍解释为不足以支持 vaccinated。

### 5.6.3 G1 — 当邻域本来就接近五五开，更多比较也没有创造额外信息

G1 的标签分布约在 8:12–12:8，且 DSBS margin 极小。A 为 53.33%，B 提高到 56.00%，表明 few-shot 的帮助有限；C 降至 **47.33%**。

这里最直观的解释是：**历史案例本身就有大量正反例，模型列出更多差异不等于找到更可靠的分类方向。** G1 的 C 有 80% 标记证据不足，预测 vaccinated 比例仅 16.67%。原本接近均衡的任务被系统性偏向负类，因而额外比较很难实现净增益。

G1 更适合检验 **uncertainty recognition 和 prediction stability**，而不是期待详细 reasoning 能消除原本缺失的标签信息。这也与后续尝试独立 uncertainty head 的方向直接衔接。

### 5.6.4 G3 — 邻域多数方向已经明显，但 4+4 examples 隐藏了原始 prevalence

G3 在原始 20NN 中是 16:4 或 4:16，按理说有相对容易利用的 majority evidence。B 的 Accuracy 为 **64.67%**，高于 A 的 60.67%；C 降至 **58.67%**。

但这里有一个重要的实验设计因素：**B/C 都只看 4 vaccinated + 4 not vaccinated 的平衡 demonstrations**。原来 20NN 中的 16:4 并没有作为比例信息展示给 Teacher。对于这种本来 local direction 很强的 case，模型看到的反而是一组人数完全均衡的正反例；如果要求它进一步放大这些 conflicting evidence，就可能淡化原有的局部趋势。

因此 G3 给出的启示很具体：若想让 Teacher 利用这些本来明显的统计规律，下一轮应比较 **balanced 4+4 demonstrations** 与 **保留原始 label prevalence / 直接呈现 20NN 统计摘要** 的结果，再测试显式 comparison 是否仍然有必要。

### 5.6.5 G4 — 为什么对原 Student 相对自信的案例，Comparative Reasoning 退化最明显？

G4 是下降最大的组：B **69.13%**，C **55.70%**，相差 **13.42 pp**。这批 respondent 的 DSBS score margin ≥0.05，说明**原 Student 在其自身的两标签评分规则下**相对有把握；但 Teacher 的 A/B/C 决策仍然是独立的，不能直接说“Teacher 本来就很有把握”。

为了看清为什么 C 会变差，我把 G4 的 prediction flips 按实际 flu label 再拆开：

| G4：从 B 切换到 C | 数量 | 对 Accuracy 的影响 |
|---|---:|---|
| B 正确预测 vaccinated → C 改成 not vaccinated | **56** | **HURT** |
| B 正确预测 not vaccinated → C 改成 vaccinated | 1 | HURT |
| B 错误预测 vaccinated → C 改成 not vaccinated | **36** | **HELP** |
| B 错误预测 not vaccinated → C 改成 vaccinated | 1 | HELP |
| **合计** | **57 HURT / 37 HELP** | **Net −20 correct** |

C 的 predicted vaccinated rate 从 B 的 **73.83%** 降到 **13.42%**。尤其明显的是，它把大量原本 B 已经正确识别的 vaccinated respondents 重新划成负类；这些损失超过了它纠正 false-positive cases 的收益。

这也体现在分类表现的方向上：在 G4，B 对 vaccinated 的 Recall 为 **94.44%**，C 只有 **18.06%**；而对 not vaccinated 的 Specificity 从 **45.45%** 提高到 **90.91%**。换句话说，C 通过大规模转向负类，**换到了更高的 specificity，却损失了更大幅度的 positive recall**。

**所以 G4 下降的直接原因并不是“原本清楚的 signal 被模型错误分析”已经得到证实，而是 comparative prompt 明显改变了最终决策的类别偏好。** 一种合理的机制假设是：当指令不断要求检查 conflicting evidence、说明为什么正类证据不足时，模型更容易把不确定性转化为保守的 `not vaccinated` 判断。当前四组中 **73%–80%** 的 `insufficient_evidence` 标记，也说明这种诊断式 prompting 倾向较普遍，而不是只发生在真正困难的 G1。

这使 G4 成为下一轮很有价值的 **non-degradation control**：新的 reasoning method 不仅要能在 G2 识别冲突，还应尽量避免在 G4 这种已存在一定有效预测信号的 case 上无必要地翻转结果。

### 5.6.6 综合解读：下一步要优化的是“比较后的决策”，而不是解释长度

这次 599-case Teacher experiment 给出了很清楚的结果层次：

1. **Historical examples 有增量价值。** B 在四类 samples 上都比 A 更准确，整体增加 **5.68 pp**。
2. **显式 comparative reasoning 尚未提供额外预测收益。** C 在四类样本上都低于 B，整体 **−8.01 pp**；最重要的 G2 也没有实现预期提升。
3. **主要可观察失败方式是负类预测偏向。** C 在全部 599 人中仅 **15.53%** 预测 vaccinated；这比单纯“比较写得不够好”更值得优先修正。
4. **不同 group 对下一步有不同用途。** G2 用于检验 feature conflict integration；G1 检验 uncertainty；G3 检验如何呈现历史多数比例；G4 检验 reasoning 是否造成 unnecessary harmful flips。

因此短期内不急着把 C 的 reasoning 输出拿去生成新的 Student SFT 数据。更直接的下一步是保留 Teacher A/B/C 的 paired setup，在**相同历史案例、相同二分类输出要求**下，调整 C 的 decision instruction，并补充 *local prevalence* / *feature evidence strength* 对照；确认比较过程确实能改善 **G2** 而不明显损害 **G4**，再考虑将这类 comparative evidence 蒸馏给 Qwen3.5-9B Student。

# 6. 两个正在推进的新方向：Uncertainty Head 与 Persona Augmentation

## 6.1 Additional Classification Head：把 Uncertainty 作为独立预测目标

对应老师在 meeting 中的建议：既然许多相似 respondent 的标签本身相互冲突，是否应该允许模型明确表达 “I am uncertain”，而不是对任何 case 都生成非常肯定的解释？

目前考虑在二分类架构旁增加一个 **uncertainty head / auxiliary classification branch**，而不是直接用 `vaccinated / not vaccinated / uncertain` 三个类别替代 NHIS 原始标签（因为 **NHIS 没有第三个真实 outcome label**）。一个可评估设计是：

```text
Respondent features / shared representation
       ├── Main label head      → vaccinated / not vaccinated
       └── Uncertainty head     → low / high uncertainty
```

Uncertainty supervision 的候选来源包括 **Train20 的 out-of-fold prediction stability、out-of-fold classification error、local label entropy / local purity、ensemble disagreement**；如果采用模型错误作为目标，必须通过 Train20 cross-validation / OOF 得到，不能以 Test80 的预测错误作为 training signal。

Evaluation 需要同时保留两套指标：

1. **Full-coverage Test80 binary Accuracy / AUC**：所有人仍必须有 vaccinated/not vaccinated prediction，可与现有 benchmark 公平比较。
2. **Selective prediction / abstention analysis**：只在允许 abstain 的 evaluation protocol 下，额外报告 coverage、retained-case accuracy、risk–coverage curve、uncertainty calibration；不能把少预测了几个人造成的 Accuracy 提升当作原 benchmark 的 improvement。

**与本周结果的直接联系：** E2 的 `insufficient_evidence` 出现率达到 77.46%，且没有集中在最困难的 G1，说明 free-form LLM uncertainty flag 不能未经验证就作为 selector。额外 uncertainty head 的目标是用**可验证的训练信号**学习不确定性，而不是让 LLM 主观说“没有证据”。

**当前状态：** 方法设计 / 初步推进中；**尚无可以报告的 full Test80 accuracy 或 risk–coverage 结果**。

## 6.2 Persona / Latent Behavioral Feature Augmentation

对应老师提出的另一个问题：NHIS 数据里有慢病、保险、医疗接触、其他疫苗接种历史等 proxy，但缺少直接测量的 **subjective vaccine risk perception、attitude、intention、provider trust** 等心理变量。是否能让 LLM 从可观测 proxies 中产生一个可用于决策的 persona / latent behavioral representation？

一个初步思路：

```text
62 observed NHIS features
        ↓
LLM / constrained persona builder
        ↓
少量结构化 latent behavioral descriptors
        ↓
Concat with original 62 features
        ↓
XGBoost / tabular baseline 或 Qwen Student
        ↓
Test80 binary vaccination prediction
```

应优先尝试 **可追溯的、低维的 behavioral style / preventive engagement descriptors**，例如从其他疫苗接种行为与 healthcare use 归纳出的可能预防行为倾向，而不是生成无法验证的具体心理属性。

必要的实验控制：

- **Original 62 features baseline** vs **62 features + persona**；还应比较 **同等维度的普通 ML feature engineering**，防止增益只是因为变量增加。
- Persona generation 不得看到 target flu vaccination label；用 Train20 内交叉验证选规则 / 调参，保持 Test80 untouched。
- 将生成内容标注为 **inferred proxy / hypothesis**，不能当作 respondent 实际报告的心理状态或健康行为事实。
- 检查 persona 是否只是在重复输入的 vaccination-history feature；如果它没有带来任何 out-of-sample 额外 predictive information，复杂的 LLM augmentation 未必值得继续。

**当前状态：** 作为 feature expansion 和模型上限问题的 ongoing approach；本周结果包没有该方向的新 Accuracy，不能编写 performance gain。此前 frozen LLM persona 尝试提升有限，但不能代替现在 proposed representation-learning / SFT 实验的验证。



# 7. FLARE Follow-up：强 ML Baseline 复核与 NHIS / FLARE 数据差异分析（下一阶段实验）

## 7.1 为什么需要重新研究 FLARE？

上次 meeting 中还有一个尚未回答的核心问题：**为什么 FLARE 能在个人行为预测上获得较明显的 performance improvement，而我们在 NHIS 流感疫苗接种预测中，尝试不同 LLM prompting、HBM reasoning 和 SFT 后，Accuracy 仍然主要集中在 76%–77%？**

目前至少有两种可能解释：

1. **Benchmark gap：** FLARE 论文采用的 tabular baseline 未充分覆盖近年来更强的 ML 方法；其优势可能部分来自原有对照模型较弱，而不完全来自 LLM reasoning。
2. **Dataset–method fit：** 野火撤离和流感疫苗接种虽然都是 individual decision prediction，但数据中能够观察到的心理状态、决策线索及外部环境不同；FLARE 的 *perception → decision* pipeline 可能恰好适合其数据，而 NHIS 当前 62 features 难以提供同等强度的 subjective decision signals。

因此，这一阶段不直接移植更多 FLARE modules，而是先通过两个相互配合的实验，区分 **baseline 是否充分** 与 **数据是否适合 theory-guided LLM reasoning**。

**参考论文：** Chen et al. (2025), *From Perceptions to Decisions: Wildfire Evacuation Decision Prediction with Behavioral Theory-informed LLMs*，ACL 2025。论文：[ACL Anthology](https://aclanthology.org/2025.acl-long.1438/)；开源实现与数据入口：[FLARE GitHub](https://github.com/SusuXu-s-Lab/FLARE)。

### 原论文中值得复核的结果

原论文在合并野火调查数据（Combined Data）的 Table 1 主要比较 Logistic Regression、Random Forest、HCM 与直接 GPT-4o inference；FLARE 分别使用不同 LLM backbones。论文报告的 **Accuracy** 为：

| Method | Combined Data Accuracy | 在我们后续实验中的作用 |
|---|---:|---|
| Logistic Regression | 0.697 | 传统线性 baseline |
| Random Forest | 0.735 | 原论文的非线性 tabular baseline |
| HCM | 0.732 | PADM-informed choice-model baseline |
| GPT-4o direct inference | 0.738 | 不使用完整 FLARE pipeline 的 LLM baseline |
| **FLARE + GPT-4o** | **0.816** | 与同一 GPT-4o direct inference 对照 |
| **FLARE + Claude-3.5** | **0.895** | 原论文使用另一模型 backbone 的结果 |

这意味着一个尚待验证的问题是：**如果在同一组原始 survey features 上，认真训练并调优 XGBoost、CatBoost、LightGBM 或 TabPFN，是否也能达到接近 FLARE 的性能？** 论文表中的 0.816 或 0.895 是论文已有结果，不能作为我们新增 ML 的结果；不同 backbone 的 FLARE 数字也不应混合成同一个严格对照。

## 7.2 Experiment F1 — 在 FLARE 原始开源数据上重新建立 Strong Tabular ML Benchmarks

### Research Question

> **FLARE 相对于传统模型的优势，在加入经过合理调参的 strong tabular learners 后是否仍然存在？**

目前 NHIS 的经验是：XGBoost、TabPFN、Qwen3.5 Student 等方法已经取得比较接近的表现，说明 ML 在当前结构化特征上的拟合能力不弱。相比之下，FLARE 论文的主要 ML baseline 侧重 Logistic Regression 和 Random Forest，并没有在 Table 1 中系统报告 XGBoost / CatBoost / LightGBM / TabPFN 的同数据、同划分比较。因此，先把 *tabular learning 本身能做到什么程度* 估计清楚，才能知道 FLARE 还有多少独立的 method advantage。

### 数据与对齐

使用作者公开的三组 post-wildfire survey：**Carr Fire、Kincade Fire、Marshall Fire**。论文 Table 4 报告：Marshall **334** 个样本、Kincade **270** 个样本、Carr **500** 个样本，并分别报告 **54.19% / 81.41% / 89.40%** 的撤离比例；Appendix A.1 又将 Carr 写作 **284 responses**。因此第一个操作不是训练，而是从 GitHub 的实际 data files 和 preprocessing code 核对每个事件的有效样本数、纳入规则、target mapping 与缺失记录，形成统一的 **Data Manifest**；避免把文中不同统计口径拼接为一个未经检查的总样本量。

首先沿用论文的 **80% train / 20% test** 评价结构；如果原作者的精确样本 split 不公开，则固定新的 split IDs，以至少 **5 个随机种子**重复实验，并明确区分“复现论文评估协议”和“严格复现论文原始划分”。所有方法必须使用同一组 train/test respondent IDs、相同 target definition、相同预处理信息范围。

### 对比模型与特征版本

| Model | 设置 | 为什么需要 |
|---|---|---|
| Majority-class baseline | 只预测训练集多数类 | 检查野火调查严重不平衡时 Accuracy 的真实参照水平 |
| Logistic Regression / Random Forest | 重新运行论文对应的基础 ML 类别 | 建立与论文 Table 1 可对照的 benchmark anchor |
| **XGBoost** | categorical encoding + train-only tuning / early stopping | 检验非线性 feature interaction 能否解释 FLARE 的优势 |
| **CatBoost** | 原始类别编码、合适的缺失值处理、train-only tuning | 适合小样本、混合类别型 survey features |
| **LightGBM** | 小样本限制下调节叶节点与正则化 | 提供另一种 boosted-tree 对照 |
| **TabPFN（可选）** | 严格控制特征类型和当前模型适用范围 | 检验 foundation-style tabular learner 是否改变相对排序 |

主比较先使用**原 FLARE prediction pipeline 可获得的同一套 respondent / event covariates**，不要为 ML 私自增加额外后验信息，也不要让 FLARE 看到不同版本的数据。为区分方法的优势与输入特征的优势，随后再做统一的三组 feature views：

- **F1-a — Demographics / household / pre-event profile：** 人口统计、家庭条件、历史风险认知或预先具备的准备状态（只选真正属于决策前可用的信息）。
- **F1-b — F1-a + observable event / situational cues：** 预警信息、撤离命令、事件位置或环境线索，以及明确可在决策前观测到的变量。
- **F1-c — F1-b + perception-related survey indicators：** 加入与 perceived threat、perceived risk 或其他 decision-related perceptions 对应的问卷条目；单独检验这部分信息的增量贡献。

一个重要的数据检查是调查问卷的**时间顺序**：这些是灾后调查，可能含撤离经历、撤离时间等已经发生的行为结果。所有 feature view 都需要明确哪些变量在实际预测时可观察；若含 post-decision variables，单独标记为 retrospective setting，不能当作 prospective prediction 的证据。主结果应使用同一、可复核的预测时间定义。

### Evaluation 与需要输出的结果

既要复现论文的 **Accuracy / Macro F1 / Weighted F1 / Stay–Evacuate per-class recall**，也补充 **Balanced Accuracy、ROC-AUC、PR-AUC（可计算时）**。所有调参只在训练集内部完成，报告 repeated splits 的 mean ± std、每事件结果与 paired comparisons。

主实验矩阵：

| Training / Test | 研究目的 |
|---|---|
| **Combined → Combined** | 在论文主 setting 下比较 FLARE 与更强 tabular ML |
| **Kincade → Kincade；Marshall → Marshall** | 观察单事件、小样本、不同 label imbalance 下的模型差异 |
| **Kincade → Marshall；Marshall → Kincade** | 对应论文跨事件 generalization：比较理论方法和数据驱动模型的 transfer robustness |
| **Feature views F1-a / F1-b / F1-c** | 解释 ML 或 FLARE 的增益到底来自哪些可观测信息 |

最后主要对比：

```text
Published FLARE performance
             vs.
Reproduced Logistic Regression / Random Forest
             vs.
Tuned XGBoost / CatBoost / LightGBM / (optional) TabPFN

同一数据、同一预测时点、同一划分、同一评价指标
```

如果 stronger ML 接近或超过 FLARE，说明论文原有 performance gap **可能主要反映 baseline coverage / tuning**，而不是 LLM 在这一类 survey data 上有天然不可替代的优势。若 FLARE 仍领先，则进一步查看领先是否集中在 **cross-event transfer、少数类 recall、perception-rich cases**，再判断 theory-guided inference 带来的额外价值。

## 7.3 Experiment F2 — NHIS 与 FLARE Dataset 的结构性差异：哪些数据真正支持 Individual Decision Reasoning？

### Research Question

> **野火撤离与流感疫苗接种的共同点都是“observed characteristics → individual action”，但两套数据是否提供了同样类型、同样靠近决策过程的信息？为什么 FLARE 的 perception-based reasoning 在野火场景有效，而 NHIS 的 HBM-guided reasoning 难以进一步超越 strong tabular ML？**

### 7.3.1 核心对比：两个 dataset 究竟给了模型什么样的“决策信息”？

FLARE 与 NHIS 的区别不能简单概括成“前者有心理变量、后者只有 demographic features”。**更准确的差别在于：数据是否让模型观察到个人如何判断风险、在什么具体情境下采取行动，以及我们能否用受访者自己报告的理由检验生成的解释。** 这三个层次分别对应 *mental-state observability*、*decision-context observability* 和 *reasoning verifiability*。

| 维度 | FLARE：Post-wildfire evacuation survey | NHIS 2024：当前冻结的 62-feature profile | 为什么会影响 LLM reasoning |
|---|---|---|---|
| **Prediction target / 时间窗口** | 对一次**具体野火事件**是否撤离（Evacuate / Stay），问卷围绕该事件回顾风险、预警、行动 | 过去 **12 个月**是否接种流感疫苗；横截面资料中并不记录每个人的明确“接种决策时刻” | FLARE 的 feature、感知与 action 可以锚定同一事件；NHIS 难以将一般健康特征直接关联到某次具体的接种选择 |
| **① 直接感知与心理测量** | 包括个人伤害或财产受威胁的感知、火灾可能性的 1–5 分评价，以及部分 risk/threat-related survey indicators；可用于构造或校准 PADM perceptions | 健康状况、慢病、保险、医疗接触、既往其他疫苗史等；**缺少流感疫苗态度、接种意愿、对流感威胁的个人判断**等直接测量（以当前 62 features 为准） | FLARE 有“观测指标 → perception”这层可拟合的关系；NHIS 的 HBM 多数只能“proxy → 猜测 perception”，可能产生没有证据支撑的个人心理解释 |
| **② 外部事件与行动线索** | 有无收到撤离命令、收到时间、是否看到/感到火焰与余烬、当时反应、家庭应急计划等；论文推理还可整合 wildfire progression 等描述性情境 | 有医疗资源与就医利用的信息，但**没有**接种前具体医生推荐、疫苗供应/预约、当时的个人接种契机等逐人事件记录 | FLARE 可以把“感知风险”与“收到什么信号、此时能否/是否需要行动”联系起来；NHIS 仅凭 chronic condition、insurance 等很难重建实际 decision episode |
| **③ 可核验的个人行为理由** | 部分调查包含开放回答；论文将生成解释与 **206/604** 名受访者的开放回答作定性对照（Appendix G） | 当前监督基准仅有 vaccinated/not vaccinated outcome，没有受访者逐一报告的**流感疫苗接种理由** | FLARE 至少有部分 person-level motivation evidence 能检查 reasoning 是否与当事人叙述一致；NHIS 的 HBM rationale 往往只能判断是否“符合理论”，难以判断是否真正解释该人的行为 |
| **Behavioral framework / 模型连接** | PADM 的 threat assessment / risk perception 被映射到 survey indicators；模型再结合 reasoning-pattern routing、perception-score calibration、context 和 error memory | HBM 提供 perceived susceptibility / severity / benefits / barriers / cues to action 等概念，但目前没有相应的逐人主观构念标签 | 理论提供的“分析框架”只有和可观测、可校准的资料连接起来，才能约束模型，而非仅增加解释文字 |
| **Data scale / label balance** | 三个事件样本量相对小，且不同事件 Evacuate 比例差异大；还需要考虑跨事件 transfer | Train20=6,414、Test80=25,656，positive rate 47.42%；在当前 feature space 上 strong ML 与 Student 结果接近 | FLARE 的优势可能来自小样本、多事件情境下的理论归纳；NHIS 则更适合检验可观测 features 是否已被强 tabular models 充分利用 |

下面把前三项拆开说明。它们并非三个平行的 feature families，而是预测个人行为所需的**三个连续层次：心理状态是否有依据 → 决策时的情境是否被观察 → 生成的解释是否有独立材料可检验**。

### 7.3.2 三个具体差异：为什么它们可能决定 FLARE 的适用性？

#### Difference 1 — Direct Perception Indicators：心理状态是否能够被数据测量和校准？

**FLARE 中有什么？**

FLARE 使用的灾后调查不仅记录年龄、收入等背景变量，还调查一个人**对那一次野火威胁的判断**。原论文 Section 3.1–3.2 及 Appendix F 的例子包括：

- 对火灾可能威胁自家财产的主观可能性评分（1–5）；
- 对 personal injury / death threat，以及 home / neighborhood damage risk 相关的评价指标；
- 以往是否知道本地区存在野火风险、是否有应急计划，以及收到预警时的反应等与风险判断相关的问卷条目。

这些变量不完全相同：有些是**直接报告的感知指标**，有些只是可能预测感知的**相关输入变量**。FLARE 的方法恰恰利用这种区分：先以调查中的 perception indicators 为被解释对象，回归筛选与之有关的 survey variables；再让 LLM 根据选择的变量生成 textual perception 和 1–5 perception score，并利用从训练样本建立的 knowledge base / retrieved exemplars 约束评分的一致性（论文 Section 3.2.1–3.2.2）。因此其 perception 并非只是一段未经校验的自然语言猜想，而有一条可与观测数据联系的统计路径。

**NHIS 中缺少什么？**

当前 62 个输入变量能够让我们知道一个人*是否患有慢病、健康状况如何、有没有保险、最近有没有接触医疗系统、是否接种过其他疫苗*，却未直接告诉我们这个人：

- 是否认为自己容易感染流感、感染后后果是否严重（perceived susceptibility / severity）；
- 是否认为流感疫苗有效、是否担心副作用或不信任疫苗（perceived benefits / barriers）；
- 是否打算在这个流感季接种，以及哪些信息会真正改变其选择（intention / cues to action）。

这些问题和“客观医学风险”不是同一回事。例如，**患有多种慢病**可以提示感染后风险较高，却无法说明当事人**是否认为自己有风险**；**经常去看医生**可以表示具备接种机会，却不能证明医生**实际推荐过接种**；**接种过其他疫苗**可能表明某种预防行为倾向，但不等于愿意接受流感疫苗。这正是我们 HBM rationale 容易出现的问题：它可以合理地把 observable proxies 映射为 HBM concepts，却缺少独立的 respondent-level mental-state measurements 检查这些推断是否成立。

**为什么这会改变模型的表现？**

对于 FLARE，潜在逻辑是：`survey variables → 可校准的 threat/risk perceptions → contextualized action prediction`。对于当前 NHIS，更接近：`health/healthcare/vaccine-history proxies → teacher 推测 HBM dimensions → binary outcome`。后一个过程的中间状态缺乏直接监督，因此即使 HBM explanation 在语言上完全通顺，也未必增加超出 tabular ML 的可预测信息。这个差异也解释了为什么我们不能把**增加 reasoning 的长度**等同于**增加 decision-relevant evidence**。

**对应的实验检验。** 在 FLARE 原始数据上，使用同一个 tuned ML model 比较 `background only`、`+ event cues`、`+ perception-related information` 的增益；另做一组 train-only / cross-fitted perception prediction，比较“使用真实 survey perception indicators”和“只根据其他变量预测 perception”的效果。这样可以直接测出 FLARE 的优势有多少依赖**直接测量到的 perception**，有多少仍能在 perception 只能被推断时保留。若加入 perception indicators 明显改善 ML，优先说明这是一类高价值输入信息；不应自动把全部收益归功于 LLM reasoning。

#### Difference 2 — Event-specific Cues：是否观察到了“为什么在那个时刻采取行动”？

**FLARE 中有什么？**

野火撤离是一个相对明确的 event-driven decision：居民可能先收到官方命令，随后看到火焰或余烬、观察他人是否离开、考虑家庭成员与财产，最后决定 Evacuate 或 Stay。FLARE 的部分问卷题目明确锚定时间顺序，例如“**在决定撤离或留下之前**是否收到 mandatory evacuation order”“何时收到通知”“在决定之前是否看到/听到/感受到火焰与余烬”（Appendix F）。论文的 CoT 也会使用此类个体线索及可获得的事件叙述来组织判断。

因此，两个 socio-demographic profile 很接近的人，即使对风险的主观评价相似，也可能因为**一人收到明确撤离命令，另一人尚未收到；一人直接看到火势，另一人没有；一人需要照料牲畜或守护财产**而采取不同动作。这些差异不一定是抽象的“心理类型”，而可以是**可观察的行动触发条件或约束**。FLARE 的 context integration 正是在把 `perception` 推向 `action` 的阶段提供额外信息。

这里需要区分：论文明确使用了个体报告的 warning / flames / preparedness 等变量，并说明可以整合 wildfire progression 等描述性信息；但作者同时指出调查仍**缺少系统的地理、气象、交通路线和避难场所等变量**（Limitation）。所以当前不能理解为 FLARE 已完整获得每个人的实时环境状态。

**NHIS 中缺少什么？**

我们预测的是“过去 12 个月有没有接种流感疫苗”，却通常不知道这个人经历了哪一次具体的接种决策。例如：是否在诊所被医生推荐、是否遇到现场接种机会、是否因为担心副作用而推迟、所在地区某时段的流感传播状况、是否已有预约或遭遇疫苗缺货，都不是当前 62 features 中逐人、逐时点记录的变量。

因此，两名 NHIS respondents 可能同样 68 岁、都有慢病、都有保险、都有 usual source of care，也都接种过某种其他成人疫苗，但一个在年度体检时接种了流感疫苗，另一个没有收到具体推荐或选择推迟。**这种接种结果的分歧未必对应 62 features 中能够定位到的一项差异。** 本周 Contradiction Audit 中出现的 highly similar profiles with opposite labels，就与这种信息缺口相吻合。

**为什么这会改变模型的表现？**

HBM 可以解释“这个人有哪些 benefits、barriers、cues to action”，但如果没有观察到发生在具体时间点的 cue，Teacher 常常只能把“看过医生”推理成“医生可能推荐”，把“慢病较多”推理成“可能重视接种”。FLARE 的一些 decision cues 则更接近实际 action trigger，能够在同样的背景画像下提供额外的区分信息。换句话说：**FLARE 的 reasoning 有机会整合“某个人遇到了什么”，而 NHIS 的 reasoning 更多是在总结“这个人通常是什么样”。**

**对应的实验检验。** FLARE 做 event-cue ablation，比较保留与删除 warning receipt/timing、direct fire exposure 和 preparedness 等变量后，ML Accuracy、local-neighborhood label consistency 和 cross-event transfer 如何变化；NHIS 则分别检查 healthcare-use 与 prior vaccination features 能减少多少 local-label conflict。更进一步比较二者在**只用背景变量**时的 contradiction rate，才能检验野火数据的相对优势是否主要来自 event-specific observability，而不是两个任务本身标签含义不同。

#### Difference 3 — Open-ended Decision Reasons：能否检验 LLM 的 explanation 是否接近受访者自己说的理由？

**FLARE 中有什么？**

论文 Section 4 和 Appendix G 专门提到，**604 名相关受访者中有 206 名提供了开放回答**，作者把生成的 reasoning 和这些回答作定性对照，作为部分的 reasoning-reference evidence。文中的一个例子，受访者选择留下，自己报告的理由包括：**认为所在位置不会有危险、看到其他人留下、想保护财产，以及需要照料宠物或牲畜**。作者随后展示 LLM reasoning 是否覆盖了这些动机主题（Appendix G）。这比只根据模型是否预测对 Evacuate/Stay 来判断 explanation quality，多了一层直接来自 respondents 的参照信息。

**NHIS 中缺少什么？**

当前 Flu vaccination benchmark 的 ground truth 是 `SHTFLU12M_A`，能够告诉我们接种或未接种，却不提供逐人对应的“**我为什么接种 / 为什么没有接种**”开放回答，也没有逐人的、独立测量的 HBM decision rationale。因此我们可以检查 Teacher explanation 是否正确引用 NHIS features、是否把 code 读对、是否自相矛盾；却很难直接检验它提出的*perceived benefits、trust、worry、provider recommendation* 等是否确实是受访者当时的真实动机。

这一点也与前面 HBM SFT 的 post-hoc explanation 问题直接关联：如果 Teacher 在生成 rationale 前已经知道了 target label，那么它可以对两个几乎相同但 outcome 相反的 profiles，各自写出一段完全自洽的 HBM story。只用最后的 classification accuracy，很难区分**真的识别了 respondent 的决策理由**与**为已知 outcome 找到一套合理叙述**。

**为什么这会改变研究能回答的问题？**

FLARE 至少可以检查 reasoning 是否覆盖了受访者亲自提及的 risk perception、warning、财产和家庭约束等主题；NHIS 目前无法进行同等层次的 *individual rationale alignment* 验证。这不等于开放回答提供了完全可靠的心理过程 ground truth——它仍是灾后自述，而且论文展示的是部分样本与定性例子——但它使**预测对不对**和**解释是否与本人自述一致**成为两个可以分别讨论的目标。我们当前 NHIS 的主要证据仍是 label-level prediction performance，不能将 HBM rationale 的流畅性直接解释成 psychological fidelity。

**对应的实验检验。** 在 FLARE 有开放回答的样本中，把理由归纳为预先定义的 categories（如 perceived low risk、received warning、property protection、family/pets、evacuation constraints），对照 A）纯表格特征预测、B）FLARE rationale、C）直接 few-shot rationale 的理由覆盖率和 unsupported-claim rate。同时检查用于评估的开放回答是否已作为该次预测的输入，区分 independent reference 与 input-repetition；在 NHIS 则把评估目标限定为 **feature grounding、contradiction consistency 和 outcome prediction**，不将未观测的个体心理动机当作事实。

**合起来看：**

```text
FLARE
  measured / calibrated threat & risk perceptions
       + event-specific warnings and action cues
       + partial self-reported reasons to audit explanations
          ↓
  theory → observable evidence → context → predicted action

NHIS (current 62-feature benchmark)
  health / access / utilization / vaccination-history proxies
       + no person-specific flu-vaccine decision episode
       + no respondent-level ground-truth vaccination rationale
          ↓
  theory → inferred HBM constructs → predicted action
```

我们目前更需要检验的，不是“PADM 在理论上是否优于 HBM”，而是**当理论构念能够被观测、行动情境能够被记录、解释能够被部分核验时，reasoning 有没有更大的实际发挥空间**。F1 的 Strong-ML Baseline 和 F2 的 Dataset Diagnostics 共同回答这一点：先测出这些变量本身能带来多少预测增益，再讨论 FLARE pipeline 的额外贡献。

### 7.3.3 为什么 FLARE 的完整方法可能更适合其数据？

FLARE 并不只是“让 LLM 多写 CoT”，而是把 **data-grounded perception construction** 放在预测之前：

```text
Event-specific survey variables
      ↓
基于调查数据选择与 threat / risk perceptions 有关的变量
      ↓
选择 case-specific reasoning pattern
      ↓
LLM 生成 perception + 1–5 quantitative scores
      ↓
利用训练数据中的 perception indicators / similar examples 进行 score calibration
      ↓
融合 warning / environmental / household context
      ↓
PADM-structured reasoning → Evacuate / Stay
      ↓
使用训练阶段积累的 error memory / reflection 修正后续预测
```

这套 pipeline 与 NHIS 目前 HBM rationale 的区别主要有三点：

1. **Perceptions 不完全依赖自由生成。** FLARE 从 survey indicators 出发选择变量，并用训练样本校准 LLM 生成的 perception scores；NHIS 中当前重要心理变量往往没有直接观测值，HBM 解释更依赖 indirect proxies。
2. **Individualized routing 与可观测环境相连接。** FLARE 会选择 reasoning pattern，结合撤离命令和事件上下文再决定行动；我们现有 HBM SFT 更偏向固定的 evidence categories，最终 feature weighting 由 Teacher 自行完成。
3. **对有矛盾的 historical cases 有专门的 feedback mechanism。** FLARE 的 memory 保存训练阶段预测错误及其 reflection，用于后续相似案例 retrieval；而这周 NHIS Experiment 2 的 4+4 comparative prompting 只是直接给出反例，再要求分析，尚未提供经过训练数据校准的 evidence strength 或 reliable routing。因此两者不是相同复杂度的 reasoning method。

这些是 **method–data alignment hypotheses**，而非已经得到验证的机制结论。下一步可以通过统一 feature views 和 matched comparisons 逐项检验。

### 7.3.4 怎样用实验证明“数据不一样”是关键，而不是仅凭直觉？

建议对两个 dataset 都生成可直接对照的 **Dataset Diagnostic Card**，并沿用相同的若干分析方法（模型和指标可以共用，但不强行把不同变量解释成同一心理量表）：

| 诊断实验 | 在 FLARE 上做什么 | 在 NHIS 上做什么 | 检验什么解释 |
|---|---|---|---|
| **D1. Feature availability / timing audit** | 按 demographics、event cues、direct perception indicators、post-decision responses 编目，并标记决策前是否可观测 | 按 demographics、health / affordability / access、prior vaccination、subjective perception proxies 编目，并标记回顾期与同期变量 | 两个 dataset 究竟有多少直接贴近决策的输入信息；高预测力是否来自决策后的信息 |
| **D2. Feature-family ablation** | `profile only` → `+ event cues` → `+ perception indicators`，在同一 strong ML 上比较增益 | `62 features`，分别去掉 vaccination history / healthcare utilization，并评估通过 persona augmentation 加回的结构化描述 | FLARE 提升是否依赖直接感知与事件线索；NHIS 是否主要依靠行为史等 proxy |
| **D3. Local-label consistency** | 在 Train 内做 kNN purity、opposite-label matched pair、feature-family conflict；必要时按事件分别计算 | 复用本周 Correctability / Contradiction Audit，再试 distance weighting 与多个 k | 各 dataset 中“相似特征、相反结果”的比例和程度是否不同 |
| **D4. Learning curve / model convergence** | 比较 ML、FLARE 在不同训练样本比例及不同事件上的表现（在固定 test 下重复抽样） | 使用已有 SFT / XGBoost / TabPFN label-fraction curve 与 fixed Test80 baseline | 是否一边存在明显未利用的学习空间，另一边已在当前 features 下接近方法间收敛 |
| **D5. Perception-alignment test** | 用训练数据中的指标拟合 threat / risk perception；比较真实感知指标、out-of-fold 推断感知与完全不使用感知三种设置 | 当前 62 features 没有相同的独立心理指标时，评估 proxy-derived descriptors 是否对 label 有独立增益 | 区分“直接观测到感知”与“根据 proxies 推断感知”的信息质量差异 |
| **D6. Explanation–self-report alignment** | 在提供开放回答的受访者中，比较模型 explanation 与受访者自述的理由类别、遗漏及无证据推断 | 检查 rationale 的 feature grounding、相反标签样本的一致性；不把 HBM 推断心理当作已验证动机 | 区分“预测准确”和“推理确实与人自己的决策理由一致” |

其中 **D2 + D3** 会直接衔接本周的 feature contradiction；而 **D5 + D6** 专门检验 subjective perception 是否能够被数据校准，以及 reasoning 是否能与个人叙述对齐。具体来说：本周 NHIS 已经看到大量 mixed-label neighborhoods 和一部分 feature-level contrasts，却很难通过单个变量确定个体行为；如果 FLARE 在加入 event / perception features 后，local label consistency 和 ML Accuracy 同时明显改善，那么我们就有更直接的证据认为 **observable decision information 的质量和贴近程度**，而不只是 reasoning structure，解释了两套任务上的差异。

### 7.3.5 两组实验合起来能够得到什么答案？

| F1 Strong ML on FLARE | F2 Dataset diagnosis | 可能的研究解释 / 后续方向 |
|---|---|---|
| Strong ML 与 FLARE 接近或超过 FLARE | FLARE 的 perception / event cues 很强，ML 加入后同样受益 | 原论文 advantage 可能部分来自 baseline 对照不足；重点转向 **data-informed features / tabular learning**，不必默认 LLM reasoning 是主要增益来源 |
| FLARE 在严格同数据对照下仍领先 | FLARE perception indicators 可校准，event cues 充分 | 支持 **theory-guided perception construction + context integration** 仍有增量价值；尝试把可验证的 evidence calibration / routing 引入 NHIS，而非只复制 CoT |
| FLARE 在 rich perception 输入下领先，在 profile-only 输入下优势缩小 | NHIS 缺少相应主观感知变量且 local conflicts 较多 | 说明 reasoning benefit 可能依赖 **dataset 的 decision-relevant observability**；优先找更适合的疫苗调查数据或真正额外的信息来源 |
| Strong ML 与 FLARE 在 FLARE 上都不稳定、跨事件大幅下降 | 事件分布 / 标签比例明显不同 | 不能仅依据单次 within-event Accuracy 得出一般性结论；重点研究 **跨环境泛化与稳定性** |

### 本阶段的直接产出

1. **FLARE Data Manifest：** 开源原始文件、三事件样本数、feature list / codebook；将 direct perception indicators、event cues、post-decision responses、open-ended reason references 分开编目，同时审计预测时点与可复现 splits。
2. **Strong-ML Benchmark Table：** LR、RF、XGBoost、CatBoost、LightGBM（可选 TabPFN）在 combined、within-event、cross-event 设置下的 Accuracy / Macro F1 / Balanced Accuracy 等结果。
3. **FLARE vs NHIS Dataset Comparison：** 把三个核心差异（direct perception、event-specific cues、self-reported reasons）与 feature-family ablation、local-label ambiguity、learning-curve saturation 配对报告。
4. **Interpretation：** 区分 FLARE 论文中可能存在的 *baseline gap*，与其 PADM / perception-based pipeline 真正适合原始 dataset 的部分，为后续是否继续改进 NHIS reasoning、增加 persona / uncertainty、或寻找新数据提供依据。

**当前状态：** 两项 follow-up 实验的设计已明确；新的 strong-ML runs 与跨数据集 diagnostics 尚未执行。本节引用的 FLARE 数字来自 Chen et al. (2025) 的原论文 Table 1、Table 4 和 Appendix A，不代表本项目重新运行后的结果。

---

# References

- Hsieh et al. (2023). *Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes*. Findings of ACL 2023（作为 two-task auxiliary rationale supervision 的方法参考）。
- Chen, R., Wang, C., Sun, Y., Zhao, X., & Xu, S. (2025). *From Perceptions to Decisions: Wildfire Evacuation Decision Prediction with Behavioral Theory-informed LLMs*. ACL 2025, [paper](https://aclanthology.org/2025.acl-long.1438/), [open-source repository](https://github.com/SusuXu-s-Lab/FLARE).
- NHIS 2024 Adult Survey：`SHTFLU12M_A`、本项目冻结 62-feature manifest、Train20/Test80 split。
- 本项目 2026-10-13 Weekly README 与 2026-10-18 meeting transcript（研究问题来源）。
