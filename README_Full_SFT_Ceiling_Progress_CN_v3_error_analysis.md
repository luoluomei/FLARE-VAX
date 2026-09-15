# NHIS Influenza Vaccination Prediction：Full SFT Baseline 与 Empirical Ceiling 分析

> 当前版本用于阶段性向导师汇报。  
> 本 README 主要总结最初的 **60% train / 40% test full-run**：数据划分、Qwen3.5-9B LoRA SFT 设置、XGBoost 对照、经验性 ceiling 分析，以及当前实验中遇到的工程问题。  
> 后续 1%–5% low-data / compute-matched 实验暂不在本版本展开。

---

## 1. 研究任务

当前任务是使用 **NHIS 2024 Sample Adult** 中的 respondent-level 信息，预测：

> **该 respondent 在过去 12 个月内是否接种过 influenza vaccine。**

二分类目标：

- `vaccinated = 1`：vaccinated
- `vaccinated = 0`：not vaccinated

清洗后的完整数据集：

- **N = 32,070**
- **69 columns**
- `vaccinated = 0`：16,864
- `vaccinated = 1`：15,206
- overall vaccination rate ≈ **47.42%**

因此该数据集不属于高度类别不平衡问题。

---

# 2. 固定的 Train/Test Split

为了让 SFT 与传统 ML 模型能够在完全相同的数据上公平比较，我们首先固定一次 train/test split，之后所有 full-run 模型都使用该划分。

## 2.1 Split 设置

采用：

- **60% train**
- **40% test**
- stratified split by `vaccinated`
- shuffle = True
- random seed = **42**

对应样本量：

```text
Total N = 32,070
Train N = 19,242
Test N  = 12,828
```

具体标签分布：

| Split | N | Not vaccinated | Vaccinated | Vaccination rate |
|---|---:|---:|---:|---:|
| Train | 19,242 | 10,118 | 9,124 | 47.42% |
| Test | 12,828 | 6,746 | 6,082 | 47.41% |
| Total | 32,070 | 16,864 | 15,206 | 47.42% |

可以看到，stratified split 后 train/test 的 outcome distribution 基本完全一致。

---

## 2.2 模型输入变量

清洗数据一共包含 69 个 columns。

其中排除以下 7 个字段：

| Variable | Exclusion reason |
|---|---|
| `vaccinated` | prediction target |
| `SHTFLU12M_A` | 原始 influenza vaccination target，存在直接 label leakage |
| `HHX` | respondent identifier |
| `PPSU` | survey design identifier |
| `PSTRAT` | survey design identifier |
| `WTFA_A` | survey analysis weight |
| `SRVY_YR` | constant survey year |

因此最终使用：

```text
62 model features
```

Qwen SFT 与 XGBoost **严格使用相同的 62 个 feature**。

---

# 3. Qwen3.5-9B SFT Baseline

## 3.1 Base model

使用：

```text
Qwen/Qwen3.5-9B
```

训练采用 parameter-efficient fine-tuning：

```text
LoRA
```

没有进行 full-parameter fine-tuning，也没有 quantization。

---

## 3.2 输入格式

每一个 respondent 被转换成一段结构化文本，例如：

```text
Respondent profile (NHIS variable=value):

RATCAT_A=...
HISPALLP_A=...
AGEP_A=...
...
PHSTAT_A=...

Did this respondent receive an influenza vaccine during the past 12 months?

Return only: vaccinated or not vaccinated.
```

当前版本直接使用：

```text
NHIS variable name = raw survey code/value
```

而不是把每一个 survey code 进一步翻译成自然语言含义。

System instruction 要求模型：

- 只使用给定 respondent information；
- 不补充未提供的信息；
- 最终只返回：
  - `vaccinated`
  - `not vaccinated`

---

# 4. SFT Training Objective

训练并不是让模型重新生成整个 prompt。

我们采用 **response-only SFT**：

- prompt / respondent profile token 不计算 loss；
- 只有 assistant 最后的正确 label token 计算 loss。

例如真实 label 为 vaccinated：

```text
Prompt:
Respondent profile ...
...

Assistant:
vaccinated
```

训练目标只作用于：

```text
vaccinated
```

对应的 assistant response tokens。

这样训练目标更加直接对应当前 binary classification task。

---

# 5. LoRA 设置

LoRA 参数：

| Parameter | Value |
|---|---:|
| LoRA rank `r` | **16** |
| LoRA alpha | **32** |
| LoRA dropout | **0.05** |
| Bias | none |
| Task type | causal LM |

LoRA target modules：

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
in_proj_qkv
in_proj_z
in_proj_b
in_proj_a
out_proj
```

训练结束后保存：

```text
final_adapter/
```

因此实际保存的是：

> Qwen3.5-9B base model + trained LoRA adapter

而不是重新保存一整套 9B base weights。

---

# 6. Full SFT Training Hyperparameters

正式 full run 使用全部：

```text
19,242 training respondents
```

主要训练设置：

| Parameter | Value |
|---|---:|
| Base model | Qwen3.5-9B |
| Training rows | **19,242** |
| Test rows | **12,828** |
| Features | **62** |
| Epochs | **2** |
| Learning rate | **1e-4** |
| Micro batch size | **4** |
| Gradient accumulation | **8** |
| Effective batch size | **32** |
| Max sequence length | **1536** |
| Precision | **BF16** |
| Quantization | none |
| Optimizer | fused AdamW |
| LR scheduler | cosine |
| Warmup | **3%** |
| Weight decay | **0.01** |
| Max grad norm | **1.0** |
| Gradient checkpointing | enabled |
| Random seed | **42** |

训练过程中使用 dynamic padding。

---

# 7. SFT Inference / Binary Decision

测试时不让模型自由生成长文本，而是固定比较两个候选答案：

```text
not vaccinated
vaccinated
```

对于每一个候选答案，计算该 answer token 序列的平均 log-likelihood。

可以简单写成：

```text
candidate_score
= candidate answer 中所有 token log-probability 的平均值
```

即对于某个 candidate：

```text
score(candidate)
= mean log P(candidate token | respondent profile + previous candidate tokens)
```

最终 prediction 取两个 candidate 中 score 更高的一个：

```text
prediction
= argmax {
    score("vaccinated"),
    score("not vaccinated")
  }
```

同时，将两个 candidate score 做二项 softmax，构造一个相对的 vaccinated score：

```text
P_pair(vaccinated)
=
exp(score_vaccinated)
/
[
  exp(score_vaccinated)
  +
  exp(score_not_vaccinated)
]
```

需要强调：

> 这里的 `P_pair(vaccinated)` 是两个 candidate likelihood 的相对归一化值，并没有经过专门的 probability calibration。

因此不能直接把它解释成严格校准后的：

```text
真实个体 influenza vaccination probability
```

---

# 8. XGBoost Baseline

为了判断 LLM fine-tuning 是否真正具有 predictive advantage，我们使用完全相同的：

- 19,242 train respondents
- 12,828 test respondents
- 62 features

训练 XGBoost。

XGBoost 设置：

| Parameter | Value |
|---|---:|
| `n_estimators` | 600 |
| `max_depth` | 5 |
| `learning_rate` | 0.05 |
| `min_child_weight` | 2 |
| `subsample` | 0.85 |
| `colsample_bytree` | 0.85 |
| `reg_alpha` | 0 |
| `reg_lambda` | 1 |
| objective | binary logistic |
| eval metric | log loss |
| tree method | hist |
| seed | 42 |

因此 SFT 与 XGBoost 的 comparison 是严格的：

```text
same train set
same test set
same 62 features
```

---

# 9. Full Test Results

测试集：

```text
N_test = 12,828
```

最终结果：

| Metric | Qwen3.5-9B SFT | XGBoost |
|---|---:|---:|
| Accuracy | **0.76715** | 0.76575 |
| Balanced Accuracy | **0.76849** | 0.76541 |
| Precision | 0.73558 | **0.74996** |
| Recall | **0.79448** | 0.75896 |
| F1 | **0.76389** | 0.75443 |
| ROC-AUC | **0.85002** | 0.84729 |
| Log Loss | 0.59516 | **0.48636** |

换成百分比：

```text
Qwen3.5-9B SFT Accuracy = 76.715%
XGBoost Accuracy         = 76.575%
```

Accuracy difference：

```text
76.715% - 76.575% = +0.140 percentage point
```

因此 full-data SFT 与 XGBoost 的 hard-label predictive performance 基本处于同一水平。

---

# 10. Paired Case-Level Comparison

因为两个模型测试的是完全相同的 12,828 respondents，可以做 paired comparison。

结果：

```text
SFT correct / XGBoost wrong = 482
XGBoost correct / SFT wrong = 464
```

因此：

```text
482 - 464 = 18
```

即 SFT 总共只比 XGBoost 多预测正确 18 个 respondent。

McNemar exact test：

```text
p ≈ 0.580
```

因此当前 full-run 下，没有证据表明 Qwen3.5-9B SFT 在 accuracy 上显著优于 XGBoost。

更合适的结论是：

> **SFT and XGBoost are approximately tied in final classification performance.**

同时：

```text
SFT ROC-AUC = 0.8500
XGB ROC-AUC = 0.8473
```

SFT 略高；但：

```text
SFT Log Loss = 0.5952
XGB Log Loss = 0.4864
```

XGBoost 明显更好。

因此当前结果说明：

> SFT 的 ranking / discrimination ability 已经达到强 tabular ML baseline 的水平，但当前 candidate-likelihood probability 的 calibration quality 仍明显弱于 XGBoost。

---


## 10.1 SFT 与 XGBoost 的错误是否相同？

虽然两个模型的总体 Accuracy 非常接近，但更值得关注的是它们是否在**同一批 respondent** 上犯错。

完整 12,828 个 test cases 可以分成四类：

| Case-level outcome | N | Test-set share |
|---|---:|---:|
| SFT correct + XGBoost correct | **9,359** | **72.96%** |
| SFT correct + XGBoost wrong | **482** | **3.76%** |
| SFT wrong + XGBoost correct | **464** | **3.62%** |
| SFT wrong + XGBoost wrong | **2,523** | **19.67%** |

因此：

```text
Model prediction agreement = 92.63%

SFT total errors = 2,987
XGBoost total errors = 3,005

Shared errors = 2,523
```

也就是说：

```text
84.47% of SFT errors are also XGBoost errors
83.96% of XGBoost errors are also SFT errors
```

如果把两个模型的 error sets 看作两个集合，其 error-set Jaccard overlap 大约为：

```text
72.73%
```

这说明：

> **两个模型的大多数错误并不是完全不同的 model-specific failure，而是集中在相当大的一批共同困难 cases 上。**

这与前面的 empirical ceiling 分析方向一致：如果两个结构完全不同的模型（LLM SFT 与 gradient-boosted trees）在大多数错误样本上仍然同时失败，那么其中相当一部分困难很可能来自当前 62 个 observable features 对个体 vaccination behavior 的区分能力有限。

但这里需要谨慎：

> shared error 并不能直接证明某个 case 是“不可预测”的；它只是说明该 case 对当前两种模型都困难。

---

## 10.2 两个模型犯错的方向并不一样

虽然 SFT 和 XGBoost 的 overall accuracy 基本持平，但 confusion matrix 显示它们有明显不同的 decision tendency。

### Qwen3.5-9B SFT

```text
True Negative  = 5,009
False Positive = 1,737
False Negative = 1,250
True Positive  = 4,832
```

对应：

```text
Sensitivity / Recall = 79.45%
Specificity          = 74.25%

False Positive Rate  = 25.75%
False Negative Rate  = 20.55%

Predicted vaccinated rate = 51.21%
```

### XGBoost

```text
True Negative  = 5,207
False Positive = 1,539
False Negative = 1,466
True Positive  = 4,616
```

对应：

```text
Sensitivity / Recall = 75.90%
Specificity          = 77.19%

False Positive Rate  = 22.81%
False Negative Rate  = 24.10%

Predicted vaccinated rate = 47.98%
```

因此两个模型虽然 Accuracy 相似，但它们采用了不同的 error trade-off：

> **SFT 更倾向于预测 vaccinated，因此 Recall 更高，但 False Positive 更多。**

而：

> **XGBoost 更保守地预测 vaccinated，因此 Specificity / Precision 更高，但会漏掉更多真实 vaccinated respondents。**

这个差异也可以从两种模型各自的 error composition 看出来：

| Error type | SFT | XGBoost |
|---|---:|---:|
| False Positive | 1,737 (**58.15% of its errors**) | 1,539 (**51.21%**) |
| False Negative | 1,250 (**41.85%**) | 1,466 (**48.79%**) |

---

## 10.3 在两个模型发生分歧时，错误方向更加明显

两个模型一共有：

```text
946
```

个 test respondents 给出了不同的 class prediction。

因为这是 binary classification，所以在 prediction 不一致时，一定有一个模型正确、另一个模型错误。

### 对真实 vaccinated respondents

在模型分歧的 vaccinated cases 中：

```text
SFT correct / XGBoost wrong = 321
XGBoost correct / SFT wrong = 105
```

也就是说：

> 当真实 respondent 是 vaccinated、而两个模型意见不一致时，SFT 大约有 **75%** 的情况下是正确的一方。

### 对真实 not-vaccinated respondents

在模型分歧的 not-vaccinated cases 中：

```text
SFT correct / XGBoost wrong = 161
XGBoost correct / SFT wrong = 359
```

也就是说：

> 当真实 respondent 是 not vaccinated、而两个模型意见不一致时，XGBoost 大约有 **69%** 的情况下是正确的一方。

因此 model-specific error 有非常明确的方向：

```text
SFT-specific weakness:
更容易把真实 not vaccinated respondent 判成 vaccinated

XGBoost-specific weakness:
更容易把真实 vaccinated respondent 判成 not vaccinated
```

更加具体地说：

```text
SFT wrong / XGBoost correct cases = 464

其中：
359 / 464 = 77.4%
是 SFT false positives
```

而：

```text
SFT correct / XGBoost wrong cases = 482

其中：
321 / 482 = 66.6%
是 XGBoost false negatives
```

因此两个模型并不是简单地“随机在不同样本上犯错”，而是表现出不同的 classification bias。

---

## 10.4 Model-specific disagreements 主要发生在 decision boundary 附近

我们进一步比较 prediction confidence。

这里需要注意：

- SFT confidence 来自两个 candidate likelihood 的相对归一化；
- XGBoost probability 来自其自身 binary classifier；
- 两者不是同一种 calibrated probability，因此**不能直接横向比较绝对数值**。

但是可以分别在每个模型内部比较 correct vs wrong cases。

### SFT

```text
Mean confidence when correct = 0.589
Mean confidence when wrong   = 0.548
```

### XGBoost

```text
Mean confidence when correct = 0.811
Mean confidence when wrong   = 0.696
```

因此两个模型在错误样本上的 confidence 都明显下降。

更重要的是，在只有一个模型正确的 disagreement cases 中，prediction 通常更加接近 decision boundary：

```text
SFT-only-correct cases:
SFT mean confidence ≈ 0.517
XGB mean confidence ≈ 0.573

XGB-only-correct cases:
SFT mean confidence ≈ 0.514
XGB mean confidence ≈ 0.583
```

这表明：

> **两个模型之间的 model-specific advantage 主要出现在本来就比较接近分类边界的 respondents 上。**

而在两个模型都预测错误的 2,523 个 shared-error cases 中：

```text
SFT mean confidence ≈ 0.555
XGBoost mean confidence ≈ 0.719
```

尤其对 XGBoost 来说，共同错误并不总是低 confidence error。

因此一部分 shared failures 更可能属于：

> 模型根据 observable profile 得到了一个较稳定但与真实 individual outcome 相反的 pattern。

这与 health behavior prediction 中可能存在的 profile-outcome conflict 是一致的。

---

## 10.5 哪些变量与错误最相关？

为了进一步判断两个模型是否在不同类型 respondent 上犯错，我们将：

```text
model error = 1
model correct = 0
```

作为 binary indicator，并把每一个输入 feature 当作 categorical variable，计算 feature 与 error indicator 的 **Cramér's V**。

这里的目的只是做 diagnostic association：

> Cramér's V 较大表示该 feature 的不同取值下 error rate 差异更明显。

它不能被解释成 causal importance。

结果显示，SFT 和 XGBoost 的 top error-associated features 高度相似。

| Feature | SFT error Cramér's V | XGBoost error Cramér's V |
|---|---:|---:|
| `SHTCVD19NM2_A` | **0.148** | **0.146** |
| `SHTCVD191_A` | **0.142** | **0.137** |
| `LASTDR_A` | 0.076 | 0.078 |
| `SHTSHINGL1_A` | 0.071 | 0.065 |
| `HINOTYR_A` | 0.071 | 0.065 |
| `NOTCOV_A` | 0.070 | 0.063 |
| `RSNHICOST_A` | 0.069 | 0.063 |
| `SHTPNUEV_A` | 0.060 | 0.054 |
| `RXDL12M_A` | 0.058 | 0.068 |
| `HICOV_A` | 0.058 | 0.053 |

从变量类型上看，较明显的 error association 主要集中在：

```text
previous vaccination / vaccine-history variables
healthcare utilization variables
insurance / healthcare-access variables
```

特别是 COVID vaccination-history variables：

```text
SHTCVD19NM2_A
SHTCVD191_A
```

对两个模型都是最明显的 error-associated features。

更重要的是：

> **两个模型的 top error-associated feature ranking 非常相似。**

因此当前结果并不支持：

> “LLM 与 XGBoost 各自存在完全不同的单一 feature blind spot。”

相反，它更支持：

> **两种模型在相似的 behavioral / healthcare profile 区域都更容易遇到困难。**

这也是 shared errors 占两个模型全部错误约 84% 的另一个佐证。

---

## 10.6 一个简单的年龄分组例子

为了给 error profile 一个更加直观的例子，我们把 test respondents 简单按年龄分为四组。

| Age group | N | Observed vaccination rate | SFT error rate | XGBoost error rate |
|---|---:|---:|---:|---:|
| 18–34 | 2,579 | 30.32% | 24.35% | 24.74% |
| 35–49 | 2,833 | 34.27% | 24.04% | 24.00% |
| 50–64 | 2,985 | 44.36% | **25.53%** | **25.09%** |
| 65+ | 4,431 | 67.82% | **20.67%** | **21.17%** |

在这个简单分组下：

- 50–64 岁附近的 respondents 对两个模型都相对困难；
- 65+ group 的 error rate 反而更低。

同时 SFT 在每个年龄段的 predicted-vaccinated rate 都高于 XGBoost，例如：

```text
50–64:
SFT predicted vaccinated = 48.24%
XGBoost                  = 44.52%

65+:
SFT predicted vaccinated = 77.07%
XGBoost                  = 73.91%
```

这再次与 SFT 整体更加倾向于预测 vaccinated 的 pattern 一致。

需要强调：

> 这些 subgroup statistics 是 descriptive diagnostics，不能直接解释为 age 对 model error 的 causal effect。

---

## 10.7 Error analysis 的整体解释

综合上面的 case-level analysis，可以把两个模型的错误结构总结为三点。

### 1. 大多数错误是 shared errors，而不是 model-specific errors

```text
Shared wrong cases = 2,523
≈ 84% of each model's errors
```

这说明大量困难 respondents 对两种非常不同的 modeling approach 都同样困难。

这支持进一步研究：

```text
information-limited cases
profile-outcome conflict
intrinsic ambiguity
```

而不是把所有错误都归因于模型 capacity 不够。

### 2. 两种模型仍然具有互补性

虽然大部分错误重合，但仍然存在：

```text
482 cases: SFT correct / XGB wrong
464 cases: XGB correct / SFT wrong
```

因此两个模型并不是完全等价。

如果构造一个不可直接实现的 oracle union：

```text
只要两个模型中至少一个预测正确，
就认为该 case 可以被 oracle 正确选择
```

那么 accuracy 可以达到：

```text
80.33%
```

这说明：

> 当前 empirical ceiling diagnostic 并不是严格 hard bound，同时也说明两个模型捕捉到了部分互补 signal。

### 3. 互补性具有方向，而不是随机互补

SFT 的主要优势在：

```text
减少 false negatives
提高 vaccinated recall
```

XGBoost 的主要优势在：

```text
减少 false positives
提高 specificity / precision
```

因此后续如果考虑：

- model ensemble；
- confidence-aware routing；
- selective prediction；
- calibration；

更合理的方向并不是简单 averaging，而是利用两种模型在 error direction 上的互补结构。

---


# 11. 为什么进一步分析 Dataset Ceiling

当 SFT 和 XGBoost 都稳定在约：

```text
76%–77%
```

时，一个关键问题是：

> 模型性能没有继续提高，到底是模型仍然没有学好，还是当前 62 个 observable features 本身只能支持有限程度的个体预测？

因此我们进一步估计：

> **在当前 feature representation 下，这个数据集可以支持的大致 empirical information ceiling。**

这里的 ceiling 不是严格数学意义上的绝对上限，而是一个：

```text
empirical / diagnostic upper-performance reference
```

主要用于判断：

> 当前模型距离数据本身可支持的 predictive information 还有多远。

---

# 12. Empirical Ceiling：1-Nearest-Neighbor / Cover–Hart Diagnostic

## 12.1 基本思想

如果两个 respondent 在当前 62 个 observable features 上非常相似，但 vaccination outcome 却经常不同，那么说明：

> 即使模型继续变复杂，仅根据当前 observable profile，也可能仍然无法完全区分这些个体。

因此我们采用 nearest-neighbor diagnostic。

对于每一个 test respondent：

1. 在 training set 中寻找 feature profile 最相似的 respondent；
2. 使用该 nearest training respondent 的 vaccination label 预测 test respondent；
3. 计算 1-NN error。

距离采用 62-feature profile 的 Hamming-style mismatch：

```text
distance(i, j)
=
两个 respondent 在 62 个 feature 中不相同的 feature 数
/
62
```

例如：

```text
如果有 10 / 62 个 feature 不一致：

distance = 10 / 62 ≈ 0.161
```

missing value 作为一个独立 category 处理。

---

## 12.2 Cover–Hart relation

经典的二分类 1-NN 结果可以写成：

```text
Bayes error
≤
asymptotic 1-NN error
≤
2 × Bayes error × (1 - Bayes error)
```

记：

```text
R*     = Bayes error
R_1NN  = asymptotic 1-NN error
```

则关系为：

```text
R* ≤ R_1NN ≤ 2 × R* × (1 - R*)
```

为了得到一个直观的 diagnostic scale，我们把 observed 1-NN error 近似代入右侧关系：

```text
R_1NN ≈ 2 × R* × (1 - R*)
```

解这个二次方程后：

```text
R*
≈
[1 - sqrt(1 - 2 × R_1NN)] / 2
```

然后将 error 转换为 accuracy：

```text
Diagnostic Bayes accuracy
≈
1 - R*
```

需要强调：

> 这里是为了获得一个 empirical diagnostic scale，并不是声称 observed finite-sample 1-NN error 就严格等于 asymptotic 1-NN error。

---

# 13. Ceiling Diagnostic Result

我们之前的 1-NN / sample-size-growth 分析给出的 practical empirical ceiling 大致位于：

```text
78.3%–79.2%
```

代表性结果中：

```text
1-NN accuracy ≈ 67%
1-NN error    ≈ 33%
```

将这一误差代入上面的 Cover–Hart diagnostic inversion 后，对应的 Bayes-accuracy scale 大约为：

```text
≈ 79%
```

同时，我们还观察了随着 reference-set size 增加，nearest-neighbor diagnostic 的变化。

增长外推后的结果大致稳定在：

```text
≈ 78.3%–78.7%
```

因此当前阶段采用：

```text
Empirical information-ceiling diagnostic range
≈ 78.3%–79.2%
```

作为当前 62-feature representation 的一个 practical reference。

---

# 14. Current Models vs Empirical Ceiling

当前主要结果：

```text
Majority-class baseline ≈ 52.59%

XGBoost Accuracy        = 76.575%
Qwen3.5-9B SFT Accuracy = 76.715%

Empirical ceiling
diagnostic              ≈ 78.3%–79.2%
```

因此 Qwen SFT 距离该 empirical ceiling diagnostic 大约还有：

```text
Lower end:
78.3% - 76.715% = 1.585 pp

Upper end:
79.2% - 76.715% = 2.485 pp
```

即：

```text
Qwen SFT is approximately
1.6–2.5 percentage points
below the empirical ceiling diagnostic.
```

XGBoost 则大约低：

```text
1.7–2.6 percentage points
```

---

## 14.1 相对于 majority baseline 的“已恢复提升”

如果以 majority baseline：

```text
≈ 52.59%
```

作为最简单参考，那么 SFT 已经获得的 improvement 为：

```text
76.715% - 52.588%
= 24.127 pp
```

如果采用 ceiling range 下端 78.3%：

```text
Maximum diagnostic improvement
= 78.3% - 52.588%
= 25.712 pp

Recovered fraction
= 24.127 / 25.712
≈ 93.8%
```

如果采用 ceiling range 上端 79.2%：

```text
Maximum diagnostic improvement
= 79.2% - 52.588%
= 26.612 pp

Recovered fraction
= 24.127 / 26.612
≈ 90.7%
```

因此可以粗略理解为：

> 当前 full-data SFT 已经恢复了相对于 majority baseline、由该 empirical ceiling diagnostic 所暗示的潜在提升中的约 **91%–94%**。

这里同样只是一个 diagnostic interpretation，而不是严格的理论比例。

---

# 16. 当前 Full-Run 阶段的核心结论

## Finding 1 — SFT 已达到强 tabular ML baseline 水平

Qwen3.5-9B 经过 LoRA SFT 后：

```text
Accuracy = 76.715%
```

XGBoost：

```text
Accuracy = 76.575%
```

两者基本持平。

因此当前结果不是：

> “LLM 显著优于 XGBoost。”

而是：

> **经过 supervised adaptation 后，Qwen3.5-9B 可以在这一 structured health-behavior prediction task 上达到强 tabular ML baseline 的水平。**

---

## Finding 2 — 两个模型已经接近当前数据的信息饱和区

当前参考：

```text
SFT                  = 76.715%
XGBoost              = 76.575%
Empirical diagnostic = 78.3%–79.2%
```

这说明继续仅仅：

- 增加 LoRA rank；
- 多训练几个 epoch；
- 换更大的 LLM；

可能只能带来有限的 hard-label accuracy improvement。

---

## Finding 3 — Remaining errors 不完全是“模型不够强”

如果当前 62 个 observable features 本身不能充分区分两类 respondent，那么：

> 模型复杂度增加无法自动解决缺失信息问题。

因此后续研究需要进一步区分：

```text
model-limited errors
information-limited errors
highly ambiguous cases
```

不能把所有 prediction error 都视为同一种 failure。

---

## Finding 4 — SFT discrimination 较强，但 probability calibration 仍有明显空间

当前：

```text
SFT AUC = 0.8500
XGB AUC = 0.8473
```

SFT 略高。

但：

```text
SFT Log Loss = 0.5952
XGB Log Loss = 0.4864
```

XGBoost 明显更好。

因此当前结果暗示：

> Qwen SFT 已经学习到较强的 respondent ranking / discrimination signal，但当前 pairwise candidate likelihood 仍不能视作充分校准的 behavioral probability。

---


## Finding 5 — 两个模型的大多数错误高度重合，但存在方向性的互补

case-level analysis 显示：

```text
Shared wrong cases = 2,523
```

约占两个模型各自全部 errors 的：

```text
84%
```

因此当前 remaining errors 中存在很强的共同困难部分。

与此同时：

```text
SFT 更容易 false positive
XGBoost 更容易 false negative
```

说明两个模型仍然捕捉到了不同的 decision tendency。

因此后续分析应同时关注：

```text
shared error → information / ambiguity limitation
model-specific error → model / decision-boundary limitation
```

而不能仅根据 overall Accuracy 判断两个模型“完全一样”。

---

# 17. 当前阶段的整体 Interpretation

目前 full-data experiment 可以理解为建立了一个：

```text
FULL-INFORMATION / FULL-SUPERVISION REFERENCE
```

即：

```text
62 features
19,242 training respondents
Qwen3.5-9B LoRA SFT

Test Accuracy = 76.715%
```

同时：

```text
XGBoost Test Accuracy = 76.575%
```

而当前 feature representation 的 empirical information-ceiling diagnostic 大约是：

```text
78%–79%
```

因此后续研究的重点不应该只剩下：

> “怎样把 76.7% 再提高 0.3%？”

更重要的问题是：

1. **LLM 需要多少 supervision 才能接近 full-data performance？**
2. **哪些 training samples 对 SFT 真正具有高 marginal value？**
3. **哪些错误属于模型没有利用好已有信息？**
4. **哪些错误来自 respondent information 本身不足？**
5. **增加更 proximal / behavioral information 是否比增加更多 training respondents 更有效？**
6. **在接近 accuracy saturation 后，calibration、uncertainty 与 selective prediction 是否成为更合理的优化目标？**

这些问题构成了后续 low-data scaling、sample-value analysis 和 behavior-aware fine-tuning 的基础。

---

# 18. 当前工程困难：FLA 加速模块未成功启用

本轮实验中一个比较实际的工程困难是 **FLA（Flash Linear Attention）加速路径没有成功、稳定地启用**。

原本希望使用 Qwen3.5 对应的 FLA / fast-kernel implementation，以降低训练与推理时间。

但在 ASU GPU 环境中，FLA 相关依赖与 kernel 安装没有稳定完成，因此最终用于上述正式结果的运行方式是：

```text
Qwen3.5 standard/default Transformers implementation
```

而没有使用 FLA fast-kernel acceleration。

普通 attention backend 使用的是：

```text
SDPA
(Scaled Dot-Product Attention)
```

因此可以把当前运行方式理解为：

```text
Qwen3.5 standard Transformers path
+ no FLA acceleration
```

这不会改变：

- train/test split；
- training labels；
- LoRA 设置；
- SFT objective；
- inference candidate scoring；
- 最终 evaluation protocol。

主要影响的是：

```text
training / inference speed
```


---

