# 本周进展 — Tabular Foundation Model、Reasoning SFT 与 Loss 诊断

## 1. 上周问题 Follow-up：Tabular Foundation Model 是否真的没有 Supervision？

上周老师重点追问了一个问题：TabPFN / TabICLv2 看起来可以直接输入 tabular matrix 并输出 probability，那么模型到底有没有看到当前任务的 labeled examples？如果完全没有 supervision，它如何知道当前需要预测的 outcome？

这里需要修正之前的表述：

> **Pretrained ICL 不等于 zero-shot，也不等于 no supervision。**

TabPFN-3.5 和 TabICLv2 都属于 tabular in-context learning / tabular foundation model。当前任务中，模型会使用 labeled training rows 作为 context，再预测 test rows；pretrained ICL 的区别在于不需要在当前任务上再做常规 gradient-based parameter update。

```text
Labeled training rows
        ↓
Pretrained Tabular Foundation Model
        ↓
In-context prediction
        ↓
No task-specific parameter update
```

当前实验采用：5% labeled Train + 15% Validation + 80% Test。其中 5% labeled Train 提供当前任务 supervision/context，15% Validation 用于 threshold selection，80% Test（25,656 respondents）只用于最终评估。

因此后续不再使用“zero-shot TabPFN”或“完全无 supervision”的表述。更准确地说：

- **TabPFN-3.5 / TabICLv2 Pretrained ICL**：有当前任务 labeled context，但没有 task-specific parameter update；
- **TabPFN-3.5 Fine-tuned**：进一步使用当前任务 labeled data 更新参数。

---

## 2. Tabular Foundation Model 结果

### 固定 Threshold = 0.50

| Model | Accuracy | Balanced Accuracy | F1 | ROC-AUC | Log Loss |
|---|---:|---:|---:|---:|---:|
| **TabPFN-3.5 — Pretrained ICL** | **0.7671** | **0.7659** | **0.7513** | **0.8464** | 0.4853 |
| TabPFN-3.5 — Fine-tuned | 0.7665 | 0.7654 | 0.7512 | 0.8464 | **0.4852** |
| TabICLv2 — Pretrained ICL | 0.7647 | 0.7630 | 0.7464 | 0.8450 | 0.4884 |

### Validation-selected Threshold

| Model | Threshold | Accuracy | Balanced Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **TabPFN-3.5 — Pretrained ICL** | 0.465 | **0.7682** | **0.7685** | 0.7603 | **0.8464** |
| TabPFN-3.5 — Fine-tuned | 0.485 | 0.7676 | 0.7672 | 0.7561 | 0.8464 |
| TabICLv2 — Pretrained ICL | 0.445 | 0.7665 | 0.7672 | **0.7603** | 0.8450 |

目前 Tabular Foundation Model 的 Accuracy 约在 **0.765–0.768**。另外，TabPFN-3.5 在当前任务上进一步 fine-tune 后并没有获得明显提升。

---

## 3. Teacher-Reasoning SFT：实验设置

当前 Reasoning SFT 使用：

```text
Train20 = 6,414 respondents
Test80  = 25,656 respondents
Input   = 62 NHIS features
```

整体流程：

```text
NHIS Respondent Profile
        ↓
Qwen3.8 Teacher
        ↓
Reasoning + Ground-truth Label
        ↓
Qwen3.5-9B Student
        ↓
LoRA SFT
        ↓
Vaccinated / Not Vaccinated
```

Student 保持之前一致的简单 LoRA 设置：rank=16、alpha=32、dropout=0.05、learning rate=1e-4、2 epochs、micro batch=2、gradient accumulation=16、max length=2048、seed=42。

---

## 4. 本周比较的三种 Reasoning Strategy

### 4.1 General Reasoning

不加入特定理论，让 teacher 自由综合 supporting evidence、opposing evidence、conflicting signals 和 overall integration。

### 4.2 HBM-Light

使用 Health Belief Model 作为 reasoning lens，在有 observable evidence 时考虑 perceived threat、benefits、barriers、self-efficacy 和 cues to action，但不要求机械覆盖所有维度，也不允许仅根据 demographics 强行推断不可观察的心理状态。

### 4.3 HBM-Structured

进一步把 reasoning 固定为：

```text
Behavioral Evidence
        ↓
HBM Interpretation
        ↓
Conflicting Evidence
        ↓
Integration
        ↓
Prediction
```

这里的核心变化不是增加更多 HBM concepts，而是将 evidence → conflict → integration → decision 显式结构化。


为了更直观地说明这一方法，下面给出一个 illustrative example。假设某位 respondent 具有以下 profile：

```text
Age: 68
Multiple chronic conditions
Insurance: Yes
Usual source of care: Yes
Frequent doctor contact
Healthcare access: Good
Other adult vaccination history: No
```

在 **HBM-Light** 中，teacher 可能会比较自由地写：

```text
The respondent is older and has multiple chronic conditions.
They also have insurance and regular healthcare access, which may
increase opportunities for vaccination. However, the lack of other
adult vaccination history may indicate lower vaccination tendency.
Overall, these factors should be considered together.
```

而在 **HBM-Structured** 中，同样的信息会被明确拆分为：

```text
Behavioral Evidence
- The respondent has insurance and a usual source of care.
- Doctor contact is frequent.
- However, other adult vaccines have not been received.

HBM Interpretation
- Good healthcare access suggests relatively low practical access barriers.
- Frequent provider contact provides more opportunities or cues to action.
- No assumption is made about unobserved beliefs or perceived vaccine benefits.

Conflicting Evidence
- Healthcare engagement supports vaccination opportunity.
- Previous adult vaccination behavior provides evidence in the opposite direction.

Integration
- The decision should depend on the balance between strong healthcare
  contact/access signals and the negative vaccination-history signal.

Final Decision
- Produce the vaccination label based on the integrated evidence.
```

这个例子只用于说明 **reasoning structure**，并不是一个实际 respondent 的完整 teacher output。它想强调的是：HBM-Structured 并不是简单加入更多 HBM 概念，而是强制模型先识别 observable evidence，再进行 theory interpretation，随后单独处理 conflicting evidence，最后才完成 integration 和 decision。

因此这一方法真正测试的是：

> **相比自由 reasoning，显式的 evidence → conflict → integration 结构是否更容易让 student 学到有用的 decision pattern？**

每一种 reasoning strategy 都比较两种 teacher mode：

```text
No Thinking:
  enable_thinking = False
  max_new_tokens = 384

Thinking-Low:
  enable_thinking = True
  reasoning_effort = low
  max_new_tokens = 640
```

目前 General / Thinking-Low 仍在补全最后少量 teacher annotations，其余五组已经得到结果。

---

## 5. Reasoning SFT 当前结果

| Method | Accuracy | Balanced Accuracy | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| General / No Thinking | 0.7177 | — | 0.6854 | 0.7849 | — |
| HBM-Light / No Thinking | 0.7204 | 0.7164 | 0.6841 | 0.7870 | 0.7751 |
| HBM-Light / Thinking-Low | 0.7189 | 0.7123 | 0.6634 | 0.7892 | 0.7706 |
| **HBM-Structured / No Thinking** | **0.7443** | **0.7434** | **0.7292** | **0.8231** | **0.7991** |
| HBM-Structured / Thinking-Low | 0.7308 | 0.7272 | 0.6986 | 0.8036 | 0.7833 |
| General / Thinking-Low | Pending | Pending | Pending | Pending | Pending |

目前 reasoning 方法中最好的版本是：

> **HBM-Structured / No Thinking：Accuracy = 0.7443，ROC-AUC = 0.8231。**

---

## 6. 与已有 Baseline 的整体对比

| Model / Method | Setting | Accuracy | ROC-AUC |
|---|---|---:|---:|
| **TabPFN-3.5 — Pretrained ICL** | 5% labeled context, threshold=0.50 | **0.7671** | **0.8464** |
| TabPFN-3.5 — Fine-tuned | 5% labeled train, threshold=0.50 | 0.7665 | 0.8464 |
| TabICLv2 — Pretrained ICL | 5% labeled context, threshold=0.50 | 0.7647 | 0.8450 |
| XGBoost | historical 20% labeled experiment | 0.7587 | — |
| Qwen3.5-9B Label-only SFT | historical 20% labeled experiment | 0.7567 | — |
| **HBM-Structured / No Thinking** | Train20 reasoning + label | **0.7443** | **0.8231** |
| HBM-Structured / Thinking-Low | Train20 reasoning + label | 0.7308 | 0.8036 |
| HBM-Light / No Thinking | Train20 reasoning + label | 0.7204 | 0.7870 |
| HBM-Light / Thinking-Low | Train20 reasoning + label | 0.7189 | 0.7892 |
| General / No Thinking | Train20 reasoning + label | 0.7177 | 0.7849 |

这里有两个 caveat：第一，Tabular Foundation Model 的 labeled-context protocol 与 Qwen SFT 的训练机制不同，因此主要作为 strong baseline；第二，当前 `0.7567` 的 label-only SFT 来自之前的 label-fraction scaling experiment，虽然同样使用 20% labeled data，但 split 未必与当前 Train20/Test80 完全一致。

因此本周正在重新跑一个 **exact same Train20/Test80 split 的 label-only Qwen3.5-9B baseline**，用于后续严格 paired comparison。

---

## 7. 当前结果的初步 Interpretation

首先，HBM 本身并没有明显提升：General / No Thinking = 0.7177，HBM-Light / No Thinking = 0.7204，两者非常接近。

其次，Structured Reasoning 明显更有效。在 No Thinking 下，HBM-Light 从 0.7204 提升到 HBM-Structured 的 0.7443，Accuracy 增加 **2.39 个百分点**；ROC-AUC 从 0.7870 提升到 0.8231，增加 **3.60 个百分点**。这说明目前真正有效的更可能是 **structured evidence organization**，而不是简单增加 theory-related prose。

第三，Thinking-Low 没有带来稳定收益。HBM-Light 从 0.7204 降至 0.7189；HBM-Structured 从 0.7443 降至 0.7308，其中 Structured setting 下降约 **1.35 个百分点 Accuracy** 和 **1.95 个百分点 ROC-AUC**。因此当前结果不支持“reasoning 越长 / thinking 越强越好”。

---

## 8. Reasoning SFT 初步诊断：Loss Dilution

本周进一步分析了 student 实际训练 target 的 token composition。

### 8.1 不是 2048 Sequence Truncation

所有当前 reasoning 数据均满足：

```text
>2048 full sequence = 0%
Target truncated    = 0%
Label truncated     = 0%
Rationale truncated = 0%
```

因此 student `max_length=2048` 不是当前性能下降的主要原因。

### 8.2 Rationale 很长，并且高度接近 Teacher Generation Budget

| Method | Mean Rationale Tokens | P95 |
|---|---:|---:|
| General / No Thinking | 374.6 | 385 |
| HBM-Light / No Thinking | 375.9 | 385 |
| HBM-Structured / No Thinking | 384.1 | 385 |
| HBM-Light / Thinking-Low | 640.6 | 641 |
| HBM-Structured / Thinking-Low | 640.5 | 641 |

No Thinking 的 generation budget 为 384，Thinking-Low 为 640。当前 rationale length 几乎全部集中在这个上限附近，但更长 reasoning 并没有带来更好的 student performance。

### 8.3 Label 在 Token-level Loss 中占比非常低

当前 target 为：

```text
<LABEL> ... </LABEL>
<RATIONALE> ... </RATIONALE>
```

整个 target 使用普通 causal-LM token-level cross entropy。

实际统计：

```text
No Thinking:
Label : Rationale ≈ 1 : 106–109
Label semantic-token share ≈ 0.91%–0.93%

Thinking-Low:
Label : Rationale ≈ 1 : 182
Label semantic-token share ≈ 0.55%
```

因此当前最重要的 hypothesis 是：

> **大部分 optimization budget 被用于学习 rationale generation，而真正与最终 classification objective 直接对应的 label supervision 占比很小。**

换句话说，student 很可能更多在优化：

```text
P(rationale | profile, label)
```

而最终真正需要的是：

```text
P(label | profile)
```

这可能是当前 reasoning SFT 没有超过 label-only SFT 的核心原因之一。

---

## 9. 下一步 1：Exact-split Label-only Baseline + Error Analysis

首先重新跑一个与当前 reasoning pipeline **完全相同 Train20/Test80 split** 的 Qwen3.5-9B label-only LoRA baseline：

```text
Train20 = 6,414
Test80  = 25,656
No teacher
No rationale
Only ground-truth label
```

完成后重点建立四类 case：

```text
1. Both correct
2. Both wrong
3. Label-only correct / Reasoning wrong
4. Reasoning correct / Label-only wrong
```

尤其重点分析哪些 case 原本 label-only 可以预测正确，但加入 reasoning supervision 后反而被改错。后续进一步结合 FP/FN、confidence、boundary cases、feature-conflict patterns、XGBoost / Tabular FM disagreement，以及 dataset-specific noise / local label inconsistency。

这一部分直接对应上周讨论的方向：一边做 error analysis，一边回到 dataset 本身寻找 noise 或现有 training signal 的具体 defect，再针对性设计新的方法。

---

## 10. 下一步 2：Distilling Step-by-Step 式 Multi-task Loss

当前方法相当于：

```text
Profile
→ Label + Long Rationale
→ one token-level CE loss
```

下一步参考 Hsieh et al. 的 **Distilling Step-by-Step**，将 label prediction 与 rationale generation 显式拆成两个任务。

对于同一个输入 `x`：

```text
[label] + x
→ y
```

以及：

```text
[rationale] + x
→ r
```

训练：

\[
L = \alpha L_{\text{label}} + (1-\alpha)L_{\text{rationale}}
\]

官方实现中，`alpha` 就是 multi-task weight，推荐值为：

```text
alpha = 0.5
```

这个设计与当前 concatenated SFT 的关键区别是：

> **rationale 不会仅仅因为 token 数更多，就自然获得远高于 label prediction 的 loss contribution。**

当前计划先复现：

```text
α = 0.50
```

然后继续测试 classification-heavy 设置：

```text
α = 0.75
α = 0.90
```

并使用 `α = 1.00` 作为 rationale-loss-off / label-only 对应参照。

第一轮优先使用目前表现最好的 **HBM-Structured / No Thinking rationale**。核心问题是：

> 如果显式提高 label prediction objective 的权重，同时保留 structured rationale 作为 auxiliary supervision，是否能够恢复甚至超过 label-only SFT？

---

## 11. 当前 Working Hypothesis

目前结果不支持“Reasoning 越多、thinking 越强，student 就越好”。当前更可能的机制是：

```text
Long rationale
        ↓
Token-level loss dominated by rationale generation
        ↓
Classification label signal diluted
        ↓
Prediction performance decreases
```

但 HBM-Structured / No Thinking 又明显优于 General / HBM-Light，因此 reasoning 本身并非完全没有价值，更可能是：

> **structured decision evidence 有价值，但需要重新设计它与 label objective 的训练关系。**

下一阶段最核心的两个实验是：

1. **Exact-split Label-only vs. Structured Reasoning error analysis**；
2. **Distilling Step-by-Step style multi-task loss + alpha ablation**。

最终希望判断：当前 reasoning SFT 的主要问题究竟来自 reasoning 缺少 discriminative information，还是来自现有 loss 对 label 与 rationale 的权重设计不合理。

---

## References

- Jäger et al. (2026). *TabPFN-3.5: Technical Report*. arXiv:2609.17895.
- Qu et al. (2026). *TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model*. arXiv:2602.11139.
- Hsieh et al. (2023). *Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes*. Findings of ACL 2023.
- Official implementation: https://github.com/google-research/distilling-step-by-step
