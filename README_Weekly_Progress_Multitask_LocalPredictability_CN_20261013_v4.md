# 本周进展 — Multi-task Reasoning、Alpha Ablation 与 Error Reframing

## 1. 本周核心问题

上周的结果已经说明：传统的 Reasoning SFT 虽然同时训练 label 与 rationale，但长 rationale 在普通 token-level causal-LM loss 中占据了绝大多数 optimization budget。与此同时，HBM-Structured / No Thinking 又明显优于更自由的 reasoning strategy，说明 structured reasoning 可能并非完全没有价值，问题更可能在于 **reasoning signal 如何进入 classification objective**。

因此本周主要围绕三个问题继续推进：

1. **No Thinking 与 Thinking-Low 到底提供了什么不同的 supervision？**
2. **如果把 label prediction 与 rationale generation 拆成独立任务，并显式控制两部分 loss 的权重，structured rationale 是否能够成为有效的 auxiliary supervision？**
3. **α=0.90 带来的 improvement / hurt 到底发生在哪些 case 上？这些错误中，哪些更像当前 feature space 本身难以区分，哪些才更值得继续做 reasoning-level correction？**

---

# 2. Follow-up：No Thinking vs. Thinking-Low

上次 meeting 中老师追问了一个很重要的问题：

> HBM-Structured 的 No Thinking 与 Thinking-Low 到底有什么区别？Thinking-Low 是否会额外产生一段“中间思维”，还是只是把最终 rationale 写得更长？

重新对 teacher annotations 做逐 case 对齐后，目前最明显的区别并不在最终 label，而在 **rationale 的长度和组织方式**。

以 HBM-Structured 为例：

| Statistic | No Thinking | Thinking-Low |
|---|---:|---:|
| Mean rationale characters | 1,447.4 | 1,755.2 |
| Mean rationale words | 200.8 | 259.5 |
| Mean rationale lines | 13.2 | 42.9 |
| Thinking-Low longer than matched No Thinking | — | 99.75% |

两种 teacher mode 在 matched cases 上的最终 label 基本保持一致，但 Thinking-Low 会产生更长、更展开的 visible rationale；No Thinking 则更接近直接按照我们要求的 HBM structure 输出整理后的 evidence 和 integration。

为了更直观地说明差别，可以看同一个 respondent profile 下两种输出的代表性形式。下面的 excerpt 做了截短，只保留最能体现 style difference 的部分。

### No Thinking：直接进入结构化 evidence → conflict → integration

```text
Behavioral Evidence
- Older age and multiple chronic conditions increase health risk.
- Insurance, a usual source of care, and frequent doctor contact
  indicate strong healthcare access.
- However, there is no history of other adult vaccination.

HBM Interpretation
- Frequent provider contact provides more opportunities for cues to action.
- Good access suggests relatively low practical barriers.

Conflicting Evidence
- Healthcare engagement supports vaccination opportunity.
- Previous adult vaccination behavior points in the opposite direction.

Integration
- Balance the strong healthcare-contact signal against the negative
  prior-vaccination signal and produce the final decision.
```

这里的特点是：模型基本不展示额外的探索过程，而是直接把输入信息整理成我们指定的几个部分，再完成 conflict 和 integration。

### Thinking-Low：同样的 evidence，但会加入更展开的分析过程

```text
Let me analyze the profile step by step.

The respondent is older and has several chronic conditions, so this
could increase perceived susceptibility. They also have insurance and
frequent contact with healthcare providers, which creates more
opportunities for vaccination.

However, the lack of other adult vaccination history points in the
opposite direction.

Wait, this negative vaccination-history signal may be important.
I should not rely only on age and healthcare access. The final decision
needs to weigh the positive access/contact evidence against the
respondent's previous vaccination behavior.

...
Final integration and decision.
```

Thinking-Low 仍然会使用相同的 HBM evidence，但更容易出现：

```text
Let me analyze...
However...
Wait...
I should reconsider...
```

这类展开、回看和 self-correction。也就是说，它不是简单多写几个 HBM concept，而是把原本可以直接压缩成结构化结论的过程展开成更长的 reasoning path。

这也解释了为什么两种 teacher mode 虽然最后 label 基本一致，但提供给 student 的 supervision 并不完全一样：

```text
No Thinking
→ 更短、更直接、更接近固定的 HBM evidence organization

Thinking-Low
→ 更长、更探索式，包含更多 intermediate deliberation
```

但更长的 Thinking-Low supervision 并没有带来更好的 downstream classification。HBM-Structured 的 Accuracy 从 No Thinking 的 0.7443 降到 Thinking-Low 的 0.7308。

因此目前更支持：

> **reasoning 的 compression、structure 与 evidence organization 比 reasoning length 本身更重要。**

---

# 3. 从普通 Reasoning SFT 到 Two-task Multi-task Training

## 3.1 为什么需要改变原来的训练目标？

原来的 Reasoning SFT 把 label 与 rationale 串在同一个 autoregressive target 中：

```text
Respondent Profile
        ↓
<LABEL> y </LABEL>
<RATIONALE> r </RATIONALE>
```

然后对整段 target 使用普通 token-level causal-LM cross entropy。

问题在于，label 与 rationale 的长度极不平衡：

```text
No Thinking:
Label : Rationale ≈ 1 : 106–109

Thinking-Low:
Label : Rationale ≈ 1 : 182
```

因此，虽然从语义上看 label 与 rationale 都属于 supervision，但在普通 token-level loss 中，大部分 optimization budget 实际上用于：

```text
reproduce teacher rationale
```

而不是：

```text
predict vaccination label
```

这会造成训练 objective 与最终 evaluation objective 不匹配：

```text
Training:
大量 token 用于学习 rationale generation

Evaluation:
只关心 vaccinated / not vaccinated 是否预测正确
```

所以这一阶段的核心问题不是“要不要 reasoning”，而是：

> **如何保留 structured rationale 中可能有用的信息，同时不让 rationale generation 淹没 label prediction objective？**

---

## 3.2 Two-task formulation：两个任务彼此独立，但共享同一套 Student 参数

参考 Distilling Step-by-Step 的思路，本周把训练显式拆成两个独立任务。

### Task 1 — Label Prediction

输入：

```text
[label] + respondent profile
```

输出：

```text
vaccinated / not vaccinated
```

对应 loss：

\[
L_{\text{label}}
\]

它直接学习：

\[
P(y\mid x)
\]

也就是当前真正需要的 classification task。

### Task 2 — Rationale Generation

输入：

```text
[rationale] + respondent profile
```

输出：

```text
HBM-Structured rationale
```

对应 loss：

\[
L_{\text{rationale}}
\]

它学习：

\[
P(r\mid x)
\]

这里需要特别强调：

> **Label Prediction 与 Rationale Generation 是两个独立的 supervised tasks。**

它们：

- 使用不同 task prompt；
- target 不同；
- 分别进行 forward pass；
- 但共享同一个 Qwen3.5-9B + LoRA student 参数。

因此 rationale 的作用不是在同一次 inference 中“先想完再决定 label”，而是作为一个 **auxiliary training signal**，通过共享参数影响 student 学到的 representation。

总 loss 为：

\[
L=\alpha L_{\text{label}}+(1-\alpha)L_{\text{rationale}}
\]

其中：

- `α = 0.50`：两个 task 等权；
- `α → 1`：逐渐强调 classification；
- `α = 1.00`：rationale objective 被关闭，对应 label-only endpoint。

所有 multi-task α 实验都使用同一套：

> **HBM-Structured / No Thinking teacher annotations**

因此这一轮 ablation 不再改变 teacher、reasoning style 或 LoRA hyperparameters，主要变化只有：

```text
label objective
vs.
rationale objective
```

的相对权重。

---

## 3.3 为什么当前 DSBS 测试时只输出 Label，而不是 Label + Reasoning？

这是当前方法中非常重要的一点。

DSBS 的两个 task 在训练时彼此独立：

```text
Task 1:
profile → label

Task 2:
profile → rationale
```

因此最终 classification evaluation 时，我们直接调用 **Label Prediction branch**：

```text
profile
   ↓
vaccinated / not vaccinated
```

而不要求模型先生成一长段 rationale。

这样做有三个原因。

第一，最终 benchmark 的目标就是预测：

```text
vaccinated
vs.
not vaccinated
```

因此 classification performance 应该由 `P(y|x)` 直接评估。

第二，在当前 DSBS formulation 中，rationale 是 auxiliary supervision，而不是 test-time decision chain。训练时的 rationale loss 可以通过共享参数改变模型，但测试时再生成 rationale 并不会“回头”改变已经由 label task 得出的 prediction。

第三，只输出 label 可以避免把 improvement 混淆成“测试时使用了额外 generation / compute”。

因此当前 α ablation 回答的是：

> **在 test-time prediction protocol 不变的情况下，加入多少 structured rationale supervision 最有利于学习 label？**

---

## 3.4 当前 Control 与下一步：Reasoning 到底在哪一个环节发挥作用？

目前 `α=0.90` 的 single-run 结果高于 exact-split label-only，但这个结果本身还不能回答：

> **到底是 rationale 的内容真的帮助了 label learning，还是 loss balancing、task formulation，甚至 test-time generation path 带来的差异？**

因此这一阶段不把 `DSBS α=0.90 > Label-only` 直接解释成“reasoning 有效”，而是把后续实验设计成一组 **mechanism controls**，逐步区分 reasoning 可能发挥作用的位置。

### Control A — Balanced Post-thinking：先隔离 Loss Balancing

目前正在补的第一个 control 是 **Balanced Post-thinking**。

#### DSBS / Two-task

训练：

```text
Task A:
profile → label

Task B:
profile → rationale
```

两个独立 task、两个 forward pass：

\[
L=0.9L_{\text{label}}+0.1L_{\text{rationale}}
\]

测试：

```text
profile → label
```

只使用 Label Prediction task 得到 classification prediction。

#### Balanced Post-thinking

训练时不再拆成两个 task，而是保留原来的 joint autoregressive sequence：

```text
profile
   ↓
<LABEL> y </LABEL>
<RATIONALE> r </RATIONALE>
```

模型学习：

\[
P(y,r\mid x)=P(y\mid x)P(r\mid x,y)
\]

但与最早的普通 Reasoning SFT 不同，这里不再使用普通 token-level CE 让 rationale token 数量自动决定 loss contribution，而是显式计算：

\[
L=0.9L_{\text{label}}+0.1L_{\text{rationale}}
\]

测试时同样生成：

```text
LABEL → RATIONALE
```

并使用最前面的 generated label 作为 classification prediction。

因此：

```text
Old Post-thinking
vs.
Balanced Post-thinking
```

主要回答：

> **原始 Reasoning SFT 的下降有多少来自长 rationale 导致的 loss dilution？**

而：

```text
Balanced Post-thinking
vs.
DSBS α=0.90
```

主要回答：

> **在 loss 权重相同的情况下，把 label 与 rationale 拆成两个独立 task 是否还有额外价值？**

---

### Control B — Reasoning-before-Label：测试 Reasoning 是否能直接参与 Label Decision

Balanced Post-thinking 虽然在 test time 会生成 rationale，但顺序仍然是：

```text
LABEL
↓
RATIONALE
```

因此从 autoregressive generation 的方向看，后生成的 rationale **不能反过来改变已经生成的 label**。

所以还需要增加一个更直接的 control：

```text
profile
   ↓
<RATIONALE> r </RATIONALE>
<LABEL> y </LABEL>
```

也就是 **Reasoning-before-Label / Pre-thinking**。

训练时仍然显式平衡 label 与 rationale loss，例如：

\[
L=0.9L_{\text{label}}+0.1L_{\text{rationale}}
\]

但测试时真正执行：

```text
profile
↓
generate rationale
↓
label is generated conditional on the rationale
```

此时模型学习的是：

\[
P(r,y\mid x)=P(r\mid x)P(y\mid x,r)
\]

因此：

```text
Balanced Post-thinking
vs.
Balanced Pre-thinking
```

在尽量保持 rationale source、loss weight 和 student model 一致的情况下，主要回答：

> **当 rationale 真正位于 label 之前、能够成为 label generation 的上下文时，是否会带来额外 prediction benefit？**

如果 Pre-thinking 明显优于 Post-thinking，才更支持：

> **test-time generated reasoning path 本身对 label decision 有额外作用。**

如果两者接近，则更可能说明 rationale 的作用主要发生在 training supervision / shared representation，而不是 inference-time reasoning chain。

---

### 当前实验矩阵

| Setting | Training structure | Loss design | Test-time path | 主要回答的问题 |
|---|---|---|---|---|
| **Label-only** | `profile → label` | label only | `profile → label` | 无 rationale baseline |
| **Old Reasoning SFT** | `LABEL → RATIONALE` joint sequence | ordinary token CE | `LABEL → RATIONALE` | 原始 reasoning SFT |
| **Balanced Post-thinking** | `LABEL → RATIONALE` joint sequence | balanced label/rationale loss | `LABEL → RATIONALE` | **loss balancing 是否足够** |
| **DSBS α=0.90** | Label / Rationale 两个独立 task | balanced 0.9/0.1 | `profile → label` | **task separation 是否有价值** |
| **Balanced Pre-thinking（planned）** | `RATIONALE → LABEL` joint sequence | balanced label/rationale loss | `RATIONALE → LABEL` | **test-time reasoning 是否直接帮助 label** |
下	
因此这一阶段真正希望回答的不是简单的：

> “reasoning 有没有用？”

而是进一步拆成：

```text
1. 是不是只是普通 token CE 的 loss dilution 问题？
2. balanced loss 本身是否已经足够？
3. label / rationale task separation 是否有额外作用？
4. reasoning 只作为 training supervision 是否有效？
5. 当 generated reasoning 真正出现在 label 之前时，是否还能带来额外收益？
```

只有把这些 control 补齐后，才能更准确地判断 **reasoning 究竟在哪一个环节发挥作用**，以及后续应该继续优化 reasoning content、objective design，还是 inference-time reasoning path。

---
# 4. Alpha Ablation Results

为了公平比较，所有 α 实验使用：

```text
Train20 = 6,414
Test80  = 25,656
Teacher rationale = HBM-Structured / No Thinking
Student = Qwen3.5-9B + LoRA
```

Exact-split label-only baseline：

| Method | Accuracy | Balanced Accuracy | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Label-only (`α=1.00`) | 0.7623 | 0.7619 | 0.7503 | 0.8447 | 0.8223 |

Multi-task α sweep：

| α | Label Weight | Rationale Weight | Accuracy | Balanced Accuracy | F1 | ROC-AUC | PR-AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 50% | 50% | 0.7644 | 0.7659 | 0.7619 | 0.8463 | 0.8234 |
| 0.75 | 75% | 25% | 0.7656 | 0.7668 | 0.7617 | **0.8464** | 0.8235 |
| **0.90** | **90%** | **10%** | **0.7670** | **0.7681** | **0.7627** | 0.8462 | 0.8236 |
| 0.92 | 92% | 8% | 0.7639 | 0.7651 | 0.7601 | 0.8438 | 0.8202 |
| 0.95 | 95% | 5% | 0.7647 | 0.7657 | 0.7596 | 0.8458 | **0.8240** |
| 0.97 | 97% | 3% | 0.7620 | 0.7637 | 0.7605 | 0.8433 | 0.8210 |
| 0.99 | 99% | 1% | Pending / unavailable | — | — | — | — |
| 1.00 | 100% | 0% | 0.7623 | 0.7619 | 0.7503 | 0.8447 | 0.8223 |

![Alpha sweep Accuracy](figures/alpha_sweep_accuracy.png)

以 classification metrics 为主，当前最佳 operating point 是：

```text
α = 0.90
```

相对于 exact-split label-only：

```text
Accuracy:
0.7623 → 0.7670
+0.46 percentage points

Balanced Accuracy:
0.7619 → 0.7681
+0.62 percentage points

F1:
0.7503 → 0.7627
+1.23 percentage points
```

这一结果目前只能作为 **preliminary evidence**：

> **在当前 single-run setting 下，低权重 rationale auxiliary objective 与更好的 classification result 同时出现；但在 Balanced Post-thinking、Pre-thinking 以及后续 multi-seed control 完成以前，还不能把这部分差异直接归因于 reasoning content 本身。**

---

# 5. 从旧 Reasoning SFT 到 α=0.90：当前 Working Interpretation

之前同样使用 HBM-Structured / No Thinking rationale，但采用普通 concatenated SFT：

```text
Old concatenated Reasoning SFT:
Accuracy = 0.7443
```

改成 two-task balanced objective 后：

```text
DSBS α=0.90:
Accuracy = 0.7670
```

因此当前结果更支持：

```text
Structured rationale
        ↓
contains potentially useful auxiliary information
        ↓
but ordinary token CE overweights rationale reproduction
        ↓
explicit label-heavy weighting protects classification objective
        ↓
small but measurable gain over label-only
```

所以目前不再把问题简单解释成：

> “Reasoning 没有用。”

但也暂时不把 `α=0.90` 的结果解释成：

> “Reasoning 已经被证明有效。”

更准确的当前表述是：

> **不同 reasoning formulation 的结果存在差异，但这种差异究竟来自 loss balancing、task separation、rationale supervision 本身，还是 reasoning 真正进入 label decision，还需要后续 control 才能区分。**

---

# 6. 为什么选择 α=0.90 做后续分析？

α=0.90 同时取得：

- 当前最高 Accuracy；
- 当前最高 Balanced Accuracy；
- 当前最高 F1；
- 高于 exact-split label-only；
- 同时仍然保留 10% rationale objective。

因此当前 error analysis 暂时以：

```text
Label-only
vs.
DSBS α=0.90
```

作为一个 **working comparison**，因为 α=0.90 是目前 single-run classification metrics 最好的 setting。

但这一比较的目的不是提前认定 `α=0.90` 已经是稳定更优的方法，而是先观察：

> **加入这一类 rationale auxiliary objective 后，哪些 prediction 发生了变化；这些变化是 substantive 的，还是主要集中在 decision boundary 附近。**

最终是否把这些变化解释成稳定的 reasoning effect，还需要结合 Section 3.4 的 control 与 repeated-run validation。

---

# 7. Error Analysis：四类 Prediction Transition

按照 label-only 与 DSBS α=0.90 是否预测正确，将 Test80 的 25,656 cases 分成四类：

| Group | Definition | N | Percentage |
|---|---|---:|---:|
| A | Both correct | 18,985 | 74.00% |
| B | Both wrong | 5,406 | 21.07% |
| C | Label-only wrong → DSBS correct | **692** | **2.70%** |
| D | Label-only correct → DSBS wrong | **573** | **2.23%** |

因此：

```text
HELP = 692
HURT = 573
Net gain = +119 correct predictions
```

HELP 中：

```text
FN → TP = 539
FP → TN = 153
```

HURT 中：

```text
TP → FN = 95
TN → FP = 478
```

所以 α=0.90 的主要方向性变化是：

> **更多地把 negative prediction 推向 positive。**

但仅仅看到 `692 HELP / 573 HURT` 还不够，因为这并不能告诉我们这些被改变的 case 本身是不是容易区分。

因此下一步重新从 **local feature-label predictability** 的角度分析全部 5,979 个 DSBS errors。

---

# 8. 重新定义 Error：哪些 case 在当前 Feature Space 中本身就很难区分？

## 8.1 为什么需要这一层分析？

如果两个 respondents 的 observable profile 非常相似，但一个实际 vaccinated、另一个实际 not vaccinated，那么仅依靠当前输入 features，模型本身就缺少稳定 signal。

这种情况下：

```text
prediction error
```

不一定代表：

```text
reasoning failure
```

它也可能来自：

- 当前 62 个 features 没有覆盖真正决定 vaccination behavior 的信息；
- individual preference / provider recommendation 等因素没有被观察；
- survey / label noise；
- Train20 样本覆盖不足；
- 相似 profile 本身就存在不同 outcome。

因此在继续寻找 reasoning failure pattern 之前，需要先问：

> **这个错误在当前 feature representation 下，到底有多“可预测”？**

---

## 8.2 Neighbor reference：为什么使用 Train20，而不是 Test80？

主分析对每一个错误的 Test80 respondent，只在：

```text
Train20 = 6,414 respondents
```

中寻找 nearest neighbors。

不使用其他 Test80 respondents 的真实 label 来定义主要 error category。

原因是：

> **我们想知道的是：模型训练时实际能够看到的 labeled data 中，是否已经存在足够清楚的 local signal。**

如果直接使用 Test80 labels 来决定哪些 case “应该可修”，会把 evaluation labels 引入方法选择。

因此：

```text
Primary analysis:
Test case → nearest neighbors in Train20
```

Test80 leave-one-out neighborhood 只作为 secondary / ex-post robustness check，不参与主要 taxonomy。

---

## 8.3 Distance 如何定义？

模型最终使用 62 个 NHIS features。

这些 feature 大多数属于 coded categorical / ordinal survey variables，因此这里不直接把 `1/2/3` 当连续距离。

处理方式是：

```text
62 model features
      ↓
categorical one-hot encoding
      ↓
cosine distance
      ↓
20 nearest Train20 neighbors
```

记 test respondent \(i\) 的 20 个最近训练邻居为：

\[
N_{20}(i)
\]

---

## 8.4 True-label Support

最关键的指标是：

\[
S_i=
\frac{1}{20}
\sum_{j\in N_{20}(i)}
\mathbf{1}(y_j=y_i)
\]

其中：

- \(y_i\)：当前 Test80 respondent 的真实 label；
- \(y_j\)：Train20 neighbor 的真实 label。

也就是说：

> **20 个最相似训练样本中，有多少比例与这个 test case 的真实 outcome 相同？**

例如：

```text
True label = vaccinated

20 nearest Train20:
18 vaccinated
 2 not vaccinated

True-label support = 18 / 20 = 0.90
```

如果模型仍然预测 `not vaccinated`，那么当前 observed features 中其实已经存在很强的正确方向 signal，这种错误更值得继续研究。

反过来：

```text
True label = vaccinated

20 nearest Train20:
 2 vaccinated
18 not vaccinated

True-label support = 2 / 20 = 0.10
```

虽然模型预测 `not vaccinated` 在 benchmark 上仍然是错误，但从当前 62 个 observable features 看，这个 respondent 本身更像典型的 `not vaccinated` profile。

这种 case 更可能受到未观测因素影响，而不一定能通过“更好的 reasoning”解决。

因为 \(k=20\)，support 以 0.05 为一个步长：

```text
1/20 = 0.05
2/20 = 0.10
...
20/20 = 1.00
```

---

## 8.5 Local Purity 与 True-label Support 的区别

还需要区分两个指标。

### Local Purity

\[
P_i=\max(p_i,1-p_i)
\]

其中 \(p_i\) 是邻域 vaccinated proportion。

它回答：

> **这个 neighborhood 自己到底一致不一致？**

例如：

```text
10 vaccinated
10 not vaccinated

Purity = 0.50
```

表示 neighborhood 非常 ambiguous。

但如果：

```text
18 not vaccinated
2 vaccinated
```

Purity = 0.90，说明 neighborhood 非常一致。

### True-label Support

它回答：

> **这个一致的 neighborhood 是否支持当前 respondent 自己的真实 label？**

因此：

```text
True = vaccinated
Neighbors = 18 not vaccinated / 2 vaccinated
```

会得到：

```text
Purity = 0.90
True-label support = 0.10
```

这个 case **不是 ambiguous**。

相反，它是一个：

> **local-label outlier：周围人高度一致，但这个 respondent 自己的 outcome 与周围主流方向相反。**

这个区别对解释 error 非常重要。

---

## 8.6 Sparse / weak Train20 support

还有一类 case 不能仅根据 neighbor labels 判断，因为 Train20 中本身就找不到足够相似的人。

为了识别这类 case，先对 Train20 自身做 leave-one-out nearest-neighbor analysis，得到训练数据正常的 local-distance distribution。

如果某个 Test80 error 的平均 20-NN distance 超过 Train20 leave-one-out distribution 的 90th percentile，则定义为：

> **Sparse / weak Train20 support**

这类 case 更准确的解释是：

```text
training coverage insufficient
```

而不是：

```text
intrinsically unpredictable
```

如果以后增加训练数据，它们可能自然改善。

---

# 9. 5,979 个 α=0.90 Errors 的 Preliminary Taxonomy

| Category | N | Share of Errors | Definition |
|---|---:|---:|---|
| **Strong local outlier** | **912** | **15.25%** | ≤4/20 neighbors support true label |
| **Low-support / likely unexplained** | **1,629** | **27.25%** | 5–7/20 support true label |
| **Locally ambiguous** | **2,503** | **41.86%** | approximately 8–12/20; neighborhood itself is mixed |
| **Moderately correctable** | **282** | **4.72%** | 13–15/20 support true label |
| **Strict correctable** | **32** | **0.54%** | ≥16/20 support true label |
| **Sparse / weak Train20 support** | **621** | **10.39%** | neighborhood too far from normal Train20 support |

这六类加起来正好是全部：

```text
5,979 DSBS errors
```

---

## 9.1 Strong Local Outlier

定义：

```text
True-label support <= 0.20
= 最多 4 / 20 neighbors 支持真实 label
```

例如：

```text
True label = vaccinated

20 nearest Train20:
 2 vaccinated
18 not vaccinated
```

这说明：

> **当前 respondent 的 observed profile 与真实 outcome 在局部训练分布中非常 atypical。**

这种 error 不应该直接归因于 reasoning。

---

## 9.2 Low-support / Likely Unexplained

定义：

```text
True-label support = 0.25–0.35
= 5–7 / 20 neighbors 支持真实 label
```

它不像 strong outlier 那么极端，但 neighborhood 仍然明显更支持相反方向。

因此这类 case 很可能也受到当前 feature set 未观察到的信息影响。

---

## 9.3 Locally Ambiguous

大致对应：

```text
True-label support = 0.40–0.60
```

例如：

```text
10 vs. 10
11 vs. 9
12 vs. 8
```

这时相似 respondents 的 outcome 本身就在两边大量存在。

这是最典型的：

> **current-feature ambiguity**

即仅靠当前 62 个 observed features 很难稳定区分。

---

## 9.4 Moderately Correctable

定义：

```text
True-label support = 0.65–0.75
= 13–15 / 20 neighbors 支持真实 label
```

这时 local training data 已经开始出现明显 majority signal。

如果模型仍然预测错误，这类 case 更值得检查：

- 是否漏掉关键 evidence；
- 是否错误加权 feature；
- 是否发生 conflict-resolution error；
- rationale 是否与 prediction 不一致。

---

## 9.5 Strict Correctable

定义：

```text
True-label support >= 0.80
= 至少 16 / 20 neighbors 支持真实 label
```

同时必须处于正常 Train20 support 范围内，而不是 sparse/OOD。

这是最干净的一批：

> **observable feature neighborhood 已经强烈支持真实 outcome，但模型仍然预测错误。**

当前共有：

```text
32 cases
```

---

## 9.6 为什么不直接说其他 5,947 个都是“不可修”？

因为 local-neighbor analysis 只是一个 **empirical predictability diagnostic**，不是 Bayes error 的证明。

当前数据仍然有几个限制：

1. Train20 只有 6,414 个 respondents，高维 feature space 本身较 sparse；
2. 62 个 features 并不包含 vaccination decision 的所有真实 determinants；
3. one-hot + cosine 默认各 feature 贡献相对均等，但现实中不同 feature 的 predictive importance 不同；
4. NHIS label / self-report 本身也可能包含 measurement noise；
5. nearest-neighbor consistency 只能描述当前 observed representation 下的 local predictability。

因此这里更准确的语言是：

```text
low local predictability
local-label atypical
sparse-support
potentially correctable
```

而不是：

```text
fixable
vs.
impossible
```

---

# 10. Local Predictability 与整个 Test80 Accuracy 的关系

为了验证这个指标不是人为构造出来的 arbitrary score，我们进一步把全部 Test80 respondents 按 Train20 local label purity 分组。

结果呈现非常清楚的单调关系：

| Train20 Local Label Purity | Test80 N | DSBS α=0.90 Accuracy |
|---|---:|---:|
| 0.50–0.60 | 7,688 | **64.14%** |
| 0.60–0.70 | 5,751 | **72.74%** |
| 0.70–0.80 | 5,286 | **80.97%** |
| 0.80–0.90 | 4,684 | **88.96%** |
| 0.90–1.00 | 2,247 | **94.17%** |

也就是说：

> **训练邻域的 label 越一致，这个 Test80 respondent 就越容易预测。**

因此 local predictability 确实捕捉到了当前任务中的一部分实际 difficulty。

这也进一步说明：

> 当前 0.76–0.77 左右的整体 performance 中，相当一部分 residual error 与 feature-space ambiguity / atypical outcomes 有关，而不应该全部解释成 reasoning quality 问题。

---

# 11. 放宽 Potentially Correctable 的标准

`≥80%` 是一个非常严格的定义。

在 20-NN 下：

```text
>= 80%
= 至少 16 / 20 neighbors 支持 true label
```

因此只得到 32 个 case。

为了避免 threshold 过于严格，可以进一步做分层：

| Definition | Neighbor support | Error N | Share of 5,979 |
|---|---:|---:|---:|
| High-confidence correctable | ≥16/20 = 80% | 32 | 0.54% |
| **Strong candidate correctable** | **≥14/20 = 70%** | **144** | **2.41%** |
| Exploratory correctable | ≥13/20 = 65% | 314 | 5.25% |

因此这三个 threshold 更适合被理解为 **preliminary screening levels**：

```text
Primary screening:
>=70% true-label support
N = 144

High-confidence screening:
>=80%
N = 32

Exploratory screening:
>=65%
N = 314
```

在真正进入 reasoning-pattern analysis 之前，还需要继续确认这些 case 在更严格的 similarity / distance-weighted / multi-scale neighborhood definition 下是否仍然稳定。换句话说，`144 / 32` 目前不是最终“可修 case”集合，而是下一步需要重点验证的候选集合。

---

# 12. 573 个 DSBS-HURT 几乎全部是 Boundary Flip

这一结果会重新改变我们对 `D_dsbs_hurt` 的解释。

Recall：

```text
D_dsbs_hurt
=
Label-only correct
→
DSBS α=0.90 wrong
```

共有：

```text
573 cases
```

如果只看 correct/wrong，很容易解释成：

> rationale supervision 把 573 个原本正确的 case “reason 错了”。

但重新检查两个模型的 prediction probability 后发现：

```text
571 / 573 = 99.65%
```

同时满足：

```text
Label-only margin <= 0.02
DSBS margin       <= 0.02
```

这里定义：

\[
\text{margin}=|p(\text{vaccinated})-0.5|
\]

例如：

```text
Label-only:
p = 0.497
→ not vaccinated
→ correct

DSBS:
p = 0.503
→ vaccinated
→ wrong
```

这和：

```text
0.20 → 0.80
```

这种 substantive prediction reversal 完全不同。

进一步看 probability shift：

```text
Mean absolute probability shift = 0.0086

325 / 573 = 56.72%
|Δp| <= 0.01

11 / 573 = 1.92%
|Δp| >= 0.02

Maximum |Δp| = 0.0283
```

因此更准确的 interpretation 是：

> **绝大多数 HURT 并不是 rationale supervision 把一个原本非常明确的 prediction 大幅推向错误方向，而是在本来就非常接近 0.5 的 boundary cases 上产生了很小的 score shift，最终刚好跨过 threshold。**

---

## 12.1 HELP 也是同样的 Boundary Phenomenon

`C_dsbs_help`：

```text
Label-only wrong
→
DSBS correct
```

共有：

```text
692 cases
```

而：

```text
692 / 692
```

全部同样处于：

```text
Label-only margin <= 0.02
DSBS margin       <= 0.02
```

因此 α=0.90 的主要 improvement 更准确地说是：

> **重新调整 decision boundary 附近 case 的 score / ranking。**

而不是：

> **突然学会了一批原来完全不会判断的新规则。**

---

# 13. 下一阶段：先定位 Reasoning 在哪里起作用，再决定哪些 Case 值得做 Pattern Analysis

当前最重要的调整是：**暂时不继续对全部 5,979 个错误做 generalized reasoning-pattern audit。**

原因是目前已经看到：

```text
prediction error
≠
reasoning failure
```

一个 error 可能来自：

```text
1. 当前 observed features 本身高度 ambiguous
2. respondent 是 local-label atypical / outlier
3. Train20 对该区域覆盖不足
4. 模型只是在 decision boundary 上发生很小的 score flip
5. 模型确实没有利用好当前 features 中已经存在的 signal
```

如果把这些来源完全不同的 case 混在一起找“固定 reasoning pattern”，很容易把一个高频的 rationale 写作习惯误认为 prediction error 的原因。

因此后续分析分成两个阶段。

---

## 13.1 Stage 1 — 先回答 Reasoning 到底有没有稳定作用，以及作用发生在哪里

第一阶段优先完成 Section 3.4 中的 mechanism controls：

```text
Label-only
↓
Old Post-thinking
↓
Balanced Post-thinking
↓
DSBS / Two-task
↓
Balanced Pre-thinking
↓
(optional) Rationale-conditioned Two-stage
```

主要回答三个层次的问题。

### A. Training objective 层面

```text
Old Post
vs.
Balanced Post
```

判断：

> 原始 Reasoning SFT 的下降主要是不是 loss dilution。

### B. Task formulation 层面

```text
Balanced Post
vs.
DSBS
```

判断：

> 把 label 与 rationale 分成独立 task 是否本身带来额外收益。

### C. Test-time reasoning path 层面

```text
Balanced Post
vs.
Balanced Pre

以及必要时：
DSBS
vs.
Rationale-conditioned Two-stage
```

判断：

> generated rationale 如果真正位于 label 之前，并成为 label predictor 的输入，是否能够进一步改变或改善 prediction。

这一步完成以后，才可以更准确地区分：

```text
reasoning as auxiliary training supervision
vs.
reasoning as task regularization / representation learning
vs.
reasoning as an actual inference-time decision path
```

同时，由于目前不同 setting 与 label-only 的 overall accuracy gap 都比较小，后续还需要通过 repeated seeds / reproducibility check 判断 observed gain 是否稳定，而不是直接把单次运行的差异解释成 method effect。

---

## 13.2 Stage 2 — 不是所有 Error 都适合寻找固定 Reasoning Pattern

只有当 Stage 1 证明某一种 reasoning formulation 存在相对稳定的效果后，才继续做 case-level mechanism analysis。

这里的关键不是：

> “从所有错误里找最常见的 reasoning pattern。”

而是先判断：

> **哪些 case 的错误有足够证据可以被解释为 model / reasoning failure。**

### 不优先做 Pattern Analysis 的 Case

#### 1. Locally ambiguous

例如：

```text
相似 training profiles:
10 vaccinated
10 not vaccinated
```

这种情况下 observed features 本身就没有给出稳定方向。

模型预测错不能自动说明 reasoning 错；模型预测对也不能自动说明 reasoning 对。

因此这类 case 更适合用于研究：

```text
uncertainty
prediction stability
```

而不是寻找一个确定的“错误 reasoning rule”。

#### 2. Local-label atypical / outlier

例如：

```text
True = vaccinated

very similar training profiles:
18 not vaccinated
2 vaccinated
```

此时模型预测 `not vaccinated` 虽然在 benchmark 上是 error，但当前 observed features 本身其实更支持 opposite label。

如果强制模型为这个 rare observed outcome 写一个完整 rationale，很可能只是在 **rationalize an atypical outcome**，而不是暴露一个真正可以被修正的 reasoning failure。

#### 3. Sparse / weak-support

如果附近没有足够相似的 Train20 respondents，就无法可靠判断：

```text
model missed a clear pattern
```

还是：

```text
training data simply did not cover this region
```

#### 4. Pure boundary flips

目前绝大多数 HELP / HURT 都发生在非常小的 score shift 附近。

对于：

```text
0.497 → 0.503
```

这类 case，不应优先解释为：

> reasoning 学到了 / 破坏了一个明确的新 decision rule。

---

## 13.3 真正值得做 Reasoning Pattern Analysis 的候选 Case

更值得研究的是：

> **当前 feature space 已经相对稳定地支持 observed true label，但某一种模型 / reasoning formulation 仍然稳定地预测错误，或者稳定地相对 baseline 产生 improvement / harm。**

理想的 reasoning-diagnostic case 应尽量满足：

```text
1. sufficient local training support
2. genuinely similar neighbors
3. neighborhood direction stable across multiple scales / similarity definitions
4. observed true label 得到明显 local support
5. prediction error / improvement 在 repeated runs 中相对稳定
6. 不是仅仅由于极小 threshold crossing 形成的偶然 correct / wrong
```

目前的：

```text
>=70% true-label support
N = 144
```

以及：

```text
>=80%
N = 32
```

只能作为 **preliminary candidate sets**。

后续会先用更严格的 local similarity definition 验证这些 case，而不是直接把它们定义为“可修 reasoning errors”。

---

## 13.4 如果某种 Reasoning Method 确实有效，Case Analysis 应该看什么？

假设后续 control 显示某一种 formulation，例如：

```text
Balanced Pre-thinking
```

或：

```text
DSBS
```

相对于 label-only 在 repeated runs 中存在稳定收益，那么 case analysis 就不再泛化到全部错误，而重点比较三类 case：

```text
A. Reasoning-stable HELP
   baseline 稳定错误
   reasoning method 稳定正确

B. Reasoning-stable HURT
   baseline 稳定正确
   reasoning method 稳定错误

C. Stable shared errors with strong data support
   两种方法都错
   但 local feature evidence 明显支持 true label
```

然后才去比较：

```text
Teacher rationale
vs.
Student rationale
vs.
actual label prediction
```

并检查是否存在稳定重复的 pattern，例如：

```text
- teacher 已经错误解释 / 错误加权 evidence
- student 在 distillation 中丢失某类 evidence
- 某类 conflict 被系统性错误解决
- rationale 支持 true label，但 label branch 持续给出 opposite prediction
- 某一类 feature 被稳定地过度依赖或忽略
```

只有当某个 pattern：

```text
在 feature-supported cases 中重复出现
+
与 prediction change 方向稳定相关
+
在不同 seed / case 中能够复现
```

才值得进一步设计：

```text
consistency loss
grounding constraint
conflict-aware supervision
feature-aware rationale
```

来做针对性 correction。

---

## 13.5 当前下一步的研究逻辑

因此下一阶段不再把研究路线写成：

```text
发现 error
↓
生成 reasoning
↓
找一个错误 pattern
↓
继续修改 prompt / SFT
```

而是：

```text
Step 1
先完成 reasoning mechanism controls
↓
确定 reasoning 是否有稳定收益，以及收益发生在 training / task / inference 哪一层

Step 2
重新定义哪些 case 在当前 data representation 下真正有清晰 signal
↓
排除 ambiguous / atypical / sparse / unstable cases

Step 3
只在 reasoning-diagnostic cases 上做 teacher–student pattern analysis
↓
判断是否存在稳定、可重复、可以被干预的 reasoning failure

Step 4
只有在找到这样的 pattern 后
↓
再设计新的 reasoning objective / supervision method
```

当前最重要的目标不是继续扩大 reasoning generation，而是：

> **先定位 reasoning 的真实作用机制，再把 case-level analysis 限制到那些确实存在可学习、可改变 signal 的样本上。**

这样后续无论结果是哪一种，都可以得到更清楚的结论：

```text
如果 reasoning gain 不稳定：
→ 不应继续把 overall accuracy difference 解释成 reasoning improvement。

如果 gain 来自 loss balancing：
→ 重点是 objective design，而不是 reasoning content。

如果 gain 来自 task separation：
→ 重点是 auxiliary multi-task representation learning。

如果 reasoning-before-label 明显更好：
→ 才有更直接证据支持 inference-time reasoning path 的价值。

如果只在少量 feature-supported cases 上存在稳定 HELP / HURT：
→ reasoning correction 应该 targeted，而不是对所有 samples universal applied。
```

---


# 14. 两个需要进一步确认的核心研究问题

在完成当前 multi-task、boundary analysis 与 local-predictability analysis 后，下一阶段最需要确认的并不是继续增加新的 reasoning variant，而是先回答两个更基础的问题：**当前 performance bottleneck 到底来自数据本身，还是来自我们还没有找到合适的方法；以及在这个任务里，theory-guided reasoning 相对于 data-driven tabular learning 到底应该承担什么作用。**

---

## 14.1 Open Question 1：什么时候可以判断当前 Feature Representation 已经接近可达到的上限？

目前不同类型的强模型最终都集中在大约 `76%–77%` 的 Accuracy 区间，而 local-predictability analysis 又显示，相当一部分 residual errors 位于：

```text
Locally ambiguous
Local-label atypical / outlier
Sparse / weak-support
```

这些 case 的共同特点是：

> **仅依靠当前 62 个 observable features，training data 本身并没有给出非常稳定的 label separation。**

因此现在需要区分两种完全不同的可能性：

```text
Possibility A
我们还没有找到更好的 model / reasoning formulation
→ 继续优化方法仍然可能获得明显 improvement

Possibility B
当前 62-feature representation 本身已经缺少足够的 individual-level information
→ 不同强模型自然会逐渐收敛到相近的 performance
→ 继续更换 reasoning format 只能带来很小的 boundary-level variation
```

之前基于 nearest-neighbor / local consistency 得到过一个大约 `78%` 左右的粗略 ceiling proxy，但它只能作为参考，不能单独证明真正的上限。

因此这里希望进一步确认的是：

> **我们应该使用什么 empirical evidence，才能比较有把握地判断“继续优化模型”与“增加 feature / 更换 dataset”哪一个更值得？**

一个可能的判断路径是同时看：

```text
1. Strong-model convergence
   XGBoost / TabPFN / TabICL / Qwen SFT 是否长期集中在同一 performance range

2. Learning-curve saturation
   增加 labeled training data 后，performance 是否仍然快速增长

3. Local predictability
   residual errors 是否主要集中在 ambiguous / outlier / sparse region

4. Feature expansion
   加入新的、非泄漏的 vaccination-relevant features 后，
   ambiguity 与 performance 是否明显改善

5. Repeated-run stability
   当前 0.x percentage-point 的 method differences
   是否明显大于 random-seed variation
```

核心问题是：

> **在继续设计更多 reasoning method 之前，是否应该先建立一个更系统的“dataset / feature headroom”判断标准？如果 feature expansion 后不同模型仍然集中在同一区间，那么是否应该把主要问题从 model design 转向 information limitation；反过来，如果加入新的 feature 后 ambiguity 明显下降，则说明当前瓶颈更多来自 representation，而不是模型本身。**

---

## 14.2 Open Question 2：在这个任务中，Theory-Guided Reasoning 相对于 ML 的真正作用应该是什么？

当前 HBM-Structured reasoning 可以把 respondent profile 分解到若干理论维度，例如：

```text
susceptibility
barriers
cues to action
healthcare access
previous preventive behavior
...
```

当这些 evidence 方向一致时，最终 decision 相对容易。

真正困难的是：

```text
some evidence → supports vaccination
other evidence → supports non-vaccination
```

此时最终仍然必须回答：

> **哪一组 evidence 应该权重更高？这些 evidence 之间应该怎样 integration？**

而 HBM 本身主要提供的是：

```text
what dimensions to consider
```

它并没有提供：

```text
which dimension is more predictive for this dataset
how much each signal should be weighted
how conflicting dimensions map to the final individual behavior
```

这就带来一个更核心的问题：

> **如果最终 prediction 的关键仍然是学习 feature importance、feature interaction 和 case-specific weighting，那么这一部分是否本来就是 XGBoost、TabPFN 等 data-driven model 更擅长的事情？在这种情况下，LLM reasoning 相对于 tabular ML 的额外价值到底在哪里？**

这也意味着，后续可能不应该继续让 LLM 单独完成：

```text
Feature extraction
↓
Theory interpretation
↓
Evidence weighting
↓
Conflict resolution
↓
Final label
```

而是考虑把不同模块的作用分开。

一个更接近当前问题的 formulation 可能是：

```text
Theory
→ 定义 decision structure / meaningful dimensions

Data / ML
→ 学习哪些 observed features 真正重要
→ 学习不同 signal 的 empirical strength / interaction

LLM
→ 在有冲突的 evidence 之间进行 structured integration
→ 将 case-specific evidence 映射到一个可解释的 decision path
```

这里与之前讨论 FLARE 时最值得借鉴的地方类似：**theory 本身不一定直接给出 individual behavior，关键是如何把 theory、data-driven mapping 和最终 decision 连接起来。**

因此下一步需要导师帮助确认的不是简单的：

> “HBM 要不要换成另一个 theory？”

而是：

> **我们是否应该把 reasoning 的研究问题重新定义成：LLM 能不能在 data 已经给出 evidence strength 的情况下，更好地完成 conflict integration 和 individualized decision？**

如果答案是肯定的，那么后续可以进一步比较：

```text
A. HBM-only reasoning
   Theory structure only

B. Process / stage-based reasoning
   更明确地表示 decision stage / transition

C. Data-grounded reasoning
   Theory structure
   +
   empirical feature importance / local evidence strength

D. Pure tabular model
   直接由 data 学 prediction
```

真正需要验证的是：

> **LLM 是否在“有足够 signal、但 evidence 相互冲突”的 case 上提供了 ML 之外的增量价值，而不是让 LLM 去重复学习 tabular model 已经更擅长的 feature weighting。**

---

## 14.3 这两个问题如何影响下一步实验选择

因此后续实验顺序可以先由这两个问题决定：

```text
Question 1:
当前 dataset / feature representation 还有多少 headroom？
        ↓
决定是否优先继续 model innovation，
还是先扩充 feature / dataset

Question 2:
LLM reasoning 的 comparative role 到底是什么？
        ↓
决定后续 reasoning 应该继续使用
theory-only supervision，
还是转向 data-grounded / conflict-aware / stage-based reasoning
```

如果这两个问题没有先回答清楚，继续增加新的 reasoning prompt 或 SFT variant 很容易变成：

```text
不断尝试不同 formulation
↓
performance 仍然集中在相近区间
↓
很难判断是 method 不够好，
还是当前 observable information 本身已经接近极限
```

所以当前更希望先明确：

> **我们应该先证明“哪里还有可学习的信息”，再决定“reasoning 应该如何利用这些信息”。**

---


## References

- Hsieh et al. (2023). *Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes*. Findings of ACL 2023.
- Official implementation: `google-research/distilling-step-by-step`
