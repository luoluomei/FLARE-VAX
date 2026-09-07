# FLARE-VAX / LLM Vaccination Prediction：阶段性研究总结与下一步计划

> **目的**：整理上周讨论后的核心发现。当前重点已经从“继续堆更复杂的 prompting / correction module”逐步转向一个更基础的问题：  
> **LLM 在什么信息条件下能够做好个体行为预测？当真正与行为相关的近端心理信息缺失时，LLM 能否利用预训练知识与推理补足这部分信息？**

---

## 0. 当前最重要的结论

目前两个数据集给出了一个比较一致的信号：

1. **存在可提取的预测信号，但 LLM 不一定能充分利用它。**  
   在 Dataset 1（NHIS）中，同一组 observable features 给监督式 ML 后，可以达到明显高于 zero-shot / few-shot LLM 的表现。因此这里确实存在一个“可达到的经验表现区域”。严格来说，它不是理论 information ceiling，而更适合称为 **empirical supervised reference**。

2. **输入信息的类型比单纯增加 prompting complexity 更关键。**  
   Dataset 2 中，只给 5 个 demographic features 时，多个 LLM 的 Balanced Accuracy 只有约 0.56–0.57；加入 17 个与 COVID risk、vaccine belief、safety、trust、experience、cue 等更相关的问题后，Balanced Accuracy 上升到约 0.80–0.82，强模型已经接近 supervised reference。

3. **因此值得研究的不只是“LLM 能不能预测”，而是“LLM 什么时候能预测”。**  
   Behavioral science 本身区分了 distal background information 与 proximal behavioral states。对普通统计/ML而言，proximal variables 更接近 outcome，本来就更容易预测；但对 LLM，我们有额外期待：LLM 可能利用预训练的社会、健康与行为知识，从 distal profile 中先推断未观察到的 behavioral state，再完成预测。

当前最自然的研究问题因此变成：

\[
D/H/G
\xrightarrow{\text{LLM reasoning}}
\hat{B}
\rightarrow
\hat{Y}
\]

其中：

- \(D\)：demographics；
- \(H\)：health / healthcare background；
- \(G\)：general beliefs / general decision orientation；
- \(B\)：与当前行为直接相关的 proximal behavioral / psychological states；
- \(Y\)：actual vaccination behavior。

---

# 1. Dataset 1：NHIS 2024 流感疫苗接种

## 1.1 数据与任务

Dataset 1 使用 **2024 National Health Interview Survey (NHIS) Sample Adult**。

预测目标：

```text
SHTFLU12M_A:
过去 12 个月是否接种流感疫苗
```

原始开发数据约包含 **32,629 名 respondents、630 个变量**。经过 feature-policy 筛选后，我们主要保留两个版本：

| Version | N | Baseline features | 主要区别 |
|---|---:|---:|---|
| V4 | 32,132 | 75 | 保留 COVID-19、pneumonia、shingles、hepatitis-A 等其他 vaccination history |
| V5 | 32,130 | 67 | 去除全部 non-target vaccination-history variables |

NHIS 的优势是 observable background 非常丰富，包括 demographics、health status、chronic conditions、insurance / affordability、healthcare access、utilization、digital-health engagement 等。

但它有一个非常重要的限制：

> **NHIS 没有直接测量 vaccination-specific psychological states。**

因此我们早期构造的 HBM threat / benefit / barrier / cue / self-efficacy 本质上是根据 observable variables 构造的 **theory-guided proxies**，而不是真实 respondent-level psychometric measurements。

---

## 1.2 上周讨论的“上限”：更准确地说是 empirical supervised reference

我们首先用同一组 observable features 训练传统 supervised models，目的是回答：

> **这些输入本身到底包含多少可以被监督模型提取出来的 outcome signal？**

结果如下：

| Version | Model | Accuracy | ROC-AUC | F1 |
|---|---|---:|---:|---:|
| V4 | Gradient Boosting | 0.7630 | 0.8444 | 0.7506 |
| V4 | Logistic Regression | 0.7604 | 0.8398 | 0.7484 |
| V4 | Random Forest | 0.7617 | 0.8377 | 0.7471 |
| V4 | **XGBoost** | **0.7647** | **0.8452** | **0.7519** |
| V5 | Gradient Boosting | 0.6867 | 0.7553 | 0.6691 |
| V5 | Logistic Regression | 0.6766 | 0.7433 | 0.6640 |
| V5 | Random Forest | 0.6829 | 0.7507 | 0.6660 |
| V5 | **XGBoost** | **0.6871** | **0.7570** | **0.6697** |

因此当前更谨慎的表述是：

```text
V4 empirical supervised reference:
Accuracy ≈ 76.47%, ROC-AUC ≈ 0.845

V5 empirical supervised reference:
Accuracy ≈ 68.71%, ROC-AUC ≈ 0.757
```

这里 **不能把 XGBoost 称为理论 ceiling**：它只是我们测试过的 supervised models 中的经验参考。但它至少说明，同一组输入中存在一部分 zero-shot LLM 没有完全利用到的可预测信号。

同时，V4 明显高于 V5，也说明 **other-vaccination history 本身含有很强的个体行为信息**。

---

## 1.3 Zero-shot / Few-shot：增加 prompting 并没有自动解决问题

以 Llama 4 Scout 为代表：

| Version | Method | Accuracy | Balanced Acc. | ROC-AUC |
|---|---|---:|---:|---:|
| V4 | Zero-shot direct | **0.6339** | **0.6453** | **0.7136** |
| V4 | Random balanced 8-shot | 0.6432 | 0.6297 | 0.6851 |
| V4 | Random 8-shot + generic CoT | 0.5830 | 0.5622 | 0.6048 |
| V4 | Similarity-selected 8-shot | 0.6189 | 0.6225 | 0.6543 |
| V4 | Representative 8-shot | 0.6053 | 0.6211 | 0.6554 |
| V5 | Zero-shot direct | **0.6307** | **0.6367** | **0.6622** |
| V5 | Random balanced 8-shot | 0.5917 | 0.5932 | 0.6229 |
| V5 | Random 8-shot + generic CoT | 0.5461 | 0.5223 | 0.5389 |
| V5 | Similarity-selected 8-shot | 0.5773 | 0.5783 | 0.6051 |
| V5 | Representative 8-shot | 0.6218 | 0.6232 | 0.6251 |

一个很直接的观察是：

> **Few-shot / generic CoT 并没有稳定提升表现，有时反而低于最简单的 zero-shot。**

所以仅仅让模型“多看几个例子”或“多推理几步”，并没有自动补足 respondent-specific information。

---

## 1.4 Theory-guided / population-pattern 方法能够逼近 reference，但仍有 gap

我们随后尝试过 HBM-CoPB、HBM pattern anchor、refined anchor 和 failure-pattern correction：

| Version | Method | Accuracy | Balanced Acc. | ROC-AUC |
|---|---|---:|---:|---:|
| V4 | Zero-shot direct | 0.6339 | 0.6453 | 0.7136 |
| V4 | HBM-CoPB | 0.6726 | 0.6753 | 0.7131 |
| V4 | HBM16 refined anchor | 0.7362 | 0.7342 | 0.8039 |
| V4 | Failure-card + LLM | **0.7412** | **0.7408** | **0.8088** |
| V4 | **XGBoost reference** | **0.7647** | — | **0.8452** |
| V5 | Zero-shot direct | 0.6307 | 0.6367 | 0.6622 |
| V5 | HBM-CoPB | 0.6376 | 0.6378 | 0.6652 |
| V5 | HBM16 refined anchor | 0.6495 | 0.6478 | 0.7046 |
| V5 | Failure-card + LLM | **0.6607** | **0.6619** | **0.7104** |
| V5 | **XGBoost reference** | **0.6871** | — | **0.7570** |

从 Accuracy 看：

```text
V4:
Zero-shot → XGB gap = 13.08 pp
Failure-card + LLM → XGB gap = 2.35 pp

V5:
Zero-shot → XGB gap = 5.64 pp
Failure-card + LLM → XGB gap = 2.64 pp
```

这里需要注意：后面的 HBM16 / failure-card 已经使用 training-side population pattern / empirical anchor，因此不能把它们理解为“LLM 单独推理能力”。它们更重要的价值是说明：

> **通过更好的 representation 和 population-level statistical structure，可以逐渐逼近 empirical reference；但这并不能证明 LLM 真正恢复了 respondent 的心理状态。**

这也是 Dataset 1 的核心局限：**没有真实 \(B\) 可以验证 \(\hat B\)**。

---

# 2. Dataset 2：2022 23-country COVID-19 Vaccine Survey

## 2.1 数据来源

Dataset 2 来源于：

> Lazarus, J. V., Wyka, K., White, T. M., et al. (2023).  
> **A survey of COVID-19 vaccine acceptance across 23 countries in 2022.**  
> *Nature Medicine, 29*, 366–375.  
> DOI: 10.1038/s41591-022-02185-4

原研究在 **2022-06-29 至 2022-07-10** 对 **23 个国家、23,000 名成年人**进行调查，每个国家约 1,000 人。原论文的 raw data 和 analysis code 公开在 Zenodo（DOI: 10.5281/zenodo.6875363）。

---

## 2.2 我们的 2,000 人构造

我们没有直接使用全部 23,000 人。

为了避免某些国家 vaccination status 接近单一类别，我们按 released data 中的 vaccination prevalence 预先筛选 **< 90% vaccinated** 的国家，最终保留 8 个：

```text
France
Ghana
Nigeria
Poland
Russia
South Africa
Sweden
United States
```

然后：

```text
250 respondents / country × 8 countries = N = 2,000
```

> 注：这里是 **每国 250 人**，不是 25 人。

最终样本：

```text
N = 2,000
Vaccinated prevalence = 79.65%
Majority-class Accuracy = 79.65%
```

Outcome：

```text
Q7 / q0007:
是否已经至少接种一剂 COVID-19 vaccine
```

---

## 2.3 当前信息分组

### D5：5 个 demographic features

```text
Country
Age
Gender
Education
Income
```

### B17：17 个更接近 vaccination decision 的 behavioral/contextual questions

当前使用：

```text
Q1-Q6
Q14-Q21
Q23-Q25
```

这些问题包括：

- COVID-19 是否是危险的 health threat；
- vaccination 是否能预防 COVID；
- disease risk 与 vaccine risk 的比较；
- COVID vaccine safety；
- 对政府 vaccine delivery 的信任；
- 对 COVID vaccine science 的信任；
- 对 Long COVID protection 的判断；
- employer / government / university / school vaccine mandate attitudes；
- vaccination proof requirements；
- personal / family COVID experience；
- family loss；
- 对新 vaccine information 的关注；
- healthcare-worker context。

这里不把 B17 全部称为严格 psychometric scale。更准确地说，它们是：

> **behaviorally informative, decision-proximal beliefs / attitudes / experiences / cues / context**

同时，我们排除了直接 vaccination intention、child branch 和一些过于接近 outcome 的问题，以减少 leakage。

---

# 3. Dataset 2：ML 显示了非常清晰的信息层级

## 3.1 Ordinary XGBoost

| Input | Accuracy |
|---|---:|
| **D5 only** | **0.7840 ± 0.0060** |
| **B17 only** | **0.8651 ± 0.0026** |
| **D5 + B17** | **0.8689 ± 0.0022** |

因为 vaccinated prevalence 已经是 79.65%，Accuracy 容易被 majority class 影响，因此 Balanced Accuracy / ROC-AUC 更关键：

| Input | Balanced Accuracy | ROC-AUC |
|---|---:|---:|
| **D5 only** | **0.6332** | **0.691** |
| **B17 only** | **0.8002** | **0.875** |
| **D5 + B17** | **0.8125** | **0.897** |

因此：

```text
D5 only      → BACC 0.633
B17 only     → BACC 0.800
D5 + B17     → BACC 0.813
```

这说明在同一个 respondent-level task 中，真实的 behavioral / attitudinal / experiential information 比 5 个 demographics 提供了明显更多的区分信息。

当前可把：

```text
D5 + B17 XGBoost
Accuracy ≈ 86.89%
Balanced Accuracy ≈ 81.25%
ROC-AUC ≈ 0.897
```

视为 **empirical supervised reference**，而不是理论上限。

---

# 4. Dataset 2：LLM 在 D5 与 D5+B17 下的巨大差异

## 4.1 只有 D5

下面使用当前最新汇总值：

| Model | N | Accuracy | Balanced Accuracy |
|---|---:|---:|---:|
| **Llama 4 Scout** | 2000 | **0.6075** | **0.5734** |
| **GPT-4o-mini** | 2000 | **0.3505** | **0.5621** |
| **GPT-5.6 Luna** | 2000 | **0.5960** | **0.5644** |

三个模型的 Balanced Accuracy 基本都只有：

```text
0.56 – 0.57
```

相比 D5 XGBoost BACC = 0.6332，仍有约 6–7 个百分点的差距。

这说明当前的 demographic-only setting 同时包含两个问题：

1. D5 本身的信息量有限；
2. LLM 也没有完全提取出 D5 中监督 ML 能提取到的全部 signal。

---

## 4.2 D5 + B17

| Model | N | Accuracy | Balanced Accuracy | ROC-AUC |
|---|---:|---:|---:|---:|
| **GPT-5.6 Luna** | 2000 | **0.8620** | **0.8018** | **0.8935** |
| **GPT-4o-mini** | 2000 | **0.7440** | **0.8027** | **0.8852** |
| **Llama 4 Scout** | 2000 | **0.8345** | **0.8184** | **0.8899** |
| **Llama 4 Maverick** | 2000 | **0.8345** | **0.7955** | **0.8804** |

同模型从 D5 到 D5+B17：

| Model | Δ Accuracy | Δ Balanced Accuracy |
|---|---:|---:|
| GPT-5.6 Luna | **+26.60 pp** | **+23.74 pp** |
| GPT-4o-mini | **+39.35 pp** | **+24.06 pp** |
| Llama 4 Scout | **+22.70 pp** | **+24.49 pp** |

最明显的是：

```text
GPT-5.6 Luna:
D5 only              Accuracy = 59.60%
D5 + B17             Accuracy = 86.20%
XGBoost D5+B17       Accuracy = 86.89%
```

也就是说，Luna 只比 D5+B17 supervised reference 低约 **0.69 pp Accuracy**；其 ROC-AUC 0.8935 也非常接近 XGBoost 的 0.897。

Llama 4 Scout 的 Balanced Accuracy 甚至略高于这一 XGBoost reference（0.8184 vs. 0.8125），但 Accuracy 较低，说明不同模型的 threshold / calibration / class tendency 仍有差异。

因此 Dataset 2 给出的信息比 Dataset 1 更直接：

> **LLM 并不是在这个 vaccination task 上“完全不会预测”。当 decision-relevant respondent information 被直接观察到后，强 LLM 可以接近 supervised reference；真正困难的是缺少这些信息时，LLM 能否把它补出来。**

---

# 5. 把两个 Dataset 放在一起：我们现在更关心“什么时候 LLM 会失效”

上周讨论中最有启发性的一点，是把研究问题从：

> “再做一个 prompting module 能不能提高 2–3 个百分点？”

转成：

> **“LLM 在什么信息条件下能够做 individual-level behavioral prediction，什么情况下会失败？”**

可以区分两类失败。

### Case A：信息有，但 LLM 没利用好

如果：

```text
ML(X) 很高
LLM(X) 明显较低
```

更像是：

```text
model / extraction / reasoning limitation
```

### Case B：输入本身缺少 respondent-specific behavioral information

如果：

```text
ML(D/H) 和 LLM(D/H) 都有限
但加入 B 后两者都明显提高
```

更像是：

```text
information limitation
```

Dataset 2 当前强烈提示第二类问题非常值得研究。

---

# 6. Behavioral-science framing：distal vs. proximal information

Vaccination literature 本身已经给了一个很自然的理论解释。

HBM、5C 等 behavioral frameworks 一般把：

```text
demographics / background / health context
```

看作相对 **distal / modifying information**；

而：

```text
perceived susceptibility
perceived severity
perceived benefits
barriers
confidence
constraints
cues / norms
```

更接近实际 decision，因此属于 **proximal information**。

这也和已有 empirical work 一致：behavioral / attitudinal variables 往往比少量 demographics 更能解释 vaccination uptake。

因此，单纯发现：

```text
B 比 D 更好预测 Y
```

本身并不足以形成一个新的 LLM contribution——behavioral science 已经预期这一点。

真正与 LLM 相关的新问题是下一步。

---

# 7. LLM 为什么在这里值得单独研究？

传统 supervised ML 在我们的实验里是一个 **information reference**：给它 labeled training data，它学习：

\[
X \rightarrow Y
\]

LLM 的额外可能性在于，它拥有来自 pretraining 的社会、健康、语言与行为知识，因此理论上可以尝试：

\[
D/H/G
\rightarrow
\hat B
\rightarrow
\hat Y
\]

也就是说，即使 respondent 没有直接回答某个心理问题，LLM 也许能够根据 profile 推断：

- 他可能如何感知 disease risk；
- 是否可能信任 healthcare / science；
- 是否可能有某种 barrier；
- 是否可能认为 vaccination beneficial / safe；
- 最后再预测 actual behavior。

这里需要一个严谨的表述：

> 不是说传统 ML 在数学上“不能推断 latent state”；如果给它 \(B\) labels，它同样可以训练 \(X\to B\)。  
> LLM 更特殊的地方是：**我们希望测试 pretrained world knowledge 是否允许它在没有针对当前数据集训练 \(B\)-reconstruction model 的情况下，zero-shot / prompted 地恢复这些 missing states。**

目前 Dataset 2 的 demographic-only 结果说明：

> **只从 5 个 demographic variables 出发，这种隐式 reconstruction 至少没有成功到足以支持高质量 individual prediction 的程度。**

但这并不意味着 LLM 永远无法 reconstruction。更合理的问题是：

> **输入需要丰富到什么程度，LLM 才开始能够可靠地推断 missing behavioral state？**

---

# 8. 下一阶段计划：直接研究 missing-state reconstruction

下一步希望找到一个同时具有以下结构的数据集：

```text
丰富 D：demographics
丰富 H：health / healthcare history
G：general beliefs / general medical-social orientation
B：针对目标行为的直接 psychological / behavioral questions
Y：actual behavior
```

最好还有 longitudinal ordering，使 \(B_t \rightarrow Y_{t+1}\) 更干净。

为了让模块边界清楚，主设置中：

- \(G\) 应尽量是 **general, non-target-specific** 的 belief / orientation；
- 过于接近 vaccination attitude 的 general-vaccine variables 可以单独作为 bridge/robustness condition；
- direct vaccination intention 应从主 \(B\) 中拿掉，因为它离 \(Y\) 太近。

---

## 8.1 Information ladder

先不改 prompting，只改变 observable information：

```text
D + H
  ↓
D + H + G
  ↓
D + H + G + B
```

分别比较 supervised ML 与 LLM。

这样可以回答：

> performance 的改善到底来自模型能力，还是来自信息本身？

---

## 8.2 Direct prediction vs. reasoning

对每个 information level，可以比较：

| Condition | 目的 |
|---|---|
| Direct zero-shot | 最基础的 behavior prediction |
| Generic CoT | 检查“多推理”本身是否改善 |
| Behavioral-theory-guided CoT | 用 HBM / 5C 等结构组织推理 |
| Explicit state reconstruction | 先输出 \(\hat B\)，再根据 \(\hat B\) 和 profile 预测 \(Y\) |

重点不再只是 final Accuracy。

---

## 8.3 直接验证 \(\hat B\)

如果数据中有真实 respondent answers \(B\)，就可以第一次真正检验：

\[
\hat B \quad vs.\quad B_{\text{actual}}
\]

例如分别让 LLM 预测：

```text
Perceived susceptibility
Perceived severity
Vaccine benefit / importance
Safety concern
Trust
Barrier
...
```

然后比较它与 respondent 的真实问卷答案。

这样就能区分：

### 结果 1：\(\hat B\) 与真实 B 很接近

说明 LLM 的 pretrained knowledge + reasoning 确实能恢复一部分 missing state。

### 结果 2：\(\hat B\) 看起来很“合理”，但和真实 respondent 不一致

说明 LLM 可能只是在生成 population-level stereotype / plausible explanation，而没有恢复 individual-level state。

这是比只看最终 vaccination Accuracy 更直接的 test。

---

## 8.4 下游 recovery：推断出的 B 到底补回了多少信息？

最终比较：

\[
Perf(D+H+G)
\]

\[
Perf(D+H+G+\hat B)
\]

\[
Perf(D+H+G+B_{\text{actual}})
\]

可以定义一个 descriptive recovery ratio：

\[
Recovery=
\frac{
Perf(D+H+G+\hat B)-Perf(D+H+G)
}{
Perf(D+H+G+B_{\text{actual}})-Perf(D+H+G)
}
\]

它回答：

> **直接测量 B 带来的 predictive gain 中，有多少能够由 LLM inference 补回来？**

---

# 9. 一个很重要的诊断框架

下一阶段最好同时保留 supervised ML reference，因为它能帮助判断 failure source：

| 观察结果 | 更可能的解释 |
|---|---|
| ML(D/H/G) 低，LLM(D/H/G) 也低 | observed profile 本身 information insufficiency |
| ML(D/H/G) 高，LLM(D/H/G) 低 | LLM extraction / reasoning limitation |
| 加 G 后 ML 与 LLM 都提升 | richer general context 确实增加 recoverability |
| \(\hat B\) 准确且下游接近 observed-B | LLM 成功补足部分 missing state |
| \(\hat B\) 不准，但 final Y 有提升 | LLM 可能只是形成了有用的 synthetic representation，而不是真实 psychological simulation |

这会让我们的结论不依赖于“LLM 必须失败”。

无论最后是哪一种结果，都能回答：

> **LLM 在哪一种 information regime 下能够做 individualized behavioral inference？在哪些 regime 下 reasoning 仍无法替代直接 measurement？**

---

# 10. 当前正在评估的候选结构：VISION-19

我们已经开始检查 VISION-19（Thorpe et al., 2024, *BMC Infectious Diseases*）的数据结构。它是三波美国 survey，930 名 respondents 完成全部 waves，并观察到 actual COVID-19 vaccine uptake。

当前 dictionary 允许比较清楚地构造：

### D + H

```text
age / gender / race / income / rural-urban / veteran status
12 pre-existing conditions
health literacy
numeracy
```

### G：general, non-COVID-vaccine-specific orientation

```text
medicalAction
general healthcare trust
general belief in science
political orientation
```

### B：COVID-specific proximal beliefs

```text
COVID worry / susceptibility / severity
COVID-vaccine importance
relative vaccine-vs-disease risk
development-too-fast concern
vaccine safety / side-effect concern
COVID-specific conspiracy / misinformation
```

为了保持信息层级清晰，目前倾向于：

- **不把 EVCI 和 FluVaxQ2 放进主 G**：它们已经属于 general-vaccine orientation，和 COVID-vaccine B 太接近，可作为单独的 bridge/robustness condition；
- **不把 vax1 / vax13 / vax2ipsos3 放进主 B**：这些是 vaccination intention，离 actual uptake \(Y\) 太近。

VISION-19 是否最终作为主数据还需要继续检查 temporal ordering、sample size 和各变量完整率，但它已经说明这种 \(D/H \rightarrow G \rightarrow B \rightarrow Y\) 的实验结构是可实现的。

---

# 11. 当前最简洁的 Research Questions

### RQ1 — Information hierarchy

**LLM-based vaccination prediction 是否呈现与 behavioral science 一致的信息层级：proximal behavioral information 是否比 distal demographic / health information 更有助于 individualized prediction？**

### RQ2 — Behavioral-state reconstruction

**当 proximal behavioral information 缺失时，LLM 能否利用 pretrained knowledge 从 demographic / health / general-belief profile 中恢复 respondent-specific behavioral states？**

### RQ3 — Theory as reasoning guidance

**HBM / 5C 等 behavioral theory 能否帮助 LLM 更准确地完成 missing-state reconstruction，而不仅仅是生成一个听起来合理的 explanation？**

### RQ4 — Information boundary

**随着 observable information 从 D/H 增加到 D/H/G，LLM 的 behavioral-state recoverability 如何变化？在哪一个 information regime 下 reasoning 开始有效，在哪些情况下直接 measurement 仍然不可替代？**

---

# 12. Takeaway for discussion

目前最值得继续推进的不是一个“更复杂的 vaccination predictor”，而是一个关于 **LLM behavioral inference boundary** 的问题：

> **Behavioral science 告诉我们，distal background 与 proximal state 并不是同一种信息。LLM 的特殊价值可能在于利用 world knowledge 在二者之间进行推断；而我们的初步结果表明，这种能力高度依赖输入信息的丰富程度。纯 demographic profile 下，LLM 没有成功补足 missing behavioral information；当真实 decision-relevant variables 被提供后，预测能力则大幅恢复并接近 supervised reference。下一步应直接测试：加入 richer general context 与 theory-guided intermediate reasoning 后，LLM 到底能恢复多少真实 respondent-specific behavioral state。**

这使项目从单纯的“vaccination prediction”转向一个更一般的问题：

\[
\boxed{
\text{When can reasoning compensate for missing information in LLM-based human prediction?}
}
\]

---

# References

- Lazarus, J. V., Wyka, K., White, T. M., et al. (2023). *A survey of COVID-19 vaccine acceptance across 23 countries in 2022*. Nature Medicine, 29, 366–375. https://doi.org/10.1038/s41591-022-02185-4
- Thorpe, A., Fagerlin, A., Drews, F. A., et al. (2024). *Predictors of COVID-19 vaccine uptake: an online three-wave survey study of US adults*. BMC Infectious Diseases, 24, 304. https://doi.org/10.1186/s12879-024-09148-9
- Limbu, Y. B., & Gautam, R. K. (2023). *How well the constructs of Health Belief Model predict vaccination intention: A systematic review on COVID-19 primary series and booster vaccines*. Vaccines, 11(4), 816.
- Eiden, A. L., Barratt, J., & Nyaku, M. K. (2022). *Drivers of and barriers to routine adult vaccination: A systematic literature review*. Human Vaccines & Immunotherapeutics, 18(6), 2127290.
- Betsch, C., Schmid, P., Heinemeier, D., Korn, L., Holtmann, C., & Böhm, R. (2018). *Beyond confidence: Development of a measure assessing the 5C psychological antecedents of vaccination*. PLOS ONE, 13(12), e0208601.
- Argyle, L. P., Busby, E. C., Fulda, N., Gubler, J. R., Rytting, C., & Wingate, D. (2023). *Out of one, many: Using language models to simulate human samples*. Political Analysis, 31(3), 337–351.
- Hu, T., & Collier, N. (2024). *Quantifying the Persona Effect in LLM Simulations*. ACL 2024, 10289–10307.
- Sun, H., Pei, J., Choi, M., & Jurgens, D. (2025). *Sociodemographic Prompting is Not Yet an Effective Approach for Simulating Subjective Judgments with LLMs*. NAACL 2025, 845–854.
