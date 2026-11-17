# 组会文献综述：LLM 行为预测与疫苗接种预测研究进展

## 开场｜先回应上周组会的问题：Top-4 Features 是什么？

上周老师问到：**为什么 NHIS 的流感疫苗接种预测只保留最重要的 4 个 Features，模型仍然有不错的表现？这 4 个变量具体是什么？** 下面先汇报我们的 Top-4 实验结果和变量含义，再进入文献综述与后续研究框架。

### 1. 仅使用 Top-4 Features 的预测表现

| 模型 | Top-4 Accuracy | Top-4 Balanced Accuracy |
|---|---:|---:|
| XGBoost | **74.86%** | **75.14%** |
| Random Forest | **75.02%** | **75.27%** |
| Logistic Regression | **75.00%** | **75.30%** |
| Qwen9B SFT | **74.78%** | **75.22%** |

**直接发现：** 仅使用 4 个特征，传统 ML 和 Qwen9B SFT 的 Accuracy、Balanced Accuracy 都已达到约 **75%**，且模型之间差距很小。这说明当前任务中，少数变量已经包含相当强的接种行为预测信号。

### 2. Top-4 变量的含义与重要性

| 排名 | NHIS 变量名 | 英文含义 | 中文解释 | XGBoost Importance |
|---:|---|---|---|---:|
| 1 | `SHTCVD19NM2_A` | Number of COVID-19 vaccinations | 累计接种 COVID-19 疫苗剂数 | 0.07768 |
| 2 | `SHTPNUEV_A` | Ever had pneumonia shot | 是否曾接种肺炎球菌疫苗 | 0.02254 |
| 3 | `SHTSHINGL1_A` | Ever had a shingles vaccination | 是否曾接种带状疱疹疫苗 | 0.01824 |
| 4 | `SHTCVD191_A` | COVID-19 vaccination | 是否至少接种过一剂 COVID-19 疫苗 | 0.00853 |

**关键结论：Top-4 全部与既往其他疫苗接种经历有关。** 它们并不是年龄、收入或慢病，而是直接反映一个人过去的疫苗接种行为。这个发现说明，**个人是否曾接受其他疫苗、接种过多少剂，对其流感疫苗接种状态具有很强的预测关联**；也与已有疫苗预测文献反复发现“既往接种史是重要预测变量”的结果相呼应。

---

> **汇报结构：** 开场先回应上周关于 Top-4 Features 的问题；第一部分回顾 LLM 预测真实人类行为的研究框架；第二部分总结疫苗接种预测领域关于 ML、心理态度变量和 LLM 的已有发现；第三部分结合前两部分，提出 NHIS 项目的初步 Research Questions 与实验安排。
>
> **阅读顺序：** Top-4 特征结果 → Marketing Science 范文与 Behavioral Prediction → 疫苗接种预测领域证据 → NHIS 四阶段研究方案。

# 第一部分｜LLM 对真实人类行为的预测：研究框架与解释方式

## 1. [P1] Can Large Language Models Capture Human Preferences?

**期刊与链接：** *Marketing Science: Frontiers*（2024）｜[论文 DOI](https://doi.org/10.1287/mksc.2023.0306)

### 1.1 Background：为什么 Marketing 领域会关心这个问题？

Marketing 经常要从人类受访者收集 *stated preferences*、conjoint answers 或购买意愿，并据此估计消费者如何权衡产品属性。真实消费者调查具有招募、时间与经济成本。若可以让 LLM 充当“虚拟受访者”，营销研究的偏好测量成本可能大幅降低。但问题在于：**语言上像人类的模型，真的会像人类一样做经济选择吗？**

作者选取 **Intertemporal Choice（跨期选择）**，例如“较早拿到较少奖励”与“较晚拿到较多奖励”的权衡。这不是随机选题，而是因为跨期选择具备成熟的行为经济学基准、可估计的时间折现参数，也存在语言结构与时间偏好之间的理论联系。**这种选择让作者不需要新建一个巨大的真实行为数据库，也可以使用既有文献中的人类参数/行为规律作为参照。**

### 1.2 Research Gap / RQs

**论文动机：** 模型可以生成流畅的人类式回答，却未被证明能恢复具有经济意义的偏好参数；更不知道其偏好是否对模型版本、语言和 prompting 敏感。

可以将论文的研究问题归纳为四个递进层次：

* **RQ1 — Fidelity：** GPT-3.5 / GPT-4 的跨期选择是否与既有人类证据相符？
* **RQ2 — Heterogeneity：** 模型与语言会不会系统改变估计出来的时间偏好？
* **RQ3 — Intervention：** 让模型说明选择理由（Chain-of-Thought Conjoint）是否减少差距？
* **RQ4 — Complementary Value：** 如果直接把 LLM 当受访者不可靠，模型生成的理由能否帮助提出未来可检验的消费者行为假设？

### 1.3 Resources：具体用了什么数据、模型、设计？

**研究单位：** LLM 对两个跨期奖励选项的选择。

* **模型：** GPT-3.5-turbo、GPT-4 的指定快照；standard conjoint 两者都跑，CoT conjoint 主要研究 GPT-4。
* **条件：** 22 种语言；每种语言构造 **63 个 choice cells**（9 个 delay 差距 × 7 个利率水平）；较早奖励为 1 个月后的 1,000 tokens，较晚奖励根据利率和时间差计算。
* **对照：** 既有跨期选择的人类研究所观察的偏好/折现规律，而非额外训练的 RF/XGBoost。
* **控制：** 随机交换两个选项出现的先后顺序；每次独立开启会话以减少上下文依赖；另外用货币单位做部分稳健性检查。
* **CoT 样本规模：** GPT-4 CoT 部分约 **22 × 63 × 10 = 13,860 次生成回答**。
* **核心测量：** 选择较晚奖励的概率、对奖励利率变化的敏感性、折现率 (\\delta)、choice noise 参数 (\\mu)，以及理由文本的 LDA topics。

### 1.4 研究如何逐层展开？

#### Layer 1 — Behavioral discrepancy：先证明“像人”这一假设站不住

将模型置于标准化的跨期选择任务中，让其在高低金额、长短延迟条件下做选择。模型整体比人类更偏向短期奖励。尤其 GPT-3.5 在固定延迟条件下对更高的未来奖励额反应不足，出现**按时间优先的字典序式偏好（lexicographic preference）**，而不是常见的“未来奖励越高，愿意等待的概率越高”。

**这一层的写作功能：** 用有经济意义的结构性偏好差异，而不仅是分数，说明模型与人类选择的差距。

#### Layer 2 — Model × Language heterogeneity：展示失败不是随机噪声

作者比较 GPT-3.5/4 与 22 种语言，并依据 future-time reference（FTR，未来时态引用）区分语言组。模型在若干弱 FTR 语言中表现出更高的等待意愿，与先前有关语言和人类时间偏好的研究方向一致；**但“相对趋势一致”并不等于绝对折现率可靠。**

**关键设计点：** 语言是有理论依据的异质性维度，能够帮助解释差异何时出现。

#### Layer 3 — CoT Conjoint：试图改善已有缺陷

标准条件下，模型仅给出选项；在 CoT 条件下，模型先生成一段解释再作选择。作者观察到 GPT-4 选择较晚奖励的倾向提高，估计出的折现偏好与人类基准差距减小，但没有完全消失。

**关键意义：** CoT 是针对前述偏差展开的干预研究；差距缩小但没有消失。

#### Layer 4 — Explanation / hypothesis generation：负面结果之后仍然寻找研究价值

针对 GPT-4 CoT 生成的文本，作者翻译成英文，进行文本预处理和 LDA（设定 4 个主题），进一步将相关主题概括为**风险与不确定性、机会成本/替代投资、紧急需要**等。随后分析这些话题在不同语言、延迟条件下的相对出现频率。

**这一层的研究定位：** 通过模型生成的理由提取候选影响因素，主要用于 **hypothesis generation**。

#### Layer 5 — Managerial / research implication：清晰规定“能做什么、不能做什么”

直接把 GPT 代替真实受访者来估计消费偏好可能误导研究；但模型回答可以用于探索潜在属性、情境差异、后续调查应检验的假设。


---
## 2. [P2] HouseholdBench: Evaluating Large Language Models as Predictors of Household Economic Behavior

**论文链接：** [arXiv:2610.07563](https://arxiv.org/abs/2610.07563)

### 2.1 Background：从“模型会谈论经济学”转向“能预测真实家庭吗？”

家庭如何消费、工作、储蓄、应对失业或政策冲击，是经济学最关心的真实世界问题之一。通用 LLM 可能具备大量经济常识，但能否用这些知识预测具体家庭在具体时点的真实结果尚不清楚。既有研究往往只覆盖少量调查和 outcome，也少有直接测量模型对**政策变化/经济冲击反应**的预测能力。

**研究动机：** 要判断 LLM 能否成为有用的家庭行为预测工具，仅在某一问卷问题上取得合理分数是不够的；需要同一评估体系、多个行为类别、真实家庭数据、强 ML Baseline 和对 policy response 的检验。

### 2.2 Research Gap / RQs（组会可使用的表述）

* **RQ1:** 给定真实家庭的人口、过去行为及宏观经济条件，LLM 对未来行为/报告结果的预测有效吗？
* **RQ2:** 和简单不变预测（no-change）以及使用同样特征训练的 XGBoost 相比，优势和短板是什么？
* **RQ3:** 模型在哪些 outcome 上有*方向一致的系统偏差*？是错误估计变化幅度、过度依赖多数类，还是忽略行为惯性？
* **RQ4:** SFT 和多次采样聚合能否改善小型开源模型？改进能否迁移到没有用于微调的 policy-response tasks？

### 2.3 数据、模型与评价

|资源|论文设计|解释|
|-|-|-|
|数据来源|**6 个美国大型家庭调查**，底层总规模约 **21.7 million observations**|按任务抽样构建预测评估集|
|任务|**32 个 tasks**：18 个 baseline tasks + 14 个 policy-response tasks|领域涵盖消费/储蓄、收入、就业/退休、宏观预期、住房|
|目标类型|**18 numeric，9 categorical，5 probability**|按任务类型使用不同指标|
|输入|家庭人口学、近期行为/结果、宏观变量；policy tasks 增加相关政策或冲击|真实世界信息条件|
|LLM|**13 个** proprietary / open-weight 模型|多模型比较|
|非 LLM|**No-change baseline + XGBoost**|XGBoost 是每项任务单独训练，使用与 LLM 对应的输入信息|
|数据划分|历史季度按 80/10/10 切分，另留出 2026-02-16 之后公布的数据|检查预训练知识截止时间/潜在污染；每项测试最多抽 200 条进行模型评价|
|统计|任务级指标、跨任务平均排名、paired bootstrap|报告平均值和抽样不确定性|
|SFT|Qwen3.5-4B 与 Qwen3.6-27B，LoRA，1 epoch；18 baseline tasks 合计 **100,702** 条训练点|每任务至多取 6,000 训练点；policy-response tasks 不用于 SFT|
|推理聚合|**K = 16 次**预测/样本；数值/概率取均值、类别取多数投票|提升稳定性，但推理成本上升|

**对比设置：** Policy-response tasks 未参与 LLM 的 SFT，而对应 XGBoost 在各自任务标签上训练，因此两者属于不同监督条件下的性能参照。

### 2.4 Layer 1 — 建立全面预测 Benchmark

作者将多个原本不可直接比较的调查，转换为统一的 input–target–split 任务。区分三个 outcome 类型：

* **Numeric：** relative MAE（相对 no-change 的平均绝对误差）。
* **Categorical：** Macro-F1，避免稀有类别被总体准确率掩盖。
* **Probability distribution：** Total Variation distance（模型概率向量与受访者报告的概率向量之差）。

再用跨 32 个任务的平均名次汇总，避免某个数量单位或单一任务主导结论。

**研究功能：** 先确定模型是否有实际可用的预测信号，再谈哪个更强。

### 2.5 Layer 2 — 与传统 ML 对比：究竟发现什么？

主要结果：**XGBoost 在多数任务中排名最高**，而领先的闭源 LLM 接近它；开源模型整体表现更弱。多数 LLM 能超过 no-change 的数值和类别预测基准，但在部分概率分布预测上连 no-change 都未能超过。

一个有解释价值的细节是：有些家庭对自己的概率预期并不改变，而 LLM 往往预测发生了变化。也就是，在概率预测任务中，**维持原预测有时是极强的基线**。

部分任务中最佳 LLM 与 XGBoost 表现接近，论文强调的是 **XGBoost 跨任务总体更稳定**。

### 2.6 Layer 3 — Error Analysis：作者如何解释传统模型为什么经常更强？

这是本篇对 NHIS **最值得详细拆解的部分**。原文 Section **4.2: LLMs’ bias on predicting household behavior**，并配有 Figure 3。

#### 诊断 A：有方向的系统高估/低估，而非仅仅平均误差大

作者在 18 个数值任务中计算带方向的误差：(\\hat y-y)，比较模型是否系统高估或低估不同 outcome。特别是在一些 policy-response tasks 中，LLM 倾向**夸大家庭对政策/收入冲击的反应幅度**。

**原文例子：** 当问“若永久增加 10% 收入，会花掉或捐出增加收入的多大比例”时，家庭回答的中位数约为 **5%**，而部分领先 LLM 预测约 **30%–40%**。这说明模型可能能讲出合理的经济反应故事，却在真实反应强度上出现可系统描述的偏差。

**迁移到 NHIS：** 二分类没有连续金额，但可比较 **平均预测接种率 − 真实接种率**、分组概率偏差、预测阳性比例和校准误差。若 LLM 默认“更愿意接种”，就可能系统性高估接种行为。

#### 诊断 B：过度预测多数类（尤其 status quo）

作者将每个类别任务的真实多数类比例与 LLM **预测该类的比例**并排展示。例如，在下月失业状态任务中，现实中约 **50%** 的相关人群继续失业，若干 LLM 却将 **94%–100%** 预测为仍失业。有的模型在某些任务中完全漏掉离开劳动力市场的少数类别。

这里的解释非常直观：模型学到的“典型家庭/稳定状态”可能过于强势，从而在个体层面压制了变化和少数结果。

**迁移到 NHIS：** 报告真实接种率与预测接种率、TPR、TNR、BA、MCC、confusion matrix；同时比较不同方法的“总是 Yes / 总是 No”基准。即使 Accuracy 接近，也可能是不同的错误结构。

#### 诊断 C：考察任务外、时间外表现以排查简单的记忆解释

论文比较知识截止日前后或不同时间段的任务表现，没有观察到普遍、系统性的截止后性能下降。这削弱了“只是背过旧的调查答案”的简单解释，但**不能彻底证明无训练污染**。

**迁移到 NHIS：** 需要明确预测标签是历史自报结果、跨时点预测还是同波次分类。若没有真正时间外测试，不应把随机划分 Test80 写成未来接种预测。

#### 这一层为我们提供了什么？

HouseholdBench 用**偏差方向、类别频率和任务层面的比较**刻画系统性错误。这种 Error Analysis 能帮助解释模型表现差距来自哪里，并为后续机制研究提供具体假设。

### 2.7 Layer 4 — SFT：不是为了让所有模型都赢，而是检验 Task Adaptation 能否补足差距

作者使用 **18 个 baseline tasks 的真实训练标签**，对两个开源模型做 LoRA SFT，未将 policy-response tasks 纳入微调训练。

重点发现有三层：

1. **较弱的 Qwen3.5-4B 明显受益：** baseline tasks 上 RelMAE、Macro-F1、TV 多项改善；迁移到 held-out policy-response tasks 时，也观察到某些数值和分类指标改善。
2. **较强的 Qwen3.6-27B 改善不一致：** Macro-F1 可以改善，但某些数值和概率指标反而变差。**“已经较强的模型不一定能从额外 SFT 获益”** 是论文的重要结论之一。
3. **SFT + 16 次预测聚合** 能进一步改善部分数值和概率结果；但这增加推理次数，不是完全免费的改进。

**对 NHIS 的参考：** 可以进一步分析 SFT 在哪些类别、哪些人群、哪些评价指标上缩小了与 ML 的差距。

### 2.8 Layer 5 — Generalization：避免只证明模型记住了训练任务

SFT 没有在 policy-response tasks 的训练标签上更新参数，之后直接评价这些任务。这样才有资格讨论：适应模型习得的家庭经济预测能力能否跨任务迁移。

这一设计在同数据内的 Train/Test 比较之外，进一步检验了跨任务迁移。

---
## 3. [P3] Simulating Public Opinion: Comparing Distributional and Individual-Level Predictions from LLMs and Random Forests

**期刊与链接：** *Entropy*（2025）｜[论文原文](https://www.mdpi.com/1099-4300/27/9/923)

### 3.1 Background：为什么只看 Demographics 不够？

LLM 被用于模拟虚拟受访者的政治和社会观点，以降低传统调查或 Agent-based Simulation 的成本。但一个风险是：仅根据年龄、教育等人口学变量“扮演”一个人，往往无法准确描述其态度、价值观与具体问题上的立场。作者因此进一步引入真实调查中的态度与社会价值变量，观察不同信息来源对 LLM 与传统分类模型的影响。

**对 NHIS 关键关联：** 你们想讨论 HBM Prompting 为什么不总有效，而 NHIS 只有客观社会人口学、医疗接触、健康和接种史等代理信息。这篇提示：输入变量的**信息类型与目标的距离**，可能比单纯计数更重要。

### 3.2 Research Gap / RQs

* **RQ1：** 使用少量背景变量的 Frozen LLM 能否预测真实受访者的个体回答，而不仅模仿群体总体？
* **RQ2：** Demographic、Attitudinal、Moral 信息的组合如何改变 LLM 和 RF 的预测表现？
* **RQ3：** 让 LLM 自主选取最相关变量，能否减少输入并保留/改善结果？
* **RQ4：** 当 LLM 与 RF 获得**完全相同的信息池**时，它们在个体分类和群体分布上的相对优势是否不同？

### 3.3 数据与方法

* **数据：** ANES 2020 选举前调查样本，一个主数据源；选 **25 个 backstory features**，预测 **10 个 survey questions/topics**（真实人类调查回答）。
* **变量分组：** **D** = Demographic；**A** = Attitudinal & Political Orientation；**M** = Moral & Social Values。
* **模型：** 正文以 **Gemma3 12B** 为主要 LLM、**Random Forest** 为监督基线；**Qwen2.5 14B** 的复制结果放在附录。LLM **不做 fine-tuning**。
* **数据切分：** **3-fold cross-validation**；每个 fold 对 LLM 直接推理，对 RF 在其余 fold 有监督训练。模型在同一个验证 fold 上评价。
* **指标：** **F1**（个体层面的分类能力）；**Jensen–Shannon Distance (JSD)**（总体回答分布是否匹配）；**Cramér’s V**（预测与真实回答的统计关联）；bootstrap 置信区间（1,000 次）。
* **信息池：** 由 D/A/M 构成 **7 组非空组合**：D、A、M、D+A、D+M、A+M、D+A+M。

**比较设置：** LLM 与 RF 使用相同验证样本和特征池，但 LLM 是 Frozen、RF 使用训练标签；这体现了两种不同适应条件下的比较。

### 3.4 原文是四个连续 Experiments，不是三个

#### Experiment 1 — Random / Constant Baselines + Minimal Backstory

为每个调查问题，让 LLM 从 D、A、M 三组**各选一个**看起来最有预测力的变量，只提供少量 persona 信息；同时比较完全随机模型和永远预测训练集多数类的 constant model。

**发现：** LLM 在大多数考查的指标上明显优于这两种弱基线。例如在 climate change topic 上，LLM F1 约 0.59，random 约 0.36，constant 约 0.41。

**设计功能：** 首先证明 Frozen LLM 有信息利用能力，再问“更多、不同类型的信息是否更重要”。

#### Experiment 2 — Seven Feature Pools × Ten Topics × LLM/RF

对同一 topic，分别让 LLM 使用七种特征池中的**全部变量**，RF 也只用同一个池的特征进行训练，比较不同池的最优表现，以及每类信息的贡献。

**具体结果与作者解释：**

1. **A/M 信息在不少 topic 的最佳配置中出现更频繁。** 如 climate change，A+M 池的 LLM F1 约 **0.70**、JSD 约 **0.15**，比 Exp 1 有改善。
2. **没有通用最优池。** 某些 topic 加入更多变量反而不利；D 在与其他信息配合时仍可能重要。
3. **评估指标改变了“谁更好”的答案。** LLM 的 JSD 在不少 topic 优于 RF，RF 在个人层面 F1 可能占优。原文比较各自最优信息池时，LLM 在 10 个 topic 中有 **6 个** JSD 优势达到其统计显著标准。

**作者讨论的关键点：** 对态度性 Outcome，直接相关的态度和价值信息往往比单纯人口变量更有效；这种作用来自信息与目标的相关性。

#### Experiment 3 — LLM Self-Selected Features

不是再增加 feature，而是让 LLM 在给定池里先挑选其认为重要的变量，再只把这些变量写入预测 persona。这样可能减少 prompt token 数量，同时检验 LLM 的自选特征能力。

**发现：** 对很多 topic，F1/Cramér's V 改善很小，但若干 topic 的 JSD 有提高；同时出现一些**漏掉关键变量**的风险。例如 Health Insurance Policy 相关任务，自选特征后 JSD 变差，作者推测可能忽略了与目标高度相关的是否拥有健康保险信息。

**与 NHIS 的关联：** 已做的 Top-4/Random-K/Bottom-K 说明特征数量之外还存在“特征怎么选”的问题；可进一步研究特征信息类型。

#### Experiment 4 — 相同信息池下的 Head-to-Head

作者发现 Exp 2/3 中“各自的最优池”可能不同，因此加入第四个实验，固定 **topic × variable pool**，比较 Gemma3 和 RF 在**相同 D/A/M 特征组合**下的三项指标差。图表直接用 heatmap 呈现差异与 bootstrap 显著性。

**核心发现：** RF 在不少同池条件下的 F1 与关联指标更好或持平；LLM 在许多条件下的 **JSD（总体分布匹配）**更好，但这一优势并不普遍，在一些 D-only 任务里 RF 也更好。

**对 NHIS 的启发：** 对同一 respondent、同一 feature pool 或 missing mask 做 LLM–ML 对比，可以更清楚地分离信息条件与模型差异。

### 3.5 作者怎样解释 LLM 和 RF 的不同优势？

论文在 Discussion 给出一种**推测性机制解释**：RF 是针对 ANES 标签训练的 discriminative model，因此更擅长单个问卷回答的分类；而 Frozen LLM 通过大规模预训练可能学习到舆论分布的统计结构，因此更容易在总体回答分布上与真实调查一致。

这提供了一个解释视角：**个体分类准确性**和**群体回答分布拟合**可能需要不同的模型能力；不同的监督条件也是比较中重要的因素。

## 4. [P4] Persona-Prompted LLM Agents Achieve Modest but Genuine Prediction of Human Social Media Reactions

**期刊与链接：** *Scientific Reports*（2026）｜[论文原文](https://www.nature.com/articles/s41598-026-66277-8)

### 4.1 Background：LLM 能“扮演用户”不等于能预测用户下一步行动

大型语言模型常被用于构建“具有人口学、态度、价值观的虚拟用户”。模型生成的行为看起来合理，或者能够接近群体调查分布，并不能证明它可以预测**特定的个人面对特定内容时会如何反应**。这一差异直接影响用户定向、平台治理和社会模拟的可信度。

**作者认为此前的核心方法问题：** 较多研究用高原始 Accuracy 证明 Agent 像人，但面对大量“不会点赞/不会转发”的样本，这种 Accuracy 可能只是类别基准率带来的幻觉。

### 4.2 Research Questions / Hypotheses 的逻辑

论文实际围绕多个假设展开，这里按组会讨论聚合为：

* **H1 / Validity：** Persona-prompted LLM 对真实个人反应是否具有超越偶然或多数类猜测的预测信号？
* **H2 / Persona specificity：** 更详细的人口、价值与态度 Profile 是否更有帮助？
* **H3–H5 / Context：** 个人画像和帖文主题匹配、内容类别、正负情感表述是否影响预测？
* **H6 / Method comparison：** LLM Agent 能否超越传统有监督预测模型？

### 4.3 Data / Models / Input：具体资源

|项目|原文设置|
|-|-|
|人类基准|**1,511 名塞尔维亚参与者**，对 **56 条帖子**报告反应；帖子覆盖政治/新闻、娱乐/生活，以及正负 framing|
|Study 1 outcome|**Like、Dislike、Comment、Share、No Reaction** 等反应；可出现多种互动，不是简单单标签五分类|
|Personas|人口学 only、价值/态度 only、Full persona 三类|
|模型规模|**27 种 LLM × 3 类 Persona**，每个参与者得到 81 种 agent 配置，合计 **120,000+ agent–persona combinations**；加上帖文和反应指标得到数百万条机器生成的反应记录|
|Study 1 metric|Hamming accuracy，混合效应模型对模型、Persona、帖子类型和 valence 做比较|
|Study 2 outcome|仅对真实人类确实选择 Like 或 Dislike 的参与者–帖子记录测试 **Like vs. Dislike**|
|Study 2 LLM|从 Study 1 选 **4 个**表现较好的 LLM；只有 Demographics vs. Full Persona 两种输入|
|Study 2 metrics|MCC、Balanced Accuracy、minority-class lift、Brier，另外报告 raw accuracy|
|监督方法|结构化 RF/GBDT/LR 与 **TF-IDF（Persona + Post）+ Logistic Regression**；某些模型进一步加入个人反应历史|

### 4.4 Study 1 — 广泛测试真实反应：为什么 70.7% Accuracy 还不够？

Study 1 先回答“这些 Agent 大致可以模拟现实反应吗”，并系统比较 27 个 LLM、Persona 详细度、帖子内容类型、正负性和 Persona–Content 匹配。

论文报告整体 Hamming accuracy 约 **70.7%**，模型之间有约 **13 个百分点**差距。但一些互动类别非常稀有，例如不转发很常见，所以只靠匹配多数的“无反应”模式就可能看起来分数很高。

**这一 Study 的研究功能：** 筛选哪些模型和上下文值得进一步检查，同时发现 raw accuracy 在类别不平衡条件下可能误导。

**部分异质性结果：** 详尽 Persona 相比人口学 only 的平均表现提升不大且不稳定；正向表述帖子预测更容易，负向反应更难。这说明“给模型更多的个人描述信息”并不保证更好的个体行为预测。

### 4.5 Study 2 — 根据 Study 1 的失败模式，收缩成一个更严谨的研究问题

Study 2 仅考虑真实人类已经给出 Like/Dislike 的观察，避免把 No Reaction、Comment、Share 与正负反应混在一个宽泛任务中。对模型也只开放 like/dislike 二选一，用**纠正类别基准率的统计指标**去判断模型是否真的预测了个体差异。

**关键数据分布：** 真实标签中约 **69.7%** 是 Like；因此无脑预测“Like”就有约 69.7% Accuracy。LLM 可能取得更低的 raw accuracy，但仍有较好的 minority detection。作者用 MCC、BA 和 lift 证明模型包含超越简单基准率的信号。

**作者测得：** 最佳 agent MCC 约 **0.29**、BA 约 **0.66**，有真实但有限的预测力；对负向反应识别的 lift 超过基准，但仍不足以称为高精度个体预测系统。

Study 2 重点研究已经表达 Like/Dislike 的用户如何选择反应方向。

### 4.6 最重要：作者具体怎样说明“传统 ML 更好”？

这是你希望在 NHIS 中补齐的部分，建议重点讲 **原文 Table 5 + Discussion**。

#### 第一步：不是仅用一个 RF 当 Baseline，而是分出三类方法

1. **Structured ML：** 只用年龄、地区、教育等表格特征，部分模型加入从训练集得到的个人历史反应率（RF、Gradient Boosting、LR）。
2. **Text-based ML：** 将 Persona 和帖文原文转成 TF-IDF 文本特征，用 Logistic Regression 学习行为标签；另一版本再加入历史特征。
3. **LLM Agents：** Zero-shot 把同样的 Persona / Post 语义信息写入 Prompt，让 LLM 直接预测。

这使得研究者可以区分：**结构化个人属性、文本语义信息、LLM 复杂语言建模**哪一步真正增加预测价值。

#### 第二步：结果显示“文本信息本身”可能比“模型是否是 LLM”更关键

|Model|MCC ↑|Balanced Accuracy ↑|Brier ↓|
|-|-:|-:|-:|
|**TF-IDF + Logistic Regression (Persona + Post)**|**0.3601**|**0.6943**|**0.2001**|
|TF-IDF LR (+ History)|0.3521|0.6879|0.2024|
|**GPT-5.2 Agent**|**0.2958**|**0.6574**|**0.3292**|
|Gemini-2.5-Flash Agent|0.2777|0.6468|0.3327|
|Structured Random Forest|0.1483|0.5797|0.2494|

文本 LR 比最佳 LLM 的 MCC 高约 **0.064**，而带结构化信息的 RF 不如 LLM。作者据此论证：帖文文本里的可预测信号很重要；在此任务中，简单、接受监督学习的文本模型即可有效利用这些信息，未显示出 LLM 自称“代理个人”的独有预测优势。

#### 第三步：多做一层“模型失败类型”的研究，而不是停留在一个平均值

* **Class prior / Majority class：** Always Like 已经接近 69.7% Accuracy，必须报告 MCC、BA、minority lift。
* **Valence bias：** 模型对积极情感帖文更容易作出正确预测，对负面反应/反对意见可能系统低估。
* **Persona richness：** 在二分类中，额外心理细节的收益变得很小。
* **Model choice：** 27 个 LLM 之间有明显差距，不能把某一个模型的失败泛化到所有模型。
* **Probabilistic quality：** 文本 LR 的 Brier 更低，提示输出概率质量也要比较。

#### 第四步：如何理解“文本 LR 超过 LLM”？

作者的结果说明，**文本中的行为预测信号可以通过简单的监督模型有效利用**。同时，TF-IDF LR 使用了该任务的监督标签，而 LLM Agents 主要是 zero-shot；因此这组结果最直接支持的是不同方法在当前训练条件下的表现比较。对 NHIS 来说，更值得借鉴的是作者把 Structured ML、Text-based ML 和 LLM Agents 逐层拆开比较的思路。

## 5. 第一部分小结：对 NHIS 研究最有启发的分析层次

这四篇论文的作用有所不同：**Goli & Singh** 是目标期刊的行文与论证模板；**HouseholdBench** 是 Frozen → ML 对比 → Error Analysis → SFT → Generalization 的完整框架；**Simulating Public Opinion** 说明 Feature Type 和评价目标的价值；**Persona-Prompted Agents** 则演示如何用更细致的 baseline 和错误结构解释 ML 与 LLM 的差异。

对 NHIS 来说，现阶段最值得讨论的是：补强 **Error Analysis**、研究 **Feature Information Type**，以及明确 **SFT 具体改善了什么**。如果要进一步寻找带有直接态度测量的数据集，可以作为与现有 NHIS 形成对照的候选研究，而不是先确定一定要增加新的 Study。



### 5.1 Error Anatomy：从平均表现走向系统性误差

**目标：** 不只说“Frozen 不如 XGBoost，SFT 追上来了”，而是确定差距来自什么样的预测错误。

建议以同一固定 Test80、每位受访者一行结果建立 `model × respondent` 的预测记录表，然后至少报告：

1. **Class distribution：** 真实接种比例、预测接种比例、always-Yes/always-No、majority-class baseline。
2. **Confusion structure：** TP/FP/TN/FN，TPR/TNR，BA，MCC，positive precision，AUC；对每个模型保持相同标签定义。
3. **Direction of bias：** 总体和分组的预测概率偏差、预测接种率偏差；区分 hard label 的阳性率与 probability mean。
4. **Paired disagreement：** XGBoost 对而 Frozen 错、Frozen 对而 XGBoost 错、SFT 修复了哪些 Frozen 错误；显著性用配对方法/受访者 bootstrap。
5. **Calibration / uncertainty：** 只有在可提取有意义的 Yes/No probability 时才报告 Brier、Log Loss、calibration slope/curve；温度等后处理用独立验证集校准。
6. **Groups：** 划分重要性较强且有实质意义的亚组，报告分母、CI，避免只展示差距最大的事后组。

**待检验的现象：** Frozen LLM 是否高估或低估接种，SFT 是否主要修正类别偏差，以及不同模型在哪些受访者身上出现不同错误。

### 5.2 Information Value：重要特征与信息可用性

* **Feature Concentration：** NHIS 的 Top-4 特征是什么？Top-K、Random-K、Bottom-K 的结果说明多少预测力集中在少数变量上？
* **Feature Availability：** 全体受访者使用相同少量特征，与每个人随机缺失不同特征时，模型表现有何差异？
* **Direct Attitudes：** 如果在另一数据集中观察到疫苗态度、风险感知和接种意愿，模型之间的相对表现会怎样变化？

这些问题可分别利用已有的 Feature Budget / Missingness 实验，以及拟补充的含心理态度数据集来回答。

### 5.3 Model Adaptation：SFT 改变了哪些预测

建议把 SFT 的作用拆开报告：

* **Gap closing：** Frozen→SFT 的提高幅度；相对于最佳 supervised ML 缺口缩小多少。
* **Which errors fixed：** SFT 主要修正的是类别先验偏差、特定人群、还是排名能力？
* **What does not improve：** BA、Calibration、少标签或 Missingness 下哪些指标仍无提升？
* **Resource dependence：** 多少真实标签后有明显收益，SFT 是否有性能平台；是否值得加入更轻量的 Steering 做资源效率对比。
* **Robustness：** 不同随机种子、相同 split / test protocol，避免把极小提升当成确定进步。

**可选延伸：** 将 Steering 视为低资源模型适应策略，与 Frozen 和 SFT 比较。

### 5.4 Individual vs. Aggregate：两种不同的预测质量

*Simulating Public Opinion* 说明个体 F1 和总体分布 JSD 可能给出不同模型排名；NHIS 则可以在真实二分类标签下区分：

* **个体预测**：Accuracy、Balanced Accuracy、AUC、MCC。
* **总体分布**：样本/调查加权的真实接种率 vs. 模型均值接种概率；分组接种率差异。
* **风险排序**：如果面向健康管理的优先联系名单，比较 Precision@K、Recall@K、top-risk capture。但这只是**预测层面的识别效用**，不是证明联系后能提高接种率。

个体预测与总体接种率拟合代表两种不同的模型能力，值得分别评价。

---

# 第二部分｜疫苗接种与疫苗犹豫预测：已有证据与关键发现

**本部分目的：** 汇总疫苗预测领域已经取得的经验发现：研究使用什么数据和特征，哪些方法表现更好，哪些变量最有预测力，以及 LLM 研究目前进展到哪里。

**文献安排：** 两篇传统 ML 的实际接种预测、一篇前瞻性心理因素研究、一篇接种犹豫动态研究，以及一篇直接使用 LLM 预测个体接种选择的研究。

## 6. 领域总体认识：三项主要发现

### 6.1 传统机器学习已经有成熟的疫苗接种预测基础

已有研究至少从两种成熟的数据环境出发：一是长期电子病历与既往多年的真实接种记录（Shaham et al., 2020），二是全国性的大样本社会调查及不同调查波次（Maxey et al., 2026）。常见算法包括 **Logistic Regression、Naïve Bayes、Random Forest、XGBoost、LightGBM、Neural Network**；研究者不仅报告预测指标，也会检验重要变量、移除变量后的性能变化。两个比较稳固的经验认识是：

* **既往疫苗接种行为通常极有预测力。** 在长期流感接种记录中，前一至前三个流感季的接种历史是主要信息来源；在近期 COVID-19 接种预测中，同一接种季的流感接种情况是 RF 的最重要预测因子。
* **预测信号可以高度集中于少数特征。** Maxey et al. 通过移除流感接种史和年龄，发现模型判别力明显下降。

这些结果说明，**传统 ML 在疫苗接种预测方面已有较成熟的研究基础**，并形成了关于既往行为、个人特征和医疗记录的重要经验发现。

### 6.2 直接疫苗态度与心理变量具有较高预测价值

除了客观人口和医疗变量，研究反复识别出 **接种意愿、疫苗安全性/危险性感知、疫苗有效性信念、疾病严重性、阴谋论倾向、社会规范、疫苗知识**等与接种或接种犹豫有关的信息。这里有两个不同层次：

1. **预测真实后续接种行为**：前瞻性研究显示，基线接种意愿、阴谋论倾向、危险性感知和既往流感接种记录，可以预测半年后是否仍不接种 COVID-19 疫苗（*Vaccine*, 2024）。
2. **预测 Vaccine Hesitancy**：中国两阶段调查显示，最有信息量的构念会随着初次接种与加强针阶段变化；社会规范、安全性感知在初期比较突出，年龄、既往接种经历、疫苗有效性感知在加强针阶段上升（Li et al., 2025）。

这里需要区分实际接种与疫苗犹豫：两者相关，但预测目标不同；SHAP / permutation importance 用于描述变量的预测贡献。

### 6.3 LLM 直接预测个体接种的证据仍然有限

在这五篇代表性文献中，**Xu et al. (2026)** 直接使用多种 LLM 预测受访者的自报接种状态。它发现：输入只有 Demographics 时，多种 LLM 几乎把所有人预测为会接种，Balanced Accuracy 接近随机；直接加入受访者报告的疫苗态度后，模型表现显著改善；用按态度构造的媒体信息环境替代直接态度，仍能恢复一定预测信息，但模型间的类别偏好差异很大。该文的重点是不同输入信息条件与 LLM 的预测偏好，而非完整的 LLM–ML/SFT 排名。

### 6.4 文献中的预测目标

|文献|主要因变量|是否直接研究真实接种|核心评价问题|
|-|-|-|-|
|Shaham et al. (2020)|**下一流感季实际接种**（电子病历）|是，较强的前瞻性行为记录|过去的个人/家庭行为能否预测下一季？|
|Maxey et al. (2026)|**2024–25 季 COVID-19 接种**（自报）|是，但为不同调查波次的横截面观察|哪些变量和 ML 模型可区分已接种/未接种？|
|*Vaccine* (2024) 心理预测研究|**基线未接种者半年后是否仍未接种**|是，前瞻性随访|哪些心理构念能识别持续未接种者？|
|Li et al. (2025) 中国研究|**初针/加强针 Vaccine Hesitancy**（按实际行为与意愿构造）|**混合行为与态度的结果变量**|核心相关因素是否随接种阶段变化？|
|Xu et al. (2026) LLM 研究|**COVID-19 是否至少接种 1 剂**（调查自报真值）|是，LLM 模拟预测该标签|只有人口资料、直接态度、媒体代理时，模型有多可靠？|

---
## 7. [V1] Personal and Social Patterns Predict Influenza Vaccination Decision

**Adir Shaham, Gabriel Chodick, Varda Shalev, Dan Yamin (2020)** · *BMC Public Health* 20:222。  
[开放全文（PMC）](https://pmc.ncbi.nlm.nih.gov/articles/PMC7017468/) · [DOI](https://doi.org/10.1186/s12889-020-8327-3)

### 研究背景与核心问题

长期以来，季节性流感疫苗覆盖率不足。作者认为，要预判哪些人下一季会接种，需要关注患者**既往实际接种记录、医疗使用模式和家庭/社区的接种行为**，而不只是某个时间点的接种态度。核心问题有两个：哪些 *personal / social behavioral patterns* 与未来接种有关？能否利用这些记录准确预测下一流感季的个人接种？

### 数据、目标与 Feature 类型

* **数据来源：** 以色列 Maccabi Healthcare Services 匿名电子病历，2007–2017 年，从 137 家诊所抽取约 **250,000 名患者**；部分患者有可连接的家庭记录。
* **目标：** 给定过去多个流感季的资料，预测个人在**下一流感季**是否接种流感疫苗。
* **个体行为历史：** 过去 1–3 个流感季是否接种、接种时间在季节中的早晚与历史接种频率。
* **医疗使用与健康：** 呼吸道疾病诊断、慢病、处方药数量、就诊次数、住院记录。
* **人口与社区：** 年龄、性别、社会经济分数、诊所所在地区、诊所历年接种率。
* **家庭/社会：** 家庭成员之前流感季的接种记录、家庭成员间的相似性。

### 模型与主要结果

研究比较 Logistic Regression、Naïve Bayes、XGBoost、LightGBM 和神经网络，并设计 Age-only、Sociodemographic、Previous-season、Previous-season + Sociodemographic 等较简单的参照。训练/验证/测试按 **70%/15%/15%** 划分，并使用交叉验证及调参。

|代表模型/设置|ROC-AUC|结果含义|
|-|-:|-|
|XGBoost / LightGBM，基本个人数据|**0.91**|基于过去医疗与接种记录的判别能力很高|
|XGBoost / LightGBM，含家庭记录的子集|**0.88**|家庭子集也较强，但与基本样本不是严格同分布比较|
|Previous-season + Sociodemographic 简化基准|**0.87**|仅利用既往接种等少数关键信息，就已具有较强预测力|

### 最重要的领域发现

**发现 1：个人以往的流感接种史是最重要的预测信息之一。** 在 XGBoost/LightGBM 的特征重要性排序里，前一、前二、前三个流感季的接种信息长期居前。作者还观察到一种 *default / habit-like behavior*：过去接种过的人，下季继续接种的概率更高；越频繁接种的人，往往也越早接种。这类规律与行为惯性或接种习惯的解释相符。

**发现 2：医疗接触具有增量信息。** 年龄、医疗系统接触次数、处方药数量与慢病出现在重要变量中；此前有呼吸道疾病的未接种者，之后接种概率更高。

**发现 3：家庭与地区具有明显关联。** 如果同一个接种季有至少一位家庭成员接种，个人接种概率显著更高；不同诊所/地区的覆盖率也不同。家庭与地区接种模式体现了社会信息的相关性；用于前瞻预测时需注意信息的时间先后。

**发现 4：传统 ML 的预测基准已相当强。** 完整医疗历史下的 XGBoost/LightGBM 表现突出，简单的既往接种 + 人口信息组合也已具有很强判别力。

---
## 8. [V2] Comparative Evaluation of Machine Learning Models for Predicting COVID-19 Vaccine Uptake in U.S. Adults

**Maxey, Griffin, Powla & Pabon-Rodriguez (2026)** · *BMC Public Health* 26:1625。  
[期刊页面](https://doi.org/10.1186/s12889-026-27199-2) · [开放全文（PMC）](https://pmc.ncbi.nlm.nih.gov/articles/PMC13196073/)

### 研究背景与核心问题

美国成年人 COVID-19 疫苗接种存在显著差异，已有可观测的人口和医疗资料。作者进一步研究 **对联邦统计及公共统计机构的信任** 是否与接种有关，同时比较常用机器学习方法的判别能力。重要研究问题包括：哪些传统算法更有效？哪些 Feature 真正提供最多预测信息？移除其中最强的变量后，模型还剩多少预测能力？

### 数据、目标与 Feature 类型

* **数据集：** 美国 Census Bureau **Household Pulse Survey** 三轮横截面调查：2024 年 10 月、2024 年 12 月至 2025 年 1 月、2025 年 2–3 月。论文 Methods 的分析样本口径为每轮经排除后的 **9,022、9,034、6,463**，合计 **24,519** 份记录；各表格因特征缺失与统计口径可能有略微不同 N。
* **目标：** 2024–25 接种季是否报告已接种 COVID-19 疫苗（不是预测同一个体未来一年是否接种）。
* **人口/社会经济：** 年龄、性别、种族、教育、就业、婚姻、儿童数量、交通可及性。
* **健康与医疗：** 保险类型及**同一接种季的流感疫苗接种情况**。
* **信任与态度：** 对联邦统计、美国 Census Bureau 及联邦统计机构的信任/信心。
* **验证：** 前两轮作为训练，第三轮作为测试；多重插补处理缺失；交叉验证调参；使用 SHAP 和特征排除敏感性分析。

### 使用的方法与模型比较

|方法|Accuracy|ROC-AUC|发现|
|-|-:|-:|-|
|Random Forest|**0.816**|**0.895**|该研究中总体最佳|
|XGBoost|0.805|0.885|性能接近 RF|
|Neural Network (MLP)|0.773|0.868|没有比 RF/XGBoost 更好|
|Logistic Regression|0.769|0.867|较简单但仍有可观判别力|
|Linear Discriminant Analysis|0.769|0.868|与 LR/MLP 接近|

### 最有说服力的证据：少数特征贡献了大部分可用预测信息

作者的 SHAP 排名显示，**流感疫苗接种状态**最突出，接下来是年龄、儿童数量、对联邦统计及统计机构的信任变量。更重要的是，他们没有停留在“SHAP 排名”，而做了删除特征的判别能力测试：

|Random Forest 设置|ROC-AUC|
|-|-:|
|全部可用特征|**0.895**|
|去掉流感接种变量|**0.795**|
|再去掉年龄|**0.728**|

**发现 1：少数高度相关的特征能主导整体预测性能。** 仅移除流感接种变量，AUC 就下降 0.100；再移除年龄，又下降 0.067。

**发现 2：其余变量仍有增量价值。** 去掉上述两个强变量后，RF 仍有 0.728 AUC，这说明信息并非全由单一变量决定。

**发现 3：有意义的态度/制度信任特征不只来自直接接种意愿。** 在这项调查里，联邦统计信任变量进入重要特征序列，但它在模型中的关联性不等于改变信任就会直接导致接种率变化。

**发现 4：复杂模型不一定更好。** 本研究中 MLP 的表现低于 RF/XGBoost。

---
## 9. [V3] Identifying Psychological Predictors of SARS-CoV-2 Vaccination: A Machine Learning Study

**2024** · *Vaccine*。  
[DOI](https://doi.org/10.1016/j.vaccine.2024.126198) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/39106578/)

### 研究背景与核心问题

疫苗服务可用后，如何识别那些**会长期保持未接种状态**的人？已有研究发现疫苗态度与接种相关，但如果要支持干预对象识别，就需要验证心理变量能否**在基线时预判半年后的真实结果**。作者因此从单次接种意愿研究走向前瞻性行为预测。

### 数据、目标与 Feature 类型

* **样本与时序：** 疫苗广泛开放接种的 2021 年，基线时尚未接种的 **325 人**接受调查，并在 **2021 年 4–10 月约六个月**的时间窗内随访。
* **目标：** 是否**持续未接种 COVID-19 疫苗**（而不是基线愿不愿意接种）。研究报告其中有 **80 名持续未接种者**。
* **人口与健康：** 年龄等人口信息，既往流感接种记录等医疗历史。
* **直接态度/理论构念：** COVID-19 接种意愿、Health Belief Model 相关变量、对疫苗危险性/副作用的感知。
* **认知/心理：** 阴谋论倾向，以及某些与识别虚假模式、阴谋论易感性有关的任务测量。

### 方法与结果

主要使用 **Random Forest**，以基线 Features 预测随访结果，并用 Permutation Importance 分析哪些变量最有信息量；按预先的阈值调整协议选择分类阈值。

|指标|研究结果|解释|
|-|-:|-|
|Precision–Recall AUC|**0.77** (95% CI 0.56–0.90)|对持续未接种类别的识别具有较好 PR 表现|
|Precision（持续未接种）|**0.83**|被判为持续未接种的人中，大部分确实如此|
|False-positive rate|**0.04**|按文中定义，错误识别为持续未接种的比例较低|

### 核心发现：直接心理变量的价值并不是抽象猜测

该研究的 Permutation Importance 依次强调：

1. **Baseline vaccination intention**：对将来是否仍不接种最有信息量；
2. **Conspiracist ideation**：第二个重要因素；
3. **Perceived vaccine dangerousness**：与疫苗风险/障碍认知相关；
4. **History of influenza vaccination**：过往疫苗行为仍提供预测信号。

**发现 1：即使都属于基线未接种群体，后续行为仍可被心理信息进一步区分。** 仅凭初始“还没接种”无法知道谁六个月后仍不接种。

**发现 2：心理变量提供了客观人口与健康资料之外的预测信息。**

**研究定位：** 文章侧重前瞻性心理预测变量，而非全面的模型 Benchmark。


---
## 10. [V4] Influencing Factors and Dynamic Changes of COVID-19 Vaccine Hesitancy in China: From the Perspective of Machine Learning Analysis

**Lei Li et al. (2025)** · *Human Vaccines & Immunotherapeutics*, 21(1):2536898。  
[开放全文（PMC）](https://pmc.ncbi.nlm.nih.gov/articles/PMC12320845/) · [DOI](https://doi.org/10.1080/21645515.2025.2536898)

### 研究背景与核心问题

作者不只问“什么样的人更犹豫接种”，而是问**接种从初始剂次进入加强针阶段以后，重要预测因素是否改变**。这与疫苗推广阶段、信息环境以及人们已有接种经验有关。该研究以中国两个时间点的调查分析疫苗犹豫的相关因素，并把重点放在变量排序的变化，而不是只寻找一个最高的 AUC。

### 数据、目标与 Feature 类型

* **数据：** 2021 年 4 月和 11 月在安徽、河南、湖北、河北开展两阶段调查；有效样本分别 **1,409** 和 **1,082**。原文称跟踪/复访调查，样本规模不完全相同。
* **目标：** COVID-19 **初始接种与加强针犹豫**；作者结合**自报实际接种行为与接种意愿/资格**构造，因此不能直接等同于下季实际接种的预测任务。
* **特征规模：** 四个维度合计 **24 个 Features**：

  * Individual：年龄、性别、教育、收入、疫苗知识、既往接种经历；
  * Contextual / Background：社会规范、社会动员、医疗信任、社会责任、信息关注；
  * Disease-related：感知易感性、感知严重性、疫情担忧；
  * Vaccine-related：疫苗安全性、有效性、必要性、预期后悔等。
* **方法：** 先比较多种 ML，选择表现较好的 **Gradient Boosting Decision Tree（GBDT）**，再用 SHAP 排序、识别核心因素并比较两阶段变化。主要不是 LLM 研究。

### 初始接种与加强针：排名和类型真的会变

|初始接种 Hesitancy 的重要变量|排名|加强针 Hesitancy 的重要变量|排名|
|-|-:|-|-:|
|Social Norms|**1**|Age|**1**|
|Vaccine Knowledge|2|Vaccination Experience|**2**|
|Anticipated Regret|3|Vaccine Knowledge|3|
|Age|4|Vaccine Effectiveness|4|
|Vaccine Safety|5|Gender|5|
|Social Responsibility|6|Perceived Severity|6|
|Education|7|Epidemic Concerns|7|
|Religion|8|Social Norms|8|
|Vaccine Effectiveness|9|Anticipated Regret|9|
|Perceived Severity|10|Social Responsibility|10|

### 研究已经发现了什么？

**发现 1：疫苗心理态度及社会环境信息重要，但没有固定的“永远最重要的构念”。** 在初始接种时，社会规范、知识、预期后悔和安全性感知很突出；到加强针阶段，年龄、既往接种经历、有效性感知和疾病严重性感知相对更重要。

**发现 2：信息价值会随接种阶段改变。** 社会规范从第 1 下降到第 8；年龄从第 4 上升到第 1；疫苗有效性感知从第 9 上升到第 4。疫苗知识在两阶段都位居前列。

**发现 3：与疫苗直接相关的信息提供了人口特征之外的预测信号。** Vaccination Experience 在加强针模型里排名第 2，强调了接种经历与行为阶段之间的关系。

**研究定位：** 其 Outcome 是 Vaccine Hesitancy，因此主要用于理解接种犹豫的预测因素及其阶段变化。


---
## 11. [V5] Comparing the Algorithmic Fidelity of Large Language Models in Predicting Human Decision Making: A Case Study of Vaccination Choice

**Shaochong Xu et al. (2026)** · *npj Digital Public Health* 1:20。  
[期刊原文](https://www.nature.com/articles/s44482-026-00026-6)

### 研究背景与核心问题

此前 ML 已能预测某些疫苗行为，但使用通用 LLM 直接扮演一个真实受访者来**预测其自报接种选择**，存在特别的可靠性问题：模型可能有与目标人群不一致的默认倾向；如果只知道受访者人口特征，是否仍能区分已接种和未接种？当直接态度信息不可用时，能否利用新闻信息环境作为某种**态度代理（proxy）**？最后，不同 LLM 对同样信息的反应是否一致？

### 数据来源、真实标签和 Feature 条件

* **真实样本：** 2022 年美国 *COVID-19 Vaccine Hesitancy in the Pandemic’s Third Year* 调查子样本 **1,000 人**，其中 **80.6% 已至少接种一剂，19.4% 未接种**。
* **真实 Outcome：** 受访者自报是否至少接种过一剂 COVID-19 疫苗；不是将来接种的前瞻性结果。
* **Demographics：** 年龄、性别、教育、收入，输入为个人自然语言 Profile。
* **Direct Survey Attitudes：** 基础人口特征 + **7 个直接调查态度题**，例如疾病风险、疫苗安全性、信任与信息关注等。
* **Media Proxy：** 基础人口特征 + 五篇从真实历史新闻材料中抽取的文章片段。文章被归到不同媒体类别（左翼回音室、中间立场、右翼回音室、低可信度）形成*模拟*的个人信息环境。
* **模型：** GPT-4.1、GPT-4o-mini、Gemini-2.5-Flash、Llama-4-Scout-17B、Grok-4.1-fast-non-reasoning，使用重复种子衡量随机性。

### 三个主要模型设定和预测效果

|输入设定|模型知道什么|主要表现|
|-|-|-|
|**LLM-Demographics**|只知道人口特征|多个模型几乎全部预测接种；不能识别未接种群体|
|**LLM-Survey**|人口 + **真实自报接种态度**|**Balanced Accuracy 0.821–0.852**|
|**LLM-Media**|人口 + **根据态度构造的模拟新闻信息**，不直接显示态度题答案|**Balanced Accuracy 0.701–0.772**|

#### 最突出的负面发现：Demographics Only 时严重偏向预测“会接种”

GPT-4.1、GPT-4o-mini 和 Llama-4-Scout-17B 在 Demographics Only 条件下，**预测接种比例 ≥99%**。其中 GPT-4o-mini、Llama-4-Scout-17B 的 **Balanced Accuracy 均为 0.500**，GPT-4.1 为 **0.508**。这不是因为这些模型完全不会输出接种相关语言，而是因为给出的个人信息不足以让它们可靠地区分受访者真实的接种类别，模型又呈现很强的默认接种倾向。

换句话说，**一个模型可以在占多数的“已接种”类别上看似表现不错，却几乎完全丧失识别“未接种”类别的能力**；只报告总体 Accuracy 会掩盖这一问题。

#### 直接态度确实大幅改善预测

在 LLM-Survey 条件下，五种模型的 Balanced Accuracy 都明显改善到 **0.821–0.852**。研究表明，**直接观察到受访者对疫苗的安全、效果、风险等看法，与仅凭人口信息推断人的态度，是两种截然不同的信息条件**。这也是该文最有说服力的实证结果之一。

#### Media：把模拟信息环境用作态度代理

**研究设想：** 媒体信息的立场、可信度与呈现方式可能反映疫苗相关态度，因此作者将 Media 作为 Direct Attitudes 不可直接输入时的 **proxy**。

**Media 条件的构造：** 作者先用受访者填写的疫苗态度题将其分配到一种模拟 media diet，再从真实新闻材料库抽取片段。因此新闻材料是真实的，但个体信息环境是**基于已知态度构建的模拟条件**；态度信息通过媒体分配间接进入模型。

**研究结果：** 在模拟的信息环境中，LLM 能利用媒体内容恢复一部分与接种结果有关的信号，但效果仍低于直接输入态度。

在 LLM-Media 中，表现虽低于直接态度条件，但明显高于只有 Demographics 的若干模型。不同模型的类别偏好却有很大区别：

|LLM-Media 模型|Balanced Accuracy|已接种 Recall|未接种 Recall|显示出的问题|
|-|-:|-:|-:|-|
|GPT-4o-mini|**0.772**|0.804|0.740|相对均衡|
|GPT-4.1|0.748|0.877|0.619|更偏向预测接种|
|Gemini-2.5-Flash|0.739|0.566|0.912|对未接种识别较强、已接种识别变弱|
|Grok-4.1-fast-non-reasoning|0.701|0.435|0.967|对未接种极敏感，但大量牺牲已接种 Recall|
|Llama-4-Scout-17B|0.715|—|—|处于中间表现；详见原文 Table 1|

论文还报告不同假设媒体条件下的反事实模拟：大量低可信度材料会使某些模型预测的接种比例大幅下降；大量权威公共卫生信息会使多数模型更倾向于预测接种。这些实验主要刻画模型面对不同信息环境时的预测变化。

#### 不同 LLM 具有不同的预测倾向

作者还检验了 Prompt 的人称（role-play vs. third-person）、信息顺序（人口资料先 vs. 新闻先）及输出格式。改变提示方式时，预测接种比例可能明显变化。例如 GPT-4o-mini 在第三人称设定下，预测接种比例从约 69.8% 降到 37.6%。由此说明：即使相同的受访者和数据，**架构、提示方式、信息位置**也会改变最终行为预测。

**研究边界：** 论文重点分析 LLM 对输入类型与提示方式的敏感性，未进行完整的 SFT 与传统 ML 对照。

---
## 12. 五篇文章的共同发现

### 综合证据矩阵

|研究主题|领域已经发现什么|主要文献|这一证据反映的内容|
|---|---|---|---|
|**同类疫苗的既往接种史**|过去几个流感季的接种情况是下一季流感接种的强预测变量|[V1]|行为延续性和习惯相关信号|
|**相关疫苗的接种记录**|流感接种与 COVID-19 接种状态高度关联，少数特征贡献大量判别力|[V2]|相关健康行为之间的预测关联|
|**人口、健康与医疗使用**|年龄、医疗接触、慢病、家庭/地区信息提供增量预测信号|[V1]、[V2]|客观背景信息及医疗接触|
|**直接接种意愿**|在基线尚未接种者中，对半年后的持续未接种具有较高预测价值|[V3]|态度与后续行为的联系|
|**疫苗风险及阴谋论倾向**|在前瞻性 COVID-19 研究中属于重要预测因素|[V3]|疫苗相关心理信念|
|**社会规范、知识、安全与有效性信念**|与接种犹豫相关，且初针和加强针阶段的特征重要性排序会改变|[V4]|预测因素的阶段差异；Outcome 为 Hesitancy|
|**LLM + Demographics**|部分模型几乎全部预测接种，未接种类别识别能力较差|[V5]|默认类别偏好与输入信息不足|
|**LLM + Direct Attitudes**|直接加入疫苗态度后，五种模型的 Balanced Accuracy 明显提升|[V5]|直接观测态度的预测价值|
|**LLM + Simulated Media**|基于已知态度构造的模拟媒体环境提供了部分有效信号，但模型差异较大|[V5]|间接信息代理及其局限|

**第二部分的整体结论：** 传统 ML 已经建立了有效的疫苗接种预测基准；既往接种史和少数关键变量经常具有较高预测力；直接疫苗态度与心理信息进一步解释了不同人群的接种差异；针对真实个体接种结果的 LLM 研究仍相对有限，现有工作主要揭示了其对输入信息和默认类别偏好的敏感性。

# 第三部分｜NHIS 论文研究框架（初步讨论稿）

## 13. 核心研究问题与整体安排

**Central RQ — How do model adaptation, available information, and systematic prediction errors shape the performance of LLMs in individual vaccination behavior prediction, relative to conventional ML?**

以 **NHIS 2024 季节性流感接种结果**为主任务，先比较预测能力，再解释有效信息与错误模式，最后用含直接心理态度变量的第二个数据集检验结论在不同信息条件下是否成立。

| 阶段 | 核心问题 | 主要实验与分析 | 进展 |
|---|---|---|---|
| **Study 1 · Predictive Capability & Adaptation** | LLM 能预测得多好？Prompting 和 SFT 能改善多少？ | Frozen / HBM Prompt / LoRA SFT / ML；Label Budget | 以已有结果为主 |
| **Study 2 · Information Sufficiency** | 预测力是否集中于少数特征？信息减少时谁更稳健？ | Top-K / Random-K / Bottom-K；关键变量剔除；Instance-wise Missingness | 已有实验 + 少量补充 |
| **Study 3 · Error Anatomy** | LLM 具体怎样出错？SFT 修正了哪些错误？ | 类别偏差、TPR/TNR、受访者级错误重叠、亚组差异、阈值对照 | 重点补充分析 |
| **Study 4 · Direct Behavioral Information** | 直接疫苗态度出现后，LLM/ML 的相对表现会改变吗？ | 第二数据集；Objective / Vaccination History / Attitudes 信息组合 | 拟新增 Study |

## 14. Study 1 — Predictive Capability and Model Adaptation

**RQ1：** *How well can LLMs predict actual vaccination outcomes, and how much can task adaptation narrow their gap with conventional ML?*

- **主实验：** 同一 NHIS 测试样本上，比较 Majority Class、LR、RF、XGBoost/GBDT、Frozen LLM、HBM Prompting、LoRA SFT。核心指标为 Accuracy、Balanced Accuracy、F1、AUC；有可靠预测概率时增加 Brier Score / Calibration。
- **适应效果：** 对照 Frozen → HBM Prompt → SFT，区分理论提示与真实标签微调各自带来的收益。
- **Label Budget：** 整合已有的 50/100 个样本及不同训练比例，展示 SFT 和 ML 的学习曲线与性能平台期。

**预期呈现：** 一张统一的模型对比表 + 一张 Label Budget 学习曲线，回答“基础能力如何、适应训练改善多少、需要多少标签”。

## 15. Study 2 — Information Sufficiency and Robustness

**RQ2：** *How concentrated is the predictive information in observed respondent features, and how do different models respond when this information is reduced or missing?*

- **重要特征：** 已确认 Top-4 均为其他疫苗接种史相关变量（COVID-19 接种剂数、肺炎球菌、带状疱疹及 COVID-19 是否接种）；沿用 Top-K、Random-K、Bottom-K（62 → 4）实验，观察性能如何随特征数变化。
- **关键变量检验：** 补充 Top-1 / Top-2 / Top-4 与 **Full minus Top-1 / Top-4**，判断少数变量的独立预测力，以及移除核心信号后模型差距是否变化。
- **缺失信息：** 整合已有的 respondent-wise random missingness、Full-feature Training 与 Mixed-mask Training，对比同一缺失预算下 Frozen、SFT、ML 的表现。

**预期呈现：** Feature Budget 曲线 + 关键变量剔除结果 + Missingness 鲁棒性图。

## 16. Study 3 — Systematic Prediction Bias and Error Anatomy

**RQ3：** *What systematic errors distinguish Frozen LLMs, SFT models, and ML, and which errors are corrected by adaptation?*

- **类别偏差：** 比较真实与预测接种率、TPR/TNR、Balanced Accuracy、MCC 和 Confusion Matrix，检查 LLM 是否系统性偏向 Yes/No。
- **个体错误：** 在相同 Test80 respondent 上比较 **ML 对但 Frozen 错、SFT 修正 Frozen 错、SFT 新增错误**等类别，并分析它们与年龄、既往接种史、医疗接触等变量的关系。
- **SFT 改善来源：** 判断 SFT 主要是纠正类别偏好，还是改善个体区分能力；有可比较的概率输出时增加 AUC / Calibration，并使用 **Frozen + validation-set threshold adjustment** 作简单对照。

**预期呈现：** 类别预测分布 / 错误类型图 + SFT 错误修正分析，解释“模型在哪些条件下失准”，不只报告总体 Accuracy。

## 17. Study 4 — Direct Behavioral and Attitudinal Information

**RQ4：** *Does directly observed vaccination-related attitude information change the predictive performance and relative advantage of LLMs and ML?*

**数据需求：** 寻找同时包含**实际接种 Outcome**、人口与健康信息、既往接种记录，以及**直接疫苗态度**（例如接种意愿、安全性/有效性感知、风险感知）的第二数据集。

| 信息设置 | 包含的变量 | 主要比较 |
|---|---|---|
| **S1** | Demographics + Health (D+H) | 客观背景的基础预测能力 |
| **S2** | D+H + Vaccination History (V) | 既往接种历史的额外价值 |
| **S3** | D+H + Direct Attitudes (A) | 直接态度信息的额外价值 |
| **S4** | D+H+V+A | 信息完整时的模型表现 |

在四种设置下比较 **Frozen、HBM Prompting、SFT 与 LR/RF/XGBoost**；重点观察加入 Attitudes 后各模型的性能增量，以及 HBM Prompting 的作用是否随直接态度可见而变化。

**预期呈现：** Model × Information Setting 结果矩阵 + 加入态度变量前后的性能变化图。新增数据集内部使用统一样本划分进行比较，避免把不同疫苗或人群的绝对分数直接归因于 Feature Type。

## 18. 研究逻辑与下一步优先级

**整体叙事：** **Predictive Performance → Information Sufficiency → Systematic Errors → Behavioral Information Boundary**。

- **先整理：** Study 1 的主结果与 Label Budget；Study 2 已确认的 Top-4 变量和已有 Feature Budget / Missingness 结果。
- **优先补充：** Study 3 的完整 Error Analysis，以及 Study 2 中少数关键变量的剔除对照。
- **并行确认：** 是否能找到满足实际接种标签与直接态度测量要求的第二数据集，再决定 Study 4 的最终规模。

**组会希望讨论的问题：** 这四个研究阶段能否支撑一篇完整的 *Marketing Science* empirical paper？Study 4 是否值得作为主要新增实验？

