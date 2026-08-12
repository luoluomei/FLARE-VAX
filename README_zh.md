# FLARE-VAX：基于 NHIS 2024 的行为理论引导流感疫苗接种预测

[English README](README.md)

本项目使用 **2024 NHIS Sample Adult** 数据预测受访者过去 12 个月是否接种流感疫苗（`SHTFLU12M_A`）。开发时使用的原始 `adult24.csv` 约包含 **32,629 名受访者、630 个变量**。V4 最终纳入 32,132 人，V5 纳入 32,130 人，默认按 **40% memory / 20% calibration / 40% test** 划分。仓库不公开原始 NHIS 数据、逐样本预测和 API 日志。

> 本项目中的 HBM 分数是由 NHIS 可观察变量构造的理论代理，并不是对个体私有心理信念的直接测量。

## 1. V4 与 V5

- **V4（包含其他疫苗史）**：允许 COVID、肺炎、带状疱疹和甲肝等非目标疫苗史进入 vaccine acceptance/benefit proxy，共 75 个 ML 特征。
- **V5（排除其他疫苗史）**：从分数、prompt、retrieval 和 memory 中排除全部非目标疫苗史，用 wellness、健康信息查询、医生沟通和结果查看等非疫苗预防行为替代，共 67 个 ML 特征。

两者都构造 threat、acceptance/engagement、barriers、healthcare cues、navigation self-efficacy 五个 proxy，再合并成 Motivation、Capability、Activation，并形成 8 个行为 pattern。Pattern prior 只从 memory split 估计，reflection 只基于训练侧错误建立并在 calibration/test 前冻结。


## 新增思路拓展：RG-FLARE-VAX 与 TRBM-FLARE-VAX

下面两个方法都是在原 V4/V5 基础上的进一步拓展，而不是新的 feature version。V4 仍然允许使用其他疫苗史，V5 仍然完全排除其他疫苗史；五个 HBM proxy、HBM8 pattern 以及原有的数据边界都继续保留。

两个改版主要回答两个新的问题：

1. 原来的 HBM8 pattern prior 比较粗，能不能先得到一个更个体化、更稳定的 numerical prior？
2. Reflective memory 不应该只依赖 LLM 自己判断“这条经验是否合理”，能不能进一步利用历史数据判断这条经验是否真的有预测价值？

### RG-FLARE-VAX：Reward-Guided HBM Integration + Reward-Valued Memory

代码：`scripts/81_rg_flare_vax_reward_memory_asu.py`

RG 可以理解为在原 FLARE-VAX 中增加了一层 **reward-guided probability calibration**，同时重新定义 reflective memory 的价值。它仍然保留原来的 HBM8 pattern，但不再直接把 pattern vaccination rate 当作最终的 probability anchor，而是先根据五个 HBM construct 得到一个更个体化的 `P_reward`，再让 LLM 和 memory 处理这个 numerical prior 仍然没有解释好的部分。

整体逻辑可以概括成四个阶段：

**1. 先沿用原 V4/V5 的 HBM 表征，得到最初的 behavioral anchor。**  
每个 respondent 仍然先得到 threat、acceptance/engagement、barrier、cue、self-efficacy 五个 proxy，再形成 Motivation、Capability、Activation 和 HBM8 pattern。Memory split 中每个 pattern 的历史 vaccination rate 继续记为 `P_pattern`，表示“这一类 respondent 平均有多大概率接种”。

**2. 在 `P_pattern` 基础上学习更个体化的 `P_reward`。**  
原版中，只要两个人属于同一个 HBM8 pattern，他们的初始 probability 基本相同；但实际上，他们在 threat、barrier、cue 等具体 construct 上仍然可能差很多。RG 因此额外拟合一个 cross-fitted numerical reward model，学习某个 respondent 在五个 HBM dimensions 上相对同 pattern 平均水平的偏离应该怎样调整 `P_pattern`，从而得到更细的 `P_reward`。

这里还可以加入一次 pattern-level 的 LLM guidance。LLM 看到的是某个 pattern 的整体 prediction error 和各 construct contribution，而不是单个 respondent；它只能建议少量 construct weight 应该稍微增加、降低或保持不变。这个建议不会直接采用，只有在独立的 OOF validation data 上确实降低 log loss 时才会被接受。因此这一层的逻辑是：

`LLM proposes → data validates → accept / reject`

**3. 以 `P_reward` 为新的 probability anchor，建立 reward-valued reflective memory。**  
在 memory sample 上，LLM 不再从 `P_pattern` 开始，而是把 `P_reward` 当作当前已经比较可靠的基础判断，再根据 respondent 更具体的 observed profile 做一个较小的 residual adjustment。也就是说，LLM 的任务不是重新预测一次，而是判断：

> 在 HBM + reward layer 已经给出 `P_reward` 之后，这个人还有没有一些更细的 observable evidence，意味着 probability 应该略微提高或降低？

高置信度错误会再次进入 reflection，LLM 将错误原因总结成 reusable correction rule。

RG 与原版最重要的区别是：这些 memory 不再只根据 reflection confidence 或语言上是否合理来判断价值。每条 rule 还会得到一个 empirical **Q-value**。系统会找历史上与该 memory source case 相似的 respondents，并检查：如果按照这条 rule 建议的 increase/decrease 方向去小幅修正他们的 probability，prediction loss 是否真的下降。

因此一条 memory 的价值同时来自：

`semantic plausibility + empirical predictive utility`

也就是说，它不仅要“解释得通”，还需要“历史上真的有用”。

**4. 检索相似且高价值的 memory，再让 LLM 做最终小幅修正。**  
到了 calibration/test 阶段，retrieval 不再只看 respondent similarity，也会考虑 memory 的 Q-value。系统更倾向于返回那些既和当前 respondent 相似、又在历史上确实改善过预测的 correction rules。

最终 LLM 输入：

`P_reward + 当前 respondent profile + retrieved reward-valued memories`

然后判断哪些 memory 真正适用、是否存在 contradiction，以及 probability 应该略微 increase、decrease 还是保持不变。最终仍然只允许围绕 `P_reward` 做一个 bounded residual correction，再由 calibration 选择 classification threshold。

因此，RG 相比原版最核心的变化可以概括成两点：

> **第一，把粗粒度的 HBM8 pattern prior 进一步学习成更个体化的 reward-calibrated prior；第二，把 reflective memory 从“LLM 生成的经验规则”升级成“经过历史 prediction reward 验证的经验规则”。**

LLM 本身不进行 fine-tuning。

### TRBM-FLARE-VAX：Theory-Residual Behavioral Memory

代码：`scripts/82_trbm_flare_vax_asu.py`  
Ablation：`scripts/83_trbm_ablation_asu.py`

TRBM 在 RG 的基础上进一步限制了 LLM 对 numerical prediction 的影响。它的核心思想是：**先让 HBM theory 自己形成一个明确的基础 probability，再专门学习“HBM theory 在什么情况下会失效”。**

因此，TRBM 不再让 LLM 直接决定 probability 或 residual magnitude。LLM 主要负责识别和匹配 **theory-failure mechanism**，而真正的 numerical correction 来自历史数据。

整体流程可以理解成四个阶段：

**1. 先把五个 HBM construct 合成为一个明确的 theory prior。**  
TRBM 仍然使用原 V4/V5 的五个 HBM construct，但先把方向统一成“数值越高，理论上越支持 vaccination”。之后，只使用这五个理论变量拟合一个带 non-negative constraint 的小型 logistic model，得到每个 respondent 的 `P_HBM`。

这里的重点是：`P_HBM` 不是普通 full-feature ML prediction。模型没有使用完整 NHIS raw feature set，而是只使用 HBM-derived constructs，因此它代表的是：

> **如果只按照当前 HBM theory 表征来判断，这个人应该有多大概率接种。**

这样，TRBM 先把“theory 本身能解释多少”单独固定下来。

**2. 再从历史数据中找出 HBM theory 明显解释失败的 case。**  
在 memory split 中，TRBM 使用 cross-fitting 为每个 respondent 得到一个 OOF `P_HBM`，然后比较这个 theory prior 与真实 vaccination outcome 的差异：

`theory residual = actual - P_HBM_OOF`

如果一个人的 residual 很大，就说明当前五维 HBM representation 对这个人的实际行为出现了明显偏差。例如：

- `P_HBM` 很低，但这个人实际接种了：theory 明显低估；
- `P_HBM` 很高，但这个人实际没有接种：theory 明显高估。

因此 TRBM 的 memory 不是从“LLM 哪里预测错了”开始，而是从：

> **HBM theory 本身在哪些 historical cases 上解释得不够好？**

开始建立。

**3. 让 LLM 总结这些 theory failure 为什么发生，并在新 respondent 上判断 mechanism 是否能够迁移。**  
对于 residual 较大的历史 case，LLM 在 memory-building 阶段读取它的 HBM state、`P_HBM`、真实 outcome 和更完整的 observed profile，然后分析：

> 当前 HBM prior 为什么会在这个人身上高估或低估 vaccination behavior？还有什么可观察的机制没有被五个 HBM construct 充分表达？

LLM 把这些 failure 总结成 reusable mechanism memory，例如某类 healthcare engagement、access/capability gap、preventive habit 或 proxy measurement gap。

当 calibration/test 中出现新的 respondent 时，系统先检索与他在 theory state 和 observed context 上相似的 historical theory-failure memories。此时 LLM 只负责做一个 **mechanism gate**：

- 这些 historical failure mechanism 是否真的适用于当前 respondent？
- 如果适用，历史证据支持 increase 还是 decrease？
- 如果当前 profile 与 memory 有明显 contradiction，则不使用 correction。

这里 LLM **不重新输出 probability，也不决定具体应该加减多少**。它只回答“这条历史 failure mechanism 能不能迁移到当前人”。

**4. correction magnitude 由 historical residual 决定，并由 calibration 控制最终使用强度。**  
如果 LLM 判断某些 memory mechanism 可以迁移，系统就读取这些 historical memory 当时真实的 signed residual，根据 respondent similarity 和 memory confidence 进行加权，得到一个 empirical numerical correction。

也就是说：

`LLM 决定 mechanism 是否适用`  
`历史 residual 决定 correction 大小`

最后，calibration split 再选择一个全局 correction scale `alpha`：

`P_TRBM = P_HBM + alpha × empirical residual correction`

`alpha` 决定最终应该多大程度相信这些 historical residual correction。如果 calibration 发现 memory correction 没有带来稳定 improvement，也可以直接选择 `alpha = 0`，此时最终 prediction 就退回到原来的 `P_HBM`。

因此 TRBM 的 division of labor 非常明确：

> **HBM theory model 负责基础 probability；historical residual 负责 numerical correction magnitude；LLM 只负责判断某个 historical theory-failure mechanism 是否适用于当前 respondent。**

这也是 TRBM 与原版 FLARE-VAX 最大的区别：它进一步把 numerical probability estimation 从 LLM 中拆出来，让 LLM 更像一个 **semantic mechanism matcher**，而不是直接的 probability predictor。

三个版本可以简单对比为：

| 方法 | Base probability | Prediction 时 LLM 的主要任务 | correction magnitude |
|---|---|---|---|
| 原始 FLARE-VAX | HBM8 pattern prior | 根据个体 evidence 直接做 residual reasoning | LLM 决定 |
| RG-FLARE-VAX | Reward-calibrated HBM prior | 读取 Q-valued memory 后做小幅 residual correction | LLM 决定，但范围更小 |
| TRBM-FLARE-VAX | Theory-constrained `P_HBM` | 判断 historical theory-failure mechanism 是否适用 | Historical residual + calibration `alpha` |

## 2. Zero-shot / Few-shot baseline

代码：`scripts/60_llm_icl_benchmark_asu.py`

- Zero-shot：不提供示例。
- Random balanced 8-shot：固定 4 个 YES + 4 个 NO。
- Similarity-selected 8-shot：每个目标样本分别检索最相似的 4 个 YES + 4 个 NO。
- Representative 8-shot：每类用 KMeans 找 4 个代表中心，再选择最近的真实样本。
- Random 8-shot + generic CoT：与随机 8-shot 使用相同示例，但要求一般性的正反证据推理，不使用 HBM 理论。

这些 baseline 都不接收 HBM score、pattern、pattern prior、reflective memory 或 FLARE correction rule。

## 3. 其他论文方法的迁移

### 不需要微调大模型

**HBM-CoPB** — 原文：[Chain-of-Planned-Behaviour Workflow Elicits Few-Shot Mobility Generation in LLMs](https://arxiv.org/abs/2402.09836)。原文用 TPB 的 attitude、subjective norms、perceived behavioral control 组织移动意图推理。本项目将其替换成五阶段 HBM 疫苗推理，使用 8 个固定平衡示例，但不提供 FLARE 的确定性分数、pattern prior、retrieval 或 reflection。

**HBM-PB&J** — 原文：[Improving Language Model Personas via Rationalization with Psychological Scaffolds](https://aclanthology.org/2025.findings-emnlp.1187/)。原文先用心理学 scaffold 为既有判断生成 rationale，再用强化 persona 预测新偏好。本项目 V4 使用两次调用：第一次在不知道流感疫苗 label 的情况下构造 HBM health persona；第二次结合 persona 与 8 个 memory 示例进行预测。目前只实现 V4。

**SILIC-inspired** — 原文：[Where You Go is Who You Are](https://arxiv.org/abs/2505.17249)。原文结合 behavioral theory、LLM 引导和 IRL，从移动轨迹反推潜在 reward。本项目的初步 V4 版本把四类非目标疫苗决策视作 contextual binary choices，拟合五维 preventive reward vector，再由 LLM 完成最终预测。由于 NHIS 没有时间顺序状态转移，因此明确称为 inverse contextual choice，而不是 sequential IRL。LLM 不微调，只优化外部 latent parameters。**状态：代码已加入，完整结果待跑。**

### 需要监督微调

**Persona-aware and Explainable Bikeability Assessment** — 原文：[arXiv:2601.03534](https://arxiv.org/abs/2601.03534)。原文使用 cyclist persona conditioning、多粒度 supervised fine-tuning 和 AI data augmentation 完成可解释的 bikeability 评分。**状态：仅完成方法调研，尚未迁移到 FLARE-VAX。**

## 4. 结果

### 4.1 ML baseline

| Version | Model | Accuracy | ROC-AUC | F1 |
|---|---|---|---|---|
| V4 | gradient_boosting | 0.7630 | 0.8444 | 0.7506 |
| V4 | knn | 0.7102 | 0.7689 | 0.6981 |
| V4 | logistic | 0.7604 | 0.8398 | 0.7484 |
| V4 | mlp | 0.7587 | 0.8402 | 0.7509 |
| V4 | random_forest | 0.7617 | 0.8377 | 0.7471 |
| V4 | svm | 0.7630 | 0.8356 | 0.7552 |
| V4 | xgboost | 0.7647 | 0.8452 | 0.7519 |
| V5 | gradient_boosting | 0.6867 | 0.7553 | 0.6691 |
| V5 | knn | 0.6289 | 0.6708 | 0.6119 |
| V5 | logistic | 0.6766 | 0.7433 | 0.6640 |
| V5 | mlp | 0.6824 | 0.7506 | 0.6580 |
| V5 | random_forest | 0.6829 | 0.7507 | 0.6660 |
| V5 | svm | 0.6786 | 0.7477 | 0.6636 |
| V5 | xgboost | 0.6871 | 0.7570 | 0.6697 |

### 4.2 Zero-shot / Few-shot

| Version | Model | Method | Test N | Threshold | Accuracy | Balanced Acc. | ROC-AUC | F1 | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| V4 | Llama 3 70B | Random balanced 8-shot | 12853 | 5 | 0.4737 | 0.5000 | 0.5000 | 0.6428 | complete |
| V4 | Llama 3 70B | Random 8-shot + generic CoT | 12853 | 29 | 0.5186 | 0.5399 | 0.5399 | 0.6504 | complete |
| V4 | Llama 3 70B | Representative 8-shot | 12853 | 5 | 0.4737 | 0.5000 | 0.5000 | 0.6428 | complete |
| V4 | Llama 3 70B | Similarity-selected 8-shot | 12853 | 5 | 0.4737 | 0.5000 | 0.5000 | 0.6428 | complete |
| V4 | Llama 3 70B | Zero-shot direct | 12853 | 46 | 0.4787 | 0.5048 | 0.5048 | 0.6449 | complete |
| V4 | Llama 4 Scout 17B | Random balanced 8-shot | 12853 | 86 | 0.6432 | 0.6297 | 0.6851 | 0.4974 | complete |
| V4 | Llama 4 Scout 17B | Random 8-shot + generic CoT | 12853 | 73 | 0.5830 | 0.5622 | 0.6048 | 0.2761 | complete |
| V4 | Llama 4 Scout 17B | Representative 8-shot | 12853 | 75 | 0.6053 | 0.6211 | 0.6554 | 0.6887 | complete |
| V4 | Llama 4 Scout 17B | Similarity-selected 8-shot | 12853 | 79 | 0.6189 | 0.6225 | 0.6543 | 0.6317 | complete |
| V4 | Llama 4 Scout 17B | Zero-shot direct | 12853 | 81 | 0.6339 | 0.6453 | 0.7136 | 0.6904 | complete |
| V5 | Llama 3 70B | Random balanced 8-shot | 12852 | 5 | 0.4739 | 0.5000 | 0.5000 | 0.6430 | complete |
| V5 | Llama 3 70B | Random 8-shot + generic CoT | 12851 | 51 | 0.4877 | 0.5122 | 0.5122 | 0.6448 | complete* |
| V5 | Llama 3 70B | Representative 8-shot | 12852 | 5 | 0.4739 | 0.5000 | 0.5000 | 0.6430 | complete |
| V5 | Llama 3 70B | Similarity-selected 8-shot | 12852 | 5 | 0.4739 | 0.5000 | 0.5000 | 0.6430 | complete |
| V5 | Llama 3 70B | Zero-shot direct | 12852 | 46 | 0.4774 | 0.5034 | 0.5034 | 0.6445 | complete |
| V5 | Llama 4 Scout 17B | Random balanced 8-shot | 12852 | 82 | 0.5917 | 0.5932 | 0.6229 | 0.5903 | complete |
| V5 | Llama 4 Scout 17B | Random 8-shot + generic CoT | 12852 | 73 | 0.5461 | 0.5223 | 0.5389 | 0.1248 | complete |
| V5 | Llama 4 Scout 17B | Representative 8-shot | 12852 | 75 | 0.6218 | 0.6232 | 0.6251 | 0.6197 | complete |
| V5 | Llama 4 Scout 17B | Similarity-selected 8-shot | 12852 | 79 | 0.5773 | 0.5783 | 0.6051 | 0.5725 | complete |
| V5 | Llama 4 Scout 17B | Zero-shot direct | 12852 | 81 | 0.6307 | 0.6367 | 0.6622 | 0.6585 | complete |

上周尚未完成的 Llama 3 70B benchmark 现在已经补齐。可以看到，多组 direct 8-shot 的输出仍然接近“几乎全部预测为接种”，因此 balanced accuracy 约为 0.50；这里把它作为真实 benchmark 结果保留，而不是继续标记为未完成。`*` V5 70B generic-CoT 最终返回 12,851 个 test prediction（test success rate = 0.999922），比配置的 12,852 少 1 个。

### 4.3 CoPB / PB&J 迁移结果

| Version | Model | Method | Test N | Threshold | Accuracy | Balanced Acc. | ROC-AUC | F1 | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| V4 | Llama 3 70B | HBM-CoPB | 12853 | 51 | 0.6419 | 0.6464 | 0.6698 | 0.6593 | complete |
| V4 | Llama 4 Scout 17B | HBM-CoPB | 12853 | 51 | 0.6726 | 0.6753 | 0.7131 | 0.6776 | complete |
| V5 | Llama 3 70B | HBM-CoPB | 12852 | 51 | 0.5664 | 0.5634 | 0.5719 | 0.5258 | complete |
| V5 | Llama 4 Scout 17B | HBM-CoPB | 12852 | 61 | 0.6376 | 0.6378 | 0.6652 | 0.6268 | complete |
| V4 | Llama 3 70B | HBM-PB&J | 12853 | 76 | 0.5831 | 0.5935 | 0.6243 | 0.6425 | complete |
| V4 | Llama 4 Scout 17B | HBM-PB&J | 12853 | 61 | 0.7007 | 0.6970 | 0.7610 | 0.6651 | complete |

上周尚未完成的 V4 Llama 4 Scout 17B HBM-PB&J 已补跑完成，ROC-AUC 为 0.7610，balanced accuracy 为 0.6970。V5 HBM-PB&J、SILIC 和需要微调的 persona-aware 方法目前仍没有可汇报的 FLARE-VAX 结果。

### 4.4 FLARE-VAX 主方法与 ablation

| Version | Model | Method | Test N | Threshold | Accuracy | Balanced Acc. | ROC-AUC | F1 | Status |
|---|---|---|---|---|---|---|---|---|---|
| V4 | Llama 4 Scout 17B | FLARE-VAX full run | 12853 | 51 | 0.7318 | 0.7312 | 0.7789 | 0.7174 | complete |
| V4 | Llama 3 70B | FLARE-VAX full run | 12853 | 46 | 0.7278 | 0.7285 | 0.7546 | 0.7207 | complete |
| V4 | No LLM | HBM8 pattern-only ablation | 12853 | 50 | 0.7277 | 0.7284 | 0.7693 | 0.7206 | complete |
| V5 | Llama 4 Scout 17B | FLARE-VAX full run | 11057 | 47 | 0.6101 | 0.6179 | 0.6334 | 0.6265 | complete_final_summary_11057_evaluated |
| V5 | Llama 3 70B | FLARE-VAX full run | 12852 | 47 | 0.6255 | 0.6325 | 0.6763 | 0.6597 | complete |
| V5 | No LLM | HBM8 pattern-only ablation | 12852 | 37 | 0.6255 | 0.6325 | 0.6797 | 0.6597 | complete |


### 4.5 RG-FLARE-VAX 拓展结果

| Version | Model | Stage | Test N | Threshold | Accuracy | Balanced Acc. | ROC-AUC | F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| V4 | Llama 3 70B | Reward-calibrated HBM prior | 12853 | 47 | 0.7371 | 0.7390 | 0.8218 | 0.7363 |
| V4 | Llama 3 70B | Final reward-valued memory | 12853 | 52 | 0.7384 | 0.7398 | 0.8210 | 0.7351 |
| V4 | Llama 4 Scout 17B | Reward-calibrated HBM prior | 12853 | 53 | 0.7417 | 0.7403 | 0.8216 | 0.7238 |
| V4 | Llama 4 Scout 17B | Final reward-valued memory | 12853 | 50 | 0.7390 | 0.7395 | 0.8221 | 0.7308 |
| V5 | Llama 3 70B | Reward-calibrated HBM prior | 12852 | 45 | 0.6590 | 0.6635 | 0.7223 | 0.6758 |
| V5 | Llama 3 70B | Final reward-valued memory | 12852 | 50 | 0.6604 | 0.6638 | 0.7208 | 0.6699 |
| V5 | Llama 4 Scout 17B | Reward-calibrated HBM prior | 12852 | 49 | 0.6580 | 0.6585 | 0.7222 | 0.6493 |
| V5 | Llama 4 Scout 17B | Final reward-valued memory | 12852 | 46 | 0.6604 | 0.6635 | 0.7204 | 0.6686 |

目前最明显的 improvement 来自 reward-calibrated prior：V4 的 ROC-AUC 从 HBM8 pattern-only 的约 0.769 提升到约 0.822，V5 从约 0.680 提升到约 0.720–0.722。Reward-valued memory 在部分配置中会改变 balanced accuracy、F1 和最终 operating point，但并没有在所有 probability metric 上稳定超过 reward prior，因此这里把两部分结果分开汇报。

### 4.6 TRBM-FLARE-VAX 拓展结果

TRBM 目前仍在继续补跑。本周 README **只把 V4 + Llama 4 Scout 17B 作为已经完成的结果保留**；其他计划中的组合继续列在表里，但统一用 `--` 占位，表示尚未跑完。

| Version | Model | Method | Test N | Memories | Threshold | Correction scale α | Accuracy | Balanced Acc. | ROC-AUC | F1 | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| V4 | Llama 4 Scout 17B | TRBM full | 12853 | 459 | 0.48 | 0.00 | 0.7390 | 0.7371 | 0.8205 | 0.7181 | complete |
| V4 | Llama 4 Scout 17B | TRBM full — survey weighted | 12853 | 459 | 0.48 | 0.00 | 0.7315 | 0.7236 | 0.8113 | 0.6812 | complete |
| V4 | Llama 3 70B | TRBM full | -- | -- | -- | -- | -- | -- | -- | -- | not completed yet |
| V4 | Llama 3 70B | TRBM full — survey weighted | -- | -- | -- | -- | -- | -- | -- | -- | not completed yet |
| V5 | Llama 4 Scout 17B | TRBM full | -- | -- | -- | -- | -- | -- | -- | -- | not completed yet |
| V5 | Llama 4 Scout 17B | TRBM full — survey weighted | -- | -- | -- | -- | -- | -- | -- | -- | not completed yet |
| V5 | Llama 3 70B | TRBM full | -- | -- | -- | -- | -- | -- | -- | -- | not completed yet |
| V5 | Llama 3 70B | TRBM full — survey weighted | -- | -- | -- | -- | -- | -- | -- | -- | not completed yet |

目前完成的两个 V4 Scout 17B run 中，calibration 都选择了 `alpha = 0.00`。因此现阶段最终 TRBM probability 实际上等于 theory-constrained `P_HBM`：reflection memory 和 mechanism gate 已经运行，但它们产生的 residual correction 没有被 calibration 保留下来。未加 survey weight 的版本 ROC-AUC 为 0.8205，survey-weighted 版本为 0.8113。其他 TRBM 组合等全部跑完后再补入正式数值。

## 5. GitHub 中保留的内容

保留：V4/V5 主代码、ML baseline、统一 ICL baseline、CoPB/PB&J、SILIC 初步实现、必要配置、方法文档和汇总级结果。

本次拓展另外加入 RG-FLARE-VAX、TRBM-FLARE-VAX、TRBM ablation，以及对应的公开汇总结果文件；原有 README 内容和旧方法结果均继续保留。

剔除：早期 HBM2 开发脚本、带本地运行状态的 notebook、checkpoint、逐样本 prediction、API 日志、support map、本地路径、失败日志以及原始 NHIS 数据。

安装依赖：

```bash
pip install -r requirements.txt
```
