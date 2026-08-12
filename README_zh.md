# FLARE-VAX：基于 NHIS 2024 的行为理论引导流感疫苗接种预测

[English README](README.md)

本项目使用 **2024 NHIS Sample Adult** 数据预测受访者过去 12 个月是否接种流感疫苗（`SHTFLU12M_A`）。开发时使用的原始 `adult24.csv` 约包含 **32,629 名受访者、630 个变量**。V4 最终纳入 32,132 人，V5 纳入 32,130 人，默认按 **40% memory / 20% calibration / 40% test** 划分。仓库不公开原始 NHIS 数据、逐样本预测和 API 日志。

> 本项目中的 HBM 分数是由 NHIS 可观察变量构造的理论代理，并不是对个体私有心理信念的直接测量。

## 1. V4 与 V5

- **V4（包含其他疫苗史）**：允许 COVID、肺炎、带状疱疹和甲肝等非目标疫苗史进入 vaccine acceptance/benefit proxy，共 75 个 ML 特征。
- **V5（排除其他疫苗史）**：从分数、prompt、retrieval 和 memory 中排除全部非目标疫苗史，用 wellness、健康信息查询、医生沟通和结果查看等非疫苗预防行为替代，共 67 个 ML 特征。

两者都构造 threat、acceptance/engagement、barriers、healthcare cues、navigation self-efficacy 五个 proxy，再合并成 Motivation、Capability、Activation，并形成 8 个行为 pattern。Pattern prior 只从 memory split 估计，reflection 只基于训练侧错误建立并在 calibration/test 前冻结。

## 2. 本次新增的两个 V4/V5 改版

这两个方法都**不是新的 feature version**，而是在原 V4/V5 的变量边界和 HBM proxy 上继续扩展。V4 仍允许其他疫苗史，V5 仍严格排除其他疫苗史；改变的是 HBM prior 如何变成最终概率，以及 reflective memory 如何建立和使用。

### 2.1 RG-FLARE-VAX：Reward-Guided HBM Integration + Reward-Valued Memory

代码：`scripts/81_rg_flare_vax_reward_memory_asu.py`

RG 版本保留原来的 HBM8 pattern prior，但在它上面增加一个小型数值 reward layer，用五个 HBM construct 学习如何调整 pattern anchor。Memory split 使用 5-fold out-of-fold 预测避免同一样本既参与拟合又用于自己的 reward 评估。LLM 可以提出很稀疏的 pattern-specific reward-weight 修改，但只有在独立 OOF validation 上真正降低 log loss 才会被接受。

第二个变化是 memory 不再只按“是否像一个好规则”来检索。每条 reflection rule 会得到一个经验性的 directional Q-value，用来衡量这个 correction direction 在相似历史样本中是否真的降低过 loss。测试时 retrieval 同时考虑 similarity、HBM8 pattern match、memory quality 和 Q-value。最终 LLM 以 reward-calibrated HBM prior 为锚，只允许做较小的 residual correction（默认 ±15 个百分点）。

所以 RG 相对原 V4/V5 的核心变化可以概括为：**重新学习 HBM construct 的数值整合方式，并用历史 reward 给 memory 定价**。LLM 本身不微调；由于 NHIS 是横截面数据，这里是 SILIC-inspired reward learning，而不是 sequential IRL。

### 2.2 TRBM-FLARE-VAX：Theory-Residual Behavioral Memory

代码：`scripts/82_trbm_flare_vax_asu.py`  
离线 ablation：`scripts/83_trbm_ablation_asu.py`  
方法说明：`docs/trbm_method_notes.md`

TRBM 进一步改变了 LLM 的角色：基础概率不再由 LLM 估计。它先只使用五个正向化的 HBM construct 拟合一个带非负系数约束的小型 logistic model，得到 `P_HBM`。这个 prior 不直接使用完整的 NHIS raw features，因此它代表的是 theory-derived prior，而不是 full-feature ML prediction。

之后 memory 专门学习 **HBM theory residual**。Memory respondent 使用 out-of-fold 的 `actual - P_HBM_OOF` 找出理论 prior 明显低估或高估的样本，再让 LLM 解释这些偏差背后的可复用机制。LLM 不允许输出 probability 或数值 delta。到了 calibration/test，LLM 只判断某个历史机制是否适用以及方向是 increase / decrease / none；真正的 correction magnitude 来自被选中历史 memory 的 signed residual，最后再由 calibration split 选择一个全局 `alpha`。

因此 TRBM 相对原 V4/V5 的重点是：**把概率估计和校准从 LLM 中拿出来，只让 LLM 做 theory failure 的语义机制识别和 gating**。


## 3. Zero-shot / Few-shot baseline

代码：`scripts/60_llm_icl_benchmark_asu.py`

- Zero-shot：不提供示例。
- Random balanced 8-shot：固定 4 个 YES + 4 个 NO。
- Similarity-selected 8-shot：每个目标样本分别检索最相似的 4 个 YES + 4 个 NO。
- Representative 8-shot：每类用 KMeans 找 4 个代表中心，再选择最近的真实样本。
- Random 8-shot + generic CoT：与随机 8-shot 使用相同示例，但要求一般性的正反证据推理，不使用 HBM 理论。

这些 baseline 都不接收 HBM score、pattern、pattern prior、reflective memory 或 FLARE correction rule。

## 4. 其他论文方法的迁移

### 不需要微调大模型

**HBM-CoPB** — 原文：[Chain-of-Planned-Behaviour Workflow Elicits Few-Shot Mobility Generation in LLMs](https://arxiv.org/abs/2402.09836)。原文用 TPB 的 attitude、subjective norms、perceived behavioral control 组织移动意图推理。本项目将其替换成五阶段 HBM 疫苗推理，使用 8 个固定平衡示例，但不提供 FLARE 的确定性分数、pattern prior、retrieval 或 reflection。

**HBM-PB&J** — 原文：[Improving Language Model Personas via Rationalization with Psychological Scaffolds](https://aclanthology.org/2025.findings-emnlp.1187/)。原文先用心理学 scaffold 为既有判断生成 rationale，再用强化 persona 预测新偏好。本项目 V4 使用两次调用：第一次在不知道流感疫苗 label 的情况下构造 HBM health persona；第二次结合 persona 与 8 个 memory 示例进行预测。目前只实现 V4。

**SILIC-inspired** — 原文：[Where You Go is Who You Are](https://arxiv.org/abs/2505.17249)。原文结合 behavioral theory、LLM 引导和 IRL，从移动轨迹反推潜在 reward。本项目的初步 V4 版本把四类非目标疫苗决策视作 contextual binary choices，拟合五维 preventive reward vector，再由 LLM 完成最终预测。由于 NHIS 没有时间顺序状态转移，因此明确称为 inverse contextual choice，而不是 sequential IRL。LLM 不微调，只优化外部 latent parameters。**状态：代码已加入，完整结果待跑。**

### 需要监督微调

**Persona-aware and Explainable Bikeability Assessment** — 原文：[arXiv:2601.03534](https://arxiv.org/abs/2601.03534)。原文使用 cyclist persona conditioning、多粒度 supervised fine-tuning 和 AI data augmentation 完成可解释的 bikeability 评分。**状态：仅完成方法调研，尚未迁移到 FLARE-VAX。**

## 5. 结果

### 5.1 ML baseline

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

### 5.2 Zero-shot / Few-shot

| Version | Model | Method | Test N | Threshold | Accuracy | Balanced Acc. | ROC-AUC | F1 | Status |
|---|---|---|---|---|---|---|---|---|---|
| V4 | Llama 3 70B | Random balanced 8-shot | 12853 | 5 | 0.4737 | 0.5000 | 0.5000 | 0.6428 | complete |
| V4 | Llama 3 70B | Random 8-shot + generic CoT | 12853 | 29 | 0.5186 | 0.5399 | 0.5399 | 0.6504 | complete |
| V4 | Llama 3 70B | Representative 8-shot | — | — | — | — | — | — | pending_rerun |
| V4 | Llama 3 70B | Similarity-selected 8-shot | — | — | — | — | — | — | pending_rerun |
| V4 | Llama 3 70B | Zero-shot direct | 12853 | 46 | 0.4787 | 0.5048 | 0.5048 | 0.6449 | complete |
| V4 | Llama 4 Scout 17B | Random balanced 8-shot | 12853 | 86 | 0.6432 | 0.6297 | 0.6851 | 0.4974 | complete |
| V4 | Llama 4 Scout 17B | Random 8-shot + generic CoT | 12853 | 73 | 0.5830 | 0.5622 | 0.6048 | 0.2761 | complete |
| V4 | Llama 4 Scout 17B | Representative 8-shot | 12853 | 75 | 0.6053 | 0.6211 | 0.6554 | 0.6887 | complete |
| V4 | Llama 4 Scout 17B | Similarity-selected 8-shot | 12853 | 79 | 0.6189 | 0.6225 | 0.6543 | 0.6317 | complete |
| V4 | Llama 4 Scout 17B | Zero-shot direct | 12853 | 81 | 0.6339 | 0.6453 | 0.7136 | 0.6904 | complete |
| V5 | Llama 3 70B | Random balanced 8-shot | — | — | — | — | — | — | pending_rerun |
| V5 | Llama 3 70B | Random 8-shot + generic CoT | — | — | — | — | — | — | pending_rerun |
| V5 | Llama 3 70B | Representative 8-shot | — | — | — | — | — | — | pending_rerun |
| V5 | Llama 3 70B | Similarity-selected 8-shot | — | — | — | — | — | — | pending_rerun |
| V5 | Llama 3 70B | Zero-shot direct | — | — | — | — | — | — | pending_rerun |
| V5 | Llama 4 Scout 17B | Random balanced 8-shot | 12852 | 82 | 0.5917 | 0.5932 | 0.6229 | 0.5903 | complete |
| V5 | Llama 4 Scout 17B | Random 8-shot + generic CoT | 12852 | 73 | 0.5461 | 0.5223 | 0.5389 | 0.1248 | complete |
| V5 | Llama 4 Scout 17B | Representative 8-shot | 12852 | 75 | 0.6218 | 0.6232 | 0.6251 | 0.6197 | complete |
| V5 | Llama 4 Scout 17B | Similarity-selected 8-shot | 12852 | 79 | 0.5773 | 0.5783 | 0.6051 | 0.5725 | complete |
| V5 | Llama 4 Scout 17B | Zero-shot direct | 12852 | 81 | 0.6307 | 0.6367 | 0.6622 | 0.6585 | complete |

V5 的全部 Llama 3 70B，以及 V4 70B 的 similarity-selected 和 representative 两项，当前公开表中保留为空并标记为 pending rerun。初步运行出现接近常数的输出，因此不作为最终结果汇报。

### 5.3 CoPB / PB&J 迁移结果

| Version | Model | Method | Test N | Threshold | Accuracy | Balanced Acc. | ROC-AUC | F1 | Status |
|---|---|---|---|---|---|---|---|---|---|
| V4 | Llama 3 70B | HBM-CoPB | 12853 | 51 | 0.6419 | 0.6464 | 0.6698 | 0.6593 | complete |
| V4 | Llama 4 Scout 17B | HBM-CoPB | 12853 | 51 | 0.6726 | 0.6753 | 0.7131 | 0.6776 | complete |
| V5 | Llama 3 70B | HBM-CoPB | 12852 | 51 | 0.5664 | 0.5634 | 0.5719 | 0.5258 | complete |
| V5 | Llama 4 Scout 17B | HBM-CoPB | 12852 | 61 | 0.6376 | 0.6378 | 0.6652 | 0.6268 | complete |
| V4 | Llama 3 70B | HBM-PB&J | 12853 | 76 | 0.5831 | 0.5935 | 0.6243 | 0.6425 | complete |
| V4 | Llama 4 Scout 17B | HBM-PB&J | — | — | — | — | — | — | pending |

PB&J V4 17B 暂时保留为空。SILIC 和需要微调的 persona-aware 方法尚无可汇报结果。

### 5.4 FLARE-VAX 主方法与 ablation

| Version | Model | Method | Test N | Threshold | Accuracy | Balanced Acc. | ROC-AUC | F1 | Status |
|---|---|---|---|---|---|---|---|---|---|
| V4 | Llama 4 Scout 17B | FLARE-VAX full run | 12853 | 51 | 0.7318 | 0.7312 | 0.7789 | 0.7174 | complete |
| V4 | Llama 3 70B | FLARE-VAX full run | 12853 | 46 | 0.7278 | 0.7285 | 0.7546 | 0.7207 | complete |
| V4 | No LLM | HBM8 pattern-only ablation | 12853 | 50 | 0.7277 | 0.7284 | 0.7693 | 0.7206 | complete |
| V5 | Llama 4 Scout 17B | FLARE-VAX full run | 11057 | 47 | 0.6101 | 0.6179 | 0.6334 | 0.6265 | complete_final_summary_11057_evaluated |
| V5 | Llama 3 70B | FLARE-VAX full run | 12852 | 47 | 0.6255 | 0.6325 | 0.6763 | 0.6597 | complete |
| V5 | No LLM | HBM8 pattern-only ablation | 12852 | 37 | 0.6255 | 0.6325 | 0.6797 | 0.6597 | complete |


### 5.5 RG-FLARE-VAX full-run 结果

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

RG 的主要提升首先来自 reward-calibrated prior：V4 的 ROC-AUC 从 HBM8 pattern anchor 的约 0.769 提升到约 0.822，V5 从约 0.680 提升到约 0.720–0.722。Reward-valued memory 会改变最终 operating point，并在部分设置下改善 F1 / balanced accuracy，但并不是所有概率指标都稳定优于 reward prior 或 no-memory，因此这里把各阶段分开汇报，不把全部提升归因于 memory。

### 5.6 TRBM-FLARE-VAX full-run 结果

| Version | Model | Test N | Threshold | Correction scale α | Accuracy | Balanced Acc. | ROC-AUC | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| V4 | Llama 3 70B | 12853 | 0.48 | 0.00 | 0.7390 | 0.7371 | 0.8205 | 0.7181 |
| V4 | Llama 4 Scout 17B | 12853 | 0.48 | 0.00 | 0.7390 | 0.7371 | 0.8205 | 0.7181 |
| V5 | Llama 3 70B | 12853 | 0.44 | 0.00 | 0.6531 | 0.6574 | 0.7200 | 0.6690 |
| V5 | Llama 4 Scout 17B | 12853 | 0.44 | 0.00 | 0.6531 | 0.6574 | 0.7200 | 0.6690 |

这批 full run 中，calibration 对所有配置都选择了 `alpha = 0.00`。因此最终 `trbm_full` 数值实际上等于 theory-constrained HBM prior；虽然 residual memories 和 LLM mechanism gate 已经建立并运行，但 calibration 没有发现继续叠加 residual correction 可以改善 log loss。V4 的 unweighted ROC-AUC 为 0.8205，V5 为 0.7200。TRBM 原始结果文件报告的 V5 test N 是 12,853，本次 update 原样保留这个数字，不人为改成旧 V5 pipeline 的 12,852。

## 6. GitHub 中保留的内容

保留：V4/V5 主代码、ML baseline、统一 ICL baseline、CoPB/PB&J、SILIC 初步实现、RG-FLARE-VAX、TRBM-FLARE-VAX、必要配置、方法文档和汇总级结果。

剔除：早期 HBM2 开发脚本、带本地运行状态的 notebook、checkpoint、逐样本 prediction、API 日志、support map、本地路径、失败日志以及原始 NHIS 数据。

安装依赖：

```bash
pip install -r requirements.txt
```
