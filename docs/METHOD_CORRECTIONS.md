# V3 方法与评测修正清单

本文件记录从 `2.1.0-reacr-rca` 到 `3.0.0-reacr-rca` 的语义与实现修正。它不是性能声明；所有受影响的实验都必须按 V3 配置重跑。

| 问题 | 修正 | 对实验/论文的影响 |
| --- | --- | --- |
| 根因标签使用无边界前缀可能误匹配相似服务名 | 仅允许节点名与标签完全相同，或以 `标签 + "-"` 开头；零匹配默认报错 | D1 的服务级标签可能映射多个实例。应报告多正候选 A@K，不能统一声称唯一实例级真值。 |
| D1 metric 特征由两个非连续区间组成，连续切片会混入 trace/log 或改变原顺序 | 配置显式 `modality_spans`，门控按索引收集并回写原位置，同时验证区间完整、互斥 | 使用模态门控的结果必须重跑；旧结果不能证明正确的多模态融合。 |
| 时间注意力可读取未来窗口位置 | 默认加入时间方向遮罩，使位置 `t` 只能读取 `<=t` | 避免同一窗口中的未来信息泄漏；旧模型结果与新模型不可直接比较。 |
| 预测分支比较隐藏状态而非下一步原始观测 | 从当前隐藏状态预测 `x[t+1]`，显式导出预测、目标、有效掩码和损失 | “预测误差”现在具有观测空间语义；所有异常分数需要重算。 |
| 正常滑动窗口只构造但未实质参与主训练 | 加入正常窗口重构/预测预训练，并在根因排序时循环加入辅助正常动态损失 | 必须加入 `no_normal_dynamics_training` 消融。 |
| D1、D2 被笼统描述为固定拓扑 | 新增只读取实际 DGL 快照的审计和 8 维节点拓扑特征 | D1 有 8 个边集/15 次切换；D2 固定。动态图贡献主要在 D1 验证。 |
| 把 `static_only` 误当成“固定拓扑”对照 | 明确 `static_only` 只关闭显式变化通道；新增 `graph_sequence_mode=freeze_first`，将同一窗口的空间图、拓扑特征、关系掩码和图先验统一冻结为首个真实图 | 真正的固定拓扑消融为 runner 变体 `frozen_window_topology`；不得用 `full-static_only` 代表全部动态图贡献。 |
| D1 变化窗口极少，总体指标可能掩盖动态窗口表现 | 审计 case/正常窗口，正常训练变化窗口默认按因子 4 加权采样，并导出 changed/stable 分层 A@K 与样本数 | D1 case 窗口仅 6/210 变化；正常滑窗 27/3705 变化且只来自 3 次底层切换。过采样不改变测试分布，需增加 `no_topology_change_oversampling` 消融。 |
| 管线拓扑解释字段曾无条件支持“固定拓扑”判断 | 根据训练/测试快照审计结果生成 `fixed_edge_topology_in_both_splits` 和 `observed_edge_changes_in_any_split` | D1 自动报告存在边变化，D2 自动报告两部分固定；结论不再由硬编码决定。 |
| 有边权时的空间聚合使用加权消息均值，分母不等于权重和 | 改为加权和除以权重和 | 当前 D1/D2 无边权，因此这是前向兼容修复，不应误写为现有数据使用了边权。 |
| 排序图先验只读取窗口末帧，早期拓扑与早期异常信号无法进入该分支 | 每个时刻分别用对应图和异常分数计算入/出邻居证据，再沿整段窗口聚合 | 排序器现在消费完整图序列；`freeze_first` 冻结图但保留逐时刻异常信号，固定拓扑消融只移除边演化。 |
| 单个全局时滞分布和固定目标查询难以表达 case/节点对差异 | 源与目标均由当前样本轨迹编码，产生 case-、target-、source-specific 的时滞权重 | 输出应称“时滞有向关系证据”，而非已识别真实因果矩阵。 |
| 趋势/季节分解允许偶数、非正或非整数移动平均核，可能使对称填充后的序列长度与输入不一致，或让池化层延迟报类型错误 | `MovingAverage` 构造时强制 `kernel_size` 为正奇整数，并增加偶数核与非整数核提前失败测试 | 配置错误会在建模前暴露；时滞轨迹保持与原窗口严格对齐。 |
| 关系 affinity 曾在窗口全部对齐时间上平均，4D 边并集只能限制时滞 logits，不能排除边尚未出现/已经消失的乘积 | 构造 `lag_alignment_topology_mask[B,L,W,target,source]`；只有边存在于目标快照 `t` 时才累计 `source(t-l)×target(t)`，并按实际有效时刻数归一化；同时保留 4D `lag_topology_mask` 作为时间并集 | 拓扑支持真正进入时间级 affinity 计算；5D 掩码、4D 时滞并集和3D窗口并集均导出审计，但仍不构成因果识别。 |
| 关系阈值后再次归一化会强制剩余来源解释全部响应 | 不再阈值后强制归一化；加入可学习空源概率；可用真实图支持掩码 | 模型可以表达“当前候选源不足以解释该响应”，但仍不是干预因果结论。 |
| 根因/症状分数由同一量的正负形式构造，且数据无症状真值 | 改为独立源角色与响应角色头；高响应负节点仅作模型挖掘伪候选 | 不得将响应角色输出报告为症状识别准确率，也不得称训练使用了症状标签。 |
| 角色辅助损失把高响应伪候选当作硬症状真值，并对其响应角色施加绝对 BCE | 移除 hard pseudo response 的绝对 BCE，只比较标注正候选与伪候选之间的源角色、响应角色和角色差值；标注正候选仍有源角色正向监督 | 伪候选只提供弱相对约束，不是症状真值；角色分支不能报告症状识别性能。 |
| 旧 coverage/residual 存在代数冗余，且“counterfactual”超出观测数据支持范围 | 使用候选源假设覆盖度与对称归一化绝对残差，分别惩罚欠解释和过解释 | 论文改称“候选源假设解释”，不称反事实干预或个体因果效应。 |
| 短窗口时滞前缀用首个异常值回填，制造了不存在的历史观测 | 改为零填充并显式使用有效位置掩码，缺失前驱不参与覆盖/残差计算 | 时滞解释不再把第一个观测复制到窗口之前。 |
| 事件链整条路径共用全局时滞，且零关系质量可能形成严格边 | 为每条源—目标路径边导出节点对特定 argmax/期望时滞与置信度；严格边要求强度大于阈值且大于零 | 事件链可报告预测时滞与异常起始时滞的一致率/MAE，但数据无真实链标签，不能表述为传播链准确率。 |
| 图先验加权聚合用节点数而非权重和作分母 | 改为加权和除以有效权重和 | 避免不同入边数/权重尺度造成系统性缩放。 |
| 服务级标签映射出的三个副本被逐个当成真根因，训练语义比 any-hit A@K 更强 | 对 D1 210 个 case 审计得到 136 个单匹配、74 个三副本候选袋；主模型和 DeepHunt 基线排序损失均以袋内最高分为正分数，主模型关系签名和辅助目标也选择袋内最受支持候选 | 两种方法的训练、run config/manifest 与“命中任一候选即成功”的 A@K 对齐；不得声称 74×3 个副本均有独立真值标注。 |
| DeepHunt 的样本内 min–max 排序归一化在常数/近并列分数时除以极小极差，可产生约 `10^11` 量级梯度；仅给极差加下界不能解决不稳定性 | 完全移除 min–max，改为原始有界分数上的候选袋 pairwise-margin：默认 `margin=1/rank_range`，袋内 max 与按分数降序的负候选比较，`1/rank` 权重按权重和归一 | 常数和近并列回归测试的 loss/梯度均有限，最大绝对梯度不超过 1；权重作为 buffer 并随当前设备/数据类型使用。不得再描述为“常数极差下界保护”。 |
| `root_label_semantics` 把通用 any-hit 口径写成仅适用于服务级标签，且未区分 D1/D2 映射粒度 | 通用值改为 `positive_candidate_bag_any_hit`，另记 `label_mapping_granularity`：D1=`mixed_service_to_instance_candidate_bag`，D2=`exact_single_instance_candidate` | 主模型与基线 manifest/run config 同时记录两层语义；D2 不再被误写为多副本映射。 |
| 事件链用 `causal_inferred` 命名仅由模型关系支持、观测图中无对应方向边的路径 | 主术语改为 `model_relation_only`；事件链 CSV 主字段使用 `source_role_score`、`response_role_score`、`source_hypothesis_coverage/residual` | `causal_inferred` 及 `root/symptom/counterfactual` 字段仅作为弃用兼容别名，不应出现在新论文主表述中。 |
| 角色诊断沿用 `role_auc`、`root_role_gap`，会把候选袋成员关系误写成角色真值或继续使用旧“根因角色”术语 | 主字段改为 `candidate_membership_auc`、`source_role_gap`、`response_role_gap`；runner 的逐运行、多 seed 和 Markdown 报告优先读取新字段 | 汇总 JSON/表中的 gap 字段可带 `_mean`；`role_auc`、`root_role_gap` 仅为弃用兼容别名。candidate-membership AUC 不是症状标签准确率。 |
| artifact、事件链、链验证和诊断继续以 root/true root/hard symptom 命名候选节点 | 主字段统一为 `is_label_candidate`；`candidate_node`/`label_candidate_nodes`；`pred_candidate`/`label_candidates`/`candidate_hit`；`label_candidate_node`/`response_candidate_node` | `is_root_label`、`root_node`、`pred_root`、`true_roots`、`hard_symptom_*` 等仅为弃用兼容别名。主诊断指标为 `response_candidate_error_rate` 与 `label_vs_response_candidate_margin_mean`，runner 标题为 “Source-Response Candidate Diagnostic”。 |
| 修正后的运行目录可能静默复用旧代码产物，且基线默认只跑单 seed | runner 记录版本与关键代码/配置的 SHA-256 implementation fingerprint；主模型和 DeepHunt 基线默认均为 seed 0–4 | 指纹不一致必须覆盖或使用新目录；正式结果必须重新运行五 seed，旧结果不得混入。 |
| changed/stable 只保存在单次评测 JSON，难以进行跨 seed 比较 | runner 将分层样本数与 A@1/A@3/A@5 收集进 `all_runs.csv`、多 seed 聚合和最终 Markdown 报告 | 正式结果应直接给出分层均值和样本数，不能只报告总体 A@K。 |
| 输出与文档术语超过证据边界 | 新增独立角色、空源、拓扑特征、观测预测等导出，并统一能力边界 | 兼容字段名可保留在代码/JSON 中，但论文图表、正文和新结果应使用修正术语。 |

## 数据集特定结论

### D1

- 7,200 个图快照、46 个节点、8 个唯一有向边集、15 次相邻拓扑切换；边数为 200、202、256、262、266 或 268。
- case 窗口中 6/210 包含变化；正式 runner 在 `time_strict`、`feedback_sample_ratio=0.4` 协议下的训练/验证/测试分别为 3/84、0/21、3/105（不是所有直接管线调用的通用划分）。正常训练滑窗中 27/3705 包含变化，但它们只源于训练序列的 3 次底层切换。
- 没有边属性或边权。
- 标签为服务级与实例级混合粒度：210 个 case 中 136 个映射一个节点，74 个映射三个副本。三副本是一个正候选袋，主模型与 DeepHunt 基线训练均采用袋内 max，A@K 采用“命中任一候选”语义。
- D1 是主模型、`no_topology_change_features`、`frozen_window_topology`、`no_explicit_topology_encoder` 和 `no_relation_topology_constraint` 等拓扑消融的主要数据集。

### D2

- 14,400 个图快照、18 个节点、1 个唯一有向边集、0 次切换；每个快照固定 66 条边。
- 没有边属性或边权。
- D2 只能验证固定拓扑兼容性和时变节点属性建模，不能作为拓扑演化收益证据。

## 术语对照

| 兼容代码/旧文档名称 | 新论文与报告名称 | 原因 |
| --- | --- | --- |
| `causal_matrix` | 时滞有向关系证据矩阵 | 观测学习未提供干预识别保证。 |
| `counterfactual_coverage/residual` | 候选源假设覆盖度/残差 | 没有执行或估计 `do` 干预。 |
| `root_role_score` | 源角色证据 | 标签仅监督根因候选，角色是模型证据。 |
| `symptom_role_score` | 响应角色证据 | 数据集没有症状节点标签。 |
| `role_auc` | 候选成员 AUC（`candidate_membership_auc`） | 衡量正候选袋成员与挖掘候选的分离，不是真实角色分类 AUC。 |
| `root_role_gap` | 源角色差值（`source_role_gap`） | 新诊断同时独立报告 `response_role_gap`。 |
| hard symptom candidate | 响应候选（response candidate） | 由模型挖掘，不是真值症状；`hard_symptom_*` 只为兼容。 |

这些旧字段为了读取已有脚本和产物仍可能存在，但不应被用来扩大科学结论。

## 必须重跑的实验

1. D1、D2 的五 seed 主实验，保持 `time_strict` 划分并重新生成所有预测和诊断；DeepHunt 基线也使用相同五个 seed。
2. D1 的 `no_topology_change_features`、`topology_dynamic_features_only`、`frozen_window_topology`、`no_explicit_topology_encoder` 与 `no_topology_change_oversampling`；固定拓扑消融必须包含 `frozen_window_topology`。
3. `no_relation_topology_constraint`，区分显式拓扑特征与时滞对齐关系支持掩码。
4. changed/stable 分层 A@K 和样本数；D2 只作为固定拓扑健全性检查。
5. `no_normal_dynamics_training`，检验正常窗口预训练和辅助损失。
6. 角色损失、候选源假设损失、事件链一致性和不变性损失的逐项消融。
7. D1 标签映射审计：统计每个 case 的匹配数量、零匹配数量和多正候选比例。
8. 使用修正后的模态区间重新运行所有启用模态门控的配置。

在以上重跑完成前，不应沿用 V2.1 的 A@K 数字，不应声称本轮修正提升了性能，也不应把单次 smoke test、非零梯度或变化窗口过采样当作论文性能结果。runner 的 implementation fingerprint 应随结果一起归档。

截至当前代码状态，61/61 项单元测试以及 D1/D2 真实数据小规模端到端运行已经通过，覆盖时间级 lag 拓扑筛选、正奇整数移动平均核（含非整数拒绝）、完整窗口图先验、主模型/基线候选袋排序、DeepHunt 并列梯度、外部 `base_dir`、拓扑汇总和兼容字段。D1 smoke 的 5D mask 为 `[8,3,10,46,46]`、动态梯度 `1.990379096e-4`；D2 为 `[4,3,10,18,18]`、动态梯度 0，真实 D2 E2E 进一步验证了新的候选字段。它们是正确性与可运行性检查；完整五 seed 主实验/基线及所需消融尚未完成。
