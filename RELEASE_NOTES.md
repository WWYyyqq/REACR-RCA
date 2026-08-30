# REACR-RCA V3.1.0 Release Notes

状态：`3.1.0-reacr-rca`，V3.0 的向后兼容增量版本。V3.0 及更早分支和标签保持不变。

## V3.1 增量内容

- 新增 D1 真实加权动态图转换器与 `D1_weighted` 配置，从原始 AIOps Jaeger Span 恢复分钟级实例调用，并生成调用量、频率、平均/P95 延迟、错误率及边类型属性。
- 原始无权 D1 不被覆盖；D2 继续保留为历史固定拓扑数据，但不进入当前论文主实验。
- 新增 `source_hypothesis_no_role`、零时滞关系和核心损失移除等严格消融接口。
- 新增 RQ1–RQ6 的基线、消融、故障类型、参数敏感性、动态图强度和受控传播实验脚本及结果审计工具。
- 实现指纹纳入 D1 转换器，拓扑审计进一步区分边集合、边权和边属性变化。
- 发布前回归测试扩充为 79 项；数据、模型权重及大规模结果产物不随仓库发布。

## V3.0 基础能力

### V3.0 新增能力

- 新增 SN/TT 原始遥测转换器：从 Jaeger `CHILD_OF` span 按时间桶构造真实调用图，保留空图、动态边、调用次数和 5 维边属性，并按会话分组划分数据。
- 新增边质量与边属性上下文编码，使 SN/TT 的观测调用强度和边属性进入空间聚合与关系证据。
- 新增不使用节点 ID 的服务无关统计校准器（ExtraTrees/线性 SVC）及样本内分数融合，用于跨服务零样本评估。
- 新增 SN/TT 配置、动态转换审计、统计校准测试和端到端动态图回归测试；V3 发布前完整单元测试为 61/61 通过。


### V3.0 修正

- 使用边界安全的根因标签匹配，并将服务级标签表示为正候选袋；零匹配默认报错。D1 的 210 个 case 中 136 个单匹配、74 个映射三个副本，主模型与 DeepHunt 基线均用袋内 max 训练并按 any-hit A@K 评测。通用 `root_label_semantics` 为 `positive_candidate_bag_any_hit`，另分别记录 D1 `mixed_service_to_instance_candidate_bag` 和 D2 `exact_single_instance_candidate`。
- 用显式特征区间修正 D1 非连续 metric 布局，并保持原始特征顺序。
- 默认启用只访问当前/过去步骤的时间注意力；下一步预测改为观测空间预测。
- 将正常窗口用于重构/预测预训练和监督训练期间的辅助动态约束。
- 新增对实际 DGL 快照的拓扑审计、8 维节点级拓扑动态特征、changed/stable 分层评测和正常变化窗口加权采样。
- 区分“关闭显式变化通道”与“冻结图序列”：`static_only` 仍读取每步观测图，真正固定窗口拓扑的 runner 消融为 `frozen_window_topology`（`graph_sequence_mode=freeze_first`）。
- 将关系模块重构为 case-和节点对特定的时滞有向关系证据；5D `lag_alignment_topology_mask[B,L,W,target,source]` 直接筛选每个 `(t-l,t)` 异常乘积，4D `lag_topology_mask` 保留为时间并集，并支持空源概率。
- 移动平均核强制为正奇整数；偶数、非正和非整数配置在构造关系头时提前报错，保证趋势/季节序列与输入窗口等长。
- 排序图先验改为逐时刻使用对应图与异常信号，再聚合完整窗口，不再只读取末帧。
- 使用彼此独立的源角色与响应角色分支；高响应负节点仅作为伪候选。
- 移除 hard pseudo response 的绝对 BCE，仅保留与标注正候选之间的弱相对角色约束。
- 将原“反事实”校正重述并实现为候选源假设覆盖度与对称残差。
- 修正缺失时滞前缀、角色弱监督、加权图先验和零质量事件链边等实现细节。
- 扩充中间产物，导出拓扑特征、观测预测、空源概率、独立角色、5D/4D 时滞拓扑掩码、逐路径边预测时滞和候选源假设证据。
- 事件链主术语改为 `model_relation_only`，CSV 主字段改用 `source_role_score`、`response_role_score` 与 `source_hypothesis_*`；`causal_inferred` 及旧角色/反事实字段仅为弃用兼容别名。
- 角色诊断主字段改为 `candidate_membership_auc`、`source_role_gap`、`response_role_gap`，runner 报告优先使用新字段；`role_auc`、`root_role_gap` 仅为弃用兼容别名。
- artifact、事件链、链验证和角色诊断主字段统一为候选语义：`is_label_candidate`，`candidate_node`/`label_candidate_nodes`，`pred_candidate`/`label_candidates`/`candidate_hit`，以及 `label_candidate_node`/`response_candidate_node`。旧 root/true-roots/hard-symptom 字段仅为弃用兼容别名。
- 诊断主指标改为 `response_candidate_error_rate` 和 `label_vs_response_candidate_margin_mean`；runner 报告标题改为 “Source-Response Candidate Diagnostic”。旧 hard-symptom 指标仅为弃用别名。
- changed/stable 样本数与 A@K 已进入逐运行 CSV、多 seed 汇总和最终 Markdown 报告；管线的固定/变化拓扑解释字段由实际审计值生成。
- DeepHunt 排序损失同步采用正候选袋 max，并完全移除不稳定的样本内 min–max 归一化。新损失直接在有界分数上计算 pairwise margin（默认 `1/rank_range`），负候选按排名权重归一；并列/近并列梯度有限且最大绝对值不超过 1。baseline manifest/run config 记录与主模型相同的通用标签语义及数据集粒度。
- 正式 runner 对主模型和 DeepHunt 基线默认使用 seed 0–4，并记录关键实现与配置的 SHA-256 fingerprint，防止静默复用旧代码产物。

### V3.0 已审计的拓扑事实

| 数据集 | 快照 | 唯一边集 | 切换 | 边数 | 边权 |
| --- | ---: | ---: | ---: | --- | --- |
| D1 | 7,200 | 8 | 15 | 200–268 | 无 |
| D2 | 14,400 | 1 | 0 | 66 | 无 |

D1 可用于检验少量观测边集变化；D2 是固定拓扑退化情形，不能验证拓扑演化建模。审计没有证明 D1 的变化一定由部署操作导致。

模型窗口中的变化更稀疏：D1 case 窗口仅 `6/210` 含变化，正常训练滑窗为 `27/3705`；后者的 27 个重叠窗口只源于训练序列的 3 次底层切换。默认过采样因子 4 只提高正常训练时抽到变化窗口的概率，不修改图、不改变测试分布，也不增加独立变化事件。D2 的所有窗口均无拓扑变化。

在真实 D1 动态窗口 `229:239` 上，边数轨迹包含 `268→256→268`，提取到 56 个非零动态特征，反向传播后动态投影梯度范数为 `0.0004352666`。该检查证明代码实际消费动态通道，不是性能收益证据。

### V3.0 术语与兼容性

新文档和论文应使用：

- “时滞有向关系证据”，不称“真实因果矩阵”；
- “候选源假设解释”，不称“反事实干预”；
- “独立源/响应角色”，不称使用了已标注症状；
- “混合粒度、多正候选根因定位”，不把所有 D1 case 称为唯一实例级真值。

`causal_*`、`counterfactual_*`、`root_symptom_*` 等代码字段为向后兼容而暂时保留。

事件链中“只得到模型关系支持、观测图中无对应方向边”的主标签为 `model_relation_only`；`causal_inferred` 是弃用别名，不能据此声称模型推断出了真实因果边。

### V3.0 结果失效范围

标签、模态布局、时序预测、训练目标、关系头、排序器和拓扑输入均有语义变化。V2.1 及更早版本的指标只代表对应历史版本，不能迁移为 V3 结果。发布论文性能结论前必须至少重跑：

- D1/D2 多 seed 主实验；
- D1 的 `no_topology_change_features`、`topology_dynamic_features_only`、`frozen_window_topology`、`no_explicit_topology_encoder` 与 `no_topology_change_oversampling`；
- `no_relation_topology_constraint`；
- `no_normal_dynamics_training`；
- 解释性损失逐项消融；
- changed/stable 分层 A@K 及对应样本数；
- D1 标签匹配数量审计和使用修正模态区间的实验。

这里的“多 seed”在正式 runner 中指主模型和基线默认均为五个 seed（0–4）；消融默认是三个 seed（0–2），若论文要求全量五 seed 结果，运行时必须显式设置 `--ablation_seeds 0 1 2 3 4`。本次 V3 代码发布不附带新的正式性能数字或动态图收益结论；这些结论必须等待完整重跑。

当前 61/61 项单元测试与 D1/D2 真实数据小规模端到端运行均已通过；回归测试覆盖 DeepHunt 并列/近并列梯度、偶数及非整数移动平均核、外部 `base_dir` 和拓扑分层汇总，D2 真实 E2E 验证了新增候选字段。D1 smoke 导出的 5D mask 为 `[8,3,10,46,46]`、动态梯度 `1.990379096e-4`；D2 为 `[4,3,10,18,18]`、动态梯度 0。训练、评估、解释和导出均正常退出。这是正确性和可运行性检查；完整五 seed 主实验/基线和所需消融仍需重跑，不能把 smoke/E2E 结果写成论文性能。

### V3.0 能力边界

数据没有逐时刻真实因果图、干预结果、症状标签或真实传播链。时滞对齐掩码、逐边预测时滞、关系、角色和事件链输出均为可检查的模型证据；它们不能单独证明真实因果关系、真实传播时滞、反事实效应或症状识别正确性。
