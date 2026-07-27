# REACR-RCA V3 Release Manifest

## 主入口

- `reacr_rca_pipeline.py`：单次训练、验证、测试、数据驱动拓扑解释、标签语义记录和产物导出。
- `tools/reacr_rca_final_study_runner.py`：主模型/基线多 seed 主实验、消融、changed/stable 分层 CSV/Markdown 汇总、运行清单及 implementation fingerprint 校验。
- `tools/reacr_rca_topology_audit.py`：只基于序列化 DGL 快照审计边集变化及边属性。
- `tools/convert_eadro_dynamic.py`：把 SN/TT Jaeger span 流转换为会话感知、按时间桶组织的真实动态图快照。

## 当前方法实现

- `models/reacr_rca_dsbg_dataset.py`：窗口构造、边界安全标签匹配与多正标签。
- `models/reacr_rca_modality_gate.py`：显式模态索引/区间及原位置回写。
- `models/reacr_rca_topology_dynamics.py`：静态度和边增删动态特征。
- `models/reacr_rca_dynamic_backbone.py`：空间编码、拓扑融合、GRU 和时间方向注意力。
- `models/reacr_rca_anomaly_head.py`：观测重构、下一步观测预测和异常演化。
- `models/reacr_rca_causal_head.py`：兼容文件名；用 5D `lag_alignment_topology_mask` 筛选时间级乘积，保留 4D `lag_topology_mask`/3D窗口并集，并输出时滞有向关系证据与空源；移动平均核必须为正奇整数，非整数在构造期拒绝。
- `models/reacr_rca_causal_ranker.py`：聚合完整图序列与逐时刻异常的图先验、独立源/响应角色及候选源假设解释排序。
- `models/reacr_rca_trainer.py`：正候选袋 max 排序、弱相对角色/解释性目标、正常动态预训练和辅助训练；`prepare_graph_sequence` 统一实现 `observed`/`freeze_first` 图序列。
- `models/reacr_rca_event_chain_explainer.py`：以 `model_relation_only` 及 candidate/label-candidate 字段为主术语，导出模型证据链及每条路径边的预测时滞/一致性；不代表真实传播链。
- `models/reacr_rca_artifact_exporter.py`：导出中间状态、5D/4D 时滞拓扑掩码、`is_label_candidate` 和主术语证据字段。
- `models/reacr_rca_statistical_calibrator.py`：不使用节点身份的时序统计校准器及神经/统计分数融合。

## 配置与数据接口

- `config/D1.yaml`：130 维特征；metric 为 `[0,52]` 与 `[73,129]` 两个闭区间，trace 为 `[53,61]`，log 为 `[62,72]`。
- `config/D2.yaml`：266 维特征；metric、trace、log 为连续且互斥的显式区间。
- `config/SN.yaml`、`config/TT.yaml`：真实动态图窗口、会话分组划分、边属性和零样本校准配置。
- `config/SN_TPL.yaml`：SN 日志模板增强实验配置。
- `utils/public_functions.py`：加载样本、节点映射和兼容初始化信息。

数据文件不随仓库发布。拓扑动态特征只读取每个样本内的 DGL 图，不从图目录补边。

## 图序列与拓扑实验接口

- `--graph_sequence_mode observed`：主模型读取窗口内每一步的实际序列化图。
- `--graph_sequence_mode freeze_first`：实验消融；同一窗口所有步骤使用窗口首个实际图，并一致作用于空间消息传递、8 维拓扑特征、关系拓扑掩码和排序图先验。
- `--topology_feature_mode static_only`：仅移除六个显式变化通道，**不冻结**上述图结构消费者。
- `--normal_topology_change_oversample_factor 4`：正式 runner 的默认正常窗口加权采样；只改变训练抽样概率，测试不采样。

runner 中对应变体为 `no_topology_change_features`、`topology_dynamic_features_only`、`frozen_window_topology`、`no_explicit_topology_encoder`、`no_relation_topology_constraint` 和 `no_topology_change_oversampling`。

审计统计：D1 有 8 个边集、15 次快照切换；case 窗口 6/210 变化，正常训练滑窗 27/3705 变化且只源于 3 次底层切换。D2 只有一个固定边集，所有窗口变化数为零。`topology_audit.json` 的固定/变化解释由这些审计值计算。评测通过 `topology_strata.changed/stable` 导出各层样本数与 A@K，并进入多 seed 聚合与最终 Markdown 报告。

标签审计：D1 的 210 个 case 中 136 个映射一个节点、74 个映射三个副本。主模型 `PairwiseRankingLoss` 和 DeepHunt `models/layers.py::RankingLoss` 均使用正候选袋 max，与 any-hit A@K 一致。运行配置/manifest 记录通用 `root_label_semantics=positive_candidate_bag_any_hit`、`positive_bag_aggregation=max`，并记录 D1 `label_mapping_granularity=mixed_service_to_instance_candidate_bag`、D2 `exact_single_instance_candidate`。

DeepHunt `RankingLoss` 不再使用 min–max 或极差下界。它直接在有界 scorer 输出上，以 `margin=1/rank_range`（可覆盖）比较袋内最高分与负候选；负候选 `1/rank` 权重除以权重和，权重 buffer 随分数设备和类型使用。并列/近并列测试的梯度均有限且最大绝对值不超过 1。

## 测试边界

- `tests/test_reacr_rca_label_and_modality.py`：标签碰撞、多正映射、零匹配与非连续模态布局。
- `tests/test_causal_temporal_anomaly.py`：未来不可见和观测空间下一步预测。
- `tests/test_reacr_rca_topology_dynamics.py`：固定/变化拓扑特征与编码。
- `tests/test_reacr_rca_topology_audit.py`：边集、切换、边属性和来源审计。
- `tests/test_graph_sequence_ablation.py`：`freeze_first` 去除窗口内变化且不制造边，并一致作用于模型图消费者。
- `tests/test_relation_and_ranker_corrections.py`：正奇整数移动平均核（含非整数拒绝）、5D 时间级/4D 时滞级拓扑掩码、完整窗口图先验、主模型/DeepHunt 正候选袋损失、并列梯度、空源、相对角色约束、有效时滞前缀与候选源解释。
- `tests/test_event_chain_topology_consistency.py`：严格边、逐路径边时滞与证据链导出语义。
- `tests/test_final_study_runner.py`：正式拓扑变体、D1 模态配置、基线安全标签匹配和 implementation fingerprint。
- `tests/test_eadro_dynamic_graph.py`：SN/TT 流式 span、空图、边证据、故障窗口和会话无泄漏划分。
- `tests/test_statistical_calibrator.py`：统计特征对齐、分类器保存/加载和融合边界。

当前单元测试为 61/61 通过，覆盖外部 `base_dir`、拓扑分层 summary/Markdown、偶数/非整数核拒绝和 DeepHunt 并列/近并列梯度。D1/D2 真实数据小规模端到端烟测的训练、评估、解释与导出均以退出码 0 完成；D2 真实 E2E 还验证了新增候选字段。这些检查不替代正式多 seed 实验。

## 运行身份与结果状态

正式 runner 的主模型与 DeepHunt 基线默认 seed 均为 `0 1 2 3 4`，消融默认 seed 为 `0 1 2`；全量五 seed 消融需要显式传入 `--ablation_seeds 0 1 2 3 4`。运行清单记录 `VERSION` 以及管线、runner、REACR-RCA 模块、基线关键代码和 D1/D2 配置的 SHA-256 implementation fingerprint；已有目录的指纹不一致时不得静默复用结果。

当前 V3 代码版本尚未完成可用于论文的五 seed 主实验和全部消融。D1 小规模烟测得到 5D mask `[8,3,10,46,46]`、动态投影梯度范数 `1.990379096e-4` 并成功写入 NPZ；D2 得到 `[4,3,10,18,18]` 且动态梯度为 0，符合固定拓扑预期。此前 D1 单窗口梯度 `0.0004352666` 的检查仍有效。这些只证明实现路径，不构成性能声明。

## 兼容字段

以下名称为旧脚本/产物兼容而保留，不是当前科学表述：

- `causal_matrix`：应解释为时滞有向关系证据矩阵；
- `counterfactual_coverage` / `counterfactual_residual`：对应候选源假设覆盖度/残差的兼容别名；
- `root_role_score` / `symptom_role_score`：对应独立源/响应角色证据的兼容别名；
- `role_auc`：对应 `candidate_membership_auc` 的弃用诊断别名；
- `root_role_gap`：对应 `source_role_gap` 的弃用诊断别名；新报告同时使用独立的 `response_role_gap`；
- `hard_symptom_error_rate` / `root_vs_hard_symptom_margin_mean`：对应 `response_candidate_error_rate` / `label_vs_response_candidate_margin_mean` 的弃用诊断别名；
- `is_root_label`、`root_node`、`pred_root`、`true_roots`、`hard_symptom_*`：分别对应 artifact/event/validation/diagnostic 中的 label-candidate、candidate 或 response-candidate 主字段；
- `causal_inferred`：对应 `model_relation_only` 的弃用事件链别名；
- `counterfactual_loss_weight`、`root_symptom_loss_weight`：兼容 CLI 名称。

## 非主体兼容代码

`models/dycau_trainer.py`、`evaluation.py`、`layers.py`、`noise_engineering.py`、`rc_scorer.py` 和 `train.py` 为 DeepHunt/baseline 兼容模块，不是 V3 主排序器的核心贡献。其中标签匹配、正候选袋有界 pairwise-margin、rank 权重归一和设备一致性属于公平评测所需的基线正确性修复。
