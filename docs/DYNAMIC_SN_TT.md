# REACR-RCA 的真实动态图数据路径（SN/TT）

## 结论

`D2` 本身没有逐时刻调用边观测，不能通过复制或扰动固定边集变成“真实动态图”。本实现因此保留 D2 作为固定拓扑退化测试，同时从 Eadro 的原始 SN/TT Jaeger spans 重建逐时间桶调用图，使 REACR-RCA 可以在有可审计边变化的数据上训练和定位。

节点集合固定用于对齐时间序列，边集合、边权和边属性随时间变化：

```text
Jaeger CHILD_OF spans
        │ 去重 traceID、过滤会话时间、校正 +8h 时差
        ▼
每个时间桶内 parent service → child service
        │
        ├── E_t：该桶实际观测到的调用边（允许为空）
        ├── weight：原始调用次数
        └── edge_attr：调用率、延迟、P95、错误率等 5 维特征
```

这与 TGSphere 的核心动态图语义一致：从 trace 父子关系按窗口生成独立图快照；本实现额外保留空图，并让边强度和边属性真正进入模型。

## 数据语义

| 数据集 | 节点 | 时间桶 | 故障窗口 | 图节点 | 动态边来源 |
| --- | ---: | ---: | ---: | --- | --- |
| SN | 12 个服务 | 10 s | 12×10 s = 120 s | 固定服务词表 | 当前桶 Jaeger 调用 |
| TT | 27 个服务 | 60 s | 10×60 s = 600 s | 固定服务词表 | 当前桶 Jaeger 调用 |

每个节点有 45 维特征：29 维指标、10 维 trace、6 维日志。归一化参数只用正常训练会话拟合，避免故障测试数据泄漏。三个模态均包含 presence mask，因此缺失 trace 或日志不会与数值零混淆。

故障标签来自数据集的受控注入记录，标签粒度是服务级注入目标；它不是逐时刻真实因果传播图。`cases.csv` 额外保存 `session_id`，训练/验证/测试按完整会话分组，确保同一轮实验的多个故障案例不会跨集合泄漏。

## 模型如何消费动态图

动态图信息经过三条路径：

1. 每一时刻的 DGL 图独立进入空间消息传递，模型不会复用窗口首图，除非显式使用 `--graph_sequence_mode freeze_first` 做消融。
2. `weight` 用于加权邻居聚合，同时 `log(1+入边调用量)` 保留绝对调用强度；5 维 `edge_attr` 聚合后参与空间表示。
3. 因果关系头按每一目标时刻的真实边约束候选关系，并用窗口内相对边强度调整关系 logits。

空时间桶保持 `E_t=∅`，不会补全成全连接图，也不会用静态调用目录替代。静态 `call_edges.pkl` 只保存训练集观测并集，供兼容代码和审计使用，不参与逐时刻图构造。

## 生成数据

从仓库根目录运行：

```bash
python tools/convert_eadro_dynamic.py \
  --dataset both \
  --input-root '/mnt/d/Users/wyq/shiyan/Eadro/chunks' \
  --output-root data \
  --overwrite
```

输出目录：

```text
data/SN/ 或 data/TT/
├── cases.csv
├── metadata.json
├── normalization.npz
├── graphs_info/
│   ├── node_hash.pkl
│   ├── call_edges.pkl
│   └── host_edges.pkl
└── samples/
    ├── train_samples.pkl
    └── test_samples.pkl
```

`metadata.json` 记录每个会话的有效 trace、重复/stale trace、边切换、边权切换和空图数量，可用于复现实验审计。

## 运行 REACR-RCA

先做快速前向检查：

```bash
python reacr_rca_pipeline.py \
  --dataset SN --device cpu \
  --max_train_windows 8 --max_case_windows 8 \
  --output_dir res/sn_dynamic_smoke

python reacr_rca_pipeline.py \
  --dataset TT --device cpu \
  --max_train_windows 8 --max_case_windows 8 \
  --output_dir res/tt_dynamic_smoke
```

正式训练时去掉两个 `--max_*_windows` 限制，并设置训练预算与随机种子：

```bash
python reacr_rca_pipeline.py \
  --dataset SN --device cuda --seed 0 \
  --train_epochs 20 --normal_pretrain_epochs 3 \
  --output_dir res/sn_dynamic_seed0
```

SN/TT 配置默认启用 `group` 会话划分、无填充故障窗口、5 维边特征和 `topology_weight_gamma=0.25`。

## 本次全量转换的可核验结果

| 检查项 | SN | TT |
| --- | ---: | ---: |
| 正常/故障会话 | 3 / 4 | 2 / 9 |
| 正常/故障图快照 | 436 / 569 | 273 / 976 |
| 故障案例 | 36 | 81 |
| 原始会话内边集切换 | 344 | 281 |
| 原始会话内边权切换 | 987 | 289 |
| 真实空图时间步 | 22 | 72 |
| 可训练正常窗口 | 403 | 255 |
| 含边变化的正常窗口 | 170 | 21 |
| 完整故障窗口 | 36 | 81 |
| 含边变化的故障窗口 | 31 | 29 |

SN、TT 均完成了正常预训练和监督训练各 1 epoch 的端到端反向传播烟测。两者的动态拓扑投影梯度均非零；SN 实测边强度/边属性投影梯度范数为 `3.23×10^-2`（正常预训练均值）和 `1.73×10^-2`（最终诊断），证明图结构变化和边证据都位于实际训练计算图中。烟测指标只用于验证实现，不能作为正式性能结论。

## 必须报告的消融

- `observed` 对比 `freeze_first`：检验时间变化边集的贡献。
- `topology_feature_mode full` 对比 `static_only`：检验显式边增删特征。
- `topology_weight_gamma=0`：检验因果关系头中的边强度贡献。
- `edge_feature_dim=0`（配套去除 edge-attribute 分支）：检验调用率、延迟和错误率边属性。
- changed/stable 分层 A@K：避免 TT 中较多稳定窗口掩盖动态图子集表现。

不要把“动态图分支可运行”写成“动态图一定提升性能”；最终结论必须由相同划分、相同 seed 和相同预算下的多随机种子消融支持。
