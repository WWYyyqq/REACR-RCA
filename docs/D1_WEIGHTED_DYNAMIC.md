# D1 原始 Trace 加权动态图

## 数据来源与边界

转换器 `tools/convert_d1_weighted_dynamic.py` 使用原始 AIOps-Dataset 五个采集日的
`trace/all/trace_jaeger-span.csv`。它不会修改旧 D1，而是读取旧 D1 的 46 节点词表、
130 维节点特征、训练/测试划分和 Pod--Node 部署关系，输出独立的
`data/D1_weighted`。

原始 CSV 的关键字段为 `timestamp`、`cmdb_id`、`span_id`、`trace_id`、
`duration`、`status_code` 和 `parent_span`。调用方向由同一 Trace 内
`(trace_id, parent_span) -> (trace_id, span_id)` 确定；同一实例内部 Span 被过滤，
跨实例父子 Span 形成 `caller -> callee` 调用边。

## 时间桶和边属性

时间戳为毫秒 Unix 时间，与旧 D1 的秒级时间戳直接对齐，不施加时区偏移。每个
60 秒桶为调用边保存 8 维 `edge_attr`：

1. `log1p_call_count`；
2. `log1p_call_rate`；
3. `log1p_mean_latency_ms`；
4. `log1p_p95_latency_ms`；
5. `error_rate`；
6. `is_call`；
7. `is_deployment`；
8. `is_self`。

`duration` 按微秒除以 1000 转为毫秒。状态码同时兼容 HTTP 和 gRPC：HTTP
1xx--3xx、`0`、`OK/Ok` 视为成功；非零 gRPC 码及 HTTP 4xx/5xx 视为错误。
调用边的标量 `weight` 为桶内调用次数；部署边和自环的权重为 1，其五个调用统计
字段为 0，并由边类型标记区分。

旧 D1 的 Pod--Pod 调用目录边不会被保留。每个桶只使用该分钟真实观察到的调用边；
Pod--Node 双向部署边和自环则从同一时间戳的旧 DGL 快照保留。因此，67 个 Node
级故障案例仍然具有可达的基础设施节点。

## 全量转换审计

| 项目 | 结果 |
| --- | ---: |
| 原始 Span | 44,858,388 |
| 图快照 | 7,200 |
| 节点 | 46 |
| 有调用的时间桶 | 7,200 |
| 唯一观察调用边 | 142 |
| 唯一完整边集 | 5,192 |
| 相邻边集切换 | 6,696 / 7,199 |
| 调用权重签名切换 | 7,199 / 7,199 |
| 每快照调用边 | 2--142 |
| 每快照全部边 | 128--268 |
| 未映射实例 | 0 |
| 非单调时间行 | 0 |
| 桶完成后的迟到调用 | 0 |

五个文件的父子关系解析率依次为 99.9997%、99.9974%、99.9945%、99.9999% 和
99.9986%。少数超出五分钟缓存窗口的未解析子 Span 被审计记录，但没有形成迟到桶
写入，也没有节点映射丢失。

## 复现命令

```bash
python -m tools.convert_d1_weighted_dynamic \
  --raw-root '/path/to/Aiops-Dataset' \
  --source-root '/path/to/processed/D1' \
  --output-root data/D1_weighted \
  --overwrite

python -m tools.reacr_rca_topology_audit \
  --dataset D1_weighted \
  --output data/D1_weighted/topology_audit.json
```

## 论文表述边界

D1_weighted 可以表述为“从 D1 原始 Jaeger Span 按分钟重建的加权动态图”。不能将
新边权表述为旧 `samples.pkl` 原生提供，也不能把观察调用关系称为经过干预识别的
真实因果图。动态图性能结论仍需与二值动态边、冻结窗口图和去除边属性的多 seed
消融共同验证。
