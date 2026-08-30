"""Rebuild D1 as a trace-weighted dynamic graph dataset.

The released D1 ``samples.pkl`` files contain time-varying, unweighted DGL
graphs.  The original AIOps dataset additionally provides Jaeger span CSVs.
This converter joins ``(trace_id, parent_span)`` to ``(trace_id, span_id)``
and aggregates cross-instance calls in 60-second buckets.  It preserves the
existing 130-dimensional node observations and the original train/test split,
while replacing legacy pod-to-pod catalogue edges with calls observed in the
corresponding time bucket.

Each output graph contains three explicitly typed edge families:

* observed call edges with count, rate, mean latency, p95 latency, error rate;
* pod--node deployment edges copied from the matching original snapshot;
* self loops copied from the matching original snapshot.

The source D1 files are never modified.  Write the output to a separate
``data/D1_weighted`` directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import shutil
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import dgl
import numpy as np
import torch


BUCKET_SECONDS = 60
CALL_FEATURE_NAMES: Tuple[str, ...] = (
    "log1p_call_count",
    "log1p_call_rate",
    "log1p_mean_latency_ms",
    "log1p_p95_latency_ms",
    "error_rate",
)
EDGE_TYPE_FEATURE_NAMES: Tuple[str, ...] = (
    "is_call",
    "is_deployment",
    "is_self",
)
EDGE_FEATURE_NAMES = CALL_FEATURE_NAMES + EDGE_TYPE_FEATURE_NAMES
REQUIRED_TRACE_COLUMNS = {
    "timestamp",
    "cmdb_id",
    "span_id",
    "trace_id",
    "duration",
    "status_code",
    "parent_span",
}


@dataclass
class RunningEdgeStats:
    call_count: int = 0
    error_count: int = 0
    latency_ms: List[float] = field(default_factory=list)

    def add(self, latency_ms: float, is_error: bool) -> None:
        self.call_count += 1
        self.error_count += int(is_error)
        self.latency_ms.append(float(max(latency_ms, 0.0)))


@dataclass(frozen=True)
class CompactEdgeStats:
    call_count: int
    mean_latency_ms: float
    p95_latency_ms: float
    error_rate: float


def _bucket(timestamp_ms: int, bucket_seconds: int = BUCKET_SECONDS) -> int:
    timestamp_seconds = int(timestamp_ms) // 1000
    return (timestamp_seconds // bucket_seconds) * bucket_seconds


def _status_is_error(value: object) -> bool:
    """Normalize the heterogeneous Jaeger status codes used by raw D1."""

    text = str(value if value is not None else "").strip().lower()
    if text in {"", "0", "ok", "unset", "none", "nan", "false"}:
        return False
    try:
        numeric = int(float(text))
    except ValueError:
        return text in {"error", "err", "failed", "failure", "true"}
    # Raw D1 mixes HTTP codes with canonical gRPC codes.  HTTP 1xx--3xx are
    # successful, while non-zero gRPC codes (1--16) denote failures.
    if 100 <= numeric < 400:
        return False
    return numeric != 0


def _compact(stats: RunningEdgeStats) -> CompactEdgeStats:
    latency = np.asarray(stats.latency_ms, dtype=np.float64)
    return CompactEdgeStats(
        call_count=int(stats.call_count),
        mean_latency_ms=float(latency.mean()) if latency.size else 0.0,
        p95_latency_ms=float(np.percentile(latency, 95)) if latency.size else 0.0,
        error_rate=float(stats.error_count / max(stats.call_count, 1)),
    )


def aggregate_trace_csv(
    trace_path: Path,
    node_hash: Mapping[str, int],
    target_buckets: Set[int],
    bucket_seconds: int = BUCKET_SECONDS,
    max_parent_gap_seconds: int = 300,
    max_rows: Optional[int] = None,
) -> Tuple[Dict[int, Dict[Tuple[int, int], CompactEdgeStats]], Dict[str, object]]:
    """Stream one raw D1 span CSV and aggregate observed cross-instance calls.

    Rows are timestamp ordered, but trace IDs are interleaved.  A bounded span
    cache plus a pending-child map handles both parent-before-child and the rare
    child-before-parent ordering without loading the multi-gigabyte CSV.
    """

    trace_path = Path(trace_path)
    gap_ms = int(max_parent_gap_seconds * 1000)
    span_cache: Dict[Tuple[str, str], Tuple[str, int]] = {}
    span_queue: deque[Tuple[int, Tuple[str, str]]] = deque()
    pending: DefaultDict[Tuple[str, str], List[Tuple[int, str, float, bool]]] = defaultdict(list)
    active: DefaultDict[int, Dict[Tuple[int, int], RunningEdgeStats]] = defaultdict(dict)
    finalized: Dict[int, Dict[Tuple[int, int], CompactEdgeStats]] = {}
    status_counts: Counter[str] = Counter()
    unmapped_nodes: Counter[str] = Counter()

    audit: Dict[str, object] = {
        "path": str(trace_path),
        "rows": 0,
        "rows_with_parent": 0,
        "resolved_parent_links": 0,
        "cross_instance_calls": 0,
        "same_instance_parent_links": 0,
        "unmapped_parent_or_child": 0,
        "late_calls_after_bucket_finalization": 0,
        "non_monotonic_rows": 0,
        "expired_span_records": 0,
        "expired_pending_children": 0,
    }

    max_seen_ms = -1
    previous_ms = -1
    last_cleanup_bucket: Optional[int] = None

    def add_call(parent_cmdb: str, child_cmdb: str, child_ts: int, latency_ms: float, is_error: bool) -> None:
        audit["resolved_parent_links"] = int(audit["resolved_parent_links"]) + 1
        if parent_cmdb == child_cmdb:
            audit["same_instance_parent_links"] = int(audit["same_instance_parent_links"]) + 1
            return
        if parent_cmdb not in node_hash or child_cmdb not in node_hash:
            audit["unmapped_parent_or_child"] = int(audit["unmapped_parent_or_child"]) + 1
            if parent_cmdb not in node_hash:
                unmapped_nodes[parent_cmdb] += 1
            if child_cmdb not in node_hash:
                unmapped_nodes[child_cmdb] += 1
            return
        bucket = _bucket(child_ts, bucket_seconds)
        if bucket not in target_buckets:
            return
        if bucket in finalized:
            audit["late_calls_after_bucket_finalization"] = int(
                audit["late_calls_after_bucket_finalization"]
            ) + 1
            return
        edge = (int(node_hash[parent_cmdb]), int(node_hash[child_cmdb]))
        accumulator = active[bucket].setdefault(edge, RunningEdgeStats())
        accumulator.add(latency_ms=latency_ms, is_error=is_error)
        audit["cross_instance_calls"] = int(audit["cross_instance_calls"]) + 1

    def finalize_old_buckets(cutoff_seconds: int) -> None:
        old = [bucket for bucket in active if bucket + bucket_seconds <= cutoff_seconds]
        for bucket in sorted(old):
            finalized[bucket] = {
                edge: _compact(stats) for edge, stats in active.pop(bucket).items()
            }

    def expire_state(cutoff_ms: int) -> None:
        while span_queue and span_queue[0][0] < cutoff_ms:
            timestamp, key = span_queue.popleft()
            cached = span_cache.get(key)
            if cached is not None and cached[1] == timestamp:
                del span_cache[key]
                audit["expired_span_records"] = int(audit["expired_span_records"]) + 1
        if not pending:
            return
        expired = 0
        for key in list(pending):
            retained = [record for record in pending[key] if record[0] >= cutoff_ms]
            expired += len(pending[key]) - len(retained)
            if retained:
                pending[key] = retained
            else:
                del pending[key]
        audit["expired_pending_children"] = int(audit["expired_pending_children"]) + expired

    with trace_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_TRACE_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{trace_path} is missing trace columns: {sorted(missing)}")

        for row_idx, row in enumerate(reader):
            if max_rows is not None and row_idx >= max_rows:
                break
            try:
                timestamp_ms = int(row["timestamp"])
                duration_ms = float(row.get("duration", 0.0) or 0.0) / 1000.0
            except (TypeError, ValueError):
                continue

            audit["rows"] = int(audit["rows"]) + 1
            if previous_ms > timestamp_ms:
                audit["non_monotonic_rows"] = int(audit["non_monotonic_rows"]) + 1
            previous_ms = timestamp_ms
            max_seen_ms = max(max_seen_ms, timestamp_ms)

            trace_id = str(row.get("trace_id", ""))
            span_id = str(row.get("span_id", ""))
            cmdb_id = str(row.get("cmdb_id", ""))
            parent_span = str(row.get("parent_span", "") or "")
            is_error = _status_is_error(row.get("status_code"))
            status_counts[str(row.get("status_code", ""))] += 1

            if parent_span:
                audit["rows_with_parent"] = int(audit["rows_with_parent"]) + 1
                parent_key = (trace_id, parent_span)
                parent = span_cache.get(parent_key)
                if parent is not None:
                    add_call(parent[0], cmdb_id, timestamp_ms, duration_ms, is_error)
                else:
                    pending[parent_key].append((timestamp_ms, cmdb_id, duration_ms, is_error))

            span_key = (trace_id, span_id)
            span_cache[span_key] = (cmdb_id, timestamp_ms)
            span_queue.append((timestamp_ms, span_key))
            for child_ts, child_cmdb, child_latency, child_error in pending.pop(span_key, []):
                add_call(cmdb_id, child_cmdb, child_ts, child_latency, child_error)

            current_bucket = _bucket(max_seen_ms, bucket_seconds)
            if last_cleanup_bucket != current_bucket:
                cutoff_ms = max_seen_ms - gap_ms
                expire_state(cutoff_ms)
                finalize_old_buckets(cutoff_ms // 1000)
                last_cleanup_bucket = current_bucket

    # Resolve no more parents after EOF; retain their count for audit.
    audit["unresolved_parent_links"] = int(sum(len(records) for records in pending.values()))
    for bucket in sorted(active):
        finalized[bucket] = {
            edge: _compact(stats) for edge, stats in active[bucket].items()
        }
    audit["status_code_counts"] = dict(status_counts)
    audit["unmapped_node_counts"] = dict(unmapped_nodes)
    audit["output_buckets_with_calls"] = int(sum(bool(edges) for edges in finalized.values()))
    audit["output_unique_call_edges"] = int(
        len({edge for edges in finalized.values() for edge in edges})
    )
    audit["parent_resolution_rate"] = float(
        int(audit["resolved_parent_links"]) / max(int(audit["rows_with_parent"]), 1)
    )
    return finalized, audit


def _inverse_node_hash(node_hash: Mapping[str, int]) -> Dict[int, str]:
    inverse = {int(index): str(name) for name, index in node_hash.items()}
    if len(inverse) != len(node_hash):
        raise ValueError("node_hash must map unique node names to unique integer IDs")
    return inverse


def _structural_edges(graph: dgl.DGLGraph, inverse: Mapping[int, str]) -> List[Tuple[int, int, str]]:
    source, target = graph.edges(order="eid")
    result: List[Tuple[int, int, str]] = []
    for src, dst in zip(source.tolist(), target.tolist()):
        if src == dst:
            result.append((int(src), int(dst), "self"))
            continue
        src_is_node = inverse[int(src)].startswith("node-")
        dst_is_node = inverse[int(dst)].startswith("node-")
        if src_is_node != dst_is_node:
            result.append((int(src), int(dst), "deployment"))
    return result


def build_weighted_graph(
    original_graph: dgl.DGLGraph,
    edge_stats: Mapping[Tuple[int, int], CompactEdgeStats],
    node_hash: Mapping[str, int],
    bucket_seconds: int = BUCKET_SECONDS,
) -> dgl.DGLGraph:
    """Combine observed calls with deployment edges and self loops."""

    inverse = _inverse_node_hash(node_hash)
    records: List[Tuple[int, int, float, List[float]]] = []
    for src, dst, edge_type in _structural_edges(original_graph, inverse):
        flags = [0.0, float(edge_type == "deployment"), float(edge_type == "self")]
        records.append((src, dst, 1.0, [0.0] * len(CALL_FEATURE_NAMES) + flags))

    for (src, dst), stats in sorted(edge_stats.items()):
        count = float(stats.call_count)
        call_features = [
            math.log1p(count),
            math.log1p(count / bucket_seconds),
            math.log1p(stats.mean_latency_ms),
            math.log1p(stats.p95_latency_ms),
            float(stats.error_rate),
        ]
        records.append((int(src), int(dst), count, call_features + [1.0, 0.0, 0.0]))

    source = [record[0] for record in records]
    target = [record[1] for record in records]
    graph = dgl.graph((source, target), num_nodes=original_graph.num_nodes())
    graph.edata["weight"] = torch.tensor(
        [record[2] for record in records], dtype=torch.float32
    )
    graph.edata["edge_attr"] = torch.tensor(
        [record[3] for record in records], dtype=torch.float32
    ).reshape(-1, len(EDGE_FEATURE_NAMES))
    for key, value in original_graph.ndata.items():
        graph.ndata[key] = value.clone()
    return graph


def convert_samples(
    samples: Sequence[Tuple[int, dgl.DGLGraph, torch.Tensor]],
    trace_by_bucket: Mapping[int, Mapping[Tuple[int, int], CompactEdgeStats]],
    node_hash: Mapping[str, int],
) -> Tuple[List[Tuple[int, dgl.DGLGraph, torch.Tensor]], Dict[str, object]]:
    converted = []
    call_edge_counts: List[int] = []
    total_edge_counts: List[int] = []
    weighted_buckets = 0
    for timestamp, original_graph, features in samples:
        stats = trace_by_bucket.get(int(timestamp), {})
        graph = build_weighted_graph(original_graph, stats, node_hash)
        converted.append((int(timestamp), graph, features))
        call_edge_counts.append(len(stats))
        total_edge_counts.append(int(graph.num_edges()))
        weighted_buckets += int(bool(stats))
    audit = {
        "sample_count": len(converted),
        "buckets_with_observed_calls": weighted_buckets,
        "empty_call_bucket_count": len(converted) - weighted_buckets,
        "call_edges_per_bucket": {
            "min": int(min(call_edge_counts, default=0)),
            "max": int(max(call_edge_counts, default=0)),
            "mean": float(np.mean(call_edge_counts)) if call_edge_counts else 0.0,
        },
        "total_edges_per_bucket": {
            "min": int(min(total_edge_counts, default=0)),
            "max": int(max(total_edge_counts, default=0)),
            "mean": float(np.mean(total_edge_counts)) if total_edge_counts else 0.0,
        },
    }
    return converted, audit


def summarize_sequence_dynamics(
    samples: Sequence[Tuple[int, dgl.DGLGraph, torch.Tensor]],
) -> Dict[str, int]:
    """Count topology and observed call-weight changes in timestamp order."""

    edge_switches = 0
    weight_switches = 0
    previous_edges = None
    previous_weights = None
    for _, graph, _ in sorted(samples, key=lambda item: int(item[0])):
        source, target = graph.edges(order="eid")
        pairs = list(zip(source.tolist(), target.tolist()))
        edge_signature = tuple(sorted((int(src), int(dst)) for src, dst in pairs))
        call_mask = graph.edata["edge_attr"][:, len(CALL_FEATURE_NAMES)] > 0.5
        weights = graph.edata["weight"].tolist()
        call_signature = tuple(
            sorted(
                (int(src), int(dst), round(float(weight), 6))
                for (src, dst), weight, is_call in zip(
                    pairs, weights, call_mask.tolist()
                )
                if is_call
            )
        )
        edge_switches += int(
            previous_edges is not None and edge_signature != previous_edges
        )
        weight_switches += int(
            previous_weights is not None and call_signature != previous_weights
        )
        previous_edges = edge_signature
        previous_weights = call_signature
    return {
        "snapshot_count": len(samples),
        "edge_set_switches": edge_switches,
        "call_weight_signature_switches": weight_switches,
    }


def _load_pickle(path: Path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _save_pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _trace_files(raw_root: Path) -> List[Path]:
    files = sorted(Path(raw_root).glob("data/*/trace/all/trace_jaeger-span.csv"))
    if not files:
        raise FileNotFoundError(f"no D1 Jaeger CSV files found below {raw_root}")
    return files


def convert_dataset(args: argparse.Namespace) -> Dict[str, object]:
    source_root = Path(args.source_root).resolve()
    raw_root = Path(args.raw_root).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"non-empty output directory: {output_root}; use --overwrite")
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    source_samples = source_root / "samples"
    train_samples = _load_pickle(source_samples / "train_samples.pkl")
    test_samples = _load_pickle(source_samples / "test_samples.pkl")
    node_hash = _load_pickle(source_root / "graphs_info" / "node_hash.pkl")
    target_buckets = {int(sample[0]) for sample in train_samples}
    target_buckets.update(int(sample[0]) for sample in test_samples)

    trace_by_bucket: Dict[int, Dict[Tuple[int, int], CompactEdgeStats]] = {}
    trace_audits = []
    files = _trace_files(raw_root)
    if args.max_files > 0:
        files = files[: args.max_files]
    for file_index, path in enumerate(files, start=1):
        print(
            f"[D1 weighted] aggregating trace file {file_index}/{len(files)}: {path}",
            flush=True,
        )
        stats, audit = aggregate_trace_csv(
            path,
            node_hash=node_hash,
            target_buckets=target_buckets,
            bucket_seconds=args.bucket_seconds,
            max_parent_gap_seconds=args.max_parent_gap_seconds,
            max_rows=None if args.max_rows_per_file <= 0 else args.max_rows_per_file,
        )
        overlap = set(trace_by_bucket).intersection(stats)
        if overlap:
            raise ValueError(f"trace files overlap in {len(overlap)} output buckets")
        trace_by_bucket.update(stats)
        trace_audits.append(audit)
        print(
            "[D1 weighted] completed "
            f"{path.parent.parent.parent.name}: rows={audit['rows']}, "
            f"resolved={audit['resolved_parent_links']}, "
            f"weighted_buckets={audit['output_buckets_with_calls']}",
            flush=True,
        )

    print("[D1 weighted] rebuilding train/test DGL snapshots", flush=True)
    converted_train, train_audit = convert_samples(train_samples, trace_by_bucket, node_hash)
    converted_test, test_audit = convert_samples(test_samples, trace_by_bucket, node_hash)
    _save_pickle(output_root / "samples" / "train_samples.pkl", converted_train)
    _save_pickle(output_root / "samples" / "test_samples.pkl", converted_test)
    _save_pickle(output_root / "graphs_info" / "node_hash.pkl", node_hash)

    call_edges = {
        (src, dst)
        for bucket_edges in trace_by_bucket.values()
        for src, dst in bucket_edges
    }
    inverse = _inverse_node_hash(node_hash)
    _save_pickle(
        output_root / "graphs_info" / "call_edges.pkl",
        {(inverse[src], inverse[dst]) for src, dst in call_edges},
    )
    structural = {
        (inverse[src], inverse[dst], edge_type)
        for samples in (train_samples, test_samples)
        for _, graph, _ in samples
        for src, dst, edge_type in _structural_edges(graph, inverse)
    }
    _save_pickle(output_root / "graphs_info" / "structural_edges.pkl", structural)
    shutil.copy2(source_root / "cases.csv", output_root / "cases.csv")

    metadata = {
        "dataset": "D1_weighted_dynamic",
        "source_processed_dataset": str(source_root),
        "source_raw_dataset": str(raw_root),
        "bucket_seconds": int(args.bucket_seconds),
        "node_count": len(node_hash),
        "feature_dim": int(train_samples[0][2].shape[-1]),
        "edge_feature_names": list(EDGE_FEATURE_NAMES),
        "edge_feature_dim": len(EDGE_FEATURE_NAMES),
        "call_edge_semantics": "parent_span_cmdb_id_to_child_span_cmdb_id",
        "latency_conversion": "duration_microseconds_divided_by_1000",
        "legacy_pod_call_edges_replaced": True,
        "deployment_and_self_edges_preserved": True,
        "train": train_audit,
        "test": test_audit,
        "trace_files": trace_audits,
        "target_bucket_count": len(target_buckets),
        "trace_bucket_count": len(trace_by_bucket),
        "unique_observed_call_edges": len(call_edges),
        "sequence_dynamics": summarize_sequence_dynamics(
            list(converted_train) + list(converted_test)
        ),
    }
    with (output_root / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return metadata


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build trace-weighted dynamic D1 graphs")
    parser.add_argument("--raw-root", required=True, help="AIOps-Dataset root")
    parser.add_argument("--source-root", required=True, help="existing processed D1 directory")
    parser.add_argument("--output-root", default=str(root / "data" / "D1_weighted"))
    parser.add_argument("--bucket-seconds", type=int, default=BUCKET_SECONDS)
    parser.add_argument("--max-parent-gap-seconds", type=int, default=300)
    parser.add_argument("--max-files", type=int, default=0, help="development only; 0 means all")
    parser.add_argument("--max-rows-per-file", type=int, default=0, help="development only; 0 means all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    metadata = convert_dataset(parse_args())
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
