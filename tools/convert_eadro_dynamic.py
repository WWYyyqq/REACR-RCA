"""Convert the raw Eadro SN/TT telemetry into REACR dynamic graph samples.

The converter deliberately does not reuse Eadro's pre-built ``chunks``.  Those
artifacts contain one static edge catalogue.  Instead, this module rebuilds a
directed invocation graph for every time bucket from Jaeger ``CHILD_OF`` span
relations.

Output contract
---------------
``samples/{train,test}_samples.pkl`` contain the legacy REACR tuple:

    (bucket_start_timestamp, dgl_graph, node_features[N, F])

The service node set and ordering stay fixed while graph edges, scalar
``edata['weight']`` values, and multi-dimensional ``edata['edge_attr']`` values
may change at every bucket.  Empty observed buckets remain empty graphs.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import dgl
import numpy as np
import pandas as pd
import torch


METRIC_NAMES: Tuple[str, ...] = (
    "cpu_usage_system",
    "cpu_usage_total",
    "cpu_usage_user",
    "memory_usage",
    "memory_working_set",
    "rx_bytes",
    "tx_bytes",
)

METRIC_FEATURE_NAMES: Tuple[str, ...] = tuple(
    f"{stat}_{metric}"
    for stat in ("mean", "max", "std", "slope")
    for metric in METRIC_NAMES
) + ("metric_present",)

TRACE_FEATURE_NAMES: Tuple[str, ...] = (
    "log1p_out_call_count",
    "log1p_in_call_count",
    "log1p_out_mean_latency_ms",
    "log1p_in_mean_latency_ms",
    "log1p_out_p95_latency_ms",
    "log1p_in_p95_latency_ms",
    "out_error_rate",
    "in_error_rate",
    "log1p_span_count",
    "trace_present",
)

LOG_FEATURE_NAMES: Tuple[str, ...] = (
    "log1p_log_count",
    "log1p_error_log_count",
    "log1p_warn_log_count",
    "log1p_info_log_count",
    "log1p_debug_log_count",
    "log_present",
)

FEATURE_NAMES: Tuple[str, ...] = METRIC_FEATURE_NAMES + TRACE_FEATURE_NAMES + LOG_FEATURE_NAMES
FEATURE_DIM = len(FEATURE_NAMES)
METRIC_DIM = len(METRIC_FEATURE_NAMES)
TRACE_DIM = len(TRACE_FEATURE_NAMES)
LOG_DIM = len(LOG_FEATURE_NAMES)
MASK_FEATURE_INDICES: Tuple[int, ...] = (
    METRIC_DIM - 1,
    METRIC_DIM + TRACE_DIM - 1,
    FEATURE_DIM - 1,
)

EDGE_FEATURE_NAMES: Tuple[str, ...] = (
    "log1p_call_count",
    "log1p_call_rate",
    "log1p_mean_latency_ms",
    "log1p_p95_latency_ms",
    "error_rate",
)


SN_SERVICES: Tuple[str, ...] = (
    "social-graph-service",
    "compose-post-service",
    "post-storage-service",
    "user-timeline-service",
    "url-shorten-service",
    "user-service",
    "media-service",
    "text-service",
    "unique-id-service",
    "user-mention-service",
    "home-timeline-service",
    "nginx-web-server",
)

TT_SERVICES: Tuple[str, ...] = (
    "ts-assurance-service",
    "ts-auth-service",
    "ts-basic-service",
    "ts-cancel-service",
    "ts-config-service",
    "ts-contacts-service",
    "ts-food-map-service",
    "ts-food-service",
    "ts-inside-payment-service",
    "ts-notification-service",
    "ts-order-other-service",
    "ts-order-service",
    "ts-payment-service",
    "ts-preserve-service",
    "ts-price-service",
    "ts-route-plan-service",
    "ts-route-service",
    "ts-seat-service",
    "ts-security-service",
    "ts-station-service",
    "ts-ticketinfo-service",
    "ts-train-service",
    "ts-travel-plan-service",
    "ts-travel-service",
    "ts-travel2-service",
    "ts-user-service",
    "ts-verification-code-service",
)

SN_FAULT_NAME_MAP: Mapping[str, str] = {
    "socialnetwork-text-service-1": "text-service",
    "socialnetwork-home-timeline-service-1": "home-timeline-service",
    "socialnetwork-media-service-1": "media-service",
    "socialnetwork-post-storage-service-1": "post-storage-service",
    "socialnetwork-social-graph-service-1": "social-graph-service",
    "socialnetwork-url-shorten-service-1": "url-shorten-service",
    "socialnetwork-nginx-thrift-1": "nginx-web-server",
    "socialnetwork-unique-id-service-1": "unique-id-service",
    "socialnetwork-user-service-1": "user-service",
    "socialnetwork-compose-post-service-1": "compose-post-service",
    "socialnetwork-user-timeline-service-1": "user-timeline-service",
    "socialnetwork-user-mention-service-1": "user-mention-service",
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    services: Tuple[str, ...]
    bucket_seconds: int
    span_time_offset_seconds: int = -8 * 3600
    log_time_offset_seconds: int = -8 * 3600

    @property
    def node_hash(self) -> Dict[str, int]:
        return {service: idx for idx, service in enumerate(self.services)}


DATASET_SPECS: Mapping[str, DatasetSpec] = {
    "SN": DatasetSpec("SN", SN_SERVICES, bucket_seconds=10),
    "TT": DatasetSpec("TT", TT_SERVICES, bucket_seconds=60),
}


@dataclass(frozen=True)
class Session:
    dataset: str
    session_id: str
    path: Path
    record_path: Path
    is_fault: bool


@dataclass
class EdgeAccumulator:
    call_count: int = 0
    error_count: int = 0
    latencies_ms: List[float] = field(default_factory=list)

    def add(self, latency_ms: float, is_error: bool) -> None:
        self.call_count += 1
        self.error_count += int(is_error)
        self.latencies_ms.append(float(max(latency_ms, 0.0)))


@dataclass
class SessionAudit:
    session_id: str
    is_fault: bool
    start_time: int
    end_time: int
    sample_count: int
    unique_valid_traces: int
    duplicate_trace_objects: int
    stale_trace_objects: int
    valid_spans: int
    cross_service_calls: int
    graph_edge_switches: int
    graph_weight_switches: int
    empty_graph_steps: int
    log_line_count: int
    unparsable_log_lines: int


def iter_json_array(path: Path, chunk_chars: int = 4 * 1024 * 1024) -> Iterator[object]:
    """Stream objects from a top-level JSON array using only the stdlib."""

    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        started = False
        eof = False

        while True:
            if not eof and len(buffer) < chunk_chars:
                chunk = handle.read(chunk_chars)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            if not started:
                stripped = buffer.lstrip()
                if not stripped:
                    if eof:
                        raise ValueError(f"{path} is empty")
                    continue
                if stripped[0] != "[":
                    raise ValueError(f"{path} must contain a top-level JSON array")
                buffer = stripped[1:]
                started = True

            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                return
            if not buffer:
                if eof:
                    raise ValueError(f"unterminated JSON array in {path}")
                continue

            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise
                chunk = handle.read(chunk_chars)
                if chunk:
                    buffer += chunk
                    continue
                eof = True
                continue
            yield value
            buffer = buffer[end:]


def _floor_bucket(timestamp: float, bucket_seconds: int) -> int:
    return int(math.floor(float(timestamp) / bucket_seconds) * bucket_seconds)


def _session_record_path(dataset_root: Path, session_dir: Path, dataset: str) -> Path:
    prefix = f"{dataset}."
    if not session_dir.name.startswith(prefix):
        raise ValueError(f"unexpected {dataset} session directory: {session_dir.name}")
    suffix = session_dir.name[len(prefix):]
    record_path = session_dir.parent / f"{dataset}.fault-{suffix}.json"
    if not record_path.exists():
        raise FileNotFoundError(f"missing fault record for {session_dir}: {record_path}")
    return record_path


def discover_sessions(dataset_root: Path, dataset: str) -> List[Session]:
    sessions: List[Session] = []
    for dirname, is_fault in (("data", True), ("no fault", False)):
        split_dir = dataset_root / dirname
        if not split_dir.exists():
            raise FileNotFoundError(split_dir)
        for session_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            sessions.append(
                Session(
                    dataset=dataset,
                    session_id=session_dir.name,
                    path=session_dir,
                    record_path=_session_record_path(dataset_root, session_dir, dataset),
                    is_fault=is_fault,
                )
            )
    return sessions


def normalize_fault_service(dataset: str, raw_name: str) -> str:
    if dataset == "SN":
        if raw_name not in SN_FAULT_NAME_MAP:
            raise KeyError(f"unknown SN fault container {raw_name!r}")
        return SN_FAULT_NAME_MAP[raw_name]
    if dataset == "TT":
        prefix = "dockercomposemanifests_"
        normalized = raw_name[len(prefix):] if raw_name.startswith(prefix) else raw_name
        if normalized.endswith("_1"):
            normalized = normalized[:-2]
        return normalized
    raise ValueError(dataset)


def _metric_time_range(session: Session, spec: DatasetSpec) -> Tuple[int, int]:
    starts: List[int] = []
    ends: List[int] = []
    for service in spec.services:
        metric_path = session.path / "metrics" / f"{service}.csv"
        if not metric_path.exists():
            raise FileNotFoundError(metric_path)
        timestamp = pd.read_csv(metric_path, usecols=["timestamp"])["timestamp"]
        if timestamp.empty:
            raise ValueError(f"empty metric file: {metric_path}")
        starts.append(int(timestamp.min()))
        ends.append(int(timestamp.max()) + 1)
    # Use the common observable interval so every node has metric coverage.
    start_time = max(starts)
    end_time = min(ends)
    if end_time <= start_time:
        raise ValueError(f"empty common metric range in {session.session_id}")
    return start_time, end_time


def _bucket_grid(start_time: int, end_time: int, bucket_seconds: int) -> List[int]:
    first = _floor_bucket(start_time, bucket_seconds)
    last = _floor_bucket(end_time - 1, bucket_seconds)
    return list(range(first, last + bucket_seconds, bucket_seconds))


def _aggregate_metrics(
    session: Session,
    spec: DatasetSpec,
    start_time: int,
    end_time: int,
    features_by_time: MutableMapping[int, np.ndarray],
) -> None:
    node_hash = spec.node_hash
    usecols = ["timestamp", *METRIC_NAMES]
    for service in spec.services:
        node_idx = node_hash[service]
        metric_path = session.path / "metrics" / f"{service}.csv"
        frame = pd.read_csv(metric_path, usecols=usecols)
        frame = frame[(frame["timestamp"] >= start_time) & (frame["timestamp"] < end_time)]
        frame["_bucket"] = (
            (frame["timestamp"].astype(np.int64) // spec.bucket_seconds)
            * spec.bucket_seconds
        )
        for bucket, group in frame.groupby("_bucket", sort=False):
            bucket = int(bucket)
            if bucket not in features_by_time:
                continue
            values = group.loc[:, METRIC_NAMES].to_numpy(dtype=np.float64)
            timestamps = group["timestamp"].to_numpy(dtype=np.float64)
            mean = np.mean(values, axis=0)
            maximum = np.max(values, axis=0)
            std = np.std(values, axis=0)
            elapsed = max(float(timestamps[-1] - timestamps[0]), 1.0)
            slope = (values[-1] - values[0]) / elapsed
            row = features_by_time[bucket][node_idx]
            row[0:7] = mean
            row[7:14] = maximum
            row[14:21] = std
            row[21:28] = slope
            row[28] = 1.0


LOG_TIME_PATTERN = re.compile(
    r"(?P<time>\d{4}-(?:[A-Za-z]{3}|\d{2})-\d{2} "
    r"\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
)


def _parse_log_timestamp(line: str, offset_seconds: int) -> Optional[float]:
    match = LOG_TIME_PATTERN.search(line)
    if match is None:
        return None
    value = match.group("time")
    formats = (
        "%Y-%b-%d %H:%M:%S.%f",
        "%Y-%b-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    )
    for fmt in formats:
        try:
            parsed = datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
            return parsed.timestamp() + offset_seconds
        except ValueError:
            continue
    return None


def _aggregate_logs(
    session: Session,
    spec: DatasetSpec,
    start_time: int,
    end_time: int,
    features_by_time: MutableMapping[int, np.ndarray],
) -> Tuple[int, int]:
    log_path = session.path / "logs.json"
    if not log_path.exists():
        return 0, 0
    with log_path.open("r", encoding="utf-8") as handle:
        logs = json.load(handle)

    counts: DefaultDict[Tuple[int, int], np.ndarray] = defaultdict(
        lambda: np.zeros(5, dtype=np.float64)
    )
    node_hash = spec.node_hash
    total_lines = 0
    unparsable = 0
    for service, lines in logs.items():
        if service not in node_hash:
            continue
        node_idx = node_hash[service]
        for line in lines:
            total_lines += 1
            timestamp = _parse_log_timestamp(str(line), spec.log_time_offset_seconds)
            if timestamp is None:
                unparsable += 1
                continue
            if not (start_time <= timestamp < end_time):
                continue
            bucket = _floor_bucket(timestamp, spec.bucket_seconds)
            if bucket not in features_by_time:
                continue
            lower = str(line).lower()
            severity_idx = None
            if "error" in lower or "<err" in lower:
                severity_idx = 1
            elif "warn" in lower:
                severity_idx = 2
            elif "info" in lower:
                severity_idx = 3
            elif "debug" in lower:
                severity_idx = 4
            counts[(bucket, node_idx)][0] += 1.0
            if severity_idx is not None:
                counts[(bucket, node_idx)][severity_idx] += 1.0

    log_start = METRIC_DIM + TRACE_DIM
    for (bucket, node_idx), value in counts.items():
        row = features_by_time[bucket][node_idx]
        row[log_start:log_start + 5] = np.log1p(value)
        row[log_start + 5] = 1.0
    return total_lines, unparsable


def _span_error(span: Mapping[str, object]) -> bool:
    for tag in span.get("tags", []) or []:
        if not isinstance(tag, Mapping):
            continue
        key = str(tag.get("key", "")).lower()
        value = tag.get("value")
        value_text = str(value).lower()
        if key in {"error", "otel.status_code", "status.code"}:
            if value is True or value_text in {"true", "error", "2"}:
                return True
        if key in {"http.status_code", "status_code"}:
            try:
                if int(value) >= 500:
                    return True
            except (TypeError, ValueError):
                pass
    return False


def _aggregate_traces(
    session: Session,
    spec: DatasetSpec,
    start_time: int,
    end_time: int,
    features_by_time: MutableMapping[int, np.ndarray],
    global_valid_trace_ids: Set[str],
    max_trace_objects: Optional[int] = None,
) -> Tuple[
    Dict[int, Dict[Tuple[int, int], EdgeAccumulator]],
    Dict[str, int],
]:
    spans_path = session.path / "spans.json"
    if not spans_path.exists():
        raise FileNotFoundError(spans_path)

    node_hash = spec.node_hash
    edge_by_time: DefaultDict[int, Dict[Tuple[int, int], EdgeAccumulator]] = defaultdict(dict)
    out_latencies: DefaultDict[Tuple[int, int], List[float]] = defaultdict(list)
    in_latencies: DefaultDict[Tuple[int, int], List[float]] = defaultdict(list)
    out_errors: DefaultDict[Tuple[int, int], int] = defaultdict(int)
    in_errors: DefaultDict[Tuple[int, int], int] = defaultdict(int)
    out_counts: DefaultDict[Tuple[int, int], int] = defaultdict(int)
    in_counts: DefaultDict[Tuple[int, int], int] = defaultdict(int)
    span_counts: DefaultDict[Tuple[int, int], int] = defaultdict(int)

    local_trace_ids: Set[str] = set()
    audit = {
        "unique_valid_traces": 0,
        "duplicate_trace_objects": 0,
        "stale_trace_objects": 0,
        "valid_spans": 0,
        "cross_service_calls": 0,
    }

    for object_idx, trace in enumerate(iter_json_array(spans_path)):
        if max_trace_objects is not None and object_idx >= max_trace_objects:
            break
        if not isinstance(trace, Mapping):
            continue
        trace_id = str(trace.get("traceID", ""))
        if trace_id and trace_id in local_trace_ids:
            audit["duplicate_trace_objects"] += 1
            continue
        if trace_id:
            local_trace_ids.add(trace_id)

        processes = trace.get("processes", {}) or {}
        spans = trace.get("spans", []) or []
        if not isinstance(processes, Mapping) or not isinstance(spans, list):
            continue

        span_map = {
            str(span.get("spanID")): span
            for span in spans
            if isinstance(span, Mapping) and span.get("spanID") is not None
        }
        valid_spans: List[Tuple[Mapping[str, object], int, int, float]] = []
        for span in spans:
            if not isinstance(span, Mapping):
                continue
            process_id = str(span.get("processID", ""))
            process = processes.get(process_id, {})
            if not isinstance(process, Mapping):
                continue
            service = str(process.get("serviceName", ""))
            if service not in node_hash:
                continue
            try:
                timestamp = (
                    int(span["startTime"]) / 1_000_000.0
                    + spec.span_time_offset_seconds
                )
                duration_ms = float(span.get("duration", 0.0)) / 1000.0
            except (KeyError, TypeError, ValueError):
                continue
            if start_time <= timestamp < end_time:
                valid_spans.append(
                    (span, node_hash[service], _floor_bucket(timestamp, spec.bucket_seconds), duration_ms)
                )

        if not valid_spans:
            audit["stale_trace_objects"] += 1
            continue
        if trace_id and trace_id in global_valid_trace_ids:
            audit["duplicate_trace_objects"] += 1
            continue
        if trace_id:
            global_valid_trace_ids.add(trace_id)
        audit["unique_valid_traces"] += 1

        for span, child_idx, bucket, duration_ms in valid_spans:
            if bucket not in features_by_time:
                continue
            span_counts[(bucket, child_idx)] += 1
            audit["valid_spans"] += 1
            child_process_id = str(span.get("processID", ""))
            child_process = processes.get(child_process_id, {})
            child_service = (
                str(child_process.get("serviceName", ""))
                if isinstance(child_process, Mapping)
                else ""
            )
            is_error = _span_error(span)
            for reference in span.get("references", []) or []:
                if not isinstance(reference, Mapping):
                    continue
                if reference.get("refType") != "CHILD_OF":
                    continue
                parent = span_map.get(str(reference.get("spanID", "")))
                if not isinstance(parent, Mapping):
                    continue
                parent_process = processes.get(str(parent.get("processID", "")), {})
                if not isinstance(parent_process, Mapping):
                    continue
                parent_service = str(parent_process.get("serviceName", ""))
                if parent_service not in node_hash or child_service not in node_hash:
                    continue
                parent_idx = node_hash[parent_service]
                if parent_idx == child_idx:
                    break

                edge = (parent_idx, child_idx)
                accumulator = edge_by_time[bucket].setdefault(edge, EdgeAccumulator())
                accumulator.add(latency_ms=duration_ms, is_error=is_error)
                out_counts[(bucket, parent_idx)] += 1
                in_counts[(bucket, child_idx)] += 1
                out_latencies[(bucket, parent_idx)].append(duration_ms)
                in_latencies[(bucket, child_idx)].append(duration_ms)
                out_errors[(bucket, parent_idx)] += int(is_error)
                in_errors[(bucket, child_idx)] += int(is_error)
                audit["cross_service_calls"] += 1
                break

    trace_start = METRIC_DIM
    for bucket in features_by_time:
        for node_idx in range(len(spec.services)):
            out_values = out_latencies.get((bucket, node_idx), [])
            in_values = in_latencies.get((bucket, node_idx), [])
            out_count = out_counts.get((bucket, node_idx), 0)
            in_count = in_counts.get((bucket, node_idx), 0)
            spans_for_node = span_counts.get((bucket, node_idx), 0)
            row = features_by_time[bucket][node_idx]
            row[trace_start + 0] = math.log1p(out_count)
            row[trace_start + 1] = math.log1p(in_count)
            row[trace_start + 2] = math.log1p(np.mean(out_values) if out_values else 0.0)
            row[trace_start + 3] = math.log1p(np.mean(in_values) if in_values else 0.0)
            row[trace_start + 4] = math.log1p(np.percentile(out_values, 95) if out_values else 0.0)
            row[trace_start + 5] = math.log1p(np.percentile(in_values, 95) if in_values else 0.0)
            row[trace_start + 6] = out_errors.get((bucket, node_idx), 0) / max(out_count, 1)
            row[trace_start + 7] = in_errors.get((bucket, node_idx), 0) / max(in_count, 1)
            row[trace_start + 8] = math.log1p(spans_for_node)
            row[trace_start + 9] = float(spans_for_node > 0)

    return dict(edge_by_time), audit


def _build_graph(
    node_num: int,
    bucket_seconds: int,
    edge_stats: Mapping[Tuple[int, int], EdgeAccumulator],
) -> dgl.DGLGraph:
    ordered_edges = sorted(edge_stats)
    sources = [edge[0] for edge in ordered_edges]
    targets = [edge[1] for edge in ordered_edges]
    graph = dgl.graph((sources, targets), num_nodes=node_num)

    weights: List[float] = []
    edge_features: List[List[float]] = []
    for edge in ordered_edges:
        accumulator = edge_stats[edge]
        count = float(accumulator.call_count)
        latency = accumulator.latencies_ms
        mean_latency = float(np.mean(latency)) if latency else 0.0
        p95_latency = float(np.percentile(latency, 95)) if latency else 0.0
        error_rate = float(accumulator.error_count / max(accumulator.call_count, 1))
        weights.append(count)
        edge_features.append(
            [
                math.log1p(count),
                math.log1p(count / bucket_seconds),
                math.log1p(mean_latency),
                math.log1p(p95_latency),
                error_rate,
            ]
        )

    graph.edata["weight"] = torch.tensor(weights, dtype=torch.float32)
    graph.edata["edge_attr"] = torch.tensor(
        edge_features,
        dtype=torch.float32,
    ).reshape(-1, len(EDGE_FEATURE_NAMES))
    return graph


def _graph_signature(graph: dgl.DGLGraph) -> Tuple[Tuple[int, int], ...]:
    source, target = graph.edges(order="eid")
    return tuple(sorted(zip(source.tolist(), target.tolist())))


def _weight_signature(graph: dgl.DGLGraph) -> Tuple[Tuple[int, int, float], ...]:
    source, target = graph.edges(order="eid")
    weights = graph.edata["weight"].tolist()
    return tuple(
        sorted(
            (int(s), int(t), round(float(w), 6))
            for s, t, w in zip(source.tolist(), target.tolist(), weights)
        )
    )


def _session_graph_dynamics(graphs: Sequence[dgl.DGLGraph]) -> Tuple[int, int, int]:
    edge_switches = 0
    weight_switches = 0
    empty_steps = 0
    previous_edges = None
    previous_weights = None
    for graph in graphs:
        edges = _graph_signature(graph)
        weights = _weight_signature(graph)
        empty_steps += int(len(edges) == 0)
        if previous_edges is not None:
            edge_switches += int(edges != previous_edges)
            weight_switches += int(weights != previous_weights)
        previous_edges = edges
        previous_weights = weights
    return edge_switches, weight_switches, empty_steps


def _load_cases(session: Session, spec: DatasetSpec) -> List[Dict[str, object]]:
    with session.record_path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    cases: List[Dict[str, object]] = []
    for fault in record.get("faults", []):
        service = normalize_fault_service(spec.name, str(fault["name"]))
        if service not in spec.node_hash:
            raise ValueError(
                f"fault label {service!r} has no graph node in {session.session_id}"
            )
        start = float(fault["start"])
        duration = float(fault["duration"])
        cases.append(
            {
                "timestamp": int(math.floor(start)),
                "fault_start": start,
                "fault_end": start + duration,
                "duration": duration,
                "cmdb_id": service,
                "failure_type": str(fault["fault"]),
                "session_id": session.session_id,
                "raw_container_id": str(fault["name"]),
            }
        )
    return cases


def convert_session(
    session: Session,
    spec: DatasetSpec,
    global_valid_trace_ids: Set[str],
    max_trace_objects: Optional[int] = None,
) -> Tuple[List[Tuple[int, dgl.DGLGraph, torch.Tensor]], List[Dict[str, object]], SessionAudit]:
    start_time, end_time = _metric_time_range(session, spec)
    bucket_times = _bucket_grid(start_time, end_time, spec.bucket_seconds)
    features_by_time = {
        timestamp: np.zeros((len(spec.services), FEATURE_DIM), dtype=np.float32)
        for timestamp in bucket_times
    }

    _aggregate_metrics(session, spec, start_time, end_time, features_by_time)
    log_line_count, unparsable_log_lines = _aggregate_logs(
        session,
        spec,
        start_time,
        end_time,
        features_by_time,
    )
    edge_by_time, trace_audit = _aggregate_traces(
        session,
        spec,
        start_time,
        end_time,
        features_by_time,
        global_valid_trace_ids=global_valid_trace_ids,
        max_trace_objects=max_trace_objects,
    )

    samples: List[Tuple[int, dgl.DGLGraph, torch.Tensor]] = []
    graphs: List[dgl.DGLGraph] = []
    for timestamp in bucket_times:
        graph = _build_graph(
            node_num=len(spec.services),
            bucket_seconds=spec.bucket_seconds,
            edge_stats=edge_by_time.get(timestamp, {}),
        )
        features = torch.tensor(features_by_time[timestamp], dtype=torch.float32)
        samples.append((timestamp, graph, features))
        graphs.append(graph)

    edge_switches, weight_switches, empty_steps = _session_graph_dynamics(graphs)
    audit = SessionAudit(
        session_id=session.session_id,
        is_fault=session.is_fault,
        start_time=start_time,
        end_time=end_time,
        sample_count=len(samples),
        unique_valid_traces=trace_audit["unique_valid_traces"],
        duplicate_trace_objects=trace_audit["duplicate_trace_objects"],
        stale_trace_objects=trace_audit["stale_trace_objects"],
        valid_spans=trace_audit["valid_spans"],
        cross_service_calls=trace_audit["cross_service_calls"],
        graph_edge_switches=edge_switches,
        graph_weight_switches=weight_switches,
        empty_graph_steps=empty_steps,
        log_line_count=log_line_count,
        unparsable_log_lines=unparsable_log_lines,
    )
    cases = _load_cases(session, spec) if session.is_fault else []
    return samples, cases, audit


def _normalize_node_features(
    train_samples: Sequence[Tuple[int, dgl.DGLGraph, torch.Tensor]],
    test_samples: Sequence[Tuple[int, dgl.DGLGraph, torch.Tensor]],
) -> Tuple[
    List[Tuple[int, dgl.DGLGraph, torch.Tensor]],
    List[Tuple[int, dgl.DGLGraph, torch.Tensor]],
    np.ndarray,
    np.ndarray,
]:
    if not train_samples:
        raise ValueError("normal training samples are required for feature normalization")
    train_stack = torch.stack([sample[2] for sample in train_samples]).numpy()
    mean = np.mean(train_stack, axis=0)
    std = np.std(train_stack, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    for mask_idx in MASK_FEATURE_INDICES:
        mean[:, mask_idx] = 0.0
        std[:, mask_idx] = 1.0

    def transform(samples):
        transformed = []
        mean_tensor = torch.tensor(mean, dtype=torch.float32)
        std_tensor = torch.tensor(std, dtype=torch.float32)
        for timestamp, graph, features in samples:
            normalized = (features - mean_tensor) / std_tensor
            for mask_idx in MASK_FEATURE_INDICES:
                normalized[:, mask_idx] = features[:, mask_idx]
            transformed.append((timestamp, graph, normalized))
        return transformed

    return transform(train_samples), transform(test_samples), mean, std


def _prior_call_edges(
    samples: Sequence[Tuple[int, dgl.DGLGraph, torch.Tensor]],
    services: Sequence[str],
) -> Set[Tuple[str, str]]:
    edges: Set[Tuple[str, str]] = set()
    for _, graph, _ in samples:
        source, target = graph.edges()
        edges.update(
            (services[int(source_idx)], services[int(target_idx)])
            for source_idx, target_idx in zip(source.tolist(), target.tolist())
        )
    return edges


def _write_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def convert_dataset(
    input_root: Path,
    output_root: Path,
    dataset: str,
    bucket_seconds: Optional[int] = None,
    span_time_offset_seconds: Optional[int] = None,
    log_time_offset_seconds: Optional[int] = None,
    max_sessions: Optional[int] = None,
    max_trace_objects: Optional[int] = None,
    overwrite: bool = False,
) -> Dict[str, object]:
    base_spec = DATASET_SPECS[dataset]
    spec = DatasetSpec(
        name=base_spec.name,
        services=base_spec.services,
        bucket_seconds=bucket_seconds or base_spec.bucket_seconds,
        span_time_offset_seconds=(
            base_spec.span_time_offset_seconds
            if span_time_offset_seconds is None
            else span_time_offset_seconds
        ),
        log_time_offset_seconds=(
            base_spec.log_time_offset_seconds
            if log_time_offset_seconds is None
            else log_time_offset_seconds
        ),
    )
    dataset_root = input_root / f"{dataset} Dataset"
    destination = output_root / dataset
    sentinel = destination / "metadata.json"
    if sentinel.exists() and not overwrite:
        raise FileExistsError(
            f"{destination} already contains converted data; pass --overwrite to replace it"
        )

    sessions = discover_sessions(dataset_root, dataset)
    if max_sessions is not None:
        fault_sessions = [session for session in sessions if session.is_fault][:max_sessions]
        normal_sessions = [session for session in sessions if not session.is_fault][:max_sessions]
        sessions = fault_sessions + normal_sessions

    train_samples: List[Tuple[int, dgl.DGLGraph, torch.Tensor]] = []
    test_samples: List[Tuple[int, dgl.DGLGraph, torch.Tensor]] = []
    cases: List[Dict[str, object]] = []
    audits: List[SessionAudit] = []
    global_valid_trace_ids: Set[str] = set()

    for session_idx, session in enumerate(sessions, start=1):
        print(
            f"[{dataset}] session {session_idx}/{len(sessions)} "
            f"{session.session_id} ({'fault' if session.is_fault else 'normal'})",
            flush=True,
        )
        samples, session_cases, audit = convert_session(
            session,
            spec,
            global_valid_trace_ids=global_valid_trace_ids,
            max_trace_objects=max_trace_objects,
        )
        if session.is_fault:
            test_samples.extend(samples)
            cases.extend(session_cases)
        else:
            train_samples.extend(samples)
        audits.append(audit)
        gc.collect()

    train_samples.sort(key=lambda sample: sample[0])
    test_samples.sort(key=lambda sample: sample[0])
    cases.sort(key=lambda case: (float(case["timestamp"]), str(case["cmdb_id"])))
    for case_id, case in enumerate(cases):
        case["case_id"] = case_id

    train_samples, test_samples, feature_mean, feature_std = _normalize_node_features(
        train_samples,
        test_samples,
    )
    prior_edges = _prior_call_edges(train_samples, spec.services)

    samples_dir = destination / "samples"
    graph_dir = destination / "graphs_info"
    destination.mkdir(parents=True, exist_ok=True)
    _write_pickle(samples_dir / "train_samples.pkl", train_samples)
    _write_pickle(samples_dir / "test_samples.pkl", test_samples)
    _write_pickle(graph_dir / "node_hash.pkl", spec.node_hash)
    _write_pickle(graph_dir / "call_edges.pkl", prior_edges)
    _write_pickle(graph_dir / "host_edges.pkl", [])
    pd.DataFrame(cases).to_csv(destination / "cases.csv", index=False)
    np.savez_compressed(
        destination / "normalization.npz",
        mean=feature_mean,
        std=feature_std,
        feature_names=np.array(FEATURE_NAMES),
    )

    audit_rows = [audit.__dict__ for audit in audits]
    total_edge_switches = int(sum(audit.graph_edge_switches for audit in audits))
    total_weight_switches = int(sum(audit.graph_weight_switches for audit in audits))
    total_empty_steps = int(sum(audit.empty_graph_steps for audit in audits))
    metadata = {
        "dataset": dataset,
        "source_path": str(dataset_root),
        "bucket_seconds": spec.bucket_seconds,
        "span_time_offset_seconds": spec.span_time_offset_seconds,
        "log_time_offset_seconds": spec.log_time_offset_seconds,
        "node_num": len(spec.services),
        "service_names": list(spec.services),
        "metric_names": list(METRIC_NAMES),
        "feature_dim": FEATURE_DIM,
        "feature_names": list(FEATURE_NAMES),
        "modality_dims": {
            "metric": METRIC_DIM,
            "trace": TRACE_DIM,
            "log": LOG_DIM,
        },
        "mask_feature_indices": list(MASK_FEATURE_INDICES),
        "edge_feature_dim": len(EDGE_FEATURE_NAMES),
        "edge_feature_names": list(EDGE_FEATURE_NAMES),
        "train_sample_count": len(train_samples),
        "test_sample_count": len(test_samples),
        "case_count": len(cases),
        "normal_session_count": sum(not session.is_fault for session in sessions),
        "fault_session_count": sum(session.is_fault for session in sessions),
        "training_prior_edge_count": len(prior_edges),
        "observed_graph_edge_switches": total_edge_switches,
        "observed_graph_weight_switches": total_weight_switches,
        "empty_observed_graph_steps": total_empty_steps,
        "empty_graph_policy": "preserve_empty_graph",
        "graph_semantics": "trace-derived time-varying invocation graph",
        "label_semantics": "service-level controlled fault injection target",
        "session_audit": audit_rows,
    }
    with sentinel.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw Eadro SN/TT telemetry into REACR dynamic graph samples."
    )
    parser.add_argument("--dataset", choices=["SN", "TT", "both"], default="both")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/mnt/d/Users/wyq/shiyan/Eadro/chunks"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--bucket-seconds", type=int, default=None)
    parser.add_argument("--span-time-offset-seconds", type=int, default=None)
    parser.add_argument("--log-time-offset-seconds", type=int, default=None)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--max-trace-objects", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = ("SN", "TT") if args.dataset == "both" else (args.dataset,)
    results = {}
    for dataset in datasets:
        results[dataset] = convert_dataset(
            input_root=args.input_root,
            output_root=args.output_root,
            dataset=dataset,
            bucket_seconds=args.bucket_seconds,
            span_time_offset_seconds=args.span_time_offset_seconds,
            log_time_offset_seconds=args.log_time_offset_seconds,
            max_sessions=args.max_sessions,
            max_trace_objects=args.max_trace_objects,
            overwrite=args.overwrite,
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
