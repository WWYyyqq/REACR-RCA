import csv
import tempfile
import unittest
from pathlib import Path

import dgl
import torch

from tools.convert_d1_weighted_dynamic import (
    EDGE_FEATURE_NAMES,
    CompactEdgeStats,
    _status_is_error,
    aggregate_trace_csv,
    build_weighted_graph,
)


class D1WeightedTraceAggregationTest(unittest.TestCase):
    def test_parent_child_join_handles_interleaved_and_child_first_rows(self):
        rows = [
            # Trace a parent, followed by an interleaved trace-b parent.
            ["60001", "frontend-0", "a-parent", "a", "1000", "http", "0", "root", ""],
            ["60002", "frontend-1", "b-parent", "b", "1000", "http", "0", "root", ""],
            # Trace c child precedes its parent; pending resolution must recover it.
            ["60003", "cartservice-0", "c-child", "c", "3000", "rpc", "500", "call", "c-parent"],
            ["60004", "productcatalogservice-0", "a-child", "a", "2000", "rpc", "0", "call", "a-parent"],
            ["60005", "recommendationservice-0", "b-child", "b", "4000", "rpc", "OK", "call", "b-parent"],
            ["60006", "checkoutservice-0", "c-parent", "c", "1000", "http", "0", "root", ""],
        ]
        header = [
            "timestamp", "cmdb_id", "span_id", "trace_id", "duration", "type",
            "status_code", "operation_name", "parent_span",
        ]
        node_hash = {
            "frontend-0": 0,
            "frontend-1": 1,
            "productcatalogservice-0": 2,
            "recommendationservice-0": 3,
            "checkoutservice-0": 4,
            "cartservice-0": 5,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                writer.writerows(rows)
            result, audit = aggregate_trace_csv(path, node_hash, target_buckets={60})

        self.assertEqual(set(result[60]), {(0, 2), (1, 3), (4, 5)})
        self.assertEqual(result[60][(0, 2)].call_count, 1)
        self.assertAlmostEqual(result[60][(0, 2)].mean_latency_ms, 2.0)
        self.assertEqual(result[60][(4, 5)].error_rate, 1.0)
        self.assertEqual(audit["resolved_parent_links"], 3)
        self.assertEqual(audit["unresolved_parent_links"], 0)

    def test_status_normalization(self):
        for value in ("", "0", "OK", "Ok", "200", "302"):
            self.assertFalse(_status_is_error(value))
        for value in ("ERROR", "500", 503, 1, 2, 4, 9, 13, 14, "true"):
            self.assertTrue(_status_is_error(value))


class D1WeightedGraphConstructionTest(unittest.TestCase):
    def test_replaces_legacy_pod_calls_and_preserves_structural_edges(self):
        node_hash = {"frontend-0": 0, "cartservice-0": 1, "node-1": 2}
        # self loops, bidirectional deployment, and one legacy pod call.
        original = dgl.graph(
            ([0, 1, 2, 0, 2, 0], [0, 1, 2, 2, 0, 1]),
            num_nodes=3,
        )
        stats = {
            (1, 0): CompactEdgeStats(
                call_count=4,
                mean_latency_ms=10.0,
                p95_latency_ms=20.0,
                error_rate=0.25,
            )
        }
        graph = build_weighted_graph(original, stats, node_hash)
        source, target = graph.edges(order="eid")
        pairs = list(zip(source.tolist(), target.tolist()))

        self.assertNotIn((0, 1), pairs)
        self.assertIn((1, 0), pairs)
        self.assertIn((0, 2), pairs)
        self.assertIn((2, 0), pairs)
        self.assertEqual(tuple(graph.edata["edge_attr"].shape), (6, len(EDGE_FEATURE_NAMES)))
        call_idx = pairs.index((1, 0))
        self.assertEqual(float(graph.edata["weight"][call_idx]), 4.0)
        self.assertEqual(graph.edata["edge_attr"][call_idx, 5:].tolist(), [1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
