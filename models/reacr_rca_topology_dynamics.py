"""Topology-dynamics features for DSBG graph sequences.

The extractor in this module never creates edges or completes a graph from a
catalogue.  Every value is computed from the edges that are present in the DGL
snapshots passed by the caller.

Feature order (last tensor dimension):

0. in_degree
1. out_degree
2. delta_in_degree
3. delta_out_degree
4. added_in_edges
5. removed_in_edges
6. added_out_edges
7. removed_out_edges

The first two channels describe the current snapshot.  The remaining six
channels compare the current snapshot with the immediately preceding snapshot
from the same sample.  They are exactly zero at the first step and remain
exactly zero when the topology is fixed.
"""

from collections import Counter
from typing import Counter as CounterType
from typing import Dict, List, Optional, Sequence, Tuple, Union

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


Edge = Tuple[int, int]
EdgeCounter = CounterType[Edge]

TOPOLOGY_FEATURE_NAMES: Tuple[str, ...] = (
    'in_degree',
    'out_degree',
    'delta_in_degree',
    'delta_out_degree',
    'added_in_edges',
    'removed_in_edges',
    'added_out_edges',
    'removed_out_edges',
)
TOPOLOGY_STATIC_FEATURE_DIM = 2
TOPOLOGY_DYNAMIC_FEATURE_DIM = 6
TOPOLOGY_FEATURE_DIM = len(TOPOLOGY_FEATURE_NAMES)


def _validate_homogeneous_graph(graph: dgl.DGLGraph) -> None:
    if not graph.is_homogeneous:
        raise ValueError(
            'topology dynamics currently require homogeneous DGL graphs; '
            'convert each relation type explicitly before extraction'
        )


def canonical_edge_multiset(graph: dgl.DGLGraph) -> Tuple[Edge, ...]:
    """Return the exact directed edge multiset in deterministic order.

    Parallel edges are retained as repeated ``(source, destination)`` pairs.
    No self-loop or other edge is added by this function.
    """

    _validate_homogeneous_graph(graph)
    sources, destinations = graph.edges(order='eid')
    edges = zip(
        sources.detach().cpu().tolist(),
        destinations.detach().cpu().tolist(),
    )
    return tuple(sorted((int(source), int(destination)) for source, destination in edges))


def edge_multiset_counter(graph: dgl.DGLGraph) -> EdgeCounter:
    """Count the exact directed edges present in ``graph``."""

    return Counter(canonical_edge_multiset(graph))


def _unbatch_steps(
    graphs: Sequence[dgl.DGLGraph],
) -> Tuple[List[List[dgl.DGLGraph]], int, int]:
    if len(graphs) == 0:
        raise ValueError('graphs must contain at least one temporal snapshot')

    unbatched_steps: List[List[dgl.DGLGraph]] = []
    expected_batch_size: Optional[int] = None
    expected_node_num: Optional[int] = None

    for step_idx, graph in enumerate(graphs):
        _validate_homogeneous_graph(graph)
        sample_graphs = list(dgl.unbatch(graph))
        if expected_batch_size is None:
            expected_batch_size = len(sample_graphs)
        elif len(sample_graphs) != expected_batch_size:
            raise ValueError(
                f'inconsistent graph batch size at step {step_idx}: '
                f'expected {expected_batch_size}, got {len(sample_graphs)}'
            )

        for sample_idx, sample_graph in enumerate(sample_graphs):
            node_num = int(sample_graph.num_nodes())
            if expected_node_num is None:
                expected_node_num = node_num
            elif node_num != expected_node_num:
                raise ValueError(
                    'all graph snapshots must use the same node count and node '
                    f'ordering; expected {expected_node_num}, got {node_num} '
                    f'at step {step_idx}, sample {sample_idx}'
                )
        unbatched_steps.append(sample_graphs)

    assert expected_batch_size is not None
    assert expected_node_num is not None
    return unbatched_steps, expected_batch_size, expected_node_num


def _degrees_from_edges(edges: EdgeCounter, node_num: int) -> Tuple[List[int], List[int]]:
    in_degree = [0] * node_num
    out_degree = [0] * node_num
    for (source, destination), multiplicity in edges.items():
        if not (0 <= source < node_num and 0 <= destination < node_num):
            raise ValueError(
                f'edge ({source}, {destination}) is outside node range [0, {node_num})'
            )
        out_degree[source] += int(multiplicity)
        in_degree[destination] += int(multiplicity)
    return in_degree, out_degree


def _accumulate_edge_changes(
    feature_row: torch.Tensor,
    added_edges: EdgeCounter,
    removed_edges: EdgeCounter,
) -> None:
    for (source, destination), multiplicity in added_edges.items():
        count = float(multiplicity)
        feature_row[destination, 4] += count
        feature_row[source, 6] += count
    for (source, destination), multiplicity in removed_edges.items():
        count = float(multiplicity)
        feature_row[destination, 5] += count
        feature_row[source, 7] += count


def extract_topology_dynamics(
    graphs: Sequence[dgl.DGLGraph],
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Extract exact node-level topology features from a graph sequence.

    Args:
        graphs: Temporal list of DGL graphs.  Each element may be an unbatched
            graph or a DGL batch.  Batch membership must be consistent over
            time, and every sample must keep the same node count and ordering.
        device: Output device.  By default, use the first graph's device.
        dtype: Floating output dtype.

    Returns:
        Tensor with shape ``[B, W, N, 8]``.

    Notes:
        Edge differences are multiset differences.  Consequently parallel-edge
        additions/removals are counted exactly, and degree deltas agree with
        the added/removed edge counts.
    """

    unbatched_steps, batch_size, node_num = _unbatch_steps(graphs)
    window_size = len(unbatched_steps)
    output_device = torch.device(device) if device is not None else graphs[0].device

    # Build the non-differentiable structural observations on CPU, then move
    # once to the requested model device.
    features = torch.zeros(
        batch_size,
        window_size,
        node_num,
        TOPOLOGY_FEATURE_DIM,
        dtype=dtype,
        device='cpu',
    )

    for sample_idx in range(batch_size):
        previous_edges: Optional[EdgeCounter] = None
        previous_in_degree: Optional[List[int]] = None
        previous_out_degree: Optional[List[int]] = None

        for step_idx in range(window_size):
            current_edges = edge_multiset_counter(unbatched_steps[step_idx][sample_idx])
            in_degree, out_degree = _degrees_from_edges(current_edges, node_num)
            feature_row = features[sample_idx, step_idx]
            feature_row[:, 0] = torch.tensor(in_degree, dtype=dtype)
            feature_row[:, 1] = torch.tensor(out_degree, dtype=dtype)

            if previous_edges is not None:
                assert previous_in_degree is not None
                assert previous_out_degree is not None
                feature_row[:, 2] = torch.tensor(
                    [
                        current_degree - previous_degree
                        for current_degree, previous_degree in zip(in_degree, previous_in_degree)
                    ],
                    dtype=dtype,
                )
                feature_row[:, 3] = torch.tensor(
                    [
                        current_degree - previous_degree
                        for current_degree, previous_degree in zip(out_degree, previous_out_degree)
                    ],
                    dtype=dtype,
                )
                _accumulate_edge_changes(
                    feature_row,
                    current_edges - previous_edges,
                    previous_edges - current_edges,
                )

            previous_edges = current_edges
            previous_in_degree = in_degree
            previous_out_degree = out_degree

    return features.to(output_device)


class TopologyDynamicsEncoder(nn.Module):
    """Project raw topology observations into node features for a backbone.

    Static degree features and dynamic change features use separate
    projections.  The dynamic projection has no bias, so a fixed topology
    produces an exactly zero ``dynamic_embedding``.

    A caller can concatenate the returned encoded tensor with observations:

    ``model_features = torch.cat([features, encoder(graphs)], dim=-1)``.
    """

    def __init__(
        self,
        output_dim: int,
        dropout: float = 0.0,
        feature_mode: str = 'full',
    ):
        super().__init__()
        if output_dim <= 0:
            raise ValueError(f'output_dim must be positive, got {output_dim}')
        if feature_mode not in {'full', 'static_only', 'dynamic_only'}:
            raise ValueError(f'unsupported topology feature_mode: {feature_mode}')
        self.output_dim = int(output_dim)
        self.feature_mode = feature_mode
        self.static_projection = nn.Linear(
            TOPOLOGY_STATIC_FEATURE_DIM,
            self.output_dim,
            bias=True,
        )
        self.dynamic_projection = nn.Linear(
            TOPOLOGY_DYNAMIC_FEATURE_DIM,
            self.output_dim,
            bias=False,
        )
        self.dropout = nn.Dropout(dropout)

    def extract_raw(
        self,
        graphs: Sequence[dgl.DGLGraph],
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        parameter = self.static_projection.weight
        output_device = parameter.device if device is None else device
        output_dtype = parameter.dtype if dtype is None else dtype
        return extract_topology_dynamics(
            graphs,
            device=output_device,
            dtype=output_dtype,
        )

    def forward(
        self,
        graphs: Sequence[dgl.DGLGraph],
        return_details: bool = False,
    ):
        raw_features = self.extract_raw(graphs)
        static_features = raw_features[..., :TOPOLOGY_STATIC_FEATURE_DIM]
        dynamic_features = raw_features[..., TOPOLOGY_STATIC_FEATURE_DIM:]
        static_embedding = F.gelu(self.static_projection(static_features))
        dynamic_embedding = F.gelu(self.dynamic_projection(dynamic_features))
        if self.feature_mode == 'static_only':
            encoded_features = self.dropout(static_embedding)
        elif self.feature_mode == 'dynamic_only':
            encoded_features = self.dropout(dynamic_embedding)
        else:
            encoded_features = self.dropout(static_embedding + dynamic_embedding)

        if not return_details:
            return encoded_features
        return {
            'encoded_features': encoded_features,
            'raw_features': raw_features,
            'static_features': static_features,
            'dynamic_features': dynamic_features,
            'static_embedding': static_embedding,
            'dynamic_embedding': dynamic_embedding,
            'feature_mode': self.feature_mode,
            'feature_names': TOPOLOGY_FEATURE_NAMES,
        }


def summarize_topology_features(features: torch.Tensor) -> Dict[str, object]:
    if features.ndim != 4 or features.shape[-1] != TOPOLOGY_FEATURE_DIM:
        raise ValueError(
            f'expected topology features [B, W, N, {TOPOLOGY_FEATURE_DIM}], '
            f'got {list(features.shape)}'
        )
    dynamic = features[..., TOPOLOGY_STATIC_FEATURE_DIM:]
    return {
        'shape': list(features.shape),
        'feature_names': list(TOPOLOGY_FEATURE_NAMES),
        'dynamic_nonzero_count': int(torch.count_nonzero(dynamic).item()),
        'changed_sample_steps': int(
            torch.any(dynamic != 0, dim=(-1, -2)).sum().item()
        ),
    }
