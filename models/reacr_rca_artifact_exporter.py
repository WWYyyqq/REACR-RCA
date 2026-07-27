"""Export REACR-RCA's aligned inputs and inference intermediates by batch.

The layout mirrors Eadro's reproducible "aligned chunks" idea, but preserves
REACR-RCA's windowed graph tensors and ranking evidence.  Batches are written
independently so a full experiment never accumulates all case tensors in RAM.
"""

import json
import os
from typing import Dict

import numpy as np
import pandas as pd
import torch

from models.reacr_rca_trainer import _move_graphs_to_device, _safe_device


def _numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def export_intermediate_artifacts(
    model,
    dataloader,
    device: torch.device,
    index_to_node: Dict[int, str],
    output_dir: str,
) -> Dict[str, object]:
    """Write one compressed NPZ per batch and a node-level case index CSV."""
    device = _safe_device(device)
    os.makedirs(output_dir, exist_ok=True)
    batch_dir = os.path.join(output_dir, 'batches')
    os.makedirs(batch_dir, exist_ok=True)

    model.eval()
    rows = []
    batch_files = []
    module_shapes = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            features = batch['features'].float().to(device)
            graphs = _move_graphs_to_device(batch['graphs'], device)
            outputs = model(graphs, features)

            gate_output = outputs.get('modality_gate_output') or {}
            backbone_output = outputs['backbone_output']
            anomaly_output = outputs['anomaly_output']
            causal_output = outputs['causal_output']
            rank_output = outputs['rank_output']
            payload = {
                'case_ids': _numpy(batch['case_ids']),
                'case_timestamps': _numpy(batch['case_timestamps']),
                'window_timestamps': _numpy(batch['timestamps']),
                'labels': _numpy(batch['labels']),
                'input_features': _numpy(features),
                'fused_features': _numpy(gate_output.get('fused_features')),
                'temporal_states': _numpy(backbone_output.get('temporal_states')),
                'topology_raw_features': _numpy(
                    (backbone_output.get('topology_output') or {}).get('raw_features')
                ),
                'anomaly_score': _numpy(anomaly_output.get('anomaly_score')),
                'reconstruction_prediction': _numpy(anomaly_output.get('reconstruction_prediction')),
                'next_step_forecast': _numpy(anomaly_output.get('next_step_forecast')),
                'causal_matrix': _numpy(causal_output.get('causal_matrix')),
                'null_source_probability': _numpy(causal_output.get('null_source_probability')),
                'lag_topology_mask': _numpy(causal_output.get('lag_topology_mask')),
                'lag_alignment_topology_mask': _numpy(
                    causal_output.get('lag_alignment_topology_mask')
                ),
                'lag_weights': _numpy(causal_output.get('lag_weights')),
                'final_score': _numpy(rank_output.get('final_score')),
                'source_role_score': _numpy(rank_output.get('source_role_score')),
                'response_role_score': _numpy(rank_output.get('response_role_score')),
                'role_margin': _numpy(rank_output.get('role_margin')),
                'root_role_score': _numpy(rank_output.get('root_role_score')),
                'symptom_role_score': _numpy(rank_output.get('symptom_role_score')),
                'source_score': _numpy(rank_output.get('source_score')),
                'victim_score': _numpy(rank_output.get('victim_score')),
                'incoming_explainedness': _numpy(rank_output.get('incoming_explainedness')),
                'counterfactual_coverage': _numpy(rank_output.get('counterfactual_coverage')),
                'counterfactual_residual': _numpy(rank_output.get('counterfactual_residual')),
                'source_hypothesis_coverage': _numpy(rank_output.get('source_hypothesis_coverage')),
                'source_hypothesis_residual': _numpy(rank_output.get('source_hypothesis_residual')),
            }
            compact_payload = {key: value for key, value in payload.items() if value is not None}
            batch_path = os.path.join(batch_dir, f'batch_{batch_idx:04d}.npz')
            np.savez_compressed(batch_path, **compact_payload)
            batch_files.append(os.path.relpath(batch_path, output_dir))

            if module_shapes is None:
                module_shapes = {key: list(value.shape) for key, value in compact_payload.items()}

            score = rank_output['final_score'].detach().cpu()
            anomaly = anomaly_output['anomaly_score'].detach().cpu()
            labels = batch['labels']
            labels = labels.cpu() if labels is not None else None
            for sample_idx in range(score.shape[0]):
                for node_idx in range(score.shape[1]):
                    is_label_candidate = int(
                        labels is not None
                        and labels[sample_idx, node_idx].item() > 0
                    )
                    rows.append({
                        'case_id': int(batch['case_ids'][sample_idx].item()),
                        'case_timestamp': int(batch['case_timestamps'][sample_idx].item()),
                        'node_index': node_idx,
                        'node': index_to_node.get(node_idx, str(node_idx)),
                        'is_label_candidate': is_label_candidate,
                        # Deprecated alias retained for historical readers.
                        'is_root_label': is_label_candidate,
                        'anomaly_peak': float(anomaly[sample_idx, :, node_idx].max().item()),
                        'final_score': float(score[sample_idx, node_idx].item()),
                        'source_role_score': float(rank_output['source_role_score'][sample_idx, node_idx].detach().cpu().item()),
                        'response_role_score': float(rank_output['response_role_score'][sample_idx, node_idx].detach().cpu().item()),
                        'source_hypothesis_coverage': float(rank_output['source_hypothesis_coverage'][sample_idx, node_idx].detach().cpu().item()),
                        'source_hypothesis_residual': float(rank_output['source_hypothesis_residual'][sample_idx, node_idx].detach().cpu().item()),
                        # Deprecated aliases retained for older analysis scripts.
                        'root_role_score': float(rank_output['root_role_score'][sample_idx, node_idx].detach().cpu().item()),
                        'symptom_role_score': float(rank_output['symptom_role_score'][sample_idx, node_idx].detach().cpu().item()),
                        'source_score': float(rank_output['source_score'][sample_idx, node_idx].detach().cpu().item()),
                        'victim_score': float(rank_output['victim_score'][sample_idx, node_idx].detach().cpu().item()),
                        'incoming_explainedness': float(rank_output['incoming_explainedness'][sample_idx, node_idx].detach().cpu().item()),
                        'counterfactual_coverage': float(rank_output['counterfactual_coverage'][sample_idx, node_idx].detach().cpu().item()),
                        'counterfactual_residual': float(rank_output['counterfactual_residual'][sample_idx, node_idx].detach().cpu().item()),
                    })

    index_path = os.path.join(output_dir, 'case_node_evidence.csv')
    pd.DataFrame(rows).to_csv(index_path, index=False)
    manifest = {
        'format': 'REACR-RCA intermediate artifacts v1',
        'purpose': 'Eadro-style aligned, inspectable inputs and module outputs',
        'batch_count': len(batch_files),
        'batch_files': batch_files,
        'module_shapes': module_shapes or {},
        'case_node_evidence': os.path.basename(index_path),
    }
    manifest_path = os.path.join(output_dir, 'artifact_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
    return {
        'artifact_dir': output_dir,
        'artifact_manifest_path': manifest_path,
        'case_node_evidence_path': index_path,
        'batch_count': len(batch_files),
    }
