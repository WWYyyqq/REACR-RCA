import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from tools.reacr_rca_final_study_runner import (
    VARIANTS,
    _node_matches_root_label,
    build_pipeline_cmd,
    collect_row,
    dataset_defaults,
    implementation_fingerprint,
    label_mapping_granularity,
    resolved_train_epochs,
    summarize_group,
)


class FinalStudyRunnerTest(unittest.TestCase):
    def test_implementation_fingerprint_is_versioned_and_stable(self):
        first = implementation_fingerprint()
        second = implementation_fingerprint()
        self.assertEqual(first, second)
        self.assertEqual(len(first['sha256']), 64)
        self.assertTrue(first['version'])

    def test_baseline_label_matching_is_separator_safe(self):
        self.assertTrue(_node_matches_root_label('adservice-0', 'adservice'))
        self.assertFalse(_node_matches_root_label('adservice2-0', 'adservice'))

    def test_label_granularity_distinguishes_d1_and_d2(self):
        self.assertEqual(
            label_mapping_granularity('D1'),
            'mixed_service_to_instance_candidate_bag',
        )
        self.assertEqual(
            label_mapping_granularity('D2'),
            'exact_single_instance_candidate',
        )

    def _args(self):
        return SimpleNamespace(
            device='cpu',
            feedback_sample_ratio=0.4,
            test_start_ratio=0.5,
            normal_pretrain_epochs=3,
            normal_auxiliary_loss_weight=0.1,
            normal_topology_change_oversample_factor=4.0,
            skip_event_chains=True,
        )

    def test_d1_official_recipe_uses_corrected_modality_layout(self):
        defaults = dataset_defaults('D1')
        self.assertTrue(defaults['enable_modality_gate'])
        self.assertEqual(defaults['modality_gate_fusion_mode'], 'raw_residual')

        with tempfile.TemporaryDirectory() as tmp_dir:
            cmd = build_pipeline_cmd(
                'D1',
                'source_hypothesis_losses',
                VARIANTS['source_hypothesis_losses'],
                seed=0,
                epochs=1,
                out_dir=Path(tmp_dir),
                args=self._args(),
                smoke=True,
            )
        self.assertIn('--enable_modality_gate', cmd)
        fusion_idx = cmd.index('--modality_gate_fusion_mode')
        self.assertEqual(cmd[fusion_idx + 1], 'raw_residual')

    def test_dynamic_datasets_use_group_split_and_configured_ratio(self):
        args = self._args()
        args.feedback_sample_ratio = None
        with tempfile.TemporaryDirectory() as tmp_dir:
            for dataset in ('SN', 'TT'):
                cmd = build_pipeline_cmd(
                    dataset,
                    'source_hypothesis_losses',
                    VARIANTS['source_hypothesis_losses'],
                    seed=0,
                    epochs=1,
                    out_dir=Path(tmp_dir) / dataset,
                    args=args,
                    smoke=True,
                )
                split_idx = cmd.index('--case_split_protocol')
                ratio_idx = cmd.index('--feedback_sample_ratio')
                self.assertEqual(cmd[split_idx + 1], 'group')
                self.assertEqual(float(cmd[ratio_idx + 1]), 0.3)

    def test_dataset_specific_default_training_budget(self):
        self.assertEqual(resolved_train_epochs('SN', None), 20)
        self.assertEqual(resolved_train_epochs('TT', None), 25)
        self.assertEqual(resolved_train_epochs('TT', 9), 9)

    def test_topology_ablation_modes_are_explicit(self):
        self.assertEqual(
            VARIANTS['no_topology_change_features']['topology_feature_mode'],
            'static_only',
        )
        self.assertEqual(
            VARIANTS['topology_dynamic_features_only']['topology_feature_mode'],
            'dynamic_only',
        )
        self.assertEqual(
            VARIANTS['frozen_window_topology']['graph_sequence_mode'],
            'freeze_first',
        )
        self.assertFalse(
            VARIANTS['no_explicit_topology_encoder']['enable_topology_dynamics']
        )

    def test_external_result_paths_and_topology_strata_are_summarized(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_path = Path(tmp_dir)
            final_eval = {
                'metrics': {'A@1': 0.5, 'A@3': 0.75, 'A@5': 1.0},
                'score_std': 0.2,
                'topology_strata': {
                    'changed': {
                        'sample_count': 2,
                        'metrics': {'A@1': 0.5, 'A@3': 1.0, 'A@5': 1.0},
                    },
                    'stable': {
                        'sample_count': 4,
                        'metrics': {'A@1': 0.5, 'A@3': 0.5, 'A@5': 1.0},
                    },
                },
            }
            (run_path / 'final_eval.json').write_text(json.dumps(final_eval))
            (run_path / 'validation_eval.json').write_text(
                json.dumps({'metrics': {'A@1': 0.25, 'A@3': 0.5, 'A@5': 0.75}})
            )
            (run_path / 'run_config.json').write_text(
                json.dumps({'seed': 0, 'case_split_protocol': 'time_strict'})
            )

            row = collect_row('D1', 'source_hypothesis_losses', run_path)
            summary = summarize_group(pd.DataFrame([row]))

        self.assertEqual(row['run_dir'], str(run_path.resolve()))
        self.assertEqual(summary.loc[0, 'changed_case_count'], 2)
        self.assertAlmostEqual(summary.loc[0, 'changed_A3_mean'], 1.0)


if __name__ == '__main__':
    unittest.main()
