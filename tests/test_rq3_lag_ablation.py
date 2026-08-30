from types import SimpleNamespace

from tools.reacr_rca_final_study_runner import VARIANTS, build_pipeline_cmd


def test_no_lag_variant_is_a_strict_zero_lag_control(tmp_path):
    variant = VARIANTS['no_lag_relation']
    assert variant['causal_max_lag'] == 0
    assert variant['ranker_mode'] == 'source_hypothesis'
    assert variant['enable_explanation_losses'] is True
    args = SimpleNamespace(
        device='cpu', feedback_sample_ratio=None, test_start_ratio=0.5,
        normal_pretrain_epochs=3, normal_auxiliary_loss_weight=0.1,
        normal_topology_change_oversample_factor=4.0,
        skip_event_chains=True,
    )
    cmd = build_pipeline_cmd(
        'SN', 'no_lag_relation', variant, 0, 1, tmp_path, args,
    )
    index = cmd.index('--causal_max_lag')
    assert cmd[index + 1] == '0'


def test_main_variant_retains_three_lags(tmp_path):
    variant = VARIANTS['source_hypothesis_losses']
    args = SimpleNamespace(
        device='cpu', feedback_sample_ratio=None, test_start_ratio=0.5,
        normal_pretrain_epochs=3, normal_auxiliary_loss_weight=0.1,
        normal_topology_change_oversample_factor=4.0,
        skip_event_chains=True,
    )
    cmd = build_pipeline_cmd(
        'SN', 'source_hypothesis_losses', variant, 0, 1, tmp_path, args,
    )
    index = cmd.index('--causal_max_lag')
    assert cmd[index + 1] == '3'
