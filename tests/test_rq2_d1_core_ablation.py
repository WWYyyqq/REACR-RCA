from types import SimpleNamespace

from tools.run_rq2_d1_core_ablation import DEFAULT_VARIANTS, runner_args


def test_rq2_variants_cover_core_components():
    assert DEFAULT_VARIANTS == (
        "source_hypothesis_losses",
        "frozen_window_topology",
        "no_topology_change_features",
        "no_lag_relation",
        "no_root_symptom_loss",
        "no_source_hypothesis_loss",
        "no_normal_dynamics_training",
    )


def test_rq2_runner_uses_locked_common_protocol():
    args = SimpleNamespace(device="cuda", overwrite=False)
    result = runner_args(args)
    assert result.feedback_sample_ratio is None
    assert result.test_start_ratio == 0.5
    assert result.normal_pretrain_epochs == 3
    assert result.normal_auxiliary_loss_weight == 0.1
    assert result.normal_topology_change_oversample_factor == 4.0
    assert result.skip_event_chains is True
