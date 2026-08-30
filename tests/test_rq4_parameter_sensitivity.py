from pathlib import Path

from tools.run_rq4_parameter_sensitivity import (
    PARAMETERS,
    build_sensitivity_cmd,
    reusable_rq2_dir,
    set_cmd_option,
    value_slug,
)


class Args:
    device = "cpu"
    train_epochs = 1
    feedback_sample_ratio = None
    test_start_ratio = 0.5
    normal_pretrain_epochs = 0
    normal_auxiliary_loss_weight = 0.0
    normal_topology_change_oversample_factor = 1.0
    skip_event_chains = True


def test_set_cmd_option_replaces_or_appends():
    assert set_cmd_option(["python", "x", "--a", "1"], "--a", 2)[-2:] == ["--a", "2"]
    assert set_cmd_option(["python", "x"], "--a", 2)[-2:] == ["--a", "2"]


def test_sensitivity_command_is_validation_only(tmp_path: Path):
    cmd = build_sensitivity_cmd("window_size", 14, 0, tmp_path, Args())
    assert "--validation_only" in cmd
    assert cmd[cmd.index("--window_size") + 1] == "14"
    assert "--output_dir" in cmd


def test_declared_defaults_and_slugs():
    assert PARAMETERS["max_lag"]["default"] == 3
    assert PARAMETERS["window_size"]["default"] == 10
    assert value_slug(0.25) == "0p25"


def test_nondefault_unrelated_setting_is_not_reused():
    assert reusable_rq2_dir("window_size", 8, 0) is None
