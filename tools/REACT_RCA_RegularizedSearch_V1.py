import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "res" / "react_rca_regularized_search"
LOG_DIR = BASE / "logs"


def common_args(run_dir: Path):
    return [
        sys.executable,
        "-u",
        "REACT_RCA_Pipeline_V1.py",
        "--device",
        "cuda",
        "--case_split_protocol",
        "fair",
        "--feedback_sample_ratio",
        "0.3",
        "--train_epochs",
        "20",
        "--enable_change_point_branch",
        "--export_event_chains",
        "--output_dir",
        str(run_dir),
    ]


def configs():
    d1_common = [
        "--dataset",
        "D1",
        "--seed",
        "1",
        "--batch_size",
        "8",
        "--change_point_weight",
        "1.0",
    ]
    d2_common = [
        "--dataset",
        "D2",
        "--seed",
        "3",
        "--batch_size",
        "4",
        "--change_point_weight",
        "0.05",
        "--causal_hidden_dim",
        "16",
        "--causal_gamma",
        "0.2",
        "--causal_kernel_size",
        "3",
        "--enable_modality_gate",
        "--modality_gate_fusion_mode",
        "raw_residual",
        "--modality_gate_residual_alpha",
        "0.1",
    ]
    return [
        {
            "name": "D1_h64_do02_wd1e4",
            "args": d1_common
            + [
                "--dropout",
                "0.2",
                "--train_weight_decay",
                "0.0001",
                "--train_patience",
                "6",
            ],
        },
        {
            "name": "D1_h48_do02_wd1e4",
            "args": d1_common
            + [
                "--spatial_hidden_dim",
                "48",
                "--temporal_hidden_dim",
                "48",
                "--anomaly_hidden_dim",
                "48",
                "--causal_hidden_dim",
                "32",
                "--rank_hidden_dim",
                "48",
                "--dropout",
                "0.2",
                "--train_weight_decay",
                "0.0001",
                "--train_patience",
                "6",
            ],
        },
        {
            "name": "D1_h32_do03_wd5e4",
            "args": d1_common
            + [
                "--spatial_hidden_dim",
                "32",
                "--temporal_hidden_dim",
                "32",
                "--anomaly_hidden_dim",
                "32",
                "--causal_hidden_dim",
                "32",
                "--rank_hidden_dim",
                "32",
                "--dropout",
                "0.3",
                "--train_weight_decay",
                "0.0005",
                "--train_patience",
                "6",
            ],
        },
        {
            "name": "D2_h64_do02_wd1e4_b005",
            "args": d2_common
            + [
                "--dropout",
                "0.2",
                "--branch_init_scale",
                "0.05",
                "--train_weight_decay",
                "0.0001",
                "--train_patience",
                "6",
            ],
        },
        {
            "name": "D2_h48_do02_wd1e4_b005",
            "args": d2_common
            + [
                "--spatial_hidden_dim",
                "48",
                "--temporal_hidden_dim",
                "48",
                "--anomaly_hidden_dim",
                "48",
                "--rank_hidden_dim",
                "48",
                "--modality_gate_hidden_dim",
                "48",
                "--dropout",
                "0.2",
                "--branch_init_scale",
                "0.05",
                "--train_weight_decay",
                "0.0001",
                "--train_patience",
                "6",
            ],
        },
        {
            "name": "D2_h32_do03_wd5e4_b005",
            "args": d2_common
            + [
                "--spatial_hidden_dim",
                "32",
                "--temporal_hidden_dim",
                "32",
                "--anomaly_hidden_dim",
                "32",
                "--rank_hidden_dim",
                "32",
                "--modality_gate_hidden_dim",
                "32",
                "--dropout",
                "0.3",
                "--branch_init_scale",
                "0.05",
                "--train_weight_decay",
                "0.0005",
                "--train_patience",
                "6",
            ],
        },
    ]


def run_one(conf):
    run_dir = BASE / conf["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{conf['name']}.stdout.log"
    cmd = common_args(run_dir) + conf["args"]
    print("RUN", conf["name"])
    print(" ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"{conf['name']} failed with exit code {ret}")


def summarize_one(name):
    run_dir = BASE / name
    summary = json.loads((run_dir / "stage1_dsbg_summary.json").read_text(encoding="utf-8"))
    final_eval = json.loads((run_dir / "final_eval.json").read_text(encoding="utf-8"))
    val_eval = json.loads((run_dir / "validation_eval.json").read_text(encoding="utf-8"))
    history = json.loads((run_dir / "train_history.json").read_text(encoding="utf-8"))
    rows = []
    for item in history:
        train = item.get("train_metrics", {})
        val = item.get("val_metrics", {})
        rows.append(
            {
                "epoch": item["epoch"],
                "train_loss": item["train_loss"],
                "train_A@1": train.get("A@1", 0.0),
                "train_A@3": train.get("A@3", 0.0),
                "train_A@5": train.get("A@5", 0.0),
                "val_loss": item["val_loss"],
                "val_A@1": val.get("A@1", 0.0),
                "val_A@3": val.get("A@3", 0.0),
                "val_A@5": val.get("A@5", 0.0),
            }
        )
    hist_df = pd.DataFrame(rows)
    hist_df["gap_A@1"] = hist_df["train_A@1"] - hist_df["val_A@1"]
    hist_df["loss_gap"] = hist_df["val_loss"] - hist_df["train_loss"]
    hist_df.to_csv(run_dir / "train_diagnostics.csv", index=False)
    best_epoch = summary["training_summary"]["train_meta"]["best_epoch"]
    best_row = hist_df[hist_df["epoch"] == best_epoch].iloc[0]
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    event = summary["training_summary"].get("event_chain_metrics") or {}
    return {
        "name": name,
        "dataset": summary["dataset"],
        "seed": summary["seed"],
        "batch_size": summary["batch_size"],
        "dropout": config.get("dropout"),
        "spatial_hidden_dim": config.get("spatial_hidden_dim"),
        "temporal_hidden_dim": config.get("temporal_hidden_dim"),
        "anomaly_hidden_dim": config.get("anomaly_hidden_dim"),
        "rank_hidden_dim": config.get("rank_hidden_dim"),
        "causal_hidden_dim": config.get("causal_hidden_dim"),
        "modality_gate_hidden_dim": config.get("modality_gate_hidden_dim"),
        "branch_init_scale": config.get("branch_init_scale"),
        "weight_decay": config.get("train_weight_decay"),
        "patience": config.get("train_patience"),
        "trained_epochs": summary["training_summary"]["train_meta"]["trained_epochs"],
        "best_epoch": best_epoch,
        "best_train_A@1": best_row["train_A@1"],
        "best_val_A@1": val_eval["metrics"]["A@1"],
        "best_gap_A@1": best_row["gap_A@1"],
        "best_train_loss": best_row["train_loss"],
        "best_val_loss": val_eval["loss"],
        "best_loss_gap": best_row["loss_gap"],
        "test_A@1": final_eval["metrics"]["A@1"],
        "test_A@3": final_eval["metrics"]["A@3"],
        "test_A@5": final_eval["metrics"]["A@5"],
        "test_loss": final_eval["loss"],
        "strict_chain_coverage": event.get("strict_chain_coverage"),
        "label_root_hit_at_topk": event.get("label_root_hit_at_topk"),
    }


def summarize_all():
    rows = [summarize_one(conf["name"]) for conf in configs()]
    df = pd.DataFrame(rows)
    df.to_csv(BASE / "regularized_search_summary.csv", index=False)
    (BASE / "regularized_search_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = ["# REACT-RCA Regularized Search Summary", ""]
    report.append(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    report.append("")
    for dataset, group in df.groupby("dataset"):
        best = group.sort_values(["test_A@1", "test_A@3", "test_A@5", "best_gap_A@1"], ascending=[False, False, False, True]).iloc[0]
        report.append(f"## Best {dataset}")
        report.append(best.to_string())
        report.append("")
    (BASE / "regularized_search_summary.md").write_text("\n".join(report), encoding="utf-8")
    print(df.to_string(index=False))


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for conf in configs():
        run_one(conf)
    summarize_all()


if __name__ == "__main__":
    main()
