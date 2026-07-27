"""Append Drain3 template-count features to converted Eadro dynamic samples.

Templates and event assignments must have been learned from normal sessions by the
Eadro preprocessing pipeline. The script never reads fault labels while building
features and estimates normalization statistics from normal training samples only.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.public_functions import load_init, load_samples


Sample = Tuple[int, object, torch.Tensor]


def load_template_lookup(
    parsed_dir: Path,
    file_pattern: str,
    services: Sequence[str],
    bucket_seconds: int,
    time_offset_seconds: int,
) -> Tuple[Dict[Tuple[int, str], np.ndarray], List[str], Dict[str, int]]:
    templates = json.loads((parsed_dir / "templates.json").read_text(encoding="utf-8"))
    feature_names = ["drain_unseen_count"] + [
        f"drain_template_{index + 1}_count" for index in range(len(templates))
    ]
    event_to_index = {event: index + 1 for index, event in enumerate(templates)}
    paths = sorted(glob.glob(str(parsed_dir / file_pattern)))
    if not paths:
        raise FileNotFoundError(f"no files match {parsed_dir / file_pattern}")

    frames = [
        pd.read_csv(path, usecols=["timestamp", "service", "events"])
        for path in paths
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame["service"].isin(set(services))].copy()
    frame["bucket"] = (
        ((frame["timestamp"] + time_offset_seconds) // bucket_seconds)
        * bucket_seconds
    ).astype(np.int64)
    frame["event_index"] = frame["events"].map(event_to_index).fillna(0).astype(int)

    lookup: Dict[Tuple[int, str], np.ndarray] = {}
    for (bucket, service, event_index), count in frame.groupby(
        ["bucket", "service", "event_index"]
    ).size().items():
        key = (int(bucket), str(service))
        vector = lookup.setdefault(
            key, np.zeros(len(feature_names), dtype=np.float32)
        )
        vector[int(event_index)] = math.log1p(float(count))

    audit = {
        "files": len(paths),
        "rows": int(len(frame)),
        "known_templates": len(templates),
        "feature_dim": len(feature_names),
        "unseen_rows": int((~frame["events"].isin(templates)).sum()),
    }
    return lookup, feature_names, audit


def append_features(
    samples: Iterable[Sample],
    lookup: Dict[Tuple[int, str], np.ndarray],
    services: Sequence[str],
    feature_dim: int,
) -> Tuple[List[Sample], int]:
    output = []
    nonzero_samples = 0
    for timestamp, graph, features in samples:
        extra = np.zeros((len(services), feature_dim), dtype=np.float32)
        for node_index, service in enumerate(services):
            value = lookup.get((int(timestamp), service))
            if value is not None:
                extra[node_index] = value
        nonzero_samples += int(np.any(extra))
        output.append(
            (
                timestamp,
                graph,
                torch.cat([features, torch.from_numpy(extra)], dim=-1),
            )
        )
    return output, nonzero_samples


def normalize_appended_features(
    train_samples: Sequence[Sample],
    test_samples: Sequence[Sample],
    original_dim: int,
) -> Tuple[List[Sample], List[Sample], np.ndarray, np.ndarray]:
    train_extra = torch.stack(
        [features[:, original_dim:] for _, _, features in train_samples]
    ).numpy()
    mean = train_extra.mean(axis=0)
    std = train_extra.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    mean_tensor = torch.tensor(mean, dtype=torch.float32)
    std_tensor = torch.tensor(std, dtype=torch.float32)

    def transform(samples: Sequence[Sample]) -> List[Sample]:
        result = []
        for timestamp, graph, features in samples:
            base = features[:, :original_dim]
            extra = (features[:, original_dim:] - mean_tensor) / std_tensor
            result.append((timestamp, graph, torch.cat([base, extra], dim=-1)))
        return result

    return transform(train_samples), transform(test_samples), mean, std


def write_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["SN", "TT"])
    parser.add_argument("--parsed-root", type=Path, required=True)
    parser.add_argument("--input-sample-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output-sample-dir", type=Path, required=True)
    parser.add_argument("--bucket-seconds", type=int, required=True)
    parser.add_argument("--time-offset-seconds", type=int, default=-8 * 3600)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    train_path = args.output_sample_dir / "train_samples.pkl"
    if train_path.exists() and not args.overwrite:
        raise FileExistsError(f"{train_path} exists; pass --overwrite")

    train_samples, test_samples = load_samples(str(args.input_sample_dir))
    node_hash, _, _ = load_init(str(args.graph_dir))
    services = [
        service for service, _ in sorted(node_hash.items(), key=lambda item: item[1])
    ]
    original_dim = int(train_samples[0][2].shape[-1])
    parsed_dir = args.parsed_root / args.dataset

    normal_lookup, names, normal_audit = load_template_lookup(
        parsed_dir,
        "no_fault_logs*.csv",
        services,
        args.bucket_seconds,
        args.time_offset_seconds,
    )
    fault_lookup, fault_names, fault_audit = load_template_lookup(
        parsed_dir,
        "logs*.csv",
        services,
        args.bucket_seconds,
        args.time_offset_seconds,
    )
    if names != fault_names:
        raise RuntimeError("normal and fault template vocabularies differ")

    augmented_train, train_nonzero = append_features(
        train_samples, normal_lookup, services, len(names)
    )
    augmented_test, test_nonzero = append_features(
        test_samples, fault_lookup, services, len(names)
    )
    augmented_train, augmented_test, mean, std = normalize_appended_features(
        augmented_train, augmented_test, original_dim
    )

    write_pickle(train_path, augmented_train)
    write_pickle(args.output_sample_dir / "test_samples.pkl", augmented_test)
    np.savez(
        args.output_sample_dir / "template_scaler.npz",
        mean=mean,
        std=std,
        feature_names=np.array(names),
    )
    audit = {
        "dataset": args.dataset,
        "provenance": "Drain3 templates learned from normal sessions by Eadro",
        "uses_fault_labels_for_features": False,
        "normalization_source": "normal_training_samples_only",
        "original_feature_dim": original_dim,
        "template_feature_dim": len(names),
        "output_feature_dim": original_dim + len(names),
        "train_samples": len(augmented_train),
        "test_samples": len(augmented_test),
        "train_samples_with_template_events": train_nonzero,
        "test_samples_with_template_events": test_nonzero,
        "normal_log_audit": normal_audit,
        "fault_log_audit": fault_audit,
    }
    (args.output_sample_dir / "template_feature_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
