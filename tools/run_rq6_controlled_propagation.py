"""Controlled benchmark for lagged-relation and source-response recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from models.reacr_rca_causal_head import RunInspiredCausalHead
from models.reacr_rca_causal_ranker import CausalPropagationRanker


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ControlledCase:
    anomaly: torch.Tensor
    graphs: list
    source: int
    response_mask: torch.Tensor
    relation: torch.Tensor
    lag: torch.Tensor


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_case(rng, node_num=8, window_size=10, max_lag=3, noise=0.15):
    source = int(rng.randint(node_num))
    available = [node for node in range(node_num) if node != source]
    rng.shuffle(available)
    path_length = int(rng.randint(2, 5))
    path = [source] + available[:path_length]
    lags = [int(rng.randint(1, max_lag + 1)) for _ in range(path_length)]
    onset = [int(rng.randint(1, 3))]
    for lag in lags:
        onset.append(onset[-1] + lag)
    if onset[-1] >= window_size - 1:
        shift = onset[-1] - (window_size - 2)
        onset = [max(0, value - shift) for value in onset]

    anomaly = torch.as_tensor(
        rng.normal(0.0, noise, size=(window_size, node_num)), dtype=torch.float32
    ).abs()
    relation = torch.zeros(node_num, node_num, dtype=torch.float32)
    lag_truth = torch.zeros(node_num, node_num, dtype=torch.long)
    response_mask = torch.zeros(node_num, dtype=torch.float32)
    for index, node in enumerate(path):
        amplitude = 2.0 + 0.55 * index + float(rng.uniform(0.0, 0.25))
        start = onset[index]
        anomaly[start, node] += amplitude
        if start + 1 < window_size:
            anomaly[start + 1, node] += amplitude * 0.65
        if index > 0:
            parent = path[index - 1]
            relation[node, parent] = 1.0
            lag_truth[node, parent] = lags[index - 1]
            response_mask[node] = 1.0

    true_edges = {(path[index - 1], path[index]) for index in range(1, len(path))}
    graph_sequence = []
    for step in range(window_size):
        edges = set(true_edges)
        for _ in range(int(rng.randint(2, 6))):
            left = int(rng.randint(node_num))
            right = int(rng.randint(node_num))
            if left != right:
                edges.add((left, right))
        # Dynamic distractor churn; propagation edges remain observable.
        if step % 2 == 1 and len(edges) > len(true_edges):
            distractors = sorted(edges - true_edges)
            edges.discard(distractors[int(rng.randint(len(distractors)))])
        src = torch.tensor([edge[0] for edge in sorted(edges)], dtype=torch.int64)
        dst = torch.tensor([edge[1] for edge in sorted(edges)], dtype=torch.int64)
        graph = dgl.graph((src, dst), num_nodes=node_num)
        graph.edata["weight"] = torch.ones(graph.num_edges())
        graph_sequence.append(graph)
    return ControlledCase(anomaly, graph_sequence, source, response_mask, relation, lag_truth)


def make_dataset(seed, count, **kwargs):
    rng = np.random.RandomState(seed)
    return [make_case(rng, **kwargs) for _ in range(count)]


def collate(cases, device):
    window_size = cases[0].anomaly.shape[0]
    graphs = [
        dgl.batch([case.graphs[step] for case in cases]).to(device)
        for step in range(window_size)
    ]
    return {
        "anomaly": torch.stack([case.anomaly for case in cases]).to(device),
        "graphs": graphs,
        "source": torch.tensor([case.source for case in cases], dtype=torch.long, device=device),
        "response": torch.stack([case.response_mask for case in cases]).to(device),
        "relation": torch.stack([case.relation for case in cases]).to(device),
        "lag": torch.stack([case.lag for case in cases]).to(device),
    }


class ControlledModel(nn.Module):
    def __init__(self, node_num, window_size, max_lag, ranker_mode):
        super().__init__()
        self.max_lag = max_lag
        self.relation = RunInspiredCausalHead(
            node_num=node_num,
            window_size=window_size,
            hidden_dim=32,
            max_lag=max_lag,
            kernel_size=3,
            gamma=0.5,
            topology_mode="directed",
            topology_weight_gamma=0.25,
        )
        self.ranker = CausalPropagationRanker(
            window_size=window_size,
            hidden_dim=32,
            branch_hidden_dim=32,
            dropout=0.0,
            anchor_init_scale=0.1,
            branch_init_scale=0.1,
            ranker_mode=ranker_mode,
        )

    def forward(self, batch):
        relation = self.relation(batch["anomaly"], batch["graphs"])
        rank = self.ranker(
            batch["graphs"],
            batch["anomaly"],
            relation["directed_relation_matrix"],
            lag_weights=relation["lag_weights"],
        )
        return relation, rank


def controlled_loss(batch, relation, rank, *, use_relation=True, use_role=True, args=None):
    final_weight = 1.0 if args is None else float(args.final_loss_weight)
    relation_weight = 0.5 if args is None else float(args.relation_loss_weight)
    lag_weight = 0.5 if args is None else float(args.lag_loss_weight)
    source_role_weight = 0.25 if args is None else float(args.source_role_loss_weight)
    response_role_weight = 0.25 if args is None else float(args.response_role_loss_weight)
    loss = final_weight * F.cross_entropy(rank["final_score"], batch["source"])
    true_edge = batch["relation"] > 0
    if use_relation:
        # Each target has one parent or the explicit null source. Training the
        # complete categorical distribution penalises false parents as well as
        # rewarding the true propagation edge.
        parent = batch["relation"].argmax(dim=-1)
        has_parent = true_edge.any(dim=-1)
        null_index = batch["relation"].shape[-1]
        parent = torch.where(has_parent, parent, torch.full_like(parent, null_index))
        distribution = torch.cat(
            [
                relation["directed_relation_matrix"],
                relation["null_source_probability"].unsqueeze(-1),
            ],
            dim=-1,
        ).clamp_min(1e-8)
        loss = loss + relation_weight * F.nll_loss(
            distribution.log().reshape(-1, null_index + 1),
            parent.reshape(-1),
        )
    if use_relation and true_edge.any():
        if relation["lag_weights"].shape[1] > 0:
            weights = relation["lag_weights"].permute(0, 2, 3, 1)[true_edge]
            target = batch["lag"][true_edge] - 1
            loss = loss + lag_weight * F.nll_loss(weights.clamp_min(1e-8).log(), target)
    if use_role:
        # Source identity is mutually exclusive, so use a listwise loss instead
        # of independent binary labels that are dominated by negative nodes.
        source_probability = rank["source_role_score"].clamp_min(1e-8)
        source_probability = source_probability / source_probability.sum(dim=1, keepdim=True)
        loss = loss + source_role_weight * F.nll_loss(
            source_probability.log(), batch["source"]
        )
        loss = loss + response_role_weight * F.binary_cross_entropy(
            rank["response_role_score"], batch["response"]
        )
    return loss


def iter_batches(cases, batch_size, seed, shuffle=True):
    indices = np.arange(len(cases))
    if shuffle:
        np.random.RandomState(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [cases[int(index)] for index in indices[start:start + batch_size]]


def train_variant(variant, seed, train_cases, test_cases, args):
    config = {
        "full": {"max_lag": args.max_lag, "ranker": "source_hypothesis", "relation": True, "role": True},
        "no_lag": {"max_lag": 0, "ranker": "source_hypothesis", "relation": False, "role": True},
        "no_role": {"max_lag": args.max_lag, "ranker": "source_hypothesis_no_role", "relation": True, "role": False},
    }[variant]
    seed_everything(seed)
    device = torch.device(args.device)
    model = ControlledModel(args.node_num, args.window_size, config["max_lag"], config["ranker"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()
    for epoch in range(args.epochs):
        for cases in iter_batches(train_cases, args.batch_size, seed + epoch):
            batch = collate(cases, device)
            relation, rank = model(batch)
            loss = controlled_loss(
                batch,
                relation,
                rank,
                use_relation=config["relation"],
                use_role=config["role"],
                args=args,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return evaluate(model, test_cases, args)


@torch.no_grad()
def evaluate(model, cases, args):
    model.eval()
    source_hits = {1: [], 3: []}
    reciprocal_ranks = []
    margins = []
    source_role_hits = []
    source_role_margins = []
    response_role_truth = []
    response_role_score = []
    direction = []
    lag_errors = []
    relation_truth = []
    relation_score = []
    device = torch.device(args.device)
    for selected in iter_batches(cases, args.batch_size, 0, shuffle=False):
        batch = collate(selected, device)
        relation, rank = model(batch)
        score = rank["final_score"]
        for k in source_hits:
            top = score.topk(k, dim=1).indices
            source_hits[k].extend((top == batch["source"].unsqueeze(1)).any(dim=1).float().cpu().tolist())
        order = torch.argsort(score, dim=1, descending=True)
        positions = (order == batch["source"].unsqueeze(1)).nonzero(as_tuple=False)[:, 1]
        reciprocal_ranks.extend((1.0 / (positions.float() + 1.0)).cpu().tolist())
        source_score = score.gather(1, batch["source"].unsqueeze(1)).squeeze(1)
        masked_response = score.masked_fill(batch["response"] <= 0, float("-inf"))
        margins.extend((source_score - masked_response.max(dim=1).values).cpu().tolist())

        source_role = rank["source_role_score"]
        response_role = rank["response_role_score"]
        source_role_hits.extend(
            (source_role.argmax(dim=1) == batch["source"]).float().cpu().tolist()
        )
        source_role_at_source = source_role.gather(1, batch["source"].unsqueeze(1)).squeeze(1)
        source_role_at_response = source_role.masked_fill(batch["response"] <= 0, float("-inf"))
        source_role_margins.extend(
            (source_role_at_source - source_role_at_response.max(dim=1).values).cpu().tolist()
        )
        response_role_truth.extend(batch["response"].reshape(-1).cpu().tolist())
        response_role_score.extend(response_role.reshape(-1).cpu().tolist())

        rel = relation["directed_relation_matrix"]
        truth = batch["relation"]
        eye = torch.eye(rel.shape[-1], dtype=torch.bool, device=device).unsqueeze(0)
        relation_truth.extend(truth.masked_select(~eye.expand_as(truth)).cpu().tolist())
        relation_score.extend(rel.masked_select(~eye.expand_as(rel)).cpu().tolist())
        true_indices = torch.nonzero(truth > 0, as_tuple=False)
        for batch_idx, target, source in true_indices.tolist():
            direction.append(float(rel[batch_idx, target, source] > rel[batch_idx, source, target]))
            if relation["lag_weights"].shape[1] > 0:
                predicted_lag = int(relation["lag_weights"][batch_idx, :, target, source].argmax().item()) + 1
                lag_errors.append(abs(predicted_lag - int(batch["lag"][batch_idx, target, source].item())))
    return {
        "source_hit_at_1": float(np.mean(source_hits[1])),
        "source_hit_at_3": float(np.mean(source_hits[3])),
        "source_mrr": float(np.mean(reciprocal_ranks)),
        "source_response_margin": float(np.mean(margins)),
        "source_role_hit_at_1": float(np.mean(source_role_hits)),
        "source_role_margin": float(np.mean(source_role_margins)),
        "response_role_auprc": float(average_precision_score(response_role_truth, response_role_score)),
        "relation_auprc": float(average_precision_score(relation_truth, relation_score)),
        "direction_accuracy": float(np.mean(direction)),
        "lag_mae": None if not lag_errors else float(np.mean(lag_errors)),
    }


def aggregate(records):
    metrics = records[0]["metrics"].keys()
    result = {}
    for metric in metrics:
        values = [record["metrics"][metric] for record in records if record["metrics"][metric] is not None]
        result[metric] = None if not values else {
            "values": values,
            "mean": float(np.mean(values)),
            "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--variants", nargs="+", choices=["full", "no_lag", "no_role"], default=["full", "no_lag", "no_role"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--node-num", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--train-cases", type=int, default=600)
    parser.add_argument("--test-cases", type=int, default=240)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--noise", type=float, default=0.15)
    parser.add_argument("--train-data-seed", type=int, default=1731)
    parser.add_argument("--test-data-seed", type=int, default=2718)
    parser.add_argument("--final-loss-weight", type=float, default=1.0)
    parser.add_argument("--relation-loss-weight", type=float, default=0.5)
    parser.add_argument("--lag-loss-weight", type=float, default=0.5)
    parser.add_argument("--source-role-loss-weight", type=float, default=1.0)
    parser.add_argument("--response-role-loss-weight", type=float, default=0.25)
    parser.add_argument("--output-dir", default="res/rq6_controlled_propagation_20260825")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_config(args, variant, seed):
    return {
        "variant": variant,
        "seed": seed,
        "device": args.device,
        "node_num": args.node_num,
        "window_size": args.window_size,
        "max_lag": args.max_lag,
        "train_cases": args.train_cases,
        "test_cases": args.test_cases,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "noise": args.noise,
        "train_data_seed": getattr(args, "train_data_seed", 1731),
        "test_data_seed": getattr(args, "test_data_seed", 2718),
        "final_loss_weight": getattr(args, "final_loss_weight", 1.0),
        "relation_loss_weight": getattr(args, "relation_loss_weight", 0.5),
        "lag_loss_weight": getattr(args, "lag_loss_weight", 0.5),
        "source_role_loss_weight": getattr(args, "source_role_loss_weight", 1.0),
        "response_role_loss_weight": getattr(args, "response_role_loss_weight", 0.25),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def run_record_path(output, variant, seed):
    return output / "runs" / variant / f"seed{seed}.json"


def main():
    args = parse_args()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    train_cases = make_dataset(args.train_data_seed, args.train_cases, node_num=args.node_num, window_size=args.window_size, max_lag=args.max_lag, noise=args.noise)
    test_cases = make_dataset(args.test_data_seed, args.test_cases, node_num=args.node_num, window_size=args.window_size, max_lag=args.max_lag, noise=args.noise)
    records = []
    for variant in args.variants:
        for seed in args.seeds:
            path = run_record_path(output, variant, seed)
            config = run_config(args, variant, seed)
            if path.exists() and not args.overwrite:
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("run_config") != config:
                    raise RuntimeError(f"incompatible existing RQ6 run: {path}")
            else:
                metrics = train_variant(variant, seed, train_cases, test_cases, args)
                record = {"variant": variant, "seed": seed, "metrics": metrics, "run_config": config}
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    summary = {
        variant: aggregate([record for record in records if record["variant"] == variant])
        for variant in args.variants
    }
    payload = {
        "protocol": {key: value for key, value in vars(args).items() if key != "overwrite"},
        "ground_truth": "source node, response nodes, directed propagation edges, and integer lag",
        "records": records,
        "summary": summary,
    }
    (output / "rq6_controlled_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
