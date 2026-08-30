import torch

from baselines.rq1_common import bag_classification_loss, topk_metrics
from baselines.rq1_models import build_model


def test_all_trainable_baselines_return_node_scores():
    spans = {'metric': list(range(3)), 'trace': [3, 4], 'log': [5]}
    x = torch.randn(2, 4, 5, 6)
    adjacency = torch.eye(5).repeat(2, 1, 1)
    labels = torch.zeros(2, 5)
    labels[0, 1] = 1
    labels[1, 2:4] = 1
    for method in ['DejaVu', 'Eadro', 'MicroIRC', 'MODIFy', 'MicroInspect', 'TVDiag', 'UDA-RCL']:
        model = build_model(method, 6, spans, hidden=16)
        scores = model(x, adjacency)
        assert scores.shape == labels.shape
        loss = bag_classification_loss(scores, labels) + model.extra_loss()
        assert torch.isfinite(loss)
        loss.backward()


def test_any_hit_topk_metrics():
    scores = torch.tensor([[0.1, 0.9, 0.2], [0.8, 0.7, 0.1]])
    labels = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    metrics = topk_metrics(scores, labels, ks=(1, 2, 3))
    assert metrics['A@1'] == 50.0
    assert metrics['A@2'] == 100.0
    assert metrics['A@3'] == 100.0
