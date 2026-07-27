import unittest

import dgl
import torch

from models.reacr_rca_anomaly_head import MultiHeadAnomalyHead
from models.reacr_rca_dynamic_backbone import DynamicSpatialTemporalBackbone


class CausalTemporalBackboneTest(unittest.TestCase):
    def _graphs(self, window_size: int):
        return [
            dgl.graph(
                (torch.tensor([0, 1]), torch.tensor([1, 0])),
                num_nodes=2,
            )
            for _ in range(window_size)
        ]

    def test_causal_attention_is_enabled_by_default(self):
        model = DynamicSpatialTemporalBackbone(
            in_dim=3,
            spatial_hidden_dim=4,
            temporal_hidden_dim=4,
            num_spatial_layers=1,
            dropout=0.0,
            num_temporal_heads=1,
        )
        self.assertTrue(model.causal_temporal_attention)

        mask = model._causal_attention_mask(4, torch.device('cpu'))
        expected = torch.tensor(
            [
                [False, True, True, True],
                [False, False, True, True],
                [False, False, False, True],
                [False, False, False, False],
            ]
        )
        torch.testing.assert_close(mask, expected)

    def test_past_representations_do_not_depend_on_future_inputs(self):
        torch.manual_seed(7)
        window_size = 5
        prefix_size = 3
        model = DynamicSpatialTemporalBackbone(
            in_dim=3,
            spatial_hidden_dim=4,
            temporal_hidden_dim=4,
            num_spatial_layers=1,
            dropout=0.0,
            num_temporal_heads=1,
        ).eval()

        features = torch.randn(1, window_size, 2, 3)
        changed_future = features.clone()
        changed_future[:, prefix_size:, :, :] += 100.0 * torch.randn_like(
            changed_future[:, prefix_size:, :, :]
        )

        with torch.no_grad():
            original_output = model(self._graphs(window_size), features)
            changed_output = model(self._graphs(window_size), changed_future)

        torch.testing.assert_close(
            original_output['temporal_states'][:, :prefix_size, :, :],
            changed_output['temporal_states'][:, :prefix_size, :, :],
            rtol=0.0,
            atol=1e-6,
        )


class ObservationForecastAnomalyHeadTest(unittest.TestCase):
    def test_forecast_targets_next_observation_and_exposes_losses(self):
        torch.manual_seed(11)
        head = MultiHeadAnomalyHead(
            hidden_dim=4,
            feat_dim=3,
            fusion_hidden_dim=5,
        )
        temporal_states = torch.randn(2, 4, 3, 4, requires_grad=True)
        observed_features = torch.randn(2, 4, 3, 3)

        output = head(temporal_states, observed_features)

        self.assertEqual(output['next_step_forecast'].shape, (2, 3, 3, 3))
        torch.testing.assert_close(
            output['next_step_target'],
            observed_features[:, 1:, :, :],
        )
        expected_forecast_error = torch.mean(
            (output['next_step_forecast'] - observed_features[:, 1:, :, :]) ** 2,
            dim=-1,
        )
        torch.testing.assert_close(
            output['forecast_error'][:, 1:, :],
            expected_forecast_error,
        )
        torch.testing.assert_close(
            output['forecast_loss'],
            expected_forecast_error.mean(),
        )
        torch.testing.assert_close(
            output['reconstruction_loss'],
            output['recon_error'].mean(),
        )

        self.assertTrue(torch.all(output['forecast_error'][:, 0, :] == 0))
        self.assertTrue(torch.all(~output['forecast_valid_mask'][:, 0, :]))
        self.assertTrue(torch.all(output['forecast_valid_mask'][:, 1:, :]))

        (output['reconstruction_loss'] + output['forecast_loss']).backward()
        self.assertIsNotNone(head.reconstruction_head[0].weight.grad)
        self.assertIsNotNone(head.forecast_head[0].weight.grad)

    def test_single_step_window_has_no_forecast_target(self):
        head = MultiHeadAnomalyHead(hidden_dim=4, feat_dim=3, fusion_hidden_dim=5)
        output = head(torch.randn(2, 1, 3, 4), torch.randn(2, 1, 3, 3))

        self.assertEqual(output['next_step_forecast'].shape, (2, 0, 3, 3))
        self.assertEqual(output['next_step_target'].shape, (2, 0, 3, 3))
        self.assertEqual(output['forecast_loss'].item(), 0.0)
        self.assertTrue(torch.all(output['forecast_error'] == 0))
        self.assertTrue(torch.all(~output['forecast_valid_mask']))


if __name__ == '__main__':
    unittest.main()
