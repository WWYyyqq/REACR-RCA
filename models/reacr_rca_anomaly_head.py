from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAnomalyHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        feat_dim: int,
        fusion_hidden_dim: int = 64,
        use_change_point: bool = False,
        change_point_weight: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim
        self.use_change_point = use_change_point
        self.change_point_weight = change_point_weight
        self.eps = eps

        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, feat_dim),
        )
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, feat_dim),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(4 if use_change_point else 3, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def _change_point_score(self, temporal_states: torch.Tensor) -> torch.Tensor:
        # Compare each state with the historical prefix distribution. This keeps
        # the Baro-style change-point signal differentiable and local to a window.
        batch_size, window_size, node_num, hidden_dim = temporal_states.shape
        score = torch.zeros(batch_size, window_size, node_num, device=temporal_states.device)
        if window_size <= 1:
            return score

        cumsum = torch.cumsum(temporal_states, dim=1)
        cumsum_sq = torch.cumsum(temporal_states ** 2, dim=1)
        prev_sum = cumsum[:, :-1, :, :]
        prev_sum_sq = cumsum_sq[:, :-1, :, :]
        counts = torch.arange(1, window_size, device=temporal_states.device, dtype=temporal_states.dtype)
        counts = counts.view(1, window_size - 1, 1, 1)

        prev_mean = prev_sum / counts
        prev_second = prev_sum_sq / counts
        prev_var = torch.clamp(prev_second - prev_mean ** 2, min=self.eps)
        standardized_delta = torch.abs(temporal_states[:, 1:, :, :] - prev_mean) / torch.sqrt(prev_var + self.eps)
        score[:, 1:, :] = torch.mean(standardized_delta, dim=-1)
        return torch.log1p(score) * self.change_point_weight

    def forward(self, temporal_states: torch.Tensor, original_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        # temporal_states: [B, W, N, H]
        # original_features: [B, W, N, F], after the optional modality gate.
        batch_size, window_size, node_num, hidden_dim = temporal_states.shape

        recon_pred = self.reconstruction_head(temporal_states)
        recon_error = torch.mean((recon_pred - original_features) ** 2, dim=-1)
        reconstruction_loss = recon_error.mean()

        temporal_shift = temporal_states.new_zeros(batch_size, window_size, node_num)
        temporal_shift[:, 1:, :] = torch.norm(
            temporal_states[:, 1:, :, :] - temporal_states[:, :-1, :, :],
            dim=-1,
            p=2,
        )

        # A forecast produced from h_t is aligned with its observation target
        # x_{t+1}. The first time step has no preceding state, so its aligned
        # error is explicitly zero and excluded from forecast_loss.
        forecast_error = temporal_states.new_zeros(batch_size, window_size, node_num)
        forecast_valid_mask = torch.zeros(
            batch_size,
            window_size,
            node_num,
            dtype=torch.bool,
            device=temporal_states.device,
        )
        if window_size > 1:
            next_step_forecast = self.forecast_head(temporal_states[:, :-1, :, :])
            next_step_target = original_features[:, 1:, :, :]
            forecast_error[:, 1:, :] = torch.mean(
                (next_step_forecast - next_step_target) ** 2,
                dim=-1,
            )
            forecast_valid_mask[:, 1:, :] = True
            forecast_loss = forecast_error[:, 1:, :].mean()
        else:
            next_step_forecast = original_features[:, :0, :, :]
            next_step_target = original_features[:, :0, :, :]
            forecast_loss = temporal_states.sum() * 0.0

        component_list = [recon_error, temporal_shift, forecast_error]
        change_point_score = torch.zeros_like(recon_error)
        if self.use_change_point:
            change_point_score = self._change_point_score(temporal_states)
            component_list.append(change_point_score)

        components = torch.stack(component_list, dim=-1)
        anomaly_score = self.fusion_head(components).squeeze(-1)
        anomaly_score = F.softplus(anomaly_score)

        return {
            'reconstruction_prediction': recon_pred,
            'recon_error': recon_error,
            'reconstruction_loss': reconstruction_loss,
            'next_step_forecast': next_step_forecast,
            'next_step_target': next_step_target,
            'temporal_shift': temporal_shift,
            'forecast_error': forecast_error,
            'forecast_valid_mask': forecast_valid_mask,
            'forecast_loss': forecast_loss,
            'change_point_score': change_point_score,
            'anomaly_components': components,
            'anomaly_score': anomaly_score,
        }


def summarize_anomaly_output(output: Dict[str, torch.Tensor]) -> Dict[str, object]:
    return {
        'reconstruction_prediction_shape': list(output['reconstruction_prediction'].shape),
        'recon_error_shape': list(output['recon_error'].shape),
        'reconstruction_loss_shape': list(output['reconstruction_loss'].shape),
        'next_step_forecast_shape': list(output['next_step_forecast'].shape),
        'next_step_target_shape': list(output['next_step_target'].shape),
        'temporal_shift_shape': list(output['temporal_shift'].shape),
        'forecast_error_shape': list(output['forecast_error'].shape),
        'forecast_valid_mask_shape': list(output['forecast_valid_mask'].shape),
        'forecast_loss_shape': list(output['forecast_loss'].shape),
        'change_point_score_shape': list(output['change_point_score'].shape),
        'anomaly_components_shape': list(output['anomaly_components'].shape),
        'anomaly_score_shape': list(output['anomaly_score'].shape),
    }
