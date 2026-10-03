"""The four training objectives. Train all four from the same starting weights."""

import torch


def oracle_loss(pred_velocity: torch.Tensor, true_velocity: torch.Tensor) -> torch.Tensor:
    """Supervised oracle: MSE against the true velocity."""
    raise NotImplementedError


def temporal_prediction_loss(pred_future: torch.Tensor, movie: torch.Tensor, k: int) -> torch.Tensor:
    """Predict the input k steps ahead. Log copy_baseline_loss beside it."""
    raise NotImplementedError


def copy_baseline_loss(movie: torch.Tensor, k: int) -> torch.Tensor:
    """Error of simply repeating the current frame k steps ahead."""
    raise NotImplementedError


def photometric_warp_loss(pred_velocity: torch.Tensor, movie: torch.Tensor,
                          smoothness: float = 0.1) -> torch.Tensor:
    """Rebuild the previous frame from the current one, plus smoothness across columns."""
    raise NotImplementedError


def latent_prediction_loss(pred_latent: torch.Tensor, target_latent: torch.Tensor,
                           variance_weight: float = 0.0) -> torch.Tensor:
    """Predict a slowly updated (EMA) copy's future hidden state. Optional."""
    raise NotImplementedError
