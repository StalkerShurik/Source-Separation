import torch
from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio

from src.metrics.base_metric import BaseMetric
from src.utils.metric_utls import compute_metric


class SiSNRI(BaseMetric):
    def __call__(
        self, mix: torch.Tensor, target: torch.Tensor, predict: torch.Tensor, **kwargs
    ):
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """
        baseline_metric = compute_metric(
            target=target,
            predict=torch.stack([mix, mix], dim=1),
            metric=scale_invariant_signal_distortion_ratio,
        )
        model_metric = compute_metric(
            target=target,
            predict=predict,
            metric=scale_invariant_signal_distortion_ratio,
        )

        return (model_metric - baseline_metric).mean()
