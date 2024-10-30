import torch
from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio

from src.metrics.base_metric import BaseMetric
from src.utils import metric_utls


class SiSNRI(BaseMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metric = metric_utls.create_permutation_metric(
            scale_invariant_signal_distortion_ratio
        )

    def __call__(
        self, mix: torch.Tensor, target: torch.Tensor, predict: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """

        baseline_metric = self.metric(torch.stack([mix, mix], dim=1), target)
        model_metric = self.metric(predict, target)

        return (model_metric - baseline_metric).mean()
