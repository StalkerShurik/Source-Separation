import typing as tp

import torch
from torchmetrics.functional.audio.snr import scale_invariant_signal_noise_ratio

from src.metrics.base_metric import BaseMetric
from src.utils.metric_utls import CustomePIT


class SiSNRI(BaseMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metric = CustomePIT(scale_invariant_signal_noise_ratio)

    def __call__(
        self,
        mix: torch.Tensor,
        predict: torch.Tensor,
        target: torch.Tensor,
        **kwargs: tp.Any
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

        return model_metric - baseline_metric


class SimpleSiSNRI(BaseMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metric_func = scale_invariant_signal_noise_ratio

    def __call__(
        self,
        mix: torch.Tensor,
        predict: torch.Tensor,
        target: torch.Tensor,
        **kwargs: tp.Any
    ) -> torch.Tensor:
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """

        baseline_metric = 0.5 * (
            self.metric_func(mix, target[:, 0, :]).mean()
            + self.metric_func(mix, target[:, 1, :]).mean()
        )
        model_metric = 0.5 * (
            self.metric_func(predict[:, 0, :], target[:, 1, :]).mean()
            + self.metric_func(predict[:, 1, :], target[:, 0, :]).mean()
        )

        return model_metric - baseline_metric
