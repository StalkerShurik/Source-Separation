import typing as tp

import torch
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility

from src.metrics.base_metric import BaseMetric
from src.utils.metric_utls import CustomePIT


class STOI(BaseMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fs = 16000
        self.metric = CustomePIT(short_time_objective_intelligibility)

    def __call__(
        self, predict: torch.Tensor, target: torch.Tensor, **kwargs: tp.Any
    ) -> torch.Tensor:
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """

        model_metric = self.metric(predict, target, fs=self.fs)

        return model_metric.mean()
