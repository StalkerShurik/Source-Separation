import typing as tp

import torch
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality

from src.metrics.base_metric import BaseMetric
from src.utils.metric_utls import CustomePIT


class PESQ(BaseMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fs = 16000
        self.mode = "wb"
        self.metric = CustomePIT(perceptual_evaluation_speech_quality)

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

        model_metric = self.metric(predict, target, fs=self.fs, mode=self.mode)
        return model_metric.mean()
