import torch

from src.metrics.base_metric import BaseMetric
from src.utils import sisnr_utls


class SiSNRI(BaseMetric):
    def __call__(
        self,
        mix: torch.Tensor,
        source_1: torch.Tensor,
        source_2: torch.Tensor,
        predicted_source_1: torch.Tensor,
        predicted_source_2: torch.Tensor,
        **kwargs
    ):
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """
        baseline_metric = sisnr_utls.compute_pair_sisnr(
            predicted_1=mix,
            predicted_2=mix,
            target_1=source_1,
            target_2=source_2,
        )
        model_metric = sisnr_utls.compute_pair_sisnr(
            predicted_1=predicted_source_1,
            predicted_2=predicted_source_2,
            target_1=source_1,
            target_2=source_2,
        )

        return (model_metric - baseline_metric).mean()
