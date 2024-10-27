import torch

from src.metrics.base_metric import BaseMetric
from src.metrics.si_snr import SiSNR


class SiSNRI(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        """
        Example of a nested metric class. Applies metric function
        object (for example, from TorchMetrics) on tensors.

        Notice that you can define your own metric calculation functions
        inside the '__call__' method.

        Args:
            metric (Callable): function to calculate metrics.
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.si_snr = SiSNR(device, *args, **kwargs)

    def __call__(
        self, 
        mix,
        source1,
        source2,
        predicted_source1,
        predicted_source2,
        **kwargs):
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """
        before = self.si_snr(source1, mix) + self.si_snr(source2, mix)

        after_v1 = self.si_snr(source1, predicted_source1) + self.si_snr(source2, predicted_source2)
        after_v2 = self.si_snr(source2, predicted_source1) + self.si_snr(source1, predicted_source2)
        permuted_snr = torch.max(after_v1, after_v2)

        return (permuted_snr - before).mean()
