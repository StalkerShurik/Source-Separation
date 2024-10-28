import torch
from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio


def compute_single_sisnr(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return scale_invariant_signal_distortion_ratio(
        preds=predicted, target=target, zero_mean=True
    )


def compute_pair_sisnr(
    predicted_1: torch.Tensor,
    predicted_2: torch.Tensor,
    target_1: torch.Tensor,
    target_2: torch.Tensor,
) -> torch.Tensor:
    return torch.max(
        (
            compute_single_sisnr(predicted_1, target_1)
            + compute_single_sisnr(predicted_2, target_2)
        ),
        (
            compute_single_sisnr(predicted_1, target_2)
            + compute_single_sisnr(predicted_2, target_1)
        ),
    )
