import torch


def psnr(denoised, ground_truth):
    """
    Computes peak signal-to-noise ratio
    Args:
        denoised: network output
        ground_truth: target

    Returns: psnr

    """
    mse = torch.mean((denoised - ground_truth) ** 2)
    return -10 * torch.log10(mse + 10**-8)
