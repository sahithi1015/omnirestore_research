"""CPU metrics with no skimage dependency; LPIPS is an optional extra."""
import warnings
import numpy as np
import torch
from torch.nn import functional as F


def mean_squared_error(a, b):
    return float(np.mean((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) ** 2))


def peak_signal_noise_ratio(a, b, data_range=1.0):
    mse = mean_squared_error(a, b)
    return float('inf') if mse == 0 else float(10 * np.log10(data_range ** 2 / mse))


def structural_similarity(a, b, channel_axis=-1, data_range=1.0, win_size=7):
    # Uniform valid windows, sample covariance: skimage's default SSIM convention.
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError('SSIM input shapes differ')
    if a.ndim == 2:
        a, b = a[..., None], b[..., None]
    elif channel_axis != -1:
        a, b = np.moveaxis(a, channel_axis, -1), np.moveaxis(b, channel_axis, -1)
    if min(a.shape[:2]) < win_size:
        raise ValueError('SSIM images are smaller than win_size')
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    def avg(z):
        integral = np.pad(z, ((1, 0), (1, 0), (0, 0))).cumsum(0).cumsum(1)
        return (integral[win_size:, win_size:] - integral[:-win_size, win_size:]
                - integral[win_size:, :-win_size] + integral[:-win_size, :-win_size]) / win_size ** 2
    ux, uy = avg(x), avg(y)
    correction = win_size ** 2 / (win_size ** 2 - 1)
    vx, vy = (avg(x*x)-ux*ux)*correction, (avg(y*y)-uy*uy)*correction
    vxy = (avg(x*y)-ux*uy)*correction
    c1, c2 = (0.01*data_range)**2, (0.03*data_range)**2
    return float((((2*ux*uy+c1)*(2*vxy+c2))/((ux*ux+uy*uy+c1)*(vx+vy+c2))).mean())


class UnavailableLPIPS:
    def __call__(self, *args):
        return torch.tensor(float('nan'))


def make_lpips(device):
    try:
        import lpips
    except ImportError:
        warnings.warn('LPIPS unavailable: reporting NaN, not a measured LPIPS score.')
        return UnavailableLPIPS()
    return lpips.LPIPS(net='vgg').to(device).eval()
