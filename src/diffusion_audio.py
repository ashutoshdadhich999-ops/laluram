"""
Forward Poisson Diffusion Process & Noise Corruption Engines.
Simulates discrete Markov Poisson jump transition dynamics: q(x_t | x_{t-1})
"""

import torch


def poison(x0: torch.Tensor, t: torch.Tensor, T_audio: int, max_rate: float,
           device: str, scale: float = 15.0, decay: float = 0.06,
           jitter_std: float = 0.05, mode: str = "poisson"):
    """
    Corrupt a clean waveform batch `x0` at diffusion step `t` using specified noise dynamics.
    Modes:
      - 'poisson': Continuous Poisson jump noise process (Default)
      - 'gaussian': Standard additive Gaussian diffusion noise
      - 'bernoulli': Bernoulli spike probability degradation
    """
    t_flat = t.view(-1, 1).float().to(device)
    gamma = torch.exp(-decay * t_flat * 6 / T_audio)

    if mode == "poisson":
        rate = x0 * gamma * max_rate
        rate = torch.clamp(rate, min=1e-4, max=max_rate)
        counts = torch.poisson(rate * scale)
        noisy = counts / scale
        noisy = noisy + torch.randn_like(noisy) * jitter_std * (1 - gamma)
    elif mode == "gaussian":
        sigma = (1.0 - gamma)
        noisy = x0 + torch.randn_like(x0) * sigma
    elif mode == "bernoulli":
        p = torch.clamp(x0 * gamma, 0.001, 0.999)
        spikes = torch.bernoulli(p)
        noisy = spikes + torch.randn_like(spikes) * jitter_std
    else:
        raise ValueError(f"Unknown corruption mode: {mode}")

    noisy = noisy.clamp(0, 1)
    return noisy, noisy - x0
