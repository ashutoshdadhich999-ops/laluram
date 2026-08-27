import torch
import torch.nn as nn

class PoissonAudioDiffusion:
    def __init__(self, T: int = 40, scale: float = 20.0, decay: float = 0.05, device: str = "cuda"):
        self.T = T
        self.scale = scale
        self.decay = decay
        self.device = device
        self.t_steps = torch.linspace(0, 1, T, device=device)
        self.gamma = torch.exp(-decay * self.t_steps * 6.0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, mode: str = "poisson"):
        t_normalized = (t.float() / self.T).unsqueeze(-1)
        gamma_t = torch.exp(-self.decay * t_normalized * 6.0).to(self.device)
        
        if mode == "poisson":
            rate = torch.clamp(x0 * gamma_t * self.scale, min=1e-4)
            counts = torch.poisson(rate)
            x_t = counts / self.scale
            jitter = torch.randn_like(x_t) * 0.02 * (1.0 - gamma_t)
            x_t = torch.clamp(x_t + jitter, 0.0, 1.0)
            
        elif mode == "gaussian":
            noise = torch.randn_like(x0)
            x_t = torch.clamp(gamma_t * x0 + (1 - gamma_t) * noise, 0.0, 1.0)
            
        elif mode == "bernoulli":
            prob = torch.clamp(x0 * gamma_t, 0.0, 1.0)
            x_t = torch.bernoulli(prob)
            
        else:
            raise ValueError(f"Unknown encoding mode: {mode}")

        target_noise = x_t - x0
        return x_t, target_noise

def poison(x0: torch.Tensor, t: torch.Tensor, T_audio: int, max_rate: float,
           device: str, scale: float = 20.0, decay: float = 0.05, jitter_std: float = 0.02):
    diff = PoissonAudioDiffusion(T=T_audio, scale=scale, decay=decay, device=device)
    return diff.q_sample(x0, t, mode="poisson")
