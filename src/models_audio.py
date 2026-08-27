import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class ContinuousAdaptiveLIF(nn.Module):
    """
    Upgrade 6: Continuous-time adaptive spike dynamics.
    Learns membrane decay (beta) dynamically for adaptive event-driven temporal processing.
    """
    def __init__(self, channels: int):
        super().__init__()
        # Learnable threshold and decay rates per channel
        self.beta_param = nn.Parameter(torch.full((1, channels, 1), 0.90))
        self.surrogate = surrogate.fast_sigmoid()

    def forward(self, x_seq, num_steps: int):
        batch, ch, length = x_seq.shape
        mem = torch.zeros(batch, ch, length, device=x_seq.device)
        spikes = []
        
        beta = torch.clamp(self.beta_param, 0.1, 0.99)
        
        for _ in range(num_steps):
            # Leaky Integration Continuous Dynamics
            mem = beta * mem + x_seq
            spk = self.surrogate(mem - 1.0) # Threshold = 1.0
            mem = mem * (1.0 - spk)         # Hard reset
            spikes.append(spk)
            
        return torch.stack(spikes).mean(0)

class ContinuousSpikingAudioNet(nn.Module):
    """SNN Audio Denoiser with Adaptive Spiking Dynamics."""
    def __init__(self, channels: int = 64, time_dim: int = 128, num_steps: int = 8, T_audio: int = 40):
        super().__init__()
        self.T_audio = T_audio
        self.num_steps = num_steps
        self.time_mlp = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, time_dim))
        
        self.in_conv = nn.Conv1d(1, channels, kernel_size=7, padding=3)
        self.lif1 = ContinuousAdaptiveLIF(channels)
        
        self.mid_conv = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.lif2 = ContinuousAdaptiveLIF(channels)
        
        self.out_conv = nn.Conv1d(channels, 1, kernel_size=7, padding=3)
        self.temb_proj = nn.Linear(time_dim, channels)

    def forward(self, x, t):
        temb = self.temb_proj(self.time_mlp((t.float() / self.T_audio).unsqueeze(-1)))[:, :, None]
        h = self.in_conv(x.unsqueeze(1)) + temb
        h = self.lif1(h, self.num_steps)
        h = self.mid_conv(h)
        h = self.lif2(h, self.num_steps)
        return self.out_conv(h).squeeze(1)


class ResidualDilatedTCN(nn.Module):
    """
    Upgrade 5: Ultra-Strong ANN Baseline (1D Residual Dilated TCN).
    Used to rigorously prove SNN superiority over strong, receptive-field expanded ANNs.
    """
    def __init__(self, channels: int = 64, time_dim: int = 128, T_audio: int = 40):
        super().__init__()
        self.T_audio = T_audio
        self.time_mlp = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, time_dim))
        self.temb_proj = nn.Linear(time_dim, channels)

        self.in_conv = nn.Conv1d(1, channels, kernel_size=7, padding=3)
        
        # Dilated residual blocks to capture multi-scale continuous audio features
        self.d1 = nn.Conv1d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.d2 = nn.Conv1d(channels, channels, kernel_size=3, padding=4, dilation=4)
        self.d3 = nn.Conv1d(channels, channels, kernel_size=3, padding=8, dilation=8)
        
        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()
        self.out_conv = nn.Conv1d(channels, 1, kernel_size=7, padding=3)

    def forward(self, x, t):
        temb = self.temb_proj(self.time_mlp((t.float() / self.T_audio).unsqueeze(-1)))[:, :, None]
        h = self.in_conv(x.unsqueeze(1)) + temb
        
        # Residual Dilated Blocks
        res = h
        h = self.act(self.norm(self.d1(h)))
        h = self.act(self.norm(self.d2(h))) + res
        
        res = h
        h = self.act(self.norm(self.d3(h))) + res
        return self.out_conv(h).squeeze(1)