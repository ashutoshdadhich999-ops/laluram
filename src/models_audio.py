"""
Audio Models: Discrete SNN, Continuous SNN, standard ANN, and Advanced 1D U-Net / TCN ANN Baseline.
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from src.continuous_snn import ContinuousLIFBlock1D


class ConvResBlock1D(nn.Module):
    """Spiking 1D residual block (LIF neurons)."""

    def __init__(self, ch: int, time_dim: int, num_steps: int):
        super().__init__()
        self.num_steps = num_steps
        self.time_proj = nn.Linear(time_dim, ch)
        self.conv1 = nn.Conv1d(ch, ch, 5, padding=2)
        self.norm1 = nn.GroupNorm(8, ch)
        self.lif1 = snn.Leaky(beta=0.92, spike_grad=surrogate.fast_sigmoid())
        self.conv2 = nn.Conv1d(ch, ch, 5, padding=2)
        self.norm2 = nn.GroupNorm(8, ch)
        self.lif2 = snn.Leaky(beta=0.92, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x, temb):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        res = x
        h = self.norm1(self.conv1(x))
        temb_p = self.time_proj(temb)[:, :, None]

        outs = []
        for _ in range(self.num_steps):
            h_in = h + temb_p
            spk, mem1 = self.lif1(h_in, mem1)
            h2 = self.norm2(self.conv2(spk))
            spk2, mem2 = self.lif2(h2, mem2)
            outs.append(spk2 + res)
        return torch.stack(outs).mean(0)


class NonSpikeConvRes1D(nn.Module):
    """Non-spiking counterpart of ConvResBlock1D."""

    def __init__(self, ch: int, time_dim: int):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, ch)
        self.conv1 = nn.Conv1d(ch, ch, 5, padding=2)
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv1d(ch, ch, 5, padding=2)
        self.norm2 = nn.GroupNorm(8, ch)
        self.act = nn.SiLU()

    def forward(self, x, temb):
        res = x
        h = self.norm1(self.conv1(x))
        h = h + self.time_proj(temb)[:, :, None]
        h = self.act(h)
        h = self.norm2(self.conv2(h))
        h = self.act(h)
        return h + res


class StrongAudioNet(nn.Module):
    """Discrete-time Spiking audio denoiser."""

    def __init__(self, channels: int = 64, time_dim: int = 128, num_steps: int = 8,
                 T_audio: int = 40):
        super().__init__()
        self.T_audio = T_audio
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, time_dim), nn.SiLU()
        )
        self.input = nn.Conv1d(1, channels, 7, padding=3)
        self.b1 = ConvResBlock1D(channels, time_dim, num_steps)
        self.b2 = ConvResBlock1D(channels, time_dim, num_steps)
        self.b3 = ConvResBlock1D(channels, time_dim, num_steps)
        self.out = nn.Conv1d(channels, 1, 7, padding=3)

    def forward(self, x, t):
        temb = self.time_mlp((t.float() / self.T_audio).unsqueeze(-1))
        h = self.input(x.unsqueeze(1))
        h = self.b1(h, temb)
        h = self.b2(h, temb)
        h = self.b3(h, temb)
        return self.out(h).squeeze(1)


class ContinuousSpikeAudioNet(nn.Module):
    """Advanced Continuous-Time Adaptive ODE Spiking audio denoiser."""

    def __init__(self, channels: int = 64, time_dim: int = 128, num_steps: int = 8, T_audio: int = 40):
        super().__init__()
        self.T_audio = T_audio
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, time_dim), nn.SiLU()
        )
        self.input = nn.Conv1d(1, channels, 7, padding=3)
        self.b1 = ContinuousLIFBlock1D(channels, time_dim, num_steps=num_steps)
        self.b2 = ContinuousLIFBlock1D(channels, time_dim, num_steps=num_steps)
        self.b3 = ContinuousLIFBlock1D(channels, time_dim, num_steps=num_steps)
        self.out = nn.Conv1d(channels, 1, 7, padding=3)

    def forward(self, x, t):
        temb = self.time_mlp((t.float() / self.T_audio).unsqueeze(-1))
        h = self.input(x.unsqueeze(1))
        h = self.b1(h, temb)
        h = self.b2(h, temb)
        h = self.b3(h, temb)
        return self.out(h).squeeze(1)


class NonSpikeAudioNet(nn.Module):
    """Standard Non-spiking (ANN) audio denoiser."""

    def __init__(self, channels: int = 64, time_dim: int = 128, T_audio: int = 40):
        super().__init__()
        self.T_audio = T_audio
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, time_dim), nn.SiLU()
        )
        self.input = nn.Conv1d(1, channels, 7, padding=3)
        self.b1 = NonSpikeConvRes1D(channels, time_dim)
        self.b2 = NonSpikeConvRes1D(channels, time_dim)
        self.b3 = NonSpikeConvRes1D(channels, time_dim)
        self.out = nn.Conv1d(channels, 1, 7, padding=3)

    def forward(self, x, t):
        temb = self.time_mlp((t.float() / self.T_audio).unsqueeze(-1))
        h = self.input(x.unsqueeze(1))
        h = self.b1(h, temb)
        h = self.b2(h, temb)
        h = self.b3(h, temb)
        return self.out(h).squeeze(1)


class StrongUNetAudioNet(nn.Module):
    """Strong ANN Baseline: Dilated 1D U-Net / TCN Architecture."""

    def __init__(self, channels: int = 64, time_dim: int = 128, T_audio: int = 40):
        super().__init__()
        self.T_audio = T_audio
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, time_dim), nn.SiLU()
        )
        self.input = nn.Conv1d(1, channels, 7, padding=3)
        
        # Dilated residual blocks for large receptive field
        self.d1 = nn.Conv1d(channels, channels, 3, padding=1, dilation=1)
        self.d2 = nn.Conv1d(channels, channels, 3, padding=2, dilation=2)
        self.d4 = nn.Conv1d(channels, channels, 3, padding=4, dilation=4)
        
        self.act = nn.SiLU()
        self.norm = nn.GroupNorm(8, channels)
        self.out = nn.Conv1d(channels, 1, 7, padding=3)

    def forward(self, x, t):
        temb = self.time_mlp((t.float() / self.T_audio).unsqueeze(-1))[:, :, None]
        h = self.input(x.unsqueeze(1))
        
        h = self.act(self.norm(self.d1(h + temb)))
        h = self.act(self.norm(self.d2(h + temb)))
        h = self.act(self.norm(self.d4(h + temb)))
        
        return self.out(h).squeeze(1)
