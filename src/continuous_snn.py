"""
Continuous-Time Adaptive SNN Dynamic Module.
Replaces fixed discrete time unrolling with Sub-step ODE Integration for continuous time membrane updates.
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class ContinuousLIFBlock1D(nn.Module):
    """Continuous-Time Adaptive 1D Spiking Residual Block using ODE sub-step integration."""

    def __init__(self, ch: int, time_dim: int, num_steps: int = 8, dt: float = 0.5):
        super().__init__()
        self.num_steps = num_steps
        self.dt = dt
        self.tau_m = 10.0  # Membrane time constant
        self.tau_s = 5.0   # Synaptic time constant
        
        self.time_proj = nn.Linear(time_dim, ch)
        self.conv1 = nn.Conv1d(ch, ch, 5, padding=2)
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv1d(ch, ch, 5, padding=2)
        self.norm2 = nn.GroupNorm(8, ch)
        self.spike_fn = surrogate.fast_sigmoid()

    def forward(self, x, temb):
        res = x
        h = self.norm1(self.conv1(x))
        temb_p = self.time_proj(temb)[:, :, None]
        h_in = h + temb_p

        batch, ch, seq_len = x.shape
        v1 = torch.zeros_like(h_in)
        i1 = torch.zeros_like(h_in)
        v2 = torch.zeros_like(h_in)
        i2 = torch.zeros_like(h_in)
        
        v_th = 1.0
        outs = []

        # Sub-step continuous Euler ODE integration
        for _ in range(self.num_steps):
            # Layer 1 ODE update
            di1 = (-i1 + h_in) / self.tau_s
            dv1 = (-v1 + i1) / self.tau_m
            i1 = i1 + di1 * self.dt
            v1 = v1 + dv1 * self.dt
            
            spk1 = self.spike_fn(v1 - v_th)
            v1 = v1 * (1.0 - spk1.detach())  # Reset mechanism
            
            # Layer 2 ODE update
            h2 = self.norm2(self.conv2(spk1))
            di2 = (-i2 + h2) / self.tau_s
            dv2 = (-v2 + i2) / self.tau_m
            i2 = i2 + di2 * self.dt
            v2 = v2 + dv2 * self.dt
            
            spk2 = self.spike_fn(v2 - v_th)
            v2 = v2 * (1.0 - spk2.detach())
            
            outs.append(spk2 + res)

        return torch.stack(outs).mean(0)
