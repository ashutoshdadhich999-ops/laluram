"""
Neuromorphic Energy Analysis Module.
Calculates per-layer operation counts (MACs vs ACs) and estimates real energy consumption
in Joules using standard 45nm CMOS literature constants:
  - E_MAC = 4.6 pJ (32-bit floating point / fixed point multiply-accumulate)
  - E_AC  = 0.9 pJ (32-bit addition / accumulator operation)
"""

import torch
import torch.nn as nn
import snntorch as snn


# Energy constants in Joules (45nm CMOS)
E_MAC = 4.6e-12  # Joules per MAC
E_AC  = 0.9e-12  # Joules per AC


def count_conv1d_ops(layer: nn.Conv1d, input_shape: tuple):
    """Calculates single forward pass MACs for a 1D Convolutional layer."""
    batch, in_ch, seq_len = input_shape
    out_ch = layer.out_channels
    k = layer.kernel_size[0]
    stride = layer.stride[0]
    padding = layer.padding[0]
    
    out_len = (seq_len + 2 * padding - k) // stride + 1
    macs_per_sample = out_ch * in_ch * k * out_len
    return macs_per_sample


def profile_energy_audio(model: nn.Module, input_tensor: torch.Tensor, t_tensor: torch.Tensor, spike_rate: float):
    """
    Computes theoretical MAC/AC count and total energy consumption for SNN vs ANN models.
    """
    model.eval()
    total_macs = 0
    total_acs = 0
    
    # Trace 1D Convolution layers
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            # Base operations per frame
            k = m.kernel_size[0]
            in_ch = m.in_channels
            out_ch = m.out_channels
            out_len = input_tensor.shape[-1]  # approximate length
            
            op_count = out_ch * in_ch * k * out_len
            
            if hasattr(model, 'b1') and ('Strong' in model.__class__.__name__ or 'Continuous' in model.__class__.__name__):
                # Spiking model executes Accumulate (AC) ops proportionally to firing rate across unrolled timesteps
                num_steps = getattr(model.b1, 'num_steps', 8)
                total_acs += op_count * num_steps * spike_rate
            else:
                # Standard ANN model executes Multiply-Accumulate (MAC) ops once
                total_macs += op_count

    energy_mac_joules = total_macs * E_MAC
    energy_ac_joules = total_acs * E_AC
    total_energy_joules = energy_mac_joules + energy_ac_joules
    
    return {
        "MACs": total_macs,
        "ACs": total_acs,
        "Energy_mJ": total_energy_joules * 1000.0,
    }
