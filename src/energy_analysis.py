import torch
import torch.nn as nn
import snntorch as snn

class NeuromorphicEnergyAnalyzer:
    """
    Rigorous Energy Estimation based on literature hardware values:
    - Standard 45nm CMOS technology estimates:
      - E_MAC (Multiply-Accumulate, 32-bit Float) = 4.6 pJ
      - E_AC  (Accumulate-only, 32-bit Integer/Float) = 0.9 pJ
    """
    E_MAC = 4.6e-12  # Joules
    E_AC  = 0.9e-12  # Joules

    def __init__(self, model: nn.Module):
        self.model = model

    def calculate_layer_ops(self, input_tensor: torch.Tensor):
        total_macs = 0
        total_acs = 0
        layer_details = []

        def conv_hook(module, inp, out):
            nonlocal total_macs
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                # Kernel params * output spatial features
                kernel_ops = module.weight.numel()
                out_elements = out.numel() // out.shape[0] # Per sample
                total_macs += kernel_ops * out_elements

        # Register temporary hooks
        hooks = []
        for m in self.model.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                hooks.append(m.register_forward_hook(conv_hook))

        # Forward pass to get total theoretical MACs
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor, torch.tensor([1], device=input_tensor.device))

        for h in hooks:
            h.remove()

        return total_macs

    def compute_energy(self, input_tensor: torch.Tensor, mean_spike_rate: float):
        total_macs = self.calculate_layer_ops(input_tensor)
        
        # ANN Energy (always performs full MAC operations)
        ann_energy_joules = total_macs * self.E_MAC
        
        # SNN Energy (converts MACs to sparse ACs based on spike rate)
        snn_acs = total_macs * mean_spike_rate
        snn_energy_joules = snn_acs * self.E_AC
        
        savings_percent = ((ann_energy_joules - snn_energy_joules) / ann_energy_joules) * 100.0

        return {
            "Total Theoretical MACs": total_macs,
            "ANN Energy (mJ)": ann_energy_joules * 1e3,
            "SNN Energy (mJ)": snn_energy_joules * 1e3,
            "Energy Savings (%)": savings_percent
        }