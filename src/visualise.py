import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_temporal_denoiser_behavior(model, sample_audio, diff_process, out_dir: str):
    """
    Upgrade 4: Plot output waveforms across corruption scale (t=0, 5, 10, 20, 30, 40)
    to visually prove temporal denoising capacity to PhD PIs.
    """
    model.eval()
    timesteps_to_plot = [0, 5, 10, 20, 30, 40]
    device = next(model.parameters()).device
    
    fig, axes = plt.subplots(len(timesteps_to_plot), 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Temporal Denoising Dynamics Across Timesteps (SNN)", fontsize=14, fontweight='bold')
    
    clean = sample_audio.unsqueeze(0).to(device)
    
    with torch.no_grad():
        for idx, t_val in enumerate(timesteps_to_plot):
            t_tensor = torch.tensor([t_val], device=device)
            if t_val == 0:
                noisy = clean
                pred_residual = torch.zeros_like(clean)
            else:
                noisy, _ = diff_process.q_sample(clean, t_tensor, mode="poisson")
                pred_residual = model(noisy, t_tensor)
            
            denoised = torch.clamp(noisy - pred_residual, 0.0, 1.0)
            
            ax = axes[idx]
            ax.plot(noisy[0].cpu().numpy(), label="Corrupted Waveform", color='gray', alpha=0.5)
            ax.plot(denoised[0].cpu().numpy(), label=f"Denoised (t={t_val})", color='crimson', linewidth=1.2)
            ax.set_ylabel(f"t={t_val}", rotation=0, labelpad=20, fontweight='bold')
            ax.grid(True, linestyle="--", alpha=0.5)
            if idx == 0:
                ax.legend(loc="upper right")

    plt.xlabel("Audio Samples")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "temporal_denoising_proof.png"), dpi=300)
    plt.close()