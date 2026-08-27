"""
Plotting and Visualization Utilities. Includes Denoising behavior across t,
energy comparisons, and standard performance metrics.
"""

import os
import matplotlib.pyplot as plt
import torch

from src.diffusion_audio import poison


def plot_image_samples(model, test_loader, diff, timesteps, device, out_dir):
    model.eval()
    x, _ = next(iter(test_loader))
    x = x[:8].to(device)
    t = torch.tensor([timesteps // 2] * 8, device=device)
    noise = torch.randn_like(x)
    xt = diff.q_sample(x, t, noise)

    with torch.no_grad():
        pred = model(xt, t)
        a = torch.sqrt(diff.alpha_bar[t])[:, None, None, None]
        b = torch.sqrt(1 - diff.alpha_bar[t])[:, None, None, None]
        x0 = ((xt - b * pred) / a.clamp(1e-8)).clamp(0, 1)

    fig, axes = plt.subplots(3, 8, figsize=(12, 4.5))
    for i in range(8):
        axes[0, i].imshow(x[i, 0].cpu(), cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Clean")

        axes[1, i].imshow(xt[i, 0].cpu(), cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Noisy")

        axes[2, i].imshow(x0[i, 0].cpu(), cmap="gray")
        axes[2, i].axis("off")
        if i == 0:
            axes[2, i].set_title("Denoised")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "image_denoising_grid.png"), dpi=200)
    plt.close()


def plot_audio_waveforms(model, test_loader, T_audio, max_rate, device, out_dir):
    model.eval()
    x0 = next(iter(test_loader))[:1].to(device)
    t = torch.tensor([T_audio // 2], device=device)
    noisy, _ = poison(x0, t, T_audio, max_rate, device)

    with torch.no_grad():
        pred = model(noisy, t)
        den = (noisy - pred).clamp(0, 1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(x0[0].cpu().numpy(), color="black")
    axes[0].set_title("Clean Waveform")
    axes[1].plot(noisy[0].cpu().numpy(), color="crimson")
    axes[1].set_title("Corrupted Waveform")
    axes[2].plot(den[0].cpu().numpy(), color="teal")
    axes[2].set_title("Spiking Denoised Output")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "audio_waveforms.png"), dpi=200)
    plt.close()


def plot_temporal_denoising(model, test_loader, T_audio, max_rate, device, out_dir):
    """Visualizes model outputs across t = [0, 5, 10, 20, 30, 40]."""
    model.eval()
    x0 = next(iter(test_loader))[:1].to(device)
    timesteps_to_plot = [0, 5, 10, 20, 30, min(40, T_audio - 1)]

    fig, axes = plt.subplots(len(timesteps_to_plot), 2, figsize=(10, 2 * len(timesteps_to_plot)))

    with torch.no_grad():
        for i, t_val in enumerate(timesteps_to_plot):
            t = torch.tensor([t_val], device=device)
            noisy, _ = poison(x0, t, T_audio, max_rate, device)
            pred = model(noisy, t)
            den = (noisy - pred).clamp(0, 1)

            axes[i, 0].plot(noisy[0].cpu().numpy(), color="crimson", alpha=0.7)
            axes[i, 0].set_ylabel(f"t={t_val}")
            if i == 0:
                axes[i, 0].set_title("Corrupted Input")

            axes[i, 1].plot(den[0].cpu().numpy(), color="teal")
            if i == 0:
                axes[i, 1].set_title("Denoised Output")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "temporal_denoising_progression.png"), dpi=200)
    plt.close()


def plot_comparison_bars(img_s_imp, img_ns_imp, aud_s_imp, aud_ns_imp, sparsity, out_dir):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    categories = ["Image (MNIST %)", "Audio (SI-SDR dB)"]
    spiking_vals = [img_s_imp, aud_s_imp]
    nonspiking_vals = [img_ns_imp, aud_ns_imp]

    x = range(len(categories))
    width = 0.35

    ax1.bar([p - width / 2 for p in x], spiking_vals, width, label="Spiking SNN", color="teal")
    ax1.bar([p + width / 2 for p in x], nonspiking_vals, width, label="Non-Spiking ANN", color="darkorange")

    ax1.set_ylabel("Improvement Metric")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend(loc="upper left")
    ax1.set_title(f"SNN vs ANN Comparison (SNN Sparsity: {sparsity * 100:.1f}%)")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "snn_vs_ann_metrics.png"), dpi=200)
    plt.close()
