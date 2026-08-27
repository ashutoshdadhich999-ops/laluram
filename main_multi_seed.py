"""
Main Experiment Script.
Runs comprehensive evaluations with multi-seed execution, rigorous energy analysis,
Poisson vs Bernoulli vs Gaussian encoding justification, and temporal visualizations.
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import torchaudio

from src.diffusion_image import Diffusion
from src.models_image import StrongImgNet, NonSpikeImgNet
from src.models_audio import StrongAudioNet, ContinuousSpikeAudioNet, NonSpikeAudioNet, StrongUNetAudioNet
from src.datasets import AudioDS
from src.train import train_img_model, train_audio_model
from src.evaluate import (
    eval_img_pair, evaluate_audio_pair, evaluate_audio, measure_sparsity, measure_time,
)
from src.energy import profile_energy_audio
from src.visualize import (
    plot_image_samples, plot_audio_waveforms, plot_temporal_denoising, plot_comparison_bars,
)


def parse_args():
    p = argparse.ArgumentParser(description="Neuromorphic Spiking vs Non-Spiking Residual Denoising")
    p.add_argument("--skip-image", action="store_true")
    p.add_argument("--skip-audio", action="store_true")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 2024])
    p.add_argument("--out-dir", type=str, default="outputs")
    p.add_argument("--run-encoding-ablation", action="store_true")
    p.add_argument("--use-continuous-snn", action="store_true")

    # Image branch
    p.add_argument("--batch-size-img", type=int, default=64)
    p.add_argument("--epochs-img", type=int, default=12)
    p.add_argument("--timesteps-img", type=int, default=20)
    p.add_argument("--num-steps-img", type=int, default=5)
    p.add_argument("--base-channels-img", type=int, default=32)
    p.add_argument("--lr-img", type=float, default=2e-4)

    # Audio branch
    p.add_argument("--audio-len", type=int, default=8000)
    p.add_argument("--audio-sr", type=int, default=16000)
    p.add_argument("--timesteps-audio", type=int, default=40)
    p.add_argument("--num-steps-audio", type=int, default=8)
    p.add_argument("--epochs-audio", type=int, default=20)
    p.add_argument("--batch-size-audio", type=int, default=32)
    p.add_argument("--max-rate-audio", type=float, default=0.9)
    p.add_argument("--lr-audio", type=float, default=3e-4)
    p.add_argument("--audio-subset-size", type=int, default=6000)

    return p.parse_args()


def run_audio_branch_multi_seed(args, device):
    print("\n" + "=" * 70)
    print("PART 2: Audio Denoising (Poisson Diffusion Benchmark)")
    print("=" * 70)

    os.makedirs("./data", exist_ok=True)
    base = torchaudio.datasets.SPEECHCOMMANDS("./data", download=True)
    subset = Subset(base, range(min(args.audio_subset_size, len(base))))
    tr_size = int(0.85 * len(subset))
    tr_sub, te_sub = random_split(subset, [tr_size, len(subset) - tr_size])

    train_loader = DataLoader(AudioDS(tr_sub, args.audio_len, args.audio_sr),
                               batch_size=args.batch_size_audio, shuffle=True,
                               num_workers=2, pin_memory=True)
    test_loader = DataLoader(AudioDS(te_sub, args.audio_len, args.audio_sr),
                              batch_size=args.batch_size_audio, shuffle=False,
                              num_workers=2, pin_memory=True)

    if args.use_continuous-snn:
        model_a = ContinuousSpikeAudioNet(num_steps=args.num_steps_audio, T_audio=args.timesteps_audio).to(device)
    else:
        model_a = StrongAudioNet(num_steps=args.num_steps_audio, T_audio=args.timesteps_audio).to(device)
        
    model_ns_a = NonSpikeAudioNet(T_audio=args.timesteps_audio).to(device)
    model_unet_a = StrongUNetAudioNet(T_audio=args.timesteps_audio).to(device)

    model_a = train_audio_model(model_a, "Spiking Audio", train_loader, args.timesteps_audio, args.max_rate_audio, args.epochs_audio, args.lr_audio, device)
    model_ns_a = train_audio_model(model_ns_a, "Non-Spiking Standard ANN", train_loader, args.timesteps_audio, args.max_rate_audio, args.epochs_audio, args.lr_audio, device)
    model_unet_a = train_audio_model(model_unet_a, "Strong 1D U-Net ANN Baseline", train_loader, args.timesteps_audio, args.max_rate_audio, args.epochs_audio, args.lr_audio, device)

    s_sdr_list, ns_sdr_list, unet_sdr_list = [], [], []

    for seed in args.seeds:
        res_s = evaluate_audio(model_a, "Spiking", test_loader, args.timesteps_audio, args.max_rate_audio, device, seed=seed)
        res_ns = evaluate_audio(model_ns_a, "Non-Spiking", test_loader, args.timesteps_audio, args.max_rate_audio, device, seed=seed)
        res_u = evaluate_audio(model_unet_a, "Strong U-Net", test_loader, args.timesteps_audio, args.max_rate_audio, device, seed=seed)
        
        s_sdr_list.append(res_s["SI-SDR Imp"])
        ns_sdr_list.append(res_ns["SI-SDR Imp"])
        unet_sdr_list.append(res_u["SI-SDR Imp"])

    spike_rate, sparsity = measure_sparsity(model_a, test_loader, args.timesteps_audio, args.max_rate_audio, device)
    (t_s, t_s_std), (t_ns, t_ns_std) = measure_time(model_a, model_ns_a, test_loader, args.timesteps_audio, args.max_rate_audio, device)

    # Rigorous Layer-wise Energy Profiling
    sample_input = torch.zeros((1, args.audio_len), device=device)
    sample_t = torch.tensor([args.timesteps_audio // 2], device=device)
    energy_snn = profile_energy_audio(model_a, sample_input, sample_t, spike_rate)
    energy_ann = profile_energy_audio(model_ns_a, sample_input, sample_t, spike_rate=1.0)

    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_audio_waveforms(model_a, test_loader, args.timesteps_audio, args.max_rate_audio, device, fig_dir)
    plot_temporal_denoising(model_a, test_loader, args.timesteps_audio, args.max_rate_audio, device, fig_dir)

    # Optional Encoding Ablation (Poisson vs Gaussian vs Bernoulli)
    if args.run_encoding-ablation:
        print("\n--- Running Encoding Justification Ablation ---")
        for mode in ["gaussian", "bernoulli"]:
            train_audio_model(model_a, f"Spiking-{mode}", train_loader, args.timesteps_audio, args.max_rate_audio, 3, args.lr_audio, device, mode=mode)
            evaluate_audio(model_a, f"Spiking-{mode}", test_loader, args.timesteps_audio, args.max_rate_audio, device, mode=mode)

    return {
        "s_sdr_mean": float(np.mean(s_sdr_list)), "s_sdr_std": float(np.std(s_sdr_list)),
        "ns_sdr_mean": float(np.mean(ns_sdr_list)), "ns_sdr_std": float(np.std(ns_sdr_list)),
        "unet_sdr_mean": float(np.mean(unet_sdr_list)), "unet_sdr_std": float(np.std(unet_sdr_list)),
        "spike_rate": spike_rate, "sparsity": sparsity,
        "t_s": t_s, "t_s_std": t_s_std, "t_ns": t_ns, "t_ns_std": t_ns_std,
        "energy_snn": energy_snn, "energy_ann": energy_ann,
        "res_s": {"SI-SDR Imp": float(np.mean(s_sdr_list)), "MSE": 0.0},
        "res_ns": {"SI-SDR Imp": float(np.mean(ns_sdr_list)), "MSE": 0.0},
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seeds[0])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    os.makedirs(args.out_dir, exist_ok=True)

    audio_results = run_audio_branch_multi_seed(args, device) if not args.skip_audio else None

    if audio_results:
        print("\n" + "=" * 70)
        print("FINAL NEUROMORPHIC BENCHMARK RESULTS")
        print("=" * 70)
        print(f"SNN SI-SDR Imp:  {audio_results['s_sdr_mean']:.2f} ± {audio_results['s_sdr_std']:.2f} dB")
        print(f"ANN SI-SDR Imp:  {audio_results['ns_sdr_mean']:.2f} ± {audio_results['ns_sdr_std']:.2f} dB")
        print(f"U-Net SI-SDR Imp:{audio_results['unet_sdr_mean']:.2f} ± {audio_results['unet_sdr_std']:.2f} dB")
        print("-" * 70)
        print(f"Energy (SNN Accumulate ops): {audio_results['energy_snn']['Energy_mJ']:.4f} mJ per sample")
        print(f"Energy (ANN Multiply-Accumulate ops): {audio_results['energy_ann']['Energy_mJ']:.4f} mJ per sample")
        print(f"Energy Savings: {((audio_results['energy_ann']['Energy_mJ'] - audio_results['energy_snn']['Energy_mJ']) / audio_results['energy_ann']['Energy_mJ']) * 100:.2f}%")
        print("=" * 70)


if __name__ == "__main__":
    main()
