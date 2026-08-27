import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
import torchaudio

from src.datasets import AudioDS
from src.diffusion_audio import PoissonAudioDiffusion
from src.models_audio import ContinuousSpikingAudioNet, ResidualDilatedTCN
from src.energy_analysis import NeuromorphicEnergyAnalyzer
from src.visualize import plot_temporal_denoiser_behavior
from src.evaluate import evaluate_audio_pair, measure_sparsity

def main():
    parser = argparse.ArgumentParser(description="PhD Final Multi-Seed Pipeline Execution")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 2024], help="Multi-seed run")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running pipeline on Device: {device} across Seeds: {args.seeds}")

    os.makedirs("outputs/figures", exist_ok=True)
    base = torchaudio.datasets.SPEECHCOMMANDS("./data", download=True)
    subset = Subset(base, range(3000))
    tr_sub, te_sub = random_split(subset, [2500, 500])

    train_loader = DataLoader(AudioDS(tr_sub, 8000, 16000), batch_size=32, shuffle=True)
    test_loader = DataLoader(AudioDS(te_sub, 8000, 16000), batch_size=32, shuffle=False)

    diff = PoissonAudioDiffusion(T=40, device=device)
    
    snn_sdr_improvements = []
    ann_sdr_improvements = []

    for seed in args.seeds:
        print("\n" + "="*60)
        print(f"RUNNING EXPERIMENT FOR SEED {seed}")
        print("="*60)
        torch.manual_seed(seed)

        snn_model = ContinuousSpikingAudioNet().to(device)
        ann_model = ResidualDilatedTCN().to(device)

        # Train SNN
        opt_snn = torch.optim.AdamW(snn_model.parameters(), lr=1e-3)
        for ep in range(args.epochs):
            snn_model.train()
            for x0 in train_loader:
                x0 = x0.to(device)
                t = torch.randint(0, 40, (x0.size(0),), device=device)
                noisy, target = diff.q_sample(x0, t, mode="poisson")
                loss = torch.nn.functional.mse_loss(snn_model(noisy, t), target)
                opt_snn.zero_grad()
                loss.backward()
                opt_snn.step()

        # Train ANN Baseline
        opt_ann = torch.optim.AdamW(ann_model.parameters(), lr=1e-3)
        for ep in range(args.epochs):
            ann_model.train()
            for x0 in train_loader:
                x0 = x0.to(device)
                t = torch.randint(0, 40, (x0.size(0),), device=device)
                noisy, target = diff.q_sample(x0, t, mode="poisson")
                loss = torch.nn.functional.mse_loss(ann_model(noisy, t), target)
                opt_ann.zero_grad()
                loss.backward()
                opt_ann.step()

        # Evaluate SNN vs Strong ANN
        res_snn = evaluate_audio(snn_model, "SNN Audio", test_loader, diff)
        res_ann = evaluate_audio(ann_model, "ANN TCN Baseline", test_loader, diff)

        snn_sdr_improvements.append(res_snn["SI-SDR Imp"])
        ann_sdr_improvements.append(res_ann["SI-SDR Imp"])

    # Multi-seed Statistical Summary
    print("\n" + "="*70)
    print("FINAL MULTI-SEED STATISTICAL RESULTS")
    print("="*70)
    print(f"SNN SI-SDR Improvement : {np.mean(snn_sdr_improvements):.2f} ± {np.std(snn_sdr_improvements):.2f} dB")
    print(f"ANN SI-SDR Improvement : {np.mean(ann_sdr_improvements):.2f} ± {np.std(ann_sdr_improvements):.2f} dB")

    # Neuromorphic Rigorous Energy Calculation
    print("\n" + "="*70)
    print("UPGRADE 2: HARDWARE NEUROMORPHIC ENERGY ANALYSIS")
    print("="*70)
    sample_batch = next(iter(test_loader)).to(device)
    rate, _ = measure_sparsity(snn_model, test_loader, 40, 0.9, device)
    
    analyzer = NeuromorphicEnergyAnalyzer(snn_model)
    energy_stats = analyzer.compute_energy(sample_batch, mean_spike_rate=rate)
    
    for metric, val in energy_stats.items():
        if isinstance(val, float):
            print(f"{metric:<30}: {val:.4f}")
        else:
            print(f"{metric:<30}: {val}")

    # Plot Temporal Behavior Figure
    print("\nGenerating Temporal Behavior Visuals...")
    sample_audio = sample_batch[0]
    plot_temporal_denoiser_behavior(snn_model, sample_audio, diff, "outputs/figures")
    print("Done! Visuals saved to outputs/figures/temporal_denoising_proof.png")

def evaluate_audio(model, name, test_loader, diff):
    model.eval()
    sdr_imps = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for x0 in test_loader:
            x0 = x0.to(device)
            t = torch.randint(10, 40, (x0.size(0),), device=device)
            noisy, _ = diff.q_sample(x0, t, mode="poisson")
            pred = model(noisy, t)
            den = torch.clamp(noisy - pred, 0.0, 1.0)
            
            # Simple SDR proxy calculation
            noise_signal = noisy - x0
            den_signal = den - x0
            sdr_imp = 10 * torch.log10((torch.sum(x0**2) + 1e-8) / (torch.sum(den_signal**2) + 1e-8)).item()
            sdr_imps.append(sdr_imp)
    return {"SI-SDR Imp": float(np.mean(sdr_imps))}

if __name__ == "__main__":
    main()