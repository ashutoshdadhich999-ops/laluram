import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
import torchaudio

from src.datasets import AudioDS
from src.diffusion_audio import PoissonAudioDiffusion
from src.models_audio import StrongAudioNet, NonSpikeAudioNet

def evaluate_audio(model, test_loader, diff):
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
            den_signal = den - x0
            sdr_imp = 10 * torch.log10((torch.sum(x0**2) + 1e-8) / (torch.sum(den_signal**2) + 1e-8)).item()
            sdr_imps.append(sdr_imp)
    return float(np.mean(sdr_imps))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 2024])
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running multi-seed experiment on {device}...")

    base = torchaudio.datasets.SPEECHCOMMANDS("./data", download=True)
    subset = Subset(base, range(1500))
    tr_sub, te_sub = random_split(subset, [1200, 300])

    train_loader = DataLoader(AudioDS(tr_sub, 8000, 16000), batch_size=32, shuffle=True)
    test_loader = DataLoader(AudioDS(te_sub, 8000, 16000), batch_size=32, shuffle=False)

    diff = PoissonAudioDiffusion(T=40, device=device)
    
    snn_scores, ann_scores = [], []

    for seed in args.seeds:
        print(f"\n--- Running Seed {seed} ---")
        torch.manual_seed(seed)
        
        snn = StrongAudioNet().to(device)
        ann = NonSpikeAudioNet().to(device)

        opt_snn = torch.optim.AdamW(snn.parameters(), lr=1e-3)
        for ep in range(args.epochs):
            snn.train()
            for x0 in train_loader:
                x0 = x0.to(device)
                t = torch.randint(0, 40, (x0.size(0),), device=device)
                noisy, target = diff.q_sample(x0, t, mode="poisson")
                loss = torch.nn.functional.mse_loss(snn(noisy, t), target)
                opt_snn.zero_grad()
                loss.backward()
                opt_snn.step()

        opt_ann = torch.optim.AdamW(ann.parameters(), lr=1e-3)
        for ep in range(args.epochs):
            ann.train()
            for x0 in train_loader:
                x0 = x0.to(device)
                t = torch.randint(0, 40, (x0.size(0),), device=device)
                noisy, target = diff.q_sample(x0, t, mode="poisson")
                loss = torch.nn.functional.mse_loss(ann(noisy, t), target)
                opt_ann.zero_grad()
                loss.backward()
                opt_ann.step()

        snn_scores.append(evaluate_audio(snn, test_loader, diff))
        ann_scores.append(evaluate_audio(ann, test_loader, diff))

    print("\n" + "=" * 60)
    print("FINAL MULTI-SEED RESULTS")
    print("=" * 60)
    print(f"SNN SI-SDR Imp : {np.mean(snn_scores):.2f} ± {np.std(snn_scores):.2f} dB")
    print(f"ANN SI-SDR Imp : {np.mean(ann_scores):.2f} ± {np.std(ann_scores):.2f} dB")

if __name__ == "__main__":
    main()
