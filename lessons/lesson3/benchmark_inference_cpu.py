"""TP optionnel : mesure comparative de latence d'inference de 2 reseaux PyTorch.

Ne mesure PAS un WCET ni une precision du modele (poids aleatoires) !
Il s'agit de mesures exploratoires dependantes du PC et de sa charge.

Installation facultative : pip install torch
Execution : python benchmark_inference_cpu.py --samples 150
"""
import argparse
import statistics
import time

import torch
from torch import nn


def percentile(values, pct):
    sorted_data = sorted(values)
    idx = (len(sorted_data) - 1) * pct / 100
    lo = int(idx)
    hi = min(lo + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (idx - lo) * (sorted_data[hi] - sorted_data[lo])


def benchmark(model, x, warmup, samples):
    model.eval()
    times = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        for _ in range(samples):
            before = time.perf_counter_ns()
            model(x)
            after = time.perf_counter_ns()
            times.append((after - before) / 1e6)
    count = sum(p.numel() for p in model.parameters())
    return count, statistics.median(times), percentile(times, 95), max(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=150)
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 2:
        parser.error("--warmup >= 0 et --samples >= 2 sont requis")

    # Stabilise la comparaison, sans rendre le systeme temps reel.
    torch.set_num_threads(1)
    torch.manual_seed(42)
    x = torch.randn(1, 784)  # Un echantillon type MNIST aplati
    models = {
        "petit_MLP": nn.Sequential(nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10)),
        "grand_MLP": nn.Sequential(
            nn.Linear(784, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 10)),
    }
    print("CPU / batch=1 / modeles non entraines (comparaison de latence uniquement)")
    print(f"Iterations : {args.samples} apres {args.warmup} echauffements")
    for name, model in models.items():
        n, p50, p95, maximum = benchmark(model, x, args.warmup, args.samples)
        print(f"{name:12s} : {n:9d} param | p50={p50:.3f} ms | p95={p95:.3f} ms | max={maximum:.3f} ms")
    print("Attention : ni p95 ni maximum observe ne sont des WCET certifies.")


if __name__ == "__main__":
    main()
