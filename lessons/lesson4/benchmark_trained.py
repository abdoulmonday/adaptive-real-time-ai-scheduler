"""Benchmark trained CPU MLPs. Empirical quantiles are NOT WCET guarantees."""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from models import make_model


def percentile(values, q):
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q / 100
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (idx - lo) * (ordered[hi] - ordered[lo])


def measure(model, input_tensor, warmup, samples):
    model.eval()
    durations = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(input_tensor)
        for _ in range(samples):
            start = time.perf_counter_ns()
            model(input_tensor)
            durations.append((time.perf_counter_ns() - start) / 1e6)
    return durations


def main():
    parser = argparse.ArgumentParser(description='Benchmark batch-1 CPU inference')
    parser.add_argument('--model-dir', type=Path, default=Path('outputs/models'))
    parser.add_argument('--summary', type=Path, default=Path('outputs/training_summary.json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/benchmarks.json'))
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.samples < 2 or args.warmup < 0 or args.threads <= 0:
        parser.error('samples>=2, warmup>=0, threads>=1 required')
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    training_summary = json.loads(args.summary.read_text(encoding='utf-8'))
    # CPU, batch 1, fixed 28x28 input; random input isolates forward-pass performance.
    image = torch.randn(1, 1, 28, 28)
    results = {
        'device': 'cpu', 'batch_size': 1, 'input_shape': [1, 1, 28, 28],
        'samples': args.samples, 'warmup': args.warmup, 'threads': args.threads,
        'torch_version': torch.__version__, 'python_version': platform.python_version(),
        'platform': platform.platform(), 'processor': platform.processor(),
        'timing_unit': 'ms', 'warning': 'Observed quantiles and maxima are NOT verified WCET bounds',
        'models': {},
    }
    for name in ('small', 'large'):
        model = make_model(name)
        checkpoint = args.model_dir / f'{name}.pt'
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        ms = measure(model, image, args.warmup, args.samples)
        observed = {
            'parameters': sum(p.numel() for p in model.parameters()),
            'test_accuracy': training_summary['models'][name]['test_accuracy'],
            'mean_ms': statistics.mean(ms), 'p50_ms': percentile(ms, 50),
            'p95_ms': percentile(ms, 95), 'p99_ms': percentile(ms, 99), 'max_observed_ms': max(ms),
        }
        results['models'][name] = observed
        print(f"{name:6s}: acc={observed['test_accuracy']:.2%} p50={observed['p50_ms']:.4f} ms p95={observed['p95_ms']:.4f} ms max={observed['max_observed_ms']:.4f} ms")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print('Saved:', args.output)
    print('IMPORTANT: measured p95 / observed max are not hard real-time guarantees.')


if __name__ == '__main__':
    main()
