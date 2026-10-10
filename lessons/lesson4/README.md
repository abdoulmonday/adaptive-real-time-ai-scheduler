# Lesson 4 — Trained Fashion-MNIST Models and Scheduling Integration

We compare two simple MLPs on the same data, train/evaluate each one, then separately benchmark **batch-one CPU inference latency**. Training accuracy depends on the experiment; a larger model does not guarantee higher validation accuracy.

## 1. Install requirements

From the repository root:

```bash
python -m pip install -r requirements-ml.txt
python lessons/lesson4/smoke_test.py
```

The smoke test uses synthetic tensors only; it checks tensor shapes and a training step but **does not establish accuracy**.

## 2. Train two networks

```bash
python lessons/lesson4/train_models.py --epochs 2 --batch-size 128
```

The first run downloads Fashion-MNIST to `data/` (requires Internet). Outputs are saved under `outputs/models/` and `outputs/training_summary.json`. Increase `--epochs` after successful validation.

For a fast *pipeline-only* test, use `--limit-train 1000 --limit-test 200 --epochs 1`. These results must **not** be reported as final benchmarks.

## 3. Benchmark trained models

```bash
python lessons/lesson4/benchmark_trained.py --model-dir outputs/models --samples 200 --warmup 30
```

Writes `outputs/benchmarks.json` with inference latency p50/p95/p99, maximum observed time, model parameter counts, device metadata, and prior test-set accuracy.

Only CPU batch size 1 is benchmarked. No real-time guarantee is claimed. Variance, data-loading costs, preprocessing, asynchronous accelerator behavior and CPU contention need additional analysis.

## 4. Explore timing hypotheses in the EDF/RM simulator

```bash
python lessons/lesson4/compare_schedulers.py --benchmarks outputs/benchmarks.json
```

The conversion from empirical p95 to a rounded integer execution budget is a **what-if experiment**, not WCET verification. The displayed scheduling conclusions are conditional on that assumed budget. Use `--min-budget-ms` to explore sensitivity.

## 5. Exercises

1. Repeat the experiment with several random seeds and compare accuracy variation.
2. Increase train epochs and verify any reported accuracy difference on the held-out test set.
3. Compare p50/p95/p99/max and explain why the p95 does not prove schedulability.
4. Test `--min-budget-ms 2` and then `--min-budget-ms 3` to observe the RM response-time change for T3.
5. Document OS, CPU, PyTorch version, batch size, warm-up count, threads and data version in your results.
