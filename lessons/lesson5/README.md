# Lesson 5 — Sub-millisecond scheduling and model selection (first step)

The initial lesson-4 scheduler rounds CPU inference costs to whole **milliseconds** and applies a 2-ms minimum. It consequently gives both trained Fashion-MNIST MLPs identical costs, despite real CPU benchmarks reporting different inference latencies.

This lesson introduces an **event-driven, fully preemptive single-core simulation** using **integer microsecond units**. It does not simulate PyTorch execution live; it represents each inference by a **deterministic cost chosen from empirical p95**, rounded to the nearest higher whole microsecond. **This does not demonstrate a WCET or a real-time guarantee**.

## Run with YOUR benchmark file

From the repository root:

```bash
python lessons/lesson5/microsecond_scheduler.py \
  --benchmarks outputs/benchmarks.json \
  --deadline-us 12000
```

Both models have large slack in the illustrated 12-ms deadline case; EDF and RM should both have zero deadline misses in this model.

Now introduce an **artificially tight 80-microsecond deadline**:

```bash
python lessons/lesson5/microsecond_scheduler.py \
  --benchmarks outputs/benchmarks.json \
  --deadline-us 80
```

With the user-reported measurements, the assumed costs are **51 µs** and **104 µs**, after conversion from p95 in milliseconds. The large model's assumed cost alone exceeds 80 µs. The small model can meet this deadline under EDF in this synthetic workload; RM's static-priority contention can still cause misses.

The synthetic background tasks are `T1=(2000,5000,5000)` and `T2=(2000,7000,7000)` in microseconds. The inference task T3 arrives every 12000 µs. Job release offsets, blocking, non-preemptible GPU kernels, time spent transferring and preparing inputs, thermal effects and context switch overhead are NOT modeled.

An **optional, simple selection heuristic** filters models by p95-derived proxy and keeps the best reported test accuracy that fits the budget:

```bash
python lessons/lesson5/select_model.py --benchmarks outputs/benchmarks.json --budget-us 80
python lessons/lesson5/select_model.py --benchmarks outputs/benchmarks.json --budget-us 12000
```

A p95-derived proxy is not a hard limit, and filtering with this rule doesn't ensure a job meets its deadline once contention is included. This is **not yet an adaptive scheduler running live models**.

## Tests

```bash
python -m unittest discover -s lessons/lesson5 -p 'test_*.py' -v
```

## Example dataset

`docs/example_data/benchmarks_user_reported.json` records the values a participant supplied in the course conversation. It is labeled as user-reported and has no invented p99 or hardware-identifying fields. To reproduce the reported outputs with this snapshot, substitute this sample path for `outputs/benchmarks.json`.

## Next work

- Several training seeds and multiple CPU benchmark repetitions, including confidence intervals and environmental metadata.
- Experiment with quantization and structured pruning; retain end-to-end accuracy and latency measures.
- Separate measured costs from bounds obtained by analysis or instrumentation.
- Compare EDF, RM, non-preemptive tasks, blocking and explicit overhead.
- Test accuracy-aware model selection under changing workloads and real scheduling constraints.
