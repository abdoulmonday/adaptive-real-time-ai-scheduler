"""Exploratory workload integration: measured p95 != certified WCET."""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmarks', type=Path, default=Path('outputs/benchmarks.json'))
    parser.add_argument('--min-budget-ms', type=int, default=1, help='Artificial minimum CPU cost for sensitivity experiments')
    parser.add_argument('--horizon-ms', type=int, default=420)
    args = parser.parse_args()
    if args.min_budget_ms < 1 or args.horizon_ms < 1:
        parser.error('Budgets and horizon must be >=1')
    # The scheduling analysis is kept in its own lesson to preserve the course progression.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'lesson3'))
    from analyse_ordonnancabilite import Task, simulate, utilization, response_time_analysis

    measured = json.loads(args.benchmarks.read_text(encoding='utf-8'))
    print('EXPLORATORY ASSUMPTION: integer-ms budget is NOT a verified WCET')
    print('Models: actual test accuracy and empirical CPU latencies; other task costs are illustrative.')
    for name, stats in measured['models'].items():
        # Conservative rounding of measured p95 within this *assumed* model; not a proof.
        c3 = max(args.min_budget_ms, math.ceil(stats['p95_ms']))
        tasks = [Task('T1',2,5,5), Task('T2',2,7,7), Task(f'T3_{name}',c3,12,12)]
        print(f'\n{name}: test accuracy={stats["test_accuracy"]:.2%}; measured p95={stats["p95_ms"]:.4f} ms')
        print(f'    Assumed C3={c3} ms; utilization={utilization(tasks):.2%}')
        _, feasible = response_time_analysis(tasks[-1], tasks)
        print(f'    RM response-time test for T3: {"PASS" if feasible else "FAIL"} (assuming C3)')
        for policy in ('EDF','RM'):
            timeline,jobs,due,missed = simulate(tasks,policy,args.horizon_ms)
            print(f'    {policy} observed misses over {args.horizon_ms}ms: {len(missed)}/{len(due)}')
    print('\nThese results are conditional on assumed costs, not hard real-time guarantees.')


if __name__ == '__main__':
    main()
