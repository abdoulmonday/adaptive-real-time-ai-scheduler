"""Exécuter : python -m unittest -v test_metrics.py"""
import unittest
from scheduler import simulate
from scheduler_metrics import SCENARIOS, analyze, percentile


class TestMetrics(unittest.TestCase):
    def test_baseline_both_policies_meet_all_deadlines(self):
        for policy in ("EDF", "RM"):
            timeline, jobs = simulate(SCENARIOS["baseline"], policy, horizon=40)
            metrics = analyze(timeline, jobs, horizon=40)
            self.assertEqual(metrics["due"], 22)
            self.assertEqual(metrics["missed"], 0)

    def test_overload_both_policies_miss_four(self):
        for policy in ("EDF", "RM"):
            timeline, jobs = simulate(SCENARIOS["overload"], policy, horizon=40)
            metrics = analyze(timeline, jobs, horizon=40)
            self.assertEqual(metrics["due"], 22)
            self.assertEqual(metrics["missed"], 4)

    def test_overload_misses_different_tasks(self):
        edf_t, edf_j = simulate(SCENARIOS["overload"], "EDF", horizon=40)
        rm_t, rm_j = simulate(SCENARIOS["overload"], "RM", horizon=40)
        edf = analyze(edf_t, edf_j, 40)
        rm = analyze(rm_t, rm_j, 40)
        self.assertEqual(edf["missed_by_task"], {"T1": 2, "T2": 2, "T3": 0})
        self.assertEqual(rm["missed_by_task"], {"T1": 0, "T2": 0, "T3": 4})

    def test_percentile(self):
        self.assertEqual(percentile([1, 2, 3, 4, 5], 50), 3)
        self.assertEqual(percentile([], 95), None)


if __name__ == "__main__":
    unittest.main()
