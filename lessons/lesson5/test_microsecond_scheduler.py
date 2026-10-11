"""Unit tests for microsecond-precision EDF/RM simulation (standard library)."""
import unittest
from microsecond_scheduler import Task, simulate, model_execution_us, tasks_for_model


class MicrosecondSchedulerTest(unittest.TestCase):
    def test_rounding_p95_microseconds(self):
        self.assertEqual(model_execution_us({"p95_ms": 0.0504}), 51)
        self.assertEqual(model_execution_us({"p95_ms": 0.1038}), 104)

    def test_finish_at_deadline_is_on_time(self):
        tasks = [Task("a", 80, 200, 80)]
        res = simulate(tasks, "EDF", 1000)
        self.assertEqual(res["missed"], 0)

    def test_tight_deadline_edf_distinguishes_models(self):
        small = tasks_for_model("small", 51, 80)
        large = tasks_for_model("large", 104, 80)
        rsmall = simulate(small, "EDF", 420000)
        rlarge = simulate(large, "EDF", 420000)
        self.assertEqual(rsmall["missed"], 0)
        self.assertGreater(rlarge["missed_by_task"]["T3_large"], 0)

    def test_default_deadline_passes_in_simulated_scenarios(self):
        for cost in (51, 104):
            for algorithm in ("EDF", "RM"):
                r = simulate(tasks_for_model("ml", cost, 12000), algorithm, 420000)
                self.assertEqual(r["missed"], 0)

    def test_basic_rm_priority_and_preemption(self):
        jobs = [Task("high", 2, 5, 5), Task("low", 3, 12, 12)]
        r = simulate(jobs, "RM", 20)
        self.assertEqual(r["timeline"][0], (0, 2, "high"))
        self.assertEqual(r["timeline"][1], (2, 5, "low"))


class ModelSelectorTests(unittest.TestCase):
    def test_accuracy_aware_selection(self):
        from select_model import select_model
        models = {
            "small": {"p95_ms": 0.0504, "test_accuracy": 0.8469},
            "large": {"p95_ms": 0.1038, "test_accuracy": 0.8666},
        }
        self.assertEqual(select_model(models, 80)[0], "small")
        self.assertEqual(select_model(models, 12000)[0], "large")
        self.assertIsNone(select_model(models, 10))


if __name__ == "__main__":
    unittest.main()
