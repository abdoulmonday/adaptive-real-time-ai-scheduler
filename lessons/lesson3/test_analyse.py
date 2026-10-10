import unittest
from analyse_ordonnancabilite import SCENARIOS, dbf, utilization, simulate, response_time_analysis


class Tests(unittest.TestCase):
    def test_edf_short_deadline_demand(self):
        tasks = SCENARIOS["deadlines_courtes"]
        self.assertAlmostEqual(utilization(tasks), 3/10 + 2/12)
        self.assertEqual(sum(dbf(t, 4) for t in tasks), 5)
        _, _, _, misses = simulate(tasks, "EDF", 4)
        self.assertEqual(len(misses), 1)

    def test_rta_initial(self):
        tasks = SCENARIOS["initial"]
        results = {t.name: response_time_analysis(t, tasks) for t in tasks}
        self.assertEqual(results["T1"][0], [2, 2])
        self.assertEqual(results["T2"][0], [2, 4, 4])
        self.assertEqual(results["T3"][0], [3, 7, 9, 11, 13])
        self.assertFalse(results["T3"][1])

    def test_rta_improvements(self):
        tasks = SCENARIOS["modele_leger"]
        steps, possible = response_time_analysis(tasks[-1], tasks)
        self.assertEqual(steps, [2, 6, 8, 10, 10])
        self.assertTrue(possible)
        tasks = SCENARIOS["periode_allongee"]
        steps, possible = response_time_analysis(tasks[-1], tasks)
        self.assertEqual(steps, [3, 7, 9, 11, 11])
        self.assertTrue(possible)

    def test_simulation_initial(self):
        tasks = SCENARIOS["initial"]
        edf, _, _, edf_miss = simulate(tasks, "EDF", 420)
        rm, _, _, rm_miss = simulate(tasks, "RM", 420)
        self.assertEqual(len(edf_miss), 0)
        self.assertEqual(len(rm_miss), 2)
        self.assertEqual(edf[:13], ["T1", "T1", "T2", "T2", "T3", "T1", "T1", "T3", "T3", "T2", "T2", "T1", "T1"])
        self.assertEqual(rm[:13], ["T1", "T1", "T2", "T2", "T3", "T1", "T1", "T2", "T2", "T3", "T1", "T1", "T3"])


if __name__ == "__main__":
    unittest.main()
