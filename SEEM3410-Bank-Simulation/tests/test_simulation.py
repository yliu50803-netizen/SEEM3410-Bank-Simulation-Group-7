import unittest
import numpy as np
from dataclasses import replace
from bank_simulation.config import make_scenarios, CounterConfig
from bank_simulation.simulation import BankSimulation, CounterState
from bank_simulation.experiments import _peak_profile_with_conserved_volume


class TestGeneralCounterSimulation(unittest.TestCase):
    def test_general_counter_only_scope(self):
        sim = BankSimulation(make_scenarios()["C4_all_day"], seed=11)
        self.assertTrue(all(not hasattr(c, "customer_class") and not hasattr(c, "channel")
                            for c in sim.customers))

    def test_arrivals_within_business_hours(self):
        sim = BankSimulation(make_scenarios()["C4_all_day"], seed=12)
        self.assertTrue(all(540 <= c.arrival_time < 1020 for c in sim.customers))

    def test_queue_conservation_and_nonnegative_wait(self):
        sim = BankSimulation(make_scenarios()["C4_all_day"], seed=13)
        sim.run()
        metrics = sim.metrics()
        self.assertEqual(metrics["completed"] + metrics["unfinished_at_end"], metrics["arrivals"])
        self.assertTrue(all(c.waiting_time is None or c.waiting_time >= 0 for c in sim.customers))
        self.assertTrue(all(c.departure_time is None or c.departure_time > c.service_start_time
                            for c in sim.customers))

    def test_reproducibility(self):
        cfg = make_scenarios()["C4_all_day"]
        a = BankSimulation(cfg, seed=99); a.run()
        b = BankSimulation(cfg, seed=99); b.run()
        self.assertEqual([c.arrival_time for c in a.customers],
                         [c.arrival_time for c in b.customers])
        self.assertAlmostEqual(a.metrics()["mean_wait"], b.metrics()["mean_wait"])

    def test_capacity_sweep(self):
        scenarios = make_scenarios()
        for n in (3, 4, 5, 6, 7):
            self.assertEqual(len(scenarios[f"C{n}_all_day"].counters), n)

    def test_break_overlap_no_intersection(self):
        state = CounterState(CounterConfig("C1", breaks=((660, 690),)))
        self.assertAlmostEqual(state.scheduled_overlap(600, 630), 30)

    def test_break_overlap_inside_break(self):
        state = CounterState(CounterConfig("C1", breaks=((660, 690),)))
        self.assertAlmostEqual(state.scheduled_overlap(665, 680), 0)

    def test_break_overlap_crossing_break(self):
        state = CounterState(CounterConfig("C1", breaks=((660, 690),)))
        self.assertAlmostEqual(state.scheduled_overlap(650, 700), 20)

    def test_break_minutes_are_clipped_to_shift(self):
        state = CounterState(CounterConfig("C1", open_minute=600, close_minute=900,
                                           breaks=((540, 630), (880, 930))))
        self.assertAlmostEqual(state.available_minutes(), 250)

    def test_peak_profiles_conserve_daily_volume(self):
        base = make_scenarios()["C4_all_day"].arrival_rates
        expected = sum((b-a)*rate for a,b,rate in base)
        for start, end in ((660,900),(720,840),(750,810)):
            profile = _peak_profile_with_conserved_volume(base, start, end)
            actual = sum((b-a)*rate for a,b,rate in profile)
            self.assertTrue(np.isclose(expected, actual))

    def test_utilization_bounded(self):
        sim = BankSimulation(make_scenarios()["C4_all_day"], seed=101)
        sim.run()
        self.assertGreaterEqual(sim.metrics()["counter_utilization"], 0)
        self.assertLessEqual(sim.metrics()["counter_utilization"], 1)


if __name__ == "__main__":
    unittest.main()
