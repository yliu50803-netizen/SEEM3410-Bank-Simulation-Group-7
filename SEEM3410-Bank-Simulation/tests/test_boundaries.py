from dataclasses import replace
import unittest

import numpy as np
from scipy.stats import gamma

from bank_simulation.config import CounterConfig, make_scenarios
from bank_simulation.simulation import (
    BankSimulation, CounterState, Customer, gamma_parameters, sample_service_times,
)

BASE = make_scenarios()["C4_all_day"]


def run_day(customers, *counters):
    config = replace(BASE, counters=counters or (CounterConfig("A"),))
    sim = BankSimulation(config, arrivals=[Customer(*row) for row in customers])
    sim.run()
    return sim


class TestDayBoundaries(unittest.TestCase):
    def test_opening_is_inclusive_and_closing_is_exclusive_for_arrivals(self):
        sim = run_day([(1, 540, 5)])
        self.assertEqual(sim.customers[0].service_start_time, 540)
        for bad_arrival in (539, 1020, 1021):
            with self.subTest(arrival=bad_arrival), self.assertRaisesRegex(ValueError, "excluding 17:00"):
                run_day([(1, bad_arrival, 5)])

    def test_fifo_uses_arrival_timestamp_before_id_and_id_for_same_time(self):
        sim = run_day([(1, 546, 2), (70, 543, 2), (60, 543, 2), (30, 540, 10)])
        self.assertEqual([c.customer_id for c in sorted(sim.customers, key=lambda c: c.service_start_time)],
                         [30, 60, 70, 1])

    def test_waiter_goes_before_new_arrival_when_service_finishes(self):
        sim = run_day([(10, 540, 10), (99, 545, 2), (1, 550, 2)])
        self.assertEqual([c.service_start_time for c in sim.customers], [540, 550, 552])

    def test_departures_at_same_time_free_both_counters_before_dispatch(self):
        sim = run_day([(10, 540, 10), (20, 540, 2), (30, 542, 8), (40, 548, 2), (50, 549, 2)],
                      CounterConfig("A"), CounterConfig("B"))
        self.assertEqual([(c.service_start_time, c.counter_id) for c in sim.customers[3:]],
                         [(550, "A"), (550, "B")])

    def test_new_shift_takes_waiter_at_open_but_stops_new_work_at_its_close(self):
        sim = run_day([(1, 545, 8), (2, 565, 10), (3, 570, 5)], CounterConfig("Temp", 550, 570))
        first, second, third = sim.customers
        self.assertEqual(first.service_start_time, 550)
        self.assertEqual(second.departure_time, 575)
        self.assertIsNone(third.service_start_time)
        self.assertEqual(sim.metrics()["off_shift_service_minutes"], 5)

    def test_break_boundaries_and_service_allowed_to_cross_break(self):
        sim = run_day([(1, 545, 10), (2, 550, 4), (3, 560, 5)],
                      CounterConfig("A", breaks=((550, 560),)))
        first, second, third = sim.customers
        self.assertEqual(first.departure_time, 555)
        self.assertEqual(second.service_start_time, 560)
        self.assertEqual(third.service_start_time, 564)
        self.assertEqual(sim.metrics()["off_shift_service_minutes"], 5)

    def test_departure_exactly_at_break_start_does_not_refill_counter(self):
        sim = run_day([(1, 540, 10), (2, 541, 3)], CounterConfig("A", breaks=((550, 560),)))
        self.assertEqual(sim.customers[1].service_start_time, 560)

    def test_overlapping_breaks_are_charged_once(self):
        state = CounterState(CounterConfig("A", 600, 700, ((625, 645), (590, 630), (680, 710))))
        self.assertEqual(state.config.breaks, ((590, 645), (680, 710)))
        self.assertEqual(state.available_minutes(), 35)
        self.assertEqual(state.scheduled_overlap(630, 690), 35)

    def test_close_snapshot_and_clearance_are_separate(self):
        sim = run_day([(1, 1010, 10), (2, 1011, 20), (3, 1012, 5)],
                      CounterConfig("A"), CounterConfig("B"))
        m = sim.metrics()
        self.assertEqual([c.status_at(1020) for c in sim.customers], ["completed", "in_service", "waiting"])
        self.assertEqual(sim.customers[2].service_start_time, 1020)
        self.assertEqual(m["arrivals"], m["completed_at_close"] + m["waiting_at_close"] + m["in_service_at_close"])
        self.assertEqual(m["completed"], 3)
        self.assertEqual(m["unfinished_at_end"], 0)
        self.assertEqual(m["overtime_service_minutes"], 16)
        self.assertAlmostEqual(m["time_avg_queue_length"], 8 / 480)

    def test_clearance_can_resume_when_a_break_ending_after_close_is_over(self):
        sim = run_day([(1, 1018, 2)], CounterConfig("A", breaks=((1015, 1025),)))
        self.assertEqual(sim.customers[0].service_start_time, 1025)
        self.assertEqual(sim.metrics()["waiting_at_close"], 1)

    def test_unserved_customers_are_kept_and_do_not_enter_wait_statistics(self):
        sim = run_day([(1, 995, 10), (2, 997, 20), (3, 999, 5)], CounterConfig("Temp", 990, 1000))
        m = sim.metrics()
        self.assertEqual((m["arrivals"], m["served_started"], m["completed"], m["waiting_at_end"]),
                         (3, 1, 1, 2))
        self.assertEqual(m["mean_wait"], 0)
        self.assertIsNone(sim.customers[1].as_dict()["waiting_time"])
        self.assertEqual(sim.customers[1].as_dict()["status_at_end"], "waiting")
        self.assertAlmostEqual(m["time_avg_queue_length"], (23 + 21) / 480)

    def test_no_service_means_wait_statistics_are_undefined(self):
        sim = run_day([(1, 999, 5)], CounterConfig("Temp", 540, 900))
        m = sim.metrics()
        self.assertTrue(np.isnan(m["mean_wait"]))
        self.assertTrue(np.isnan(m["p_wait_gt_20"]))
        self.assertEqual(m["waiting_at_end"], 1)

    def test_metrics_follow_hand_calculation(self):
        sim = run_day([(3, 545, 10), (2, 540, 30), (4, 550, 10), (5, 562, 10)])
        m = sim.metrics()
        self.assertAlmostEqual(m["mean_wait"], 20.75)
        self.assertAlmostEqual(m["p95_wait"], 29.7)
        self.assertEqual(m["p_wait_gt_20"], 0.75)
        self.assertEqual(m["p_wait_gt_30"], 0)
        self.assertEqual(m["mean_service_time"], 15)
        self.assertEqual(m["mean_system_time"], 35.75)
        self.assertEqual(m["max_queue_length"], 3)
        self.assertAlmostEqual(m["time_avg_queue_length"], 83 / 480)
        self.assertAlmostEqual(m["counter_utilization"], 60 / 480)
        self.assertAlmostEqual(sim.counter_metrics()[0]["counter_utilization"], 60 / 480)

    def test_invalid_config_inputs_are_rejected_before_sampling(self):
        invalid = [
            {"arrival_rates": ((610, 650, 5), (640, 680, 4))},
            {"arrival_rates": ((530, 650, 5),)},
            {"arrival_rates": ((540, 650, -5),)},
            {"arrival_rates": ((540, 650, float("nan")),)},
            {"service_mean": 0}, {"service_sd": float("inf")},
            {"counters": (CounterConfig("A"), CounterConfig("A"))},
        ]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(BASE, **changes)

    def test_gamma_sampler_matches_moments_and_quantiles(self):
        data = sample_service_times(np.random.default_rng(924), 8, 3, 50_000)
        shape, scale = gamma_parameters(8, 3)
        self.assertTrue(np.all(data > 0))
        self.assertAlmostEqual(data.mean(), 8, delta=0.05)
        self.assertAlmostEqual(data.std(ddof=1), 3, delta=0.05)
        ps = [0.05, 0.5, 0.95]
        np.testing.assert_allclose(np.quantile(data, ps), gamma.ppf(ps, shape, scale=scale), atol=0.12)


if __name__ == "__main__":
    unittest.main()
