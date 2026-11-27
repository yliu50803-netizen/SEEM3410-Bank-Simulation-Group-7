from dataclasses import replace
from io import StringIO
import unittest

import numpy as np
import pandas as pd

from bank_simulation.config import make_scenarios
from bank_simulation.experiments import (
    METRICS, REFERENCE, _paired_days, capacity_margins, estimate, run_experiments,
    run_sensitivity, sensitivity_cases, sensitivity_comparisons,
)
from bank_simulation.simulation import BankSimulation


class TestMatchedExperiments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main, cls.main_summary, cls.customers, cls.pairs = run_experiments(replications=3)
        cls.sensitivity, cls.sensitivity_summary = run_sensitivity(replications=3)

    def test_replication_outputs_with_csv_roundtrip(self):
        cases = [
            (self.main, ["scenario"], 8, 24),
            (self.sensitivity, ["factor", "level"], 21, 63),
        ]
        for frame, labels, groups, rows in cases:
            with self.subTest(keys=labels):
                reread = pd.read_csv(StringIO(frame.to_csv(index=False)))
                self.assertEqual(len(reread), rows)
                self.assertFalse(reread.isna().any().any())
                self.assertFalse(reread.duplicated(labels + ["replication"]).any())
                sizes = reread.groupby(labels).size()
                self.assertEqual(len(sizes), groups)
                self.assertTrue((sizes == 3).all())
                self.assertEqual(set(reread.columns), set(labels + ["seed", "replication", *METRICS]))
                self.assertTrue((reread.groupby("replication").seed.nunique() == 1).all())

    def test_summary_contains_sd_ci_and_sample_size(self):
        for frame in (self.main_summary, self.sensitivity_summary):
            self.assertFalse(frame.isna().any().any())
            for metric in METRICS:
                self.assertTrue((frame[f"{metric}_n"] == 3).all())
                self.assertTrue((frame[f"{metric}_sd"] >= 0).all())
                self.assertTrue((frame[f"{metric}_ci95_low"] <= frame[f"{metric}_mean"]).all())

    def test_identity_levels_match_main_scenario_exactly(self):
        base = make_scenarios()[REFERENCE]
        identity = [("arrival_rate_scale", 1.0), ("service_time_scale", 1.0), ("counter_number", 4)]
        by_level = {(factor, level): config for factor, level, config in sensitivity_cases()}
        expected = self.main.loc[self.main.scenario == REFERENCE].drop(columns="scenario").reset_index(drop=True)
        for factor, level in identity:
            with self.subTest(factor=factor):
                self.assertEqual(by_level[factor, level], base)
                actual = self.sensitivity.loc[
                    (self.sensitivity.factor == factor) & (self.sensitivity.level == level)
                ].drop(columns=["factor", "level"]).reset_index(drop=True)
                pd.testing.assert_frame_equal(expected, actual)

    def test_unchanged_arrival_and_service_inputs_across_capacity(self):
        base = make_scenarios()[REFERENCE]
        first = BankSimulation(base, seed=671)
        last = BankSimulation(make_scenarios()["C7_all_day"], seed=671)
        self.assertEqual(first.customers, last.customers)
        first.run()
        same = BankSimulation(base, seed=671)
        same.run()
        self.assertEqual(first.customers, same.customers)
        self.assertEqual(first.metrics(), same.metrics())

    def test_demand_scales_reuse_the_poisson_clock(self):
        cases = {level: cfg for factor, level, cfg in sensitivity_cases() if factor == "arrival_rate_scale"}
        low = BankSimulation(cases[0.8], seed=611)
        high = BankSimulation(cases[1.2], seed=611)
        self.assertLess(len(low.customers), len(high.customers))
        for first, second in zip(low.customers, high.customers):
            self.assertLessEqual(second.arrival_time, first.arrival_time)
            self.assertEqual(second.service_time, first.service_time)

    def test_service_scales_reuse_arrivals_and_customer_percentiles(self):
        base = make_scenarios()[REFERENCE]
        slower = replace(base, service_mean=9.6, service_sd=3.6)
        a, b = BankSimulation(base, seed=11), BankSimulation(slower, seed=11)
        self.assertEqual([c.arrival_time for c in a.customers], [c.arrival_time for c in b.customers])
        np.testing.assert_allclose([1.2 * c.service_time for c in a.customers],
                                   [c.service_time for c in b.customers], rtol=1e-12)

    def test_peak_levels_keep_expected_and_matched_daily_counts(self):
        for profile in ("commercial", "mall_residential"):
            base = make_scenarios(profile)[REFERENCE]
            for factor, _, cfg in sensitivity_cases(profile):
                if factor == "peak_concentration":
                    self.assertAlmostEqual(cfg.expected_daily_arrivals, base.expected_daily_arrivals)
        peak = self.sensitivity.loc[self.sensitivity.factor == "peak_concentration"]
        self.assertTrue((peak.groupby("replication").arrivals.nunique() == 1).all())

    def test_paired_differences_match_raw_and_reject_wrong_seeds(self):
        candidate = self.main.loc[self.main.scenario == "C5_all_day"]
        base = self.main.loc[self.main.scenario == REFERENCE]
        difference = _paired_days(candidate, base)
        result = self.pairs.loc[(self.pairs.scenario == "C5_all_day") & (self.pairs.metric == "mean_wait")].iloc[0]
        self.assertAlmostEqual(result.mean_difference_vs_C4_reference, difference["mean_wait"].mean())
        self.assertAlmostEqual(result.sd, difference["mean_wait"].std(ddof=1))
        with self.assertRaisesRegex(ValueError, "matching replication"):
            _paired_days(candidate.assign(seed=candidate.seed + 1), base)
        with self.assertRaisesRegex(ValueError, "one row per replication"):
            _paired_days(pd.concat([candidate, candidate]), base)

    def test_margins_and_factor_pairs_have_correct_group_counts(self):
        margins, pairs = capacity_margins(self.sensitivity), sensitivity_comparisons(self.sensitivity)
        self.assertTrue((margins.additional_counter_hours == 8).all())
        self.assertEqual(list(margins.to_counters), [4, 5, 6, 7])
        self.assertEqual(len(pairs), (21 - 5) * len(METRICS))
        self.assertTrue((pairs.replications == 3).all())

    def test_student_t_interval_matches_hand_calculation(self):
        measured = estimate(pd.Series([1.0, 2.0, 3.0]))
        expected_radius = 4.30265273 / np.sqrt(3)
        self.assertAlmostEqual(measured["mean"], 2.0)
        self.assertAlmostEqual(measured["sd"], 1.0)
        self.assertAlmostEqual(measured["ci95_low"], 2.0 - expected_radius)
        self.assertAlmostEqual(measured["ci95_high"], 2.0 + expected_radius)

    def test_customer_rows_reconcile_with_day_counts(self):
        self.assertFalse(self.customers.duplicated(["scenario", "replication", "customer_id"]).any())
        counts = self.customers.groupby(["scenario", "replication"]).size()
        np.testing.assert_array_equal(counts, self.main.set_index(["scenario", "replication"]).sort_index().arrivals)


if __name__ == "__main__":
    unittest.main()
