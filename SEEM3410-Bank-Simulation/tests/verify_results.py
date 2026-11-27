"""Check the saved commercial run against its manifest and raw tables."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

MAIN_KEYS = ["scenario", "replication", "seed"]
SWEEP_KEYS = ["factor", "level", "replication", "seed"]


def verify(output: Path):
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    for filename, digest in config["source_sha256"].items():
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == digest, filename

    def table(name):
        return pd.read_csv(output / f"{name}.csv")

    main = table("replication_metrics")
    sweep = table("sensitivity_replications")
    for raw, summaries, keys, days, cases in (
        (main, table("scenario_summary"), MAIN_KEYS, config["main_replications"], 8),
        (sweep, table("sensitivity_summary"), SWEEP_KEYS, config["sensitivity_replications"], 21),
    ):
        groups = keys[:-2]
        assert not raw.duplicated(keys[:-1]).any()
        assert not raw.isna().any().any()
        assert not summaries.duplicated(groups).any()
        assert not summaries.isna().any().any()
        assert len(summaries) == len(raw.groupby(groups)) == cases
        assert (raw.groupby(groups).size() == days).all()
        assert (summaries.replications == days).all()
        assert (summaries.filter(regex=r"_n$") == days).all().all()
        assert raw.seed.eq(config["base_seed"] + (raw.replication - 1) * 10007).all()
        assert raw.arrivals.eq(raw.completed_at_close + raw.waiting_at_close + raw.in_service_at_close).all()
        assert raw.arrivals.eq(raw.completed + raw.waiting_at_end + raw.in_service_at_end).all()
        assert raw.unfinished_at_end.eq(raw.waiting_at_end + raw.in_service_at_end).all()
        assert raw.counter_utilization.between(0, 1).all()

    reference = main.loc[main.scenario == config["main_reference"]].drop(columns="scenario").reset_index(drop=True)
    for factor, level in (("arrival_rate_scale", "1.0"), ("service_time_scale", "1.0"), ("counter_number", "4")):
        rows = sweep.loc[sweep.factor.eq(factor) & sweep.level.eq(level)]
        pd.testing.assert_frame_equal(reference, rows.drop(columns=["factor", "level"]).reset_index(drop=True),
                                      check_exact=True)

    peaks = sweep.loc[sweep.factor == "peak_concentration"]
    assert (peaks.groupby("replication").arrivals.nunique() == 1).all()
    expected_volume = config["main_scenarios"][config["main_reference"]]["expected_daily_arrivals"]
    for case in config["sensitivity_cases"]:
        if case["factor"] == "peak_concentration":
            np.testing.assert_allclose(case["expected_daily_arrivals"], expected_volume, rtol=1e-12)

    customers = table("customer_level_results")
    assert not customers.duplicated(MAIN_KEYS + ["customer_id"]).any()
    assert customers.arrival_time.ge(540).all() and customers.arrival_time.lt(1020).all()
    reached = customers.service_start_time.notna()
    left = customers.departure_time.notna()
    assert (customers.waiting_time.notna() == reached).all()
    assert (customers.system_time.notna() == left).all()
    assert left.le(reached).all()
    np.testing.assert_allclose(customers.loc[reached, "waiting_time"],
                               (customers.service_start_time - customers.arrival_time)[reached])
    np.testing.assert_allclose(customers.loc[left, "service_time"],
                               (customers.departure_time - customers.service_start_time)[left])
    close_status = np.select([customers.departure_time.le(1020), customers.service_start_time.lt(1020)],
                             ["completed", "in_service"], default="waiting")
    end_status = np.select([left, reached], ["completed", "in_service"], default="waiting")
    np.testing.assert_array_equal(customers.status_at_close, close_status)
    np.testing.assert_array_equal(customers.status_at_end, end_status)
    customers["queue_minutes_before_close"] = customers.service_start_time.fillna(1020).clip(upper=1020) - customers.arrival_time
    grouped = customers.groupby(MAIN_KEYS)
    main_indexed = main.set_index(MAIN_KEYS).sort_index()
    np.testing.assert_array_equal(grouped.size(), main_indexed.arrivals)
    np.testing.assert_allclose(grouped.queue_minutes_before_close.sum() / 480, main_indexed.time_avg_queue_length)
    np.testing.assert_allclose(grouped.waiting_time.quantile(0.95), main_indexed.p95_wait)
    for cutoff, expected in (("close", close_status), ("end", end_status)):
        by_status = customers.assign(status=expected).groupby(MAIN_KEYS + ["status"]).size().unstack(fill_value=0)
        for status in ("completed", "waiting", "in_service"):
            column = "completed" if (cutoff, status) == ("end", "completed") else f"{status}_at_{cutoff}"
            actual = by_status.get(status, np.zeros(len(by_status)))
            np.testing.assert_array_equal(actual, main_indexed[column])

    for raw, keys, file in ((main, MAIN_KEYS, "counter_level_results"),
                            (sweep, SWEEP_KEYS, "sensitivity_counter_level_results")):
        counters = table(file)
        assert not counters.duplicated(keys + ["counter_id"]).any()
        assert not counters.isna().any().any()
        assert counters.counter_utilization.between(0, 1).all()
        np.testing.assert_allclose(counters.total_service_minutes,
                                   counters.busy_available_minutes + counters.off_shift_service_minutes
                                   + counters.overtime_service_minutes, atol=1e-8)
        sums = counters.groupby(keys).sum(numeric_only=True)
        run = raw.set_index(keys).sort_index()
        np.testing.assert_allclose(sums.available_minutes / 60, run.counter_hours_available)
        np.testing.assert_allclose(sums.busy_available_minutes / sums.available_minutes, run.counter_utilization)
        np.testing.assert_allclose(sums.overtime_service_minutes, run.overtime_service_minutes)

    plots = sorted((output / "plots").glob("*.png"))
    assert len(plots) == 9 and all(p.stat().st_size > 0 for p in plots)
    print(f"PASS: {len(main)} main runs, {len(sweep)} sensitivity runs, {len(customers)} customer rows, {len(plots)} plots.")
    print("PASS: exact C4 baseline matches, peak-volume conservation, run seeds, close/end conservation,")
    print("      customer timestamps and queue area, per-counter reconciliation and source hashes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=Path("results"))
    verify(parser.parse_args().output)
