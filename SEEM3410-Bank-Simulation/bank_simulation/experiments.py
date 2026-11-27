"""Matched simulation days for the main comparison and five OFAT sweeps."""

from dataclasses import replace
from functools import lru_cache
import math

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .config import CLOSE_MINUTE, OPEN_MINUTE, ScenarioConfig, make_scenarios
from .simulation import BankSimulation

REFERENCE = "C4_all_day"
DEFAULT_SEED = 20261124
PEAK_WINDOWS = {"wide": (660, 900), "medium": (720, 840), "narrow": (750, 810)}
FACTOR_REFERENCES = {
    "arrival_rate_scale": 1.0,
    "service_time_scale": 1.0,
    "counter_number": 4,
    "peak_concentration": "medium",
    "break_schedule_offset": 0,
}
METRICS = (
    "arrivals", "served_started", "completed", "completed_at_close", "waiting_at_close",
    "in_service_at_close", "waiting_at_end", "in_service_at_end", "unfinished_at_end",
    "mean_wait", "median_wait", "p90_wait", "p95_wait", "p_wait_gt_20", "p_wait_gt_30",
    "mean_service_time", "mean_system_time", "time_avg_queue_length", "max_queue_length",
    "counter_utilization", "counter_hours_available", "counter_hours_busy",
    "overtime_service_minutes", "off_shift_service_minutes",
)
REPLICATION_KEYS = ["replication", "seed"]


@lru_cache(maxsize=32)
def _t_critical(n: int) -> float:
    return float(student_t.ppf(0.975, df=n - 1))


def estimate(values: pd.Series) -> dict:
    """Mean and pointwise Student's t interval across independent days."""
    sample = values.dropna().to_numpy(dtype=float)
    n = len(sample)
    mean = float(sample.mean()) if n else np.nan
    sd = float(sample.std(ddof=1)) if n >= 2 else np.nan
    half_width = _t_critical(n) * sd / math.sqrt(n) if n >= 2 else np.nan
    return {"mean": mean, "sd": sd, "ci95_low": mean - half_width,
            "ci95_high": mean + half_width, "n": n}


def _summarise(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for metric in METRICS:
            row.update({f"{metric}_{field}": value for field, value in estimate(group[metric]).items()})
        rows.append({**row, "replications": len(group)})
    return pd.DataFrame(rows)


def _seeds(replications: int, base_seed: int):
    if replications < 2:
        raise ValueError("Use at least 2 replications to estimate uncertainty")
    if base_seed < 0:
        raise ValueError("Seed must be non-negative")
    for day in range(1, replications + 1):
        yield day, base_seed + (day - 1) * 10007


def _paired_days(candidate: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    candidate = candidate.set_index(REPLICATION_KEYS).sort_index()
    reference = reference.set_index(REPLICATION_KEYS).sort_index()
    if not (candidate.index.is_unique and reference.index.is_unique):
        raise ValueError("Each paired case must have one row per replication and seed")
    if not candidate.index.equals(reference.index):
        raise ValueError("Paired comparisons need matching replication numbers and seeds")
    return candidate[list(METRICS)] - reference[list(METRICS)]


def _paired_rows(candidate, reference, fields, mean_label) -> list[dict]:
    differences = _paired_days(candidate, reference)
    rows = []
    for metric in METRICS:
        result = estimate(differences[metric])
        result[mean_label] = result.pop("mean")
        result["replications"] = result.pop("n")
        rows.append({**fields, "metric": metric, **result})
    return rows


def run_experiments(branch_profile="commercial", replications=30, base_seed=DEFAULT_SEED,
                    *, counter_rows: list | None = None):
    scenarios = make_scenarios(branch_profile)
    rows, customer_rows = [], []
    for day, seed in _seeds(replications, base_seed):
        for scenario, config in scenarios.items():
            meta = {"scenario": scenario, "replication": day, "seed": seed}
            sim = BankSimulation(config, replication=day, seed=seed)
            sim.run()
            rows.append({**meta, **sim.metrics()})
            customer_rows.extend({**meta, **customer.as_dict()} for customer in sim.customers)
            if counter_rows is not None:
                counter_rows.extend({**meta, **counter} for counter in sim.counter_metrics())

    raw = pd.DataFrame(rows)
    baseline = raw.loc[raw.scenario == REFERENCE]
    paired = []
    for scenario, candidate in raw.groupby("scenario", sort=False):
        if scenario != REFERENCE:
            paired.extend(_paired_rows(candidate, baseline, {"scenario": scenario, "reference": REFERENCE},
                                       "mean_difference_vs_C4_reference"))
    return raw, _summarise(raw, ["scenario"]), pd.DataFrame(customer_rows), pd.DataFrame(paired)


def _peak_profile_with_conserved_volume(base_rates, peak_start, peak_end, peak_to_shoulder=2.0):
    """Build a two-rate shape with the baseline's expected daily arrivals."""
    if not OPEN_MINUTE < peak_start < peak_end < CLOSE_MINUTE:
        raise ValueError("Peak window must be inside the business day")
    if not math.isfinite(peak_to_shoulder) or peak_to_shoulder <= 0:
        raise ValueError("Peak-to-shoulder ratio must be finite and positive")
    minute_rate_total = sum((end - start) * rate for start, end, rate in base_rates)
    peak_minutes = peak_end - peak_start
    shoulder_minutes = CLOSE_MINUTE - OPEN_MINUTE - peak_minutes
    shoulder_rate = minute_rate_total / (shoulder_minutes + peak_to_shoulder * peak_minutes)
    return ((OPEN_MINUTE, peak_start, shoulder_rate),
            (peak_start, peak_end, peak_to_shoulder * shoulder_rate),
            (peak_end, CLOSE_MINUTE, shoulder_rate))


def sensitivity_cases(branch_profile="commercial") -> list[tuple[str, int | float | str, ScenarioConfig]]:
    scenarios = make_scenarios(branch_profile)
    base = scenarios[REFERENCE]
    scales = (0.8, 0.9, 1.0, 1.1, 1.2)
    cases = []
    for scale in scales:
        rates = tuple((start, end, rate * scale) for start, end, rate in base.arrival_rates)
        cases.append(("arrival_rate_scale", scale, replace(base, arrival_rates=rates)))
    for scale in scales:
        cases.append(("service_time_scale", scale, replace(
            base, service_mean=base.service_mean * scale, service_sd=base.service_sd * scale
        )))
    for number in (3, 4, 5, 6, 7):
        cases.append(("counter_number", number, scenarios[f"C{number}_all_day"]))
    for level, window in PEAK_WINDOWS.items():
        profile = _peak_profile_with_conserved_volume(base.arrival_rates, *window)
        cases.append(("peak_concentration", level, replace(base, arrival_rates=profile)))
    for offset in (0, 30, 60):
        counters = []
        for i, counter in enumerate(base.counters):
            break_start = 660 + offset + i * 15
            counters.append(replace(counter, breaks=((break_start, break_start + 30),)))
        cases.append(("break_schedule_offset", offset, replace(base, counters=tuple(counters))))
    return cases


def run_sensitivity(branch_profile="commercial", replications=30, base_seed=DEFAULT_SEED,
                    *, counter_rows: list | None = None):
    rows = []
    days = list(_seeds(replications, base_seed))
    for factor, level, config in sensitivity_cases(branch_profile):
        for day, seed in days:
            meta = {"factor": factor, "level": level, "replication": day, "seed": seed}
            sim = BankSimulation(config, replication=day, seed=seed)
            sim.run()
            rows.append({**meta, **sim.metrics()})
            if counter_rows is not None:
                counter_rows.extend({**meta, **counter} for counter in sim.counter_metrics())
    raw = pd.DataFrame(rows)
    return raw, _summarise(raw, ["factor", "level"])


def sensitivity_comparisons(raw: pd.DataFrame) -> pd.DataFrame:
    """Compare each level to its own factor's reference, using matching days."""
    rows = []
    for factor, frame in raw.groupby("factor", sort=False):
        ref_level = FACTOR_REFERENCES[factor]
        reference = frame.loc[frame.level == ref_level]
        for level, candidate in frame.groupby("level", sort=False):
            if level != ref_level:
                rows.extend(_paired_rows(candidate, reference, {"factor": factor, "level": level,
                                                               "reference_level": ref_level},
                                         "mean_difference"))
    return pd.DataFrame(rows)


def capacity_margins(raw: pd.DataFrame) -> pd.DataFrame:
    """Minutes saved and extra staffed hours when adding one counter."""
    levels = raw.loc[raw.factor == "counter_number"]
    rows = []
    for smaller in (3, 4, 5, 6):
        fewer = levels.loc[levels.level == smaller]
        more = levels.loc[levels.level == smaller + 1]
        saved = _paired_days(fewer, more)
        row = {"from_counters": smaller, "to_counters": smaller + 1,
               "additional_counter_hours": -saved["counter_hours_available"].mean(),
               "counter_hours_after": more["counter_hours_available"].mean()}
        for metric in ("mean_wait", "p95_wait"):
            row.update({f"{metric}_saved_{field}": value for field, value in estimate(saved[metric]).items()})
        rows.append(row)
    return pd.DataFrame(rows)
