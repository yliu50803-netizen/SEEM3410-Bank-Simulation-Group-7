"""General-counter-only model configuration.

All rates, service parameters and counter schedules are illustrative assumptions,
not verified Hang Seng Bank operational data.
"""
from dataclasses import dataclass
from math import isfinite
from typing import Tuple

OPEN_MINUTE, CLOSE_MINUTE = 540, 1020


@dataclass(frozen=True)
class CounterConfig:
    counter_id: str
    open_minute: int = OPEN_MINUTE
    close_minute: int = CLOSE_MINUTE
    breaks: Tuple[Tuple[int, int], ...] = ()

    def __post_init__(self):
        if not self.counter_id or not OPEN_MINUTE <= self.open_minute < self.close_minute <= CLOSE_MINUTE:
            raise ValueError("A counter needs an ID and a shift within 09:00–17:00")
        for start, end in self.breaks:
            if not (isfinite(start) and isfinite(end) and start < end):
                raise ValueError(f"{self.counter_id}: each break needs finite start < end")
        # A minute covered by two overlapping breaks is still just one minute off.
        merged = []
        for start, end in sorted(self.breaks):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        object.__setattr__(self, "breaks", tuple(merged))


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    description: str
    arrival_rates: Tuple[Tuple[int, int, float], ...]
    service_mean: float
    service_sd: float
    counters: Tuple[CounterConfig, ...]
    seed: int = 20261124

    def __post_init__(self):
        previous_end = OPEN_MINUTE
        for start, end, rate in self.arrival_rates:
            if not OPEN_MINUTE <= start < end <= CLOSE_MINUTE:
                raise ValueError("Arrival intervals must lie within 09:00–17:00")
            if start < previous_end:
                raise ValueError("Arrival intervals must be ordered and non-overlapping")
            if not (isfinite(rate) and rate >= 0):
                raise ValueError("Arrival rates must be finite and non-negative")
            previous_end = end
        if not all(isfinite(x) and x > 0 for x in (self.service_mean, self.service_sd)):
            raise ValueError("Service mean and SD must be finite and positive")
        ids = [counter.counter_id for counter in self.counters]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Provide at least one counter, with a unique ID for each")

    @property
    def expected_daily_arrivals(self) -> float:
        return sum((end - start) / 60 * rate for start, end, rate in self.arrival_rates)

    def rate_for(self, minute: float) -> float:
        for start, end, rate in self.arrival_rates:
            if start <= minute < end:
                return rate
        return 0.0


def commercial_arrival_rates():
    return ((540, 660, 16.0), (660, 720, 29.0), (720, 840, 43.0),
            (840, 960, 36.0), (960, 1020, 13.0))


def mall_residential_arrival_rates():
    return ((540, 660, 23.0), (660, 780, 40.0), (780, 840, 27.0),
            (840, 960, 37.0), (960, 1020, 17.0))


def make_scenarios(branch_profile="commercial"):
    """Capacity sweep: 4 is a neutral reference, not observed staffing."""
    if branch_profile == "commercial":
        rates = commercial_arrival_rates()
    elif branch_profile == "mall_residential":
        rates = mall_residential_arrival_rates()
    else:
        raise ValueError("Unknown branch profile")

    scenarios = {}
    for n in (4, 3, 5, 6, 7):
        key = f"C{n}_all_day"
        label = "reference" if n == 4 else "capacity test"
        counters = tuple(CounterConfig(f"C{i}") for i in range(1, n + 1))
        scenarios[key] = ScenarioConfig(
            key, f"{n} general counters all day ({label}; assumed)",
            rates, 8.0, 3.0, counters
        )

    four = tuple(CounterConfig(f"C{i}") for i in range(1, 5))
    scenarios["S_peak_reinforcement"] = ScenarioConfig(
        "S_peak_reinforcement", "One extra counter from 12:00 to 16:00 (assumed)",
        rates, 8.0, 3.0, four + (CounterConfig("C5_peak", 720, 960),)
    )
    scenarios["S_two_peak_flex"] = ScenarioConfig(
        "S_two_peak_flex", "One extra counter during 11:00-13:00 and 14:00-16:00 (assumed)",
        rates, 8.0, 3.0,
        four + (CounterConfig("C5_flex_1", 660, 780),
                CounterConfig("C6_flex_2", 840, 960))
    )
    scenarios["S_staggered_breaks"] = ScenarioConfig(
        "S_staggered_breaks", "Four counters with staggered 30-minute breaks (assumed)",
        rates, 8.0, 3.0,
        tuple(CounterConfig(f"C{i}", breaks=((660 + (i-1)*30, 690 + (i-1)*30),))
              for i in range(1, 5))
    )
    return scenarios
