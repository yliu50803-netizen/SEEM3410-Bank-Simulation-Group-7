"""One General Counter queue, with scheduled staff and after-hours clearance."""

from collections import deque
from dataclasses import asdict, dataclass
from heapq import heappop, heappush
from math import isfinite
from typing import Iterable

import numpy as np
from scipy.special import gammaincinv

from .config import CLOSE_MINUTE, OPEN_MINUTE, CounterConfig, ScenarioConfig

DEPARTURE, CAPACITY_CHANGE, ARRIVAL = range(3)
BUSINESS_MINUTES = CLOSE_MINUTE - OPEN_MINUTE


@dataclass
class Customer:
    customer_id: int
    arrival_time: float
    service_time: float
    service_start_time: float | None = None
    departure_time: float | None = None
    counter_id: str | None = None

    @property
    def waiting_time(self) -> float | None:
        if self.service_start_time is None:
            return None
        return self.service_start_time - self.arrival_time

    @property
    def system_time(self) -> float | None:
        if self.departure_time is None:
            return None
        return self.departure_time - self.arrival_time

    def status_at(self, minute: float) -> str:
        # At 17:00, count departures due at closing before any overtime starts.
        if self.departure_time is not None and self.departure_time <= minute:
            return "completed"
        if self.service_start_time is not None and self.service_start_time < minute:
            return "in_service"
        return "waiting"

    def as_dict(self) -> dict:
        return {
            **asdict(self),
            "waiting_time": self.waiting_time,
            "system_time": self.system_time,
            "status_at_close": self.status_at(CLOSE_MINUTE),
            "status_at_end": self.status_at(float("inf")),
        }


@dataclass
class CounterState:
    config: CounterConfig
    customer: Customer | None = None
    busy_minutes: float = 0.0
    busy_available_minutes: float = 0.0
    overtime_service_minutes: float = 0.0
    off_shift_service_minutes: float = 0.0

    def scheduled(self, minute: float) -> bool:
        c = self.config
        on_shift = c.open_minute <= minute < c.close_minute
        can_clear = c.close_minute == CLOSE_MINUTE and minute >= CLOSE_MINUTE
        on_break = any(start <= minute < end for start, end in c.breaks)
        return (on_shift or can_clear) and not on_break

    def available(self, minute: float) -> bool:
        return self.customer is None and self.scheduled(minute)

    def available_minutes(self) -> float:
        return self.scheduled_overlap(OPEN_MINUTE, CLOSE_MINUTE)

    def scheduled_overlap(self, start: float, end: float) -> float:
        """Part of [start, end) inside the shift, excluding breaks."""
        c = self.config
        left, right = max(start, c.open_minute), min(end, c.close_minute)
        if right <= left:
            return 0.0
        break_minutes = sum(max(0.0, min(right, b) - max(left, a)) for a, b in c.breaks)
        return right - left - break_minutes


def gamma_parameters(mean: float, sd: float) -> tuple[float, float]:
    shape = (mean / sd) ** 2
    return shape, mean / shape


def sample_service_times(rng: np.random.Generator, mean: float, sd: float, size: int) -> np.ndarray:
    shape, scale = gamma_parameters(mean, sd)
    # The same arrival rank gets the same percentile, even when the Gamma parameters change.
    percentiles = np.maximum(rng.random(size), np.finfo(float).tiny)
    return gammaincinv(shape, percentiles) * scale


def _poisson_arrivals(rng: np.random.Generator, rates) -> list[float]:
    """Run a unit-rate Poisson clock, then map it onto each time band's rate."""
    minutes = []
    cumulative_rate = 0.0
    next_event = rng.exponential()
    for start, end, per_hour in rates:
        if per_hour == 0:
            continue
        per_minute = per_hour / 60
        band_end = cumulative_rate + (end - start) * per_minute
        while next_event < band_end:
            minute = start + (next_event - cumulative_rate) / per_minute
            minutes.append(min(minute, float(np.nextafter(end, start))))
            next_event += rng.exponential()
        cumulative_rate = band_end
    return minutes


class BankSimulation:
    def __init__(
        self, config: ScenarioConfig, replication: int = 0, seed: int | None = None,
        *, arrivals: Iterable[Customer] | None = None,
    ):
        self.config = config
        self.replication = replication
        self.seed = config.seed + replication if seed is None else seed
        arrival_seed, service_seed = np.random.SeedSequence(self.seed).spawn(2)
        self.rng_arrival = np.random.default_rng(arrival_seed)
        self.rng_service = np.random.default_rng(service_seed)
        if arrivals is None:
            times = _poisson_arrivals(self.rng_arrival, config.arrival_rates)
            demands = sample_service_times(
                self.rng_service, config.service_mean, config.service_sd, len(times)
            )
            arrivals = (Customer(i, t, float(d)) for i, (t, d) in enumerate(zip(times, demands), 1))
        # Replay supplied arrivals without mutating the caller's input.
        self.customers = sorted(
            (Customer(c.customer_id, c.arrival_time, c.service_time) for c in arrivals),
            key=lambda c: (c.arrival_time, c.customer_id),
        )
        self.counters = [CounterState(c) for c in sorted(config.counters, key=lambda c: c.counter_id)]
        self.events = []
        self.sequence = 0
        self.queue: deque[Customer] = deque()
        self.queue_area = 0.0
        self.last_queue_update = float(OPEN_MINUTE)
        self.max_queue = 0
        self._ran = False

        seen_ids = set()
        for c in self.customers:
            if not OPEN_MINUTE <= c.arrival_time < CLOSE_MINUTE:
                raise ValueError("Customer arrivals must be during 09:00–17:00, excluding 17:00")
            if not (isfinite(c.service_time) and c.service_time > 0):
                raise ValueError("Customer service demand must be finite and positive")
            if c.customer_id in seen_ids:
                raise ValueError("Customer IDs must be unique within a run")
            seen_ids.add(c.customer_id)
            self._push(c.arrival_time, ARRIVAL, c)

        # A schedule change can free capacity even when no customer arrives then.
        change_times = {CLOSE_MINUTE}
        for counter in self.counters:
            c = counter.config
            change_times.update((c.open_minute, c.close_minute))
            for start, end in c.breaks:
                clears_overtime = c.close_minute == CLOSE_MINUTE
                if end > c.open_minute and (start < c.close_minute or clears_overtime):
                    change_times.add(end)
                    if start >= c.open_minute:
                        change_times.add(start)
        for minute in sorted(change_times):
            self._push(minute, CAPACITY_CHANGE, None)

    def _push(self, minute, kind, payload):
        self.sequence += 1
        heappush(self.events, (minute, kind, self.sequence, payload))

    def _update_queue_area(self, minute: float):
        end = min(minute, CLOSE_MINUTE)
        self.queue_area += len(self.queue) * (end - self.last_queue_update)
        self.last_queue_update = end

    def _start_service(self, customer: Customer, counter: CounterState, minute: float):
        customer.service_start_time = minute
        customer.counter_id = counter.config.counter_id
        counter.customer = customer
        finish = minute + customer.service_time
        regular = counter.scheduled_overlap(minute, finish)
        overtime = max(0.0, finish - max(minute, CLOSE_MINUTE))
        counter.busy_minutes += customer.service_time
        counter.busy_available_minutes += regular
        counter.overtime_service_minutes += overtime
        counter.off_shift_service_minutes += max(0.0, min(finish, CLOSE_MINUTE) - minute - regular)
        self._push(finish, DEPARTURE, counter)

    def _dispatch(self, minute: float):
        for counter in self.counters:
            if not self.queue:
                break
            if counter.available(minute):
                self._start_service(self.queue.popleft(), counter, minute)

    def run(self) -> list[Customer]:
        while self.events:
            minute = self.events[0][0]
            self._update_queue_area(minute)
            # Finish all due services before assigning any waiting customers. Shift and
            # break changes take effect by the clock, so dispatch sees the new schedule.
            while self.events and self.events[0][0] == minute:
                _, kind, _, payload = heappop(self.events)
                if kind == DEPARTURE:
                    payload.customer.departure_time = minute
                    payload.customer = None
                elif kind == ARRIVAL:
                    self.queue.append(payload)
            self._dispatch(minute)
            if minute < CLOSE_MINUTE:
                self.max_queue = max(self.max_queue, len(self.queue))
        self._ran = True
        return self.customers

    def metrics(self) -> dict:
        if not self._ran:
            raise RuntimeError("Run the simulation before calculating metrics")
        started = [c for c in self.customers if c.service_start_time is not None]
        completed = [c for c in started if c.departure_time is not None]
        waits = np.array([c.waiting_time for c in started])
        close_statuses = [c.status_at(CLOSE_MINUTE) for c in self.customers]
        available = sum(c.available_minutes() for c in self.counters)
        busy = sum(c.busy_available_minutes for c in self.counters)
        median, p90, p95 = np.percentile(waits, [50, 90, 95]) if len(waits) else [np.nan] * 3

        return {
            "arrivals": len(self.customers),
            "served_started": len(started),
            "completed": len(completed),
            "completed_at_close": close_statuses.count("completed"),
            "waiting_at_close": close_statuses.count("waiting"),
            "in_service_at_close": close_statuses.count("in_service"),
            "waiting_at_end": len(self.customers) - len(started),
            "in_service_at_end": len(started) - len(completed),
            "unfinished_at_end": len(self.customers) - len(completed),
            "mean_wait": float(waits.mean()) if len(waits) else np.nan,
            "median_wait": float(median),
            "p90_wait": float(p90),
            "p95_wait": float(p95),
            "p_wait_gt_20": float((waits > 20).mean()) if len(waits) else np.nan,
            "p_wait_gt_30": float((waits > 30).mean()) if len(waits) else np.nan,
            "mean_service_time": float(np.mean([c.departure_time - c.service_start_time for c in completed]))
            if completed else np.nan,
            "mean_system_time": float(np.mean([c.system_time for c in completed])) if completed else np.nan,
            "time_avg_queue_length": self.queue_area / BUSINESS_MINUTES,
            "max_queue_length": self.max_queue,
            "counter_utilization": busy / available if available else np.nan,
            "counter_hours_available": available / 60,
            "counter_hours_busy": busy / 60,
            "overtime_service_minutes": sum(c.overtime_service_minutes for c in self.counters),
            "off_shift_service_minutes": sum(c.off_shift_service_minutes for c in self.counters),
        }

    def counter_metrics(self) -> list[dict]:
        if not self._ran:
            raise RuntimeError("Run the simulation before calculating counter metrics")
        rows = []
        for counter in self.counters:
            available = counter.available_minutes()
            rows.append({
                "counter_id": counter.config.counter_id,
                "available_minutes": available,
                "busy_available_minutes": counter.busy_available_minutes,
                "total_service_minutes": counter.busy_minutes,
                "counter_utilization": counter.busy_available_minutes / available if available else np.nan,
                "overtime_service_minutes": counter.overtime_service_minutes,
                "off_shift_service_minutes": counter.off_shift_service_minutes,
            })
        return rows
