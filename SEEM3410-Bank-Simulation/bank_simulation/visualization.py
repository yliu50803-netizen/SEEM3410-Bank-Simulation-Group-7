"""Small-multiple charts with real factor levels and pointwise 95% intervals."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from .experiments import PEAK_WINDOWS

BLUE, GREEN, INK, GRID = "#174C84", "#26776B", "#18324A", "#E3EAF1"
METRIC_LABELS = {
    "mean_wait": "Mean wait (min)",
    "p95_wait": "Daily P95 wait (min)",
    "p_wait_gt_20": "Wait over 20 min",
    "counter_utilization": "Staffed-time utilization",
    "counter_hours_available": "Staffed counter-hours",
    "waiting_at_close": "Customers waiting at 17:00",
    "in_service_at_close": "Customers in service at 17:00",
    "overtime_service_minutes": "Overtime service (counter-min)",
    "mean_wait_saved": "Mean wait saved (min)",
    "p95_wait_saved": "Daily P95 wait saved (min)",
}
SENSITIVITY_PANELS = {
    "arrival_rate_scale": ("Arrival rate", "Multiplier on each time band's arrival rate",
                           ("mean_wait", "p_wait_gt_20", "p95_wait", "waiting_at_close")),
    "service_time_scale": ("Service time", "Multiplier on service mean and standard deviation",
                            ("mean_wait", "counter_utilization", "p95_wait", "overtime_service_minutes")),
    "counter_number": ("Counter capacity", "Number of all-day counters",
                        ("mean_wait", "p95_wait", "counter_utilization", "counter_hours_available")),
    "peak_concentration": ("Peak concentration", "Peak window (same expected daily volume)",
                           ("mean_wait", "p95_wait", "p_wait_gt_20", "waiting_at_close")),
    "break_schedule_offset": ("Break timing", "Breaks moved later by this many minutes",
                               ("mean_wait", "p95_wait", "counter_utilization", "waiting_at_close")),
}


def _style(ax, axis="y"):
    ax.grid(axis=axis, color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(axis="both", length=0, pad=5, colors=INK)


def _finish(fig, output, name, title, note):
    fig.suptitle(title, y=0.96, fontsize=17, color=INK, weight="bold")
    fig.text(0.07, 0.035, note, color=INK, fontsize=9)
    fig.savefig(output / f"{name}.png", dpi=180, facecolor="white")
    plt.close(fig)


def _ci(frame, metric, scale=1.0):
    mean = frame[f"{metric}_mean"].to_numpy(dtype=float) * scale
    low = frame[f"{metric}_ci95_low"].to_numpy(dtype=float) * scale
    high = frame[f"{metric}_ci95_high"].to_numpy(dtype=float) * scale
    return mean, np.column_stack((mean - low, high - mean)).T


def _metric_panel(ax, x, labels, frame, metric, color=BLUE, connect=True):
    is_percent = metric in ("counter_utilization", "p_wait_gt_20")
    mean, errors = _ci(frame, metric, 100 if is_percent else 1)
    ax.errorbar(x, mean, yerr=errors, color=color, fmt="o-" if connect else "o",
                markersize=5, linewidth=1.4, capsize=4)
    ax.set_xticks(x, labels)
    ax.set_title(METRIC_LABELS[metric], loc="left", fontsize=11, weight="bold", color=INK)
    if is_percent:
        ax.yaxis.set_major_formatter(PercentFormatter())
    _style(ax)


def _panels(size):
    fig, axes = plt.subplots(2, 2, figsize=size)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.85, bottom=0.18, hspace=0.53, wspace=0.26)
    return fig, axes


def _scenario_plots(summary, customers, output):
    order = ["C3_all_day", "C4_all_day", "C5_all_day", "C6_all_day", "C7_all_day",
             "S_peak_reinforcement", "S_two_peak_flex", "S_staggered_breaks"]
    labels = ["C3", "C4", "C5", "C6", "C7", "Peak\nextra", "Split\nextra", "Breaks"]
    frame = summary.set_index("scenario").loc[order]
    note = "Points show the average across simulated days. Bars show pointwise 95% confidence intervals."
    fig, axes = _panels((10.7, 6.7))
    for ax, metric in zip(axes.flat, ("mean_wait", "p95_wait", "p_wait_gt_20", "counter_utilization")):
        _metric_panel(ax, range(len(labels)), labels, frame, metric, connect=False)
    _finish(fig, output, "scenarios_wait_and_utilization", "Waiting and utilization by staffing plan", note)

    fig, axes = _panels((10.7, 6.7))
    for ax, metric in zip(axes.flat, ("counter_hours_available", "waiting_at_close",
                                     "in_service_at_close", "overtime_service_minutes")):
        _metric_panel(ax, range(len(labels)), labels, frame, metric, color=GREEN, connect=False)
    _finish(fig, output, "scenarios_staffing_and_clearance", "Staffing effort and closing-time workload", note)

    if not customers.empty:
        fig, ax = plt.subplots(figsize=(9.3, 5.2))
        fig.subplots_adjust(left=0.12, right=0.97, top=0.87, bottom=0.19)
        colors = ["#895B3B", BLUE, GREEN, "#7F629A", "#6C7A89"]
        for i, scenario in enumerate(order[:5]):
            waits = np.sort(customers.loc[customers.scenario == scenario, "waiting_time"].dropna())
            if len(waits):
                ax.step(np.r_[0, waits], np.arange(len(waits) + 1) / len(waits),
                        where="post", label=labels[i], color=colors[i])
        ax.set(xlabel="Wait (minutes)", ylabel="Share of customers at or below this wait", ylim=(0, 1.025))
        ax.axvline(20, color=INK, linestyle=":", linewidth=1)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.legend(title="All-day counters", loc="lower right", frameon=False)
        _style(ax)
        _finish(fig, output, "waiting_time_distributions", "How long customers waited",
                "Empirical distributions pool customers who started service across all simulated days.")


def _clock(minute):
    return f"{minute // 60:02}:{minute % 60:02}"


def _factor_axis(factor, levels):
    if factor == "peak_concentration":
        labels = [f"{level.title()}\n{_clock(PEAK_WINDOWS[level][0])}–{_clock(PEAK_WINDOWS[level][1])}"
                  for level in levels]
        return np.arange(len(levels)), labels
    if factor.endswith("_scale"):
        return np.array(levels, dtype=float), [f"{float(level):.1f}×" for level in levels]
    return np.array(levels, dtype=int), [str(level) for level in levels]


def _sensitivity_plots(summary, margins, output):
    for factor, frame in summary.groupby("factor", sort=False):
        title, xlabel, metrics = SENSITIVITY_PANELS[factor]
        x, labels = _factor_axis(factor, frame.level)
        fig, axes = _panels((9.2, 6.2))
        for index, (ax, metric) in enumerate(zip(axes.flat, metrics)):
            _metric_panel(ax, x, labels, frame, metric, color=BLUE if index % 2 == 0 else GREEN)
        fig.supxlabel(xlabel, y=0.081, color=INK, fontsize=10)
        _finish(fig, output, f"sensitivity_{factor}", title,
                "Points show the average across simulated days. Bars show pointwise 95% confidence intervals.")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    fig.subplots_adjust(left=0.08, right=0.96, top=0.81, bottom=0.24, wspace=0.25)
    labels = [f"{a} → {b}\n{h:g} h/day total"
              for a, b, h in zip(margins.from_counters, margins.to_counters, margins.counter_hours_after)]
    for ax, metric in zip(axes, ("mean_wait_saved", "p95_wait_saved")):
        _metric_panel(ax, range(len(labels)), labels, margins, metric, connect=False)
        ax.axhline(0, color=INK, linewidth=0.75)
    extra_hours = margins.additional_counter_hours.iloc[0]
    _finish(fig, output, "sensitivity_capacity_margins", "What one more counter buys",
            f"Each step adds {extra_hours:g} staffed counter-hours/day. Bars show paired 95% confidence intervals.")


def create_report_plots(scenario_summary, customers, sensitivity_summary=None, margins=None,
                        output_dir="results/plots"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _scenario_plots(scenario_summary, customers, output)
    if sensitivity_summary is not None:
        _sensitivity_plots(sensitivity_summary, margins, output)
