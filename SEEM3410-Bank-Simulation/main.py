"""Run the assumed General Counter model and write tables and charts."""

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from scipy.stats import gamma

from bank_simulation.config import CLOSE_MINUTE, OPEN_MINUTE, make_scenarios
from bank_simulation.experiments import (
    DEFAULT_SEED, FACTOR_REFERENCES, REFERENCE, capacity_margins, run_experiments,
    run_sensitivity, sensitivity_cases, sensitivity_comparisons,
)
from bank_simulation.simulation import gamma_parameters, sample_service_times
from bank_simulation.visualization import create_report_plots


def _describe(config):
    return {**asdict(config), "expected_daily_arrivals": config.expected_daily_arrivals}


def _run_config(args):
    root = Path(__file__).resolve().parent
    files = [root / "main.py", *sorted((root / "bank_simulation").glob("*.py"))]
    return {
        "profile": args.profile,
        "input_status": "Illustrative assumptions; the branch inputs have not been verified.",
        "python": platform.python_version(),
        "package_versions": {name: version(name) for name in ("numpy", "pandas", "matplotlib", "scipy")},
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "base_seed": args.seed,
        "replication_seed_formula": "base_seed + (replication - 1) * 10007",
        "main_replications": args.replications,
        "sensitivity_replications": 0 if args.skip_sensitivity else args.sensitivity_replications,
        "ci": "Pointwise Student t, 95%, across daily estimates or matched daily differences",
        "business_minutes": {"open": OPEN_MINUTE, "close": CLOSE_MINUTE},
        "main_reference": REFERENCE,
        "main_scenarios": {name: _describe(cfg) for name, cfg in make_scenarios(args.profile).items()},
        "sensitivity_references": {} if args.skip_sensitivity else FACTOR_REFERENCES,
        "sensitivity_cases": [] if args.skip_sensitivity else [
            {"factor": factor, "level": level, **_describe(cfg)}
            for factor, level, cfg in sensitivity_cases(args.profile)
        ],
    }


def _service_check(profile, seed):
    base = make_scenarios(profile)[REFERENCE]
    shape, scale = gamma_parameters(base.service_mean, base.service_sd)
    size = 50_000
    # A separate stream for diagnostics, never used to generate the experiment results.
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(2,)))
    sample = sample_service_times(rng, base.service_mean, base.service_sd, size)
    quantiles = (0.05, 0.50, 0.95)
    expected = [base.service_mean, base.service_sd, *gamma.ppf(quantiles, shape, scale=scale)]
    observed = [sample.mean(), sample.std(ddof=1), *np.quantile(sample, quantiles)]
    return pd.DataFrame({
        "statistic": ["mean", "sd", "p05", "p50", "p95"],
        "theoretical_minutes": expected, "sample_minutes": observed, "draws": size,
    })


def main():
    parser = argparse.ArgumentParser(description="General Counter queue simulation (assumed inputs)")
    parser.add_argument("--profile", choices=["commercial", "mall_residential"], default="commercial")
    parser.add_argument("--replications", type=int, default=30)
    parser.add_argument("--sensitivity-replications", type=int, default=30)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--output-dir", type=Path, help="Where to write CSVs, plots and run settings")
    args = parser.parse_args()
    if args.replications < 2 or (not args.skip_sensitivity and args.sensitivity_replications < 2):
        parser.error("use at least 2 replications per case (30 or more for reported results)")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    output = args.output_dir or Path("results/main_only" if args.skip_sensitivity else "results")
    if args.skip_sensitivity and (output / "sensitivity_summary.csv").is_file():
        parser.error("this folder already has a sensitivity run; choose a different --output-dir")

    print(f"Profile {args.profile} | {args.replications} days per main scenario | seed {args.seed}", flush=True)
    try:
        output.mkdir(parents=True, exist_ok=True)
        counter_rows = []
        reps, summary, customers, paired = run_experiments(
            args.profile, args.replications, args.seed, counter_rows=counter_rows
        )
        tables = {
            "replication_metrics": reps,
            "scenario_summary": summary,
            "customer_level_results": customers,
            "counter_level_results": pd.DataFrame(counter_rows),
            "paired_comparisons_vs_C4_reference": paired,
            "service_sampler_check": _service_check(args.profile, args.seed),
        }

        sensitivity_summary = margins = None
        if not args.skip_sensitivity:
            print(f"Running five sensitivity sweeps | {args.sensitivity_replications} days per level", flush=True)
            counter_rows = []
            raw, sensitivity_summary = run_sensitivity(
                args.profile, args.sensitivity_replications, args.seed, counter_rows=counter_rows
            )
            margins = capacity_margins(raw)
            tables.update(
                sensitivity_replications=raw, sensitivity_summary=sensitivity_summary,
                sensitivity_counter_level_results=pd.DataFrame(counter_rows),
                sensitivity_paired_comparisons=sensitivity_comparisons(raw),
                capacity_margins=margins,
            )

        for name, table in tables.items():
            table.to_csv(output / f"{name}.csv", index=False)
        create_report_plots(summary, customers, sensitivity_summary, margins, output / "plots")
        manifest = json.dumps(_run_config(args), indent=2, ensure_ascii=False)
        (output / "run_config.json").write_text(manifest + "\n", encoding="utf-8")
    except OSError as error:
        parser.exit(1, f"Could not write to {output}: {error}\n")

    columns = ["scenario", "mean_wait_mean", "p95_wait_mean", "p_wait_gt_20_mean",
               "counter_utilization_mean", "counter_hours_available_mean"]
    display = summary[columns].rename(columns={
        "mean_wait_mean": "Mean wait (min)", "p95_wait_mean": "Daily P95 (min)",
        "p_wait_gt_20_mean": "Wait >20 min (%)", "counter_utilization_mean": "Utilization (%)",
        "counter_hours_available_mean": "Staffed hours",
    })
    display[["Wait >20 min (%)", "Utilization (%)"]] *= 100
    print(display.round(2).to_string(index=False))
    sensitivity_count = 0 if args.skip_sensitivity else len(raw)
    print(f"\nWrote {len(reps)} main and {sensitivity_count} sensitivity runs to {output}/")


if __name__ == "__main__":
    main()
