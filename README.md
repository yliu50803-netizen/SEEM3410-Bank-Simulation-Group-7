# SEEM3410-Bank-Simulation-Group-7

A Python-based discrete-event simulation of the General Counter service system at Hang Seng Bank Branch.

## Project Overview

This project evaluates alternative counter allocation policies to reduce customer waiting time while achieving a better resource utilization.

The model uses:
- A single FIFO queue.
- Time-dependent Poisson customer arrivals.
- Gamma-distributed service times.
- Multiple service counters.
- Alternative staffing and break schedules.

All input parameters are illustrative assumptions, not official Hang Seng Bank operational data.

## Scenario Experiments

Eight strategies are evaluated:

| Scenario | Counter Configuration |
|---|---|
| S0 | Four fixed counters |
| S1 | One additional counter during peak hours |
| S2 | Two additional counters during peak hours |
| S3 | Three fixed counters |
| S3 | Five fixed counters |
| S3 | Six fixed counters |
| S3 | Seven fixed counters |
| S4 | Staggered counter breaks |

The project also includes sensitivity analysis for arrival rates, service times, counter capacity, peak concentration and break schedules.

## Installation

Python 3.10 or later is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Simulation

Run the main experiments and sensitivity analysis:

```bash
python main.py
```

Run the main scenarios only:

```bash
python main.py --skip-sensitivity
```

Run with a specified random seed:

```bash
python main.py --seed 20261124
```

Results, CSV summaries and plots are automatically generated in the `results/` directory.

## Run Tests

```bash
python -m unittest discover -s tests -v
```

## Reproducibility

The default random seed is `20261124`. Replication seeds are generated using:

```text
seed = base_seed + (replication - 1) * 10007
```

The same configuration and seed reproduce the corresponding simulation experiments.

## Project Structure

- `bank_simulation/` – Simulation model, configuration, experiments and visualization.
- `tests/` – Model and experiment tests.
- `main.py` – Main execution script.
- `requirements.txt` – Python dependencies.

Course project: SEEM3410 System Modeling and Simulation (2026–27 Term 1).
