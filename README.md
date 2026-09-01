# Support-aware permafrost residual flow

This repository contains the code used for the manuscript *Preserving Measurement Support in Probabilistic Three-Dimensional Characterization of Ice-Rich Permafrost*.

The implementation represents borehole intervals, finite-volume geophysical measurements, Gaussian-kernel observations, point measurements, and active-layer crossings through explicit observation operators. A tree-derived reference volume is combined with a deterministic residual and centered stochastic anomalies. The evaluation code includes calibration, out-of-distribution control, support-preserving ablations, public complete-borehole holdouts, and engineering-response diagnostics.

## Repository structure

- `cold_recon/`: data adapters, observation operators, models, training utilities, evaluation routines, and synthetic generators.
- `configs/`: experiment registry and model configuration.
- `scripts/`: benchmark construction, training, evaluation, and result aggregation used for the manuscript.
- `tests/`: unit tests for the support-aware and probabilistic components.

## Installation

Python 3.11 or later is recommended.

```bash
git clone https://github.com/corbrander/support-aware-permafrost-residual-flow.git
cd support-aware-permafrost-residual-flow
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

## Checks

```bash
pytest -q
```

The included tests use synthetic arrays and do not require the public borehole archives or trained checkpoints.

## Main workflow

Build the controlled benchmark:

```bash
python scripts/build_m1_support_guided_benchmark.py --config configs/m1_support_guided.yaml
```

Train the factorized autoencoder and the support-aware residual-flow models:

```bash
python scripts/train_m1_factorized_autoencoder.py --config configs/m1_support_guided.yaml
python scripts/train_m1_support_guided_flow.py --config configs/m1_support_guided.yaml
```

Run the controlled evaluation:

```bash
python scripts/evaluate_m1_controlled.py --config configs/m1_support_guided.yaml
```

Additional scripts reproduce the anchor-sensitivity, geostatistical, probability-tree, missing-source, noise, out-of-distribution, public complete-borehole, and sequential-investigation analyses. Paths, split identifiers, and fixed experiment settings are recorded in `configs/m1_support_guided.yaml` and `configs/m1_experiment_registry.yaml`.

Full training and posterior evaluation require a CUDA-capable system and the benchmark or public-source files described in the manuscript. Public-source records are not redistributed here; obtain them from the cited USGS and NSF Arctic Data Center releases.

## Evidence boundary

The controlled benchmark supports full-volume comparisons. The public-data routines test held-out observations and the configured fallback logic; they do not establish dense field-scale ground truth or validate individual ice-body geometry.

## License

The source code is released under the MIT License. Dataset licenses remain those of the original data providers.

