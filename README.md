# UMDO Lab: UAV Design Working Example

> **Status:** Under Construction

This repository provides a working example for uncertainty-based multidisciplinary design optimization (UMDO) applied to the conceptual sizing of a quadrotor biplane tailsitter (QBiT) unmanned aerial vehicle.

The deterministic sizing model is based on the QBiT formulation presented by Govindarajan et al. (2020)[^1] and Kaneko & Martins (2023)[^2]. The model was reimplemented in OpenMDAO and extended with uncertainty propagation and robust design optimization under uncertain hover-time requirements using Monte Carlo Simulation and Polynomial Chaos Expansion via UQPCE[^3].

## Context

The sizing formulation follows published conceptual UAV sizing methods for package-delivery missions and extends them with stochastic modeling. Two vehicle architectures are represented in the code base:

- QBiT, a transition-capable tailsitter configuration
- Hexarotor, used here as a multirotor comparison baseline

The uncertainty studies focus on how variability in hover requirements and related performance parameters affects key sizing quantities such as takeoff mass. 

## What The Repository Explores

At a high level, the code supports three related activities:

1. Deterministic sizing of the UAV mission design point.
2. Uncertainty propagation using Monte Carlo simulation and polynomial chaos methods.
3. Robust sizing and comparison between design architectures across mission ranges and payload masses.

The example mission is a logistics delivery scenario with multiple customer stops, a fixed total mission range, and a payload representative of small package transport.

## Working Example

The example considers a logistics delivery mission with:
- 30 km total mission range
- 2 customer stops
- 3 kg payload
- Uncertain hover-time requirement

The design objective is to minimize total takeoff weight, used as a proxy for vehicle cost.

The optimization problem includes the following design variables:
- Cruise speed ($V_\infty$)
- Rotor radius ($r$)
- Propeller advance ratio ($J$)
- Wing area ($S_w$)

Subject to:
- Weight closure constraint
- Disk loading constraint
- Blade loading constraint
- Cruise lift coefficient constraint

## Repository Structure

The main implementation lives in the `sizing_openmdao` directory and is organized around model components, group definitions, and analysis scripts.

- `qbit/` contains the QBiT sizing model.
- `hexarotor/` contains the comparison multirotor model.
- `run_qbit.py` and related scripts execute the deterministic baseline and uncertainty-aware studies.
- `run_qbit_monte_carlo.py`, `run_qbit_UQ_static.py`, and `run_qbit_UQPCE.py` provide stochastic and robust-design workflows.


## Dependencies

The workflow relies on a small scientific Python stack:

- `numpy`
- `scipy`
- `openmdao`
- `matplotlib`
- `pyyaml`
- `uqpce`

## References

**Sizing models based on**:

[^1]: Govindarajan, B., Sridharan, A., 2020. *Conceptual Sizing of Vertical Lift Package Delivery Platforms*. Journal of Aircraft 57, 1170–1188. https://doi.org/10.2514/1.C035805

**Design context and sizing models based on**:

[^2]: Kaneko, S., Martins, J.R.R.A., 2023. *Fleet Design Optimization of Package Delivery Unmanned Aerial Vehicles Considering Operations*. Journal of Aircraft 60, 1061–1077. https://doi.org/10.2514/1.C036921

**PCE uncertainty propagation based on**:

[^3]: Ben D. Phillips, Joanna Schmidt, Robert D. Falck, Eliot D. Aretskin-Hariton (2025). *End-to-End Uncertainty Quantification with Analytical Derivatives for Design Under Uncertainty*. Journal of Aircraft, Volume 62, Number6. https://doi.org/10.2514/6.2024-4219

## Disclaimer

This repository is intended as a research and educational example in uncertainty-aware aerospace design. The implementation is under active development and should not be considered a validated engineering design tool.