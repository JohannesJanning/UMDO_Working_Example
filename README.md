# UMDO Lab: UAV Design Working Example

> **Status:** Under Construction

This repository contains the preliminary code for a working example on Uncertainty-Based Multidisciplinary Design Optimization (UMDO) applied to the conceptual sizing of a Quadrotor Biplane Tailsitter (QBiT) UAV.

The deterministic sizing model is based on the QBiT formulation presented by Govindarajan et al. (2020) and Kaneko & Martins (2023). The model was reimplemented in OpenMDAO and extended with uncertainty propagation and robust design optimization under uncertain hover-time requirements.

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

```text
run_qbit_MCS.py
```

Monte Carlo-based robust sizing optimization using uncertainty propagation through repeated model evaluations.

```text
run_qbit_UQPCE.py
```

Polynomial-chaos-based robust sizing optimization using NASA's UQPCE framework.

## Dependencies

```text
numpy
scipy
openmdao
matplotlib
pyyaml
uqpce
```

## References

**Design context based on**:

Kaneko, S., Martins, J.R.R.A., 2023. *Fleet Design Optimization of Package Delivery Unmanned Aerial Vehicles Considering Operations*. Journal of Aircraft 60, 1061–1077. https://doi.org/10.2514/1.C036921

**Sizing models based on**:

Govindarajan, B., Sridharan, A., 2020. *Conceptual Sizing of Vertical Lift Package Delivery Platforms*. Journal of Aircraft 57, 1170–1188. https://doi.org/10.2514/1.C035805

**PCE uncertainty propagation based on**:

Ben D. Phillips, Joanna Schmidt, Robert D. Falck, Eliot D. Aretskin-Hariton (2025). *End-to-End Uncertainty Quantification with Analytical Derivatives for Design Under Uncertainty*. Journal of Aircraft, Volume 62, Number6. https://doi.org/10.2514/6.2024-4219

## Disclaimer

This repository is intended as a research and educational example in uncertainty-aware aerospace design. The implementation is under active development and should not be considered a validated engineering design tool.