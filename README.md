# Causal Emergence and Intervention Distributions

Code accompanying the paper **"Intervention Distribution Matters: Identifying Causal Emergence in Flocking Systems"** (submitted at SISSY 2026 @ ACSOS).

This repository investigates how the choice of intervention distribution affects the Rosas et al. (2020) causal emergence criterion $\Psi$, and whether that choice can distinguish genuine emergent macro-variables from confounder-driven false positives. Experiments use Reynolds boids flocking simulations as a testbed.

## Repository structure

```
src/
  basic_example.py        – Boolean parity baseline: exact MI computation over discrete states
  flocking.py             – Reynolds boids simulation (positions, headings, alignment/cohesion/separation)
  flocking_observables.py – Macro feature extractors (polarization, pairwise distance)
  mi_estimators.py        – KSG mutual information estimator with surrogate-based bias correction
  psi.py                  – Ψ score: I(C_M, E_M) − Σ_j I(C_m^j, E_M)
  main.py                 – Regime × macro × prior sweep; noise-perturbation grid search
requirement.txt           – Python dependencies
```

## Installation

```bash
pip install -r requirement.txt
```

Dependencies: `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `tqdm`, `pygame`.

Python 3.9+ recommended.

## Usage

**Run the main sweep** (regime × macro feature × intervention prior):

```bash
python src/main.py
```

This prints a table of $\Psi$, $I(C_M, E_M)$, and $\sum_j I(C_m^j, E_M)$ for each combination of flocking regime (tight, attractor, wind, …), macro feature (polarization, pairwise distance), and intervention distribution (trajectory, uniform, noise-perturbed, confounder).

**Noise-perturbation grid search** (sweep over number of perturbed variables and noise amplitude):

```python
from main import make_noise_sweep, render_noise_sweep
make_noise_sweep("tight", "distance", max_n=15, max_factor=0.7, n_values=16)
render_noise_sweep("tight", "distance", max_n=15, max_factor=0.7, n_values=5)
```

Results are saved to `data/` and figures to `figures/`.

## Reference

F. Rosas, P. A. M. Mediano, H. J. Jensen, A. K. Seth, A. B. Barrett, R. Carhart-Harris, D. Bor, "Reconciling emergences: An information-theoretic approach to identify causal emergence in multivariate data," *PLOS Computational Biology*, 2020.

## License

MIT
