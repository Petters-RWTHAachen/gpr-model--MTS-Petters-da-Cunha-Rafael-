# Casting Die-Temperature GPR Surrogate Model

Gaussian Process Regression (GPR) surrogate model relating two casting die
temperatures to solidification time and porosity, developed as part of a
thesis project on casting process optimization.

## Overview

Casting simulations are computationally expensive, so only a limited number
of simulation runs are available for any given set of process parameters.
This project fits a GPR surrogate model to a small set of simulation
results, allowing:

- Fast prediction of solidification time and porosity for any die
  temperature combination within the sampled range.
- Inverse design: given a desired solidification time and porosity, solving
  for the die temperature combination that best achieves them.
- Quantification of prediction uncertainty, including detection of when a
  requested prediction falls outside the region covered by the original
  simulation data.

## Inputs

The two controllable process parameters:

- **T1** — die temperature, module 4
- **T2** — die temperature, modules 3 and 5

The two model outputs, evaluated at a single reference point in the
casting:

- **Solidification time** [s]
- **Porosity** [%]

## Data format

Input files are Excel spreadsheets with a two-row header and the following
columns:

| Column | Content |
|---|---|
| Number | Simulation run index |
| T1 | Die temperature, module 4 [°C] |
| T2 | Die temperature, modules 3 and 5 [°C] |
| Point 3 – Sol. Time | Solidification time [s] |
| Point 3 – Porosity | Porosity [%] |

Three example datasets are used in the associated thesis, differing in how
finely T1 was sampled (100 °C, 50 °C, and 25 °C steps).

## Model

- **Kernel:** anisotropic RBF (one length scale per input dimension),
  scaled by a signal-variance term (`ConstantKernel`), with an additive
  observation-noise term (`WhiteKernel`).
- **Hyperparameters** (length scales, signal variance, noise variance) are
  learned automatically by maximizing the log marginal likelihood, with the
  optimizer restarted from 15 random initializations to reduce the risk of
  a poor local optimum.
- **Preprocessing:** inputs standardized to zero mean/unit variance;
  outputs normalized internally (`normalize_y=True`).
- **Validation:** leave-one-out cross-validation (diagnostic only — the
  deployed model is always fit on the full dataset).
- **Inverse design:** global search (`differential_evolution`) followed by
  local refinement (`L-BFGS-B`) to find the die temperature combination
  matching a desired solidification time and porosity, with the model's
  predictive uncertainty reported at the solution.

Two independent GPR models are fit — one for solidification time, one for
porosity — both using the same two inputs.

## Usage

```bash
pip install -r requirements.txt
```

```python
from gpr_core import load_data, fit_models, loo_cv, inverse_solve

# Fit the model
X, y_time, y_poro = load_data("Model_data_25_C_steps.xlsx")
scaler, gpr_time, gpr_poro = fit_models(X, y_time, y_poro)

# Validate
r2_t, mae_t, rmse_t = loo_cv(X, y_time)
r2_p, mae_p, rmse_p = loo_cv(X, y_poro)
print(f"Time:     R2={r2_t:.4f}  MAE={mae_t:.3f} s  RMSE={rmse_t:.3f} s")
print(f"Porosity: R2={r2_p:.4f}  MAE={mae_p:.3f}    RMSE={rmse_p:.3f}")

# Solve the inverse problem: find T1, T2 for a target time & porosity
solution = inverse_solve(t_target=13.81, p_target=1.42,
                          X=X, scaler=scaler, gpr_time=gpr_time, gpr_poro=gpr_poro)
print(solution)
```

## Environment

Results in the associated thesis were produced with:

- Python 3.12.3
- scikit-learn 1.8.0
- NumPy 2.4.4
- pandas 3.0.2
- SciPy 1.17.1
- Matplotlib 3.10.8
- openpyxl 3.1.5

See `requirements.txt` for exact versions.

## Repository contents

- `gpr_core.py` — model implementation (data loading, fitting, validation,
  inverse solver).
- `requirements.txt` — exact Python package versions used.
- Example input data files (`.xlsx`).

## Citation / context

Developed as part of a thesis on Gaussian Process Regression surrogate
modeling for casting process optimization.
