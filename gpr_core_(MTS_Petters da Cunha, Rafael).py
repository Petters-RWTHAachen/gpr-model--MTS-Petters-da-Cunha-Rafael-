"""
gpr_core.py

Gaussian Process Regression (GPR) surrogate modeling of casting
solidification time and porosity as a function of two die temperatures
(T1: temperature in module 4; T2: temperature in modules 3 and 5).

Data format: Excel file, 2-row header, columns:
    Number | T1 | T2 | Point 3 - Sol. Time | Point 3 - Porosity

Two independent GPR models are fit (one per output). Provides:
  - data loading
  - model construction and fitting
  - leave-one-out cross-validation (diagnostic only)
  - convex-hull interpolation/extrapolation check
  - inverse design (solve for T1, T2 given a target time & porosity)

Environment used to produce the thesis results:
  Python 3.12.3 | scikit-learn 1.8.0 | numpy 2.4.4 | pandas 3.0.2 |
  scipy 1.17.1 | matplotlib 3.10.8 | openpyxl 3.1.5
  Ubuntu 24.04.4 LTS, Intel Xeon @ 2.10 GHz, 4 GB RAM
"""

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from scipy.optimize import minimize, differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_data(path):
    """
    Load T1, T2, solidification time, and porosity from the Excel file.

    Returns
    -------
    X : ndarray, shape (n, 2)   -- columns [T1, T2]
    y_time : ndarray, shape (n,)
    y_poro : ndarray, shape (n,)
    """
    df = pd.read_excel(path, header=[0, 1])
    df.columns = ["Number", "T1", "T2", "t3", "p3"]
    X = df[["T1", "T2"]].to_numpy(dtype=float)
    y_time = df["t3"].to_numpy(dtype=float)
    y_poro = df["p3"].to_numpy(dtype=float)
    return X, y_time, y_poro


# ---------------------------------------------------------------------------
# 2. GPR model construction
# ---------------------------------------------------------------------------

def build_gpr():
    """
    Anisotropic RBF kernel (one length scale per input dimension),
    scaled by a signal-variance (ConstantKernel) term, plus an additive
    WhiteKernel noise term.

    Hyperparameter search bounds (deliberately chosen, not learned):
      - length_scale:  init 1.0,  bounds (1e-2, 1e2)   [per dimension]
      - signal variance (ConstantKernel): init 1.0, bounds (1e-2, 1e3)
      - noise variance (WhiteKernel):     init 1e-2, bounds (1e-6, 1e1)

    Optimizer: scikit-learn default (L-BFGS-B), restarted from
    n_restarts_optimizer=15 random initializations in addition to the
    single default initialization above; the restart achieving the
    highest log marginal likelihood is kept.
    """
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3))
        * RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))
    )
    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=15, random_state=0
    )


def fit_models(X, y_time, y_poro):
    """Standardize inputs (fit on X), then fit one GPR per output."""
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    gpr_time = build_gpr().fit(Xs, y_time)
    gpr_poro = build_gpr().fit(Xs, y_poro)
    return scaler, gpr_time, gpr_poro


# ---------------------------------------------------------------------------
# 3. Leave-one-out cross-validation (diagnostic only - does NOT change the
#    final deployed model, which is always fit on the FULL dataset via
#    fit_models() above)
# ---------------------------------------------------------------------------

def loo_cv(X, y):
    """
    Leave-one-out cross-validation for a single output (time OR porosity).
    Returns (r2, mae, rmse) computed on the held-out predictions.
    """
    loo = LeaveOneOut()
    pred = np.zeros_like(y)
    for train_idx, test_idx in loo.split(X):
        scaler = StandardScaler().fit(X[train_idx])
        g = build_gpr().fit(scaler.transform(X[train_idx]), y[train_idx])
        pred[test_idx] = g.predict(scaler.transform(X[test_idx]))
    r2 = r2_score(y, pred)
    mae = mean_absolute_error(y, pred)
    rmse = mean_squared_error(y, pred) ** 0.5
    return r2, mae, rmse


# ---------------------------------------------------------------------------
# 4. Sampled-domain / convex-hull check (interpolation vs extrapolation)
# ---------------------------------------------------------------------------

def in_sampled_domain(points, X):
    """True for `points` that fall inside the convex hull of sampled X."""
    hull = ConvexHull(X)
    path = Path(X[hull.vertices])
    return path.contains_points(points)


# ---------------------------------------------------------------------------
# 5. Inverse design: find (T1, T2) matching a desired time & porosity
# ---------------------------------------------------------------------------

def inverse_solve(t_target, p_target, X, scaler, gpr_time, gpr_poro, w_poro=1.0):
    """
    Minimize normalized squared error between predicted and desired
    (time, porosity), searching within the bounding box of sampled T1/T2.
    Global search (differential_evolution) + local polish (L-BFGS-B).

    Returns a dict with T1, T2, time_pred, time_std, poro_pred, poro_std,
    inside_sampled_domain (bool), objective (residual at solution).
    """
    t1_bounds = (X[:, 0].min(), X[:, 0].max())
    t2_bounds = (X[:, 1].min(), X[:, 1].max())

    def objective(T):
        Ts = scaler.transform([T])
        t_pred = gpr_time.predict(Ts)[0]
        p_pred = gpr_poro.predict(Ts)[0]
        t_err = (t_pred - t_target) / t_target
        p_err = (p_pred - p_target) / p_target
        return t_err ** 2 + w_poro * p_err ** 2

    result_global = differential_evolution(
        objective, bounds=[t1_bounds, t2_bounds], seed=0, tol=1e-10, polish=True
    )
    result = minimize(
        objective, x0=result_global.x, bounds=[t1_bounds, t2_bounds], method="L-BFGS-B"
    )

    T_opt = result.x
    Ts_opt = scaler.transform([T_opt])
    t_pred, t_std = gpr_time.predict(Ts_opt, return_std=True)
    p_pred, p_std = gpr_poro.predict(Ts_opt, return_std=True)
    inside = in_sampled_domain(np.array([T_opt]), X)[0]

    return {
        "T1": T_opt[0], "T2": T_opt[1],
        "time_pred": t_pred[0], "time_std": t_std[0],
        "poro_pred": p_pred[0], "poro_std": p_std[0],
        "inside_sampled_domain": bool(inside),
        "objective": result.fun,
    }


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    X, y_time, y_poro = load_data("Model_data_25_C_steps.xlsx")
    scaler, gpr_time, gpr_poro = fit_models(X, y_time, y_poro)

    r2_t, mae_t, rmse_t = loo_cv(X, y_time)
    r2_p, mae_p, rmse_p = loo_cv(X, y_poro)
    print(f"Time:     R2={r2_t:.4f}  MAE={mae_t:.3f} s  RMSE={rmse_t:.3f} s")
    print(f"Porosity: R2={r2_p:.4f}  MAE={mae_p:.3f}    RMSE={rmse_p:.3f}")

    solution = inverse_solve(13.81, 1.42, X, scaler, gpr_time, gpr_poro)
    print(solution)
