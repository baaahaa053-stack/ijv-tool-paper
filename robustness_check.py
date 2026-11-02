"""
Solver robustness check for the IJV Thermal Comfort Tool (reviewer comment 15).
Run from the repository root:  python robustness_check.py
Reports: convergence rate and residual norms over the validated envelope,
first-start vs multi-start success, and sensitivity to random initial guesses.
"""
import itertools
import warnings
import numpy as np
from scipy.optimize import fsolve
from thermal_model import ThermalModel

warnings.filterwarnings("ignore")

A, B, HR, AP, PT, HP, S = 5.0, 5.0, 3.6, 2.56, 120.0, 1.5, 0.175


def make(ts, vs, N):
    m = ThermalModel()
    m.set_case(A, B, HR, AP, PT, N, ts, vs, S, HP)
    return m


# 1. validated envelope: ts 15-19 C, vs 1.6-2.8 m/s, N 32-60 (1.28-2.4 persons/m2)
ts_grid = np.arange(15, 19.01, 0.5)
vs_grid = np.round(np.arange(1.6, 2.81, 0.1), 2)
N_grid = [32, 40, 48, 60]

n = conv = first_ok = 0
res = []
for ts, vs, N in itertools.product(ts_grid, vs_grid, N_grid):
    m = make(ts, vs, N)
    r = m.solve()
    n += 1
    conv += bool(r["converged"])
    res.append(r["res_norm"] if "res_norm" in r else float(np.max(np.abs(
        m.equations_system(np.array([r[k] for k in m.NAMES]))))))
    sk, _, ier, _ = fsolve(m.equations_system, m.T0_DEFAULT, full_output=True)
    rn = float(np.max(np.abs(m.equations_system(sk))))
    first_ok += (ier == 1) and rn < m.RES_TOL and m._is_plausible(sk)
print(f"Validated envelope: {n} cases, converged {conv} ({100*conv/n:.1f}%), "
      f"first start alone {first_ok}")
print(f"max |residual| (W): max = {np.nanmax(res):.2e}, median = {np.nanmedian(res):.2e}")

# 2. random initial guesses at representative corner/centre cases
rng = np.random.default_rng(0)
cases = [(15, 1.6, 32), (17, 2.2, 48), (19, 2.8, 60), (18, 2.0, 32),
         (16, 1.9, 60), (19, 1.6, 32), (15, 2.8, 60)]
tot = ok = implausible = 0
maxdev = 0.0
for ts, vs, N in cases:
    m = make(ts, vs, N)
    ref = m.solve()
    refv = np.array([ref[k] for k in m.NAMES])
    for _ in range(100):
        x0 = rng.uniform(10, 45, 10)
        sk, _, ier, _ = fsolve(m.equations_system, x0, full_output=True)
        rn = float(np.max(np.abs(m.equations_system(sk))))
        tot += 1
        if ier == 1 and rn < m.RES_TOL:
            if m._is_plausible(sk):
                ok += 1
                maxdev = max(maxdev, float(np.max(np.abs(sk - refv))))
            else:
                implausible += 1
print(f"Random starts: {tot}, converged and plausible {ok}, "
      f"converged but implausible {implausible}, max deviation from reference root {maxdev:.1e} K")
