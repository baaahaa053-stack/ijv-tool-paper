"""
Regression / reference tests for the IJV Thermal Comfort Tool.

Run from the repository root:
    pip install pytest pythermalcomfort
    pytest tests/ -v
"""
import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from thermal_model import ThermalModel, calc_PD, calc_E, calc_PMV  # noqa: E402

# ── PMV/PPD vs. the ISO 7730 model in pythermalcomfort ───────────────────────
# (ta, tr, v, RH, met, clo)
PMV_CASES = [
    (19.0, 19.0, 0.0, 40, 1.0, 1.0),
    (25.0, 25.0, 0.1, 50, 1.0, 0.6),
    (25.7, 30.9, 0.3, 50, 1.0, 0.8),   # close to the single-case example
    (22.0, 22.0, 0.1, 60, 1.2, 0.5),
    (27.0, 27.0, 0.3, 60, 1.0, 0.5),
    (24.0, 28.0, 0.1, 50, 1.2, 1.0),
    (26.0, 30.0, 0.3, 50, 1.0, 0.8),
]


@pytest.mark.parametrize("ta,tr,v,rh,met,clo", PMV_CASES)
def test_pmv_ppd_against_pythermalcomfort(ta, tr, v, rh, met, clo):
    pmv_ppd_iso = pytest.importorskip("pythermalcomfort.models").pmv_ppd_iso
    ref = pmv_ppd_iso(tdb=ta, tr=tr, vr=v, rh=rh, met=met, clo=clo,
                      model="7730-2005", limit_inputs=False)
    # calc_PMV takes the three surface temperatures separately; with
    # tf = tc = tp = tr the area-weighted MRT is exactly tr.
    pmv, ppd = calc_PMV(ta, tr, tr, tr, a=5, b=5, N=32, Ap=2.56,
                        M=met * 58.15, W=0.0, I_cl=clo * 0.155, U=v, RH=rh)
    assert pmv == pytest.approx(float(ref["pmv"]), abs=0.011)
    assert ppd == pytest.approx(float(ref["ppd"]), abs=0.11)


# ── Draft rate (Fanger et al. 1988) ──────────────────────────────────────────
def test_draft_rate_value():
    # (34 - 25.665) * (0.3 - 0.05)^0.62 * (0.37 * 0.3 * 10 + 3.14) = 15.0 %
    assert calc_PD(25.665, 0.3, 10.0) == pytest.approx(15.0, abs=0.05)


def test_draft_rate_limits():
    assert calc_PD(34.0, 0.3) == 0.0        # no draught when ta >= 34 °C
    assert 0.0 <= calc_PD(10.0, 5.0) <= 100.0


# ── Energy utilisation coefficient ───────────────────────────────────────────
def test_energy_coefficient():
    assert calc_E(28.0, 26.0, 18.0) == pytest.approx(10.0 / 8.0)
    assert calc_E(28.0, 18.0, 18.0) != calc_E(28.0, 18.0, 18.0)   # nan, no ZeroDivisionError


# ── Single-case example of the paper (Section 3.1) ───────────────────────────
@pytest.fixture(scope="module")
def example():
    m = ThermalModel()
    m.set_case(a=5.0, b=5.0, hr=3.6, Ap=2.56, Pt=120.0, N=32,
               ts=18.0, vs=2.0, S=0.175, hp=1.5)
    return m, m.solve()


def test_example_converges(example):
    _, res = example
    assert res["converged"]
    assert res["res_norm"] < 1e-3


def test_example_values(example):
    m, res = example
    assert m.Qs == pytest.approx(0.350, abs=1e-3)
    assert m.ACH == pytest.approx(14.00, abs=0.01)
    assert res["toz"] == pytest.approx(25.7, abs=0.05)
    assert res["tf"] == pytest.approx(26.1, abs=0.05)
    assert res["tc"] == pytest.approx(31.4, abs=0.05)
    assert res["PMV"] == pytest.approx(0.73, abs=0.005)
    assert res["PPD"] == pytest.approx(16.21, abs=0.01)
    assert res["E"] == pytest.approx(1.268, abs=0.0005)
    assert calc_PD(res["toz"], 0.3) == pytest.approx(15.0, abs=0.05)


def test_toz_minus_ts_nearly_independent_of_ts():
    """toz - ts is almost invariant with ts at fixed vs (Fig. 4, Reviewer 4 #7)."""
    diffs = []
    for ts in (15.0, 17.0, 19.0):
        m = ThermalModel()
        m.set_case(5.0, 5.0, 3.6, 2.56, 120.0, 32, ts, 1.8, 0.175, 1.5)
        r = m.solve()
        assert r["converged"]
        diffs.append(r["toz"] - ts)
    assert max(diffs) - min(diffs) < 0.01


# ── Solver robustness (Reviewer comment 15) ──────────────────────────────────
def test_converges_over_validated_envelope():
    """All grid cases inside the validated ranges converge with small residuals."""
    for ts, vs, N in itertools.product(np.arange(15.0, 19.01, 1.0),
                                       np.round(np.arange(1.6, 2.81, 0.3), 2),
                                       [32, 48, 60]):
        m = ThermalModel()
        m.set_case(5.0, 5.0, 3.6, 2.56, 120.0, N, ts, vs, 0.175, 1.5)
        r = m.solve()
        assert r["converged"], (ts, vs, N)
        assert r["res_norm"] < 1e-3, (ts, vs, N)


def test_unique_root_for_different_initial_guesses():
    """Different initial guesses reach the same root (no multiple roots)."""
    m = ThermalModel()
    m.set_case(5.0, 5.0, 3.6, 2.56, 120.0, 32, 17.0, 2.2, 0.175, 1.5)
    ref = m.solve()
    for shift in (-6.0, -3.0, 3.0, 6.0, 10.0):
        r = m.solve(T0=m.T0_DEFAULT + shift, multi_start=False)
        assert r["converged"]
        assert r["toz"] == pytest.approx(ref["toz"], abs=1e-3)
