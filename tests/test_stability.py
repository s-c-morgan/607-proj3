"""Numerical safeguards in methods.py, metrics.py and the analysis pipeline."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze import summarize  # noqa: E402
from dgps import make_population, randomize  # noqa: E402
from methods import (CONVERGED, FITTED_BOUNDARY, MAX_ITER, SEPARATED,  # noqa: E402
                     SINGULAR_HESSIAN, STATUS,
                     ConvergenceWarning, _expit, _expit_and_weight, logit_mle,
                     plug_in, separated_by_treatment)
from metrics import true_delta  # noqa: E402

EXTREME = np.array([-1e308, -1e4, -800.0, -745.0, -40.0, -1.0, 0.0,
                    1.0, 40.0, 745.0, 800.0, 1e4, 1e308])


def _naive_expit(t):
    return 1.0 / (1.0 + np.exp(-t))


def _study_like(n, seed):
    rng = np.random.default_rng(seed)
    Z, YC, YT = make_population(n, rng)
    X, Y, _ = randomize(Z, YC, YT, rng=rng)
    return X, Y, Z


# ---------- overflow ----------

def test_naive_expit_overflows():
    # documents the problem the fix addresses
    with np.errstate(over="raise"), pytest.raises(FloatingPointError):
        _naive_expit(np.array([-800.0]))


def test_expit_no_overflow_at_extremes():
    with np.errstate(over="raise"):
        p = _expit(EXTREME)
    assert np.all(np.isfinite(p))
    assert np.all((p >= 0) & (p <= 1))
    assert p[0] == 0.0 and p[-1] == 1.0
    assert p[6] == 0.5


def test_expit_matches_naive_in_safe_range():
    t = np.linspace(-30, 30, 2001)
    np.testing.assert_allclose(_expit(t), _naive_expit(t), rtol=1e-14, atol=0)


def test_plug_in_no_overflow_for_very_negative_predictor():
    Z = np.linspace(0.01, 0.99, 50)
    with np.errstate(over="raise", divide="ignore", invalid="ignore"):
        _, aT, aC = plug_in(np.array([-900.0, 0.0, 5.0]), Z)
    assert aT == 0.0 and aC == 0.0


# ---------- Newton weights (underflow / cancellation) ----------

def test_weights_stay_positive_past_cancellation_point():
    t = np.array([-700.0, -40.0, 0.0, 40.0, 700.0])
    p, w = _expit_and_weight(t)
    naive_w = _naive_expit(t) * (1 - _naive_expit(t))
    assert naive_w[3] == 0.0                 # 1 - p cancels to 0 at t = 40
    assert np.all(w > 0)                     # stable weight does not
    np.testing.assert_allclose(w, np.exp(-np.abs(t)) / (1 + np.exp(-np.abs(t))) ** 2,
                               rtol=1e-14)
    np.testing.assert_array_equal(p, _expit(t))


# ---------- separation and convergence status ----------

def test_separation_detected():
    X = np.array([0, 0, 1, 1, 1])
    assert separated_by_treatment(X, np.array([1, 1, 0, 1, 0]))   # control constant
    assert separated_by_treatment(X, np.array([0, 1, 1, 1, 1]))   # treated constant
    assert separated_by_treatment(np.ones(5, int), np.array([0, 1, 1, 0, 1]))  # empty arm
    assert not separated_by_treatment(X, np.array([0, 1, 0, 1, 1]))


def test_separated_fit_returns_nan_and_status():
    X = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    Y = np.array([1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0])
    Z = np.linspace(0.05, 0.95, 12)
    with warnings.catch_warnings():
        warnings.simplefilter("error")       # full_output: no warning, no crash
        beta, info = logit_mle(X, Y, Z, full_output=True)
    assert info["status"] == SEPARATED and info["n_iter"] == 0
    assert np.all(np.isnan(beta))
    delta, aT, aC = plug_in(beta, Z)
    assert np.isnan(delta) and np.isnan(aT) and np.isnan(aC)


def test_failed_fit_warns_without_full_output():
    X = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    Y = np.array([1, 1, 1, 0, 1, 0, 1, 1])
    with pytest.warns(ConvergenceWarning, match="separated"):
        beta = logit_mle(X, Y, np.linspace(0, 1, 8))
    assert np.all(np.isnan(beta))


def test_separation_through_z_is_not_reported_as_converged():
    # Y = 1 exactly when Z > 0.5: both arms have both outcomes, so the arm
    # check passes, but the MLE does not exist (beta3 -> +inf).
    X = np.array([0, 0, 1, 1, 1, 1, 1, 1])
    Z = np.array([0.1, 0.9, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    Y = (Z > 0.5).astype(int)
    assert not separated_by_treatment(X, Y)
    with np.errstate(all="ignore"):
        beta, info = logit_mle(X, Y, Z, full_output=True)
    assert info["status"] != CONVERGED, STATUS[info["status"]]
    assert np.all(np.isnan(beta))


def test_false_convergence_at_boundary_is_flagged():
    # Real n = 12 experiment from the stress test (seed 2026). Separation is
    # through X and Z jointly; Newton's step drops below tol at iteration 12
    # only because every residual has underflowed, with fitted p exactly 0 or
    # 1. (Chosen because the path is well conditioned: some n = 8 cases reach
    # Hessian condition numbers ~1e16, and their failure label then depends
    # on rounding.)
    X = np.array([1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1])
    Y = np.array([1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1])
    Z = np.array([0.23945055502757184, 0.7420916336848787, 0.35325680620445443,
                  0.6944599012936342, 0.3982310889166055, 0.876262662460279,
                  0.7080599070911773, 0.08987604900799306, 0.7609189948878938,
                  0.2928378438147393, 0.20014812715635588, 0.5732693808223385])
    assert not separated_by_treatment(X, Y)
    with np.errstate(all="ignore"):
        beta, info = logit_mle(X, Y, Z, full_output=True)
    assert info["status"] == FITTED_BOUNDARY and info["n_iter"] == 12
    assert np.all(np.isnan(beta))


def test_singular_hessian_caught():
    # Z identical to X makes the design matrix rank 2
    X = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    Y = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    beta, info = logit_mle(X, Y, X.astype(float), full_output=True)
    assert info["status"] == SINGULAR_HESSIAN
    assert np.all(np.isnan(beta))


def test_max_iter_reported():
    X, Y, Z = _study_like(100, 1)
    beta, info = logit_mle(X, Y, Z, max_iter=2, full_output=True)
    assert info["status"] == MAX_ITER and info["n_iter"] == 2
    assert np.all(np.isnan(beta))


def test_converged_fit_satisfies_score_equations():
    for n, seed in ((100, 2), (1000, 3)):
        X, Y, Z = _study_like(n, seed)
        with np.errstate(all="raise"):
            beta, info = logit_mle(X, Y, Z, full_output=True)
        assert info["status"] == CONVERGED and info["n_iter"] <= 10
        M = np.column_stack([np.ones(n), X, Z])
        assert np.max(np.abs(M.T @ (Y - _expit(M @ beta)))) < 1e-8
        np.testing.assert_array_equal(logit_mle(X, Y, Z), beta)


# ---------- plug-in and true delta ----------

def test_plug_in_matches_naive_formula():
    rng = np.random.default_rng(1)
    Z = rng.uniform(size=200)
    beta = np.array([-0.7, 1.3, 2.1])
    aT = _naive_expit(beta[0] + beta[1] + beta[2] * Z).mean()
    aC = _naive_expit(beta[0] + beta[2] * Z).mean()
    naive = np.log(aT / (1 - aT)) - np.log(aC / (1 - aC))
    delta, aT2, aC2 = plug_in(beta, Z)
    np.testing.assert_allclose(delta, naive, rtol=1e-12)
    np.testing.assert_allclose([aT2, aC2], [aT, aC], rtol=1e-14)


def test_true_delta_exact_and_guarded():
    YC = np.array([0, 1, 1, 0])
    YT = np.array([1, 1, 1, 0])
    assert true_delta(YC, YT) == pytest.approx(np.log(3.0))
    with pytest.raises(ValueError, match="log odds undefined"):
        true_delta(YC, np.ones(4, dtype=int))
    with pytest.raises(ValueError, match="log odds undefined"):
        true_delta(np.zeros(4, dtype=int), YT)


# ---------- analysis excludes failed fits ----------

def test_summarize_excludes_and_counts_failures():
    nan = np.nan
    #          n  sim  b1   b2   b3  plug delta status iters
    est = np.array([
        [10, 1, 0.1, 1.0, 0.2, 0.5, 0.4, 0, 6],
        [10, 2, 0.1, 3.0, 0.2, 0.7, 0.4, 0, 6],
        [10, 3, nan, nan, nan, nan, 0.4, SEPARATED, 0],
    ])
    (row,) = summarize(est)
    n, delta, b2_mean, b2_sd, _, plug_mean, _, _, n_fits, n_failed = row
    assert (n, n_fits, n_failed) == (10, 3, 1)
    assert b2_mean == pytest.approx(2.0) and plug_mean == pytest.approx(0.6)
    assert np.isfinite(b2_sd)
