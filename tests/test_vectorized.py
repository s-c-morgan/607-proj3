"""Batched estimation (logit_mle_batch, 2-D plug_in) agrees with single fits."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dgps import make_population, randomize  # noqa: E402
from methods import (CONVERGED, FITTED_BOUNDARY, MAX_ITER, SEPARATED,  # noqa: E402
                     SINGULAR_HESSIAN, logit_mle, logit_mle_batch, plug_in,
                     separated_by_treatment)
from simulation import estimate, simulate_data  # noqa: E402


def _experiments(n, S, seed):
    rng = np.random.default_rng(seed)
    Z, YC, YT = make_population(n, rng)
    X, Y = zip(*(randomize(Z, YC, YT, rng=rng)[:2] for _ in range(S)))
    return np.array(X), np.array(Y), Z


def test_batch_matches_single_fits():
    for n, seed in ((100, 11), (1000, 12)):
        X, Y, Z = _experiments(n, 50, seed)
        with np.errstate(all="raise"):
            B, info = logit_mle_batch(X, Y, Z)
        assert np.all(info["status"] == CONVERGED)
        for s in range(len(X)):
            beta, single = logit_mle(X[s], Y[s], Z, full_output=True)
            np.testing.assert_allclose(B[s], beta, rtol=1e-12, atol=1e-14)
            assert info["n_iter"][s] == single["n_iter"]


def test_chunking_does_not_change_results():
    X, Y, Z = _experiments(100, 40, 13)
    B, info = logit_mle_batch(X, Y, Z)
    B2, info2 = logit_mle_batch(X, Y, Z, chunk_elements=700)   # 7 rows per chunk
    np.testing.assert_allclose(B2, B, rtol=1e-13, atol=1e-15)
    np.testing.assert_array_equal(info2["status"], info["status"])
    np.testing.assert_array_equal(info2["n_iter"], info["n_iter"])


def test_mixed_batch_gets_status_per_row():
    # Every row of one n = 8 batch shares Z, so each failure is built around it.
    Z = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    X_ok = np.array([0, 1, 1, 0, 1, 1, 1, 1])
    Y_ok = np.array([0, 1, 0, 1, 1, 0, 1, 1])
    X_sep = np.array([0, 0, 1, 1, 1, 1, 1, 1])
    Y_sep = np.array([1, 1, 0, 1, 0, 1, 1, 1])              # control arm constant
    X_z = np.array([0, 1, 1, 1, 0, 1, 1, 1])
    Y_z = (Z > 0.5).astype(int)                             # separated through Z
    X = np.array([X_ok, X_sep, X_z, X_ok])
    Y = np.array([Y_ok, Y_sep, Y_z, Y_ok])
    assert list(separated_by_treatment(X, Y)) == [False, True, False, False]
    with np.errstate(all="ignore"):
        B, info = logit_mle_batch(X, Y, Z)
    st = info["status"]
    assert st[0] == CONVERGED and st[3] == CONVERGED
    assert st[1] == SEPARATED and info["n_iter"][1] == 0
    assert st[2] in (MAX_ITER, FITTED_BOUNDARY, SINGULAR_HESSIAN)
    assert np.all(np.isnan(B[1:3])) and np.all(np.isfinite(B[[0, 3]]))
    np.testing.assert_array_equal(B[0], B[3])
    np.testing.assert_allclose(B[0], logit_mle(X_ok, Y_ok, Z), rtol=1e-12)


def test_singular_hessian_in_batch_does_not_affect_other_rows():
    # A design with X equal to Z is rank 2; a separate batch row is fine.
    Xs = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    Z = Xs.astype(float)
    X = np.array([Xs, Xs])
    Y = np.array([[0, 1, 1, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 1, 1, 0]])
    B, info = logit_mle_batch(X, Y, Z)
    assert list(info["status"]) == [SINGULAR_HESSIAN, SINGULAR_HESSIAN]
    assert np.all(np.isnan(B))


def test_plug_in_2d_matches_1d():
    rng = np.random.default_rng(14)
    Z = rng.uniform(size=300)
    B = rng.normal(size=(25, 3))
    B[4] = np.nan                                           # a failed fit
    delta, aT, aC = plug_in(B, Z)
    assert delta.shape == aT.shape == aC.shape == (25,)
    for s in range(25):
        d1, t1, c1 = plug_in(B[s], Z)
        np.testing.assert_array_equal([delta[s], aT[s], aC[s]], [d1, t1, c1])
    assert np.isnan(delta[4]) and np.isnan(aT[4]) and np.isnan(aC[4])


def test_estimate_layout():
    data = simulate_data(sample_sizes=[60, 120], n_sims=30, seed=5)
    est = estimate(data)
    assert est.shape == (60, 9)
    np.testing.assert_array_equal(est[:30, 0], 60)
    np.testing.assert_array_equal(est[30:, 1], np.arange(1, 31))
    d = data[120]
    for s in (0, 17):
        beta, info = logit_mle(d["X"][s], d["Y"][s], d["Z"], full_output=True)
        row = est[30 + s]
        np.testing.assert_allclose(row[2:5], beta, rtol=1e-12, equal_nan=True)
        np.testing.assert_allclose(row[5], plug_in(beta, d["Z"])[0], rtol=1e-12,
                                   equal_nan=True)
        assert row[6] == d["delta"] and row[7] == info["status"] and row[8] == info["n_iter"]
    np.testing.assert_array_equal(d["Y"], d["X"] * d["YT"] + (1 - d["X"]) * d["YC"])
