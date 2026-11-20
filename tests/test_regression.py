"""Regression tests: the optimized code reproduces the baseline's results.

The baseline is the code at commit bd9ce33 (docs/BASELINE.md), extracted with
`git archive` and imported under its own module names, so both versions run
side by side in this process. The tests check that:

1. the full study (seed 607, n = 100/500/1000/5000, 1000 replications) gives
   identical simulated data, per-replication estimates equal to rounding, and
   matching summary statistics;
2. the distributions of beta hat 2 and the plug-in estimate are the same,
   both on the same draws (two-sample KS statistic exactly 0) and on
   independent re-randomizations of the same frozen populations (KS and
   Welch tests do not reject);
3. edge cases (small n, where the baseline can crash or fail silently) give
   identical estimates wherever the optimized code reports convergence, and
   a flagged NaN wherever the baseline crashed or diverged;
4. the threaded parallel estimate equals the serial one.

Skipped if git or the baseline commit is unavailable.
"""
import importlib
import io
import math
import subprocess
import sys
import tarfile
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import analyze  # noqa: E402
import methods  # noqa: E402
import simulation  # noqa: E402
from dgps import make_population, randomize  # noqa: E402

BASELINE_REV = "bd9ce33"
MODULES = ("dgps", "metrics", "methods", "simulation", "analyze")
B2, PLUG, STATUS = 3, 5, 7          # estimates columns: beta hat 2, plug-in, status


# ---------------------------------------------------------------- helpers ----

def _load_baseline(tmp):
    """Import the baseline's src modules without disturbing the current ones."""
    try:
        blob = subprocess.run(["git", "archive", BASELINE_REV, "src"], cwd=ROOT,
                              check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"baseline commit {BASELINE_REV} not available")
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        tar.extractall(tmp)
    saved = {m: sys.modules.pop(m) for m in MODULES if m in sys.modules}
    sys.path.insert(0, str(tmp / "src"))
    try:
        return {m: importlib.import_module(m) for m in MODULES}
    finally:
        sys.path.remove(str(tmp / "src"))
        for m in MODULES:
            sys.modules.pop(m, None)
        sys.modules.update(saved)


def ks_2samp(a, b):
    """Two-sample Kolmogorov-Smirnov statistic and asymptotic p-value."""
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    d = np.max(np.abs(np.searchsorted(a, grid, side="right") / a.size
                      - np.searchsorted(b, grid, side="right") / b.size))
    en = math.sqrt(a.size * b.size / (a.size + b.size))
    lam = (en + 0.12 + 0.11 / en) * d
    if lam < 0.3:                    # the series below does not converge; p is 1 here
        return d, 1.0
    p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return d, min(max(p, 0.0), 1.0)


def welch_p(a, b):
    """Two-sided p-value of Welch's test for equal means (normal approximation)."""
    z = (a.mean() - b.mean()) / math.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    return math.erfc(abs(z) / math.sqrt(2))


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return _load_baseline(tmp_path_factory.mktemp("baseline"))


@pytest.fixture(scope="module")
def study(base):
    """Baseline and optimized runs of the full study, same seed."""
    data_b = base["simulation"].simulate_data()
    data_o = simulation.simulate_data()
    with np.errstate(all="raise"):           # the study must be warning-free
        est_b = base["simulation"].estimate(data_b)
        est_o = simulation.estimate(data_o)
    return data_b, data_o, est_b, est_o


# ------------------------------------------------------------ full study ----

def test_simulated_data_identical(study):
    data_b, data_o, _, _ = study
    assert data_b.keys() == data_o.keys()
    for n in data_b:
        for key in ("Z", "YC", "YT", "X", "Y"):
            np.testing.assert_array_equal(data_o[n][key], data_b[n][key])
        assert data_o[n]["delta"] == pytest.approx(data_b[n]["delta"], rel=1e-15)


def test_per_replication_estimates_match(study):
    _, _, est_b, est_o = study
    assert est_b.shape == (4000, 7) and est_o.shape == (4000, 9)
    np.testing.assert_array_equal(est_o[:, :2], est_b[:, :2])       # n, sim
    assert np.all(est_o[:, STATUS] == methods.CONVERGED)
    np.testing.assert_allclose(est_o[:, 2:7], est_b[:, 2:7], rtol=1e-11, atol=0)


def test_summary_statistics_match(study):
    _, _, est_b, est_o = study
    sb = np.array(base_summary := _baseline_summary(est_b))
    so = np.array(analyze.summarize(est_o))
    assert len(base_summary) == 4
    np.testing.assert_array_equal(so[:, -2:], [[1000, 0]] * 4)    # n_fits, n_failed
    np.testing.assert_allclose(so[:, :8], sb, rtol=1e-10, atol=1e-13)


def _baseline_summary(est):
    # BASELINE's analyze.summarize, restated here so it does not depend on the
    # status column the baseline did not have
    rows = []
    for n in np.unique(est[:, 0]):
        r = est[est[:, 0] == n]
        delta = r[0, 6]
        rows.append([n, delta, r[:, B2].mean(), r[:, B2].std(ddof=1), r[:, B2].mean() - delta,
                     r[:, PLUG].mean(), r[:, PLUG].std(ddof=1), r[:, PLUG].mean() - delta])
    return rows


def test_distributions_identical_on_same_draws(study):
    # Values agree to ~1e-13, which can still swap the order of two nearly tied
    # estimates and give D = 1/1000; rounding to 10 digits removes that.
    _, _, est_b, est_o = study
    for n in (100, 500, 1000, 5000):
        for col in (B2, PLUG):
            a = np.round(est_b[est_b[:, 0] == n, col], 10)
            b = np.round(est_o[est_o[:, 0] == n, col], 10)
            d, p = ks_2samp(a, b)
            assert d == 0.0 and p == 1.0, (n, col, d)


def test_distributions_equivalent_on_independent_draws(study):
    # Same frozen populations, independent randomizations (a different random
    # stream, as batching the randomization would give): the estimators'
    # sampling distributions must be statistically indistinguishable.
    data_b, _, est_b, _ = study
    rng = np.random.default_rng(2027)
    for n in (100, 500, 1000, 5000):
        d = data_b[n]
        X = np.array([randomize(d["Z"], d["YC"], d["YT"], rng=rng)[0] for _ in range(1000)])
        Y = np.where(X == 1, d["YT"], d["YC"])
        est = simulation.estimate({n: dict(d, X=X, Y=Y)})
        assert np.all(est[:, STATUS] == methods.CONVERGED)
        for col in (B2, PLUG):
            a, b = est_b[est_b[:, 0] == n, col], est[:, col]
            # 16 comparisons (4 n x 2 estimators x 2 tests): a 0.001 threshold
            # keeps the family-wise false-alarm rate under ~2%
            _, p_ks = ks_2samp(a, b)
            assert p_ks > 0.001, (n, col, p_ks)
            assert welch_p(a, b) > 0.001, (n, col)
            assert b.std(ddof=1) == pytest.approx(a.std(ddof=1), rel=0.15)


# ------------------------------------------------------------- edge cases ----

def _stress_cases(seed=2026, sizes=(8, 12, 16, 20, 30, 50), pops=5, reps=40):
    rng = np.random.default_rng(seed)
    for n in sizes:
        for _ in range(pops):
            Z, YC, YT = make_population(n, rng)
            for _ in range(reps):
                X, Y, _ = randomize(Z, YC, YT, rng=rng)
                yield n, X, Y, Z


def test_edge_cases(base):
    """Small n: identical where both are defined, flagged where baseline fails."""
    counts = dict(same=0, flagged_crash=0, flagged_diverged=0)
    for n, X, Y, Z in _stress_cases():
        beta_o, info = methods.logit_mle(X, Y, Z, full_output=True)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            try:
                beta_b = base["methods"].logit_mle(X, Y, Z)
            except np.linalg.LinAlgError:
                beta_b = None
        if info["status"] == methods.CONVERGED:
            # the optimized code only reports convergence where the baseline
            # also gives a well-defined answer, and then the answers agree
            assert beta_b is not None, (n, X, Y)
            np.testing.assert_allclose(beta_o, beta_b, rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(methods.plug_in(beta_o, Z)[0],
                                       base["methods"].plug_in(beta_b, Z)[0], rtol=1e-8)
            counts["same"] += 1
        else:
            assert np.all(np.isnan(beta_o))
            if beta_b is None:
                counts["flagged_crash"] += 1
            else:
                counts["flagged_diverged"] += 1
    assert counts["same"] > 0 and counts["flagged_crash"] > 0, counts


def test_true_delta_identical(base):
    rng = np.random.default_rng(3)
    for n in (20, 100, 5000):
        _, YC, YT = make_population(n, rng)
        assert simulation.true_delta(YC, YT) == pytest.approx(
            base["metrics"].true_delta(YC, YT), rel=1e-14)


# --------------------------------------------------------------- parallel ----

def test_parallel_matches_serial(study):
    _, data_o, _, est_o = study
    for workers in (2, 4):
        est_p = simulation.estimate_parallel(data_o, workers)
        np.testing.assert_array_equal(est_p[:, [0, 1, 6, 7, 8]], est_o[:, [0, 1, 6, 7, 8]])
        np.testing.assert_allclose(est_p[:, 2:6], est_o[:, 2:6], rtol=1e-12, atol=0)
