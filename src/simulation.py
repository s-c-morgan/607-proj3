"""Replicate the Freedman (2008) Section 5 simulations.

Step 1 (simulate_data): for each n, the population (Z, YC, YT) is generated
once and frozen, and 1000 randomized experiments are drawn on it.
Saved to data/simulated.pk as {n: dict} with keys
    Z, YC, YT, delta (true differential log odds),
    X, Y (n_sims x n arrays: assignment and observed response per experiment)

Step 2 (estimate): the logit MLE and plug-in estimator are computed for every
experiment. Saved to results/raw/estimates.pk as a numpy array with columns
    n, simulation number, beta hat 1, beta hat 2, beta hat 3, plug-in estimate,
    true delta, fit status, Newton iterations
Fit status is a methods.STATUS code (0 = converged). Failed fits have NaN
betas and plug-in estimate and are excluded by analyze.py.

`python simulation.py --workers K` runs step 2 across K threads
(estimate_parallel); the output is the same as the serial run.
"""
import argparse
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from dgps import make_population, randomize
from metrics import true_delta
from methods import logit_mle_batch, plug_in

SAMPLE_SIZES = [100, 500, 1000, 5000]
N_SIMS = 1000
SEED = 607
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "simulated.pk"
RESULTS_PATH = ROOT / "results" / "raw" / "estimates.pk"


def simulate_data(sample_sizes=SAMPLE_SIZES, n_sims=N_SIMS, seed=SEED):
    rng = np.random.default_rng(seed)
    data = {}
    for n in sample_sizes:
        Z, YC, YT = make_population(n, rng)
        X = np.empty((n_sims, n), dtype=int)
        for s in range(n_sims):
            X[s], _, _ = randomize(Z, YC, YT, frac_treated=0.75, rng=rng)
        Y = np.where(X == 1, YT, YC)
        data[n] = dict(Z=Z, YC=YC, YT=YT, delta=true_delta(YC, YT), X=X, Y=Y)
    return data


def estimate(data):
    """All experiments for a given n are fitted in one batched call."""
    return np.vstack([_estimate_block(n, 0, d["X"], d["Y"], d["Z"], d["delta"])
                      for n, d in data.items()])


def estimate_parallel(data, workers=4):
    """Same output as ``estimate``, with the work split across threads.

    Each n is cut into blocks of replications holding about the same number
    of (replication, subject) elements, so n = 5000 (three quarters of the
    work) is spread over several threads. NumPy releases the GIL inside its
    array arithmetic, so the blocks run concurrently. Blocks come back in
    order, so rows are ordered by n and then simulation number, as in
    ``estimate``.

    Threads rather than processes: starting a process pool on macOS (spawn)
    costs ~0.8 s, more than the whole serial estimate, and fork-based pools
    crash under Apple's Accelerate BLAS. See docs/OPTIMIZATION.md.
    """
    total = sum(d["X"].size for d in data.values())
    per_task = max(1, total // (2 * workers))
    tasks = []
    for n, d in data.items():
        rows = max(1, per_task // n)
        for lo in range(0, d["X"].shape[0], rows):
            tasks.append((n, lo, d["X"][lo:lo + rows], d["Y"][lo:lo + rows],
                          d["Z"], d["delta"]))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        blocks = list(pool.map(_estimate_block, *zip(*tasks)))
    return np.vstack(blocks)


def _estimate_block(n, first_sim, X, Y, Z, delta):
    """Estimates for experiments first_sim+1, ..., first_sim+len(X) of one n."""
    S = X.shape[0]
    beta, info = logit_mle_batch(X, Y, Z)
    delta_tilde, _, _ = plug_in(beta, Z)
    return np.column_stack([
        np.full(S, n), np.arange(first_sim + 1, first_sim + S + 1), beta, delta_tilde,
        np.full(S, delta), info["status"], info["n_iter"]]).astype(float)


def _save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=0,
                    help="estimate in parallel with this many threads (0 = serial)")
    args = ap.parse_args()

    data = simulate_data()
    _save(data, DATA_PATH)
    print(f"Saved simulated data for n={list(data)} to {DATA_PATH}")

    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    results = estimate_parallel(data, args.workers) if args.workers else estimate(data)
    _save(results, RESULTS_PATH)
    print(f"Saved {results.shape[0]} rows to {RESULTS_PATH}")
