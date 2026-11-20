"""Empirical computational complexity of one simulation replication vs n.

One replication = randomize() + logit_mle() + plug_in() on a frozen population
of size n (exactly what the study repeats n_sims times per n). Each component is
timed separately; we report the median over many repetitions, fit log-log slopes,
and also record the number of Newton-Raphson iterations.

Writes results/profiling/complexity.csv and complexity.png.
"""
import time

import matplotlib.pyplot as plt
import numpy as np

from _common import OUT, done

from dgps import make_population, randomize
from methods import logit_mle, plug_in

GRID = [50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000,
        100_000, 200_000, 500_000]
STUDY_NS = [100, 500, 1000, 5000]


def n_iter(X, Y, Z, tol=1e-10, max_iter=100):
    """Same Newton loop as methods.logit_mle, but returns the iteration count."""
    M = np.column_stack([np.ones(len(Y)), X, Z])
    beta = np.zeros(3)
    for k in range(1, max_iter + 1):
        p = 1.0 / (1.0 + np.exp(-(M @ beta)))
        step = np.linalg.solve((M * (p * (1 - p))[:, None]).T @ M, M.T @ (Y - p))
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            return k
    return max_iter


def time_n(n, rng):
    reps = int(np.clip(2e6 / n, 30, 1000))   # keep each n to ~1-3 s
    Z, YC, YT = make_population(n, rng)
    t_rand, t_mle, t_plug, iters = [], [], [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        X, Y, _ = randomize(Z, YC, YT, rng=rng)
        t1 = time.perf_counter()
        beta = logit_mle(X, Y, Z)
        t2 = time.perf_counter()
        plug_in(beta, Z)
        t3 = time.perf_counter()
        t_rand.append(t1 - t0); t_mle.append(t2 - t1); t_plug.append(t3 - t2)
        if len(iters) < 50:
            iters.append(n_iter(X, Y, Z))
    med = lambda a: float(np.median(a))
    return dict(n=n, reps=reps, randomize=med(t_rand), logit_mle=med(t_mle),
                plug_in=med(t_plug),
                total=med(np.add(np.add(t_rand, t_mle), t_plug)),
                newton_iters=float(np.mean(iters)))


def slope(ns, ts):
    b, a = np.polyfit(np.log10(ns), np.log10(ts), 1)
    return b, a


if __name__ == "__main__":
    rng = np.random.default_rng(607)
    for n in (100, 1000):  # warm-up
        time_n(n, rng)
    rows = [time_n(n, rng) for n in GRID]
    keys = ["n", "reps", "randomize", "logit_mle", "plug_in", "total", "newton_iters"]
    with open(OUT / "complexity.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.6g}" for k in keys) + "\n")

    ns = np.array([r["n"] for r in rows], float)
    report = []
    for comp in ("randomize", "logit_mle", "plug_in", "total"):
        ts = np.array([r[comp] for r in rows])
        b_all, _ = slope(ns, ts)
        big = ns >= 5000
        b_big, _ = slope(ns[big], ts[big])
        small = ns <= 1000
        b_small, _ = slope(ns[small], ts[small])
        report.append(f"{comp:10s} slope all n: {b_all:.2f}   "
                      f"n<=1000: {b_small:.2f}   n>=5000: {b_big:.2f}")
    # Predicted study runtime from per-replication medians (1000 sims per n)
    pred = sum(1000 * np.interp(n, ns, [r["total"] for r in rows]) for n in STUDY_NS)
    report.append(f"Predicted estimate+randomize time for the study (1000 x {STUDY_NS}): "
                  f"{pred:.2f} s")
    for n in STUDY_NS:
        t = np.interp(n, ns, [r["total"] for r in rows])
        report.append(f"  n={n:5d}: {1000 * t:.3f} s ({100 * 1000 * t / pred:.0f}%)")
    text = "\n".join(report)
    (OUT / "complexity_fit.txt").write_text(text + "\n")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    styles = {"total": ("k", "o", "Total per replication"),
              "logit_mle": ("C0", "s", "logit_mle (Newton-Raphson)"),
              "randomize": ("C1", "^", "randomize"),
              "plug_in": ("C2", "v", "plug_in")}
    for comp, (c, m, lab) in styles.items():
        ax.loglog(ns, [r[comp] * 1e3 for r in rows], marker=m, color=c, label=lab)
    big = ns >= 5000
    b, a = slope(ns[big], [r["total"] for r in np.array(rows)[big]])
    xs = np.array([2000, ns.max()])
    ax.loglog(xs, 1e3 * 10 ** a * xs ** b, "k--", lw=1,
              label=f"fit n≥5000: slope {b:.2f}")
    for n in STUDY_NS:
        ax.axvline(n, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("Sample size n (dotted lines: study values)")
    ax.set_ylabel("Median time per replication (ms)")
    ax.set_title("Runtime of one simulation replication vs n")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "complexity.png", dpi=150)
    done(__file__)
