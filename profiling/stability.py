"""Numerical-stability audit of the Unit 2 simulation study.

Part A re-runs the exact study (seed 607, n = 100/500/1000/5000, 1000 sims)
with every numpy floating-point event counted (np.errstate(all="call")),
Python warnings recorded, and exceptions caught. Each fit's status code from
logit_mle(full_output=True) is recorded. For converged fits we also replay the
Newton-Raphson loop (identical arithmetic to methods.logit_mle) to log the
final gradient norm, Hessian condition number and fitted-probability range,
and check the replay reproduces logit_mle exactly.

Part B stresses the same code outside the study's settings (smaller n) to find
the conditions under which it does break down.

Writes results/profiling/stability_study.csv, stability_stress.csv and
stability_log.txt.
"""
import warnings
from collections import Counter

import numpy as np

from _common import OUT

from dgps import make_population, randomize
from methods import (CONVERGED, STATUS, _expit, _expit_and_weight, logit_mle,
                     plug_in)
from simulation import simulate_data

TOL, MAX_ITER = 1e-10, 100
fp_events = Counter()


def _fp_handler(kind, flag):
    fp_events[kind] += 1


np.seterrcall(_fp_handler)


def newton_trace(X, Y, Z):
    """Replays methods.logit_mle on a fit known to converge (same arithmetic)."""
    M = np.column_stack([np.ones(len(Y)), X, Z])
    beta = np.zeros(3)
    conds = []
    converged = False
    for k in range(1, MAX_ITER + 1):
        p, w = _expit_and_weight(M @ beta)
        grad = M.T @ (Y - p)
        hess = (M * w[:, None]).T @ M
        conds.append(np.linalg.cond(hess))
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < TOL:
            converged = True
            break
    p = _expit(M @ beta)
    return beta, dict(iters=k, converged=converged,
                      final_step=float(np.max(np.abs(step))),
                      grad_norm=float(np.max(np.abs(M.T @ (Y - p)))),
                      cond_max=float(max(conds)), cond_final=float(conds[-1]),
                      p_min=float(p.min()), p_max=float(p.max()),
                      beta_absmax=float(np.max(np.abs(beta))))


def one_rep(X, Y, Z):
    """Run the real estimator code, catching warnings/exceptions."""
    out = dict(exception="", warnings="",
               arm_constant=bool(np.ptp(Y[X == 1]) == 0 or np.ptp(Y[X == 0]) == 0))
    with warnings.catch_warnings(record=True) as w, np.errstate(all="call"):
        warnings.simplefilter("always")
        try:
            beta, info = logit_mle(X, Y, Z, full_output=True)
            delta, aT, aC = plug_in(beta, Z)
            out.update(beta=beta, delta=delta, aT=aT, aC=aC,
                       status=STATUS[info["status"]], iters=info["n_iter"],
                       converged=info["status"] == CONVERGED)
        except Exception as e:  # noqa: BLE001
            out["exception"] = type(e).__name__
        out["warnings"] = ";".join(sorted({str(x.category.__name__) for x in w}))
    if not out["exception"]:
        out["finite"] = bool(np.all(np.isfinite(out["beta"])) and np.isfinite(out["delta"]))
        if out["converged"]:
            with np.errstate(all="ignore"):
                b2, diag = newton_trace(X, Y, Z)
            diag.pop("iters"), diag.pop("converged")
            out.update(diag)
            out["replay_matches"] = bool(np.array_equal(b2, out["beta"]))
    return out


def summarize(rows, label):
    lines = [f"--- {label}: {len(rows)} replications"]
    exc = Counter(r["exception"] for r in rows if r["exception"])
    warn = Counter(r["warnings"] for r in rows if r["warnings"])
    ok = [r for r in rows if not r["exception"]]
    lines.append(f"exceptions: {dict(exc) or 'none'}")
    lines.append(f"python warnings (numpy fp errors go to the counter below instead): "
                 f"{dict(warn) or 'none'}")
    sep = [r for r in rows if r["arm_constant"]]
    bad = [r for r in rows if r["exception"] or not r.get("converged", True)
           or not r.get("finite", True)]
    lines.append(f"an arm with constant Y (separation by X): {len(sep)}; "
                 f"failures (exception / no convergence / non-finite): {len(bad)}, "
                 f"of which in separated data: {sum(r['arm_constant'] for r in bad)}")
    st = Counter(r["status"] for r in rows if not r["exception"])
    lines.append("fit status: " + ", ".join(f"{k} {v}" for k, v in sorted(st.items())))
    failed = [r for r in rows if not r["exception"] and not r["converged"]]
    conv = [r for r in rows if not r["exception"] and r["converged"]]
    lines.append(f"silent failures (failed fit with non-NaN beta): "
                 f"{sum(not np.all(np.isnan(r['beta'])) for r in failed)}; "
                 f"converged fits with non-finite estimate: "
                 f"{sum(not r['finite'] for r in conv)}")
    ok = conv
    if ok:
        it = np.array([r["iters"] for r in ok])
        lines.append(f"Newton iterations (converged fits): min {it.min()}, "
                     f"mean {it.mean():.2f}, max {it.max()}")
        lines.append(f"replay != logit_mle: {sum(not r['replay_matches'] for r in ok)}")
        g = np.array([r["grad_norm"] for r in ok if r["converged"]])
        if g.size:
            lines.append(f"max |gradient| at solution (converged): {g.max():.2e}")
        c = np.array([r["cond_max"] for r in ok])
        lines.append(f"Hessian cond. number (max over iters): median {np.median(c):.3g}, "
                     f"max {c.max():.3g}")
        lines.append(f"fitted p range: [{min(r['p_min'] for r in ok):.3g}, "
                     f"{max(r['p_max'] for r in ok):.6g}]; "
                     f"max |beta|: {max(r['beta_absmax'] for r in ok):.3g}")
        fin = [r for r in ok if r["finite"]]
        if fin:
            lines.append(f"plug-in alpha range: aT [{min(r['aT'] for r in fin):.4f}, "
                         f"{max(r['aT'] for r in fin):.4f}], aC [{min(r['aC'] for r in fin):.4f}, "
                         f"{max(r['aC'] for r in fin):.4f}]")
    return lines


def write_csv(path, rows, extra):
    keys = list(extra) + ["arm_constant", "exception", "warnings", "status", "iters",
                          "converged", "final_step",
                          "grad_norm", "cond_max", "p_min", "p_max", "beta_absmax"]
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")


if __name__ == "__main__":
    log = []

    # ---------- Part A: the study exactly as submitted ----------
    fp_events.clear()
    with warnings.catch_warnings(record=True) as w_sim, np.errstate(all="call"):
        warnings.simplefilter("always")
        data = simulate_data()
    log.append("=== Part A: Unit 2 study (seed 607, 1000 sims per n) ===")
    log.append(f"simulate_data: python warnings {len(w_sim)}, fp events {dict(fp_events) or 'none'}")
    for n, d in data.items():
        fp_events.clear()
        rows = []
        for s in range(d["X"].shape[0]):
            r = one_rep(d["X"][s], d["Y"][s], d["Z"])
            r.update(n=n, sim=s + 1)
            rows.append(r)
        log += summarize(rows, f"n={n} (true delta {d['delta']:.4f})")
        log.append(f"numpy floating-point events (incl. underflow): {dict(fp_events) or 'none'}")
        write_csv(OUT / f"stability_study_n{n}.csv", rows, ["n", "sim"])

    # ---------- Part B: stress test outside the study settings ----------
    log.append("")
    log.append("=== Part B: stress test, 1000 experiments per n, frac_treated=0.75 ===")
    rng = np.random.default_rng(2026)
    stress_rows = []
    for n in [8, 12, 16, 20, 30, 50]:
        fp_events.clear()
        rows = []
        for pop in range(10):          # 10 populations x 100 randomizations
            Z, YC, YT = make_population(n, rng)
            for s in range(100):
                X, Y, _ = randomize(Z, YC, YT, rng=rng)
                r = one_rep(X, Y, Z)
                r.update(n=n, pop=pop, sim=s + 1)
                rows.append(r)
        log += summarize(rows, f"n={n}")
        log.append(f"numpy floating-point events (incl. underflow): {dict(fp_events) or 'none'}")
        stress_rows += rows
    write_csv(OUT / "stability_stress.csv", stress_rows, ["n", "pop", "sim"])

    text = "\n".join(log)
    (OUT / "stability_log.txt").write_text(text + "\n")
    print(text)
