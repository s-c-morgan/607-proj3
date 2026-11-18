"""Per-stage time complexity, as measured by cProfile.

For each n in a grid, the Unit 2 pipeline (simulate_data -> pickle save/load ->
estimate -> analyze.summarize) is run under cProfile for a single sample size,
and the cumulative time of each stage's function is read from the profile.
S = 200 replications at every n (1000 would need ~1.6 GB at n = 100k); times
are divided by S, giving per-replication cost.

Writes results/profiling/stage_scaling.csv, stage_scaling_fit.txt and
stage_scaling.png.
"""
import cProfile
import io
import pickle
import pstats

import matplotlib.pyplot as plt
import numpy as np

from _common import OUT

import analyze
import simulation

GRID = [50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]
STUDY_NS = [100, 500, 1000, 5000]

# stage label -> (file suffix, function name) as cProfile records them
STAGES = {
    "simulate: randomize": ("dgps.py", "randomize"),
    "estimate: logit_mle": ("methods.py", "logit_mle"),
    "estimate: plug_in": ("methods.py", "plug_in"),
    "save/load: pickle": ("~", "<built-in method _pickle.dumps>|<built-in method _pickle.loads>"),
    "analyze: summarize": ("analyze.py", "summarize"),
}


S_FIXED = 200   # same S at every n, so n is the only thing that changes


def pipeline(n, S):
    data = simulation.simulate_data(sample_sizes=[n], n_sims=S, seed=607)
    blob = pickle.dumps(data)                     # the save + reload step of
    data = pickle.loads(blob)                     # simulation.py, in memory
    est = simulation.estimate(data)
    blob = pickle.dumps(est)
    est = pickle.loads(blob)
    analyze.summarize(est)


def cumtimes(prof):
    stats = pstats.Stats(prof).stats   # {(file, line, func): (cc, nc, tt, ct, callers)}
    out = {k: 0.0 for k in STAGES}
    for (fname, _, func), (_, _, _, ct, _) in stats.items():
        for label, (suffix, names) in STAGES.items():
            if fname.endswith(suffix) and func in names.split("|"):
                out[label] += ct
        if fname.endswith("profile_stages.py") and func == "pipeline":
            out["total"] = ct
    return out


def main():
    pipeline(100, 10)  # warm-up
    rows = []
    for n in GRID:
        S = S_FIXED
        prof = cProfile.Profile()
        prof.runcall(pipeline, n, S)
        t = cumtimes(prof)
        rows.append(dict(n=n, S=S, **{k: v / S for k, v in t.items()}))
        print(f"n={n:>6} S={S:>4} " + "  ".join(
            f"{k.split(': ')[-1]}={1e3 * v / S:.4f}ms" for k, v in t.items()))

    labels = list(STAGES) + ["total"]
    with open(OUT / "stage_scaling.csv", "w") as f:
        f.write("n,S," + ",".join(l.replace(": ", "_").replace("/", "_")
                                   for l in labels) + "\n")
        for r in rows:
            f.write(f"{r['n']},{r['S']}," + ",".join(f"{r[l]:.6g}" for l in labels) + "\n")

    ns = np.array([r["n"] for r in rows], float)
    fit_lines = ["log-log slope of per-replication cumulative time (cProfile)",
                 f"{'stage':22s} {'n<=1000':>8s} {'n>=5000':>8s}"]
    slopes = {}
    for l in labels:
        ts = np.array([r[l] for r in rows])
        lo, hi = ns <= 1000, ns >= 5000
        if np.any(ts <= 0):
            continue
        s_lo = np.polyfit(np.log10(ns[lo]), np.log10(ts[lo]), 1)[0]
        s_hi = np.polyfit(np.log10(ns[hi]), np.log10(ts[hi]), 1)[0]
        slopes[l] = s_hi
        fit_lines.append(f"{l:22s} {s_lo:8.2f} {s_hi:8.2f}")
    text = "\n".join(fit_lines)
    (OUT / "stage_scaling_fit.txt").write_text(text + "\n")
    print(text)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    styles = ["C0s-", "C2v-", "C1^-", "C4D-", "C3P-", "ko-"]
    for l, st in zip(labels, styles):
        lw = 2 if l == "total" else 1.3
        name = "Total pipeline" if l == "total" else l
        ax.loglog(ns, [1e3 * r[l] for r in rows], st, lw=lw, ms=5,
                  label=f"{name}  (slope {round(slopes[l], 2) + 0.0:.2f} for n≥5000)")
    ref = np.array([5000, ns.max()])
    t0 = 1e3 * rows[GRID.index(5000)]["total"]
    ax.loglog(ref, t0 * ref / 5000, "k:", lw=1, label="O(n) reference")
    for n in STUDY_NS:
        ax.axvline(n, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("Sample size n (dotted lines: study values)")
    ax.set_ylabel("cProfile cumulative time per replication (ms)")
    ax.set_title("Time complexity of each pipeline stage (cProfile)")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "stage_scaling.png", dpi=150)
    print(f"Saved {OUT / 'stage_scaling.png'}")


if __name__ == "__main__":
    main()
