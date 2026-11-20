"""Before/after comparison of the optimizations in docs/OPTIMIZATION.md.

Three versions of src/ are measured on the same study (seed 607):
    baseline    commit bd9ce33 (docs/BASELINE.md)
    stability   commit 4e63d9a (numerical-stability fixes)
    vectorized  the working tree (batched estimation)
The two commits are extracted with `git archive`. Each version runs in its own
subprocess (``--worker``) so that the `methods`/`simulation` modules of
different versions never share an interpreter.

Per version the worker records:
  * the time of each pipeline stage (simulate_data, pickle save, pickle load,
    estimate, summarize), each best of REPEATS
  * estimate() time and peak traced memory (tracemalloc) for each n
  * a cProfile of estimate() on the full study data: top functions by own
    time, and the call tree used for the flame graph

Writes to results/profiling/:
  optimization_compare.csv     per-n time and memory for each version
  optimization_stages.csv      per-stage times for each version
  optimization_profile.txt     cProfile top functions, before vs after
  cprofile_estimate_<version>.prof
  optimization_scaling.png     estimate() time and peak memory vs n
  optimization_stages.png      where the pipeline's time goes, per version
  optimization_flame.png       flame graphs of estimate(), baseline vs vectorized
"""
import argparse
import io
import json
import pickle
import pstats
import subprocess
import sys
import tarfile
import tempfile
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "profiling"
VERSIONS = {"baseline": "bd9ce33", "stability": "4e63d9a", "vectorized": None}
REPEATS = 3

# light-mode categorical palette (slots 1-5), text and surface colors
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------- worker ----

def _best(f, *args):
    ts = []
    for _ in range(REPEATS):
        t = time.perf_counter()
        out = f(*args)
        ts.append(time.perf_counter() - t)
    return min(ts), out


def _label(key):
    fname, _, func = key
    if fname == "~":                                   # C function
        func = func.replace("<built-in method ", "").replace("<method ", "")
        func = func.rstrip(">").replace(" of 'numpy.ndarray' objects", "")
        func = func.replace("numpy.", "np.").replace("_pickle.", "")
        return func
    if "numpy" in fname:                               # NumPy's Python-level helpers
        return f"np.{func}"
    return func                                        # project code (methods, simulation)


def _flame(stats, root, min_frac=0.01, max_depth=6):
    """Rectangles (depth, start, width, label) of the call tree under root.

    A child's width is the cumulative time it spent when called from its
    parent (pstats' per-caller record), so siblings never overlap.
    """
    children = {}
    for key, (_, _, _, _, callers) in stats.items():
        for caller, rec in callers.items():
            children.setdefault(caller, []).append((key, rec[3]))
    total = stats[root][3]
    rects = []

    def walk(key, start, width, depth):
        rects.append((depth, start, width, _label(key)))
        if depth >= max_depth:
            return
        x = start
        for child, ct in sorted(children.get(key, []), key=lambda c: -c[1]):
            w = min(ct, width - (x - start))           # recursion can overcount
            if w >= min_frac * total and child != key:
                walk(child, x, w, depth + 1)
            x += max(w, 0)

    walk(root, 0.0, total, 0)
    return rects


def worker(src):
    import cProfile
    sys.path.insert(0, src)
    import analyze
    import simulation

    res = {"stages": {}, "per_n": {}}
    t, data = _best(simulation.simulate_data)
    res["stages"]["simulate_data"] = t
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "simulated.pk"

        def save():
            with open(path, "wb") as f:
                pickle.dump(data, f)

        def load():
            with open(path, "rb") as f:
                return pickle.load(f)

        res["stages"]["pickle save"], _ = _best(save)
        res["stages"]["pickle load"], _ = _best(load)
    simulation.estimate({100: data[100]})              # warm-up
    res["stages"]["estimate"], est = _best(simulation.estimate, data)
    res["stages"]["summarize"], _ = _best(analyze.summarize, est)

    for n in data:
        sub = {n: data[n]}
        t, _ = _best(simulation.estimate, sub)
        tracemalloc.start()
        simulation.estimate(sub)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        res["per_n"][str(n)] = {"time": t, "peak_mb": peak / 1e6}

    prof = cProfile.Profile()
    prof.runcall(simulation.estimate, data)
    prof.dump_stats(str(OUT / f"cprofile_estimate_{Path(src).parent.name}.prof"))
    stats = pstats.Stats(prof).stats
    root = next(k for k in stats if k[2] == "estimate" and k[0].endswith("simulation.py"))
    res["profile_total"] = stats[root][3]
    res["top"] = [(_label(k), v[1], v[2], v[3]) for k, v in
                  sorted(stats.items(), key=lambda kv: -kv[1][2])[:10]]
    res["flame"] = _flame(stats, root)
    print(json.dumps(res))


# ------------------------------------------------------------------ main ----

def extract(rev, dest):
    blob = subprocess.run(["git", "archive", rev, "src"], cwd=ROOT, check=True,
                          capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        tar.extractall(dest)
    return dest / "src"


def run_all():
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, rev in VERSIONS.items():
            if rev is None:
                src = ROOT / "src"
                # name the .prof after the version, not the folder
                src_dir = Path(tmp) / name
                src_dir.mkdir()
                (src_dir / "src").symlink_to(src)
                src = src_dir / "src"
            else:
                src = extract(rev, Path(tmp) / name)
            out = subprocess.run([sys.executable, __file__, "--worker", str(src)],
                                 capture_output=True, text=True)
            if out.returncode != 0:
                raise RuntimeError(f"measuring the {name} version failed:\n{out.stderr}")
            results[name] = json.loads(out.stdout)
    return results


def write_tables(results):
    names = list(results)
    ns = list(results[names[0]]["per_n"])
    with open(OUT / "optimization_compare.csv", "w") as f:
        f.write("n," + ",".join(f"{v}_time_s,{v}_peak_mb" for v in names) + "\n")
        for n in ns:
            f.write(n + "," + ",".join(
                f"{results[v]['per_n'][n]['time']:.6g},{results[v]['per_n'][n]['peak_mb']:.4g}"
                for v in names) + "\n")
    stages = list(results[names[0]]["stages"])
    with open(OUT / "optimization_stages.csv", "w") as f:
        f.write("version," + ",".join(stages) + ",total\n")
        for v in names:
            st = results[v]["stages"]
            f.write(v + "," + ",".join(f"{st[s]:.4f}" for s in stages)
                    + f",{sum(st.values()):.4f}\n")
    lines = []
    for v in ("baseline", "vectorized"):
        r = results[v]
        lines.append(f"=== {v}: cProfile of simulation.estimate on the study data "
                     f"({r['profile_total']:.3f} s under the profiler) ===")
        lines.append(f"{'function':45s} {'calls':>9s} {'own time (s)':>13s} {'cumulative (s)':>15s}")
        for lab, nc, tt, ct in r["top"]:
            lines.append(f"{lab[:45]:45s} {nc:9d} {tt:13.3f} {ct:15.3f}")
        lines.append("")
    (OUT / "optimization_profile.txt").write_text("\n".join(lines))


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def plot_scaling(results, plt):
    names = list(results)
    ns = [int(n) for n in results[names[0]]["per_n"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
    panels = [("time", 1e3, "estimate() time (ms)", "Runtime", [10, 30, 100, 300, 1000]),
              ("peak_mb", 1, "peak memory allocated (MB)", "Peak memory",
               [0.3, 1, 3, 10, 30, 100])]
    for ax, (key, scale, ylabel, title, yticks) in zip(axes, panels):
        _style(ax)
        for color, v in zip(SERIES, names):
            ys = [results[v]["per_n"][str(n)][key] * scale for n in ns]
            ax.loglog(ns, ys, "-o", color=color, lw=2, ms=8, mec=SURFACE, mew=2,
                      label=v, zorder=3)
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.set_yticks(yticks, [f"{y:g}" for y in yticks])
        ax.minorticks_off()
        ax.set_xlim(70, 9000)
        ax.set_xlabel("sample size n (1000 replications each)", color=INK2, fontsize=9)
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "optimization_scaling.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_stages(results, plt, figures_s):
    names = list(results)
    stages = list(results[names[0]]["stages"]) + ["figures (2 scripts)"]
    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    colors = SERIES + ["#9b9a94"]
    for row, v in enumerate(names):
        vals = [results[v]["stages"][s] for s in stages[:-1]] + [figures_s]
        x = 0.0
        for s, val, c in zip(stages, vals, colors):
            ax.barh(row, val, left=x, height=0.62, color=c, edgecolor=SURFACE,
                    linewidth=2, label=s if row == 0 else None, zorder=3)
            if val > 0.12:
                ax.text(x + val / 2, row, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=INK)
            x += val
        ax.text(x + 0.04, row, f"{x:.2f} s", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlabel("seconds (each stage best of 3; Python start-up and imports not included)",
                  color=INK2, fontsize=9)
    ax.set_title("Where the pipeline's time goes", color=INK, fontsize=11, loc="left")
    ax.legend(ncol=6, frameon=False, fontsize=8, loc="upper left",
              bbox_to_anchor=(0, -0.28), labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "optimization_stages.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_flame(results, plt):
    pair = ("baseline", "vectorized")
    xmax = max(results[v]["profile_total"] for v in pair) * 1.01
    depth = max(max(r[0] for r in results[v]["flame"]) for v in pair) + 1
    fig, axes = plt.subplots(2, 1, figsize=(11, 1.0 + 0.42 * depth * 2), facecolor=SURFACE,
                             sharex=True)
    for ax, v in zip(axes, pair):
        ax.set_facecolor(SURFACE)
        for d, start, w, lab in results[v]["flame"]:
            ax.barh(d, w, left=start, height=0.92, color=BLUE_RAMP[min(d, len(BLUE_RAMP) - 1)],
                    edgecolor=SURFACE, linewidth=1.5)
            fit = int(w / (0.0062 * xmax))             # characters that fit in the bar
            text = lab if len(lab) <= fit else (lab[:fit - 1] + "…" if fit >= 4 else "")
            if text:
                ax.text(start + 0.004 * xmax, d, text, va="center", ha="left", fontsize=7.5,
                        color=INK if d < 3 else SURFACE, clip_on=True)
        ax.set_ylim(depth - 0.5, -0.5)
        ax.set_yticks([])
        ax.set_xlim(0, xmax)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.set_title(f"{v}: estimate() = {results[v]['profile_total']:.2f} s under cProfile",
                     color=INK, fontsize=10, loc="left")
    axes[-1].set_xlabel("seconds (bar width = time inside that call, including its callees; "
                        "space under a bar = its own time, mostly NumPy arithmetic)",
                        color=INK2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "optimization_flame.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def time_figures():
    ts = []
    for _ in range(REPEATS):
        t = time.perf_counter()
        for script in ("plot_plugin.py", "plot_beta2.py"):
            out = subprocess.run([sys.executable, str(ROOT / "results" / "figures" / script)],
                                 capture_output=True, text=True)
            if out.returncode != 0:
                raise RuntimeError(f"{script} failed:\n{out.stderr}")
        ts.append(time.perf_counter() - t)
    return min(ts)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    results = run_all()
    figures_s = time_figures()     # same scripts for every version
    write_tables(results)
    plot_scaling(results, plt)
    plot_stages(results, plt, figures_s)
    plot_flame(results, plt)
    print(f"compare_versions.py: done, outputs in {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", metavar="SRC")
    args = ap.parse_args()
    worker(args.worker) if args.worker else main()
