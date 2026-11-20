"""Runtime profile of the Unit 2 simulation study (unchanged source code).

1. cProfile of `src/simulation.py` run as __main__ (simulate + save + load +
   estimate + save), i.e. exactly what `make simulate` does.
2. line_profiler on the hot functions (logit_mle, randomize, plug_in, and the
   two driver loops).
3. Wall-clock split of each `make` stage, including how much of the figures
   stage is just importing matplotlib.

Writes text reports to results/profiling/.
"""
import contextlib
import cProfile
import io
import pstats
import runpy
import sys
import time

from line_profiler import LineProfiler

from _common import OUT, ROOT, SRC, done, run_checked

import dgps
import methods
import simulation


def run_cprofile():
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    with contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path(str(SRC / "simulation.py"), run_name="__main__")
    prof.disable()
    wall = time.perf_counter() - t0

    buf = io.StringIO()
    buf.write(f"Wall time of simulation.py under cProfile: {wall:.2f} s\n\n")
    for key in ("tottime", "cumulative"):
        buf.write(f"===== top 20 by {key} =====\n")
        pstats.Stats(prof, stream=buf).strip_dirs().sort_stats(key).print_stats(20)
    (OUT / "cprofile_simulation.txt").write_text(buf.getvalue())
    prof.dump_stats(str(OUT / "cprofile_simulation.prof"))


def run_line_profiler():
    lp = LineProfiler()
    for f in (methods.logit_mle, methods.plug_in, dgps.randomize,
              simulation.simulate_data, simulation.estimate):
        lp.add_function(f)
    data = lp.runcall(simulation.simulate_data)
    lp.runcall(simulation.estimate, data)
    buf = io.StringIO()
    lp.print_stats(stream=buf, output_unit=1e-3)
    (OUT / "line_profile.txt").write_text(buf.getvalue())


def stage_timings():
    """Wall time of each make stage in a fresh interpreter (median of 3)."""
    py = sys.executable
    cmds = {
        "simulate (simulation.py)": ([py, "simulation.py"], SRC),
        "analyze (analyze.py)": ([py, "analyze.py"], SRC),
        "figures (plot_plugin.py)": ([py, "results/figures/plot_plugin.py"], ROOT),
        "figures (plot_beta2.py)": ([py, "results/figures/plot_beta2.py"], ROOT),
        "  of which: import numpy": ([py, "-c", "import numpy"], ROOT),
        "  of which: import matplotlib.pyplot": (
            [py, "-c", "import matplotlib.pyplot"], ROOT),
    }
    lines = ["stage,median_wall_s,runs"]
    for name, (cmd, cwd) in cmds.items():
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            run_checked(cmd, cwd)
            ts.append(time.perf_counter() - t0)
        ts.sort()
        lines.append(f"{name},{ts[1]:.3f},{'|'.join(f'{t:.3f}' for t in ts)}")
    (OUT / "stage_timings.csv").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    run_cprofile()
    run_line_profiler()
    stage_timings()
    done(__file__)
