# Baseline Performance: Unit 2 Simulation Study

**Study settings.** n ∈ {100, 500, 1000, 5000} with S = 1000 randomized experiments per n, giving 4000 logistic regressions fit by Newton–Raphson plus 4000 plug-in estimates. Seed 607.

---

## 1. Total runtime


The full study takes about 3.6 s end to end. Breakdown by `make` stage (from `results/profiling/stage_timings.csv`):

| Stage | Script | Wall time | Share |
|---|---|---|---|
| simulate | `src/simulation.py` (generate, pickle, reload, estimate, pickle) | 1.55 s | 44% |
| analyze | `src/analyze.py` | 0.18 s | 5% |
| figures | `plot_plugin.py` + `plot_beta2.py` | 0.90 + 0.89 s | 51% |
| *for reference* | `import numpy` alone | 0.16 s | |
| *for reference* | `import matplotlib.pyplot` alone | 0.54 s | |

About half the wall time goes to plotting, and most of that is fixed cost: each of the two plot scripts imports matplotlib (0.54 s) and loads the 106 MB `simulated.pk` only to read `delta`. Neither cost depends on the statistics. The scientific computation is entirely inside the **simulate** stage, and that stage is what the rest of this document profiles.

## 2. Profiler results

### 2.1 cProfile of `simulation.py` (function level)

The run took 1.60 s under cProfile (`results/profiling/cprofile_simulation.txt`, with the `.prof` file alongside it for snakeviz).

| Function | Calls | Own time | Cumulative time | % of run |
|---|---|---|---|---|
| `simulation.estimate` | 1 | 0.008 s | **1.378 s** | **86%** |
| ↳ `methods.logit_mle` | 4,000 | 0.655 s | **1.221 s** | **76%** |
| ↳↳ `methods._expit` | 32,352 | 0.336 s | 0.336 s | 21% |
| ↳↳ `numpy.linalg.solve` (3×3) | 24,352 | 0.096 s | 0.219 s | 14% |
| ↳ `methods.plug_in` | 4,000 | 0.025 s | 0.148 s | 9% |
| `simulation.simulate_data` | 1 | 0.015 s | 0.157 s | 10% |
| ↳ `dgps.randomize` | 4,000 | 0.091 s | 0.115 s | 7% |
| `pickle.dump` + `pickle.load` | 3 | 0.062 s | 0.062 s | 4% |

24,352 Newton iterations across 4,000 fits works out to **6.1 iterations per fit**. One call to `np.linalg.solve` on a 3×3 system costs about 9 µs. Almost all of that is Python and NumPy dispatch overhead (`_commonType`, `_makearray`, error-state context managers): the 3×3 solve itself takes nanoseconds.

### 2.2 line_profiler on the hot path

From `results/profiling/line_profile.txt`, `methods.logit_mle` (1.30 s)

| Line | Code | % of `logit_mle` |
|---|---|---|
| 20 | `hess = (M * (p * (1 - p))[:, None]).T @ M` | **25.7%** |
| 18 | `p = _expit(M @ beta)` | **23.9%** |
| 21 | `step = np.linalg.solve(hess, grad)` | **21.6%** |
| 19 | `grad = M.T @ (Y - p)` | 17.2% |
| 23 | `if np.max(np.abs(step)) < tol` | 7.4% |
| 15 | `M = np.column_stack([...])` | 2.9% |

**Main bottlenecks, in order:**

1. **The Newton–Raphson loop in `logit_mle`, at about 76–80% of compute.** The time is spread fairly evenly over its four array operations. No single line dominates, which suggests the cost is mostly per-call overhead repeated about 24k times.
2. **`plug_in` and `randomize`, at about 7–10% each.** Both are O(n) array work plus per-call overhead.
3. **Pickle I/O plus the duplicated loads and imports in the figure scripts.** These cost little compute but a lot of wall time in `make all`.

## 3. Computational complexity

### 3.1 Theoretical analysis (per Claude)

Notation: n is the sample size, p = 3 is the number of logit parameters, S = 1000 is the number of replications per n, and K is the number of Newton iterations.

| Step | Work per replication | Notes |
|---|---|---|
| `make_population` (once per n) | O(n) | negligible |
| `randomize` | O(n) | `choice` without replacement, then elementwise Y |
| `logit_mle`, per iteration | O(np) for `M@β`, expit and gradient; O(np²) for the Hessian; O(p³) for the solve | |
| `logit_mle`, total | **O(K·n·p²)** | p = 3 is fixed and K ≈ 6 , so this is **O(n)** |
| `plug_in` | O(n) | two expit evaluations and two means |
| `true_delta` (once per n) | O(n) | |

So **one replication costs Θ(n)**, and the whole study costs Θ(S · Σ n) = Θ(1000 × 6600). Newton–Raphson converges quadratically once it is near the optimum, and the log-likelihood here is well conditioned (Hessian condition number < 90; see Section 4). That is why K does not grow with n: the measured K is 6.0 for every n ≥ 500 and 6.35 at n = 100.

### 3.2 Empirical analysis

`profiling/complexity.py` times a single replication (randomize, then `logit_mle`, then `plug_in`) on a frozen population for n from 50 to 500,000. It reports the median over 30 to 1000 repetitions. Full data is in `results/profiling/complexity.csv`.

![Runtime of one replication vs n](../results/profiling/complexity.png)

| n | randomize | logit_mle | plug_in | **total** | Newton iters |
|---|---|---|---|---|---|
| 100 | 0.009 ms | 0.088 ms | 0.011 ms | **0.108 ms** | 6.1 |
| 500 | 0.013 ms | 0.140 ms | 0.016 ms | **0.169 ms** | 6.0 |
| 1,000 | 0.019 ms | 0.190 ms | 0.024 ms | **0.233 ms** | 6.0 |
| 5,000 | 0.063 ms | 0.558 ms | 0.071 ms | **0.693 ms** | 6.0 |
| 50,000 | 0.277 ms | 4.716 ms | 0.604 ms | **5.597 ms** | 6.0 |
| 500,000 | 3.007 ms | 63.96 ms | 6.063 ms | **73.42 ms** | 6.0 |

Slopes fitted on a log–log scale (`results/profiling/complexity_fit.txt`):

| Component | n ≤ 1000 | n ≥ 5000 | all n |
|---|---|---|---|
| `logit_mle` | 0.27 | **1.01** | 0.72 |
| `plug_in` | 0.25 | 0.97 | 0.72 |
| `randomize` | 0.28 | 0.83 | 0.65 |
| **total** | **0.26** | **0.99** | 0.72 |

The empirical results match the theory.

**Where the study's compute goes**, predicted as 1000 × t(n) from the table above:

| n | Predicted time for 1000 reps | Share |
|---|---|---|
| 100 | 0.11 s | 9% |
| 500 | 0.17 s | 14% |
| 1,000 | 0.23 s | 19% |
| 5,000 | 0.69 s | 58% |
| **Total** | **1.20 s** | |

This agrees with the 1.38 s measured by cProfile for `estimate`, which includes profiler overhead.

### 3.3 Per-stage scaling measured by the profiler

![Per-stage time complexity measured by cProfile](../results/profiling/stage_scaling.png)

cProfile cumulative time per replication, in ms:

| Stage | n = 100 | n = 500 | n = 1,000 | n = 5,000 | n = 100,000 | Slope, n ≤ 1000 | Slope, n ≥ 5000 |
|---|---|---|---|---|---|---|---|
| simulate: `randomize` | 0.014 | 0.013 | 0.019 | 0.066 | 0.538 | 0.21 | 0.66 |
| estimate: `logit_mle` | 0.132 | 0.184 | 0.240 | 0.617 | 9.681 | 0.21 | **0.92** |
| estimate: `plug_in` | 0.015 | 0.021 | 0.030 | 0.076 | 1.354 | 0.20 | 0.95 |
| save/load: pickle | 0.001 | 0.003 | 0.004 | 0.020 | 0.753 | 0.69 | **1.25** |
| analyze: `summarize` | 0.0004 | 0.0005 | 0.0005 | 0.0005 | 0.0005 | −0.04 | 0.04 |
| **Total pipeline** | **0.166** | **0.225** | **0.298** | **0.792** | **12.43** | 0.21 | 0.92 |

The per-stage results agree with the analysis in 3.1:

- **`logit_mle` dominates at every n,** taking 75–80% of each replication.
- **`plug_in` and `randomize` are O(n)** with small constants.
- **`summarize` does not depend on n.** 


## 4. Numerical warnings and convergence (per Claude)

`profiling/stability.py` re-runs the study exactly (seed 607) and logs the following for each replication:

- every NumPy floating-point event (overflow, divide-by-zero, invalid and underflow, counted with `np.errstate(all="call")`)
- every Python warning
- every exception

It also replays the Newton loop to record iterations, convergence, gradient size, Hessian condition number and the range of fitted probabilities. The replay reproduces `logit_mle` exactly, bit for bit, in all 4000 replications. Output is in `results/profiling/stability_log.txt` and `stability_study_n*.csv`.

### 4.1 The Unit 2 study: no instability observed

| n | Exceptions | Warnings / FP events | Not converged | Newton iters (mean / max) | max \|∇ℓ\| at solution | Hessian cond. (max) | Fitted p range | max \|β̂\| |
|---|---|---|---|---|---|---|---|---|
| 100 | 0 | 0 | 0 / 1000 | 6.35 / 7 | 9.0e-15 | 88 | [0.009, 0.994] | 7.5 |
| 500 | 0 | 0 | 0 / 1000 | 6.00 / 6 | 3.0e-14 | 42 | [0.105, 0.959] | 3.7 |
| 1,000 | 0 | 0 | 0 / 1000 | 6.00 / 6 | 5.8e-14 | 42 | [0.104, 0.963] | 4.1 |
| 5,000 | 0 | 0 | 0 / 1000 | 6.00 / 6 | 3.1e-13 | 37 | [0.132, 0.953] | 3.7 |


n = 100 is the only setting with any sign of strain, with the widest range of fitted probabilities and the largest coefficients (|β̂| up to 7.5).

### 4.2 Stress test: where the same code breaks

To find the conditions under which the code becomes unstable, the same functions were run at smaller n. Each n uses 10 populations × 100 randomizations, frac_treated = 0.75 and seed 2026.

| n | Arm with constant Y (separation) | `LinAlgError: Singular matrix` | Hit `max_iter=100` silently | Non-finite δ̃ | Overflow events (`exp`) | Divide-by-zero (`log`) | Total failures |
|---|---|---|---|---|---|---|---|
| 8 | 59.9% | 68.5% | 1.1% | 1.0% | 470 | 10 | **70.6%** |
| 12 | 27.7% | 39.4% | 1.5% | 0.6% | 5153 | 6 | **41.5%** |
| 16 | 15.6% | 15.0% | 0 | 0.8% | 206 | 8 | **15.8%** |
| 20 | 9.3% | 9.0% | 0.2% | 0.2% | 145 | 2 | **9.4%** |
| 30 | 1.7% | 1.5% | 0 | 0.2% | 1 | 2 | **1.7%** |
| 50 | 0.1% | 0.1% | 0 | 0 | 0 | 0 | **0.1%** |
| ≥ 100 (study) | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

When the control arm (only 25% of subjects) happens to have all Y = 0 or all Y = 1, the logit MLE does not exist (β̂ → ±∞). Newton then pushes fitted probabilities to exactly 0 or 1, the weights p(1 − p) underflow to zero, and the Hessian becomes singular.

---

## Etcetera (reproducing, file bloat/descriptions)

```bash
pip install -r requirements.txt line_profiler
make all        # the study itself (time it with `time make all`)
make profile    # profile_runtime.py, complexity.py, profile_stages.py, stability.py → results/profiling/
```

| File | Contents |
|---|---|
| `profiling/profile_runtime.py` | cProfile + line_profiler + per-stage wall times |
| `profiling/complexity.py` | per-replication timing vs n, log–log fits, plot |
| `profiling/stability.py` | warning/exception/convergence audit and small-n stress test |
| `results/profiling/cprofile_simulation.txt` / `.prof` | full cProfile output |
| `results/profiling/line_profile.txt` | line-by-line timings |
| `results/profiling/complexity.csv`, `complexity_fit.txt`, `complexity.png` | complexity data, fits, figure |
| `profiling/profile_stages.py` | cProfile of the whole pipeline at each n, timing each stage separately |
| `results/profiling/stage_scaling.csv`, `stage_scaling_fit.txt`, `stage_scaling.png` | per-stage scaling data, fits, figure |
| `results/profiling/stability_log.txt`, `stability_*.csv` | per-replication diagnostics |
