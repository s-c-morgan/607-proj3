PYTHON ?= python
WORKERS ?= 4

.PHONY: all simulate analyze figures clean test \
        profile complexity benchmark parallel stability-check all-with-profilers

all: simulate analyze figures

simulate:
	cd src && $(PYTHON) simulation.py

analyze:
	cd src && $(PYTHON) analyze.py

figures:
	$(PYTHON) results/figures/plot_plugin.py
	$(PYTHON) results/figures/plot_beta2.py

test:
	$(PYTHON) -m pytest tests

clean:
	rm -f data/simulated.pk results/raw/estimates.pk results/summary.csv
	rm -f results/figures/*.png
	rm -f results/profiling/*.csv results/profiling/*.txt results/profiling/*.prof
	rm -rf src/__pycache__ tests/__pycache__ profiling/__pycache__ .pytest_cache

# Run profiling on representative simulation (cProfile + line_profiler)
profile:
	cd profiling && $(PYTHON) profile_runtime.py

# Run computational complexity analysis (timing vs n, overall and per stage)
complexity:
	cd profiling && $(PYTHON) complexity.py
	cd profiling && $(PYTHON) profile_stages.py

# Run timing comparison: baseline vs optimized
benchmark:
	cd profiling && $(PYTHON) compare_versions.py

# Run optimized version with parallelization: threaded estimation and the
# two figure scripts in parallel
parallel:
	cd src && $(PYTHON) simulation.py --workers $(WORKERS)
	cd src && $(PYTHON) analyze.py
	$(PYTHON) results/figures/plot_plugin.py & pid=$$!; \
	$(PYTHON) results/figures/plot_beta2.py; status=$$?; \
	wait $$pid && exit $$status

# Check for warnings/convergence issues across conditions
stability-check:
	cd profiling && $(PYTHON) stability.py

# The full pipeline followed by every profiling and stability analysis
# (in order: benchmark times the figure scripts, so it needs `all` first)
all-with-profilers: all profile complexity benchmark stability-check
