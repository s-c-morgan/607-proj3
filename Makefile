PYTHON ?= python

.PHONY: all simulate analyze figures clean test profile

all: simulate analyze figures

simulate:
	cd src && $(PYTHON) simulation.py

analyze:
	cd src && $(PYTHON) analyze.py

figures:
	$(PYTHON) results/figures/plot_plugin.py
	$(PYTHON) results/figures/plot_beta2.py

profile:
	cd profiling && $(PYTHON) profile_runtime.py
	cd profiling && $(PYTHON) complexity.py
	cd profiling && $(PYTHON) profile_stages.py
	cd profiling && $(PYTHON) stability.py

test:
	$(PYTHON) -m pytest tests

clean:
	rm -f data/simulated.pk results/raw/estimates.pk results/summary.csv
	rm -f results/figures/*.png
	rm -rf src/__pycache__ tests/__pycache__ .pytest_cache
