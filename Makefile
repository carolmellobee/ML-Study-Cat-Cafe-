PY ?= python
export PYTHONPATH := src

.PHONY: all test data clean

all:            ## simulate data, run the audit, model, SHAP, revenue, experiment, write reports
	$(PY) -m catcafe.pipeline

test:
	$(PY) -m pytest -q

data:           ## only regenerate the raw tables
	$(PY) -m catcafe.simulate

clean:
	rm -rf data/raw data/warehouse data/processed
