# One-command entry points. Targets are filled in as each phase lands.

PYTHON ?= python

.PHONY: model test-model lint sim regress clean

model:                     ## Train float MLP, quantise to INT8, run the integer golden model
	$(PYTHON) model/train.py
	$(PYTHON) model/quantize.py
	$(PYTHON) model/golden.py

test-model:                ## Python unit tests for the golden model
	$(PYTHON) -m pytest model -q

lint:                      ## Verilator lint on all RTL
	@echo "TODO: verilator --lint-only (Phase 2)"

sim:                       ## Unit testbenches
	@echo "TODO: per-unit simulations (Phase 2-4)"

regress: test-model lint sim  ## Verification levels 1-5, non-zero exit on any failure

clean:
	rm -rf model/data/*.npz model/data/*.hex obj_dir xsim.dir
