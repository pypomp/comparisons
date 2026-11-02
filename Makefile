PYTHON ?= python3
PACKAGE ?= cuthbert
BUILD ?= build
PORT ?= 8765

.PHONY: help prepare overview figures tables check check-archives browse
.DEFAULT_GOAL := help

help:
	@echo "make prepare PACKAGE=cuthbert  Restore original source and apply the recorded diff"
	@echo "make overview                  Summarize current native-baseline results and times"
	@echo "make figures                   Draw current native-baseline figures (NumPy/Matplotlib)"
	@echo "make tables                    Rebuild seven earlier saved tables (PFJAX extension retained)"
	@echo "make check                     Verify files, source recovery and saved tables"
	@echo "make check-archives            Also read all compressed archive contents"
	@echo "make browse                    Open a read-only server at 127.0.0.1:8765"
	@echo "Set BUILD to a new directory when repeating source or report generation."

prepare:
	$(PYTHON) scripts/prepare.py $(PACKAGE) --output $(BUILD)/$(PACKAGE)

tables:
	$(PYTHON) scripts/reproduce.py --output $(BUILD)/tables

overview:
	$(PYTHON) scripts/summarize_current.py --output $(BUILD)/overview

figures:
	$(PYTHON) scripts/plot_dacca_native_comparison.py --data reports/dacca_native_figures_20261102T0718/figure_data.json --output $(BUILD)/figures

check:
	$(PYTHON) scripts/verify.py --output $(BUILD)/verification.json

check-archives:
	$(PYTHON) scripts/verify.py --archives --output $(BUILD)/verification.json

browse:
	$(PYTHON) scripts/browse.py --port $(PORT)
