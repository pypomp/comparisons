PYTHON ?= python3
PACKAGE ?= cuthbert
BUILD ?= build
PORT ?= 8765

.PHONY: help prepare tables check check-archives browse
.DEFAULT_GOAL := help

help:
	@echo "make prepare PACKAGE=cuthbert  Restore original source and apply the recorded diff"
	@echo "make tables                    Rebuild summary tables from saved scores"
	@echo "make check                     Verify files, source recovery and saved tables"
	@echo "make check-archives            Also read all compressed archive contents"
	@echo "make browse                    Open a read-only server at 127.0.0.1:8765"
	@echo "Set BUILD to a new directory when repeating prepare or tables."

prepare:
	$(PYTHON) scripts/prepare.py $(PACKAGE) --output $(BUILD)/$(PACKAGE)

tables:
	$(PYTHON) scripts/reproduce.py --output $(BUILD)/tables

check:
	$(PYTHON) scripts/verify.py --output $(BUILD)/verification.json

check-archives:
	$(PYTHON) scripts/verify.py --archives --output $(BUILD)/verification.json

browse:
	$(PYTHON) scripts/browse.py --port $(PORT)
