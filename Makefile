PYTHON ?= python3

.PHONY: paper reproduce test

paper:
	$(PYTHON) -B scripts/build_paper.py

reproduce:
	$(PYTHON) -B scripts/reproduce.py --output build/reproduction

test:
	$(PYTHON) -B -m unittest discover -s tests -v
