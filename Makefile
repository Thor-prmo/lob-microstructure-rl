.PHONY: install test figures paper clean all

PYTHON := python3

all: test figures

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/

figures:
	$(PYTHON) scripts/run_experiments.py

paper: figures
	cd paper && pdflatex -interaction=nonstopmode microstructure_execution_note.tex && pdflatex -interaction=nonstopmode microstructure_execution_note.tex

clean:
	rm -rf __pycache__ .pytest_cache src/**/__pycache__ tests/__pycache__
	rm -f paper/*.aux paper/*.log paper/*.out

