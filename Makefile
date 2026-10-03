.PHONY: setup test lab clean

setup:  ## create .venv and install the package with dev tools
	python3 -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
	.venv/bin/pip install -e '.[dev]'

test:   ## run the unit tests
	.venv/bin/pytest -q

lab:    ## open JupyterLab in notebooks/
	.venv/bin/jupyter lab notebooks/

clean:
	rm -rf .pytest_cache build dist src/*.egg-info
