.PHONY: setup test lab fmt clean

setup:  ## create .venv and install everything from uv.lock
	uv sync

test:   ## run the unit tests
	uv run pytest -q

lab:    ## open JupyterLab in notebooks/
	uv run jupyter lab notebooks/

fmt:    ## format and lint-fix with ruff
	uv run ruff format . && uv run ruff check --fix .

clean:
	rm -rf .pytest_cache build dist src/*.egg-info
