Uplift Model Talk
=================

A small repository with code and notebooks for demonstrating uplift modeling experiments on Criteo data.

## Repository structure
- `nbs/` — Jupyter notebooks exploring dataset processing and uplift metrics (`dataset.ipynb`, `uplift_conversion.ipynb`, `uplift_visit.ipynb`).
- `src/` — small Python package with helper modules (`bins.py`, `curve.py`).
- `slides.pdf` — presentation slides

## Quick setup

This project uses Poetry for dependency management. If you don't have Poetry installed, see https://python-poetry.org/docs/

Install dependencies (from repo root):

```bash
poetry install
```

Run a notebook

Open the notebooks with Jupyter Lab/Notebook from the repo root after activating the poetry shell:

```bash
poetry shell
jupyter lab nbs/
```

Usage examples

- Run the notebooks interactively to reproduce dataset exploration and uplift metric plots.
- Import helpers from `src` in your own scripts or notebooks:

```python
from src.bins import plot_uplift_bins  # replace with actual function names
from src.curve import plot_qini_curve
```

Notes

- The data files are compressed CSVs, which will be downloaded when the notebooks are run; the notebooks show how they are loaded.
- This repository is intended for demos and experiments, not production use.