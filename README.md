# ODIL-wave: a tool for solving the wave equation by Optimising a Discrete Loss (ODIL)

![Tests](https://github.com/lookdnr/odil-wave/actions/workflows/test-package.yml/badge.svg?branch=main)
![Build Report](https://github.com/lookdnr/odil-wave/actions/workflows/build-report.yml/badge.svg?branch=main)

## Table of contents

- [Overview](#overview)
- [Getting started](#getting-started)
- [Repository structure](#repository-structure)
- [Examples](#examples)
- [Documentation](#documentation)
- [Tests](#tests)
- [Reproducing the report](#reproducing-the-report)

## Overview

This repository presents `odil_wave`, a modular, extensible library for solving the two dimensional acoustic wave equation using the [Optimising a Discrete Loss](https://pmc.ncbi.nlm.nih.gov/articles/PMC10799659/) (ODIL) framework. ODIL can be used to solve the discretised wave equation by expressing it as a functional $\mathcal{L}$,

$$\mathcal{L} = \frac{1}{2}\|A\mathbf{u} - \mathbf{s}\|^2.$$

which we seek to minimise, where $\mathbf{s}$ is a forcing term and $\mathbf{u}$ is the pressure field we seek to find. The pressure field is the optimisation variable, and it is optimised directly on the discretising grid. The approach used in this work is to formulate the PDE as a linear system discretised by sparse matrix operators derived from finite difference stencils, whose corresponding ODIL problem is primarily inteded to be solved using a method derived from the Gauss-Newton algorithm. The described approach is enabled by the ParaDiag-II preconditioner, which introduces embarassing parallelism that is exploited using Python's `multiprocessing` module.

The contents of this repository were developed by Luke Dinsdale in collaboration with Sonalis as part of the capstone module of the MSc Applied Computational Science and Engineering at Imperial College London.

## Getting started

To begin, clone the repository. You may then install the package from the root of the project by running

```
pip install .
```

Alternatively, to install in editable mode with the optional dependencies (for use of dev tools or to run tests locally), run one of the following

```
pip install -e .[dev]
pip install -e .[test]
```

Note that this library has been tested only for Python version >=3.12.

## Repository structure

```ASCII
.
├── README.md
├── deliverables/               # project plan and repository
├── experiments/                # scripts for generating experimental results in report
├── logbook/                    # tracking log over the IRP period
├── notebooks/                  # examples and scrapbooks
├── pyproject.toml
├── odil_wave/                  # source code
├── scripts/                    # scripts for figure generation and PBS jobs
├── tests/                      # module tests
└── title/                      # IRP project title tracker
```
## Examples

This library has been designed for programmatic usage, no CLI or GUI has been developed as of this moment. A full introductory walkthrough can be found in [this Jupyter notebook](/notebooks/example-usage.ipynb).

As a quick reference, however, you may find the following useful as an overview of the intended workflow:

```python
from odil_wave import (
    Grid, 
    OverDensityModel, 
    Wavefield,
    Sources,
    Receivers,
    AcqusitionGeometry
)

# define an 100x100 grid
grid = Grid(nx=100, ny=100, xmin=0.0, xmax=80.0, ymin=0.0, ymax=80.0, c_min=1200.0, c_max=1500.0, t_max=0.07, cfl_safety=0.9)

print(grid.summary)

# build a circular inclusion model
model = OverDensityModel(grid, background_c=1200.0, contrast=300.0, centre=(40.0, 25.0), radius=15)

# define an acquisition geometry in spatial coords
src = Sources(grid, f0=120.0, source_locs=((40.0, 50.0),))
recs = Receivers(grid, receiver_locs=((20.0, 25.0), (40.0, 15.0), (60.0, 40.0)))
geom = AcqusitionGeometry(src, rcvs)

# plot the geometry on the model
geom.show(model)

"""
...

from here, a WaveEquation can be defined and piped into a
Problem which configures an Optimiser, and so forth.

Consult the example notebook for a more thorough walkthrough.
"""
```

## Documentation

API documentation is built with MkDocs and generated from NumPy-style docstrings. It can be served locally in a live preview format, or static docs can be built as follow

```
pip install -e .[dev]
mkdocs serve          # live preview at http://127.0.0.1:8000
mkdocs build          # static site in site/
```

## Tests

Tests were built primarily for verification of the correctness of mathematical components, but also comprise formatting checks against PEP8 style guidelines and basic smoke tests. They can be run locally using

```bash
pip install -e .[test]
pytest
```

Note that `[test]` pulls in `devito`, which is used as the reference finite difference solver in the comparison tests.

## Reproducing the report

Experiment drivers are in `experiments/`, with PBS submission scripts in `scripts/jobs/` and figure generation in `scripts/figures/`. Each experiment writes JSON/JSONL results into `results/`, which the figure scripts then consume.
