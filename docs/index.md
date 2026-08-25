# ODIL-wave: a tool for solving the wave equation by Optimising a Discrete Loss (ODIL)

## Overview

A modular, extensible framework for solving the two dimensional acoustic wave equation using the [Optimising a Discrete Loss](https://pmc.ncbi.nlm.nih.gov/articles/PMC10799659/) (ODIL) framework.

The contents of this repository were developed Luke Dinsdale as part of the capstone module of the MSc Applied Computational Science and Engineering at Imperial College London.

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

## Repository structure

```
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
