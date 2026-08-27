# ODIL-wave: a tool for solving the wave equation by Optimising a Discrete Loss (ODIL)

## Overview

This repository presents `odil_wave`, a modular, extensible library for solving the two dimensional acoustic wave equation using the [Optimising a Discrete Loss](https://pmc.ncbi.nlm.nih.gov/articles/PMC10799659/) (ODIL) framework.

The contents of this repository were developed by Luke Dinsdale in collaboration with Sonalis as part of the capstone module of the MSc Applied Computational Science and Engineering at Imperial College London.

Please consult the GitHub repository for information on repo structure, examples, and instructions for reproducing the report.

## Installation

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

## API reference

Start with [`Grid`](reference/grid.md) to set up a discretisation, then [`WaveEquation`](reference/operator.md) and [`Optimisation`](reference/optimisation.md) for the solve.

Consult the repository's `example-solve.ipynb` for a walkthrough.