# A guide to `/experiments`

This directory (and its subdirectories) consists of scripts that are used to generate the results explored in my thesis submission. The scripts are _not_ intended to be run directly. The results presented in my report were collected on a branch of Imperial's High Performance Computing cluster, CX3. For more information on result collection, consult the relevant [GUIDE](../scripts/jobs/GUIDE.md). 

## Structure

```ASCII
├── README.md
│ 
├── accuracy/           # accuracy measuremetns
├── common/             # common components
├── performance/        # performance measurements
├── wave_specific/      # e.g. dispersion, cfl
│ 
├── exp0_baseline.py                # baseline results
├── exp1_accuracy_sweep.py          # accuracy results
├── exp2_performance_scaling.py     # perf results
├── exp3_dispersion.py              # dispersion results
└── exp4_cfl_sweep.py               # stability results
```