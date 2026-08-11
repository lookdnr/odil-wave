# Guide to the `.pbs` scripts

The scripts here are written to be run on an AMD Rome node of Imperial's CX3 HPC cluster.

To run the experiments on CX3, first clone the repository after ssh'ing on to CX3.

Next, to collect the results for the accuracy sweep and strong scaling experiments, run

```
qsub -N accuracy_sweep scripts/jobs/accuracy_sweep.pbs
qsub -N strong_scaling scripts/jobs/strong_scaling.pbs
```

To collect the baseline (no hardware parallelism) performance data, run

```
qsub -N f0_swp_cached -v SOLVER=odil,MODE=cached,NCORES=1 scripts/jobs/f0_sweep.pbs
qsub -N f0_swp_uncached -v SOLVER=odil,MODE=uncached,NCORES=1 scripts/jobs/f0_sweep.pbs
qsub -N f0_swp_devito -v SOLVER=devito,MODE=na,NCORES=1 scripts/jobs/f0_sweep.pbs
```

To collect performance data at the full thread count, run

```
qsub -N f0_swp_cached -v SOLVER=odil,MODE=cached,NCORES=128 scripts/jobs/f0_sweep.pbs
qsub -N f0_swp_uncached -v SOLVER=odil,MODE=uncached,NCORES=128 scripts/jobs/f0_sweep.pbs
qsub -N f0_swp_devito -v SOLVER=devito,MODE=na,NCORES=128 scripts/jobs/f0_sweep.pbs
```