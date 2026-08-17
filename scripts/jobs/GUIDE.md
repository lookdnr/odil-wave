# Guide to the `.pbs` scripts

The scripts here are written to be run on an AMD Rome node of Imperial's CX3 HPC cluster.

To run the experiments on CX3, first clone the repository after ssh'ing on to CX3.

`scripts/collect-results` submits every job below in one go, with the correct arguments already filled in. Use it unless you specifically want to rerun just one experiment. To collect all results, simply run from the root of the repo

```bash
./scripts/collect-results
```

Note you may have to make this executable via

```bash
chmod u+x scripts/collect-results
```

## exp0: GMRES/L-BFGS-B baseline comparison

```bash
qsub -N baseline scripts/jobs/exp0_baseline.pbs
```

No args, output path is hardcoded (`results/baseline.pkl`).

## exp1: accuracy sweep

```bash
qsub -N accuracy-sweep scripts/jobs/exp1_accuracy_sweep.pbs
```
No args. Outputs `results/accuracy/sweep_50khz_nxi.jsonl`, for `i=80, 100, 120, 140, 160, 180, 200` describing problem configurations, along with a `results/accuracy/sweep_50khz.pkl`, a pickle file encapsulating the results of each sweep.

## exp2a: frequency sweep

To collect the baseline (no hardware parallelism) performance data, run

```bash
qsub -N f0-sweep-cached-c1 -v SOLVER=odil,MODE=cached,NCORES=1 scripts/jobs/exp2a_f0_sweep.pbs
qsub -N f0-sweep-uncached-c1 -v SOLVER=odil,MODE=uncached,NCORES=1 scripts/jobs/exp2a_f0_sweep.pbs
qsub -N f0-sweep-devito-c1 -v SOLVER=devito,MODE=na,NCORES=1 scripts/jobs/exp2a_f0_sweep.pbs
```

To collect performance data at the full thread count, run

```bash
qsub -N f0-sweep-cached-c128 -v SOLVER=odil,MODE=cached,NCORES=128 scripts/jobs/exp2a_f0_sweep.pbs
qsub -N f0-sweep-uncached-c128 -v SOLVER=odil,MODE=uncached,NCORES=128 scripts/jobs/exp2a_f0_sweep.pbs
qsub -N f0-sweep-devito-c128 -v SOLVER=devito,MODE=na,NCORES=128 scripts/jobs/exp2a_f0_sweep.pbs
```

Note: the `c1` and `c128` runs for the same `SOLVER`/`MODE` combination write to the same output file (`results/performance/f0_sweep_${SOLVER}_${MODE}.jsonl`, appended, distinguished only by the `ncores` field within each row). If you submit both at once they can, in principle, race on the same file if their writes overlap. Hasn't caused a problem in practice, but worth knowing if a file ever looks corrupted.

## exp2b: strong scaling

```bash
qsub -N strong-scaling scripts/jobs/exp2b_strong_scaling.pbs
```

No args, sweeps `ncores` and `solver`/`mode` internally. Output is `results/performance/f0_sweep_{solver}_{mode}.jsonl` for `solver` in `devito, odil` and `mode` in `na, cached, uncached`.

## exp3a: dispersion/dissipation ppw sweep

```bash
qsub -N dispersion scripts/jobs/exp3a_dispersion.pbs
```

No args, sweeps `ppw` internally, output is `results/wave/dispersion.pkl`.

## exp3b: CFL sweep

```bash
qsub -N cfl-sweep-homog -v WHICH=homog scripts/jobs/exp3b_cfl_sweep.pbs
qsub -N cfl-sweep-sl -v WHICH=sl scripts/jobs/exp3b_cfl_sweep.pbs
```

`WHICH` selects the model (`homog` = homogeneous, quantitative; `sl` = Shepp-Logan+skull, qualitative). No analytic reference exists for a heterogeneous medium, so accuracy metrics are dropped and field snapshots are saved instead. Sweeps `cfl_safety` internally, and each submission writes its own file (`results/wave/cfl_sweep_${WHICH}.jsonl`).
