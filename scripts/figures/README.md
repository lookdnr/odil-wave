# Guide to `/figures`

In this directory are scripts for generating the figures presented in my [final report](../../deliverables/ld2022-final-report.pdf). To run them, you must have collected the relevant results beforehand by running the result collection scripts.

With all results in hand, you may run

```bash
chmod u+x collect-plots && ./collect-plots
```

To produce all plots. This may take some time. If instead you wish to reproduce a single figure, run for instance

```bash
python3 fig1_op_structure.py
```

None of the scripts require any arguments.