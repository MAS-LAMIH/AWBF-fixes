# Paper Results Comparison

The reproduction script now writes a data-specific `PAPER_RESULTS_COMPARISON.md`
inside the selected `--output-dir`. The generated table includes, per method:

- paper AP / reproduced AP / AP delta
- paper AP50 / reproduced AP50 / AP50 delta
- paper AP75 / reproduced AP75 / AP75 delta
- paper AR / reproduced AR / AR delta
- paper AR50 / reproduced custom AR50 / AR50 delta
- paper AR75 / reproduced custom AR75 / AR75 delta
- paper AR50 / reproduced AR50 / AR50 delta
- paper AR75 / reproduced AR75 / AR75 delta
- AP small/medium/large comparison
- AR small/medium/large comparison

`n/a` is used when the paper target table available in this repository does not
provide that metric for a method.

## Current environment status

No full benchmark metrics were computed in this environment. The requested
command failed before fusion/evaluation because benchmark CSVs and COCO
annotations were not present locally, and benchmark download was blocked by the
network/proxy (`Tunnel connection failed: 403 Forbidden`).

A successful full run will write the data-specific comparison to:

`outputs/reproduction_benchmark/PAPER_RESULTS_COMPARISON.md`
