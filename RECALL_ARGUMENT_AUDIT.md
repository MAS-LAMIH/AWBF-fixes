# Recall Argument Audit

The reproduction script now writes `RECALL_ARGUMENT_AUDIT.md` inside the selected
`--output-dir`. The generated report uses computed COCO metrics to assess:

- whether AWBF variants improve or preserve AR relative to WBF,
- whether recall gains are stronger than precision/AP gains,
- which method has the best AR,
- whether medium/large-object AR follows the recall-oriented trend,
- whether competition/negotiation preserve more detections than WBF,
- whether the computed results support the revised recall-oriented paper argument.

## Acceptance rule

The recall-oriented argument is not considered supported by this repository
unless nonzero COCO AR metrics are computed and AP-vs-AR trends are explicitly
compared against WBF.

## Current environment status

The full benchmark reproduction did not complete here because the benchmark CSVs
and COCO annotations were unavailable, and the benchmark download attempt failed
with `Tunnel connection failed: 403 Forbidden`. Therefore this root-level file
does not claim support for the recall argument. A successful run will write the
computed audit to:

`outputs/reproduction_benchmark/RECALL_ARGUMENT_AUDIT.md`
