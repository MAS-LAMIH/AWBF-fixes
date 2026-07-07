# Detection Flow Report

The reproduction script now writes `DETECTION_FLOW_REPORT.md` inside the selected
`--output-dir`. The generated report includes, for each selected method:

- input detections,
- output detections,
- detections per image,
- average input confidence,
- average output confidence,
- number of clusters,
- average cluster size,
- largest cluster size,
- detections removed,
- detections fused,
- detections retained,
- average coordinate displacement,
- average negotiation rounds used,
- invalid boxes removed,
- invalid categories.

These diagnostics are intended to explain whether a method preserves/refines
more detections or aggressively suppresses them.

## Current environment status

No full benchmark detection-flow report could be computed in this environment
because the required benchmark CSVs were missing and the download attempt was
blocked. A successful full run will write the data-specific flow report to:

`outputs/reproduction_benchmark/DETECTION_FLOW_REPORT.md`
