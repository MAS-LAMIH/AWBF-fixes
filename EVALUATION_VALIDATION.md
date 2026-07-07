# Evaluation Validation

This repository now generates `EVALUATION_VALIDATION.md` in every reproduction
`--output-dir`. The generated report validates, per method:

- COCO image IDs exist in the annotation file.
- Category IDs exist in the annotation file.
- Scores are finite and in `[0, 1]`.
- Bboxes are finite pixel-space COCO `xywh` values before evaluation.
- COCO AR metrics use correct names: standard stats are `AR_maxDets1`, `AR_maxDets10`, and `AR`; custom `AR50`/`AR75` are computed separately from the recall tensor.
- Empty/non-positive bboxes are removed before COCOeval.
- Normalized-looking bboxes are counted so scaling mistakes are visible.
- Bbox conversion counts from `--bbox-scale auto|normalized|pixel` are reported.

## Current environment status

The requested full benchmark reproduction could not complete in this environment
because `data/benchmark/benchmark/` contains only `.gitkeep`, the expected CSVs
are absent, and the fallback download from GitHub failed with `Tunnel connection
failed: 403 Forbidden`.

Because the full reproduction did not run, this root-level file is a validation
capability/status document, not a claim of paper-metric reproduction. A
successful run will write the data-specific validation report to:

`outputs/reproduction_benchmark/EVALUATION_VALIDATION.md`
