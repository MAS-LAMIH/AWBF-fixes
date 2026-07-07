# Reproduction Limitations

## Full reproduction attempt

The requested command was run:

```bash
python scripts/reproduce_paper_results.py \
  --benchmark-dir data/benchmark \
  --annotations data/coco/annotations/instances_val2017.json \
  --bbox-scale auto \
  --output-dir outputs/reproduction_benchmark \
  --profile
```

It did not complete in this environment.

## Blocking issue

`data/benchmark/benchmark/` contains only `.gitkeep`; the expected benchmark CSV
files are not present. The script attempted to download
`https://github.com/ZFTurbo/Weighted-Boxes-Fusion/releases/download/v1.0.5/benchmark.zip`,
but the request failed with:

```text
Tunnel connection failed: 403 Forbidden
```

The COCO annotation path `data/coco/annotations/instances_val2017.json` is also
not present in this checkout.


## Parameter sweep status

The parameter sweep script is available and was smoke-tested with `--sample`, but
the full benchmark sweep was not run in this environment for the same missing
benchmark/annotation and blocked-download reasons. Run it locally with:

```bash
python scripts/sweep_awbf_parameters.py \
  --benchmark-dir data/benchmark \
  --annotations data/coco/annotations/instances_val2017.json \
  --bbox-scale auto \
  --output-dir outputs/parameter_sweep
```

## Missing detector provenance

The benchmark archive is expected to provide detector CSVs, but the original
paper's exact detector provenance, calibration details, and any unreleased
post-processing settings are not available in the repository. This remains a
potential source of mismatch even when the benchmark CSVs are supplied.

## Parameter/implementation differences to track

- IoU threshold, score threshold, competition threshold, negotiation threshold,
  negotiation rounds, and negotiation update weight.
- Reconstructed AWBF competition/negotiation logic may differ from unreleased
  code used for the paper.
- Benchmark labels must match COCO category IDs; otherwise evaluation validation
  will report invalid categories.
- Normalized boxes must be converted to pixel-space COCO `xywh` before COCOeval.
- Standard COCOeval AR maxDets stats must not be mislabeled as AR50/AR75; this code now reports maxDets fields separately and computes custom AR50/AR75.

## Current conclusion

Exact or approximate paper-metric reproduction is not claimed from this
environment. The qualitative recall-oriented argument should only be assessed
after a successful run produces nonzero COCO AR metrics and the generated
`RECALL_ARGUMENT_AUDIT.md` compares AR trends against AP trends.
