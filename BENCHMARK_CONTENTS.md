# Benchmark contents and reproduction status

## Commands run

```bash
python scripts/download_benchmark.py --output-dir data/benchmark
```

Result in this execution environment: **failed before download** because outbound
access to the GitHub release asset was blocked by the configured proxy:

```text
urllib.error.URLError: <urlopen error Tunnel connection failed: 403 Forbidden>
```

Because `benchmark.zip` could not be downloaded in this environment, the archive
contents could not be inspected here and the benchmark reproduction command could
not be completed against the real benchmark files.

```bash
python scripts/reproduce_paper_results.py \
  --download-benchmark \
  --benchmark-dir data/benchmark \
  --output-dir outputs/reproduction_benchmark
```

This command is wired to download/extract automatically when the benchmark files
are missing, but it depends on access to the same GitHub release asset.

## Archive path

Expected local cache path after a successful download:

```text
data/benchmark/benchmark.zip
```

Expected extraction directory:

```text
data/benchmark/benchmark/
```

## Exact files found

No benchmark CSV files were found in this environment because the download was
blocked before `benchmark.zip` could be cached or extracted.

## Expected benchmark CSV files validated by the downloader

The downloader validates these expected benchmark files after extraction:

| CSV file | Detector/model name used by reproduction script |
| --- | --- |
| `EffNetB0-preds.csv` | `EffNetB0-preds` |
| `EffNetB0-mirror-preds.csv` | `EffNetB0-mirror-preds` |
| `EffNetB1-preds.csv` | `EffNetB1-preds` |
| `EffNetB1-mirror-preds.csv` | `EffNetB1-mirror-preds` |
| `EffNetB2-preds.csv` | `EffNetB2-preds` |
| `EffNetB2-mirror-preds.csv` | `EffNetB2-mirror-preds` |
| `EffNetB3-preds.csv` | `EffNetB3-preds` |
| `EffNetB3-mirror-preds.csv` | `EffNetB3-mirror-preds` |
| `EffNetB4-preds.csv` | `EffNetB4-preds` |
| `EffNetB4-mirror-preds.csv` | `EffNetB4-mirror-preds` |
| `EffNetB5-preds.csv` | `EffNetB5-preds` |
| `EffNetB5-mirror-preds.csv` | `EffNetB5-mirror-preds` |
| `EffNetB6-preds.csv` | `EffNetB6-preds` |
| `EffNetB6-mirror-preds.csv` | `EffNetB6-mirror-preds` |
| `EffNetB7-preds.csv` | `EffNetB7-preds` |
| `EffNetB7-mirror-preds.csv` | `EffNetB7-mirror-preds` |
| `DetRS-valid.csv` | `DetRS-valid` |
| `DetRS-mirror-valid.csv` | `DetRS-mirror-valid` |
| `DetRS_resnet50-valid.csv` | `DetRS_resnet50-valid` |
| `DetRS_resnet50-mirror-valid.csv` | `DetRS_resnet50-mirror-valid` |
| `yolov5x_tta.csv` | `yolov5x_tta` |

## Rows/images/detections per CSV

Not available from this environment because the real benchmark CSV files were
not downloadable. In an environment with access to the release asset, run:

```bash
python scripts/download_benchmark.py --output-dir data/benchmark
python scripts/reproduce_paper_results.py \
  --download-benchmark \
  --benchmark-dir data/benchmark \
  --output-dir outputs/reproduction_benchmark
```

Then count rows/images/detections from each extracted CSV in
`data/benchmark/benchmark/`.

## Annotations and AP/AR computation

The benchmark release contains detector prediction CSV files, not COCO annotation
JSON. Therefore AP/AR can only be computed if the user also supplies COCO
annotations with `--annotations`, for example:

```bash
python scripts/reproduce_paper_results.py \
  --download-benchmark \
  --benchmark-dir data/benchmark \
  --annotations data/coco/annotations/instances_val2017.json \
  --output-dir outputs/reproduction_benchmark_eval
```

If annotations are not supplied, the reproduction script emits this explicit
message:

```text
Benchmark predictions were loaded and fused, but COCO AP/AR cannot be computed without annotations.
```

## Outputs generated

The smoke reproduction path was run successfully and generated the same fused
output filenames that the benchmark path will produce after successful download:

```text
outputs/reproduction_sample/wbf_predictions.json
outputs/reproduction_sample/awbf_predictions.json
outputs/reproduction_sample/awbf_competition_predictions.json
outputs/reproduction_sample/awbf_negotiation_predictions.json
outputs/reproduction_sample/reproduction_report.json
```

For a successful benchmark run, the corresponding files are written under
`outputs/reproduction_benchmark/`:

```text
outputs/reproduction_benchmark/wbf_predictions.json
outputs/reproduction_benchmark/awbf_predictions.json
outputs/reproduction_benchmark/awbf_competition_predictions.json
outputs/reproduction_benchmark/awbf_negotiation_predictions.json
outputs/reproduction_benchmark/reproduction_report.json
```
