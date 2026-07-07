# Benchmark contents

## Download source

```text
https://github.com/ZFTurbo/Weighted-Boxes-Fusion/releases/download/v1.0.5/benchmark.zip
```

## Commands attempted

```bash
python scripts/download_benchmark.py --output-dir data/benchmark
```

and:

```bash
python scripts/reproduce_paper_results.py \
  --benchmark-dir data/benchmark \
  --output-dir outputs/reproduction_benchmark
```

## Download/extraction result in this environment

The archive could not be downloaded in this execution environment because the
configured proxy rejects the GitHub release-asset connection:

```text
urllib.error.URLError: <urlopen error Tunnel connection failed: 403 Forbidden>
```

A direct no-proxy attempt also failed DNS resolution. Therefore the real
`data/benchmark/benchmark.zip` archive and `data/benchmark/benchmark/` extracted
files are not present in this checkout.

## Full directory tree found

No benchmark directory tree was produced from the real archive because download
failed before the archive could be cached or extracted.

Expected paths after a successful run are:

```text
data/
└── benchmark/
    ├── benchmark.zip
    └── benchmark/
        ├── EffNetB0-preds.csv
        ├── EffNetB0-mirror-preds.csv
        ├── EffNetB1-preds.csv
        ├── EffNetB1-mirror-preds.csv
        ├── EffNetB2-preds.csv
        ├── EffNetB2-mirror-preds.csv
        ├── EffNetB3-preds.csv
        ├── EffNetB3-mirror-preds.csv
        ├── EffNetB4-preds.csv
        ├── EffNetB4-mirror-preds.csv
        ├── EffNetB5-preds.csv
        ├── EffNetB5-mirror-preds.csv
        ├── EffNetB6-preds.csv
        ├── EffNetB6-mirror-preds.csv
        ├── EffNetB7-preds.csv
        ├── EffNetB7-mirror-preds.csv
        ├── DetRS-valid.csv
        ├── DetRS-mirror-valid.csv
        ├── DetRS_resnet50-valid.csv
        ├── DetRS_resnet50-mirror-valid.csv
        └── yolov5x_tta.csv
```

## CSV files found

No real CSV files were found because the archive was unavailable.

## Expected CSV verification list and detector/model mapping

The downloader verifies these expected benchmark files after extraction. The
reproduction script maps each CSV to the detector/model name by stripping the
`.csv` suffix.

| Expected CSV file | Detector/model mapping | Row count | Column names |
| --- | --- | ---: | --- |
| `EffNetB0-preds.csv` | `EffNetB0-preds` | unavailable | unavailable |
| `EffNetB0-mirror-preds.csv` | `EffNetB0-mirror-preds` | unavailable | unavailable |
| `EffNetB1-preds.csv` | `EffNetB1-preds` | unavailable | unavailable |
| `EffNetB1-mirror-preds.csv` | `EffNetB1-mirror-preds` | unavailable | unavailable |
| `EffNetB2-preds.csv` | `EffNetB2-preds` | unavailable | unavailable |
| `EffNetB2-mirror-preds.csv` | `EffNetB2-mirror-preds` | unavailable | unavailable |
| `EffNetB3-preds.csv` | `EffNetB3-preds` | unavailable | unavailable |
| `EffNetB3-mirror-preds.csv` | `EffNetB3-mirror-preds` | unavailable | unavailable |
| `EffNetB4-preds.csv` | `EffNetB4-preds` | unavailable | unavailable |
| `EffNetB4-mirror-preds.csv` | `EffNetB4-mirror-preds` | unavailable | unavailable |
| `EffNetB5-preds.csv` | `EffNetB5-preds` | unavailable | unavailable |
| `EffNetB5-mirror-preds.csv` | `EffNetB5-mirror-preds` | unavailable | unavailable |
| `EffNetB6-preds.csv` | `EffNetB6-preds` | unavailable | unavailable |
| `EffNetB6-mirror-preds.csv` | `EffNetB6-mirror-preds` | unavailable | unavailable |
| `EffNetB7-preds.csv` | `EffNetB7-preds` | unavailable | unavailable |
| `EffNetB7-mirror-preds.csv` | `EffNetB7-mirror-preds` | unavailable | unavailable |
| `DetRS-valid.csv` | `DetRS-valid` | unavailable | unavailable |
| `DetRS-mirror-valid.csv` | `DetRS-mirror-valid` | unavailable | unavailable |
| `DetRS_resnet50-valid.csv` | `DetRS_resnet50-valid` | unavailable | unavailable |
| `DetRS_resnet50-mirror-valid.csv` | `DetRS_resnet50-mirror-valid` | unavailable | unavailable |
| `yolov5x_tta.csv` | `yolov5x_tta` | unavailable | unavailable |

## Expected CSV schema used by reproduction

When the archive is available, each CSV must contain these columns for the
reproduction loader:

```text
img_id,label,score,x1,x2,y1,y2
```

## Annotations and AP/AR computation

The benchmark archive is expected to contain detector prediction CSVs, not COCO
annotation JSON. COCO AP/AR can only be computed when `--annotations` points to a
COCO annotations file. Without annotations, the reproduction script does not fail
after loading/fusing predictions and reports:

```text
COCO AP/AR metrics cannot be computed because annotations are missing.
```
