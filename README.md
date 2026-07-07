
## Adaptive Weighted boxes fusion

Repository based on [![DOI](https://zenodo.org/badge/217881799.svg)](https://zenodo.org/badge/latestdoi/217881799)
containing Python implementation of several methods for ensembling boxes from object detection models: 

* Non-maximum Suppression (NMS)
* Soft-NMS [[1]](https://arxiv.org/abs/1704.04503)
* Non-maximum weighted (NMW) [[2]](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w14/Zhou_CAD_Scale_Invariant_ICCV_2017_paper.pdf)
* **Weighted boxes fusion (WBF)** [[3]](https://arxiv.org/abs/1910.13302) - new method which gives better results comparing to others 

In addition to a multi-agent system (MAS) that implement WBF in a decentralized and adaptive manner.

## Requirements

Python 3.*, Numpy, Numba

## Examples 
Pleas refer to main.py and AWBF_Examples folder (jupiter notebook for visualizing the bounding box evolution)



## Description of AWBF method and citation

* https://ceur-ws.org/Vol-3813/13.pdf

If you find this code useful please cite:

```
@inproceedings{daoud2024introducing,
  title={Introducing Multiagent Systems to AV Visual Perception Sub-tasks: A proof-of-concept implementation for bounding-box improvement},
  author={Daoud, Alaa and Bunel, Corentin and Gu{\'e}riau, Maxime},
  booktitle={13th International Workshop on Agents in Traffic and Transportation (ATT 2024) held in conjunction with ECAI 2024},
  year={2024},
  organization={CEUR-WS}
}
```

## Reproducing paper/benchmark fusion outputs

The reproduction script can run without COCO annotations when you only need to
validate download, fusion, and COCO-format prediction export. COCO AP/AR metrics
are computed only when `--annotations` points to a COCO annotations JSON.

### Download the WBF benchmark files

```bash
python scripts/download_benchmark.py --output-dir data/benchmark
```

For a manually downloaded archive, skip the network download and extract/validate
the local file instead:

```bash
python scripts/download_benchmark.py \
  --benchmark-zip /path/to/benchmark.zip \
  --output-dir data/benchmark
```

This downloads `benchmark.zip` from:

```text
https://github.com/ZFTurbo/Weighted-Boxes-Fusion/releases/download/v1.0.5/benchmark.zip
```

The archive is cached at `data/benchmark/benchmark.zip`. If the zip already
exists, the downloader reuses it. Files are extracted with zip-slip protection to
`data/benchmark/benchmark/`, and the downloader validates the expected benchmark
CSV files after extraction.

### Run benchmark fusion/export without COCO evaluation

```bash
python scripts/reproduce_paper_results.py \
  --download-benchmark \
  --benchmark-dir data/benchmark \
  --output-dir outputs/reproduction_benchmark
```

If you manually downloaded `benchmark.zip` (for example on a machine with browser
access to GitHub release assets), use the local zip directly:

```bash
python scripts/reproduce_paper_results.py \
  --benchmark-zip /path/to/benchmark.zip \
  --benchmark-dir data/benchmark \
  --output-dir outputs/reproduction_benchmark
```

This command downloads/extracts the benchmark only if expected files are missing,
runs WBF/AWBF/AWBF-competition/AWBF-Negotiation fusion, writes COCO-format
prediction JSONs, and stores `outputs/reproduction_benchmark/reproduction_report.json`.
Because no COCO annotations are supplied, AP/AR metric evaluation is skipped with: `COCO AP/AR metrics cannot be computed because annotations are missing.`

### Run full COCO metric evaluation

Provide COCO annotations when you want AP/AR metrics:

```bash
python scripts/reproduce_paper_results.py \
  --benchmark-dir data/benchmark \
  --annotations data/coco/annotations/instances_val2017.json \
  --bbox-scale auto \
  --output-dir outputs/reproduction_benchmark
```

You can also bypass the benchmark downloader and provide your own COCO detection
JSON files, one file per detector/model:

```bash
python scripts/reproduce_paper_results.py \
  --predictions-dir data/predictions \
  --annotations data/coco/annotations/instances_val2017.json \
  --output-dir outputs/reproduction_custom
```

### Smoke test without external data

```bash
python scripts/reproduce_paper_results.py --sample --output-dir outputs/reproduction_sample
```

### Progress output for long benchmark runs

Benchmark reproduction can take a while on large CSV files. The reproduction
script now prints progress while loading CSVs, fusing images, exporting JSONs,
and evaluating COCO metrics. Adjust image-level fusion progress frequency with:

```bash
python scripts/reproduce_paper_results.py \
  --benchmark-dir data/benchmark \
  --output-dir outputs/reproduction_benchmark \
  --progress-interval 25 \
  --profile
```

`--profile` adds per-strategy timings, cluster statistics, and per-negotiation-pass comparison counts. Pre-clustering is enabled by default; add `--disable-preclustering` only when you need to compare against the previous global per-image behavior.


### Selecting fusion methods

Use `--methods` to run one method, several methods, or `all` (the default).
This is useful when you only need to rerun a slow or experimental variant.

Run only Incremental_AWBF:

```bash
python scripts/reproduce_paper_results.py \
  --benchmark-dir data/benchmark \
  --annotations data/coco/annotations/instances_val2017.json \
  --bbox-scale auto \
  --methods Incremental_AWBF \
  --output-dir outputs/reproduction_incremental_awbf
```

Run WBF and Incremental_AWBF only:

```bash
python scripts/reproduce_paper_results.py \
  --benchmark-dir data/benchmark \
  --methods WBF Incremental_AWBF \
  --output-dir outputs/reproduction_wbf_incremental
```

Run all methods explicitly:

```bash
python scripts/reproduce_paper_results.py \
  --benchmark-dir data/benchmark \
  --methods all \
  --output-dir outputs/reproduction_benchmark
```

Valid method names are `WBF`, `AWBF`, `Incremental_AWBF`,
`AWBF-competition`, `AWBF-competition-IncrementalState`,
`AWBF-Negotiation`, and `AWBF-Negotiation-IncrementalState`.

### Direct evaluation of existing COCO prediction JSONs

If fused prediction JSONs already exist, evaluate them directly without loading
benchmark CSVs or running fusion:

```bash
python scripts/reproduce_paper_results.py \
  --annotations data/coco/annotations/instances_val2017.json \
  --evaluate-predictions outputs/reproduction_benchmark/wbf_predictions.json \
  --bbox-scale auto \
  --output-dir outputs/eval_wbf
```

Benchmark CSVs and some exported prediction JSONs may contain normalized COCO
`xywh` coordinates in `[0, 1]`; COCO evaluation requires pixel-space `xywh`. Use
`--bbox-scale auto` to convert normalized boxes using image width/height from the
annotations file. Direct evaluation mode requires `--annotations`, prints metrics
for each input file, saves converted files like
`outputs/eval_wbf/wbf_predictions_pixel_xywh.json` when conversion is needed, and
writes `outputs/eval_wbf/evaluation_report.json`.

### Incremental AWBF experiment

`Incremental_AWBF` is exported alongside the paper-style fusion strategies. It
uses the same WBF cluster-assignment logic as AWBF, but keeps each cluster
representative as running score/weight state instead of recomputing the final
confidence-weighted box in one shot. With fixed membership this is an
agent-style execution variant of AWBF, not necessarily a distinct paper metric
unless you explicitly report it as such.

Competition and negotiation incremental-state variants are always written
separately so they cannot overwrite the paper-style outputs:

- `awbf_competition_predictions.json`
- `awbf_competition_incremental_state_predictions.json`
- `awbf_negotiation_predictions.json`
- `awbf_negotiation_incremental_state_predictions.json`
- `incremental_awbf_predictions.json`

```bash
python scripts/reproduce_paper_results.py \
  --benchmark-dir data/benchmark \
  --output-dir outputs/reproduction_benchmark \
  --profile
```

The run also writes `outputs/reproduction_benchmark/METHOD_EQUIVALENCE_AUDIT.md`
and includes the same comparison data under `method_equivalence_audit` in
`reproduction_report.json`.

### Parameter sweeps

Use the sweep script to test the paper's parameter claims for competition
threshold `T`, negotiation weight `W`, and maximum rounds `Rmax`:

```bash
python scripts/sweep_awbf_parameters.py \
  --benchmark-dir data/benchmark \
  --annotations data/coco/annotations/instances_val2017.json \
  --bbox-scale auto \
  --output-dir outputs/parameter_sweep
```

The script writes `PARAMETER_SWEEP_RESULTS.csv` and, when `matplotlib` is
available, plots under `outputs/parameter_sweep/plots/`.
