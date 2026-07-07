# Reproduction status

## Download success/failure

Status: **failed in this execution environment**.

The required benchmark archive URL was attempted:

```text
https://github.com/ZFTurbo/Weighted-Boxes-Fusion/releases/download/v1.0.5/benchmark.zip
```

Command attempted:

```bash
python scripts/download_benchmark.py --output-dir data/benchmark
```

Observed failure:

```text
urllib.error.URLError: <urlopen error Tunnel connection failed: 403 Forbidden>
```

This is an environment network/proxy limitation. The code now supports a manually
downloaded archive via `--benchmark-zip`, which lets users run the same extraction
and validation path without network access from the execution environment.

## Extracted files

No files were extracted from the real benchmark archive in this environment.
The expected extraction directory is:

```text
data/benchmark/benchmark/
```

The expected file list is documented in `BENCHMARK_CONTENTS.md` and enforced by
`scripts/download_benchmark.py`.

## Benchmark statistics

Unavailable in this environment because the real CSV files were not downloaded.
`BENCHMARK_CONTENTS.md` lists the expected CSV files and records row counts and
columns as unavailable for this run.

## Reproduction command status

Command attempted:

```bash
python scripts/reproduce_paper_results.py \
  --benchmark-dir data/benchmark \
  --output-dir outputs/reproduction_benchmark
```

Result: **failed before fusion** because `data/benchmark/benchmark.zip` and the
extracted benchmark CSVs were unavailable, and the automatic download attempt was
blocked by the same proxy restriction.

## Generated outputs

The real benchmark outputs could not be generated in this environment. After a
successful benchmark download or by passing a manually downloaded zip, the script
will write:

```text
outputs/reproduction_benchmark/wbf_predictions.json
outputs/reproduction_benchmark/awbf_predictions.json
outputs/reproduction_benchmark/awbf_competition_predictions.json
outputs/reproduction_benchmark/awbf_negotiation_predictions.json
outputs/reproduction_benchmark/reproduction_report.json
```

The output-filename path is covered by automated tests using a synthetic local
benchmark zip.

## AP/AR metrics

AP/AR metrics were **not computed** in this environment.

Reasons:

1. Benchmark predictions could not be downloaded.
2. COCO annotations were not present.

When predictions are available but annotations are omitted, the reproduction
script reports:

```text
COCO AP/AR metrics cannot be computed because annotations are missing.
```

## Remaining requirements to reproduce the paper

1. Provide network access to the GitHub release asset or manually download
   `benchmark.zip` and pass it with `--benchmark-zip`.
2. Run:

   ```bash
   python scripts/reproduce_paper_results.py \
     --benchmark-zip /path/to/benchmark.zip \
     --benchmark-dir data/benchmark \
     --output-dir outputs/reproduction_benchmark
   ```

3. To compute COCO AP/AR, also provide COCO annotations:

   ```bash
   python scripts/reproduce_paper_results.py \
     --benchmark-zip /path/to/benchmark.zip \
     --benchmark-dir data/benchmark \
     --annotations data/coco/annotations/instances_val2017.json \
     --output-dir outputs/reproduction_benchmark
   ```

4. Compare generated metrics with the reported paper targets in
   `PAPER_SPEC.md`.
