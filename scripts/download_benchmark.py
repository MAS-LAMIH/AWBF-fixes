#!/usr/bin/env python3
"""Download and safely extract the WBF COCO benchmark files."""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

BENCHMARK_URL = "https://github.com/ZFTurbo/Weighted-Boxes-Fusion/releases/download/v1.0.5/benchmark.zip"
ZIP_NAME = "benchmark.zip"
EXTRACT_DIR_NAME = "benchmark"
EXPECTED_FILES = (
    "EffNetB0-preds.csv",
    "EffNetB0-mirror-preds.csv",
    "EffNetB1-preds.csv",
    "EffNetB1-mirror-preds.csv",
    "EffNetB2-preds.csv",
    "EffNetB2-mirror-preds.csv",
    "EffNetB3-preds.csv",
    "EffNetB3-mirror-preds.csv",
    "EffNetB4-preds.csv",
    "EffNetB4-mirror-preds.csv",
    "EffNetB5-preds.csv",
    "EffNetB5-mirror-preds.csv",
    "EffNetB6-preds.csv",
    "EffNetB6-mirror-preds.csv",
    "EffNetB7-preds.csv",
    "EffNetB7-mirror-preds.csv",
    "DetRS-valid.csv",
    "DetRS-mirror-valid.csv",
    "DetRS_resnet50-valid.csv",
    "DetRS_resnet50-mirror-valid.csv",
    "yolov5x_tta.csv",
)


def benchmark_zip_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / ZIP_NAME


def benchmark_extract_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / EXTRACT_DIR_NAME


def download_benchmark_zip(output_dir: str | Path, url: str = BENCHMARK_URL) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    zip_path = benchmark_zip_path(output)
    if zip_path.exists() and zip_path.stat().st_size > 0:
        print(f"Using cached benchmark archive: {zip_path}")
        return zip_path
    print(f"Downloading benchmark archive from {url} to {zip_path}")
    urlretrieve(url, zip_path)
    return zip_path


def _safe_target(base: Path, member_name: str) -> Path:
    target = (base / member_name).resolve()
    base_resolved = base.resolve()
    if target != base_resolved and base_resolved not in target.parents:
        raise ValueError(f"Unsafe path in benchmark zip: {member_name}")
    return target


def safe_extract(zip_path: str | Path, extract_dir: str | Path) -> Path:
    extract_path = Path(extract_dir)
    extract_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            _safe_target(extract_path, member.filename)
        zf.extractall(extract_path)
    return extract_path


def find_expected_file(extract_dir: str | Path, filename: str) -> Path | None:
    matches = list(Path(extract_dir).rglob(filename))
    return matches[0] if matches else None


def validate_benchmark_files(extract_dir: str | Path, expected_files: tuple[str, ...] = EXPECTED_FILES) -> list[Path]:
    missing = [name for name in expected_files if find_expected_file(extract_dir, name) is None]
    if missing:
        raise FileNotFoundError(
            "Benchmark extraction is missing expected files: "
            + ", ".join(missing)
            + f". Re-run scripts/download_benchmark.py --output-dir {Path(extract_dir).parent} or check the archive."
        )
    return [find_expected_file(extract_dir, name) for name in expected_files if find_expected_file(extract_dir, name) is not None]


def ensure_benchmark(output_dir: str | Path) -> Path:
    zip_path = download_benchmark_zip(output_dir)
    extract_dir = benchmark_extract_dir(output_dir)
    try:
        validate_benchmark_files(extract_dir)
        print(f"Using existing extracted benchmark files: {extract_dir}")
        return extract_dir
    except FileNotFoundError:
        pass
    safe_extract(zip_path, extract_dir)
    validate_benchmark_files(extract_dir)
    print(f"Benchmark ready: {extract_dir}")
    return extract_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/benchmark", help="Directory for benchmark.zip and extracted benchmark/ files")
    args = parser.parse_args(argv)
    ensure_benchmark(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
