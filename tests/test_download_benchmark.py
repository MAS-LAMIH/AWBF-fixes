import zipfile

import pytest

from scripts.download_benchmark import (
    BENCHMARK_URL,
    EXPECTED_FILES,
    download_benchmark_zip,
    safe_extract,
    validate_benchmark_files,
)


def test_benchmark_url_constant():
    assert BENCHMARK_URL == "https://github.com/ZFTurbo/Weighted-Boxes-Fusion/releases/download/v1.0.5/benchmark.zip"


def test_download_uses_cached_zip(tmp_path, monkeypatch):
    cached = tmp_path / "benchmark.zip"
    cached.write_bytes(b"already here")

    def fail_urlopen(_url, _path):
        raise AssertionError("urlopen should not be called when cache exists")

    monkeypatch.setattr("scripts.download_benchmark.urlopen", fail_urlopen)
    assert download_benchmark_zip(tmp_path) == cached
    assert cached.read_bytes() == b"already here"


def test_safe_extract_prevents_zip_slip(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../evil.txt", "owned")
    with pytest.raises(ValueError, match="Unsafe path"):
        safe_extract(archive, tmp_path / "benchmark")


def test_missing_file_error_message(tmp_path):
    extract_dir = tmp_path / "benchmark"
    extract_dir.mkdir()
    with pytest.raises(FileNotFoundError, match=EXPECTED_FILES[0]):
        validate_benchmark_files(extract_dir)

from scripts.download_benchmark import ensure_benchmark, use_local_benchmark_zip


def _write_minimal_benchmark_zip(path):
    row = "img_id,label,score,x1,x2,y1,y2\n1,1,0.9,0.0,1.0,0.0,1.0\n"
    with zipfile.ZipFile(path, "w") as zf:
        for name in EXPECTED_FILES:
            zf.writestr(name, row)


def test_local_zip_input_is_copied_extracted_and_validated(tmp_path, monkeypatch):
    local_zip = tmp_path / "manual_benchmark.zip"
    _write_minimal_benchmark_zip(local_zip)
    output_dir = tmp_path / "cache"

    def fail_urlopen(_url, _path):
        raise AssertionError("urlopen should not be called for --benchmark-zip")

    monkeypatch.setattr("scripts.download_benchmark.urlopen", fail_urlopen)
    cached_zip = use_local_benchmark_zip(output_dir, local_zip)
    assert cached_zip == output_dir / "benchmark.zip"
    assert cached_zip.read_bytes() == local_zip.read_bytes()

    extract_dir = ensure_benchmark(output_dir, benchmark_zip=local_zip)
    validated = validate_benchmark_files(extract_dir)
    assert len(validated) == len(EXPECTED_FILES)
