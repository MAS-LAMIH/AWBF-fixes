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

    def fail_urlretrieve(_url, _path):
        raise AssertionError("urlretrieve should not be called when cache exists")

    monkeypatch.setattr("scripts.download_benchmark.urlretrieve", fail_urlretrieve)
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
