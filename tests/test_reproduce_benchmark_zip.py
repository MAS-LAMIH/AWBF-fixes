import json
import subprocess
import sys
import zipfile

from scripts.download_benchmark import EXPECTED_FILES


def test_reproduce_uses_local_benchmark_zip(tmp_path):
    benchmark_zip = tmp_path / "benchmark.zip"
    row = "img_id,label,score,x1,x2,y1,y2\n1,1,0.9,0.0,1.0,0.0,1.0\n"
    with zipfile.ZipFile(benchmark_zip, "w") as zf:
        for name in EXPECTED_FILES:
            zf.writestr(name, row)

    benchmark_dir = tmp_path / "benchmark_cache"
    output_dir = tmp_path / "outputs"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_paper_results.py",
            "--benchmark-zip",
            str(benchmark_zip),
            "--benchmark-dir",
            str(benchmark_dir),
            "--output-dir",
            str(output_dir),
            "--incremental-cluster-state",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "COCO AP/AR metrics cannot be computed because annotations are missing" in result.stdout
    for filename in [
        "wbf_predictions.json",
        "awbf_predictions.json",
        "awbf_competition_predictions.json",
        "awbf_negotiation_predictions.json",
        "incremental_awbf_predictions.json",
        "reproduction_report.json",
    ]:
        assert (output_dir / filename).is_file()
    report = json.loads((output_dir / "reproduction_report.json").read_text())
    assert "Incremental_AWBF" in report["metrics"]
    assert "Incremental_AWBF" in report["timings_seconds"]
    assert report["notes"] == ["COCO AP/AR metrics cannot be computed because annotations are missing."]
