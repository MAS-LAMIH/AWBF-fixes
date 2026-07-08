import csv

from scripts import sweep_awbf_parameters as sweep


def test_parameter_sweep_sample_writes_csv(tmp_path):
    output_dir = tmp_path / "sweep"
    sweep.main(["--sample", "--output-dir", str(output_dir)])
    assert (output_dir / "PARAMETER_SWEEP_RESULTS.csv").is_file()


def test_parameter_sweep_run_sweep_returns_expected_rows(tmp_path):
    class Args:
        sample = True
        benchmark_dir = None
        predictions_dir = None
        annotations = None
        bbox_scale = "auto"
        output_dir = str(tmp_path / "sweep")
        iou_threshold = 0.55
        score_threshold = 0.0
        cooperation_threshold = 0.5
        negotiation_threshold = 0.05
        rounds = 5
        weight = 0.3
        disable_preclustering = False

    csv_path = sweep.run_sweep(Args())
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    assert len(rows) == len(sweep.T_VALUES) + len(sweep.W_VALUES) + len(sweep.RMAX_VALUES)
    assert {"T", "W", "Rmax"} == {row["parameter"] for row in rows}
    assert "AP" in rows[0]
    assert "AR" in rows[0]
