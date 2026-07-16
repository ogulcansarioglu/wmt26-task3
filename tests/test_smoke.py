"""End-to-end pipeline smoke: synthetic data -> mock scoring -> calibration ->
submission packaging, all through the real CLIs. Mirrors `make smoke`."""

import json
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(*argv: str) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [sys.executable, "-m", *argv],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"{argv} failed:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
    return proc


def test_full_pipeline_on_synthetic_data(tmp_path):
    data = tmp_path / "smoke_dev.jsonl"
    run_dir = tmp_path / "run"
    sub_dir = tmp_path / "sub"

    run("src.data", "make-smoke", "--out", str(data))
    run(
        "src.score",
        "--model",
        "mock",
        "--input",
        str(data),
        "--run",
        str(run_dir),
        "--mode",
        "chunked",
        "--device",
        "cpu",
    )
    run(
        "src.calibrate",
        "--run",
        str(run_dir),
        "--model",
        "mock",
        "--mode",
        "chunked",
        "--agg",
        "min",
        "--min-n",
        "4",
        "--min-class",
        "1",
    )
    run(
        "src.submit",
        "--run",
        str(run_dir),
        "--model",
        "mock",
        "--mode",
        "chunked",
        "--agg",
        "min",
        "--task",
        "3",
        "--out",
        str(sub_dir),
    )

    # score checkpoints exist per LP
    parquets = list((run_dir / "scores" / "mock" / "chunked").glob("*.parquet"))
    assert len(parquets) == 2
    # manifest recorded provenance and per-LP stats
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert "score" in manifest["stages"]
    # thresholds + archive produced and validated
    thresholds = json.loads(
        (run_dir / "calibration" / "mock_chunked_min" / "thresholds.json").read_text()
    )
    assert "global" in thresholds
    archive = sub_dir / "submission_task3.zip"
    assert archive.exists()
    with zipfile.ZipFile(archive) as zf:
        assert zf.namelist() == ["task3.jsonl"]

    # resumability: rerunning score skips completed LPs
    proc = run(
        "src.score",
        "--model",
        "mock",
        "--input",
        str(data),
        "--run",
        str(run_dir),
        "--mode",
        "chunked",
        "--device",
        "cpu",
    )
    assert proc.stdout.count("skipping") == 2
