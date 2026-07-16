import json

import pandas as pd
import pytest

from src.submit import apply_thresholds, validate, write_task_file


@pytest.fixture
def scored_df():
    return pd.DataFrame(
        {
            "lp": ["en-de_DE", "en-de_DE", "en-xx_XX"],
            "doc_id": ["d1", "d2", "d3"],
            "system": ["sysA", "sysA", "sysB"],
            "segment_id": ["d1::sysA", "d2::sysA", "d3::sysB"],
            "score_min": [0.9, 0.4, 0.7],
        }
    )


THRESHOLDS = {
    "global": {"threshold": 0.6},
    "per_lp": {"en-de_DE": {"threshold": 0.5, "source": "per_lp"}},
}


def test_apply_thresholds_uses_per_lp_then_global(scored_df):
    out = apply_thresholds(scored_df, THRESHOLDS, "min")
    assert out["label"].tolist() == [1, 0, 1]


def test_write_and_validate_roundtrip(tmp_path, scored_df):
    df = apply_thresholds(scored_df, THRESHOLDS, "min")
    path = write_task_file(df, 3, tmp_path)
    assert validate(path, df, 3) == []


def test_validator_catches_missing_and_duplicate_rows(tmp_path, scored_df):
    df = apply_thresholds(scored_df, THRESHOLDS, "min")
    path = write_task_file(df, 3, tmp_path)
    lines = path.read_text().strip().split("\n")
    path.write_text("\n".join([lines[0], lines[0], lines[2]]) + "\n")
    problems = validate(path, df, 3)
    assert any("duplicate" in p for p in problems)
    assert any("missing" in p for p in problems)


def test_validator_catches_bad_label_domain(tmp_path, scored_df):
    df = apply_thresholds(scored_df, THRESHOLDS, "min")
    path = write_task_file(df, 3, tmp_path)
    lines = path.read_text().strip().split("\n")
    rec = json.loads(lines[0])
    rec["error_free"] = 0.7
    lines[0] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n")
    problems = validate(path, df, 3)
    assert any("not in {0,1}" in p for p in problems)


def test_validator_catches_task2_score_domain(tmp_path, scored_df):
    df = apply_thresholds(scored_df, THRESHOLDS, "min")
    path = write_task_file(df, 2, tmp_path)
    lines = path.read_text().strip().split("\n")
    rec = json.loads(lines[0])
    rec["score"] = 250.0
    lines[0] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n")
    problems = validate(path, df, 2)
    assert any("outside [0, 100]" in p for p in problems)
