import json

import pandas as pd
import pytest

from src.submit import _item_key_column, apply_thresholds, validate, write_task_file


@pytest.fixture
def scored_df():
    # two systems on itemA, one on itemB — exercises per-item nesting
    return pd.DataFrame(
        {
            "lp": ["en-de_DE", "en-de_DE", "en-xx_XX"],
            "item_id": ["en_###_de_DE_###_news_###_d1_###_0"] * 2
            + ["en_###_xx_XX_###_social_###_d3_###_0"],
            "doc_id": ["en_###_de_DE_###_news_###_d1_###_0"] * 2
            + ["en_###_xx_XX_###_social_###_d3_###_0"],
            "system": ["sysA", "sysB", "sysA"],
            "segment_id": ["d1::sysA", "d1::sysB", "d3::sysA"],
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


def test_item_key_falls_back_to_doc_id(scored_df):
    assert _item_key_column(scored_df) == "item_id"
    assert _item_key_column(scored_df.drop(columns=["item_id"])) == "doc_id"


def test_write_nests_predictions_per_item(tmp_path, scored_df):
    df = apply_thresholds(scored_df, THRESHOLDS, "min")
    path = write_task_file(df, 3, tmp_path)
    records = [json.loads(ln) for ln in path.read_text().strip().split("\n")]
    assert len(records) == 2  # two items, not three rows
    by_item = {r["item_id"]: r["task3_pred"] for r in records}
    assert by_item["en_###_de_DE_###_news_###_d1_###_0"] == {"sysA": 1, "sysB": 0}
    assert by_item["en_###_xx_XX_###_social_###_d3_###_0"] == {"sysA": 1}


def test_write_and_validate_roundtrip(tmp_path, scored_df):
    df = apply_thresholds(scored_df, THRESHOLDS, "min")
    for task in (2, 3):
        path = write_task_file(df, task, tmp_path)
        assert validate(path, df, task) == []


def test_validator_catches_duplicate_and_missing_items(tmp_path, scored_df):
    df = apply_thresholds(scored_df, THRESHOLDS, "min")
    path = write_task_file(df, 3, tmp_path)
    lines = path.read_text().strip().split("\n")
    path.write_text("\n".join([lines[0], lines[0]]) + "\n")  # dup itemA, drop itemB
    problems = validate(path, df, 3)
    assert any("duplicate item_id" in p for p in problems)
    assert any("missing" in p for p in problems)


def test_validator_catches_missing_system_within_item(tmp_path, scored_df):
    df = apply_thresholds(scored_df, THRESHOLDS, "min")
    path = write_task_file(df, 3, tmp_path)
    records = [json.loads(ln) for ln in path.read_text().strip().split("\n")]
    del records[0]["task3_pred"]["sysB"]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    problems = validate(path, df, 3)
    assert any("pairs missing" in p for p in problems)


def test_validator_catches_bad_label_domain(tmp_path, scored_df):
    df = apply_thresholds(scored_df, THRESHOLDS, "min")
    path = write_task_file(df, 3, tmp_path)
    records = [json.loads(ln) for ln in path.read_text().strip().split("\n")]
    records[0]["task3_pred"]["sysA"] = 0.7
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    problems = validate(path, df, 3)
    assert any("not in {0,1}" in p for p in problems)


def test_validator_rejects_boolean_labels(tmp_path, scored_df):
    df = apply_thresholds(scored_df, THRESHOLDS, "min")
    path = write_task_file(df, 3, tmp_path)
    records = [json.loads(ln) for ln in path.read_text().strip().split("\n")]
    records[0]["task3_pred"]["sysA"] = True
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    problems = validate(path, df, 3)
    assert any("not in {0,1}" in p for p in problems)


def test_validator_catches_task2_score_domain(tmp_path, scored_df):
    df = apply_thresholds(scored_df, THRESHOLDS, "min")
    path = write_task_file(df, 2, tmp_path)
    records = [json.loads(ln) for ln in path.read_text().strip().split("\n")]
    records[0]["task2_pred"]["sysA"] = 250.0
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    problems = validate(path, df, 2)
    assert any("outside [0, 100]" in p for p in problems)
