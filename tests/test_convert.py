import json
from argparse import Namespace

import pandas as pd

from src.data import cmd_convert_test


def test_convert_test_keeps_all_pairs_including_empty_hyps(tmp_path):
    raw = tmp_path / "test.jsonl"
    recs = [
        {
            "item_id": "en_###_de_DE_###_news_###_doc1_###_0",
            "src": "Hello world.",
            "ref": {"text": "Hallo Welt.", "type": "human"},
            "hyps": {"sysA": "Hallo Welt.", "sysB": ""},
        },
        {
            "item_id": "cs_###_uk_UA_###_social_###_doc2_###_3",
            "src": "Ahoj.",
            "ref": None,
            "hyps": {"sysA": "Привіт."},
        },
    ]
    raw.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8")
    out = tmp_path / "test.parquet"
    cmd_convert_test(Namespace(raw=str(raw), out=str(out)))
    df = pd.read_parquet(out)

    assert len(df) == 3  # every (item, system) pair kept
    assert set(df["lp"]) == {"en-de_DE", "cs-uk_UA"}
    assert set(df["domain"]) == {"news", "social"}
    assert (df["item_id"] == df["doc_id"]).all()
    empty = df[df["mt"] == ""]
    assert len(empty) == 1 and empty.iloc[0]["system"] == "sysB"
    assert df.loc[df["system"] == "sysA", "segment_id"].is_unique
