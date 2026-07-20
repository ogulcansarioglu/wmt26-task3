"""Data acquisition and conversion to the canonical segment table.

Subcommands:
    download    fetch WMT25 General-MT source + ESA humeval files into data/raw/
    build-dev   convert the humeval JSONL into the canonical dev parquet
    stats       per-language-pair coverage / label-balance table for the dev set
    make-smoke  emit a tiny synthetic canonical dataset for pipeline smoke tests
"""

from __future__ import annotations

import argparse
import random
import statistics
import subprocess
import sys
from pathlib import Path

from src.common import (
    ERROR_FREE_ESA_THRESHOLD,
    read_jsonl,
    write_jsonl,
)

RAW_DIR = Path("data/raw")
DEV_PARQUET = Path("data/dev/wmt25_esa_dev.parquet")

SOURCES = {
    "wmt25-genmt.jsonl": (
        "https://raw.githubusercontent.com/wmt-conference/wmt25-general-mt/"
        "main/data/wmt25-genmt.jsonl"
    ),
    "wmt25-genmt-humeval.jsonl": (
        "https://media.githubusercontent.com/media/wmt-conference/"
        "wmt25-general-mt/main/data/wmt25-genmt-humeval.jsonl"
    ),
}


def cmd_download(args: argparse.Namespace) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        dest = RAW_DIR / name
        if dest.exists() and dest.stat().st_size > 1024 and not args.force:
            print(f"[download] {name} already present ({dest.stat().st_size:,} B), skipping")
            continue
        print(f"[download] {url} -> {dest}")
        subprocess.run(["curl", "-sSL", "--fail", "-o", str(dest), url], check=True)
        print(f"[download] done: {dest.stat().st_size:,} B")


def _parse_doc_id(doc_id: str) -> tuple[str, str]:
    """`cs-de_DE_#_news_#_blesk.cz.112043_#_0` -> ("cs-de_DE", "news")."""
    parts = doc_id.split("_#_")
    if len(parts) < 2:
        return doc_id, "unknown"
    return parts[0], parts[1]


def _dev_rows(humeval_path: Path):
    for rec in read_jsonl(humeval_path):
        doc_id = rec["doc_id"]
        lp, domain = _parse_doc_id(doc_id)
        src = rec.get("src_text") or ""
        tgt_by_system = rec.get("tgt_text") or {}
        for system, annots in (rec.get("scores") or {}).items():
            mt = tgt_by_system.get(system)
            if not annots or mt is None or not str(mt).strip() or not src.strip():
                continue
            scores = [a["score"] for a in annots if a.get("score") is not None]
            if not scores:
                continue
            n_errors = sum(len(a.get("errors") or []) for a in annots)
            esa_mean = statistics.fmean(scores)
            per_annot_free = [
                (a["score"] >= ERROR_FREE_ESA_THRESHOLD and not a.get("errors"))
                for a in annots
                if a.get("score") is not None
            ]
            yield {
                "lp": lp,
                "domain": domain,
                "doc_id": doc_id,
                "item_id": doc_id,  # dev predates the official item_id; doc_id is the item key
                "system": system,
                "segment_id": f"{doc_id}::{system}",
                "src": src,
                "mt": str(mt),
                "esa_gold": esa_mean,
                "esa_min": min(scores),
                "n_annot": len(scores),
                "n_errors_total": n_errors,
                "error_free": int(esa_mean >= ERROR_FREE_ESA_THRESHOLD and n_errors == 0),
                "error_free_all": int(all(per_annot_free)),
            }


def cmd_build_dev(args: argparse.Namespace) -> None:
    import pandas as pd

    humeval = RAW_DIR / "wmt25-genmt-humeval.jsonl"
    if not humeval.exists():
        sys.exit(f"missing {humeval}; run `python -m src.data download` first")
    rows = list(_dev_rows(humeval))
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("no rows built — humeval file malformed?")
    dup = df["segment_id"].duplicated()
    if dup.any():
        sys.exit(f"duplicate segment_ids in dev build: {df.loc[dup, 'segment_id'].head()}")
    DEV_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DEV_PARQUET, index=False)
    print(f"[build-dev] wrote {len(df):,} rows, {df['lp'].nunique()} LPs -> {DEV_PARQUET}")
    _print_stats(df)


def _print_stats(df) -> None:
    g = df.groupby("lp").agg(
        n=("segment_id", "size"),
        error_free_rate=("error_free", "mean"),
        esa_mean=("esa_gold", "mean"),
        systems=("system", "nunique"),
    )
    g = g.sort_values("n", ascending=False)
    print("\n| lp | n | error_free_rate | esa_mean | systems |")
    print("|---|---|---|---|---|")
    for lp, row in g.iterrows():
        print(
            f"| {lp} | {int(row['n'])} | {row['error_free_rate']:.3f} "
            f"| {row['esa_mean']:.1f} | {int(row['systems'])} |"
        )
    total_free = df["error_free"].mean()
    print(f"\nTOTAL: {len(df):,} rows, global error_free rate {total_free:.3f}")


def cmd_stats(args: argparse.Namespace) -> None:
    import pandas as pd

    df = pd.read_parquet(DEV_PARQUET)
    _print_stats(df)


# ---------------------------------------------------------------------------
# official WMT26 test data (unified schema, published 2026-07-18)
# ---------------------------------------------------------------------------


def _parse_item_id(item_id: str) -> tuple[str, str]:
    """`srcLang_###_tgtLang_###_domain_###_docID_###_segID` -> (lp, domain)."""
    parts = item_id.split("_###_")
    if len(parts) < 5:
        sys.exit(f"unexpected item_id format (want 5 '_###_' fields): {item_id!r}")
    return f"{parts[0]}-{parts[1]}", parts[2]


def cmd_convert_test(args: argparse.Namespace) -> None:
    """Official test JSONL -> canonical parquet. Unlike the dev build, EVERY
    (item, system) pair is kept — the submission must cover all of them, so an
    empty hypothesis becomes an empty-string row (scores near zero, label 0)
    rather than a silently dropped prediction."""
    import pandas as pd

    rows, n_empty_hyp = [], 0
    for rec in read_jsonl(Path(args.raw)):
        item_id = rec["item_id"]
        lp, domain = _parse_item_id(item_id)
        src = rec.get("src") or ""
        ref = rec.get("ref") or {}
        for system, mt in (rec.get("hyps") or {}).items():
            mt = "" if mt is None else str(mt)
            if not mt.strip():
                n_empty_hyp += 1
            rows.append(
                {
                    "lp": lp,
                    "domain": domain,
                    "doc_id": item_id,
                    "item_id": item_id,
                    "system": str(system),
                    "segment_id": f"{item_id}::{system}",
                    "src": src,
                    "mt": mt,
                    "ref_text": (ref.get("text") if isinstance(ref, dict) else None),
                    "ref_type": (ref.get("type") if isinstance(ref, dict) else None),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("no rows converted — wrong input file?")
    dup = df["segment_id"].duplicated()
    if dup.any():
        sys.exit(f"duplicate (item, system) pairs: {df.loc[dup, 'segment_id'].head().tolist()}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(
        f"[convert-test] {len(df):,} (item, system) rows, {df['item_id'].nunique():,} items, "
        f"{df['lp'].nunique()} LPs, {n_empty_hyp} empty hypotheses -> {out}"
    )
    print(df.groupby("lp").size().to_string())


# ---------------------------------------------------------------------------
# synthetic smoke data
# ---------------------------------------------------------------------------

_WORDS = (
    "orbit lantern crystal meadow harbor velvet quantum ember stride canyon "
    "monsoon parchment gable turbine sable lattice pearl arbor cinder fjord"
).split()


def _fake_sentence(rng: random.Random, n_words: int, num: int | None = None) -> str:
    words = [rng.choice(_WORDS) for _ in range(n_words)]
    if num is not None:
        words.insert(rng.randrange(len(words)), str(num))
    return " ".join(words).capitalize() + "."


def _fake_translation(src_sentence: str) -> str:
    """Deterministic pseudo-translation: every word gains a '-tr' suffix.
    The mock scorer keys on this, so corruptions are detectable."""
    out = []
    for w in src_sentence.rstrip(".").split():
        out.append(w if w.isdigit() else w + "-tr")
    return " ".join(out) + "."


def _corrupt(rng: random.Random, mt_sentence: str, mode: str) -> str:
    words = mt_sentence.rstrip(".").split()
    if mode == "drop_suffix":
        idx = rng.randrange(len(words))
        words[idx] = "ERRTOKEN"
    elif mode == "number":
        for i, w in enumerate(words):
            if w.isdigit():
                words[i] = str(int(w) + rng.randrange(1, 9))
                break
        else:
            words.append("999")
    elif mode == "duplicate":
        idx = rng.randrange(len(words))
        words.insert(idx, "ERRTOKEN")
    return " ".join(words) + "."


def cmd_make_smoke(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    records = []
    plan = [
        # (lp, n_sentences, corruptions) — long segments force the chunking path
        ("en-de_DE", 3, 0),
        ("en-de_DE", 4, 2),
        ("en-de_DE", 60, 0),
        ("en-de_DE", 70, 12),
        ("en-de_DE", 5, 1),
        ("en-xx_XX", 3, 0),
        ("en-xx_XX", 50, 0),
        ("en-xx_XX", 55, 10),
        ("en-xx_XX", 4, 2),
        ("en-xx_XX", 6, 0),
    ]
    for i, (lp, n_sents, n_corrupt) in enumerate(plan):
        src_sents = [
            _fake_sentence(rng, rng.randrange(6, 14), num=rng.randrange(10, 99))
            for _ in range(n_sents)
        ]
        mt_sents = [_fake_translation(s) for s in src_sents]
        for k in range(n_corrupt):
            idx = rng.randrange(len(mt_sents))
            mode = ("drop_suffix", "number", "duplicate")[k % 3]
            mt_sents[idx] = _corrupt(rng, mt_sents[idx], mode)
        error_free = int(n_corrupt == 0)
        esa = 100.0 - 9.0 * min(n_corrupt, 6) - (0.0 if error_free else 3.0)
        doc_id = f"{lp}_#_smoke_#_doc{i}_#_0"
        records.append(
            {
                "lp": lp,
                "domain": "smoke",
                "doc_id": doc_id,
                "item_id": doc_id,
                "system": "synthetic",
                "segment_id": f"{doc_id}::synthetic",
                "src": "\n".join(src_sents),
                "mt": "\n".join(mt_sents),
                "esa_gold": esa,
                "esa_min": esa,
                "n_annot": 1,
                "n_errors_total": n_corrupt,
                "error_free": error_free,
                "error_free_all": error_free,
            }
        )
    out = Path(args.out)
    n = write_jsonl(out, records)
    print(f"[make-smoke] wrote {n} synthetic segments -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="src.data")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("download", help="fetch WMT25 dev source files")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("build-dev", help="humeval JSONL -> canonical dev parquet")
    p.set_defaults(func=cmd_build_dev)

    p = sub.add_parser("stats", help="per-LP dev coverage table")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("convert-test", help="official WMT26 test JSONL -> canonical parquet")
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_convert_test)

    p = sub.add_parser("make-smoke", help="synthetic smoke dataset")
    p.add_argument("--out", default="data/smoke/smoke_dev.jsonl")
    p.add_argument("--seed", type=int, default=13)
    p.set_defaults(func=cmd_make_smoke)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
