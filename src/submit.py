"""Submission formatter + validator + packager.

FORMAT STATUS (2026-07-16): the WMT26 unified JSON schema and Codabench
packaging are announced ~1 week before the 23 July test release. Until then
this module emits a clearly-marked DRAFT format modeled on the General-MT
JSONL the organizers said they will follow. ALL format-specific logic lives in
`write_task_file()` — update that one function (and the validator's field
list) when the real schema drops, re-run, done.

Usage:
    python -m src.submit --run runs/dev --model cometkiwi --mode chunked --agg min \
        --thresholds runs/dev/calibration/cometkiwi_chunked_min/thresholds.json \
        --task 3 --out submissions/dev-rehearsal

The validator re-reads what was written and checks row counts, ID alignment
against the scored inputs, label domain, score domain, and UTF-8 before
zipping. A submission that fails validation is deleted, not packaged.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

from src.calibrate import load_scores

FORMAT_VERSION = "draft-2026-07-16"  # bump when the official schema lands

TEAM_NAME = "osar-mps"  # placeholder team id; confirm on Codabench registration


def apply_thresholds(df: pd.DataFrame, thresholds: dict, agg: str) -> pd.DataFrame:
    score_col = f"score_{agg}"
    g = float(thresholds["global"]["threshold"])
    per_lp = thresholds.get("per_lp", {})

    def label(row) -> int:
        thr = per_lp.get(row["lp"], {}).get("threshold", g)
        return int(row[score_col] > thr)

    out = df.copy()
    out["label"] = out.apply(label, axis=1)
    out["score_0_100"] = (out[score_col].clip(0.0, 1.0) * 100.0).round(4)
    return out


def write_task_file(df: pd.DataFrame, task: int, out_dir: Path) -> Path:
    """THE single place that knows the on-disk submission format."""
    path = out_dir / f"task{task}.jsonl"
    records = []
    for _, row in df.iterrows():
        rec = {
            "doc_id": row["doc_id"],
            "lp": row["lp"],
            "system": row["system"],
        }
        if task == 3:
            rec["error_free"] = int(row["label"])
        elif task == 2:
            rec["score"] = float(row["score_0_100"])
        else:
            raise SystemExit(f"task {task} not supported by this formatter")
        records.append(rec)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def validate(path: Path, expected: pd.DataFrame, task: int) -> list[str]:
    problems: list[str] = []
    try:
        raw = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        return [f"not valid UTF-8: {exc}"]
    lines = [ln for ln in raw.split("\n") if ln.strip()]
    if len(lines) != len(expected):
        problems.append(f"row count {len(lines)} != expected {len(expected)}")
    seen = set()
    for i, ln in enumerate(lines):
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError:
            problems.append(f"line {i + 1}: invalid JSON")
            continue
        key = (rec.get("lp"), rec.get("doc_id"), rec.get("system"))
        if None in key:
            problems.append(f"line {i + 1}: missing id fields")
        if key in seen:
            problems.append(f"line {i + 1}: duplicate id {key}")
        seen.add(key)
        if task == 3:
            if rec.get("error_free") not in (0, 1):
                problems.append(f"line {i + 1}: label {rec.get('error_free')!r} not in {{0,1}}")
        elif task == 2:
            score = rec.get("score")
            if not isinstance(score, (int, float)) or not (0.0 <= float(score) <= 100.0):
                problems.append(f"line {i + 1}: score {score!r} outside [0, 100]")
    expected_keys = set(zip(expected["lp"], expected["doc_id"], expected["system"], strict=True))
    missing = expected_keys - seen
    extra = seen - expected_keys
    if missing:
        problems.append(f"{len(missing)} expected ids missing (e.g. {sorted(missing)[:3]})")
    if extra:
        problems.append(f"{len(extra)} unexpected ids present (e.g. {sorted(extra)[:3]})")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(prog="src.submit")
    parser.add_argument("--run", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", default="chunked")
    parser.add_argument("--agg", default="min")
    parser.add_argument(
        "--thresholds",
        default=None,
        help="thresholds.json from src.calibrate (default: matching calibration dir)",
    )
    parser.add_argument("--task", type=int, default=3, choices=[2, 3])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run = Path(args.run)
    thr_path = Path(
        args.thresholds
        or run / "calibration" / f"{args.model}_{args.mode}_{args.agg}" / "thresholds.json"
    )
    if not thr_path.exists():
        sys.exit(f"thresholds file not found: {thr_path}")
    thresholds = json.loads(thr_path.read_text())

    df = load_scores(run, args.model, args.mode)
    df = apply_thresholds(df, thresholds, args.agg)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_file = write_task_file(df, args.task, out_dir)

    problems = validate(task_file, df, args.task)
    if problems:
        task_file.unlink()
        print("[submit] VALIDATION FAILED — nothing packaged:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    meta = {
        "format_version": FORMAT_VERSION,
        "team": TEAM_NAME,
        "task": args.task,
        "model": args.model,
        "mode": args.mode,
        "agg": args.agg,
        "thresholds": str(thr_path),
        "global_threshold": thresholds["global"]["threshold"],
        "cascade_note": (
            "For Task 2 -> Task 3 auto-cascade, declare the global threshold above "
            "with STRICT '>' semantics on 0-1 scores (or x100 if submitting 0-100)."
        ),
        "n_rows": int(len(df)),
        "label_balance": float(df["label"].mean()) if args.task == 3 else None,
    }
    (out_dir / "submission_meta.json").write_text(json.dumps(meta, indent=2))

    archive = out_dir / f"submission_task{args.task}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(task_file, arcname=task_file.name)
    print(
        f"[submit] validated {len(df):,} rows; label balance "
        f"{meta['label_balance']}; archive -> {archive}"
    )
    print(
        f"[submit] FORMAT IS {FORMAT_VERSION} — re-verify against the official "
        "schema before any real upload (docs/task_facts.md)."
    )


if __name__ == "__main__":
    main()
