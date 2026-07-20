"""Submission formatter + validator + packager.

FORMAT STATUS (2026-07-18): the unified WMT26 schema is now published on the
Task 2 page (updated 2026-07-18):

  input record:   {"item_id": "srcLang_###_tgtLang_###_domain_###_docID_###_segID",
                   "src": ..., "ref": {"text":..., "type": "human|postedit|pseudo"},
                   "hyps": {systemName: translation, ...}, "resources": {...}}
  task2 output:   {"item_id": ..., "task2_pred": {systemName: score, ...}}

Task 3 direct-submission output is NOT yet published (its page predates the
schema drop); we emit the obvious unified-format mirror

  task3 output:   {"item_id": ..., "task3_pred": {systemName: 0|1, ...}}

**marked INFERRED — confirm against the Task 3 page / Codabench before the
real upload.** ALL format-specific logic lives in `write_task_file()` and
`validate()` — update those two and bump FORMAT_VERSION on any change.

The Task 2 -> Task 3 cascade threshold is declared in the Codabench UI at
submission time (value + strict '>' vs inclusive '≥'). We submit Task 2 scores
on the 0-100 scale, so the declared threshold must be calibrated_threshold*100.

The validator re-reads what was written and checks item counts, (item, system)
alignment against the scored inputs, label/score domains, and UTF-8 before
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

FORMAT_VERSION = "official-2026-07-18+task3-inferred"

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


def _item_key_column(df: pd.DataFrame) -> str:
    # Real test data carries the official item_id; dev tables predate it and
    # use doc_id as the item key (one record per segment there too).
    return "item_id" if "item_id" in df.columns else "doc_id"


def write_task_file(df: pd.DataFrame, task: int, out_dir: Path) -> Path:
    """THE single place that knows the on-disk submission format."""
    if task not in (2, 3):
        raise SystemExit(f"task {task} not supported by this formatter")
    key_col = _item_key_column(df)
    pred_field = f"task{task}_pred"
    value_col = "label" if task == 3 else "score_0_100"
    path = out_dir / f"task{task}.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for item_id, group in df.groupby(key_col, sort=True):
            preds = {
                str(row["system"]): (int(row[value_col]) if task == 3 else float(row[value_col]))
                for _, row in group.iterrows()
            }
            rec = {"item_id": str(item_id), pred_field: preds}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def validate(path: Path, expected: pd.DataFrame, task: int) -> list[str]:
    problems: list[str] = []
    try:
        raw = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        return [f"not valid UTF-8: {exc}"]
    pred_field = f"task{task}_pred"
    key_col = _item_key_column(expected)
    expected_pairs = set(
        zip(expected[key_col].astype(str), expected["system"].astype(str), strict=True)
    )
    expected_items = {k for k, _ in expected_pairs}

    lines = [ln for ln in raw.split("\n") if ln.strip()]
    if len(lines) != len(expected_items):
        problems.append(f"item count {len(lines)} != expected {len(expected_items)}")
    seen_pairs: set[tuple[str, str]] = set()
    seen_items: set[str] = set()
    for i, ln in enumerate(lines):
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError:
            problems.append(f"line {i + 1}: invalid JSON")
            continue
        item = rec.get("item_id")
        preds = rec.get(pred_field)
        if not isinstance(item, str) or not isinstance(preds, dict) or not preds:
            problems.append(f"line {i + 1}: needs item_id + non-empty {pred_field} dict")
            continue
        if item in seen_items:
            problems.append(f"line {i + 1}: duplicate item_id {item}")
        seen_items.add(item)
        for system, value in preds.items():
            if task == 3:
                if isinstance(value, bool) or value not in (0, 1):
                    problems.append(f"line {i + 1}: {system}: label {value!r} not in {{0,1}}")
            else:
                ok = isinstance(value, (int, float)) and not isinstance(value, bool)
                if not ok or not (0.0 <= float(value) <= 100.0):
                    problems.append(f"line {i + 1}: {system}: score {value!r} outside [0, 100]")
            seen_pairs.add((item, str(system)))
    missing = expected_pairs - seen_pairs
    extra = seen_pairs - expected_pairs
    if missing:
        problems.append(
            f"{len(missing)} expected (item, system) pairs missing (e.g. {sorted(missing)[:3]})"
        )
    if extra:
        problems.append(
            f"{len(extra)} unexpected (item, system) pairs present (e.g. {sorted(extra)[:3]})"
        )
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

    g_thr = float(thresholds["global"]["threshold"])
    meta = {
        "format_version": FORMAT_VERSION,
        "team": TEAM_NAME,
        "task": args.task,
        "model": args.model,
        "mode": args.mode,
        "agg": args.agg,
        "thresholds": str(thr_path),
        "global_threshold_raw": g_thr,
        "cascade_note": (
            f"Task 2 cascade: declare threshold {g_thr * 100:.2f} on the submitted 0-100 "
            "scale with STRICT '>' semantics (matches apply_thresholds)."
        ),
        "n_rows": int(len(df)),
        "n_items": int(df[_item_key_column(df)].nunique()),
        "label_balance": float(df["label"].mean()) if args.task == 3 else None,
    }
    (out_dir / "submission_meta.json").write_text(json.dumps(meta, indent=2))

    archive = out_dir / f"submission_task{args.task}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(task_file, arcname=task_file.name)
    print(
        f"[submit] validated {meta['n_items']:,} items / {len(df):,} predictions; "
        f"label balance {meta['label_balance']}; archive -> {archive}"
    )
    if args.task == 3:
        print(
            "[submit] REMINDER: task3_pred field name is INFERRED from the Task 2 "
            "schema — confirm on Codabench/Task 3 page before real upload."
        )


if __name__ == "__main__":
    main()
