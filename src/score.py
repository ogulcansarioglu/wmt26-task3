"""Model inference with chunked long-segment handling and per-LP checkpointing.

Usage:
    python -m src.score --model mock       --input data/smoke/smoke_dev.jsonl --run runs/smoke --device cpu
    python -m src.score --model cometkiwi  --input data/dev/wmt25_esa_dev.parquet --run runs/dev
    python -m src.score --model xcomet-xl  --input ... --run runs/dev --batch-size 4

Scores are written to  <run>/scores/<model>/<mode>/<lp>.parquet  after every
language pair; re-running skips completed pairs, so a crash at pair 18/21
costs minutes, not a night.

Modes:
    chunked   (default) over-length segments are split into aligned chunks,
              scored per chunk, aggregated as min/mean/weighted-mean
    truncate  full segments handed to the model as-is (silent model-side
              truncation) — the ablation baseline
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Must be set before torch initializes MPS; harmless otherwise.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.common import (
    DEFAULT_CHUNK_BUDGET,
    DEFAULT_SEED,
    MAX_JOINT_TOKENS,
    RunManifest,
    aggregate_chunk_scores,
    load_segments,
    make_chunk_plan,
    resolve_device,
    set_seeds,
)

MODEL_IDS = {
    "cometkiwi": "Unbabel/wmt22-cometkiwi-da",
    "xcomet-xl": "Unbabel/XCOMET-XL",
    "xcomet-xxl": "Unbabel/XCOMET-XXL",
}


class MockScorer:
    """Deterministic heuristic scorer for pipeline smoke tests. Keys on the
    synthetic corruption markers produced by `src.data make-smoke`; no torch,
    no network, no model weights."""

    name = "mock"
    provenance = {"model_id": "mock", "revision": None}

    def tok_len(self, text: str) -> int:
        return len(text.split())

    def score_pairs(self, pairs: list[dict]) -> tuple[list[float], list]:
        scores = []
        for p in pairs:
            mt_words = p["mt"].replace(".", " ").split()
            if not mt_words:
                scores.append(0.0)
                continue
            bad = sum(1 for w in mt_words if w == "ERRTOKEN")
            src_digits = {w for w in p["src"].replace(".", " ").split() if w.isdigit()}
            mt_digits = {w for w in mt_words if w.isdigit()}
            digit_miss = len(mt_digits - src_digits) + len(src_digits - mt_digits)
            translated = sum(1 for w in mt_words if w.endswith("-tr") or w.isdigit())
            coverage = translated / len(mt_words)
            score = max(0.0, min(1.0, 0.55 + 0.45 * coverage - 0.18 * bad - 0.12 * digit_miss))
            scores.append(score)
        return scores, [None] * len(pairs)


class CometScorer:
    """Wraps unbabel-comet checkpoints (CometKiwi / xCOMET-*). Reference-free:
    only src+mt are passed even when references exist."""

    def __init__(self, key: str, device: str, batch_size: int):
        from comet import download_model, load_from_checkpoint

        self.name = key
        self.model_id = MODEL_IDS[key]
        self.device = device
        self.batch_size = batch_size
        print(f"[score] loading {self.model_id} (gated — needs `hf auth login` once)")
        ckpt_path = download_model(self.model_id)
        self.model = load_from_checkpoint(ckpt_path)
        self.model.eval()
        self.tokenizer = self.model.encoder.tokenizer
        self.provenance = {
            "model_id": self.model_id,
            "checkpoint": str(ckpt_path),
            "revision": _hf_revision_from_path(ckpt_path),
        }
        self._predict_kwargs: dict | None = None

    def tok_len(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def score_pairs(self, pairs: list[dict]) -> tuple[list[float], list]:
        data = [{"src": p["src"], "mt": p["mt"]} for p in pairs]
        out = self._predict(data)
        spans = getattr(getattr(out, "metadata", None), "error_spans", None)
        return list(out.scores), list(spans) if spans else [None] * len(data)

    def _predict(self, data: list[dict]):
        """comet's predict() signature and MPS support vary across versions;
        try the preferred config once, remember what worked."""
        if self._predict_kwargs is not None:
            return self.model.predict(data, **self._predict_kwargs)
        candidates = []
        if self.device == "mps":
            candidates.append(
                dict(batch_size=self.batch_size, gpus=1, accelerator="mps", num_workers=0)
            )
            candidates.append(dict(batch_size=self.batch_size, gpus=1, accelerator="mps"))
        candidates.append(
            dict(batch_size=self.batch_size, gpus=0, accelerator="cpu", num_workers=0)
        )
        candidates.append(dict(batch_size=self.batch_size, gpus=0))
        last_error: Exception | None = None
        for kwargs in candidates:
            try:
                result = self.model.predict(data, **kwargs)
                self._predict_kwargs = kwargs
                print(f"[score] predict() config locked in: {kwargs}")
                return result
            except (TypeError, ValueError, RuntimeError) as exc:  # signature or backend mismatch
                last_error = exc
                print(
                    f"[score] predict config {kwargs} failed ({type(exc).__name__}: {exc}); trying next"
                )
        raise RuntimeError(f"all predict() configs failed; last error: {last_error}")


def _hf_revision_from_path(ckpt_path) -> str | None:
    # HF cache layout: .../models--Unbabel--X/snapshots/<commit>/...
    parts = Path(ckpt_path).parts
    try:
        return parts[parts.index("snapshots") + 1]
    except (ValueError, IndexError):
        return None


def build_scorer(model_key: str, device: str, batch_size: int):
    if model_key == "mock":
        return MockScorer()
    return CometScorer(model_key, device, batch_size)


def score_lp(df_lp, scorer, mode: str, budget: int) -> tuple[list[dict], dict]:
    """Score one language pair. Returns (rows, stats)."""
    rows: list[dict] = []
    pair_index: list[tuple[int, int]] = []  # (row_idx, chunk_idx)
    flat_pairs: list[dict] = []
    plans = []

    records = df_lp.to_dict("records")
    for i, rec in enumerate(records):
        if mode == "chunked":
            plan = make_chunk_plan(rec["src"], rec["mt"], scorer.tok_len, budget)
        else:
            joint = scorer.tok_len(rec["src"]) + scorer.tok_len(rec["mt"]) + 4
            plan = None
            rec["_joint"] = joint
        if plan is not None:
            plans.append(plan)
            for k, (s, m) in enumerate(zip(plan.src_chunks, plan.mt_chunks, strict=True)):
                flat_pairs.append({"src": s, "mt": m})
                pair_index.append((i, k))
        else:
            flat_pairs.append({"src": rec["src"], "mt": rec["mt"]})
            pair_index.append((i, 0))
            plans.append(None)

    scores, spans = scorer.score_pairs(flat_pairs)

    by_row: dict[int, list[tuple[int, float, object]]] = {}
    for (row_idx, chunk_idx), score, span in zip(pair_index, scores, spans, strict=True):
        by_row.setdefault(row_idx, []).append((chunk_idx, float(score), span))

    n_over, n_oversized = 0, 0
    for i, rec in enumerate(records):
        chunk_results = sorted(by_row[i])
        chunk_scores = [s for _, s, _ in chunk_results]
        chunk_spans = [sp for _, _, sp in chunk_results if sp is not None]
        row = {k: rec[k] for k in rec if not k.startswith("_")}
        row.pop("src", None)
        row.pop("mt", None)
        plan = plans[i]
        if plan is not None:
            weights = [max(1, scorer.tok_len(c)) for c in plan.mt_chunks]
            row.update(aggregate_chunk_scores(chunk_scores, weights))
            row.update(
                n_chunks=plan.n_chunks,
                joint_tok_len=plan.joint_tokens,
                over_limit=plan.over_limit,
                oversized_chunk=plan.oversized_chunk,
            )
            n_over += int(plan.over_limit)
            n_oversized += int(plan.oversized_chunk)
        else:
            joint = rec["_joint"]
            row.update(
                score_min=chunk_scores[0],
                score_mean=chunk_scores[0],
                score_wmean=chunk_scores[0],
                n_chunks=1,
                joint_tok_len=joint,
                over_limit=joint > MAX_JOINT_TOKENS,
                oversized_chunk=False,
            )
            n_over += int(joint > MAX_JOINT_TOKENS)
        row["chunk_scores"] = json.dumps(chunk_scores)
        row["error_spans"] = json.dumps(chunk_spans) if chunk_spans else None
        rows.append(row)

    stats = {
        "n_segments": len(records),
        "n_over_limit": n_over,
        "over_limit_rate": round(n_over / max(1, len(records)), 4),
        "n_oversized_chunk": n_oversized,
    }
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(prog="src.score")
    parser.add_argument("--model", required=True, choices=["mock", *MODEL_IDS])
    parser.add_argument("--input", required=True)
    parser.add_argument("--run", required=True, help="run directory, e.g. runs/dev")
    parser.add_argument("--mode", default="chunked", choices=["chunked", "truncate"])
    parser.add_argument("--budget", type=int, default=DEFAULT_CHUNK_BUDGET)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--lps", default=None, help="comma-separated subset of language pairs")
    parser.add_argument(
        "--sample-per-lp",
        type=int,
        default=None,
        help="stratified random cap per LP (dev-time speedup)",
    )
    parser.add_argument("--limit", type=int, default=None, help="first N rows overall (smoke)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import pandas as pd

    set_seeds(args.seed)
    device = "cpu" if args.model == "mock" else resolve_device(args.device)

    df = load_segments(Path(args.input))
    if args.lps:
        keep = {lp.strip() for lp in args.lps.split(",")}
        df = df[df["lp"].isin(keep)]
    if args.limit:
        df = df.head(args.limit)
    if args.sample_per_lp:
        parts = [
            g.sample(min(len(g), args.sample_per_lp), random_state=args.seed)
            for _, g in df.groupby("lp")
        ]
        df = pd.concat(parts).reset_index(drop=True)

    run_dir = Path(args.run)
    out_dir = run_dir / "scores" / args.model / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = RunManifest(run_dir)

    scorer = build_scorer(args.model, device, args.batch_size)
    manifest.record(
        "score",
        f"{args.model}/{args.mode}/_config",
        {
            "input": str(args.input),
            "device": device,
            "batch_size": args.batch_size,
            "budget": args.budget,
            "seed": args.seed,
            "sample_per_lp": args.sample_per_lp,
            "provenance": scorer.provenance,
        },
    )

    lps = sorted(df["lp"].unique())
    print(f"[score] {len(df):,} segments over {len(lps)} LPs -> {out_dir}")
    for n, lp in enumerate(lps, 1):
        out_path = out_dir / f"{lp}.parquet"
        df_lp = df[df["lp"] == lp]
        if out_path.exists() and not args.overwrite:
            existing = pd.read_parquet(out_path)
            if len(existing) == len(df_lp):
                print(
                    f"[score] ({n}/{len(lps)}) {lp}: checkpoint exists ({len(existing)} rows), skipping"
                )
                continue
            print(
                f"[score] ({n}/{len(lps)}) {lp}: stale checkpoint ({len(existing)} != {len(df_lp)}), rescoring"
            )
        t0 = time.time()
        rows, stats = score_lp(df_lp, scorer, args.mode, args.budget)
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        wall = round(time.time() - t0, 2)
        stats["wall_clock_s"] = wall
        manifest.record("score", f"{args.model}/{args.mode}/{lp}", stats)
        print(
            f"[score] ({n}/{len(lps)}) {lp}: {stats['n_segments']} segments, "
            f"over-limit {stats['over_limit_rate']:.1%}, {wall}s"
        )
    print("[score] all language pairs complete")


if __name__ == "__main__":
    main()
