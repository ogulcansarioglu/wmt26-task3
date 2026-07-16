"""Shared plumbing: canonical schema, sentence splitting, chunk planning,
run manifests, device resolution, and small IO helpers.

Canonical segment table (one row per scored translation):
    lp           str   language pair as it appears in the data, e.g. "cs-de_DE"
    doc_id       str   source document/segment id from the task data
    system       str   MT system name ("refA" etc. count as systems in dev data)
    segment_id   str   unique row id, f"{doc_id}::{system}" in dev data
    src          str   source text (long, multi-sentence)
    mt           str   translation to be scored
    esa_gold     float gold ESA score 0-100 (dev only; NaN when absent)
    n_errors_total int total annotated error spans across annotators (dev only)
    error_free   int   gold Task 3 label: (esa_mean >= 85) AND no error spans
    error_free_all int stricter variant: every annotation individually error-free
"""

from __future__ import annotations

import dataclasses
import json
import math
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ID_COLUMNS = ["lp", "doc_id", "system", "segment_id"]
TEXT_COLUMNS = ["src", "mt"]
GOLD_COLUMNS = ["esa_gold", "n_errors_total", "error_free", "error_free_all"]

# Documented WMT26 Task 3 gold rule: ESA >= 85 and no annotated error spans.
ERROR_FREE_ESA_THRESHOLD = 85.0

# XLM-R based COMET models encode (mt, src) jointly with a 512-token window.
MAX_JOINT_TOKENS = 512
DEFAULT_CHUNK_BUDGET = 220  # per-side token budget, leaves headroom for specials
DEFAULT_SEED = 13


# ---------------------------------------------------------------------------
# sentence splitting / chunking
# ---------------------------------------------------------------------------

# Sentence-final punctuation incl. CJK, Arabic question mark, ellipsis.
_SENT_RE = re.compile(r'[^.!?。！？؟…]+(?:[.!?。！？؟…]+["»”’\')\]]*\s*|\s*$)')


def split_sentences(text: str) -> list[str]:
    """Newline-first, punctuation-second splitter.

    Oversplitting (e.g. on abbreviations) is harmless here — sentences are
    regrouped into chunks anyway. Undersplitting is what we bias against.
    """
    sents: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for match in _SENT_RE.finditer(line):
            piece = match.group().strip()
            if piece:
                sents.append(piece)
    if not sents:
        return [text.strip() or ""]
    return sents


def partition_by_mass(items: list[str], weights: list[int], n: int) -> list[list[str]]:
    """Order-preserving partition of items into exactly n contiguous,
    non-empty groups with roughly equal total weight. Requires n <= len(items).
    """
    if n <= 1:
        return [list(items)]
    if n > len(items):
        raise ValueError(f"cannot split {len(items)} items into {n} groups")
    total = float(sum(weights)) or 1.0
    target = total / n
    groups: list[list[str]] = [[] for _ in range(n)]
    cum = 0.0
    for item, weight in zip(items, weights, strict=True):
        mid = cum + weight / 2.0
        idx = min(n - 1, int(mid / target))
        cum += weight
        groups[idx].append(item)
    # A very heavy item can leave gaps; borrow the first item of the next
    # non-empty group (order is preserved: borrowed items precede the donor's).
    for i in range(n):
        if not groups[i]:
            for j in range(i + 1, n):
                if len(groups[j]) > 1 or (groups[j] and all(not g for g in groups[i + 1 : j])):
                    groups[i].append(groups[j].pop(0))
                    break
            else:
                for j in range(i - 1, -1, -1):
                    if len(groups[j]) > 1:
                        groups[i].insert(0, groups[j].pop())
                        break
    if any(not g for g in groups):
        raise AssertionError("partition produced an empty group")
    return groups


@dataclasses.dataclass
class ChunkPlan:
    src_chunks: list[str]
    mt_chunks: list[str]
    joint_tokens: int  # tokens of the un-chunked (mt + src) encoding
    over_limit: bool  # would have been truncated without chunking
    oversized_chunk: bool  # some chunk still exceeds budget (unsplittable)

    @property
    def n_chunks(self) -> int:
        return len(self.src_chunks)


def make_chunk_plan(src: str, mt: str, tok_len, budget: int = DEFAULT_CHUNK_BUDGET) -> ChunkPlan:
    """Plan aligned (src, mt) chunks so each side stays under `budget` tokens.

    Both sides are split into the SAME number of contiguous chunks, aligned by
    relative token mass — chunk k of the source is scored against chunk k of
    the translation. When one side has too few sentences to split further, the
    chunk count is clamped and the oversized chunk is flagged (the model will
    truncate that chunk; this is logged, not hidden).
    """
    src_len, mt_len = tok_len(src), tok_len(mt)
    joint = src_len + mt_len + 4  # specials margin for the joint encoding
    if joint <= MAX_JOINT_TOKENS:
        return ChunkPlan([src], [mt], joint, False, False)

    src_sents = split_sentences(src)
    mt_sents = split_sentences(mt)
    src_lens = [max(1, tok_len(s)) for s in src_sents]
    mt_lens = [max(1, tok_len(s)) for s in mt_sents]

    need = max(
        1,
        math.ceil(sum(src_lens) / budget),
        math.ceil(sum(mt_lens) / budget),
    )
    n = max(1, min(need, len(src_sents), len(mt_sents)))

    src_groups = partition_by_mass(src_sents, src_lens, n)
    mt_groups = partition_by_mass(mt_sents, mt_lens, n)
    src_chunks = [" ".join(g) for g in src_groups]
    mt_chunks = [" ".join(g) for g in mt_groups]

    oversized = any(tok_len(c) > MAX_JOINT_TOKENS - 8 for c in src_chunks + mt_chunks) or (n < need)
    return ChunkPlan(src_chunks, mt_chunks, joint, True, oversized)


def aggregate_chunk_scores(scores: list[float], weights: list[int]) -> dict[str, float]:
    """Task 3 wants min (one error anywhere breaks error-freeness); Task 2
    compares mean vs min on dev. All three aggregates are persisted."""
    total = float(sum(weights)) or 1.0
    return {
        "score_min": min(scores),
        "score_mean": sum(scores) / len(scores),
        "score_wmean": sum(s * w for s, w in zip(scores, weights, strict=True)) / total,
    }


# ---------------------------------------------------------------------------
# runs, manifests, determinism
# ---------------------------------------------------------------------------


def set_seeds(seed: int = DEFAULT_SEED) -> None:
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def resolve_device(pref: str = "auto") -> str:
    """MPS is required on this project's target machine — fail loudly rather
    than silently falling back to a CPU run that takes 20x longer."""
    if pref == "cpu":
        return "cpu"
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if pref in ("auto", "mps"):
        sys.exit(
            "FATAL: torch.backends.mps.is_available() is False and --device cpu "
            "was not explicitly requested. Refusing to start a silent CPU run."
        )
    return "cpu"


def git_commit() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).resolve().parent,
                timeout=5,
            ).stdout.strip()
            or None
        )
    except Exception:
        return None


def environment_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": git_commit(),
    }
    packages = {}
    from importlib import metadata

    for pkg in ("unbabel-comet", "torch", "transformers", "pandas", "scikit-learn", "pyarrow"):
        try:
            packages[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            packages[pkg] = None
    info["packages"] = packages
    return info


class RunManifest:
    """Append-friendly manifest at runs/<run>/manifest.json. Every scoring run
    records environment, model provenance, and per-LP wall clock, so any number
    in the paper can be traced to a run."""

    def __init__(self, run_dir: Path):
        self.path = Path(run_dir) / "manifest.json"
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "environment": environment_info(),
                "stages": {},
            }

    def record(self, stage: str, key: str, payload: dict) -> None:
        self.data.setdefault("stages", {}).setdefault(stage, {})[key] = payload
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2, default=str))
        tmp.replace(self.path)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_segments(path: Path):
    """Load a canonical segment table from .jsonl or .parquet into a DataFrame."""
    import pandas as pd

    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".jsonl":
        df = pd.DataFrame(list(read_jsonl(path)))
    else:
        raise ValueError(f"unsupported input format: {path}")
    missing = [c for c in ID_COLUMNS + TEXT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing canonical columns: {missing}")
    if df["segment_id"].duplicated().any():
        raise ValueError(f"{path} has duplicated segment_id values")
    return df
