# wmt26-task3

System for the **WMT 2026 Shared Task on Automated Translation Quality
Evaluation Systems — Task 3: Detection of Error-Free Segments** (binary:
1 = error-free, 0 = contains errors; ranked by Matthews Correlation
Coefficient), with Task 2 (segment-level ESA score prediction) as a by-product
via the organizers' auto-cascade.

Everything runs locally on an Apple-Silicon Mac (PyTorch MPS). Reference-free
throughout: only (source, translation) pairs are ever scored. No paid APIs.

- Task facts, formats, and open questions: [docs/task_facts.md](docs/task_facts.md)
- Deadlines: test data **23 July 2026**, submission **30 July 2026** (AoE)

## Architecture

| Stage | Model | Role |
|-------|-------|------|
| A | `Unbabel/wmt22-cometkiwi-da` (~0.5B) | baseline + banked fallback submission |
| B | `Unbabel/XCOMET-XL` (~3.5B) | primary system (also emits error spans) |
| C | logistic-regression ensemble | optional, only if A+B calibrated early |

Segments this year are **long multi-sentence units**; encoder QE models
silently truncate at 512 tokens. `src/score.py` therefore plans **aligned
chunks** (sentence-boundary based, equal token mass per side), scores each
chunk, and aggregates (`min` for Task 3 — one error anywhere breaks
error-freeness; `min` vs `mean` compared on dev for Task 2). Truncation
exposure is measured and reported per language pair; `--mode truncate` runs
the ablation baseline.

Binary labels come from per-language-pair MCC-optimal thresholds with a global
fallback for low-resource pairs and degeneracy guards (`src/calibrate.py` —
protocol in the module docstring).

## Setup

```bash
make venv                     # python3.12 venv + requirements
source .venv/bin/activate     # optional; make targets use .venv directly
make smoke                    # full pipeline on synthetic data, <2 min, no model
make test                     # unit + e2e tests
```

One-time (owner, personal accounts — see docs/task_facts.md §escalations):

1. Hugging Face: accept licenses on the model pages of
   [wmt22-cometkiwi-da](https://huggingface.co/Unbabel/wmt22-cometkiwi-da) and
   [XCOMET-XL](https://huggingface.co/Unbabel/XCOMET-XL), then `hf auth login`.
   Heads-up: without login, comet raises a misleading
   `KeyError: "Model ... not supported by COMET."` — the underlying error is
   `GatedRepoError: 401` (verified 2026-07-16). Log in and it goes away.
2. Codabench account + competition registration (links appear on the
   [task page](https://www2.statmt.org/wmt26/mteval-task.html) before 23 July).
3. Join the [wmt-tasks group](http://groups.google.com/group/wmt-tasks).

## Runbook

```bash
# 1. data
make download-dev             # WMT25 sources + ESA humeval (143 MB)
make build-dev                # -> data/dev/wmt25_esa_dev.parquet (+ per-LP table)

# 2. first GPU sanity check (needs HF login)
make smoke-model              # 10 real segments through CometKiwi on MPS

# 3. Stage A on full dev (checkpointed per LP; resumable; hours)
make score-dev-baseline

# 4. calibrate + inspect
make calibrate-baseline
.venv/bin/python -m src.analyze truncation --run runs/dev --model cometkiwi
.venv/bin/python -m src.analyze aggs --run runs/dev --model cometkiwi

# 5. Stage B (overnight)
make score-dev-primary
.venv/bin/python -m src.calibrate --run runs/dev --model xcomet-xl --mode chunked --agg min

# 6. dress rehearsal: package dev as if it were test
make rehearsal
```

Long runs: always prefix with `caffeinate -is` (the `score-dev-*` targets
already do) so the Mac doesn't sleep mid-inference. Every scoring run writes
per-LP parquet checkpoints under `runs/<run>/scores/<model>/<mode>/` and
appends provenance (package versions, HF model revision, device, wall clock
per LP) to `runs/<run>/manifest.json`. Re-running skips completed pairs.

## Submission

`src/submit.py` applies thresholds, writes the task file, **validates**
(row counts, ID alignment, label domain, UTF-8) and zips. The on-disk format
is `draft-2026-07-16` until the organizers publish the real schema (~17 July);
the only place that knows the format is `write_task_file()` — update it, rerun,
re-validate. **Never upload without re-checking docs/task_facts.md against the
task pages.**

Plan for the submission window: submit the Stage A baseline as a pipeline test
on 23 July, primary xCOMET-XL system by 28 July, buffer until 30 July.

## Repository layout

```
docs/task_facts.md     Day-1 recon findings (authoritative over any plan)
docs/analysis_notes.md analysis artifacts -> system-description paper
src/common.py          canonical schema, chunk planner, manifests
src/data.py            download / build-dev / stats / make-smoke
src/score.py           model inference, chunking, per-LP checkpointing
src/calibrate.py       MCC threshold sweeps, per-LP vs global tables
src/submit.py          formatter + validator + packager
src/analyze.py         truncation/aggregation/distribution/comparison reports
tests/                 unit tests + end-to-end smoke
data/, runs/, submissions/   gitignored artifacts
```

## Licensing

Code is MIT. Model weights (CometKiwi, xCOMET) are gated on Hugging Face under
CC-BY-NC-SA — used for scoring only, never redistributed. Task data belongs to
the WMT organizers and is never committed.
