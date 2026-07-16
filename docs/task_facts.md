# WMT 2026 Task Facts — Day-1 Recon (2026-07-16)

Primary sources consulted on 2026-07-16:

- https://www2.statmt.org/wmt26/ (conference page)
- https://www2.statmt.org/wmt26/mteval-task.html (task overview)
- https://www2.statmt.org/wmt26/mteval-subtask2.html (Task 2 detail)
- https://www2.statmt.org/wmt26/mteval-subtask3.html (Task 3 detail)
- https://www2.statmt.org/wmt25/mteval-subtask.html (last year's edition, for context)
- https://github.com/wmt-conference/wmt25-general-mt (dev data, inspected via API + raw files)

**Rule: if anything below contradicts the project spec, this file wins.** Re-verify
the task pages around **17–20 July** (exact schema announcement) and again on
**23 July** (test data release).

## Subtasks (2026 numbering — differs from 2025!)

| Task | Predicts | Metric |
|------|----------|--------|
| 1 | Error spans (start/end indices) + severity (major/minor) | span F-style (TBD) |
| 2 | Segment-level ESA quality score, **0–100**, higher = better | correlation w/ human ESA, segment + corpus level |
| 3 | **Binary: 1 = error-free, 0 = contains errors** | **MCC** (primary); precision/recall secondary |
| 4 | Challenge sets (not our task) | — |

2025's "Task 3" was error *correction* — do not reuse 2025 Task-3 artifacts/formats.

## Key dates (AoE)

- **23 July 2026** — test data released, Codabench submission opens
- **30 July 2026** — Tasks 1/2/3 submission deadline
- ~**17 July 2026** — exact test-set JSON schema announced ("approximately one week prior to release")

## Language pairs (23 total)

cs→de_DE, cs→uk_UA, cs→vi_VN, zh_CN→ja_JP, en→ar_EG, en→hy_AM, en→be_BY,
en→zh_CN, en→zh_TW, en→cs_CZ, en→et_EE, en→de_DE, en→is_IS, en→id_ID,
en→ja_JP, en→kk_KZ, en→ko_KR, en→lld_IT (Ladin), en→lij_IT (Ligurian),
en→ru_RU, en→sme_NO (Northern Sámi), en→th_TH, en→uk_UA

**High-priority subset (8):** cs→de, cs→uk, zh→ja, en→ar_EG, en→et, en→is, en→id, en→sme.

**Partial coverage: explicitly permitted.** Organizers "strongly encourage" all 23
but say to prioritize the 8 above if compute-limited. Not stated how partial
coverage affects official ranking tables — *residual question for organizers if
we end up dropping pairs; with 128 GB unified memory we target all 23.*

New/low-resource pairs with little or no prior human-eval data (get global-threshold
fallback + special analysis): en→lld, en→lij, en→sme, en→kk, en→hy, en→be, cs→vi,
zh→ja, en→ar_EG, en→th, en→zh_TW.

## Data properties

- Segments are **long multi-sentence units** ("similar to last year"), evaluated in document context. Encoder QE models WILL truncate without countermeasures → chunked scoring is required.
- **References optional** — "provided as optional input for some but likely not all language pairs". Design fully reference-free (QE setting). Confirmed correct.

## Formats

- Test input: "unified JSON data format" across Tasks 1–3, following the
  **JSON-lines format of the General MT shared task**. Exact schema announced
  ~1 week before 23 July. **ACTION: re-fetch subtask pages ~17–20 July.**
- General-MT JSONL reference schema (from `wmt25-genmt.jsonl`, inspected 2026-07-16):
  `dataset_id, collection_id, doc_id, domain, src_lang, tgt_lang, src_text,
  video, screenshot, prompt_instruction, refs{refA{ref}}`
- Submission output format + packaging: **not yet published**. Codabench links:
  **not yet published** ("check here later for the link"). **ACTION: watch pages;
  register on Codabench as soon as links appear.**

## Task 3 specifics

- Gold label derivation (documented by organizers): a segment is error-free iff
  it **meets a score threshold (≥ 85, ESA-like)** AND **contains no annotated
  error spans**. Organizers caution the threshold "is sensitive to different
  annotation and score calibration schema."
- Baselines we must beat: thresholded reference-free COMET-QE, Bicleaner,
  Always-Negative (all-0).
- No task-specific train/dev data is provided ("since this is a new task").
  Recommended: use Task 2 resources (below).

## Task 2 → Task 3 cascade (mechanics confirmed)

- Direct Task 2 submissions are auto-evaluated on Task 3 **using a
  participant-supplied custom threshold** (participant chooses value and
  strict `>` vs inclusive `≥` semantics). Opt-out possible.
- **Direct Task 3 submissions (binary labels) are also accepted** — this is the
  only path that supports *per-language-pair* thresholds, so our primary Task 3
  submission is direct; the Task 2 cascade (single global threshold) comes free.
- **Submission limit: 2 system variants per organization** (stated on Task 2
  page; assume same for Task 3). Plan: variant 1 = xCOMET-XL calibrated,
  variant 2 = CometKiwi baseline.
- Unknown: whether Codabench allows repeated/overwritten submission attempts
  during the window (needed for the 23-July dry-run plan). **ACTION: check on
  Codabench once competition pages exist; else email organizers.**

## Dev/training data (chosen: WMT25 General-MT ESA human eval)

Task 2 page lists: ESA annotations from WMT25 General MT (github.com/wmt-conference/wmt25-general-mt),
DA/MQM from QE tasks 2022–24 (github.com/WMT-QE-Task/), MQM 2020–24
(github.com/google/wmt-mqm-human-evaluation), DA 2016–22.

We use **WMT25 General-MT ESA annotations** as the dev set — same long-segment
regime as this year's test data, same annotation scheme (ESA + error spans),
and it directly supports the documented Task 3 gold rule.

- `data/wmt25-genmt.jsonl` — sources/metadata, 14.5 MB, plain file:
  `https://raw.githubusercontent.com/wmt-conference/wmt25-general-mt/main/data/wmt25-genmt.jsonl`
- `data/wmt25-genmt-humeval.jsonl` — **the dev corpus**, 128.7 MB, Git-LFS:
  `https://media.githubusercontent.com/media/wmt-conference/wmt25-general-mt/main/data/wmt25-genmt-humeval.jsonl`
- Humeval schema (inspected 2026-07-16): one record per segment:
  `doc_id` (encodes lp + domain: `cs-de_DE_#_news_#_<origin>_#_<seg>`),
  `src_text`, `tgt_text` (dict: system name → MT output, incl. `refA`),
  `scores` (dict: system name → list of annotations, each
  `{score: 0–100, annotator, times, errors: [{start_i, end_i, severity}]}`).
  Multiple annotations per system output occur.
- Our gold-label rule for dev (mirrors organizers'): aggregate annotations per
  (doc, system): `esa_mean` = mean of scores; `n_errors_total` = total error
  spans across annotations; primary label `error_free = (esa_mean >= 85) AND
  (n_errors_total == 0)`. Stricter variant `error_free_all` (every annotation
  individually error-free) kept for sensitivity analysis.
- Coverage note: WMT25 humeval covers WMT25's language pairs — the 2026-new
  pairs (Ladin, Ligurian, etc.) have **no dev data** → global-threshold
  fallback per calibration protocol.

### Dev set as built (2026-07-16, `python -m src.data build-dev`)

**54,419 rows, 14 LPs, global error-free rate 0.134** (imbalanced — MCC
calibration is where the points are). Per-LP:

| lp | n | error_free_rate | esa_mean |
|---|---|---|---|
| cs-de_DE | 4851 | 0.093 | 81.9 |
| cs-uk_UA | 4275 | 0.265 | 87.1 |
| en-cs_CZ | 4240 | 0.136 | 76.9 |
| en-sr_Cyrl_RS | 3914 | 0.063 | 77.2 |
| en-it_IT | 3870 | 0.117 | 67.9 |
| en-is_IS | 3838 | 0.033 | 44.8 |
| en-ru_RU | 3800 | 0.131 | 69.4 |
| en-ar_EG | 3781 | 0.041 | 30.3 |
| en-uk_UA | 3781 | 0.390 | 85.9 |
| en-zh_CN | 3762 | 0.237 | 82.0 |
| en-et_EE | 3610 | 0.039 | 57.7 |
| en-bho_IN | 3591 | 0.062 | 71.0 |
| en-ja_JP | 3591 | 0.250 | 79.5 |
| en-mas_KE | 3515 | 0.000 | 3.7 |

Notes: en-mas_KE has ZERO error-free rows (all MT garbage; ESA mean 3.7) —
exercises the degenerate-LP guard, and is a candidate to exclude from global
threshold fitting (decide during calibration). **10 of the 23 WMT26 pairs have
direct dev coverage** (cs-de, cs-uk, en-cs, en-is, en-ru, en-ar_EG, en-uk,
en-zh_CN, en-et, en-ja); the other 13 (incl. en-de!) ride the global
threshold — per-LP-vs-global transfer is a core paper analysis.

## Hardware / feasibility (this machine)

Apple M4 Max, **128 GB unified memory**, 278 GB free disk (2026-07-16).
- CometKiwi (`Unbabel/wmt22-cometkiwi-da`, ~0.5 B): trivially fits.
- xCOMET-XL (~3.5 B): fits comfortably, even fp32.
- xCOMET-XXL (~10.7 B): plausible stretch goal, fp16 — only if XL is done early.
- Both Unbabel models are **HF-gated** (license acceptance + `hf auth login`
  required — owner action) and CC-BY-NC-SA licensed (fine for shared-task use;
  weights never redistributed; code stays MIT).

## System description paper (recon 2026-07-16, conference page)

- **Deadline: 7 August 2026** — same as research papers, only 8 days after the
  30 July submission deadline. Analysis + prose must be essentially done during
  the submission window. Notification 2 Sept; **camera-ready 11 Sept** (official
  test rankings get added then if not out earlier).
- Length: "no maximum length for system papers, but normally a short paper
  (4–6 pages) is appropriate". EMNLP formatting (ACL style files);
  **not anonymized**; Limitations/Ethics sections optional.
- Submit via SoftConf: https://softconf.com/emnlp2026/wmt2026/ (owner needs a
  SoftConf/START account).
- "System papers must describe one or more shared task submissions" — cite the
  task overview paper (placeholder cite from organizers), CometKiwi (Rei et
  al., 2022), xCOMET (Guerreiro et al., 2024), WMT25 General-MT ESA data.

## Open questions / owner escalations

1. **HF gated access** — no HF token on this machine yet; blocks CometKiwi/xCOMET download.
2. **Codabench account** — create now; competition links not yet live, watch task pages.
3. **wmt-tasks Google group** — join: http://groups.google.com/group/wmt-tasks (state a reason). Organizer contact: `wmt-qe-metrics-organizers@googlegroups.com`.
4. Exact schema + Codabench URLs → re-recon 17–20 July.
5. Repeated-submission (dry-run) policy on Codabench → check when links live.
6. Task 3 ranking aggregation (per-LP vs pooled MCC) not stated → affects how much per-LP calibration matters for the official table; ask organizers if still unclear after schema drop.
