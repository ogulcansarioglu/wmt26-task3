PY := .venv/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1

.PHONY: venv freeze smoke test lint download-dev build-dev smoke-model \
        score-dev-baseline score-dev-primary calibrate-baseline rehearsal

venv:  ## fresh environment (python3.12; `uv venv` also works)
	python3.12 -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt

freeze:  ## lock exact versions for provenance
	$(PY) -m pip freeze > requirements-lock.txt

# ---- gates: nothing merges without these two passing -----------------------

smoke:  ## full pipeline on 10 synthetic samples, no model, <2 min
	$(PY) -m src.data make-smoke --out data/smoke/smoke_dev.jsonl
	$(PY) -m src.score --model mock --input data/smoke/smoke_dev.jsonl \
		--run runs/smoke --mode chunked --device cpu --overwrite
	$(PY) -m src.calibrate --run runs/smoke --model mock --mode chunked \
		--agg min --min-n 4 --min-class 1
	$(PY) -m src.submit --run runs/smoke --model mock --mode chunked \
		--agg min --task 3 --out submissions/smoke
	@echo "SMOKE OK"

test:  ## unit + e2e tests
	$(PY) -m pytest tests/ -q

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

# ---- data -------------------------------------------------------------------

download-dev:  ## WMT25 General-MT sources + ESA humeval (143 MB)
	$(PY) -m src.data download

build-dev:  ## canonical dev parquet with Task 3 gold labels
	$(PY) -m src.data build-dev

# ---- model runs (require `hf auth login` + gated-model acceptance) ----------

smoke-model:  ## 10 real samples through CometKiwi on MPS — first GPU sanity check
	$(PY) -m src.data make-smoke --out data/smoke/smoke_dev.jsonl
	$(PY) -m src.score --model cometkiwi --input data/dev/wmt25_esa_dev.parquet \
		--run runs/model-smoke --limit 10 --batch-size 4 --overwrite

# Long runs: prefix with `caffeinate -is` so the Mac doesn't sleep mid-inference.
score-dev-baseline:  ## Stage A: CometKiwi over full dev, chunked + truncate ablation
	caffeinate -is $(PY) -m src.score --model cometkiwi \
		--input data/dev/wmt25_esa_dev.parquet --run runs/dev --mode chunked --batch-size 16
	caffeinate -is $(PY) -m src.score --model cometkiwi \
		--input data/dev/wmt25_esa_dev.parquet --run runs/dev --mode truncate --batch-size 16

score-dev-primary:  ## Stage B: xCOMET-XL over full dev (overnight)
	caffeinate -is $(PY) -m src.score --model xcomet-xl \
		--input data/dev/wmt25_esa_dev.parquet --run runs/dev --mode chunked --batch-size 4

calibrate-baseline:
	$(PY) -m src.calibrate --run runs/dev --model cometkiwi --mode chunked --agg min

rehearsal:  ## dress rehearsal: format dev as if it were test, validate, package
	$(PY) -m src.submit --run runs/dev --model cometkiwi --mode chunked --agg min \
		--task 3 --out submissions/rehearsal
