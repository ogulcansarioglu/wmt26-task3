#!/usr/bin/env bash
# Thin shim so "download scripts live in scripts/" — logic is in src/data.py.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m src.data download "$@"
