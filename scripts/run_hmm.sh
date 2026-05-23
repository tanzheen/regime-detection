#!/usr/bin/env bash
set -euo pipefail

uv run python src/train/hmm_train.py "$@"
