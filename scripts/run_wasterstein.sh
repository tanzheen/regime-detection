#!/usr/bin/env bash
set -euo pipefail

uv run python src/train/wasterstein_train.py "$@"
