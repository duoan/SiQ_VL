#!/bin/bash

# Thin wrapper for Stage 2 training.
# Delegates all logic (host detection, venv, accelerate, etc.) to train_launch.sh.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STAGE=2 bash "$SCRIPT_DIR/train_launch.sh" "$@"


