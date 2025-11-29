#!/bin/bash

# Generic training entrypoint.
# By default this runs Stage 1 via the unified launcher.
# You can still call train_stage_1.sh / train_stage_2.sh directly if you prefer.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STAGE="${STAGE:-1}" bash "$SCRIPT_DIR/train_launch.sh" "$@"

