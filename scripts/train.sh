#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Training Stage 1..."
STAGE=1 bash "$SCRIPT_DIR/train_launch.sh" "$@" || {
    echo "Training Stage 1 failed."
    exit 1
}
echo "Training Stage 1 completed."

print "Training Stage 2..."
STAGE=2 bash "$SCRIPT_DIR/train_launch.sh" "$@" || {
    echo "Training Stage 2 failed."
    exit 1
}
echo "Training Stage 2 completed."

