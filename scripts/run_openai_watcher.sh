#!/bin/bash
echo "Time: $(date)"

set -e

source .env

uv run python3 -m data_generation.openai.watcher \ "$@"

echo "Finished at $(date)"
