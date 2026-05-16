#!/bin/bash
echo "Time: $(date)"

set -e

source .env

uv run python3 -m data_generation.run_openai \
    --years 2021 2024 \
    --days 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
    --max-new-tokens 8192 \
    "$@"

echo "Finished at $(date)"
