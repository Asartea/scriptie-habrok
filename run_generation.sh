#!/bin/bash

set -e

module load EESSI/2025.06
module load Python/3.13.5-GCCcore-14.3.0
module load uv

source .env
uv run generation/main.py
