#!/usr/bin/env bash
set -euo pipefail

# Minimal runner: call researchers.py with predefined args.
python3 researchers.py \
  --ensemble-dir "ensemble" \
  --max-researcher-calls 20
