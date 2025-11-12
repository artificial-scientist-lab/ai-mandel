#!/usr/bin/env bash
set -euo pipefail

# Minimal runner: call expert.py with predefined args.
python3 expert.py \
  --ensemble-dir "ensemble" \
  --journal-dir "journal_o4-mini" \
  --all-journals \
  --max-internal-expert-calls 20 \
  --max-external-expert-calls 20 \
  --max-researcher-calls 20
