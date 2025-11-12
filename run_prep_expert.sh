#!/usr/bin/env bash
set -euo pipefail

# Minimal runner: call prep_expert.py with predefined args.
python3 prep_expert.py \
  --ensemble-dir "ensemble" \
  --examples-root "assets/pytheus_examples" \
  --pytheus-infos "assets/PYTHEUS_EXPLICIT_INFOS.txt"