#!/usr/bin/env bash
# Metadata-only preparation by default. Pass --submit for explicit submission.
set -euo pipefail
SOURCE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
exec python3 -B -S "$SOURCE_ROOT/simulation/discovery_case_aggregation.py" prepare "$@"
