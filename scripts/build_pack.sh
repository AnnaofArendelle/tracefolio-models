#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
offline=()
if [[ "${1:-}" == --offline ]]; then offline=(--offline); shift; fi
if [[ $# != 0 ]]; then echo 'Usage: build_pack.sh [--offline]' >&2; exit 1; fi
# Verify weights before dependency setup; missing local models must never trigger
# an upstream download. Offline dependency misses also fail explicitly.
uv run --no-project "${offline[@]}" --python 3.12.13 python scripts/verify_local_models.py
(cd workers && uv sync --frozen "${offline[@]}")
workers/.venv/bin/python scripts/fetch_release_assets.py "${offline[@]}" --sources-only
if ! workers/.venv/bin/python scripts/media-receipt.py verify; then
  bash scripts/build_media.sh
  workers/.venv/bin/python scripts/media-receipt.py write
fi
bash scripts/build_description.sh
(cd workers && uv run --frozen "${offline[@]}" pytest -q)
target=$(workers/.venv/bin/python scripts/release_platform.py)
stage="build/runtime-${target}"
workers/.venv/bin/python scripts/stage_runtime.py "$stage" --include-models --media build/media-runtime --description build/description-runtime
workers/.venv/bin/python scripts/verify_pack.py --runtime "$stage" --report "build/reports/models-${target}.json"
mkdir -p dist
workers/.venv/bin/python scripts/pack_runtime.py "$stage" "dist/tracefolio-models-${target}.tar.gz" > "build/reports/archive-${target}.json"
workers/.venv/bin/python scripts/pack_assets.py --runtime "$stage" --directory dist
