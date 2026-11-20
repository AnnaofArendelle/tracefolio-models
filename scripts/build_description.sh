#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Build the fixed CPU runtime natively on the Ubuntu 22.04 release baseline.
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ "$(getconf GNU_LIBC_VERSION)" != "glibc 2.35" ]]; then
  echo "Description release builds require Ubuntu 22.04 / glibc 2.35." >&2
  exit 1
fi
"$project_root/workers/.venv/bin/python" - "$project_root" <<'PY'
import hashlib,json,shutil,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]);lock=json.loads((root/'packaging/locks/llama-source.json').read_text())
archive=root/'build/sources'/lock['filename']
with archive.open('rb') as stream:checksum=hashlib.file_digest(stream,'sha256').hexdigest()
if checksum!=lock['sha256']:raise SystemExit('llama.cpp source digest mismatch')
out=root/'build/description-source';out.mkdir(exist_ok=True)
with tarfile.open(archive) as tar:tar.extractall(out,filter='data')
# Upstream may reuse previously downloaded UI assets even with both UI flags
# disabled. They are not part of this fixed, server-only runtime.
ui_cache=root/'build/description-release-build/tools/ui/dist'
if ui_cache.exists():shutil.rmtree(ui_cache)
PY
source_revision=7d7f0287530bc8981690f361d8c1627dbae1c99a
runtime_source="$project_root/build/description-source/llama.cpp-$source_revision"
runtime_build="$project_root/build/description-release-build"
runtime_prefix="$project_root/build/description-runtime"
cmake -S "$runtime_source" -B "$runtime_build" \
 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF \
 -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX512=OFF -DGGML_AVX_VNNI=OFF \
 -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_CPU_ARM_ARCH=armv8-a \
 -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
 -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON \
 -DLLAMA_BUILD_COMMIT="$source_revision" -DLLAMA_BUILD_NUMBER=0 \
 -DCMAKE_EXE_LINKER_FLAGS='-static-libstdc++ -static-libgcc'
cmake --build "$runtime_build" --target llama-server -j2
mkdir -p "$runtime_prefix/bin"
install -m 0755 "$runtime_build/bin/llama-server" "$runtime_prefix/bin/llama-server"
"$project_root/workers/.venv/bin/python" "$project_root/scripts/elf_audit.py" "$runtime_prefix" --output "$project_root/build/reports/description-elf-audit.json"
