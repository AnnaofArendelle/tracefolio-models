#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Reproducible local media build; source archives are pinned in packaging/locks.
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
media_prefix="$project_root/build/media-runtime"
source_dir="$project_root/build/media-source"
build_glibc=$(getconf GNU_LIBC_VERSION)
if [[ "$build_glibc" != "glibc 2.35" ]]; then
  echo "Media release builds require Ubuntu 22.04 / glibc 2.35 (found $build_glibc)." >&2
  exit 1
fi
mkdir -p "$source_dir" "$media_prefix/bin" "$project_root/build/reports"
"$project_root/workers/.venv/bin/python" - "$project_root" "$source_dir" <<'PY'
import hashlib,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]);out=Path(sys.argv[2])
for component in ['ffmpeg','x264','zlib']:
    lock=json.loads((root/'packaging/locks'/f'{component}-source.json').read_text())
    archive=root/'build/sources'/lock['url'].split('/')[-1]
    with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    if digest!=lock['sha256']:raise SystemExit('source digest mismatch: '+component)
    with tarfile.open(archive) as tar:
        names=tar.getnames();top=Path(names[0]).parts[0]
        if not (out/top).exists():tar.extractall(out,filter='data')
PY
export PATH="$project_root/build/toolchain/nasm/usr/bin:$PATH"
export SOURCE_DATE_EPOCH=1794268800
export LC_ALL=C
x264_source="$source_dir/x264-b35605ace3ddf7c1a5d67a2eb553f034aef41d55"
ffmpeg_source="$source_dir/ffmpeg-9.0.2"
cd "$x264_source"
if [[ ! -f "$media_prefix/lib/libx264.a" || ! -f "$media_prefix/.glibc-2.35" ]]; then
  ./configure --prefix="$media_prefix" --enable-static --enable-pic --disable-cli --disable-opencl
  make -j2
  make install-lib-static install-lib-dev
fi
cd "$source_dir/zlib-1.4"
./configure --prefix="$media_prefix" --static
make -j2
make install
cd "$ffmpeg_source"
PKG_CONFIG_PATH="$media_prefix/lib/pkgconfig" ./configure \
  --prefix="$media_prefix" --pkg-config-flags=--static \
  --extra-cflags="-I$media_prefix/include" --extra-ldflags="-L$media_prefix/lib" \
  --disable-shared --enable-static --disable-autodetect --disable-debug --disable-doc \
  --disable-devices --disable-network --disable-ffplay --enable-gpl --enable-libx264 --enable-zlib \
  --disable-protocols --enable-protocol=file,pipe \
  --disable-demuxers --enable-demuxer=mov,matroska,amr,aac,flac,mp3,wav,ogg,image2,png_pipe,jpeg_pipe,webvtt,srt,ass,concat,rawvideo \
  --disable-encoders --enable-encoder=libx264,aac,flac,pcm_s16le,pcm_f32le,png,mjpeg,webvtt,subrip,ass,rawvideo,wrapped_avframe \
  --disable-muxers --enable-muxer=mp4,matroska,wav,flac,adts,image2,image2pipe,rawvideo,s16le,f32le,srt,webvtt,null \
  --disable-filters --enable-filter=aresample,atrim,trim,asetpts,scale,format,fps,setpts,select,showinfo,ashowinfo,aformat,anull,null,pad,transpose,crop,split,asplit,buffer,abuffer,buffersink,abuffersink
make -j2 ffmpeg ffprobe
install -m 0755 ffmpeg ffprobe "$media_prefix/bin/"
mkdir -p "$media_prefix/licenses"
install -m 0644 COPYING.GPLv2 COPYING.GPLv3 COPYING.LGPLv2.1 COPYING.LGPLv3 LICENSE.md "$media_prefix/licenses/"
install -m 0644 "$x264_source/COPYING" "$media_prefix/licenses/x264-COPYING"
install -m 0644 "$source_dir/zlib-1.4/LICENSE" "$media_prefix/licenses/zlib-LICENSE"
"$media_prefix/bin/ffmpeg" -version > "$project_root/build/reports/media-version.txt"
"$media_prefix/bin/ffmpeg" -buildconf > "$project_root/build/reports/media-buildconf.txt" 2>&1
"$media_prefix/bin/ffmpeg" -decoders > "$project_root/build/reports/media-decoders.txt" 2>&1
ldd "$media_prefix/bin/ffmpeg" > "$project_root/build/reports/media-ldd.txt"

"$project_root/workers/.venv/bin/python" "$project_root/scripts/elf_audit.py" "$media_prefix" --output "$project_root/build/reports/media-elf-audit.json"
touch "$media_prefix/.glibc-2.35"
