#!/usr/bin/env bash
# Decode a 48 kHz DRM I/Q wav and dump the raw MSC logical frames.
# Usage: ./run_dump.sh /path/to/kcbs_..._48k.wav [out.bin]
set -e
cd "$(dirname "$0")"
IN="$1"; OUT="${2:-$PWD/sf_2026-03-17_3205.bin}"
[ -f "$IN" ] || { echo "input not found: $IN"; exit 1; }
BIN=$(find dream-src -maxdepth 2 -name dream -type f | head -1)
[ -n "$BIN" ] || { echo "dream binary not built; run ./build.sh"; exit 1; }
export KCBS_DUMP_MSC="$OUT"
export KCBS_DUMP_AUD="${OUT%.bin}_aud.bin"
export QT_QPA_PLATFORM=offscreen     # headless
echo "dumping MSC -> $KCBS_DUMP_MSC"
"$BIN" -c 6 --sigsrate 48000 -m 1 -g 0 -f "$IN" || true
ls -la "$KCBS_DUMP_MSC" "$KCBS_DUMP_AUD" 2>/dev/null || true
