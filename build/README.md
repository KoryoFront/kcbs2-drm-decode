# KCBS-instrumented Dream — ready-to-build tree

Goal: produce the raw MSC logical frames for a KCBS-2 DRM capture, in the same
format as `reference/sf_3205.bin` (`<uint32 LE length><bytes>` records), so the
codec can be fitted against known audio.

## Why this instead of the repo's `patches/dream/`

`patches/dream/*.cpp` are full replacement files from **Dream 2.2**, whose source is
only on SourceForge (Cloudflare-blocked here). Every public GitHub mirror carries a
later, refactored Dream (`AudioSourceDecoder.cpp` is 424 lines vs the patch's 1367 —
audio decoding was split into `AudioCodec`/`fdk_aac_codec`), so those files will not
drop in.

They also hard-code `D:\KCBSDRM\temp\sf_input.bin`.

`apply_kcbs_patch.py` instead inserts a **small, version-independent** dump at the two
entry points, keyed off `pvecInputData`, which exist in every Dream generation:

| hook | fires for | env var | default |
|---|---|---|---|
| `CDataDecoder::ProcessDataInternal` | KCBS-2 (signalled as a *data* service) | `KCBS_DUMP_MSC` | `kcbs_msc.bin` |
| `CAudioSourceDecoder::ProcessDataInternal` | audio-routed builds | `KCBS_DUMP_AUD` | `kcbs_aud.bin` |

KCBS-2 declares `AD_DAB_SPEC_APP`, so the **data** hook is the one that fires. Confirmed
on this machine: stock Dream 2.2 loads a FAAD2 shim but never calls it, and every RSCI
profile (A/B/C/D/M) emits `rbp0..rbp3`/`str0` with zero length.

## Build (Ubuntu / WSL)

```bash
./build.sh                      # clones if needed, patches, installs deps, qmake && make
./run_dump.sh /mnt/c/Users/culli/OneDrive/Documents/Claude/KCBS/KCBSDRM/kcbs_2026-03-17_3205_48k.wav
```

`run_dump.sh` sets `QT_QPA_PLATFORM=offscreen` so it runs headless, and writes
`sf_2026-03-17_3205.bin` next to this README.

## Then

```bash
python scripts/fit_codec.py build/sf_2026-03-17_3205.bin
```

which aligns the frames to `out/ref_features_16k.npz` (45 501 frames of per-20 ms
energy / 16 LSFs / F0 / voicing / tilt from the KCRadio1 recording of the same morning)
and regresses every bit-field against every acoustic parameter.

## Sanity checks on the dump

- records should be **728 bytes** each, one per 400 ms
- CRC-16 (x^16+x^12+x^5+1, init 0xFFFF, final complement) over `pkt[0:180]` vs the
  big-endian word at `[180:182]`, for each of the 4 packets — expect ~70 % pass
- `scripts/extract_frames.py` then yields 32-byte codec frames + mode bytes
