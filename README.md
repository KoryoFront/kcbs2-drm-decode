# KCBS-2 DRM decoder

Tools for analysing the **KCBS-2** Digital Radio Mondiale shortwave broadcast from
raw KiwiSDR IQ recordings. The repo parses the DRM container, recovers some
side-channel text, and renders experimental audio through a candidate codec.
**There is no verified broadcast audio decode** and no demonstrated complete
Korean-text recovery in this repo.

## Render the bundled 19:06 recording

Three adjacent Daegu 3205 kHz captures from March 16, 2026 are in `recordings/`.
Their MSC dumps are in `dumps/`, so this example does not require a demodulator.

1. Install Python 3.10+, `pip install -r requirements.txt`, and WSL2 with Ubuntu.
   The Python codec wrappers invoke the Linux binary through WSL.
2. Obtain the 3GPP TS 26.443 EVS floating-point reference source (not bundled)
   and place it at `work/ref/evs/src/c-code`. Build with:

   ```bash
   python scripts/build_evs_gcc_independent.py --out work/evs_gcc_independent
   ```

3. From the repo root in Git Bash, run:

   ```bash
   python scripts/kcbs_decode_fast.py recordings/*_iq.wav --out work/march_1906
   ```

The output, `work/march_1906/KCBS2_MARCH_UNVERIFIED_32k_25463783-25464737.wav`,
is 19:06 of mono 32 kHz PCM. Rendering from cached dumps takes about 30 seconds
with the codec already built. Use a new `--out` directory for each run.

- **865 of 955** timeline groups (90.6%) have a CRC-supported 16-byte mask prefix.
  That leaves **12,975 fully unmasked frame slots out of 57,300**. Unmasked
  bytes are not the same as verified speech.
- In each keyed group, one in four slots is fully unmasked. The next slot uses
  six substitute bytes from the preceding slot. The last two are marked lost.
- Counter gaps keep their original duration. By default the decoder bridges up
  to four missing groups; longer gaps are silent with 20 ms fades at the edges.
  The longest silence is **3:18–4:00**. See `report.json` for all intervals.
- `--merge-gap 0` silences every missing group. `--rate` accepts 16000, 32000,
  or 48000. `--fresh` ignores the bundled dumps and requires patched Dream.

An optional listening filter reduces the repeating four-slot loudness swing. It
only attenuates samples, leaves silent gaps intact, and does not recover speech:

```bash
python scripts/march_listening_filter.py \
    work/march_1906/KCBS2_MARCH_UNVERIFIED_32k_25463783-25464737.wav \
    --out work/march_1906/KCBS2_MARCH_UNVERIFIED_32k_LESS_FLUTTER.wav
```

## Record and decode your own captures

Use **IQ**, not the KiwiSDR’s audio output. A good starting point is
`hl5ntr.ddns.net:8074` in Daegu on **3205 kHz**. **6140 kHz** can carry the same
payload and can be better under different reception conditions. Our pooled
windows yielded 76% CRC-good on 3205 and 30% on 6140; those are sample results,
not intrinsic properties of the frequencies. Daejeon (`hik.iptime.org`) and
Japanese receivers are also useful for simultaneous captures. Reported DRM hours
are **05:00–03:00 KST** (20:00–18:00 UTC); a failed lock does not prove it was off-air.

```bash
python kiwirecorder.py -s hl5ntr.ddns.net -p 8074 -f 3205 -m iq \
    --tlimit=1800 --station=drm -d ./captures
```

`kiwirecorder.py` is from [kiwiclient](https://github.com/jks-prv/kiwiclient).
For new IQ captures, install FFmpeg and build patched Dream with `bash build/build.sh`
in WSL Ubuntu (see [build/README.md](build/README.md)). The patch writes the MSC
stream via `KCBS_DUMP_MSC`. The general pipeline demodulates the IQ, salvages
packets, and writes `text/kcbs2_text.txt` and experimental audio:

```bash
python scripts/kcbs_decode_audio.py captures/mycapture_iq.wav --out work/mycapture
```

The text output distinguishes CRC-supported records from heuristic candidates.
`--allow-candidate-keys` permits the per-run renderer to use heuristic keys when
no full CRC-supported keys are available. The experimental `--montage` path can
also use heuristics without that flag; do not treat montage output as CRC-only.

To combine simultaneous length-prefixed MSC dumps without re-demodulating:

```bash
python scripts/salvage_msc_packets.py receiver-a.msc.bin receiver-b.msc.bin \
    --out work/msc_salvage
```

Salvage merges packet-CRC-good payloads at the same counter and phase, rejects
conflicting copies, and requires the reassembled data unit to pass its CRC.
The March fast path instead uses complete CRC-good groups in the cached dumps.

## Container and partial mask

- An MSC record has a four-byte little-endian length, then normally four 182-byte
  packets. Each packet has a one-byte header, 179 payload bytes, and a two-byte
  CRC. The concatenated 716-byte data unit (`du`) has its own CRC at `du[714:716]`.
  CRC-16 uses polynomial `0x1021`, initial value `0xFFFF`, and final XOR `0xFFFF`.
- Three data units with `du[1] >> 4` equal to 0, 1, and 2 make a 1.2 s group.
  The big-endian counter in phase 0 at `du[664:668]` increments once per group.
- Each data unit has 20 nominal 20 ms frames. The four 32-byte payloads in cycle
  `c` start at `7`, `40`, `73`, and `106` plus `131*c`, for `c=0..4`. The
  first mode byte is at `du[2+c]`; the others immediately precede their payload.
- Payload bytes 0, 3, 6, …, 30 are XOR-masked. For `role = frame_index % 4`,
  `plain[3*j] = cipher[3*j] XOR ks[11*role+j]`, for `j=0..10`. The 44-byte
  mask is reused within the group. The 21 other bytes per frame are unmasked.

The text tail `du[698:714]` shares `ks[0..15]` with the audio layout. This covers
all 11 role-0 mask bytes and five of role 1’s 11. The other six role-1 bytes and
all 22 role-2/3 bytes have not been recovered. Similarity checks between roles do
not establish independence. A one-step change of counter changes about 64 of the
128 known bits on average; that does not identify AES-CTR, establish cipher
strength, or locate a secret in receiver hardware. Byte-identical payloads across
two transmitters support a common upstream masked stream.

**Opaque trailer.** All 865 bundled March groups match this layout immediately
before the text tail:

| Phase | Layout, starting at `du[662]` |
|---|---|
| 0 | 2 zero bytes, 4-byte counter, `du[668:698]` (30 variable bytes) |
| 1 | 3 zero bytes, `du[665:698]` (33 variable bytes) |
| 2 | 2 zero bytes, `du[664:697]` (33 variable bytes), 1 zero byte |

The resulting 96-byte blocks are distinct in the merged capture; their pooled
bit-one fraction is 0.5005. Their function is unknown. Simple mappings tested so
far did not produce supported recovery of additional mask bytes.

## Text recovery and validation

**Idle text:** In bundled March groups the phase-0 plaintext tail is sixteen
`0xFF` bytes. The mask derived from it makes phases 1 and 2 match the framed
`TextMessage is sending...` placeholder and its CRC. This validates the 16-byte
prefix, not codec identity or the resulting audio.

**Scrolling text:** The implemented path uses `0xBB` fill and repeated text to
recover prefixes. It checks a common 112-byte layout: header `05` or `85`, then
`(seq << 4) | 0x0C`, 92 bytes of UTF-16LE text, a big-endian CRC over the header
and text, and 16 bytes of `0xBB` fill. CRC is an integrity check, not authentication.
Hangul density is only a heuristic: even the fill word `0xBBBB` counts as a Hangul
syllable, and forcing a fill biases this score.

**Variable-length records:** External analysis reports a block-aligned header
with length `((b0 & 7) << 4) | (b1 & 0x0f)`, a last-record flag and sequence
number. It reports zero padding after the text CRC and fill after non-final
records; final records and titles do not fit the fixed grid. Text recurs after
323 groups (6:27.6); the reported per-item toggle can require two loops for exact
header equality. The external counts **469/505** and **247/267** are fractions
of records, not fractions of unique groups with all 16 mask bytes. The parser
and those results have not been reproduced in this repo. A fair comparison
needs per-counter byte masks and provenance, against both the number of received
groups and the full timeline. Fixed-grid coverage is not a ceiling.

## Audio evidence and limitations

- The `[mode byte][32-byte payload]` reading gives 264 bits, consistent with
  EVS 13.2. In a GCC decoder test, **0/255** supplied frames raised error flags,
  versus **53–79** for uniform wrong-key controls. Those controls also change
  byte statistics; this does not identify the codec. Stronger controls preserve
  byte or bit statistics within each mode and use the same frame-loss pattern.
- The candidate did not beat wrong-key controls against the paired November
  stream. At the selected lag, held-out correlation was **−0.025**. Known EVS
  audio provided a positive control; a successful round-trip validates the build,
  not that the broadcast uses EVS.
- The old March 1 kHz audio reference was **KCBS-1**, not KCBS-2. A change in
  DRM frame statistics does not identify a 1 kHz tone or digital silence. An
  independently verified KCBS-2 recording is needed to label those intervals.
- The role-1 continuity pilot substituted a constant plaintext byte across
  different frames. It did not apply a shared XOR mask to those frames and does
  not establish that the six bytes are impossible to recover. Substitutes and
  decoder concealment can sound unnatural or speech-like without matching speech.

## Tests and licensing

```bash
for t in test_key_recovery test_november_recovery test_speech_reconstruction test_aio_pipeline test_march_playback; do
    python scripts/$t.py || exit 1
done
```

The tests use synthetic inputs to check CRC and salvage, key evidence, alignment,
frame placement, donor-byte coordinates, gap length, and fades. The bundled March
command is the end-to-end check. Dream is GPL; see [docs/NOTICE.md](docs/NOTICE.md)
and [docs/COPYING.FAAD2](docs/COPYING.FAAD2) for third-party notices. The 3GPP EVS
reference codec is not redistributed here.
