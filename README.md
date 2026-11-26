# KCBS-2 DRM decoder

Tools for analysing the **KCBS-2** Digital Radio Mondiale shortwave broadcast from
raw KiwiSDR IQ recordings. The repo parses the DRM container, recovers some
side-channel text, and renders experimental audio through a candidate codec.
The March role-0 EVS output reproduces a specific spectral feature of the RF
capture, but a complete programme/audio match is not verified. The November
contribution adds variable-length records and greatly improves usable text-tail
coverage.

## Play or render the bundled 19:06 recording

[**Download the quieter listening WAV**](https://github.com/KoryoFront/kcbs2-drm-decode/releases/download/march-19m-quiet/KCBS2_MARCH_UNVERIFIED_REPLAY_QUIETER_32k.wav)
(19:06, mono 32 kHz PCM, −24.4 LUFS). It is a reconstruction from three adjacent
Daegu 3205 kHz captures from March 16, 2026, not a verified broadcast decode.

To reproduce it from the bundled MSC dumps without rebuilding the DRM demodulator:

1. Install Python 3.10+, `pip install -r requirements.txt`, and WSL2 with Ubuntu.
   The Python codec wrappers invoke the Linux binary through WSL.
2. Obtain the 3GPP TS 26.443 EVS floating-point reference source (not bundled)
   and place it at `work/ref/evs/src/c-code`. Build with:

   ```bash
   python scripts/build_evs_gcc_independent.py --out work/evs_gcc_independent
   ```

3. From the repo root in Git Bash, run:

   ```bash
   python scripts/kcbs_decode_fast.py recordings/*_iq.wav --out work/march_1906 \
       --replay-nearby --attenuate-db 2.2
   ```

This writes `KCBS2_MARCH_UNVERIFIED_REPLAY_QUIETER_32k_25463783-25464737.wav`
and `report.json`. Use a new `--out` directory for each render.

- **865 of 955** timeline groups (90.6%) have a CRC-supported 16-byte mask prefix.
  That gives **12,975 fully unmasked frame slots out of 57,300**, not 90.6%
  verified audio.
- Within a keyed four-frame cycle, the first slot is fully unmasked and the
  second has six substitute bytes. With `--replay-nearby`, the third copies the
  second, and the fourth copies the next fully unmasked frame when available
  or falls back to the second. These copies do not restore missing frames.
- The quieter edition uses a fixed **2.2 dB attenuation** relative to the raw
  replay render, with no compression or peak limiting.
- Counter gaps retain their duration. Gaps of up to four groups use decoder
  concealment; longer gaps are digital silence with 20 ms fades at the edges.
  The longest silence is **3:18–4:00**. `--merge-gap 0` silences every gap.
- Omit `--replay-nearby` to use decoder concealment in the last two slots of each
  keyed cycle instead. `--rate` accepts 16000, 32000, or 48000. `--fresh` ignores
  the bundled dumps and requires patched Dream.

## Record and decode your own captures

Use **IQ**, not the KiwiSDR’s audio output. A good starting point is
`hl5ntr.ddns.net:8074` in Daegu on **3205 kHz**. **6140 kHz** can carry the same
payload and can be better under different reception conditions. Our pooled
windows yielded 76% CRC-good on 3205 and 30% on 6140; those are sample results,
not intrinsic properties of the frequencies. Daejeon (`hik.iptime.org`) and
Japanese receivers are also useful for simultaneous captures. Current reported DRM
hours are **05:00–23:00 KST** (20:00 UTC the previous day to 14:00 UTC), confirmed
on 2026-11-26. A failed lock does not prove the transmitter was off-air.

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
`--allow-candidate-keys` explicitly permits heuristic keys only when no full
CRC-supported prefixes are available. The `--montage` timeline respects the
same rule and labels heuristic audio `HANGUL_CANDIDATE`, never `CRC_PREFIX`.

To combine simultaneous length-prefixed MSC dumps without re-demodulating:

```bash
python scripts/salvage_msc_packets.py receiver-a.msc.bin receiver-b.msc.bin \
    --out work/msc_salvage
```

Salvage merges packet-CRC-good payloads at the same counter and phase, rejects
conflicting copies, and requires the reassembled data unit to pass its CRC.
The March fast path instead uses complete CRC-good groups in the cached dumps.
On the longest bundled March capture, both paths independently return the same
708 counters and data units; salvage adds zero, so it does not explain the
reported 702 count. Martyn’s 702 groups are a byte-identical subset of those
708; his dump lacks the other six. For June, combining CRC-good packets across the two
frequencies yields **410 complete groups instead of 408**.
The two added groups are not additional verified text prefixes.

## Render the contributed November recording

[**Download the quieter 17:40.8 listening WAV**](https://github.com/KoryoFront/kcbs2-drm-decode/releases/download/november-quiet-review/KCBS2_NOV25_UNVERIFIED_QUIET_17m40_32k.wav).
It uses the same counter span as the previous November 24 UTC / November 25 KST
sample. This time **671 of 884 timeline groups** have full 16-byte prefixes checked
against repeated records. This is group coverage, not a percentage of verified
speech or music. Role 2 is concealed; role 3 can replay the next fully unmasked
frame. The copy measures **−26.7 LUFS**, with a **−2.6 dBTP** true peak and
11 silent gaps of their original durations. The March release is unchanged.

[Martyn’s contribution](contrib/kcbsdrm/README.md) supplies six matching groups of
dumps, prefix archives, record reports and Korean text. All **3,517** exported full prefixes matched complete received groups by
counter; the six captures contain **4,421** complete groups in total. Rechecking header/body CRC, padding, fill, and matching records
at the same position in the carousel kept **3,007 full prefixes**. Repeat agreement
is an extra consistency check, not authentication; the exports are not a source
of keys for future broadcasts.

With FFmpeg and the EVS build from the March quickstart, reproduce the release
from the contributed data (Git Bash):

```bash
python scripts/salvage_msc_packets.py \
    contrib/kcbsdrm/reference/sf_2026-11-25_04*3205_hik.bin --out work/november
python scripts/review_supplied_records.py \
    --data-units work/november/data_units.npz \
    --keys contrib/kcbsdrm/keys/keys_2026-11-25.npz \
    --repeat-blocks 969 --out work/november_review
python scripts/render_supported_nonidle.py \
    --groups work/november/groups.npz --keys work/november_review/keys.npz \
    --timeline --start-counter 43679908 --end-counter 43680791 \
    --loss-policy future --attenuate-db 2.2 --out work/november_audio
python scripts/finish_listening_wav.py \
    work/november_audio/NONIDLE_CRC_PREFIX_UNVERIFIED_FUTURE_TIMELINE_32k.wav \
    --report work/november_audio/report.json --gain-db 2.2 \
    --out work/november_audio/KCBS2_NOV25_UNVERIFIED_QUIET_17m40_32k.wav
```

Use a new output directory or filenames for reruns. Omit `--timeline` for an
excerpt reel with half-second separators. `--loss-policy replay` matches the
quieter March playback policy; `--loss-policy plc` uses concealment instead of
frame replay. None of the options recovers the missing payload or guarantees
natural music. Each render writes a counter and timing report.

To check for a pre-programme RF pulse from two overlapping recordings, use
`python scripts/rf_pulse_check.py --help`. In one June pair on 3205 and 6140 kHz
no shared pulse persisted across adjacent 40-second windows of the common
interval. A planted 0.8 Hz modulation passed the same check. A listener reported
pre-start pulsing. This must not be equated to the narrow component in the March
EVS output below.
This two-frequency power test does not check what is audible in programme
content. A confirmed audio pulse can help align recordings of the same programme;
it does not recover missing frame bytes. The capture layout audit reports exact
mode-edge counters without mistaking gaps for transitions:

```bash
python scripts/audit_capture_layout.py work/mycapture/salvage/groups.npz
```

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

**Opaque trailer.** This layout immediately before the text tail matched all **2,310** complete
groups checked across the March, June, November 24, and November 25 samples:

| Phase | Layout, starting at `du[662]` |
|---|---|
| 0 | 2 zero bytes, 4-byte counter, `du[668:698]` (30 variable bytes) |
| 1 | 3 zero bytes, `du[665:698]` (33 variable bytes) |
| 2 | 2 zero bytes, `du[664:697]` (33 variable bytes), 1 zero byte |

The resulting 96-byte blocks are distinct within each sample, with bit-one
fractions from 0.4994 to 0.5022. Their function is unknown. Simple mappings tested so
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

**Variable-length records:** `scripts/parse_text_records.py` checks already-available
plaintext using the reported block-aligned header, length `((b0 & 7) << 4) | (b1 & 0x0f)`,
last-record flag, sequence number, CRC, zero padding and a fill block after a non-final
record. The optional same-length `--known-bytes` map (0 or 1 per input byte) prevents
unknown bytes from being treated as evidence. Title and item signalling are not parsed.

```bash
python scripts/parse_text_records.py plaintext.bin --known-bytes known_mask.bin
```

In the contributed archives, the parser rechecked **1,324** full records,
including **42** final records. Observed bodies span 8–92 bytes rather than a
fixed 92. The reported ~93% figures count records; the group counts above use
a different denominator. `review_supplied_records.py` only narrows existing
masks to observations that pass the review; it never infers new mask bytes.

Observed carousel repetition depends on the recording: **969**, **700**, and
**878** 16-byte blocks (6:27.6, 4:40, and 5:51.2 at 0.4 s per block). The item
toggle can require two loops for byte-identical headers, so the repeat check
compares body, sequence, final flag and position without requiring the toggle.

## Audio evidence and limitations

- The `[mode byte][32-byte payload]` reading gives 264 bits, consistent with
  EVS 13.2. A traced GCC decoder flagged **0 of 12,975** March role-0 frames.
  Across 24 trials per control, shuffling each keyed byte column within the same
  exact mode flagged **2,983–3,117**; moving whole keyed vectors within mode
  flagged **2,963–3,116**; draws from matched bit frequencies flagged
  **3,154–3,329**. November gave **0/255** versus **17–29** for the column
  shuffles. Modes, clear bytes, frame positions and erasures were unchanged.
  So uniform wrong-key statistics do not explain the separation in these tests.
  Flags are not a universal check: some shuffles of known EVS controls also had
  zero flags. The broadcast codec and audio correspondence remain unverified.
- The candidate did not beat wrong-key controls against the simultaneously
  recorded November comparison stream. Its programme identity was not established
  by the channel label alone. At the selected lag, held-out correlation was
  **−0.025**. Known EVS audio provided a positive control; a successful round-trip
  validates the build, not that the broadcast uses EVS.
- Martyn’s March spectral result reproduces independently on both raw dumps.
  From 20–80 s on the longest capture’s counter timeline, a role-0-only EVS
  render has a **1,000 Hz** peak containing **98.15%** of the 100–7,000 Hz
  power in the 980–1,020 Hz band. Wrong-prefix and exact-mode shuffled controls
  show 0.77% and 3.20–4.24%, respectively. The candidate then falls to roughly
  six PCM counts RMS at 100–220 s. This is evidence for EVS on those particular
  inputs. It does not identify a separate satellite programme or the usual
  pre-start cue. The satellite WAV is not bundled, so its reported correlation
  was not checked. The RF mode edge is counter **25464052**, phase 0, zero-based
  slot 1 (82.82 s from the longest capture’s first group, 322.82 s in the merge).
  `scripts/audit_march_contribution.py` reproduces the RF-only test.
- The role-1 continuity pilot substituted a constant plaintext byte across
  different frames. It did not apply a shared XOR mask to those frames and does
  not establish that the six bytes are impossible to recover. Substitutes and
  decoder concealment can sound unnatural or speech-like without matching speech.

## Playback controls

The replay method improved held-out eSTOI from **0.437 to 0.520** on four genuine
EVS speech excerpts with the March loss pattern. The metric compares against the
full EVS encode, not the broadcast. Three synthetic instrumental controls showed
no meaningful gain in spectral similarity. Music in the listening copy remains
rough. These results neither identify the broadcast codec nor verify speech.

To benchmark your own local speech material after building the codec, install
FFmpeg and `pystoi`, then run:

```bash
python scripts/benchmark_loss_handling.py --speech local_speech.wav --out work/replay_controls
```

The benchmark needs at least 18 minutes of audio for the selected offsets. It
encodes independent controls locally, compares seven strategies, and keeps its
selection and held-out slices separate.

## Can this run live?

The audio stage is fast enough to prototype. On one desktop, a 17:40.8 render
with the prefixes already on disk took about 30 seconds. An offline 16-group
check took a median **0.15 s** to decode each **1.2 s** group, but opening WSL for
one group stalled for **4.84 s**. A live listener would need a persistent decoder
and a playout buffer, with missing groups passed as losses on the original clock.
These are decoder measurements, not an end-to-end live test.

A live receiver also needs stable RF lock and usable prefixes for the incoming
counter before playout. The November archives apply only to the recordings they
match. For new scrolling-text broadcasts the demonstrated workflow is delayed
review and playback, not immediate audio. No live capture is started by these
scripts. The reported operating hours remain **05:00–23:00 KST**.

## Tests and licensing

```bash
for t in test_key_recovery test_november_recovery test_speech_reconstruction test_aio_pipeline test_march_playback test_text_records test_capture_layout test_supplied_records; do
    python scripts/$t.py || exit 1
done
```

The tests use synthetic inputs to check CRC and salvage, key evidence, alignment,
frame placement, donor-byte coordinates, gap length, and fades. The bundled March and November
commands are end-to-end checks. Dream is GPL; see [docs/NOTICE.md](docs/NOTICE.md)
and [docs/COPYING.FAAD2](docs/COPYING.FAAD2) for third-party notices. The 3GPP EVS
reference codec is not redistributed here.
