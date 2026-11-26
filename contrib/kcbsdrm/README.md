# contrib/kcbsdrm — keys, record parser and an EVS check against ground truth

Code and data from [martynw/KCBSDRM](https://github.com/martynw/KCBSDRM), packaged so it runs inside this repo.
GPL-3.0, like that repo. The scripts expect this folder layout (`scripts/` next to `reference/`).

## 1. EVS 13.2 matches the March satellite audio

`scripts/evs_march_check.py` decodes the March 16 sign-on through the 3GPP EVS decoder. It uses your framing:
`[mode byte][32-byte payload]` as one 264-bit G.192 frame. The input is our 3205 kHz dump of that broadcast
(`reference/sf_signon_3205_2026-03-16.bin`, counters 25463983–25464737, the last 755 groups of your 19:06
timeline). It keeps the 702 groups whose idle key passes the CRC.

Only **role-0** frames are supplied (all 11 keyed bytes known); every other slot is marked lost. The controls
keep the same loss pattern and the mode byte: `wrongkey` decrypts with the key of a group 50 away, and
`randpay` replaces the 32 payload bytes with random bytes.

| | real | wrongkey | randpay |
|---|---|---|---|
| 1 kHz share of 100–7000 Hz power, t = 20–80 s | **0.98** | 0.01 | 0.01–0.02 |
| t = 90–265 s | digital silence (rms 6) | noise | noise |
| 50 ms log-energy r vs satellite audio, programme 320–600 s | **+0.63** (null max 0.36) | 0.11 | 0.02 |

Time t is measured from the start of our dump. The satellite recording is time-aligned to the same capture.
It has a 1 kHz line-up tone, then digital silence, chimes and the programme. The decode reproduces all of
these, at a lag of about 50 s, as measured independently earlier. Random payload removes the effect, so this
is not mode-byte leakage.

On your README's point that the March 1 kHz reference was KCBS-1: whatever that satellite channel is called,
this DRM stream carries the same audio. That makes the recording usable ground truth for KCBS-2 audio work.

```bash
python scripts/evs_march_check.py --evs 'wsl|-d|Ubuntu|--|/path/to/EVS_dec' [--sat satellite_aligned.wav] [--out wavs/]
```

`--evs` takes the decoder command with `|` between arguments. Without `--sat`, only the tone and silence test
runs. The satellite WAV (46 MB) is not in this PR; ask if you want it.

## 2. November keys in your format

`keys/keys_<capture>.npz` has `counters` (uint32, the phase-0 counter at `du[664:668]`), `ks` (N × 16 uint8)
and `crc_supported` (N × 16 bool). That is the layout `render_supported_nonidle.py --keys` reads. The
originals are `reference/keys_records_*.npz`, with `verified` in place of `crc_supported`.

| capture | groups | all 16 bytes |
|---|---|---|
| 2026-11-25 04:48+04:57 KST, 3205, hik | 1316 | 1114 |
| 2026-11-25 12:46 KST, 6140, merged | 679 | 586 |
| 2026-11-25 22:42 KST, 3205, hl5ntr | 760 | 599 |
| 2026-11-26 05:00 KST, 3205, hl5ntr | 379 | 221 |
| 2026-11-26 06:42 KST, merged | 518 | 387 |
| 2026-11-26 10:08 KST, 6140, merged | 760 | 610 |

A byte counts as supported when its record passed the CRC **and** the structure checks (0000 padding, and a
BBBB fill block after a non-final record). Caveat: a few such records were later shown to be garbled. The key
search can satisfy a 16-bit CRC by chance; we found 6 garbled copies in the 10:08 capture and a handful in
the others. Their groups can carry wrong key bytes. `pair_confirm.py` finds them (below).

To match these keys to your dumps, use the counter. Ours come from Dream super-frame dumps
(`reference/sf_*.bin`), not your length-prefixed MSC dumps.

## 3. Record parser and text

| script | what it does |
|---|---|
| `pair_records.py` | variable-length record model of the side channel, then a key search per record. Writes `keys_records_*.npz`, `items_*.json` and `sidechannel_items_*.txt` |
| `pair_confirm.py TAG sf.bin [--apply]` | cross-copy check. Plants a record's units at another loop copy, where groups align differently; true text gives zero key contradictions. Catches CRC false passes |
| `pair_title_joint.py`, `pair_plaintext.py` | item openings, then the final plain text (`reference/KCBS2_text_*.txt`) |
| `pair_keypredict.py` | key-prediction tests with random and planted controls. All null |
| `pair_bytebias.py` | per-bit P(1) of the decrypted March frames |

Record header: `len = ((b0 & 7) << 4) | (b1 & 0x0F)`, `seq = b1 >> 4`, last-record flag `b0 & 0x20`, toggle
`b0 & 0x80`. Then the text, CRC-16, 0000 padding to the 16-byte block, and one BBBB fill block after a
non-final record. See the `pair_records.py` docstring.

Tested from this folder: `evs_march_check.py`, `pair_keypredict.py`, `pair_bytebias.py`, and
`pair_confirm.py` on 10:08 (115 positions confirmed, 1 solved from several copies, 2 unverified). The full
`pair_records.py` key search was not rerun here; it takes much longer. On Windows, set
`PYTHONIOENCODING=utf-8` for the Korean output.

## What limits the audio now

28 of the 44 key bytes per group are unknown: role 1 bytes j=5–10, and all of roles 2 and 3. Two shortcuts
fail:

- Silence and tone frames are not constant (2062 of 2062 and 719 of 719 role-0 frames are distinct), so there
  is no constant-plaintext crib.
- Guessing each byte from its value distribution, tested on the known role-1 bytes j=0–4, gets 71–83 of 351
  right for j=0 and at most 17 of 351 for j=1–4.

XOR differences between same-role frames within a group do not depend on the key. A useful score has to
compare candidate plaintext with frames under other keys, or with the satellite audio. Our next step is a
decoder-in-the-loop search against the tone interval.
