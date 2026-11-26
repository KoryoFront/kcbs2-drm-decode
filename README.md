# KCBS-2 DRM decoder

Tools for decoding the **KCBS-2** Digital Radio Mondiale shortwave service (Korean
Central Broadcasting Station, DPRK) from raw KiwiSDR IQ recordings.

The payload is partly encrypted. These scripts parse the DRM container, extract the
side-channel text stream, recover the keystream bytes that stream exposes, and decode
the audio frames those keys cover.

A **working recording is bundled**, together with its demodulated bitstream, so the
main path runs end to end in about 20 seconds without building the demodulator.

---

## Scope

| | Status |
|---|---|
| DRM container, packet and data-unit CRCs | Fully parsed and CRC-checked |
| Side-channel keystream `ks[0..15]` | Recovered wherever a crib exists |
| Korean broadcast text | 24–36% of groups on scrolling-text captures |
| Audio, role-0 frames (1 in 4) | Exact keys wherever a crib exists |
| Audio, role-1 frames (1 in 4) | 5 of 11 keyed bytes known, 6 substituted |
| Audio, role-2 and role-3 frames (2 in 4) | Not recovered — decoder concealment |
| Codec identity | EVS 13.2 passes syntax checks; not confirmed |
| Match to the broadcast | Not demonstrated |

There is **no complete audio decode** and **no complete text decode**. At most half the
audio frames carry real data. Audio output is labelled `UNVERIFIED` throughout.

---

## Two kinds of capture

Everything depends on whether the text channel provides a known-plaintext crib.

**Idle captures.** When the text channel is idle the phase-0 tail is sixteen `0xFF`
bytes, so `ks[0..15] = tail XOR 0xFF` falls straight out, and the other two phases
decode to a framed message whose CRC-16 confirms the key. The bundled March recording
is one of these: **708 of 708 groups verify, 93.8% coverage**. The text is only the
placeholder `TextMessage is sending...`, so idle captures are excellent for audio and
carry no text content.

**Scrolling-text captures.** With real Korean text scrolling there is no idle seed. The
crib is the `0xBB` fill padding each 112-byte segment: where a tail is all fill,
`ks = tail XOR 0xBB`. Only a minority of groups contain a fill, which caps coverage
near a third. These captures carry **real broadcast text** and give choppier audio.

---

## Requirements

To decode the bundled recording:

- Python 3.10+ — `pip install -r requirements.txt`
- The **3GPP EVS reference codec**, built by `scripts/build_evs_gcc_independent.py`
- Linux, or Windows with **WSL2 (Ubuntu)** (the codec is invoked through WSL)

To decode **your own** captures you also need:

- `ffmpeg` on `PATH`
- A **patched Dream** demodulator, built from `build/` and `patches/`

### Build the demodulator (only for new captures)

Stock Dream does not expose the MSC bitstream and its FAAD2 rejects this stream. The
patches add a `KCBS_DUMP_MSC` environment variable that writes every super-frame to a
file.

```bash
# from WSL Ubuntu
bash build/build.sh          # manual steps in build/README.md
```

Produces `/root/kcbs/dream-src/dream`. Not needed for the bundled recording: its
demodulated super-frames ship in `dumps/`, and `kcbs_decode_fast.py` finds them by
capture name.

### Build the codec

Obtain the 3GPP TS 26.443 EVS floating-point reference source (not redistributable
here) and place it at `work/ref/evs/src/c-code`, then:

```bash
python scripts/build_evs_gcc_independent.py --out work/evs_gcc_independent
```

This compiles the **unmodified** reference sources with GCC. A miscompiled codec can
degrade audio silently while still appearing to run, so verify any build by
round-tripping known speech through its own encoder and decoder: a sound build scores
roughly +4 to +5 dB SNR and STOI above 0.8 at 13.2 kbps.

---

## Decode the bundled recording

```bash
python scripts/kcbs_decode_fast.py \
    recordings/hl5ntr.ddns.net_2026-03-16T19_53_42Z_3205.000_iq.wav \
    --out work/march
```

```
idle keys: 708 of 708 groups pass the known text + CRC (708 verified)
{
  "file": "KCBS2_MARCH_IDLEKEYS_32k_25463983-25464737.wav",
  "seconds": 906.0,
  "keyed_groups": 708,
  "coverage": 0.938,
  "wall_seconds": 25.0
}
```

About **20 seconds for 906 seconds of audio**: nearly all of it is the codec, since the
keys need no search and the whole capture goes through in one call. The shipped
`dumps/` are used automatically, so this run needs no demodulator.

Options: `--rate 16000|32000|48000`, and `--fresh` to ignore the shipped dumps and
re-demodulate from the IQ (requires patched Dream, and takes several minutes).

---

## Decode a scrolling-text capture

```bash
python scripts/kcbs_decode_audio.py mycapture_iq.wav \
    --out work/mycapture --allow-candidate-keys --montage
```

Stages, each resumable:

1. **demod** — bounded 600 s windows through patched Dream
2. **salvage** — packet and data-unit CRCs, groups deduplicated by counter
3. **keys** — fill cribs plus XOR propagation across repeated text, kept only when a
   full 112-byte segment passes header, fill and CRC checks
4. **text** — writes `text/kcbs2_text.txt`
5. **audio** — one WAV per contiguous run, or a full timeline with `--montage`

| Flag | Effect |
|---|---|
| `--allow-candidate-keys` | use Korean-text-validated keys when none pass CRC; output marked `HANGULKEYS` |
| `--montage` | one continuous timeline, silence where no key exists |
| `--merge-gap N` | bridge up to N unkeyed groups within one decoder call (default 4; 8 covers more timeline with more concealment) |
| `--rate` | 16000, 32000 or 48000 |

Passing several captures of the same broadcast raises coverage: they merge by counter.
Two transmitters of one June broadcast took 354 groups to 408.

`text/kcbs2_text.txt` separates CRC-valid segments, deduplicated text bodies, candidates
that failed a check with their flags, and text from Korean-validated keys.

---

## Key validation

Two independent tests, because a plausible-looking key can still be wrong.

**CRC-16.** A segment's CRC covers its header and text, so a wrong key fails with
probability 65535/65536. This is the acceptance test wherever a complete 112-byte
segment can be reconstructed. CRC-16 is an integrity check, not authentication.

**Korean text density.** Where no complete segment is available, each group's 24
decrypted characters are counted for Hangul syllables (U+AC00–U+D7A3). A wrong key
yields uniform 16-bit values, which land in that block with probability 0.171 — about
4 of 24. Against a random-key control measured on the same data:

| Set | Hangul per 24 characters |
|---|---|
| genuine fill keys | mean **17.2–17.6**, min 10 |
| random keys | mean **4.1**, max 12 |

The distributions separate above 13. The control is measured per capture, not assumed.

---

## How the encryption works

The audio payload is encrypted, but not the whole of it, and the scheme leaks through
its own structure rather than through any weakness in the cipher.

### What is encrypted

Each 32-byte audio packet has **every third byte** XORed with a keystream:

```
byte  0  1  2  3  4  5  6  7  8 ...  30 31
      ^        ^        ^             ^
   encrypted            encrypted (11 of 32 bytes)
```

Positions 0, 3, 6 … 30 — **11 bytes** — are ciphertext. The other **21 bytes are
plaintext codec data**, transmitted in the clear. The mode byte preceding each packet
is also clear. So roughly two thirds of every frame is readable without any key, which
is why frame layout, modes and gross structure can be studied directly.

### Key lifetime and reuse

The keystream is organised per group and per packet position:

```
role = frame_index % 4                       four roles, 20 frames per super-frame
plain[3j] = cipher[3j] XOR ks[11*role + j]   j = 0..10
```

- One key covers a whole **1.2 s group** (3 super-frames, 60 packets).
- It is **44 bytes** total: 4 roles × 11 bytes.
- Each key byte is **reused 15 times per group** (60 packets ÷ 4 roles).

That reuse is the first structural weakness: XORing two packets 4 apart in the same
group cancels the keystream entirely, exposing the difference of two plaintexts.

The four role keys are independent — comparing role-0 and role-1 bytes gives an equality
rate of 0.0040 against a chance value of 0.0039 — so recovering one role says nothing
about another.

### The keystream generator

Each group carries a 32-bit counter at `du[664:668]`, incrementing by one per group.
The keystream behaves like a strong counter-driven pseudorandom function, `E_K(counter)`
in the manner of AES-CTR:

| Counter delta | Mean Hamming distance between keystreams |
|---|---|
| 1 | 63.9 / 128 |
| 2 | 64.5 / 128 |
| 4 | 64.1 / 128 |
| 16 | 63.9 / 128 |
| 256 | 64.0 / 128 |

A random pair would average 64.0, so a single counter increment already produces full
avalanche. There is no linear relation between counter and keystream, and MT19937 and
LCG recovery both fail. **The cipher is not the way in** — attacking it would need the
key, which is held in the receiver hardware.

### The way in: known plaintext in the side channel

Every data unit ends with a 16-byte tail at `du[698:714]` carrying a DRM text-message
stream, and that tail is encrypted with **the same first 16 keystream bytes the audio
uses**:

```
du[698:714]  =  ks[0..15]  XOR  text
                 |
                 +-- ks[0..10]  = role-0 audio key  (all 11 bytes)
                     ks[11..15] = role-1 audio key  (first 5 of 11 bytes)
```

So any guessable byte of that text yields audio keystream directly. Two situations make
the text guessable:

1. **Idle text.** When the text channel is idle the phase-0 tail is sixteen `0xFF`
   bytes, giving `ks[0..15] = tail XOR 0xFF` immediately. The remaining two phases then
   decrypt to a framed message ending in a CRC-16 that confirms the key — the check that
   validates 708 of 708 groups in the bundled recording.
2. **Segment fill.** Each 112-byte text segment ends with sixteen `0xBB` fill bytes.
   Wherever a tail lands entirely inside that fill, `ks = tail XOR 0xBB`.

Both are classic known-plaintext cribs. The cipher is untouched; the implementation
simply reuses keystream across two channels with very different plaintext entropy.

### Why it stops there

The tail is only 16 bytes, so it can only ever expose `ks[0..15]`. The remaining 28
bytes of each group's key —

```
ks[16..21]   role 1, remaining 6 bytes
ks[22..32]   role 2, all 11 bytes
ks[33..43]   role 3, all 11 bytes
```

— are never covered by any known plaintext. Role 0 decodes exactly; role 1 decodes
partially; roles 2 and 3 do not decode at all. Since each role is a quarter of the
frames, **at most half the audio frames can ever carry real data from this route**, and
in practice role 1 is approximate because 6 of its 11 keyed bytes are substituted.

### Where the encryption is applied

Two transmitters carrying the same programme were compared across 106 shared counters
and are **byte-identical over `du[:714]`** — identical ciphertext, identical counters.
The encryption therefore happens once, upstream of distribution, not per transmitter.
No DRM conditional access is signalled in the multiplex, so this is a private scheme
layered inside an otherwise standard DRM stream rather than a DRM feature.

### Attack chain in brief

```
IQ capture
  -> patched Dream           DRM demodulation, MSC super-frames
  -> container parse         packet + data-unit CRC, group assembly, counter
  -> side-channel tail       du[698:714] = ks[0..15] XOR text
  -> crib                    idle 0xFF, or 0xBB segment fill
  -> ks[0..15]               validated by segment CRC-16 or Korean text density
  -> audio frames            role 0 exact, role 1 partial, roles 2/3 concealed
  -> EVS 13.2 (hypothesis)   264-bit frames, 20 ms, explicit loss for unkeyed roles
```

---

## Format

- **Super-frame.** 4 packets of 182 bytes: 1 header, 179 payload, 2 CRC. Concatenated
  payloads give a 716-byte data unit; `du[714:716]` is its own CRC-16.
- **Group.** Three super-frames with `du[1] >> 4` = 0, 1, 2 form one 1.2 s group.
- **Counter.** `du[664:668]`, big-endian, +1 per group — the timebase throughout.
- **Audio frames.** 20 packets of 32 bytes per super-frame, starting at
  `7 + 131c`, `40 + 131c`, `73 + 131c`, `106 + 131c` for c = 0..4. Mode bytes: role 0
  hoisted to `du[2+c]`, roles 1–3 inline immediately before their packet.
- **Encryption.** Packet bytes 0, 3, 6 … 30 are XORed with a keystream. One key serves
  a whole group and repeats every 4 packets:
  `role = frame_index % 4`, `plain[3j] = cipher[3j] XOR ks[11*role + j]`.
- **Side channel.** `du[698:714]` is 16 bytes of keystream XOR text. This exposes
  `ks[0..15]`: all of role 0, plus the first five keyed bytes of role 1.
- **Text segment.** 112 bytes — header `05`/`85`, `(seq << 4) | 0x0C`, 46 UTF-16LE
  characters, big-endian CRC-16 over bytes 0:94, then sixteen `0xBB` fill bytes.
- **Carousel.** The text stream repeats with a measured period of 323 groups (387.6 s).
- **Upstream encryption.** Two transmitters carrying one programme are byte-identical
  over `du[:714]` — verified across 106 shared counters — so the payload is encrypted
  before distribution. No DRM conditional access is signalled.

---

## Limitations

**The missing keystream.** The side channel only ever exposes `ks[0..15]`. The
remainder — `ks[16..21]` for role 1 and `ks[22..43]` for roles 2 and 3 — never appears
in the signal and is not derivable by any method here. Roles 2 and 3 are always decoder
concealment, which is why at most half the audio frames carry real data.

**Role 1 cannot be completed.** Its six unknown bytes were tested for identifiability
against genuine speech with known values: **0 of 18 ranked first, median rank 152 of
256**, and all 256 candidates decode with zero decoder error flags. With no constraint
to search against, those bytes are substitutes chosen for continuity, not recovered
keys — a search would invent plausible values rather than find true ones.

**The codec is unconfirmed.** Reading each frame as `[mode byte][32-byte payload]`
MSB-first gives 264 bits, matching EVS 13.2. Across 255 frames the reference decoder
raised **zero** bad-frame or bit-error flags against **53–79** for same-mode wrong-key
controls. That is genuine syntax evidence, but a wrong-but-valid parse can also pass.
Directly RFC-packed AMR-WB 12.65 is excluded: its required padding bits were non-zero
in 10,448 of 11,520 slots versus 0 of 1,200 genuine controls.

**No demonstrated match to the broadcast.** Scoring decoded audio against a
simultaneously recorded reference stream performed **no better than wrong-key
controls**, and a lag chosen on three runs scored −0.025 on two held-out runs. The same
scorer recovers a planted offset at 0.799 on genuine audio, so the test is sound — the
decoded audio did not match.

**Coverage is capped by the capture.** Idle captures reach ~94%; scrolling-text
captures reach ~24–36%, limited by how many groups contain a fill. Longer captures,
multiple receivers, and intervals containing idle text all raise coverage; no analysis
of an existing capture can.

---

## Layout

```
scripts/
  kcbs_decode_fast.py            idle-tail captures, one command (start here)
  kcbs_decode_audio.py           general pipeline: demod, keys, text, audio
  demod_iq_window.py             one bounded IQ window through patched Dream
  tail_fill_audit.py             CRC-checked group parsing and fill cribs
  salvage_msc_packets.py         recover groups from partially corrupt observations
  repeated_text_keys.py          XOR propagation across repeated text, CRC-gated
  hangul_key_filter.py           Korean-text key validation, carousel propagation
  render_continuous.py           continuous timeline rendering
  partial_frame_quality.py       role-1 fill strategies, scored on genuine controls
  role1_key_search.py            identifiability test for unknown role-1 bytes
  role1_smooth_fill.py           continuity-chosen role-1 substitutes
  build_evs_gcc_independent.py   GCC build of the unmodified EVS reference
  kcbs_tail_key.py               reference CRC and loader implementation
  evs_dll_cli.py                 Windows DLL fallback for the codec
  test_*.py                      regression tests
build/                           patched Dream build tooling
patches/                         Dream and FAAD2 source patches
recordings/                      bundled March capture (IQ)
dumps/                           its demodulated super-frames, so no Dream build needed
docs/                            third-party notices
```

## Tests

```bash
for t in test_key_recovery test_november_recovery test_speech_reconstruction test_aio_pipeline; do
    python scripts/$t.py
done
```

All synthetic, no capture required. They cover CRC against a known vector, fill-crib
recovery with a random-ciphertext negative control, packet salvage, XOR constraint
consistency, alignment scoring at a known offset, fills leaving known bytes untouched,
and timeline stretch grouping.

## Legal and attribution

Recorded from a publicly transmitted shortwave broadcast. Dream is GPL; FAAD2 patches
fall under its own licence (`docs/COPYING.FAAD2`, `docs/NOTICE.md`). The EVS reference
codec is 3GPP TS 26.443 and is **not** redistributed here — obtain it from 3GPP.
