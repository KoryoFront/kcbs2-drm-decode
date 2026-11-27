# Measured-mask replay (roles 1–3)

This brings the **measured-mask** method from skgsara's
[kcbs-drm](https://github.com/skgsara/kcbs-drm) into this repository, with the
example data needed to reproduce it. It raises per-group audio coverage well
past what the text-tail prefix alone supplies, by filling the frame positions
whose masks the text tail never reaches (role 1's last six bytes, and all of
roles 2 and 3).

Credit: the method, the measured masks, and the November 19/27 recordings are
skgsara's, building in turn on KoryoFront's receiver and Martyn's
([martynw/KCBSDRM](https://github.com/martynw/KCBSDRM)) text-record and framing
work. This directory republishes skgsara's **results** (the masks) and the
applying renderer; it does **not** contain the search that produced them.

## What the text tail cannot reach, and what this adds

The phase-0 text tail shares `ks[0..15]`, which covers all 11 role-0 mask bytes
and 5 of role 1's 11. That leaves **27 unknown mask bytes per group**: 6 in
role 1, 11 in role 2, 10 in role 3 (the fourth frame's terminal byte is
received clear). So text-only recovery gives at most role 0 and a partial
role 1 — one to one-and-a-bit of every four frames — and conceals the rest.

skgsara's masks are inferred in the **EVS codec domain against external
reference recordings**: the broadcast is a commercially released performance
(identified, e.g. *행복한 내 나라* on the 1991 Wangjaesan CD, and the Nov 19
Moranbong Band tracks), so an aligned reference encode gives expected
EVS-internal values (sign-bit addresses, TCX coefficients, LP/HQ fields).
Candidate mask bytes are ranked and cross-checked against two independent
references (FM + CD) and across cycle splits, and are kept only where the choice
is unique under the codec's bit budget. This fills role-1/2/3 slots the text
tail never touches — the substantive coverage gain over prefix-only decoding.

**Honesty, carried over from upstream.** These are *inferred* masks, not
text-CRC-verified keys. The renderer labels each supplied slot `established`
(text-checksum-supported, position 1) vs `inferred` (positions 1–4 from the
acoustic search) vs `erased`. The search itself needs the instrumented EVS
field reader and the aligned reference recordings, and is **not** in this repo
(nor in skgsara's public repo — only the resulting masks are). Nothing here
recovers the per-counter keystream in general; a group with no measured mask
and no text prefix is still an erasure. The codec-domain match validates the
EVS framing on these inputs; it is not independent proof of the broadcast codec.

## Reproduce (bit-exact)

With this repo's EVS build at `work/evs_gcc_independent/EVS_dec`:

```bash
python scripts/render_measured_evs.py examples/nov27/receiver \
    --masks dumps/nov27-measured-masks.json --received-span --out work/nov27
```

Expected `programme.192` SHA-256:
`2ae4c7d6f186558db6f96cfc9c4c613e4d9702175f3980c855784fb3b4b32738`
(verified on this repo's decoder when this was added).

Of the 1,500 EVS slots in the 30-second November 27 capture:

| class | slots |
| --- | ---: |
| established (text-checksum, role 0) | 355 |
| inferred (acoustic, roles 1–4) | 795 |
| erased (concealed) | 350 |

By 1-based frame position (roles 0–3): position 1 = 375, position 2 = 285,
position 3 = 265, position 4 = 225. That is **77 % of frames supplied** versus
roughly the role-0 share (~25 %) from the text tail alone.

`python examples/nov27/validation/check.py` recomputes all ten weather-record
CRCs, rebuilds each supplied EVS word from the bundled packet bodies and masks,
and verifies every frame against the bundled `programme.192.gz`,
`text-supported.192.gz`, and `frames.csv`. Reproducing the acoustic inference
additionally requires the aligned FM and CD references and the instrumented
reader, which are not bundled.
