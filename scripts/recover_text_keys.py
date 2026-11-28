"""Additively recover text-tail prefixes for groups a supplied archive left unkeyed.

`review_supplied_records.py` only *narrows* an existing prefix archive. This script
*adds* prefixes, under the same evidence bar: a recovered prefix is kept only when the
tail it decrypts takes part in a record whose CRC passes **and** whose identical body
recurs at the broadcast's carousel period (so a 16-bit CRC a wrong key hit by chance
cannot survive — the same wrong text would have to reappear at the exact carousel
offset), or when two or more phases of a group independently reproduce a plaintext block
already established that way (128-bit agreement).

Two recall aids feed the gate; neither can pass a group on its own:
  * a phrase proposal (skgsara's method, https://github.com/skgsara/kcbs-drm): an
    eight-unit fragment from previously decoded bulletins/titles fixes one 16-byte key,
    scored against the group's other blocks before any CRC is read;
  * carousel propagation: once a record's plaintext is known it is known at every
    carousel copy (with the item-toggle bit and its recomputed CRC), so a group whose
    received tails match it is keyed.

Proposals are deliberately loose: a wrong proposal simply yields no gated record and the
group stays a miss. Nothing here infers a per-counter keystream, and a recovered prefix
supports the 16-byte text tail only, never the role-1/2/3 audio masks or codec identity.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from parse_text_records import scan_records, crc
from tail_fill_audit import text_unit

CAROUSEL_SPAN = 4          # carousel copies searched either side for propagation
OPEN_HEADER = (0x45, 0xFC)  # item-opening / title record header bytes


def build_corpus(ref_dir):
    """Phrase tables from already decoded bulletin bodies and item openings."""
    ref_dir = Path(ref_dir)
    texts = set()
    for path in sorted(ref_dir.glob("items_*.json")):
        for item in json.loads(path.read_text(encoding="utf-8"))["items"]:
            texts.update(row["text"] for row in item if row.get("ok"))
    openings = set()
    for path in sorted(ref_dir.glob("titles_*.json")):
        for row in json.loads(path.read_text(encoding="utf-8")):
            openings.update((row["opening"], row["opening"].replace("\n", "\r"),
                             row["opening"].replace("\n", "\r\n")))
    if not texts and not openings:
        raise ValueError(f"no decoded text in {ref_dir}/items_*.json or titles_*.json")
    texts |= openings
    words = {n: {s[i:i + n] for s in texts for i in range(len(s) - n + 1)}
             for n in (1, 2, 3, 4)}
    eight = {s[i:i + 8].encode("utf-16le") for s in texts for i in range(len(s) - 7)}
    # Opening title blocks carry three leading zero units then text; match those too.
    for opening in openings:
        padded = ("\x00\x00\x00" + opening).ljust(31, "\x00")[:31]
        for i in range(len(padded) - 7):
            eight.add(padded[i:i + 8].encode("utf-16le"))
    phrases = np.stack([np.frombuffer(b, "<u2") for b in sorted(eight)])
    charset = np.array([text_unit(i) or i in (0, 0x3000) for i in range(65536)])
    return phrases, words, charset


def _opening_header(first, second):
    return (first & 0x7F) == OPEN_HEADER[0] and second == OPEN_HEADER[1]


def _body_header(first, second):
    length = (first & 7) * 16 + (second & 15)
    return (((first & 0x7F) == 5 and length == 92)
            or (0x20 <= (first & 0x7F) <= 0x27 and 2 <= length <= 92 and length % 2 == 0))


def propose(parts, phrases, words, charset):
    """Return one candidate 16-byte key for a group, or None. Recall only; not a check."""
    phases = sorted(parts)
    data = np.stack([parts[p] for p in phases]).view("<u2").reshape(len(phases), 8)
    fills = {}
    for phase, row in parts.items():
        key = row ^ 0xBB
        after = parts.get(phase + 1)
        header = opening = False
        if after is not None:
            first, second = (after ^ key)[:2].tolist()
            opening = _opening_header(first, second)
            header = opening or _body_header(first, second)
        others = [(v ^ key).tobytes().decode("utf-16le", errors="replace")
                  for p, v in parts.items() if p != phase]
        trigrams = sum(s[i:i + 3] in words[3] for s in others for i in range(6))
        if (header and trigrams >= 2) or opening or (phase == 2 and len(parts) == 3 and trigrams >= 4):
            fills[key.tobytes()] = (header, trigrams)
    if fills:
        best = max(fills.values())
        winners = [k for k, score in fills.items() if score == best]
        if len(winners) == 1:
            return np.frombuffer(winners[0], np.uint8)
    candidates = set()
    for row in data:
        possible = phrases ^ row
        clear = possible[:, None, :] ^ data
        good = charset[clear].sum(axis=(1, 2)) >= data.size - 3
        candidates.update(key.tobytes() for key in possible[good])
    best_score, best = -1, None
    for key in candidates:
        clear = data ^ np.frombuffer(key, "<u2")
        strings = []
        for stretch in np.split(clear, np.flatnonzero(np.diff(phases) > 1) + 1):
            strings.extend("".join(chr(int(x)) for x in stretch.ravel()).split("뮻" * 8))
        score = sum(s[i:i + n] in gram for s in strings for n, gram in words.items()
                    for i in range(len(s) - n + 1))
        if score > best_score:
            best_score, best = score, key
        elif score == best_score:
            best = None
    if best is None or best_score <= (24 if len(phases) == 3 else 16):
        return None
    return np.frombuffer(best, np.uint8)


def _scan_verified(keys, tails, first, span, repeat_blocks):
    """Records whose body recurs at the carousel period, with the keyed groups they cover."""
    plain = np.zeros((3 * span, 16), np.uint8)
    known = np.zeros_like(plain, bool)
    for tick, tail in tails.items():
        c = tick // 3
        if c in keys:
            plain[tick - 3 * first] = tail ^ keys[c]
            known[tick - 3 * first] = True
    records = [r for r in scan_records(plain.tobytes(), known_bytes=known.tobytes())]

    def family(r):
        return (r.offset // 16 % repeat_blocks, r.sequence, r.final, r.text)

    counts = {}
    for r in records:
        counts[family(r)] = counts.get(family(r), 0) + 1
    verified, group_keys = [], {}
    for r in records:
        if counts[family(r)] < 2:
            continue
        verified.append(r)
        for block in range(r.offset // 16, (r.offset + r.size) // 16):
            tick = block + 3 * first
            if tick in tails:
                group_keys[tick // 3] = tails[tick] ^ plain[block]
    return verified, group_keys, plain


def _carousel_plaintext(verified, plain, first, repeat_blocks):
    """Known plaintext at each carousel copy of a verified record.

    A body record is identical across copies apart from its item-toggle bit and the
    CRC that follows, so both toggle states propagate every block. An item-opening
    record carries per-copy metadata (serial, clock, header toggle, check word) in
    blocks 0, 1 and 5, so only its copy-invariant blocks (2, 3, 4 and the fill) do.
    """
    expected = {}
    for r in verified:
        base = r.offset // 16
        nblk = r.size // 16
        raw = plain[base:base + nblk].tobytes()
        if _opening_header(raw[0], raw[1]):
            emit = {k: [raw[16 * k:16 * k + 16]] for k in (2, 3, 4, 6) if k < nblk}
        else:
            toggled = bytearray(raw)
            toggled[0] ^= 0x80
            length = (toggled[0] & 7) * 16 + (toggled[1] & 15)
            toggled[2 + length:4 + length] = crc(bytes(toggled[:2 + length])).to_bytes(2, "big")
            emit = {k: [raw[16 * k:16 * k + 16], bytes(toggled[16 * k:16 * k + 16])]
                    for k in range(nblk)}
        for m in range(-CAROUSEL_SPAN, CAROUSEL_SPAN + 1):
            for k, blocks in emit.items():
                tick = (base + m * repeat_blocks + k) + 3 * first
                expected.setdefault(tick, set()).update(blocks)
    return expected


def recover(data_units_npz, supplied_keys_npz, ref_dir, repeat_blocks, out, rounds=6):
    with np.load(data_units_npz) as d:
        du_c, du_p, du = d["counters"].astype(np.int64), d["phases"].astype(np.int64), d["du"]
    tails = {int(3 * c + p): du[i, 698:714].copy() for i, (c, p) in enumerate(zip(du_c, du_p))}
    with np.load(supplied_keys_npz) as k:
        keys = {int(c): row.copy() for c, row, m in zip(k["counters"], k["ks"], k["crc_supported"])
                if m.all()}
    supplied = set(keys)
    all_c = sorted({t // 3 for t in tails})
    first, span = all_c[0], all_c[-1] - all_c[0] + 1

    phrases, words, charset = build_corpus(ref_dir)
    for c in all_c:
        if c in keys:
            continue
        parts = {p: tails[3 * c + p] for p in range(3) if 3 * c + p in tails}
        if len(parts) >= 2:
            cand = propose(parts, phrases, words, charset)
            if cand is not None:
                keys[c] = cand

    confirmed = {}
    for _ in range(rounds):
        verified, group_keys, plain = _scan_verified(keys, tails, first, span, repeat_blocks)
        confirmed.update(group_keys)
        keys.update(group_keys)
        expected = _carousel_plaintext(verified, plain, first, repeat_blocks)
        added = 0
        for c in all_c:
            if c in confirmed:
                continue
            matches = {}
            for p in range(3):
                tick = 3 * c + p
                if tick in tails and tick in expected:
                    cands = {(tails[tick] ^ np.frombuffer(pb, np.uint8)).tobytes()
                             for pb in expected[tick]}
                    matches[p] = cands
            union = set().union(*matches.values()) if matches else set()
            winners = [k for k in union if sum(k in s for s in matches.values()) >= 2]
            if len(winners) == 1:
                kv = np.frombuffer(winners[0], np.uint8)
                confirmed[c] = kv
                keys[c] = kv
                added += 1
        if not added:
            break

    # Only prefixes confirmed by the carousel/CRC gate, or already supplied, are exported.
    out_counters = np.array(sorted(set(confirmed) | supplied), dtype=np.uint32)
    ks = np.zeros((len(out_counters), 16), np.uint8)
    mask = np.zeros_like(ks, bool)
    seed = {**{c: keys[c] for c in supplied}, **confirmed}
    for i, c in enumerate(out_counters):
        ks[i] = seed[int(c)]
        mask[i] = True
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "keys.npz", counters=out_counters, ks=ks, crc_supported=mask)
    report = dict(received_groups=len(all_c), supplied_prefixes=len(supplied),
                  recovered_prefixes=len(set(confirmed) - supplied),
                  exported_prefixes=int(len(out_counters)),
                  carousel_blocks=repeat_blocks,
                  note="Recovered prefixes pass a CRC that also recurs at the carousel "
                       "period, or >=2-phase 128-bit agreement. Text tail only; not audio "
                       "masks or codec identity.")
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-units", required=True, type=Path,
                    help="data_units.npz from salvage_msc_packets.py")
    ap.add_argument("--keys", required=True, type=Path, help="supplied prefix archive")
    ap.add_argument("--reference", required=True, type=Path,
                    help="directory with decoded items_*.json and titles_*.json")
    ap.add_argument("--repeat-blocks", type=int, required=True,
                    help="carousel period in 16-byte blocks for this recording")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    if a.repeat_blocks <= 0:
        ap.error("--repeat-blocks must be positive")
    recover(a.data_units, a.keys, a.reference, a.repeat_blocks, a.out)


if __name__ == "__main__":
    main()
