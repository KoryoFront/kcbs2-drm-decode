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
from collections import Counter
import json
from pathlib import Path

import numpy as np

from parse_text_records import scan_records
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


def detect_period(keys, tails, first, span):
    """Carousel period in blocks: the dominant repeat distance of supplied records.

    A record body recurs once per carousel, so the most common block distance between
    two records with the same (sequence, final, text) is the period. Returns None when
    there is no repeat (a capture shorter than one carousel).
    """
    plain = np.zeros((3 * span, 16), np.uint8)
    known = np.zeros_like(plain, bool)
    for tick, tail in tails.items():
        c = tick // 3
        if c in keys:
            plain[tick - 3 * first] = tail ^ keys[c]
            known[tick - 3 * first] = True
    positions = {}
    for r in scan_records(plain.tobytes(), known_bytes=known.tobytes()):
        positions.setdefault((r.sequence, r.final, r.text), []).append(r.offset // 16)
    distances = Counter()
    for offs in positions.values():
        offs.sort()
        for a, b in zip(offs, offs[1:]):
            if b > a:
                distances[b - a] += 1
    return distances.most_common(1)[0][0] if distances else None


def _carousel_expected(trusted, tails, repeat_blocks):
    """Plaintext expected at each carousel sibling, from every trusted group's blocks.

    Carousel content repeats every `repeat_blocks`, so a trusted group's decrypted
    block is the plaintext of the matching block one or more periods away. This holds
    for any repeating content, not only CRC-verified records, which lets a key reach a
    sibling whose own content never forms a verifiable record. Per-copy fields (item
    toggle, serial, clock, CRC) differ between copies and simply fail to agree, so the
    two-phase check downstream never accepts a key from them.
    """
    expected = {}
    for c, key in trusted.items():
        for p in range(3):
            tick = 3 * c + p
            if tick not in tails:
                continue
            block = (tails[tick] ^ key).tobytes()
            for m in range(-CAROUSEL_SPAN, CAROUSEL_SPAN + 1):
                if m:
                    expected.setdefault(tick + m * repeat_blocks, set()).add(block)
    return expected


def recover(data_units_npz, supplied_keys_npz, ref_dir, repeat_blocks, out, rounds=8):
    with np.load(data_units_npz) as d:
        du_c, du_p, du = d["counters"].astype(np.int64), d["phases"].astype(np.int64), d["du"]
    tails = {int(3 * c + p): du[i, 698:714].copy() for i, (c, p) in enumerate(zip(du_c, du_p))}
    with np.load(supplied_keys_npz) as k:
        keys = {int(c): row.copy() for c, row, m in zip(k["counters"], k["ks"], k["crc_supported"])
                if m.all()}
    supplied = set(keys)
    all_c = sorted({t // 3 for t in tails})
    first, span = all_c[0], all_c[-1] - all_c[0] + 1

    if repeat_blocks is None:
        repeat_blocks = detect_period(keys, tails, first, span)
        if repeat_blocks is None:
            raise ValueError("no carousel repeat detected; pass --repeat-blocks explicitly")

    phrases, words, charset = build_corpus(ref_dir)
    for c in all_c:
        if c in keys:
            continue
        parts = {p: tails[3 * c + p] for p in range(3) if 3 * c + p in tails}
        if len(parts) >= 2:
            cand = propose(parts, phrases, words, charset)
            if cand is not None:
                keys[c] = cand

    # Trusted = supplied prefixes plus those confirmed by the carousel/CRC gate below.
    confirmed = {c: keys[c] for c in supplied}
    for _ in range(rounds):
        _, group_keys, _ = _scan_verified(keys, tails, first, span, repeat_blocks)
        for c, kv in group_keys.items():
            confirmed[c] = kv
            keys[c] = kv
        # Carousel XOR-link: a trusted group's decrypted block is the plaintext of the
        # same block one period away; a sibling two of whose phases reproduce the same
        # 16-byte key (128-bit agreement) is confirmed.
        expected = _carousel_expected(confirmed, tails, repeat_blocks)
        added = 0
        for c in all_c:
            if c in confirmed:
                continue
            matches = {}
            for p in range(3):
                tick = 3 * c + p
                if tick in tails and tick in expected:
                    matches[p] = {(tails[tick] ^ np.frombuffer(pb, np.uint8)).tobytes()
                                  for pb in expected[tick]}
            union = set().union(*matches.values()) if matches else set()
            winners = [k for k in union if sum(k in s for s in matches.values()) >= 2]
            if len(winners) == 1:
                kv = np.frombuffer(winners[0], np.uint8)
                confirmed[c] = kv
                keys[c] = kv
                added += 1
        if not added:
            break

    out_counters = np.array(sorted(confirmed), dtype=np.uint32)
    ks = np.zeros((len(out_counters), 16), np.uint8)
    mask = np.ones_like(ks, bool)
    for i, c in enumerate(out_counters):
        ks[i] = confirmed[int(c)]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "keys.npz", counters=out_counters, ks=ks, crc_supported=mask)
    report = dict(received_groups=len(all_c), supplied_prefixes=len(supplied),
                  recovered_prefixes=len(set(confirmed) - supplied),
                  exported_prefixes=int(len(out_counters)),
                  carousel_blocks=int(repeat_blocks),
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
    ap.add_argument("--repeat-blocks", type=int, default=None,
                    help="carousel period in 16-byte blocks (auto-detected from the supplied "
                         "records when omitted)")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    if a.repeat_blocks is not None and a.repeat_blocks <= 0:
        ap.error("--repeat-blocks must be positive")
    recover(a.data_units, a.keys, a.reference, a.repeat_blocks, a.out)


if __name__ == "__main__":
    main()
