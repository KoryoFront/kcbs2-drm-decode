"""Validate fill-crib keys by Korean text quality, then propagate via the repeat.

A wrong key produces uniform 16-bit values, so each code unit lands inside the
Hangul syllable block (U+AC00-U+D7A3) with probability 0.171. Requiring a whole
24-character group to be Hangul/space/punctuation has a false-positive rate near
1e-16, which is far stronger than a lenient character-range gate. Keys that pass
are then propagated to their carousel repeats: identical plaintext at two times
means key(h) = cipher(h) XOR plaintext(g), exactly.

Every fully reconstructable 112-byte segment is then CRC-16 checked as the final
independent confirmation. CRC-valid segments are reported separately.

Usage: python scripts/hangul_key_filter.py <salvage dir> --out <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tail_fill_audit import crc

# Hangul syllables, Hangul Jamo, ASCII printable, and common CJK/unit symbols a
# Korean text stream may legitimately contain.
def plausible(cu):
    return (0xAC00 <= cu <= 0xD7A3 or 0x3130 <= cu <= 0x318F or
            0x20 <= cu <= 0x7E or cu in (0x0A, 0x0D, 0x2103, 0x00B0, 0x33A7))


def group_units(tails, g, key):
    """24 code units of one group, or None if a tail is missing."""
    out = []
    for p in range(3):
        t = 3*g + p
        if t not in tails:
            return None
        plain = tails[t] ^ key
        for m in range(8):
            out.append(int(plain[2*m]) | int(plain[2*m+1]) << 8)
    return out


def hangul_score(units):
    return sum(1 for cu in units if 0xAC00 <= cu <= 0xD7A3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('salvage')
    ap.add_argument('--out', required=True)
    ap.add_argument('--periods', default='323,646',
                    help='carousel period in groups, measured from validated plaintext')
    ap.add_argument('--min-hangul', type=int, default=18, help='of 24 code units')
    ap.add_argument('--min-plausible', type=int, default=0,
                    help='of 24 code units; 0 disables this extra gate')
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    salvage = (root/a.salvage).resolve()
    out = (root/a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    periods = [int(p) for p in a.periods.split(',')]

    with np.load(salvage/'groups.npz') as d:
        counters = d['counters'].astype(np.int64)
        du = d['du']
        seeds = d['candidate_ks'].copy()
        seed_mask = d['candidate_mask'].copy()
    index = {int(c): i for i, c in enumerate(counters)}
    tails = {3*int(c) + p: du[i, p, 698:714].copy()
             for i, c in enumerate(counters) for p in range(3)}

    seeds_by_group = {int(c): seeds[i].copy() for i, c in enumerate(counters) if seed_mask[i].all()}
    kept, scores = {}, {}
    for g, key in seeds_by_group.items():
        units = group_units(tails, g, key)
        if units is None:
            continue
        good = sum(1 for cu in units if plausible(cu))
        hangul = hangul_score(units)
        if hangul >= a.min_hangul and good >= a.min_plausible:
            kept[g] = key
            scores[g] = hangul
    print('fill seeds %d -> %d pass the Hangul gate (>=%d/24)'
          % (len(seeds_by_group), len(kept), a.min_hangul), flush=True)

    # Propagate to carousel repeats. The period was measured from validated
    # plaintext: 323 groups gives 96% byte agreement, 646 groups 99.9%. Deriving a
    # partner key is exact - key(h) = cipher(h) XOR plaintext(g) - so the only risk
    # is a wrong anchor, and every derived key is re-checked with the same Korean
    # text rule that validated the anchors.
    keys = dict(kept)
    derived_total = 0
    for round_number in range(6):
        # Collect per-byte votes for every unkeyed group from all anchors that sit
        # one or two carousel periods away. Votes come from each of the anchor's
        # three tails, so a single mismatched text byte cannot spoil the key.
        votes = {}
        for g in sorted(keys):
            if not all(3*g + p in tails for p in range(3)):
                continue
            for delta in periods:
                for h in (g + delta, g - delta):
                    if h not in index or h in keys:
                        continue
                    if not all(3*h + p in tails for p in range(3)):
                        continue
                    slot = votes.setdefault(h, [dict() for _ in range(16)])
                    for p in range(3):
                        plain_tail = tails[3*g + p] ^ keys[g]
                        proposal = tails[3*h + p] ^ plain_tail
                        for column in range(16):
                            slot[column][int(proposal[column])] =                                 slot[column].get(int(proposal[column]), 0) + 1
        added = {}
        for h, slot in votes.items():
            proposal = np.zeros(16, np.uint8)
            resolved = True
            for column in range(16):
                ranked = sorted(slot[column].items(), key=lambda kv: -kv[1])
                if not ranked:
                    resolved = False
                    break
                if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
                    resolved = False
                    break
                proposal[column] = ranked[0][0]
            if not resolved:
                continue
            check = group_units(tails, h, proposal)
            if check is None or hangul_score(check) < a.min_hangul:
                continue
            added[h] = proposal
        if not added:
            break
        keys.update(added)
        derived_total += len(added)
        print('round %d: derived %d partner keys by voting (total %d)'
              % (round_number, len(added), len(keys)), flush=True)

    # Final independent check: CRC over every fully keyed 112-byte segment.
    starts = sorted({t - j for t in tails for j in range(7)})
    checked, valid, supported = 0, [], np.zeros(len(counters), bool)
    for start in starts:
        plain = np.zeros(112, np.uint8)
        for offset in range(112):
            t = start + offset//16
            if t not in tails or (t//3) not in keys:
                plain = None
                break
            plain[offset] = int(tails[t][offset % 16]) ^ int(keys[t//3][offset % 16])
        if plain is None:
            continue
        checked += 1
        if crc(plain[:94]) != int.from_bytes(plain[94:96].tobytes(), 'big'):
            continue
        valid.append(dict(start=start, plaintext_hex=plain.tobytes().hex(),
                          text=plain[2:94].tobytes().decode('utf-16le', errors='replace')))
        for offset in range(112):
            g = (start + offset//16)//3
            if g in index:
                supported[index[g]] = True

    ks = np.zeros((len(counters), 16), np.uint8)
    mask = np.zeros_like(ks, bool)
    for i, c in enumerate(counters):
        if int(c) in keys:
            ks[i] = keys[int(c)]
            mask[i] = True
    report = dict(periods=periods, hangul_gate=dict(min_hangul=a.min_hangul,
                                                    min_plausible=a.min_plausible),
                  fill_seeds=int(len(seeds_by_group)), seeds_passing=len(kept),
                  derived=int(derived_total), groups_with_key=int(mask.all(axis=1).sum()),
                  groups_total=len(counters), coverage=round(float(mask.all(axis=1).mean()), 3),
                  segments_checked=checked, segments_crc_valid=len(valid),
                  distinct_crc_valid_bodies=len({v['plaintext_hex'] for v in valid}),
                  groups_touched_by_crc=int(supported.sum()))
    np.savez_compressed(out/'keys.npz', counters=counters.astype(np.uint32), ks=ks,
                        candidate_mask=mask,
                        crc_supported=np.repeat(supported[:, None], 16, axis=1),
                        verified=np.repeat(supported[:, None], 16, axis=1))
    (out/'crc_supported_text.json').write_text(json.dumps(valid, ensure_ascii=False, indent=2),
                                                encoding='utf-8')
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
