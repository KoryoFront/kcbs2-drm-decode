"""Test exact clear-byte codeword reuse against CRC-supported March keys.

This is a codeword-reuse hypothesis, not proof that clear-byte fingerprints
uniquely determine a frame. Reports held-out tests, collisions, and independent
fingerprint support. Output proposals are deliberately NOT labelled verified.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

STARTS = [s + 131*c for c in range(5) for s in (7, 40, 73, 106)]
MODES = [p for c in range(5) for p in (2+c, 39+131*c, 72+131*c, 105+131*c)]
KEYED = np.arange(0, 32, 3)
CLEAR = np.array([i for i in range(32) if i % 3])


def fingerprints(packets, modes):
    return [[bytes([int(modes[g, f])]) + packets[g, f, CLEAR].tobytes()
             for f in range(60)] for g in range(len(packets))]


def attack(path, out):
    data = np.load(path)
    keep = data['idle_text_crc_ok']
    if 'du' in data.files:
        du, keys, counters = data['du'][keep], data['idle_ks'][keep], data['counters'][keep]
        p = np.stack([du[:, phase, s:s+32] for phase in range(3) for s in STARTS], axis=1)
        modes = np.stack([du[:, phase, m] for phase in range(3) for m in MODES], axis=1)
    else:
        if 'mode_source_positions' not in data.files:
            raise ValueError('corpus must have source-checked corrected mode labels')
        keys, counters = data['ks0_15'][keep], data['counter'][keep]
        p, modes = data['packets_raw'][keep], data['mode'][keep]
    d = p.copy()
    for j, b in enumerate(KEYED):
        d[:, 0::4, b] ^= keys[:, j, None]
    fp = fingerprints(p, modes)
    train, test = {}, []
    # Split contiguous counter range, not shuffled frames, to reduce leakage.
    split = int(np.median(counters))
    for g, counter in enumerate(counters):
        for f in range(0, 60, 4):
            fingerprint = fp[g][f]
            plain = d[g, f, KEYED].tobytes()
            if counter <= split:
                train.setdefault(fingerprint, set()).add(plain)
            else:
                test.append((fingerprint, plain))
    unambiguous = {k: next(iter(v)) for k, v in train.items() if len(v) == 1}
    hits = [(unambiguous[k], plain) for k, plain in test if k in unambiguous]
    summary = dict(crc_supported_groups=len(counters),
                   training_fingerprints=len(train),
                   training_collisions=sum(len(v) > 1 for v in train.values()),
                   heldout_frames=len(test), heldout_matches=len(hits),
                   heldout_exact=sum(a == b for a, b in hits))
    # Use all known role-0 frames only after measuring held-out performance.
    dictionary = defaultdict(set)
    for g in range(len(p)):
        for f in range(0, 60, 4):
            dictionary[fp[g][f]].add(d[g, f, KEYED].tobytes())
    unique = {k: next(iter(v)) for k, v in dictionary.items() if len(v) == 1}
    result = []
    for role in (1, 2, 3):
        stats = Counter()
        for g, counter in enumerate(counters):
            votes, support = Counter(), defaultdict(set)
            for f in range(role, 60, 4):
                fingerprint = fp[g][f]
                if fingerprint not in unique:
                    continue
                stats['matched_frames'] += 1
                plaintext = np.frombuffer(unique[fingerprint], dtype=np.uint8)
                proposal = (p[g, f, KEYED] ^ plaintext).tobytes()
                if role == 1:
                    stats['known_prefix_matches'] += proposal[:5] == keys[g, 11:16].tobytes()
                votes[proposal] += 1
                support[proposal].add(fingerprint)
            if not votes:
                continue
            stats['groups_with_proposals'] += 1
            if len(votes) != 1:
                stats['conflicting_groups'] += 1
                continue
            proposal = next(iter(votes))
            n = len(support[proposal])
            stats['consistent_groups'] += 1
            stats['multiple_fingerprint_groups'] += n >= 2
            result.append(dict(counter=int(counter), role=role,
                               candidate_key_hex=proposal.hex(),
                               supporting_frames=votes[proposal], distinct_fingerprints=n,
                               known_prefix_matches=(proposal[:5] == keys[g, 11:16].tobytes())
                               if role == 1 else None))
        summary[f'role{role}'] = dict(stats)
    summary['all_role0_fingerprints'] = len(dictionary)
    summary['all_role0_ambiguous_fingerprints'] = sum(len(v) > 1 for v in dictionary.values())
    summary['note'] = 'Matching clear bytes is a hypothesis; proposals are not verified keys.'
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out / 'proposals.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('groups_npz')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    attack(args.groups_npz, args.out)
